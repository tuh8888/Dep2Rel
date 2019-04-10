(ns edu.ucdenver.ccp.nlp.sentence
  (:require [clojure.math.combinatorics :as combo]
            [clojure.string :as str]
            [ubergraph.alg]
            [graph]
            [word2vec]
            [taoensso.timbre :as t]))

(defrecord Sentence [concepts entities context context-vector])


;(defn walk-dep
;  ([dependency tok1 tok2]
;   (assert (= (:DOC tok1) (:DOC tok2)))
;   (assert (= (:SENT tok1) (:SENT tok2)))
;   (let [sent (:SENT tok1)
;         doc (:DOC tok1)
;         dependency (filter #(and (= (:SENT %) sent)
;                                  (= (:DOC %) doc))
;                            dependency)]
;     (loop [tok tok1
;            path []]
;       (let [next-tok (some #(when (= (:HEAD tok) (:ID %)) %)
;                            dependency)]
;         (cond (= tok tok2) path
;               (not next-tok) (conj path (reverse (walk-dep dependency tok2)))
;               :else (recur next-tok (conj path tok)))))))
;  ([dependency tok]
;   (let [sent (:SENT tok)
;         doc (:DOC tok)
;         dependency (filter #(and (= (:SENT %) sent)
;                                  (= (:DOC %) doc))
;                            dependency)]
;     (loop [tok tok
;            path []]
;       (let [next-tok (some #(when (= (:HEAD tok) (:ID %)) %)
;                            dependency)]
;         (if (not next-tok)
;           path
;           (recur next-tok (conj path tok))))))))

(defn annotation->entity
  [model ann]
  (let [concept-start (-> ann :spans vals first :start)
        concept-end (-> ann :spans vals first :end)]
    (some
      (fn [tok]
        (let [tok-start (-> tok :spans vals first :start)
              tok-end (-> tok :spans vals first :end)]
          (when (or (<= tok-start concept-start concept-end tok-end)
                    (<= concept-start tok-start tok-end concept-end))
            tok)))
      (vals (:structure-annotations model)))))

(defn tok-sent
  [model tok]
  (first
    (filter #(get-in % [:node-map (:id tok)])
            (vals (:structure-graphs model)))))

(defn annotations->entities
  [model]
  (reduce
    (fn [model [id ann]]
      (t/info id)
      (let [tok (annotation->entity model ann)
            sent (tok-sent model tok)]
        (-> model
            (assoc-in [:concept-annotations id :tok] tok)
            (assoc-in [:concept-annotations id :sent] sent))))
    model
    (get model :concept-annotations)))

(defn entities->sentences
  [model]
  (reduce
    (fn [model [sent sentence-entities]]
      (update model :sentences into
              (keep identity
                    (pmap
                      (fn [[e1 e2 :as entities]]
                        (when-not (= (:tok e1)
                                     (:tok e2))
                          (t/info (:id e1))
                          (let [concepts (->> entities
                                              (map :concept)
                                              (map set))
                                context (-> (graph/undirected-graph sent)
                                            (ubergraph.alg/shortest-path
                                              (-> e1 :tok :id)
                                              (-> e2 :tok :id))
                                            (ubergraph.alg/nodes-in-path))
                                context-vector (when-let [vectors (->> context
                                                                       (map #(get (:structure-annotations model) %))
                                                                       (keep :VEC)
                                                                       (seq))]
                                                 (apply math/unit-vec-sum vectors))]
                            (->Sentence concepts entities context context-vector))))
                      (combo/combinations sentence-entities 2)))))
    model
    (->> model
         :concept-annotations
         vals
         (group-by :sent)
         (remove (comp nil? first)))))

(defn assign-word-embedding
  [annotation]
  (assoc annotation
    :VEC (word2vec/word-embedding
           (str/lower-case
             (-> annotation
                 :spans
                 vals
                 first
                 :text)))))

(defn make-sentences
  [model]
  (-> model
      (update :structure-annotations
              #(zipmap (keys %)
                       (pmap assign-word-embedding (vals %))))
      annotations->entities
      entities->sentences))

(defn sentences-with-ann
  [sentences id]
  (filter (fn [s]
            (some (fn [e]
                    (= id (:id e)))
                  (get s :entities)))
          sentences))
