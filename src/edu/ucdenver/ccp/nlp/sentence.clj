(ns edu.ucdenver.ccp.nlp.sentence
  (:require [clojure.math.combinatorics :as combo]
            [clojure.string :as str]
            [ubergraph.alg :as uber-alg]
            [graph :as graph]
            [math :as math]
            [word2vec :as word2vec]
            [taoensso.timbre :as log]))

(defrecord Sentence [concepts entities context context-vector])

(defn ann-tok
  [model {[[_ {concept-start :start concept-end :end}]] :spans :keys [doc] :as ann}]
  (let [tok-id (->> model
                    :structure-annotations
                    (vals)
                    (filter #(= (:doc %) doc))
                    (reduce
                      (fn [{[[_ {old-tok-start :start old-tok-end :end}]] :spans :as old-tok}
                           {[[_ {new-tok-start :start new-tok-end :end}]] :spans :as new-tok}]
                        (cond (<= new-tok-start concept-start concept-end new-tok-end) new-tok
                              (<= concept-start new-tok-start new-tok-end concept-end) new-tok
                              (<= old-tok-start concept-start concept-end old-tok-end) old-tok
                              (<= concept-start old-tok-start old-tok-end concept-end) old-tok
                              (<= new-tok-start concept-start new-tok-end) new-tok
                              (<= old-tok-start concept-start old-tok-end) old-tok
                              :else old-tok))
                      nil))]
    (when-not tok-id (log/warn "No token found for" ann))
    tok-id))


(defn tok-sent-id
  [{:keys [structure-graphs]} {:keys [id]}]
  (some
    (fn [[sent-id sent]]
      (when (get-in sent [:node-map id])
        sent-id))
    structure-graphs))


(defn dependency-path
  [undirected-sent tok1 tok2]
  (-> undirected-sent
      (uber-alg/shortest-path tok1 tok2)
      (uber-alg/nodes-in-path)))

(defn dependency-embedding
  [dependency-path structure-annotations]
  (when-let [vectors (->> dependency-path
                          (keep #(get-in structure-annotations [% :VEC]))
                          (seq))]
    (apply math/unit-vec-sum vectors)))

(defn sentence-entities
  [{:keys [structure-annotations structure-graphs]} sent entities]
  (let [undirected-sent (graph/undirected-graph (get structure-graphs sent))]
    (keep
      (fn [[{tok1 :tok c1 :concept id1 :id}
            {tok2 :tok c2 :concept id2 :id}]]
        (when-not (= tok1 tok2)
          (log/info id1)
          (let [concepts (conj #{} #{c1} #{c2})
                context (dependency-path undirected-sent tok1 tok2)
                context-vector (dependency-embedding context structure-annotations)
                entities #{id1 id2}]
            (->Sentence concepts entities context context-vector))))
      (combo/combinations entities 2))))

(defn concept-annotations->sentences
  [{:keys [concept-annotations] :as model}]
  (mapcat
    (fn [[sent entities]]
      (log/info sent)
      (sentence-entities model sent entities))
    (->> concept-annotations
         vals
         (group-by :sent)
         (remove (comp nil? first)))))

(defn assign-word-embedding
  [{:as annotation [_ {:keys [text]}] :spans}]
  (assoc annotation :VEC (-> text
                             (str/lower-case)
                             (word2vec/word-embedding))))

(defn assign-sent-id
  [model tok]
  (assoc tok :sent tok-sent-id model tok))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [{:keys [entities]}]
      (some #(= id %) entities))
    sentences))

(defn assign-tok
  [model ann]
  (let [{:keys [sent id]} (ann-tok model ann)]
    (assoc ann :tok id
               :sent sent)))

