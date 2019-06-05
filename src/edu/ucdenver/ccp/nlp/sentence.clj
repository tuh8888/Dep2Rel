(ns edu.ucdenver.ccp.nlp.sentence
  (:require [clojure.math.combinatorics :as combo]
            [clojure.string :as str]
            [ubergraph.alg]
            [graph]
            [math]
            [word2vec]
            [taoensso.timbre :as log]))

(defrecord Sentence [concepts entities context context-vector])

(defn annotation-tok-id
  [model ann]
  (log/info (:id ann))
  (let [concept-start (-> ann :spans vals first :start)
        concept-end (-> ann :spans vals first :end)
        doc-toks (filter #(= (:doc %) (:doc ann)) (vals (:structure-annotations model)))
        first-tok-id (first doc-toks)]
    (:id
      (reduce
        (fn [old-tok new-tok]
          (let [new-tok-start (-> new-tok :spans vals first :start)
                new-tok-end (-> new-tok :spans vals first :end)
                old-tok-start (-> old-tok :spans vals first :start)
                old-tok-end (-> old-tok :spans vals first :end)]
            (cond (<= new-tok-start concept-start concept-end new-tok-end) new-tok
                  (<= concept-start new-tok-start new-tok-end concept-end) new-tok
                  (<= old-tok-start concept-start concept-end old-tok-end) old-tok
                  (<= concept-start old-tok-start old-tok-end concept-end) old-tok
                  (<= new-tok-start concept-start new-tok-end) new-tok
                  (<= old-tok-start concept-start old-tok-end) old-tok
                  :else old-tok)))
        first-tok-id doc-toks))
    #_(some
      (fn [tok]
        (let [tok-start (-> tok :spans vals first :start)
              tok-end (-> tok :spans vals first :end)]
          (when (or (<= tok-start concept-start concept-end tok-end)
                    (<= concept-start tok-start tok-end concept-end))
            (:id tok))))
      doc-toks)))

(defn tok-sent-id
  [model tok-id]
  (some
    (fn [[id sent]]
      (when (get-in sent [:node-map tok-id])
        id))
    (:structure-graphs model)))

(defn dependency-path
  [undirected-sent tok1 tok2]
  (-> undirected-sent
      (ubergraph.alg/shortest-path tok1 tok2)
      (ubergraph.alg/nodes-in-path)))

(defn dependency-embedding
  [dependency-path structure-annotations]
  (when-let [vectors (->> dependency-path
                          (keep #(get-in structure-annotations [% :VEC]))
                          (seq))]
    (apply math/unit-vec-sum vectors)))

(defn sentence-entities
  [{:keys [structure-annotations structure-graphs]} sent, entities]
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
  [annotation]
  (assoc annotation
    :VEC (word2vec/word-embedding
           (str/lower-case
             (-> annotation
                 :spans
                 vals
                 first
                 :text)))))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [s]
      (some
        (fn [e]
          (= id e))
        (:entities s)))
    sentences))

(defn concept-annotations-with-toks
  [model]
  (zipmap (keys (:concept-annotations model))
          (pmap
            #(let [tok-id (annotation-tok-id model %)
                   sent-id (tok-sent-id model tok-id)]
               (assoc % :tok tok-id
                        :sent sent-id))
            (vals (:concept-annotations model)))))

(defn structures-annotations-with-embeddings
  [model]
  (zipmap (keys (:structure-annotations model))
          (doall
            (pmap assign-word-embedding
                  (vals (:structure-annotations model))))))
