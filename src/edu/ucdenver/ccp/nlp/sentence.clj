(ns edu.ucdenver.ccp.nlp.sentence
  (:require [clojure.string :as str]
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

(defn make-context-path
  [undirected-sent tok1 tok2]
  (-> undirected-sent
      (uber-alg/shortest-path tok1 tok2)
      (uber-alg/nodes-in-path)))

(defn make-context-vector
  [dependency-path structure-annotations ann1 ann2]
  (when-let [vectors (let [vectors (->> dependency-path
                                        (keep #(get-in structure-annotations [% :word-vector]))
                                        (seq))]
                          (-> vectors
                              (conj (:word-vector ann1))
                              (conj (:word-vector ann2))))]
    (apply math/unit-vec-sum vectors)))

(defn make-sentence
  "Make a sentence using the sentence graph and entities"
  [{:keys [structure-annotations]} undirected-sent ann1 ann2]
  (let [anns [ann1 ann2]
        concepts (->> anns (map :concept) (map #(conj #{} %)) (set))
        context (->> anns (map :tok) (apply make-context-path undirected-sent))
        context-vector (make-context-vector context structure-annotations ann1 ann2)
        entities (->> anns (map :id) (set))]
    (->Sentence concepts entities context context-vector)))

(defn concept-annotations->sentences
  [{:keys [concept-annotations structure-graphs] :as model}]
  (log/info "Making sentences for concept annotations")
  (apply concat
    (pmap
      (fn [id g]
        (let [sent-annotations (filter #(= id (:sent %) concept-annotations))]
          (when (seq sent-annotations)
            (let [g (graph/undirected-graph g)]
              (for [ann1 sent-annotations
                    ann2 sent-annotations
                    :when (not= ann1 ann2)]
                (make-sentence model g ann1 ann2))))))
      structure-graphs)))

(defn assign-word-embedding
  [{:as annotation [_ {:keys [text]}] :spans}]
  (assoc annotation :word-vector (-> text
                                     (str/lower-case)
                                     (word2vec/word-embedding))))

(defn assign-sent-id
  [model tok]
  (update tok :sent tok-sent-id model))

(defn assign-tok
  [model ann]
  (let [{:keys [sent id]} (ann-tok model ann)]
    (assoc ann :tok id
               :sent sent
               :word-vector (->> ann :spans (map :text) (map word2vec/word-embedding) (apply math/unit-vec-sum)))))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [{:keys [entities]}]
      (some #(= id %) entities))
    sentences))

