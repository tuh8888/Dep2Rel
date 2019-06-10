(ns edu.ucdenver.ccp.nlp.sentence
  (:require [clojure.string :as str]
            [ubergraph.alg :as uber-alg]
            [graph :as graph]
            [math :as math]
            [word2vec :as word2vec]
            [taoensso.timbre :as log]))

(defrecord Sentence [concepts entities context context-vector])

(defn ann-tok
  [model {:keys [doc spans id] :as ann}]
  (log/debug "Ann:" id)
  (let [{concept-start :start concept-end :end} (first (vals spans))
        tok-id (->> model
                    :structure-annotations
                    (vals)
                    (filter #(= (:doc %) doc))
                    (reduce
                      (fn [{old-spans :spans :as old-tok} {new-spans :spans :as new-tok}]
                        (let [{old-tok-start :start old-tok-end :end} (first (vals old-spans))
                              {new-tok-start :start new-tok-end :end} (first (vals new-spans))]
                          (cond (<= new-tok-start concept-start concept-end new-tok-end) new-tok
                                (<= concept-start new-tok-start new-tok-end concept-end) new-tok
                                (and old-tok (<= old-tok-start concept-start concept-end old-tok-end)) old-tok
                                (and old-tok (<= concept-start old-tok-start old-tok-end concept-end)) old-tok
                                (<= new-tok-start concept-start new-tok-end) new-tok
                                (and old-tok (<= old-tok-start concept-start old-tok-end)) old-tok
                                :else old-tok)))
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
  [undirected-sent toks]
  (-> toks
      (map :id)
      (apply uber-alg/shortest-path undirected-sent)
      (uber-alg/nodes-in-path)))

(defn sum-vectors
  [vectors]
  (let [vectors (seq (keep identity vectors))]
    (when vectors
      (apply math/unit-vec-sum vectors))))


(defn make-context-vector
  [dependency-path structure-annotations [ann1 ann2]]
  (-> (map #(get-in structure-annotations [% :word-vector]) dependency-path)
      (conj (:word-vector ann1))
      (conj (:word-vector ann2))
      (sum-vectors)))

(defn make-sentence
  "Make a sentence using the sentence graph and entities"
  [{:keys [structure-annotations]} undirected-sent anns]
  (let [concepts (->> anns (map :concept) (map #(conj #{} %)) (set))
        context (->> anns (map :tok) (make-context-path undirected-sent))
        context-vector (make-context-vector context structure-annotations anns)
        entities (->> anns (map :id) (set))]
    (->Sentence concepts entities context context-vector)))

(defn make-sentences
  [model undirected-sent sent-annotations]
  (->> (clojure.math.combinatorics/combinations sent-annotations 2)
       (map #(make-sentence model undirected-sent %))))

(defn concept-annotations->sentences
  [{:keys [concept-annotations structure-graphs] :as model}]
  (log/info "Making sentences for concept annotations")
  (let [undirected-sents (util/map-kv graph/undirected-graph structure-graphs)]
    (->> concept-annotations
         (vals)
         (group-by :sent)
         (pmap (fn [[sent sent-annotations]]
                 (log/debug "Sentence:" sent)
                 (make-sentences model (get undirected-sents sent) sent-annotations)))

         (apply concat))))



(defn assign-word-embedding
  [{:keys [spans] :as annotation}]
  (let [{:keys [text]} (first (vals spans))]
    (assoc annotation :word-vector (-> text
                                       (str/lower-case)
                                       (word2vec/word-embedding)))))

(defn assign-sent-id
  [model tok]
  (assoc tok :sent (tok-sent-id model tok)))

(defn assign-tok
  [model ann]
  (let [{:keys [sent id]} (ann-tok model ann)]
    (assoc ann :tok id
               :sent sent
               :word-vector (->> ann
                                 :spans
                                 vals
                                 (keep :text)
                                 (map #(str/split % #" "))
                                 (apply concat)
                                 (map str/lower-case)
                                 (map word2vec/word-embedding)
                                 (sum-vectors)))))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [{:keys [entities]}]
      (some #(= id %) entities))
    sentences))

