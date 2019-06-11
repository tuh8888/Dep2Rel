(ns edu.ucdenver.ccp.nlp.sentence
  (:require [clojure.string :as str]
            [ubergraph.alg :as uber-alg]
            [graph :as graph]
            [math :as math]
            [util :as util]
            [word2vec :as word2vec]
            [taoensso.timbre :as log]
            [uncomplicate-context-alg :as context])
  (:import (clojure.lang PersistentArrayMap)))

(extend-type PersistentArrayMap
  context/ContextVector
  (context-vector [self {:keys [factory]}]
    (or (:VEC self)
        (->> self
             :spans
             vals
             (keep :text)
             (mapcat #(str/split % #" "))
             (map str/lower-case)
             (map word2vec/word-embedding)
             (doall)
             (apply math/unit-vec-sum factory)))))

(defrecord Sentence [concepts entities context]
  context/ContextVector
  (context-vector [self {:keys [structure-annotations concept-annotations factory] :as model}]
    (or (:VEC self)
        (let [context-toks (->> self
                                :context
                                (map #(get structure-annotations %)))]
          (->> self
               :entities
               (map (fn [e] (get concept-annotations e)))
               (lazy-cat context-toks)
               (map #(context/context-vector % model))
               (apply math/unit-vec-sum factory))))))

(defn pprint-sent
  [model sent]
  (->> sent
       (map #(get-in model [:structure-annotations %]))
       (map (comp first vals :spans))
       (sort-by :start)
       (map :text)
       (interpose " ")
       (apply str)))

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
  (->> toks
       (apply uber-alg/shortest-path undirected-sent)
       (uber-alg/nodes-in-path)))


(defn make-sentence
  "Make a sentence using the sentence graph and entities"
  [undirected-sent anns]
  ;; TODO: Remove context toks that are part of the entities
  (let [concepts (->> anns (map :concept) (map #(conj #{} %)) (set))
        context (->> anns (map :tok) (make-context-path undirected-sent))
        entities (->> anns (map :id) (set))]
    (->Sentence concepts entities context)))

(defn make-sentences
  [undirected-sent sent-annotations]
  (->> (clojure.math.combinatorics/combinations sent-annotations 2)
       (map #(make-sentence undirected-sent %))))

(defn concept-annotations->sentences
  [{:keys [concept-annotations structure-graphs]}]
  (log/info "Making sentences for concept annotations")
  (let [undirected-sents (util/map-kv graph/undirected-graph structure-graphs)]
    (->> concept-annotations
         (vals)
         (group-by :sent)
         (pmap (fn [[sent sent-annotations]]
                 (log/debug "Sentence:" sent)
                 (make-sentences (get undirected-sents sent) sent-annotations)))
         (apply concat))))

(defn assign-embedding
  [model tok]
  (assoc tok :VEC (context/context-vector tok model)))

(defn assign-sent-id
  [model tok]
  (assoc tok :sent (tok-sent-id model tok)))

(defn assign-tok
  [model ann]
  (let [{:keys [sent id]} (ann-tok model ann)]
    (assoc ann :tok id
               :sent sent)))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [{:keys [entities]}]
      (some #(= id %) entities))
    sentences))