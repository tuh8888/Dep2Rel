(ns edu.ucdenver.ccp.nlp.re-model
  (:require [clojure.string :as str]
            [ubergraph.alg :as uber-alg]
            [ubergraph.core :as uber]
            [graph :as graph]
            [math :as math]
            [util :as util]
            [word2vec :as word2vec]
            [taoensso.timbre :as log]
            [uncomplicate-context-alg :as context]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [incanter.core :as incanter])
  (:import (clojure.lang PersistentArrayMap)))

(extend-type PersistentArrayMap
  context/ContextVector
  (context-vector [self {:keys [factory]}]
    (or (:VEC self)
        (->> self
             :spans
             vals
             (keep :text)
             (mapcat #(str/split % #" |-"))
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
               (doall)
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

(defn min-start
  [a]
  (->> a
       :spans
       (vals)
       (map :start)
       (reduce min)))

(defn max-end
  [a]
  (->> a
       :spans
       (vals)
       (map :start)
       (reduce min)))

(defn overlap
  "Finds overlap between two annotations"
  [a1 a2]
  (let [min-a1-start (min-start a1)
        min-a2-start (min-start a2)
        max-a1-end (max-end a1)
        max-a2-end (max-end a2)]
    (<= min-a1-start min-a2-start max-a2-end max-a1-end)))

(defn remove-context-toks-in-entities
  [{:keys [structure-annotations concept-annotations]} entities context]
  (let [context-toks (map #(get structure-annotations %) context)
        entity-anns (map #(get concept-annotations %) entities)]
    (->> context-toks
         (remove (fn [tok]
                   (some (fn [ann]
                           (overlap ann tok))
                         entity-anns)))
         (map :id))))


(defn make-sentence
  "Make a sentence using the sentence graph and entities"
  [model undirected-sent anns]
  ;; TODO: Remove context toks that are part of the entities
  (let [concepts (->> anns
                      (map :concept)
                      (map #(conj #{} %))
                      (set))
        entities (->> anns
                      (map :id)
                      (set))
        context (->> anns
                     (map :tok)
                     (make-context-path undirected-sent)
                     (remove-context-toks-in-entities model entities))]
    (->Sentence concepts entities context)))

(defn combination-sentences
  [model undirected-sent sent-annotations]
  (->> (clojure.math.combinatorics/combinations sent-annotations 2)
       (map #(make-sentence model undirected-sent %))))

(defn concept-annotations->sentences
  [{:keys [concept-annotations structure-graphs] :as model}]
  (let [undirected-sents (util/map-kv graph/undirected-graph structure-graphs)]
    (->> concept-annotations
         (vals)
         (group-by :sent)
         (pmap (fn [[sent sent-annotations]]
                 (log/debug "Sentence:" sent)
                 (combination-sentences model (get undirected-sents sent) sent-annotations)))
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

(defn sent-property
  [{:keys [concept-graphs]} [id1 id2]]
  (some
    (fn [g]
      (when-let [e (or (uber/find-edge g id2 id1) (uber/find-edge g id1 id2))]
        (:value (uber/attrs g e))))
    (vals concept-graphs)))

(defn assign-property
  "Assign the associated property with the sentence"
  [model s]
  (assoc s :property (sent-property model (vec (:entities s)))))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [{:keys [entities]}]
      (some #(= id %) entities))
    sentences))

(defn make-model
  [v factory]
  (log/info "Making model")
  (let [model (as-> (k/simple-model v) model
                    (assoc model :factory factory)
                    (update model :structure-annotations (fn [structure-annotations]
                                                           (log/info "Making structure annotations")
                                                           (util/pmap-kv (fn [s]
                                                                           (->> s
                                                                                (assign-embedding model)
                                                                                (assign-sent-id model)))
                                                                         structure-annotations)))
                    (update model :concept-annotations (fn [concept-annotations]
                                                         (log/info "Making concept annotations")
                                                         (util/pmap-kv (fn [s]
                                                                         (assign-tok model s))
                                                                       concept-annotations))))]
    (log/info "Model" (util/map-kv count (dissoc model :factory)))
    model))

(defn make-sentences
  [model]
  (log/info "Making sentences")
  (let [sentences (->> model
                       (concept-annotations->sentences)
                       (pmap #(assign-property model %)))]
    (log/info "Num sentences:" (count sentences))
    (log/info "Num sentences with property:" (->> sentences
                                                  (group-by :property)
                                                  (util/map-kv count)))
    sentences))

(defn frac-seeds
  [property sentences frac seed]
  (let [pot (->> sentences
                 (filter #(= (:property %) property))
                 (util/deterministic-shuffle seed))]
    (-> pot
        (count)
        (* frac)
        (take pot)
        (set))))

(defn split-train-test
  "Splits model into train and test sets"
  [sentences model frac properties seed]
  (let [seeds (->> (group-by :property sentences)
                   (filter #(properties (first %)))
                   (map (fn [[property sentences]] (frac-seeds property sentences frac seed)))
                   (apply clojure.set/union))]

    (-> model
        (assoc :samples (remove seeds sentences)
               :seeds (->> seeds
                           (map #(assoc % :predicted (:property %)))
                           (set)))
        (update :samples (fn [samples] (->> samples
                                            (map #(assign-embedding model %))
                                            (doall))))
        (update :seeds (fn [seeds] (->> seeds
                                        (map #(assign-embedding model %))
                                        (doall))))
        (assoc :properties properties))))