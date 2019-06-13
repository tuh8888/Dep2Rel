(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [linear-algebra :as linear-algebra]
            [util :as util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]))

(def PARAM-KEYS [:context-thresh
                 :cluster-thresh
                 :context-path-length-cap
                 :rng
                 :seed-frac
                 :min-match-support
                 :re-clustering?
                 :max-iterations 100
                 :max-matches 3000])

(defn re-params
  [model]
  (->> PARAM-KEYS
       (map #(find model %))
       (into {})))

(defn sent-pattern-concepts-match?
  [{:keys [concepts]} {:keys [support]}]
  (->> support
       (map :concepts)
       (some #(= concepts %))))

(defn log-starting-values
  [{:keys [properties seeds all-samples]}]
  (let [p1 (util/map-kv count (group-by :predicted seeds))]
    (->> properties
         (map (fn [property]
                {:Seeds    (get p1 property)
                 :Property property
                 :Samples  (count all-samples)}))
         (incanter/to-dataset)
         (log/info))))

(defn log-current-values
  [{:keys [properties] :as model}]
  (let [p1 (->> [:patterns :new-matches :matches :patterns]
                (map #(find model %))
                (into {})
                (util/map-kv #(->> %
                                   (group-by :predicted)
                                   (util/map-kv count))))]
    (->> properties
         (map (fn [property]
                (util/map-kv #(get % property)
                             p1)))
         (incanter/to-dataset)
         (log/info))))

(defn cap-nones
  [matches]
  (let [nones (filter #(= re-model/NONE (:property %)) matches)
        others (remove #(= re-model/NONE (:property %)) matches)
        num-nones-to-keep (->> others
                               (group-by :predicted)
                               (util/map-kv count)
                               (vals)
                               (reduce max))
        nones (take num-nones-to-keep nones)]
    (lazy-cat nones others)))

(defn bootstrap
  [{:keys [seeds all-samples
           terminate? context-match-fn pattern-update-fn
           context-path-filter-fn
           support-filter decluster]

    :as   model}]
  (log-starting-values model)
  (loop [iteration 0
         new-matches (set seeds)
         matches #{}
         patterns #{}
         samples (context-path-filter-fn model all-samples)]
    (let [model (assoc model :patterns patterns
                             :matches matches
                             :new-matches new-matches
                             :samples samples
                             :iteration iteration)
          model (update model :patterns (fn [patterns] (pattern-update-fn model patterns)))
          unclustered (decluster model)
          model (update model :patterns (fn [patterns] (filter (fn [pattern] (support-filter model pattern)) patterns)))
          model (assoc model :new-matches (context-match-fn model))
          model (assoc model :samples (->> model
                                           :new-matches
                                           (remove :predicted)))
          model (update model :new-matches (fn [new-matches] (filter :predicted new-matches)))
          model (update model :matches (fn [matches] (->> model :new-matches
                                                          (into matches))))]
      (if-let [results (terminate? model)]
        results
        (do
          (log-current-values model)
          (recur (inc iteration) (->> new-matches
                                      (cap-nones)
                                      (lazy-cat unclustered)) matches patterns samples))))))

(defn concept-filter
  "Filter samples that don't share pattern concepts"
  [samples patterns]
  (filter (fn [s]
            (some (fn [p]
                    (sent-pattern-concepts-match? s p))
                  patterns))
          samples))

(defn concept-context-match
  [{:keys [context-thresh vector-fn samples patterns] :as params}]
  #_(log/info (count (remove vector-fn samples)) (count (remove vector-fn patterns)))
  (when (and (seq samples) (seq patterns))
    (let [filtered-samples (-> samples
                               (concept-filter patterns)
                               (vec))
          patterns (vec patterns)
          sample-vectors (->> filtered-samples
                              (map vector-fn)
                              (pmap #(linear-algebra/unit-vec params %))
                              (vec))
          pattern-vectors (->> patterns
                               (map vector-fn)
                               (pmap #(linear-algebra/unit-vec params %))
                               (vec))
          matches-map (->> pattern-vectors
                           (linear-algebra/find-best-row-matches params sample-vectors)
                           (filter (fn [{:keys [score]}] (< context-thresh score)))
                           (map #(let [s (get filtered-samples (:i %))
                                       p (get patterns (:j %))]
                                   (when-not s (log/warn (:i %) "sample not found"))
                                   (when-not p (log/warn (:j %) "pattern not found"))
                                   [s p]))
                           (filter (fn [[s p]] (sent-pattern-concepts-match? s p)))
                           (into {}))]
      (map (fn [s]
             (let [match (get matches-map s)]
               (assoc s :predicted (:predicted match))))
           samples))))

(defn pattern-update
  [{:keys [properties new-matches] :as model} patterns]
  (mapcat (fn [property]
            (let [samples (filter #(= (:predicted %) property) new-matches)
                  patterns (filter #(= (:predicted %) property) patterns)]
              (when (seq samples)
                (log/info "Clustering" property)
                (->> samples
                     (partition-all 1000)
                     (mapcat (fn [sample-part]
                               (->> patterns
                                    (cluster-tools/single-pass-cluster model sample-part)
                                    (map #(assoc % :predicted property)))))))))
          properties))

(defn terminate?
  [{:keys [max-iterations max-matches
           iteration seeds
           new-matches matches patterns samples] :as model}]

  ;; Remaining matches added to negative group
  (let [success-model (assoc model :matches (->> samples
                                                 (map #(assoc % :predicted re-model/NONE))
                                                 (into matches))
                                   :patterns patterns)]
    (cond (<= max-iterations iteration)
          (do (log/info "Max iteration reached")
              success-model)
          (empty? new-matches)
          (do (log/info "No new matches")
              success-model)
          (empty? samples)
          (do (log/info "No more samples")
              success-model)
          (<= max-matches (count (remove #(= re-model/NONE (:property %)) matches)))
          (do (log/info "Too many matches")
              model)
          (empty? seeds)
          (do (log/info "No seeds")
              model)
          (empty? (remove #(= re-model/NONE %) (map :property samples)))
          (do (log/info "Only negative examples left")
              success-model))))

(defn support-filter
  [{:keys [min-match-support new-matches]} pattern]
  (or (empty? new-matches)
      (->> pattern
           :support
           (count)
           (<= min-match-support))))


(defn decluster
  [{:keys [re-clustering? support-filter patterns] :as model}]
  (when re-clustering?
    (->> patterns
         (remove #(support-filter model %))
         (mapcat :support))))

(defn context-path-filter
  [{:keys [context-path-length-cap]} coll]
  (filter #(<= (count (:context %)) context-path-length-cap) coll))

