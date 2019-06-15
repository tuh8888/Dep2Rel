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
                {:seeds    (get p1 property)
                 :property property
                 :samples  (count all-samples)
                 :actual-positive (count  (re-model/actual-positive property all-samples))}))
         (incanter/to-dataset)
         (log/info))))

(defn log-current-values
  [{:keys [properties samples] :as model}]
  (let [p1 (->> [:patterns :new-matches :matches :patterns]
                (map #(find model %))
                (into {})
                (util/map-kv #(->> %
                                   (group-by :predicted)
                                   (util/map-kv count))))]
    (->> properties
         (map (fn [property]
                (assoc (util/map-kv #(get % property) p1)
                  :property property
                  :samples (count samples))))
         (incanter/to-dataset)
         (log/info))))

(defn cap-nones
  [matches]
  (let [nones             (filter #(= re-model/NONE (:property %)) matches)
        others            (remove #(= re-model/NONE (:property %)) matches)
        num-nones-to-keep (->> others
                               (group-by :predicted)
                               (util/map-kv count)
                               (vals)
                               (reduce max 0))

        nones             (take num-nones-to-keep nones)]
    (lazy-cat nones others)))

(defn concept-filter
  "Filter samples that don't share pattern concepts"
  [samples patterns]
  (filter (fn [s]
            (some (fn [p]
                    (sent-pattern-concepts-match? s p))
                  patterns))
          samples))

(defn concept-context-match
  [{:keys [context-thresh vector-fn samples patterns factory]}]
  #_(log/info (count (remove vector-fn samples)) (count (remove vector-fn patterns)))
  (when (and (seq samples) (seq patterns))
    (log/info "Finding matches")
    (let [max-cluster-support-m (->> patterns
                                     (group-by :property)
                                     (util/map-kv :support)
                                     (util/map-kv count))
          filtered-samples      (-> samples
                                    (concept-filter patterns)
                                    (vec))
          patterns              (vec patterns)
          sample-vectors        (->> filtered-samples
                                     (map vector-fn)
                                     (pmap #(linear-algebra/unit-vec factory %))
                                     (vec))
          pattern-vectors       (->> patterns
                                     (map vector-fn)
                                     (pmap #(linear-algebra/unit-vec factory %))
                                     (vec))
          matches-map           (->> sample-vectors
                                     (linear-algebra/find-best-col-matches factory pattern-vectors)
                                     (filter (fn [{:keys [score]}] (< context-thresh score)))
                                     (map #(let [s (get filtered-samples (:j %))
                                                 p (get patterns (:i %))]
                                             (when-not s (log/warn (:j %) "sample not found"))
                                             (when-not p (log/warn (:i %) "pattern not found"))
                                             [s [p (:score %)]]))
                                     (filter (fn [[s p]] (sent-pattern-concepts-match? s p)))
                                     (into {}))]
      (map (fn [s]
             (let [[match score] (get matches-map s)]
               (assoc s :predicted (:predicted match)
                        :confidence (* score
                                       (/ (count (:support match))
                                          (get max-cluster-support-m (:predicted match)))))))
           samples))))

(defn pattern-update
  [{:keys [properties new-matches patterns confidence-thresh] :as model}]
  (mapcat (fn [property]
            (let [samples  (->> new-matches
                                (filter #(< (:confidence %) confidence-thresh))
                                (filter #(= (:predicted %) property)))
                  patterns (filter #(= (:predicted %) property) patterns)]
              (if (seq samples)
                (do
                  (log/info "Clustering" property)
                  (->> samples
                       (partition-all 1000)
                       (mapcat (fn [sample-part]
                                 (->> patterns
                                      (cluster-tools/single-pass-cluster model sample-part)
                                      (map #(assoc % :predicted property)))))))
                patterns)))
          properties))

(defn terminate?
  [{:keys [max-iterations iteration seeds new-matches matches patterns samples] :as model}]

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
  [{:keys [context-path-length-cap all-samples]}]
  (filter #(<= (count (:context %)) context-path-length-cap) all-samples))

(defn bootstrap
  [{:keys [seeds factory vector-fn] :as model}]
  (let [model (assoc model :samples (->> model
                                         (context-path-filter)
                                         (map #(->> %
                                                    (vector-fn)
                                                    (linear-algebra/unit-vec factory)
                                                    (assoc % :VEC))))
                           :matches #{}
                           :new-matches seeds
                           :iteration 0)]
    (log/info (re-params model))
    (log-starting-values model)
    (loop [model model]
      (let [model       (assoc model :patterns (pattern-update model))
            unclustered (decluster model)
            model       (update model :patterns (fn [patterns] (filter (fn [pattern] (support-filter model pattern)) patterns)))
            model       (assoc model :new-matches (concept-context-match model))
            model       (update model :samples (fn [samples] (let [new-matches (:new-matches model)]
                                                               (if (seq new-matches)
                                                                 (remove :predicted new-matches)
                                                                 samples))))
            model       (update model :new-matches (fn [new-matches] (filter :predicted new-matches)))
            model       (update model :matches (fn [matches] (->> model
                                                                  :new-matches
                                                                  (into matches))))]
        (if-let [results (terminate? model)]
          results
          (do
            (log-current-values model)
            (let [model (update model :iteration inc)
                  model (update model :new-matches (fn [new-matches] (->> new-matches
                                                                          (cap-nones)
                                                                          (lazy-cat unclustered))))]
              (recur model))))))))



