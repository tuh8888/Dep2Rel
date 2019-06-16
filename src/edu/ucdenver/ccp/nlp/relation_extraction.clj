(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [linear-algebra :as linear-algebra]
            [util :as util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]))

(def PARAM-KEYS [:match-thresh
                 :cluster-thresh
                 :context-path-length-cap
                 :confidence-thresh
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
                {:seeds           (get p1 property)
                 :property        property
                 :samples         (count all-samples)
                 :actual-positive (count (re-model/actual-positive property all-samples))}))
         (incanter/to-dataset)
         (log/info))))

(defn log-current-values
  [{:keys [properties samples] :as model}]
  (let [p1 (->> [:patterns :seeds :matches :patterns]
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
  [{:keys [match-thresh vector-fn samples patterns factory]}]
  #_(log/info (count (remove vector-fn samples)) (count (remove vector-fn patterns)))
  (when (and (seq samples) (seq patterns))
    (log/info "Finding matches")
    (let [max-cluster-support-m (->> patterns
                                     (group-by :predicted)
                                     (util/map-kv #(map :support %))
                                     (util/map-kv #(map count %))
                                     (util/map-kv #(reduce max %)))
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
          new-matches           (->> sample-vectors
                                     (linear-algebra/find-best-col-matches factory pattern-vectors)
                                     (filter (fn [{:keys [score]}] (< match-thresh score)))
                                     (map #(let [s (get filtered-samples (:j %))
                                                 p (get patterns (:i %))]
                                             (when-not s (log/warn (:j %) "sample not found"))
                                             (when-not p (log/warn (:i %) "pattern not found"))
                                             [s (assoc p :score (:score %))]))
                                     (into {}))]
      (map (fn [s] (let [p (get new-matches s)]
                     (if (sent-pattern-concepts-match? s p)
                       (assoc s :predicted (:predicted p)
                                :confidence (* (:score p)
                                               (/ (count (:support p))
                                                  (get max-cluster-support-m (:predicted p)))))
                       s)))
           samples))))


(defn pattern-update
  [{:keys [properties seeds patterns confidence-thresh] :as model}]
  (let [seeds    (->> seeds
                      (filter #(< confidence-thresh (:confidence %)))
                      (group-by :predicted))
        patterns (group-by :predicted patterns)]
    (mapcat (fn [property]
              (let [samples  (get seeds property)
                    patterns (get patterns property)]
                (log/info "Clustering" property)
                (->> samples
                     (partition-all 1000)
                     (mapcat (fn [sample-part]
                               (->> patterns
                                    (cluster-tools/single-pass-cluster model sample-part)
                                    (map #(assoc % :predicted property))))))))
            properties)))

(defn terminate?
  [{:keys [max-iterations iteration seeds matches patterns samples] :as model}]

  ;; Remaining matches added to negative group
  (let [success-model (assoc model :matches (->> samples
                                                 (map #(assoc % :predicted re-model/NONE))
                                                 (into matches))
                                   :patterns patterns)]
    (cond (<= max-iterations iteration)
          (do (log/info "Max iteration reached")
              success-model)
          (empty? seeds)
          (do (log/info "No new matches")
              success-model)
          (empty? samples)
          (do (log/info "No more samples")
              success-model)
          (empty? (remove #(= re-model/NONE %) (map :property samples)))
          (do (log/info "Only negative examples left")
              success-model))))

(defn support-filter
  [{:keys [min-match-support seeds]} pattern]
  (or (empty? seeds)
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
                           :seeds seeds
                           :iteration 0)]
    (log/info (re-params model))
    (log-starting-values model)
    (loop [model model]
      (let [model       (assoc model :patterns (pattern-update model))
            unclustered (decluster model)
            model       (update model :patterns (fn [patterns] (filter (fn [pattern] (support-filter model pattern)) patterns)))
            model       (assoc model :seeds (concept-context-match model))
            model       (update model :samples (fn [samples] (let [new-matches (:seeds model)]
                                                               (if (seq new-matches)
                                                                 (remove :predicted new-matches)
                                                                 samples))))
            model       (update model :seeds (fn [new-matches] (filter :predicted new-matches)))
            model       (update model :matches (fn [matches] (->> model
                                                                  :seeds
                                                                  (into matches))))]
        (if-let [results (terminate? model)]
          results
          (do
            (log-current-values model)
            (let [model (update model :iteration inc)
                  model (update model :seeds (fn [new-matches] (->> new-matches
                                                                    (cap-nones)
                                                                    (lazy-cat unclustered))))]
              (recur model))))))))



