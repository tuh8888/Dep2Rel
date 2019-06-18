(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [linear-algebra :as linear-algebra]
            [util :as util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [incanter.stats :as inc-stats]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]))

(def PARAM-KEYS #{:match-thresh
                  :cluster-thresh
                  :context-path-length-cap
                  :confidence-thresh
                  :rng
                  :seed-frac
                  :min-pattern-support
                  :re-clustering?
                  :max-iterations
                  :max-matches})

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
                 :predicted       property
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
                  :predicted property
                  :samples (count samples))))
         (incanter/to-dataset)
         (log/info))))

(defn cap-nones
  [matches]
  (let [nones             (filter #(= re-model/NONE (:predicted %)) matches)
        others            (remove #(= re-model/NONE (:predicted %)) matches)
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

#_(defn concept-context-match
    [{:keys [match-thresh vector-fn samples patterns factory]}]
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
            sample-vectors        (map vector-fn filtered-samples)
            pattern-vectors       (map vector-fn patterns)
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

(defn support-weighted-sim-distribution-context-match
  [{:keys [vector-fn samples patterns match-thresh factory]}]
  (when (and (seq samples) (seq patterns))
    (log/info "Finding matches")
    (let [samples                (vec samples)
          patterns               (vec patterns)
          pattern-vectors        (map vector-fn patterns)
          pattern-support-counts (->> patterns
                                      (map :support)
                                      (map count))
          total-support          (reduce + pattern-support-counts)
          pattern-weights        (map #(/ % total-support) pattern-support-counts)]
      (->> samples
           (map vector-fn)
           (linear-algebra/mdot factory pattern-vectors)
           (map vector samples)
           (pmap (fn [[sample sample-scores]]
                   (let [[best-j best-score] (apply max-key second (map-indexed vector sample-scores))
                         {best-predicted :predicted best-support :support} (get patterns best-j)
                         weighted-best-score (* best-score (/ (count best-support)
                                                              total-support))
                         other-scores        sample-scores
                         weights             pattern-weights
                         weighted-scores     (map * other-scores weights)]
                     (let [{:keys [p-value]} (inc-stats/t-test weighted-scores :mu weighted-best-score)
                           confidence (- 1 p-value)]
                       (if (< match-thresh confidence)
                         (assoc sample :predicted best-predicted
                                       :confidence confidence)
                         sample)))))))))

(defn sim-to-support-in-pattern-match
  [{:keys [samples vector-fn patterns match-thresh factory]}]
  (let [patterns        (vec patterns)
        support         (mapcat :support patterns)
        support-vectors (map vector-fn support)
        samples         (vec samples)]
    (->> samples
         (map vector-fn)
         (linear-algebra/mdot factory support-vectors)
         (map vector samples)
         (map (fn [[sample sample-scores]]
                (let [[best-score best-pattern good] (->> patterns
                                                          (reduce (fn [[[best-score _ _ :as best] offset] pattern]
                                                                    (let [support-count  (count (:support pattern))
                                                                          support-offset (+ offset support-count)
                                                                          pattern-scores (->> support-offset
                                                                                              (range offset)
                                                                                              (select-keys sample-scores)
                                                                                              (map second))]
                                                                      (let [best-pattern-score (apply max pattern-scores)
                                                                            good               (->> pattern-scores
                                                                                                    (filter #(< match-thresh %))
                                                                                                    (count))
                                                                            bad                (- support-count good)]

                                                                        (if (and (< bad good)
                                                                                 (< best-score best-pattern-score))
                                                                          [[best-pattern-score pattern good] support-offset]
                                                                          [best support-count]))))
                                                                  [[match-thresh nil nil] 0])
                                                          (first))]
                  (if good
                    (assoc sample :predicted (:predicted best-pattern)
                                  :confidence (* best-score (/ good
                                                               (count (:support best-pattern)))))
                    sample)))))))


(defn pattern-update
  [{:keys [properties seeds patterns confidence-thresh] :as model}]
  (let [seeds    (->> seeds
                      (filter #(< confidence-thresh (:confidence %)))
                      (group-by :predicted))
        patterns (group-by :predicted patterns)
        patterns (->> properties
                      (pmap (fn [property]
                              (let [samples  (get seeds property)
                                    patterns (get patterns property)]
                                (if (seq samples)
                                  (do
                                    (log/info "Clustering" property)
                                    (->> patterns
                                         (cluster-tools/single-pass-cluster model samples)
                                         (map #(assoc % :predicted property))))
                                  patterns))))
                      (apply concat))]
    patterns))


(defn terminate?
  [{:keys [max-iterations iteration seeds matches patterns samples max-matches] :as model}]

  ;; Remaining matches added to negative group
  (let [success-model (assoc model :matches (->> samples
                                                 (map #(assoc % :predicted re-model/NONE))
                                                 (into matches))
                                   :patterns patterns)]
    (cond (<= max-iterations iteration)
          (do (log/info "Max iteration reached")
              success-model)
          (<= max-matches (count (remove #(= (:predicted %) re-model/NONE) matches)))
          (do (log/info "Max matches")
              success-model)
          (empty? seeds)
          (do (log/info "No new matches")
              success-model)
          (empty? samples)
          (do (log/info "No more samples")
              success-model)
          (empty? (remove #(= re-model/NONE %) (map :predicted seeds)))
          (do (log/info "Only negative examples left")
              success-model))))

(defn support-filter
  [{:keys [min-pattern-support]} pattern]
  (<= min-pattern-support (count (:support pattern))))


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
  [{:keys [seeds factory vector-fn match-fn] :as model}]
  (let [model (assoc model :samples (->> model
                                         (context-path-filter)
                                         (map #(->> %
                                                    (vector-fn)
                                                    (linear-algebra/unit-vec factory)
                                                    (assoc % :VEC))))
                           :matches #{}
                           :seeds seeds
                           :iteration 0)]
    (log/info (select-keys model PARAM-KEYS))
    (log-starting-values model)
    (loop [model model]
      (let [model       (assoc model :patterns (pattern-update model))
            unclustered (decluster model)
            model       (update model :patterns (fn [patterns] (filter (fn [pattern] (support-filter model pattern)) patterns)))
            model       (assoc model :seeds (match-fn model))
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



