(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [linear-algebra :as linear-algebra]
            [util :as util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
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
    (log/info "Finding matches" (count patterns) (count samples))
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
  [{:keys [vector-fn samples patterns match-thresh properties] :as model}]
  (when (and (seq samples) (seq patterns))
    (log/info "Finding matches" (count patterns) (count samples))
    (let [samples                      (vec samples)
          patterns                     (vec patterns)
          sample-vectors               (map vector-fn samples)
          pattern-vectors              (map vector-fn patterns)
          predicted-pattern-property-m (group-by :predicted patterns)
          property-support-m           (->> predicted-pattern-property-m
                                            (util/map-kv #(map :support %))
                                            (util/map-kv #(map count %))
                                            (util/map-kv #(reduce + %)))
          other-property-weights       (->> properties
                                            (map (fn [property]
                                                   (let [other-patterns-support       (->> patterns
                                                                                           (remove #(= (:predicted %) property))
                                                                                           (map :support)
                                                                                           (map count))
                                                         other-property-total-support (->> property
                                                                                           (dissoc property-support-m)
                                                                                           (vals)
                                                                                           (reduce +))]
                                                     [property (map #(/ % other-property-total-support) other-patterns-support)])))
                                            (into {}))
          scores                       (cluster-tools/update-score-cache model sample-vectors pattern-vectors nil 0)]
      (->> samples
           (map-indexed vector)
           (pmap (fn [[i sample]]
                   (let [sample-scores                  (filter #(= (:i %) i) scores)
                         {best-score :score best-j :j :as best} (apply max-key :score sample-scores)
                         {best-predicted :predicted best-support :support :as best-pattern} (get patterns best-j)
                         other-properties-sample-scores (remove (fn [{:keys [j]}] (= best-predicted
                                                                                     (get-in patterns [j :predicted])))
                                                                sample-scores)
                         weights                        (get other-property-weights best-predicted)
                         weighted-scores                (->> other-properties-sample-scores
                                                             (map :score)
                                                             (map * weights))]
                     (let [{:keys [p-value]} (incanter.stats/t-test weighted-scores :mu best-score)
                           confidence (- 1 p-value)]
                       (if (< match-thresh confidence)
                         (assoc sample :predicted best-predicted
                                       :confidence confidence)
                         sample)))))))))

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
                                (log/info (count samples) (count patterns))
                                (if (seq samples)
                                  (do
                                    (log/info "Clustering" property)
                                    (->> patterns
                                         (cluster-tools/single-pass-cluster model samples)
                                         (map #(assoc % :predicted property))))
                                  patterns))))
                      (apply concat))]
    (log/info (count seeds) (count patterns))
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
          (<= max-matches (count (remove #(= (:property %) re-model/NONE) matches)))
          (do (log/info "Max matches")
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
    (log/info (re-params model))
    (log-starting-values model)
    (loop [model model]
      (log/info (count (:seeds model)) (count (:patterns model)))
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



