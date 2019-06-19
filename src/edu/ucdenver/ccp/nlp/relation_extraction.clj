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
                  :max-matches
                  :match-fn})

(defn sent-pattern-concepts-match?
  [{:keys [concepts]} {:keys [support]}]
  (->> support
       (map :concepts)
       (some #(= concepts %))))

(defn log-starting-values
  [{:keys [properties seeds all-samples patterns]}]
  (let [p1 (util/map-kv count (group-by :predicted seeds))
        p2 (util/map-kv count (group-by :predicted patterns))]
    (->> properties
         (map (fn [property]
                {:seeds           (get p1 property)
                 :predicted       property
                 :samples         (count all-samples)
                 :patterns        (get p2 property)
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

(defn concept-context-match
  [{:keys [match-thresh samples patterns factory] :as model}]
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
          patterns              (->> patterns
                                     (map #(assoc % :VEC (re-model/context-vector % model)))
                                     (vec))]
      (->> filtered-samples
           (map #(re-model/context-vector % model))
           (linear-algebra/mdot factory (map :VEC patterns))
           (map vector filtered-samples)
           (pmap (fn [[sample sample-scores]]
                   (let [[j score] (apply max-key second (map-indexed vector sample-scores))
                         {:keys [predicted support]} (get patterns j)]
                     (if (< match-thresh score)
                       (assoc sample :predicted predicted
                                     :confidence (* score
                                                    (/ (count support)
                                                       (get max-cluster-support-m predicted))))
                       sample))))))))

(defn patterns-with-support-weight
  [patterns]
  (let [total-support (->> patterns
                           (map :support)
                           (map count)
                           (reduce +))]
    (->> patterns
         (vec)
         (map #(assoc % :weight (/ (count (:support %))
                                   total-support))))))

(defn support-weighted-sim-distribution-context-match
  [{:keys [samples patterns match-thresh factory] :as model}]
  (when (and (seq samples) (seq patterns))
    (log/info "Finding matches")
    (let [patterns (->> patterns
                        (patterns-with-support-weight)
                        (map #(assoc % :VEC (re-model/context-vector % model)))
                        (vec))
          samples  (vec samples)]
      (->> samples
           (map #(re-model/context-vector % model))
           (linear-algebra/mdot factory (map :VEC patterns))
           (map vector samples)
           (pmap (fn [[sample sample-scores]]
                   (let [[j score] (apply max-key second (map-indexed vector sample-scores))
                         {:keys [predicted weight]} (get patterns j)
                         weighted-best-score (* score weight)
                         weighted-scores     (->> patterns
                                                  (map :weight)
                                                  (map * sample-scores))]
                     (let [{:keys [p-value]} (inc-stats/t-test weighted-scores :mu weighted-best-score)
                           confidence (- 1 p-value)]
                       (if (< match-thresh confidence)
                         (assoc sample :predicted predicted
                                       :confidence confidence)
                         sample)))))))))

(defn support-pattern-scores
  [{:keys [match-thresh]} patterns sample-scores]
  (let [scores (->> (reduce (fn [[pattern-scores offset] pattern]
                              (let [support-count (count (:support pattern))
                                    new-offset    (+ offset support-count)
                                    scores        (->> new-offset
                                                       (range offset)
                                                       (select-keys sample-scores)
                                                       (map second))
                                    score         (apply max scores)
                                    good          (->> scores
                                                       (filter #(< match-thresh %))
                                                       (count))
                                    bad           (- support-count good)]
                                [(conj pattern-scores (assoc pattern
                                                        :score score
                                                        :scores scores
                                                        :good good
                                                        :bad bad))
                                 new-offset]))
                            [nil 0] patterns)
                    (first)
                    (filter (fn [{:keys [bad good]}] (< bad good))))]
    scores))


(defn support-weighted-sim-pattern-distribution-context-match
  [{:keys [samples patterns match-thresh factory] :as params}]
  (when (and (seq samples) (seq patterns))
    (log/info "Finding matches")
    (let [patterns        (patterns-with-support-weight patterns)
          samples         (vec samples)
          support         (mapcat :support patterns)
          support-vectors (map #(re-model/context-vector % params) support)]
      (->> samples
           (map #(re-model/context-vector % params))
           (linear-algebra/mdot factory support-vectors)
           (map vector samples)
           (pmap (fn [[sample sample-scores]]
                   (let [scores (support-pattern-scores params patterns sample-scores)]
                     (if (seq scores)
                       (let [{:keys [predicted weight] best-scores :scores} (apply max-key :score scores)
                             other-scores (->> scores
                                               (remove #(= (:predicted %) predicted))
                                               (mapcat #(map (fn [score]
                                                               (* score (:weight %)))
                                                             (:scores %))))
                             best-scores  (map #(* % weight) best-scores)]
                         (try
                           (let [mu         (inc-stats/mean other-scores)
                                 {:keys [p-value]} (inc-stats/t-test best-scores :mu mu)
                                 confidence (- 1 p-value)]
                             (if (< match-thresh confidence)
                               (assoc sample :predicted predicted
                                             :confidence confidence)
                               sample)
                             (assoc sample :predicted predicted
                                           :confidence confidence))
                           (catch Exception _
                             sample)))
                       sample))))))))

(defn sim-to-support-in-pattern-match
  [{:keys [samples patterns factory] :as params}]
  (let [patterns        (vec patterns)
        support         (mapcat :support patterns)
        support-vectors (map #(re-model/context-vector % params) support)
        samples         (vec samples)]
    (->> samples
         (map #(re-model/context-vector % params))
         (linear-algebra/mdot factory support-vectors)
         (map vector samples)
         (pmap (fn [[sample sample-scores]]
                 (let [scores (support-pattern-scores params patterns sample-scores)]
                   (if (seq scores)
                     (let [{:keys [score good predicted support]} (apply max-key :score scores)]
                       (if good
                         (assoc sample :predicted predicted
                                       :confidence (* score (/ good (count support))))
                         sample))
                     sample)))))))

(defn pattern-seed-match-scores
  [{:keys [seeds patterns factory] :as params}]
  (let [seeds           (vec seeds)
        pattern-vectors (map #(re-model/context-vector % params) patterns)
        sample-vectors  (map #(re-model/context-vector % params) seeds)]
    (log/info "Calculating seed matches")
    (->> pattern-vectors
         (linear-algebra/mdot factory sample-vectors)
         (map vector patterns)
         (pmap (fn [[pattern scores]]
                 (assoc pattern :seed-match-scores scores)))))
  patterns)

(defn pattern-seed-match-ratio
  [{:keys [seeds match-thresh] :as model}]
  (let [patterns (pattern-seed-match-scores model)
        seeds-m  (->> seeds
                      (map-indexed vector)
                      (group-by #(:predicted (second %)))
                      (util/map-kv #(map first %)))]
    (map (fn [p]
           (let [predicted-positive (->> p
                                         :seed-match-scores
                                         (filter #(< match-thresh %))
                                         (count))]
             (if (= 0 predicted-positive)
               (assoc p :recall 0
                        :precision 0
                        :f1 0)
               (let [tp              (->> p
                                          :predicted
                                          (get seeds-m)
                                          (select-keys (:seed-match-scores p))
                                          (vals)
                                          (filter #(< match-thresh %))
                                          (count))
                     actual-positive (count (get seeds-m (:predicted p)))
                     fp              (- predicted-positive tp)
                     fn              (- actual-positive tp)
                     precision       (/ tp predicted-positive)
                     recall          (/ tp actual-positive)]
                 (assoc p
                   :tp tp
                   :fp fp
                   :fn fn
                   :fn (- (count seeds) tp fp fn)
                   :recall recall
                   :precision precision
                   :f1 (if (or (= 0 precision) (= 0 recall))
                         0
                         (/ (* 2 precision recall)
                            (+ precision recall))))))))

         patterns)))



(defn pattern-update
  [{:keys [properties seeds patterns] :as model}]
  (let [seeds    (group-by :predicted seeds)
        patterns (group-by :predicted patterns)]
    (->> properties
         (pmap (fn [property]
                 (let [samples  (vec (get seeds property))
                       patterns (get patterns property)]
                   (if (seq samples)
                     (do
                       (log/info "Clustering" property)
                       (->> patterns
                            (cluster-tools/single-pass-cluster (assoc model
                                                                 :cluster-merge-fn re-model/add-to-pattern
                                                                 :vector-fn #(re-model/context-vector % model))
                                                               samples)
                            (map #(assoc % :predicted property))))
                     patterns))))

         (apply concat))))

(defn terminate?
  [{:keys [max-iterations iteration seeds matches patterns samples max-matches] :as model}]

  ;; Remaining samples added to negative group
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
  [{:keys [seeds factory match-fn confidence-thresh] :as model}]
  (let [model (assoc model :samples (->> model
                                         (context-path-filter)
                                         (map #(->> (re-model/context-vector % model)
                                                    (linear-algebra/unit-vec factory)
                                                    (assoc % :VEC))))
                           :matches #{}
                           :seeds seeds
                           :iteration 0)]
    (log/info (select-keys model PARAM-KEYS))
    (log-starting-values model)
    (loop [model model]
      (let [model       (update model :seeds (fn [seeds] (filter #(< confidence-thresh (:confidence %)) seeds)))
            model       (assoc model :patterns (pattern-update model))
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
        (log-current-values model)
        (if-let [results (terminate? model)]
          results
          (let [model (update model :iteration inc)
                model (update model :seeds (fn [new-matches] (->> new-matches
                                                                  (cap-nones)
                                                                  (lazy-cat unclustered))))]
            (recur model)))))))



