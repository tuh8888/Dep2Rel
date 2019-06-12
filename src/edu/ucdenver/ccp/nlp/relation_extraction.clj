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
                 :max-iterations          100
                 :max-matches             3000])

(defn model-params
  [model]
  (->> model
       (juxt PARAM-KEYS)
       (interpose PARAM-KEYS)
       (into {})))

(defn sent-pattern-concepts-match?
  [{:keys [concepts]} {:keys [support]}]
  (->> support
       (map :concepts)
       (some #(= concepts %))))



(defn log-starting-values
  [{:keys [properties seeds samples]}]
  (let [p1 (util/map-kv count (group-by :predicted seeds))]
    (->> properties
         (map (fn [property]
                {:Seeds    (get p1 property)
                 :Property property
                 :Samples  (count samples)}))
         (incanter/to-dataset)
         (log/info))))

(defn log-current-values
  [properties new-matches matches patterns]
  (let [p1 (util/map-kv count (group-by :predicted patterns))
        p2 (util/map-kv count (group-by :predicted new-matches))
        p3 (util/map-kv count (group-by :predicted matches))]
    (->> properties
         (map (fn [property]
                {:Patterns    (get p1 property)
                 :New-Matches (get p2 property)
                 :Matches     (get p3 property)
                 :Property    property}))
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
  [{:keys [properties seeds samples
           terminate? context-match-fn pattern-update-fn
           support-filter decluster]
    :as model}]
  (log-starting-values model)
  (loop [iteration 0
         new-matches (set seeds)
         matches #{}
         patterns #{}
         samples samples]
    (let [patterns (mapcat #(pattern-update-fn model new-matches patterns %) properties)
          unclustered (decluster model new-matches patterns)
          patterns (filter #(support-filter model new-matches %) patterns)
          new-matches (context-match-fn model samples patterns)
          samples (remove :predicted new-matches)
          new-matches (filter :predicted new-matches)
          matches (into matches new-matches)
          new-matches-and-unclustered (->> new-matches
                                           (cap-nones)
                                           (lazy-cat unclustered))]
      (if-let [results (terminate? model {:iteration   iteration
                                          :seeds       seeds
                                          :new-matches new-matches
                                          :matches     matches
                                          :patterns    patterns
                                          :samples     samples})]
        results
        (do
          (log-current-values properties new-matches matches patterns)
          (recur (inc iteration) new-matches-and-unclustered matches patterns samples))))))


(defn concept-context-match
  [{:keys [context-thresh vector-fn] :as params} samples patterns]
  #_(log/info (count (remove vector-fn samples)) (count (remove vector-fn patterns)))
  (when (and (seq samples) (seq patterns))
    (let [samples (vec samples)
          patterns (vec patterns)
          sample-vectors (->> samples
                              (map vector-fn)
                              (pmap #(linear-algebra/unit-vec params %))
                              (vec))
          pattern-vectors (->> patterns
                               (map vector-fn)
                               (pmap #(linear-algebra/unit-vec params %))
                               (vec))]
      (->> sample-vectors
           (linear-algebra/find-best-row-matches params pattern-vectors)
           (map #(let [s (get samples (:i %))]
                   (when-not s (log/warn (:i %) "sample not found"))
                   (assoc % :sample s)))
           (map #(let [p (get patterns (:j %))]
                   (when-not p (log/warn (:j %) "pattern not found"))
                   (assoc % :match p)))
           (map (fn [{:keys [score] :as best}] (if (< context-thresh score)
                                                 best
                                                 (dissoc best :match))))
           (map (fn [{:keys [sample match] :as best}] (if (sent-pattern-concepts-match? sample match)
                                                        best
                                                        (dissoc best :match))))
           (map (fn [{:keys [sample match]}]
                  (assoc sample :predicted (:predicted match))))))))

(defn pattern-update
  [model new-matches patterns property]
  (let [samples (->> new-matches
                     (filter #(= (:predicted %) property))
                     (set))]
    (->> patterns
         (filter #(= (:predicted %) property))
         (set)
         (cluster-tools/single-pass-cluster model samples)
         (map #(assoc % :predicted property)))))

(defn terminate?
  [{:keys [max-iterations max-matches] :as model}
   {:keys [iteration seeds new-matches matches patterns samples]}]
  (let [success-model (assoc model :matches matches
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
  [{:keys [min-match-support]} new-matches p]
  (or (empty? new-matches) (<= min-match-support (count (:support p)))))

(defn decluster
  [{:keys [re-clustering? support-filter]} new-matches patterns]
  (when re-clustering?
    (->> patterns
         (remove #(support-filter new-matches %))
         (mapcat :support))))

