(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [math :as math]
            [util :as util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [uncomplicate-context-alg :as context]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [incanter.core :as incanter])
  (:import (org.tukaani.xz.lz Matches)))

(defrecord Pattern [support]
  context/ContextVector
  (context-vector [self {:keys [factory] :as model}]
    (or (:VEC self)
        (->> self
             :support
             (map #(context/context-vector % model))
             (apply math/unit-vec-sum factory)))))

(defn sent-pattern-concepts-match?
  [{:keys [concepts]} {:keys [support]}]
  (->> support
       (map :concepts)
       (some #(= concepts %))))

(defn add-to-pattern
  [model p s]
  (let [p (->Pattern (conj (set (:support p)) s))]
    (re-model/assign-embedding model p)))

(defn bootstrap
  [{:keys [properties seeds samples] :as model} {:keys [terminate? context-match-fn pattern-update-fn
                                                        support-filter decluster]}]
  (let [p1 (util/map-kv count (group-by :predicted seeds))]
    (->> properties
         (map (fn [property]
                {:Seeds       (get p1 property)
                 :Property    property
                 :Samples (count samples)}))
         (incanter/to-dataset)
         (log/info)))
  (loop [iteration 0
         new-matches (set seeds)
         matches #{}
         patterns #{}
         samples samples]
    (let [patterns (mapcat #(pattern-update-fn new-matches patterns %) properties)
          unclustered (decluster new-matches patterns)
          patterns (filter #(support-filter new-matches %) patterns)
          last-new-matches new-matches
          new-matches (-> samples
                          (context-match-fn patterns)
                          (lazy-cat unclustered))
          samples (remove :predicted new-matches)
          new-matches (filter :predicted new-matches)
          matches (into matches new-matches)]
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
             (log/info)))
      (if-let [results (terminate? model {:iteration        iteration
                                          :seeds            seeds
                                          :new-matches      new-matches
                                          :matches          matches
                                          :patterns         patterns
                                          :samples          samples
                                          :last-new-matches last-new-matches})]
        results
        (recur (inc iteration) new-matches matches patterns samples)))))


(defn concept-context-match
  [{:keys [context-thresh vector-fn] :as params} samples patterns]
  #_(log/info (count (remove vector-fn samples)) (count (remove vector-fn patterns)))
  (when (and (seq samples) (seq patterns))
    (->> patterns
         (math/find-best-row-matches params samples)
         (map (fn [{:keys [score] :as best}] (if (< context-thresh score)
                                               best
                                               (dissoc best :match))))
         (map (fn [{:keys [sample match] :as best}] (if (sent-pattern-concepts-match? sample match)
                                                      best
                                                      (dissoc best :match))))
         (map (fn [{:keys [sample match]}]
                (assoc sample :predicted (:predicted match)))))))

(defn pattern-update
  [params model new-matches patterns property]
  (let [samples (->> new-matches
                     (filter #(= (:predicted %) property))
                     (set))
        params (merge params {:cluster-merge-fn (partial add-to-pattern model)})]
    (->> patterns
         (filter #(= (:predicted %) property))
         (set)
         (cluster-tools/single-pass-cluster params samples)
         (map #(assoc % :predicted property)))))

(defn terminate?
  [{:keys [max-iterations max-matches]} model
   {:keys [iteration seeds new-matches matches patterns samples last-new-matches]}]
  (let [success-model (assoc model :matches matches
                                   :patterns patterns)]
    (cond (<= max-iterations iteration) (do (log/info "Max iteration reached")
                                            success-model)
          (= last-new-matches new-matches) (do (log/info "No new matches")
                                               success-model)
          (empty? samples) (do (log/info "No more samples")
                               success-model)
          (<= max-matches (count matches)) (do (log/info "Too many matches")
                                               model)
          (empty? seeds) (do (log/info "No seeds")
                             model))))

(defn support-filter
  [{:keys [min-match-support]} new-matches p]
  (or (empty? new-matches) (<= min-match-support (count (:support p)))))

(defn decluster
  [{:keys [re-clustering?]} support-filter new-matches patterns]
  (when re-clustering?
    (->> patterns
         (remove #(support-filter new-matches %))
         (mapcat :support))))

