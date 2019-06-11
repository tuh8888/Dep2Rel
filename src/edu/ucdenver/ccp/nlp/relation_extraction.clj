(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [math :as math]
            [util :as util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [uncomplicate-context-alg :as context]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]))

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
    (sentence/assign-embedding model p)))

(defn bootstrap
  [{:keys [properties seeds samples] :as model} {:keys [terminate? context-match-fn pattern-update-fn
                                                        support-filter decluster]}]
  (log/info
    "\nSeeds" (util/map-kv count (group-by :predicted seeds))
    "\nSamples" (count samples))
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
      (log/info "\nPatterns" (util/map-kv count (group-by :predicted patterns))
                "\nNew matches " (util/map-kv count (group-by :predicted new-matches))
                "\nMatches " (util/map-kv count (group-by :predicted matches)))
      (if-let [results (terminate? model {:iteration        iteration
                                          :seeds            seeds
                                          :new-matches      new-matches
                                          :matches          matches
                                          :patterns         patterns
                                          :samples          samples
                                          :last-new-matches last-new-matches})]
        results
        (recur (inc iteration) new-matches matches patterns samples)))))

