(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [taoensso.timbre :as t]
            [math]
            [util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]
            [uncomplicate.commons.core :as uncomplicate]
            [uncomplicate.neanderthal.core :as thal]
            [uncomplicate.neanderthal.native :as thal-native]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]))

(defrecord Pattern [context-vector support])

(defn sent-pattern-concepts-match?
  [{:keys [concepts]} {:keys [support]}]
  (->> support
       (map :concepts)
       (some #(= concepts %))))

(defn add-to-pattern
  [p s]
  (map->Pattern {:context-vector (if (:context-vector p)
                                   (sentence/sum-vectors (map :context-vector [p s]))
                                   (:context-vector s))
                 :support        (conj (set (:support p)) s)}))

(defn context-vector-cosine-sim
  [s1 s2]
  (uncomplicate/with-release [vec1 (thal-native/dv (:context-vector s1))
                              vec2 (thal-native/dv (:context-vector s2))]
    (if (and vec1 vec2)
      (thal/dot vec1 vec2)
      0)))

(defn bootstrap
  [{:keys [properties seeds samples] :as model} {:keys [terminate? context-match-fn pattern-update-fn]}]
  (log/info "Seeds" (util/map-kv count (group-by :predicted seeds)))
  (loop [iteration 0
         new-matches (set seeds)
         matches #{}
         patterns #{}
         samples samples]
    (let [[patterns unclustered] (mapcat #(pattern-update-fn new-matches patterns %) properties)
          last-new-matches new-matches
          new-matches (-> samples
                           (context-match-fn patterns)
                           (lazy-cat unclustered))
          samples (remove :predicted new-matches)
          new-matches (filter :predicted new-matches)
          matches (into matches new-matches)]
      (t/debug "\nPatterns" (util/map-kv count (group-by :predicted patterns))
               "\nNew matches " (util/map-kv count (group-by :predicted new-matches))
               "\nMatches " (util/map-kv count (group-by :predicted matches)))
      (if-let [results (terminate? model {:iteration   iteration
                                          :seeds       seeds
                                          :new-matches new-matches
                                          :matches     matches
                                          :patterns    patterns
                                          :samples   samples
                                          :last-new-matches last-new-matches})]
        results
        (recur (inc iteration) new-matches matches patterns samples)))))

