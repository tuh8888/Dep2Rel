(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [taoensso.timbre :as t]
            [math]
            [util]
            [cluster-tools]
            [clojure.set :refer [subset? intersection]]
            [taoensso.timbre :as log]))

(defrecord Pattern [context-vector support])

(defn sent-pattern-concepts-match?
  [{:keys [concepts]} {:keys [support]}]
  (->> support
       (map :concepts)
       (some #(= concepts %))))

(defn add-to-pattern
  [p s]
  (map->Pattern {:context-vector (if (:context-vector p)
                                   (when-let [vectors (seq (keep :context-vector [p s]))]
                                     (apply math/unit-vec-sum vectors))
                                   (:context-vector s))
                 :support (conj (set (:support p)) s)}))

(defn context-vector-cosine-sim
  [s1 s2]
  (let [vec1 (:context-vector s1)
        vec2 (:context-vector s2)]
    (if (and vec1 vec2)
      (math/cosine-sim vec1 vec2)
      0)))

(defn bootstrap
  [{:keys [properties seeds samples] :as model} {:keys [terminate? context-match-fn pattern-update-fn]}]
  (log/info "Seeds" (util/map-kv count (group-by :predicted seeds)))
  (loop [iteration 0
         new-matches (set seeds)
         matches #{}
         patterns #{}
         samples samples]
    (let [patterns (mapcat #(pattern-update-fn new-matches patterns %) properties)
          new-matches (map #(context-match-fn % patterns) samples)
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
                                          :samples   samples})]
        results
        (recur (inc iteration) new-matches matches patterns samples)))))

