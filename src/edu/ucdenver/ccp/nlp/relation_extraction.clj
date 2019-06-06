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
  (map->Pattern {:concepts       (into (set (:concepts p)) (:concepts s))
                 :context-vector (if (:context-vector p)
                                   (when-let [vectors (seq (keep :context-vector [p s]))]
                                     (apply math/unit-vec-sum vectors))
                                   (:context-vector s))
                 :support        (conj (set (:support p)) s)}))

(defn concepts-match?
  [sample seed]
  (every?
    (fn [sample-concept-set]
      (some
        (fn [seed-concept-set]
          (clojure.set/subset? sample-concept-set seed-concept-set))
        (set (:concepts seed))))
    (set (:concepts sample))))

(defn context-vector-cosine-sim
  [s1 s2]
  (let [vec1 (:context-vector s1)
        vec2 (:context-vector s2)]
    (if (and vec1 vec2)
      (math/cosine-sim vec1 vec2)
      0)))

(defn bootstrap
  [seeds sentences & [{:keys [terminate? context-match-fn pattern-update-fn]}]]
  (log/info "Seeds" (count seeds))
  (loop [iteration 0
         new-matches (set seeds)
         matches #{}
         patterns #{}
         sentences sentences]
    (t/debug "Patterns" (count patterns))
    (t/debug "Matches" (count matches))
    (let [patterns (pattern-update-fn seeds new-matches matches patterns)
          new-matches (util/find-matches sentences patterns context-match-fn)
          matches (into matches new-matches)
          sentences (remove (set new-matches) sentences)]

      (if-let [results (terminate? iteration seeds new-matches matches patterns sentences)]
        results
        (recur (inc iteration) new-matches matches patterns sentences)))))

