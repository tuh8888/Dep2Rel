(ns edu.ucdenver.ccp.nlp.relation-extraction
  (:require [taoensso.timbre :as t]
            [math]
            [util]
            [clojure.set :refer [subset? intersection]]
            [edu.ucdenver.ccp.clustering :refer [single-pass-cluster]]))

(defrecord Pattern [context-vector support])

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

(defn naive-extract-relations
  "Finds sentences with the same concepts as the seeds.
  These are then used to find other sentences with similar contexts but potentially
  different concepts."
  [seeds sentences & [{:keys [seed-match-fn context-match-fn]}]]
  (let [seed-matches (util/find-matches sentences seeds seed-match-fn)]
    (t/info "Seeds:" (count seed-matches))
    (into seeds (util/find-matches sentences seed-matches context-match-fn))))

(defn cluster-extract-relations
  [seeds sentences & [{:keys [seed-match-fn context-match-fn min-support] :as params}]]
  (let [seeds (into seeds (util/find-matches sentences seeds seed-match-fn))
        patterns (cluster-tools/single-pass-cluster seeds #{} params)
        patterns (filter #(<= min-support (count (:support %))) patterns)]
    (t/info "Seeds" (count seeds))
    (t/info "Patterns" (count patterns))
    (into seeds (util/find-matches sentences patterns context-match-fn))))

(defn bootstrap
  [seeds sentences update-fn]
  (loop [matches (set seeds)
         samples sentences]
    (let [new-matches (set (update-fn matches samples))
          num-new-matches (count (clojure.set/difference new-matches matches))]
      (t/info "New matches" num-new-matches)
      (if (= num-new-matches 0)
        matches
        (recur new-matches (remove #(new-matches %) samples))))))

(defn naive-bootstrap-extract-relations
  [seeds sentences & [params]]
  (bootstrap seeds sentences #(naive-extract-relations %1 %2 params)))

(defn cluster-bootstrap-extract-relations
  [seeds sentences & [params]]
  (bootstrap seeds sentences #(cluster-extract-relations %1 %2 params)))

#_(defn extract-all-relations
    [seed-thresh cluster-thresh min-support sources]
    (let [all-concepts (map
                         #(sentence/->Entity % nil nil nil nil)
                         (knowtator/all-concepts (:annotations sources)))]
      (info "All concepts" (count all-concepts))
      (->> (combo/combinations all-concepts 2)
           (pmap set)
           (pmap #(sentence/->Sentence % nil nil))
           (pmap
             #(map
                (fn [r]
                  (assoc r :seed %))
                (extract-relations (list %) (:sentences sources)
                                   :seed-thresh seed-thresh
                                   :cluster-thresh cluster-thresh
                                   :min-support min-support
                                   :max-iter 100)))
           (mapcat identity))))
