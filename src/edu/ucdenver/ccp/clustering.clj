(ns edu.ucdenver.ccp.clustering)

(defn nearest
  [clusters sample & [{:keys [cluster-match-fn]}]]
  (first
    (reduce
      (fn [[best-cluster best-score] cluster]
        (if-let [score (cluster-match-fn cluster sample best-score)]
          [cluster score]
          [best-cluster best-score]))
      [nil nil]
      clusters)))

(defn add-to-cluster
  [clusters cluster sample merge-fn]
  (let [clusters (disj clusters cluster)
        cluster (merge-fn cluster sample)]
    (conj clusters cluster)))

(defn add-new-cluster
  [clusters sample merge-fn]
  (conj clusters (merge-fn nil sample)))

(defn single-pass-cluster
  [samples clusters & [{:keys [cluster-merge-fn] :as params}]]
  (reduce
    (fn [clusters sample]
      (if-let [cluster (nearest clusters sample params)]
        (add-to-cluster clusters cluster sample cluster-merge-fn)
        (add-new-cluster clusters sample cluster-merge-fn)))
    clusters
    samples))