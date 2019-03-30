(ns util
  (:require [uncomplicate.neanderthal.core :refer [nrm2 dot scal xpy]]))

(defn unit-vec
  [v]
  (scal (/ (nrm2 v)) v))

(defn cosine-sim
  [v1 v2]
  (dot (unit-vec v1)
       (unit-vec v2)))

(defn unit-vec-sum
  [& vectors]
  (if (<= 2 (count vectors))
    (util/unit-vec (apply xpy vectors))
    (util/unit-vec (first vectors))))

(defn parse-int
  [x]
  (try (Integer/parseInt x)
       (catch NumberFormatException _
         x)))

(defn find-matches
  [coll1 coll2 match-fn]
  (filter
    (fn [s1]
      (some
        (fn [s2]
          (match-fn s1 s2))
        coll2))
    coll1))
