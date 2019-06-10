(ns uncomplicate-context-alg
  (:require [math :as math]
            [uncomplicate.commons.core :as uncomplicate]
            [uncomplicate.neanderthal.native :as thal-native]
            [uncomplicate.neanderthal.core :as thal]))

(defprotocol ContextVector
  (context-vector [self model]))

(defn context-vector-cosine-sim
  [s1 s2 model]
  (uncomplicate/with-release [vec1 (math/unit-vec (thal-native/dv (context-vector s1 model)))
                              vec2 (math/unit-vec (thal-native/dv (context-vector s2 model)))]
    (if (and vec1 vec2)
      (thal/dot vec1 vec2)
      0)))

(defn context-matrix
  [params model coll]
  (uncomplicate/with-release [vectors (map #(context-vector % model) coll)]
    (math/vectors->matrix params vectors)))
