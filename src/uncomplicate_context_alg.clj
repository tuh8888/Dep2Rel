(ns uncomplicate-context-alg
  (:require [math :as math]
            [uncomplicate.commons.core :as uncomplicate]
            [uncomplicate.neanderthal.core :as thal]))

(defprotocol ContextVector
  (context-vector [self model]))

(defn context-vector-cosine-sim
  [{:keys [factory]} s1 s2 model]
  (uncomplicate/with-release [vec1 (math/unit-vec (thal/vctr factory (context-vector s1 model)))
                              vec2 (math/unit-vec (thal/vctr factory (context-vector s2 model)))]
    (if (and vec1 vec2)
      (thal/dot vec1 vec2)
      0)))
