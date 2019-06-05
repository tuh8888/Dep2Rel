(ns scripts.examples
  (:require [ubergraph.core :as uber]
            [clojure.string :as str]))

(def sentence "Little is known about genetic factors affecting intraocular pressure -LRB- IOP -RRB- in mice and other mammals .")

(def g (let [x (mapv keyword (str/split sentence #" "))]
         (uber/digraph [(get x 2) (get x 0) {:label :nsubjpass}]
                       [(get x 2) (get x 1) {:label :auxpass}]
                       [:ROOT (get x 2) {:label :root}]
                       [(get x 2) (get x 3) {:label :prep}]
                       [(get x 5) (get x 4) {:label :amod}]
                       [(get x 3) (get x 5) {:label :pobj}]
                       [(get x 5) (get x 6) {:label :acl}]
                       [(get x 8) (get x 7) {:label :amod}]
                       [(get x 6) (get x 8) {:label :dobj}]
                       [(get x 10) (get x 9) {:label :punct}]
                       [(get x 8) (get x 10) {:label :appos}]
                       [(get x 8) (get x 11) {:label :punct}]
                       [(get x 8) (get x 12) {:label :prep}]
                       [(get x 12) (get x 13) {:label :pobj}]
                       [(get x 13) (get x 14) {:label :cc}]
                       [(get x 16) (get x 15) {:label :amod}]
                       [(get x 13) (get x 16) {:label :conj}]
                       [(get x 2) (get x 17) {:label :punct}])))

(uber/viz-graph g {:save {:filename "resources/dep_example.png" :format :png}})

(def output-color :blue)
(def input-color :green)
(def important-color :red)

(def algorithm (uber/digraph [:text-sources {:color input-color}]
                             [:patterns {:color output-color}]
                             [:matches {:color output-color}]
                             [:text-sources :dependency-annotations]
                             [:text-sources :concept-annotations]
                             [:concept-annotations :context-paths]
                             [:dependency-annotations :context-paths]
                             [:context-paths :seeds]
                             [:context-paths :sentences]
                             [:seeds :patterns {:label :clustering}]
                             [:patterns :filtering]
                             [:sentences :filtering]
                             [:filtering :matches]
                             [:matches :seeds {:label :bootstrapping :color important-color}]))

(uber/viz-graph algorithm {:save {:filename "resources/algorithm.png" :format :png}})
