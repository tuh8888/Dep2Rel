(ns scripts.relation-extraction-script
  (:require [edu.ucdenver.ccp.nlp.relation-extraction :refer :all]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [taoensso.timbre :as t]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation])
  (:import (edu.ucdenver.ccp.knowtator.model KnowtatorModel)))
(t/set-level! :debug)

(def home-dir
  (io/file "/" "media" "tuh8888" "Seagate Expansion Drive" "data"))

(def annotations-file
  (io/file home-dir "craft-versions" "concepts+assertions_1_article" "concepts+assertions.knowtator"))

(def annotations-file
  (io/file home-dir "craft-versions" "concepts+assertions_64" "CRAFT_assertions.knowtator" ))

(def ^KnowtatorModel annotations (k/model annotations-file nil))

(.save annotations)

(def word-vector-dir
  (io/file home-dir "WordVectors"))
(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def model
  (word2vec/with-word2vec word2vec-db
    (sentence/make-sentences (k/simple-model annotations))))

(def reasoner (k/reasoner annotations))

(def mem-descs
  (memoize
    (fn [c]
      (t/info c)
      (k/get-owl-descendants reasoner c))))

(def sentences (->> model
                    :sentences
                    (map
                      #(update % :concepts
                               (fn [concepts]
                                 (map
                                   (fn [concepts]
                                     (into concepts (mem-descs (first concepts))))
                                   concepts))))
                    (doall)))

(t/info "Num sentences:" (count sentences))

(comment
  (defn sent->triple
    [match]
    (set (map :id (:entities match))))

  (defn edge->triple
    [model edge]
    (let [concept-annotations (:concept-annotations model)
          source (->> edge
                      :src
                      (get concept-annotations)
                      :id)
          target (->> edge
                      :dest
                      (get concept-annotations)
                      :id)]
      #{source target}))


  ;(k/display annotations)
  ;(k/selected-annotation annotations)

  ;; Mutation located in gene
  (def property (.get (.getOwlObjectPropertyById annotations "has_location_in")))
  (def possible-triples (map #(edge->triple model %) (k/triples-for-property model property)))

  (defn c-metrics
    [matches]
    (evaluation/calc-metrics (set (map sent->triple matches)) (set possible-triples) (set (map sent->triple sentences))))

  (def matches (let [seeds (into
                             (clojure.set/intersection
                               (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21741"))
                               (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21947")))
                             (clojure.set/intersection
                               (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21945"))
                               (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21819"))))
                     seed-thresh 0.975
                     context-thresh 0.9
                     cluster-thresh 0.9
                     min-support 10
                     params {:seed             (first seeds)
                             :seed-thresh      seed-thresh
                             :context-thresh   context-thresh
                             :seed-match-fn    #(and (concepts-match? %1 %2)
                                                     (< seed-thresh (context-vector-cosine-sim %1 %2)))
                             :context-match-fn #(< context-thresh (context-vector-cosine-sim %1 %2))
                             :cluster-merge-fn add-to-pattern
                             :cluster-match-fn #(let [score (context-vector-cosine-sim %1 %2)]
                                                  (and (< (or %3 cluster-thresh) score)
                                                       score))
                             :min-support      min-support}
                     matches (->> (cluster-bootstrap-extract-relations seeds sentences params)
                                  (map #(merge % params))
                                  (map #(let [matched (filter
                                                        (fn [triple]
                                                          (= triple (sent->triple %)))
                                                        possible-triples)]
                                          (assoc % :num-matches (count matched)
                                                   :triples matched))))]
                 (t/info "Final matches:" (count matches))
                 (t/info "Triples matched" (count (distinct (mapcat :triples matches))))
                 (t/info "Metrics" (c-metrics matches))
                 matches))

  (def metrics (c-metrics matches))

  (t/info "Metrics" metrics)

  (clojure.set/difference (set possible-triples) (set (mapcat :triples matches)))

  (evaluation/format-matches model matches)
  (evaluation/to-csv (io/file "." "matches.csv") matches model)

  (def param-results (evaluation/parameter-walk annotations
                                                "has_location_in"
                                                (clojure.set/intersection
                                                  (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21741"))
                                                  (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21947")))
                                                sentences))

  (def p2 (map last (partition 4 param-results)))
  (def p3 (map (fn [[a b c d]] [a b c (count d) (reduce + (map :num-matches d))]) (partition 4 param-results)))

  (count param-results)
  (let [f (io/file "." "params.csv")
        p p3
        col-names [:seed-thresh :cluster-thresh :min-support :count :num-matches]
        csv-form (str (apply str col-names) "\n"
                      (apply str
                             (map
                               #(str (apply str (interpose "," %)) "\n")
                               p3)))]
    (spit f csv-form)))






