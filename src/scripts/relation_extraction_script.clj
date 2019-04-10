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
  (io/file home-dir "craft-versions" "concepts+assertions1" "CRAFT_assertions.knowtator"))

#_(def annotations-file
    (io/file home-dir "craft-versions" "concepts+assertions64" "CRAFT_assertions.knowtator"))

(def annotations (k/view annotations-file))

;(.save annotations)

(def word-vector-dir
  (io/file home-dir "WordVectors"))
(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def model
  (word2vec/with-word2vec word2vec-db
    (sentence/make-sentences (k/simple-model annotations))))

;(spit "/tmp/model.edn" (with-out-str (pr model)))

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
                                   (fn [concept-set]
                                     (into concept-set (mem-descs (first concept-set))))
                                   concepts))))
                    (doall)))

(t/info "Num sentences:" (count sentences))


;(k/display annotations)
;(k/selected-annotation annotations)

;; Mutation located in gene
(def property (.get (.getOwlObjectPropertyById ^KnowtatorModel (k/model annotations) "exists_at_or_derives_from")))

(def actual-true (set (map #(evaluation/edge->triple model %)
                           (k/edges-for-property model property))))
(def all-triples (set (map evaluation/sent->triple sentences)))
(defn predicted-true
  [matches]
  (set (map evaluation/sent->triple matches)))

(defn c-metrics
  [matches]
  (evaluation/calc-metrics {:predicted-true (predicted-true matches)
                            :actual-true    actual-true
                            :all            all-triples}))

(def matches (let [seeds (clojure.set/union
                           (clojure.set/intersection
                             (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21437"))
                             (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_22305")))
                           (clojure.set/intersection
                             (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21583"))
                             (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21881")))
                           (clojure.set/intersection
                             (set (sentence/sentences-with-ann sentences"CRAFT_aggregate_ontology_Instance_21543"))
                             (set (sentence/sentences-with-ann sentences"CRAFT_aggregate_ontology_Instance_21887"))))
                   seed-thresh 0.95
                   context-thresh 0.9
                   cluster-thresh 0.95
                   min-support 2
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
                                (map #(merge % params)))]
               (t/info "Metrics" (c-metrics matches))
               matches))

(def metrics (c-metrics matches))

(t/info "Metrics" metrics)

(evaluation/fn {:predicted-true (predicted-true matches)
                :actual-true    actual-true
                :all            all-triples})

(comment
  (evaluation/format-matches model matches)
  (evaluation/->csv (io/file "." "matches.csv") matches model)

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






