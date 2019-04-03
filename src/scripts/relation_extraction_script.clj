(ns scripts.relation-extraction-script
  (:require [edu.ucdenver.ccp.nlp.relation-extraction :refer :all]
            [clojure.java.io :as io]
            [taoensso.timbre :as t]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [edu.ucdenver.ccp.clustering :refer [single-pass-cluster]]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [util :refer [cosine-sim]]
            [clojure.set :as set1]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]))
(t/set-level! :debug)

(def home-dir
  (io/file "/" "media" "tuh8888" "Seagate Expansion Drive" "data"))

(def craft-dir
  (io/file home-dir "craft-versions" "concepts+assertions_1_article"))
(def word-vector-dir
  (io/file home-dir "WordVectors"))

(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))


(def annotations-file
  (io/file craft-dir "concepts+assertions.knowtator"))

(def dependency-dir
  (io/file craft-dir "Structures"))
(def references-dir
  (io/file craft-dir "Articles"))


(def articles
  [(first (rdr/article-names-in-dir references-dir "txt"))])

(def references (rdr/read-references articles references-dir))
(def annotations (k/view annotations-file))
(def dependency (rdr/read-dependency word2vec-db articles references dependency-dir))
(def sentences (rdr/read-sentences annotations dependency articles))
(t/info "Num sentences:" (count sentences))


(comment
  (k/display annotations)
  (k/selected-annotation annotations)

  ;; Mutation located in gene
  (def matches (let [property "has_location_in"
                     seeds (set1/intersection
                             (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21741"))
                             (set (sentence/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21947")))
                     seed-thresh 0.9
                     context-thresh 0.9
                     cluster-thresh 0.75
                     min-support 20
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
                                  (map #(let [t (evaluation/matched-triples % annotations property)]
                                          (assoc % :num-matches (count t) :triples t))))]
                 (t/info "Final matches:" (count matches))
                 (t/info "Triples matched" (count (distinct (mapcat :triples matches))))
                 matches))

  (evaluation/format-matches matches)
  (evaluation/to-csv (io/file "." "matches.csv") matches)

  (def param-results (evaluation/parameter-walk annotations
                                                "has_location_in"
                                                (set1/intersection
                                                  (set (get-sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21741"))
                                                  (set (get-sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21947")))
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






