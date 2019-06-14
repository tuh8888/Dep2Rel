(ns scripts.relation-extraction-script
  (:require [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation])
  (:import (edu.ucdenver.ccp.knowtator.model KnowtatorModel)))

(log/set-level! :debug)

(def home-dir
  (io/file "/" "media" "tuh8888" "Seagate Expansion Drive" "data"))

(def annotations-file
  (io/file home-dir "craft-versions" "concepts+assertions1" "CRAFT_assertions.knowtator"))

(def annotations-file
  (io/file home-dir "craft-versions" "concepts+assertions64" "CRAFT_assertions.knowtator"))

(def annotations (k/view annotations-file))

;(.save annotations)

(def word-vector-dir
  (io/file home-dir "WordVectors"))
(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def model (k/simple-model annotations))

(def structures-annotations-with-embeddings
  (zipmap (keys (:structure-annotations model))
          (word2vec/with-word2vec word2vec-db
            (doall
              (pmap re-model/assign-word-embedding
                    (vals (:structure-annotations model)))))))

(def concepts-with-toks
  (zipmap (keys (:concept-annotations model))
          (pmap
            #(let [tok-id (re-model/ann-tok model %)
                   sent-id (re-model/tok-sent-id model tok-id)]
               (assoc % :tok tok-id
                        :sent sent-id))
            (vals (:concept-annotations model)))))

(def reasoner (k/reasoner annotations))

(def mem-descs
  (memoize
    (fn [c]
      (log/info c)
      (k/get-owl-descendants reasoner c))))

(def model (assoc model
             :concept-annotations concepts-with-toks
             :structure-annotations structures-annotations-with-embeddings))


(def sentences (->>
                 (re-model/concept-annotations->sentences model)
                 (map
                   #(update % :concepts
                            (fn [concepts]
                              (map
                                (fn [concept-set]
                                  (into concept-set (mem-descs (first concept-set))))
                                concepts))))))

(log/info "Num sentences:" (count sentences))


(def model (assoc model :sentences sentences))


;(k/display annotations)
;(k/selected-annotation annotations)

;; Mutation located in gene
(def property (.get (.getOwlObjectPropertyById ^KnowtatorModel (k/model annotations) "exists_at_or_derives_from")))



(defn c-metrics
  [matches]
  (math/calc-metrics {:predicted-positive
                      :actual-positive    actual-true
                      :all            all-triples}))

(comment
  (def actual-true-sentences (filter #(actual-true (set (:entities %))) sentences))
  (evaluation/cluster-sentences actual-true-sentences))

(comment

  (def matches (let [seeds (clojure.set/union
                             (evaluation/make-seeds sentences
                               "CRAFT_aggregate_ontology_Instance_21437"
                               "CRAFT_aggregate_ontology_Instance_22305")
                             (evaluation/make-seeds sentences
                               "CRAFT_aggregate_ontology_Instance_21365"
                               "CRAFT_aggregate_ontology_Instance_22495"))
                     seed-thresh 0.95
                     context-thresh 0.95
                     cluster-thresh 0.7
                     min-support 10
                     params {:seed             (first seeds)
                             :seed-thresh      seed-thresh
                             :context-thresh   context-thresh
                             :context-match-fn #(< context-thresh (re/context-vector-cosine-sim %1 %2))
                             :cluster-merge-fn re/add-to-pattern
                             :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                  (and (< (or %3 cluster-thresh) score)
                                                       score))
                             :min-seed-support min-support}
                     matches (->> (re/cluster-bootstrap-extract-relations seeds sentences params)
                                  (map #(merge % params)))]
                 (log/info "Metrics" (c-metrics matches))
                 matches))

  (def metrics (c-metrics matches))

  (log/info "Metrics" metrics)

  (def params {:predicted-positive
               :actual-positive    actual-true
               :all            all-triples}))

(comment
  (evaluation/format-matches model matches)
  (evaluation/->csv (io/file "." "matches.csv") matches model)

  (def param-results (evaluation/parameter-walk annotations
                                                "has_location_in"
                                                (clojure.set/intersection
                                                  (set (re-model/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21741"))
                                                  (set (re-model/sentences-with-ann sentences "CRAFT_aggregate_ontology_Instance_21947")))
                                                sentences))

  (def p2 (map last (partition 4 param-results)))
  (def p3 (map (fn [[a b c d]] [a b c (count d) (reduce + (map :num-matches d))]) (partition 4 param-results)))

  (count param-results)
  (let [f (io/file "." "params.csv")
        p p3
        col-names [:seed-thresh :cluster-thresh :min-seed-support :count :num-matches]
        csv-form (str (apply str col-names) "\n"
                      (apply str
                             (map
                               #(str (apply str (interpose "," %)) "\n")
                               p3)))]
    (spit f csv-form)))






