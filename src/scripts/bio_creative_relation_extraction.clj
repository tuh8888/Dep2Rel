(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
    #_[incanter.stats :as stats]
    #_[incanter.core :as incanter]
    #_[incanter.charts :as charts]
            [org.clojurenlp.core :as corenlp]))

(def home-dir (io/file "/" "home" "harrison"))
#_(def home-dir (io/file "/" "media" "harrison" "Seagate Expansion Drive" "data"))
(def biocreative-dir (io/file home-dir "BioCreative" "BCVI-2017" "ChemProt_Corpus"))
(def training-dir (io/file biocreative-dir "chemprot_training"))
(def word-vector-dir (io/file home-dir "WordVectors"))
(def word2vec-db (.getAbsolutePath (io/file word-vector-dir "bio-word-vectors-clj.vec")))

;(comment
;  (def sentences-dir (io/file training-dir "chem_prot_training_sentences"))
;  (doseq [f (file-seq (io/file training-dir "Articles"))]
;    (let [content (slurp f)
;          content (->> content
;                       (corenlp/sentenize)
;                       (map :text)
;                       (interpose "\n")
;                       (apply str))
;          (io/file sentences-dir (str (name k) ".txt"))]
;      (spit sentence-f content))))

(def annotations (k/model training-dir nil))

#_(rdr/biocreative-read-abstracts (k/model annotations) (io/file training-dir "chemprot_training_abstracts.tsv"))
#_(rdr/biocreative-read-entities (k/model annotations) (io/file training-dir "chemprot_training_entities.tsv"))
#_(rdr/biocreative-read-relations (k/model annotations) (io/file training-dir "chemprot_training_relations.tsv"))
;; Read from conll files
;#_(rdr/biocreative-read-dependency annotations training-dir word2vec-db)
#_(.save (k/model annotations))

(def model (let [model (k/simple-model annotations)
                 structures-annotations-with-embeddings (word2vec/with-word2vec word2vec-db
                                                          (sentence/structures-annotations-with-embeddings model))

                 concept-annotations-with-toks (sentence/concept-annotations-with-toks model)

                 model (assoc model
                         :concept-annotations concept-annotations-with-toks
                         :structure-annotations structures-annotations-with-embeddings)
                 sentences (sentence/concept-annotations->sentences model)]

             (assoc model :sentences sentences)))

(get-in model [:structure-annotations (sentence/annotation-tok-id model (get (:concept-annotations model) "23402364-T37"))])
(get-in model [:structure-annotations "23402364-859768"])
(map #(:text (first (vals (get-in model [:structure-annotations % :spans])))) (keys (get-in model [:structure-graphs "23402364-Sentence 1" :node-map])))
(log/info "Num sentences:" (count (:sentences model)))

(def property "INHIBITOR")
(def dep-filt 10)

;; #{"12871155-T7" "12871155-T20"} has a ridiculously long context due to the number of tokens in 4-amino-6,7,8,9-tetrahydro-2,3-diphenyl-5H-cyclohepta[e]thieno[2,3-b]pyridine
;;(filter #(= 35 (count (:context %))) (make-all-seeds model property (:sentences model) 100))

;;; CLUSTERING ;;;

(-> (evaluation/make-all-seeds model property (:sentences model))
    (cluster-tools/single-pass-cluster #{}
                                       {:cluster-merge-fn re/add-to-pattern
                                        :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                             (and (< (or %3 0.75) score)
                                                                  score))})
    (count))
#_(let [x (range -3 3 0.1)]
    (incanter/view
      (charts/dynamic-scatter-plot
        [cluster-similarity-score-threshold (range 0 1 0.01)]
        [x (cluster-tools/single-pass-cluster sentences #{}
                                              {:cluster-merge-fn re/add-to-pattern
                                               :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                                    (and (< (or %3 cluster-similarity-score-threshold) score)
                                                                         score))})])))

;;; RELATION EXTRACTION

(def matches (let [num-seeds 100
                   sentences (evaluation/context-filt dep-filt (:sentences model))
                   seeds (take num-seeds (evaluation/make-all-seeds model property sentences))
                   ;seed-thresh 0.85
                   context-thresh 0.9
                   cluster-thresh 0.75
                   min-support 10
                   params {:seed             (first seeds)
                           ;:seed-thresh      seed-thresh
                           :context-thresh   context-thresh
                           ;:seed-match-fn    #(and (re/concepts-match? %1 %2)
                           ;                        (< seed-thresh (re/context-vector-cosine-sim %1 %2)))
                           :context-match-fn #(< context-thresh (re/context-vector-cosine-sim %1 %2))
                           :cluster-merge-fn re/add-to-pattern
                           :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                (and (< (or %3 cluster-thresh) score)
                                                     score))
                           :min-support      min-support}
                   matches (->> (re/cluster-bootstrap-extract-relations seeds sentences params)
                                (map #(merge % params)))]
               (log/info "Metrics:" (math/calc-metrics {:predicted-true (evaluation/predicted-true matches)
                                                        :actual-true    (evaluation/actual-true model property)
                                                        :all            (evaluation/all-triples model)}))
               matches))

(evaluation/format-matches model matches)