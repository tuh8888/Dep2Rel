(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [incanter.stats :as stats]
            [incanter.core :as incanter]
            [incanter.charts :as charts]))

(def home-dir (io/file "/" "home" "harrison"))

#_(def home-dir
    (io/file "/" "media" "harrison" "Seagate Expansion Drive" "data"))

(def biocreative-dir
  (io/file home-dir "BioCreative" "BCVI-2017" "ChemProt_Corpus"))

(def training-dir
  (io/file biocreative-dir "chemprot_training"))
(def word-vector-dir
  (io/file home-dir "WordVectors"))

(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def annotations (k/model training-dir nil))

(def abstracts-f (io/file training-dir "chemprot_training_abstracts.tsv"))
(rdr/biocreative-read-abstracts (k/model annotations) abstracts-f)

(def entities-f (io/file training-dir "chemprot_training_entities.tsv"))
(rdr/biocreative-read-entities (k/model annotations) entities-f)

(def relations-f (io/file training-dir "chemprot_training_relations.tsv"))
(rdr/biocreative-read-relations (k/model annotations) relations-f)
#_(.save (k/model annotations))

(comment
  ;; Read from conll files
  (rdr/biocreative-read-dependency annotations training-dir word2vec-db))


(def model1 (k/simple-model annotations))

(def structures-annotations-with-embeddings (word2vec/with-word2vec word2vec-db
                                              (sentence/structures-annotations-with-embeddings model1)))

(def concept-annotations-with-toks (sentence/concept-annotations-with-toks model1))

(def model (assoc model1
             :concept-annotations concept-annotations-with-toks
             :structure-annotations structures-annotations-with-embeddings))

(def sentences (sentence/concept-annotations->sentences model))
(log/info "Num sentences:" (count sentences))


(let [x (range -3 3 0.1)]
  (incanter/view (charts/dynamic-scatter-plot [cluster-similarity-score-threshold (range 0 1 0.01)]
                                              [x (cluster-tools/single-pass-cluster sentences #{}
                                                                                    {:cluster-merge-fn re/add-to-pattern
                                                                                     :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                                                                          (and (< (or %3 cluster-similarity-score-threshold) score)
                                                                                                               score))})])))

(comment
  (def matches (let [property "INHIBITOR"

                     ;sentences (filter #(<= (count (:context %)) 2) sentences)
                     actual-true (set (->> property
                                           (k/edges-for-property model)
                                           (map evaluation/edge->triple)
                                           (filter (fn [t] (some #(= t (:entities %)) sentences)))))
                     all-triples (set (map evaluation/sent->triple sentences))

                     seeds (clojure.set/union
                             (apply evaluation/make-seeds sentences (first actual-true))
                             (apply evaluation/make-seeds sentences (second actual-true)))
                     seed-thresh 0.85
                     context-thresh 0.9
                     cluster-thresh 0.95
                     min-support 1
                     params {:seed             (first seeds)
                             :seed-thresh      seed-thresh
                             :context-thresh   context-thresh
                             :seed-match-fn    #(and (re/concepts-match? %1 %2)
                                                     (< seed-thresh (re/context-vector-cosine-sim %1 %2)))
                             :context-match-fn #(< context-thresh (re/context-vector-cosine-sim %1 %2))
                             :cluster-merge-fn re/add-to-pattern
                             :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                  (and (< (or %3 cluster-thresh) score)
                                                       score))
                             :min-support      min-support}
                     matches (->> (re/cluster-bootstrap-extract-relations seeds sentences params)
                                  (map #(merge % params)))]
                 (log/info "Metrics:" (math/calc-metrics {:predicted-true (evaluation/predicted-true matches)
                                                          :actual-true    actual-true
                                                          :all            all-triples}))
                 matches)))





;(comment
;  (def sentences (rdr/sentenize annotations))
;  (first sentences)
;
;  (def sentences-dir dependency-dir)
;  (doall
;    (map
;      (fn [[k v]]
;        (let [sentence-f (io/file sentences-dir (str (name k) ".txt"))
;              content (apply str (interpose "\n" (map :text v)))]
;          (spit sentence-f content)))
;      sentences)))