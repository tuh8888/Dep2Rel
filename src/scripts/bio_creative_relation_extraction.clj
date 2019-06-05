(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            #_[incanter.stats :as stats]
            #_[incanter.core :as incanter]
            #_[incanter.charts :as charts]))

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

#_(get-in model [:structure-annotations (sentence/annotation-tok-id model (get (:concept-annotations model) "23402364-T37"))])
#_(get-in model [:structure-annotations "23402364-859768"])
#_(map #(:text (first (vals (get-in model [:structure-annotations % :spans])))) (keys (get-in model [:structure-graphs "23402364-Sentence 1" :node-map])))
(log/info "Num sentences:" (count (:sentences model)))

(def property "INHIBITOR")

;; #{"12871155-T7" "12871155-T20"} has a ridiculously long context due to the number of tokens in 4-amino-6,7,8,9-tetrahydro-2,3-diphenyl-5H-cyclohepta[e]thieno[2,3-b]pyridine
;;(filter #(= 35 (count (:context %))) (make-all-seeds model property (:sentences model) 100))

;;; CLUSTERING ;;;

(comment
  (-> (evaluation/make-all-seeds model property (:sentences model))
      (cluster-tools/single-pass-cluster #{}
                                         {:cluster-merge-fn re/add-to-pattern
                                          :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                               (and (< (or %3 0.75) score)
                                                                    score))})
      (count)))
#_(let [x (range -3 3 0.1)]
    (incanter/view
      (charts/dynamic-scatter-plot
        [cluster-similarity-score-threshold (range 0 1 0.01)]
        [x (cluster-tools/single-pass-cluster sentences #{}
                                              {:cluster-merge-fn re/add-to-pattern
                                               :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                                    (and (< (or %3 cluster-similarity-score-threshold) score)
                                                                         score))})])))

;;; PCA ;;;
#_(def X (->> model :sentences
            (map :context-vector)
            (mapv seq)
            (incanter/to-dataset)
            (incanter/to-matrix)))
#_(def pca (stats/principal-components X))

#_(def components (:rotation pca))
#_(def pc1 (incanter/sel components :cols 0))
#_(def pc2 (incanter/sel components :cols 1))
#_(def x1 (incanter/mmult X pc1))
#_(def x2 (incanter/mmult X pc2))
#_(incanter/view (charts/scatter-plot x1 x2
                    :x-label "PC1"
                    :y-label "PC2"
                    :title "PCA"))

;;; RELATION EXTRACTION

#_(def matches (let [context-path-length-cap 10
                   sentences (->> model
                                  :sentences
                                  #_(evaluation/context-path-filter context-path-length-cap))
                   seed-frac 0.2
                   context-thresh 0.95
                   cluster-thresh 0.95
                   min-support 1
                   params {:seed-fn           #(evaluation/frac-seeds % sentences property seed-frac)
                           #_:context-match-fn #_#(< context-thresh (re/context-vector-cosine-sim %1 %2))
                           :context-match-fn  (fn [s p]
                                                (and (re/sent-pattern-concepts-match? s p)
                                                     (< context-thresh (re/context-vector-cosine-sim s p))))
                           :cluster-merge-fn  re/add-to-pattern
                           :cluster-match-fn  #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                 (and (< (or %3 cluster-thresh) score)
                                                      score))
                           :pattern-filter-fn #(filter (fn [p] (<= min-support (count (:support p)))) %)
                           :pattern-update-fn #(filter (fn [p] (<= min-support (count (:support p)))) %)}
                   [model matches patterns] (re/init-bootstrap-persistent-patterns re/cluster-bootstrap-extract-relations-persistent-patterns model params)]
               (log/info "Metrics:" (math/calc-metrics {:predicted-true (evaluation/predicted-true matches)
                                                        :actual-true    (evaluation/actual-true model property)
                                                        :all            (evaluation/all-triples model)}))
               matches))

#_(evaluation/format-matches model matches)

#_(evaluation/parameter-walk property model)
(def parameter-walk (evaluation/parameter-walk property model))
(spit (io/file training-dir "results" "results.edn") parameter-walk)