(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [incanter.stats :as stats]
            [incanter.core :as incanter]
            [incanter.charts :as charts]))

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

(def annotations (k/view training-dir))

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
                         :structure-annotations structures-annotations-with-embeddings)]
             (assoc model :sentences (sentence/concept-annotations->sentences model))))

#_(get-in model [:structure-annotations (sentence/annotation-tok-id model (get-in model [:concept-annotations "23402364-T37"]))])
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

;;; PCA ;;;

(def triples-dataset (evaluation/triples->dataset model))
(def groups (incanter/sel triples-dataset :cols :property))

(def x (evaluation/pca-2 triples-dataset))
(count (distinct groups))
(incanter/view (charts/scatter-plot (get x 0) (get x 1)
                                    :group-by groups
                                    :legend true
                                    :x-label "PC1"
                                    :y-label "PC2"
                                    :title "PCA"))

(def sent-dataset (evaluation/sentences->dataset (:sentences model)))
(def x2 (evaluation/pca-2 sent-dataset))
(incanter/view (charts/scatter-plot (get x2 0) (get x2 1)
                                    :legend true
                                    :x-label "PC1"
                                    :y-label "PC2"
                                    :title "PCA"))

(comment
  (pca-2 (:sentences model))
  (-> (:sentences model)
      (cluster-tools/single-pass-cluster #{}
        {:cluster-merge-fn re/add-to-pattern
         :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                              (and (< (or %3 0.75) score)
                                   score))})
      (pca-2)))

;;; RELATION EXTRACTION
(def split-model (let [seed-frac 0.2]
                   (evaluation/frac-seeds model property seed-frac)))

(def results (let [context-path-length-cap 100
                   context-thresh 0.95
                   cluster-thresh 0.95
                   min-match-support 3
                   min-seed-support 3
                   min-match-matches 0]
               (letfn [(context-match-fn [s p]
                         (and (re/sent-pattern-concepts-match? s p)
                              (< context-thresh (re/context-vector-cosine-sim s p))))
                       (pattern-update-fn [seeds new-matches matches patterns]
                         (-> new-matches
                             (cluster-tools/single-pass-cluster patterns
                               {:cluster-merge-fn re/add-to-pattern
                                :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                     (and (< (or %3 cluster-thresh) score)
                                                          score))})
                             (cond->>
                               (and (< 0 min-seed-support)
                                    (empty? new-matches)) (remove #(> min-seed-support (count (:support %))))
                               (and (< 0 min-match-support)
                                    (seq new-matches)) (remove #(> (+ min-match-support) (count (:support %))))
                               (and (< 0 min-match-matches)
                                    (seq new-matches)) (remove #(> min-match-matches
                                                                   (count (filter (fn [s] (context-match-fn s %)) matches)))))
                             (set)))]
                 (let [params {:context-match-fn  context-match-fn
                               :pattern-update-fn pattern-update-fn}
                       sentences (evaluation/context-path-filter context-path-length-cap (get-in split-model [0 :sentences]))
                       [matches patterns] (re/bootstrap-persistent-patterns (get-in split-model [1]) sentences params)]
                   (log/info "Metrics:" (-> (get-in split-model [0])
                                            (assoc :predicted-true (evaluation/predicted-true matches))
                                            (math/calc-metrics)))
                   [matches patterns]))))

(apply evaluation/format-matches model results)

(comment
  #_(evaluation/parameter-walk property model)
  (def parameter-walk (evaluation/parameter-walk property model
                                                 :context-path-length-cap [100] #_[2 3 4 5 10 20 35]
                                                 :context-thresh [0.95] #_[0.975 0.95 0.925 0.9 0.85]
                                                 :cluster-thresh [0.95] #_[0.95 0.9 0.8 0.7 0.6 0.5]
                                                 :min-support [0] #_[0 3 5 10 20 30]
                                                 :seed-frac [0.05 0.25 0.45 0.65 0.75]))
  (spit (io/file training-dir "results" "results.edn") parameter-walk))