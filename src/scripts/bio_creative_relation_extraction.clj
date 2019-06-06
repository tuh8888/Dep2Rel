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
(let [x (range -3 3 0.1)]
    (incanter/view
      (charts/dynamic-scatter-plot
        [cluster-similarity-score-threshold (range 0 1 0.01)]
        [x (cluster-tools/single-pass-cluster (:sentences model) #{}
             {:cluster-merge-fn re/add-to-pattern
              :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                   (and (< (or %3 cluster-similarity-score-threshold) score)
                                        score))})])))

;;; PCA ;;;
(defn show-pca
  [coll]
  (let [X (->> coll
               (map :context-vector)
               (filter identity)
               (map seq)
               (map #(take 2 %))
               (map vec)
               (vec)
               (incanter/to-dataset)
               (incanter/to-matrix))
        pca (stats/principal-components X)

        components (:rotation pca)
        pc1 (incanter/sel components :cols 0)
        pc2 (incanter/sel components :cols 1)
        x1 (incanter/mmult X pc1)
        x2 (incanter/mmult X pc2)]
    (incanter/view (charts/scatter-plot x1 x2
                                        :x-label "PC1"
                                        :y-label "PC2"
                                        :title "PCA"))))

(comment (show-pca (:sentences model))
         (-> (:sentences model)
             (cluster-tools/single-pass-cluster #{}
               {:cluster-merge-fn re/add-to-pattern
                :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                     (and (< (or %3 0.75) score)
                                          score))})
             (show-pca)))

;;; RELATION EXTRACTION
(def split-model (let [seed-frac 0.2]
                   (evaluation/frac-seeds model property seed-frac)))

(def results (let [context-path-length-cap 100
                   context-thresh 0.94
                   cluster-thresh 0.95
                   min-match-support 10
                   min-seed-support 3
                   min-seed-matches 0
                   min-match-matches 0
                   sentences (evaluation/context-path-filter context-path-length-cap (get-in split-model [0 :sentences]))
                   context-match-fn (fn [s p]
                                      (and (re/sent-pattern-concepts-match? s p)
                                           (< context-thresh (re/context-vector-cosine-sim s p))))
                   params {:context-match-fn  context-match-fn
                           :make-pattern-fn   (fn [samples clusters]
                                                (cluster-tools/single-pass-cluster samples clusters
                                                  {:cluster-merge-fn re/add-to-pattern
                                                   :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                                        (and (< (or %3 cluster-thresh) score)
                                                                             score))}))
                           :pattern-update-fn (fn [patterns seeds matches]

                                                (if (empty? matches)
                                                  (remove (fn [{:keys [support]}]
                                                            (> min-seed-support (count support)))
                                                          patterns)
                                                  (->> patterns
                                                       (remove (fn [{:keys [support]}]
                                                                 (> (+ min-match-support) (count support))))
                                                       #_(remove (fn [p]
                                                                   (> min-seed-matches (->> seeds
                                                                                            (filter #(context-match-fn % p))
                                                                                            (count)))))
                                                       (remove (fn [p]
                                                                 (> min-match-matches (->> matches
                                                                                           (filter #(context-match-fn % p))
                                                                                           (count))))))))}
                   [matches patterns] (re/bootstrap-persistent-patterns (get-in split-model [1]) sentences params)]
               (log/info "Metrics:" (-> (get-in split-model [0])
                                        (assoc :predicted-true (evaluation/predicted-true matches))
                                        (math/calc-metrics)))
               [matches patterns]))

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