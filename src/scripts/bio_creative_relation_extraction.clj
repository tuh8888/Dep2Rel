(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [incanter.core :as incanter]
            [incanter.charts :as charts]
            [edu.ucdenver.ccp.nlp.readers :as rdr]))

(def home-dir (io/file "/" "home" "harrison"))
#_(def home-dir (io/file "/" "media" "harrison" "Seagate Expansion Drive" "data"))
(def biocreative-dir (io/file home-dir "BioCreative" "BCVI-2017" "ChemProt_Corpus"))
(def sep "_")
(def chemprot "chemprot")
(def training-string "training")
(def training-prefix (apply str (interpose sep [chemprot training-string])))
(def training-pattern (apply str (interpose sep [training-prefix "%s"])))
(def testing-string "test")
(def gs-string "gs")
(def testing-prefix (apply str (interpose sep [chemprot testing-string gs-string])))
(def testing-pattern (apply str (interpose sep [chemprot testing-string "%s" gs-string])))
(def training-dir (io/file biocreative-dir training-prefix))
(def testing-dir (io/file biocreative-dir testing-prefix))
(def word-vector-dir (io/file home-dir "WordVectors"))
(def word2vec-db (.getAbsolutePath (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def training-knowtator-view (k/view training-dir))
(def testing-knowtator-view (k/view testing-dir))

#_(rdr/biocreative-read-abstracts (k/model training-knowtator-view) (io/file training-dir "chemprot_training_abstracts.tsv"))
#_(rdr/biocreative-read-entities (k/model training-knowtator-view) (io/file training-dir "chemprot_training_entities.tsv"))
#_(rdr/biocreative-read-relations (k/model training-knowtator-view) (io/file training-dir "chemprot_training_relations.tsv"))
(->> "abstracts"
     (format (str testing-pattern ".tsv"))
     (io/file testing-dir)
     (rdr/biocreative-read-abstracts (k/model testing-knowtator-view)))
#_(->> "entities"
       (format (str testing-pattern ".tsv"))
       (io/file testing-dir)
       (rdr/biocreative-read-entities (k/model testing-knowtator-view)))
#_(->> "relations"
       (format (str testing-pattern ".tsv"))
       (io/file testing-dir)
       (rdr/biocreative-read-relations (k/model testing-knowtator-view)))

;; Read from conll files
;#_(rdr/biocreative-read-dependency annotations training-dir word2vec-db)
(.save (k/model training-knowtator-view))


(let [sentences-dir (->> "sentences"
                         (format testing-pattern)
                         (io/file testing-dir))]
  (doseq [[id content] (rdr/sentenize (k/model testing-knowtator-view))]
    (let [content (->> content
                       (map :text)
                       (interpose "\n")
                       (apply str))
          sentence-f (io/file sentences-dir (str id ".txt"))]
      (spit sentence-f content))))



(def training-model (let [model (k/simple-model training-knowtator-view)
                          structures-annotations-with-embeddings (word2vec/with-word2vec word2vec-db
                                                                   (sentence/structures-annotations-with-embeddings model))

                          concept-annotations-with-toks (sentence/concept-annotations-with-toks model)

                          model (assoc model
                                  :concept-annotations concept-annotations-with-toks
                                  :structure-annotations structures-annotations-with-embeddings)
                          sentences (->> model
                                         (sentence/concept-annotations->sentences)
                                         (map #(assoc % :property (evaluation/sent-property model %))))]
                      (assoc model :sentences sentences)))

#_(get-in training-model [:structure-annotations (sentence/annotation-tok-id training-model (get-in training-model [:concept-annotations "23402364-T37"]))])
#_(get-in training-model [:structure-annotations "23402364-859768"])
#_(map #(:text (first (vals (get-in training-model [:structure-annotations % :spans])))) (keys (get-in training-model [:structure-graphs "23402364-Sentence 1" :node-map])))
(log/info "Num sentences:" (count (:sentences training-model)))

(def property "INHIBITOR")

;; #{"12871155-T7" "12871155-T20"} has a ridiculously long context due to the number of tokens in 4-amino-6,7,8,9-tetrahydro-2,3-diphenyl-5H-cyclohepta[e]thieno[2,3-b]pyridine
;;(filter #(= 35 (count (:context %))) (make-all-seeds model property (:sentences model) 100))

;;; CLUSTERING ;;;

(defn default-cluster
  [samples clusters cluster-thresh]
  (cluster-tools/single-pass-cluster samples clusters
    {:cluster-merge-fn re/add-to-pattern
     :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                          (and (< (or %3 cluster-thresh) score)
                               score))}))

(comment
  (-> (evaluation/make-all-seeds training-model property (:sentences training-model))
      (default-cluster #{} 0.75)
      (count)))

;;; PCA ;;;
(comment
  (def triples-dataset (evaluation/triples->dataset training-model))
  (def groups (incanter/sel triples-dataset :cols :property))

  (def x (evaluation/pca-2 triples-dataset))
  (incanter/view (charts/scatter-plot (get x 0) (get x 1)
                                      :group-by groups
                                      :legend true
                                      :x-label "PC1"
                                      :y-label "PC2"
                                      :title "PCA")))
(comment
  (def sent-dataset (evaluation/sentences->dataset (:sentences training-model)))
  (def x2 (evaluation/pca-2 sent-dataset))
  (incanter/view (charts/scatter-plot (get x2 0) (get x2 1)
                                      :legend true
                                      :x-label "PC1"
                                      :y-label "PC2"
                                      :title "PCA")))

(comment
  (def clusters (-> training-model
                    :sentences
                    (default-cluster #{} 0.75)))
  (def clust-sent-dataset (evaluation/sentences->dataset clusters))

  (def x3 (evaluation/pca-2 clust-sent-dataset))
  (incanter/view (charts/scatter-plot (get x3 0) (get x3 1)
                                      :legend true
                                      :x-label "PC1"
                                      :y-label "PC2"
                                      :title "PCA")))


;;; RELATION EXTRACTION
(defn concept-context-match
  [{:keys [context-thresh]} s p]
  (and (re/sent-pattern-concepts-match? s p)
       (< context-thresh (re/context-vector-cosine-sim s p))))
(defn pattern-update
  [context-match-fn {:keys [cluster-thresh min-seed-support min-match-support min-match-matches]} _ new-matches matches patterns]
  (-> new-matches
      (default-cluster patterns cluster-thresh)
      (cond->>
        (and (< 0 min-seed-support)
             (empty? new-matches)) (remove #(> min-seed-support (count (:support %))))
        (and (< 0 min-match-support)
             (seq new-matches)) (remove #(> (+ min-match-support) (count (:support %))))
        (and (< 0 min-match-matches)
             (seq new-matches)) (remove #(> min-match-matches
                                            (count (filter (fn [s] (context-match-fn s %)) matches)))))
      (set)))
(defn terminate?
  [iteration seeds new-matches matches patterns sentences]
  (cond (= 100 iteration) [matches patterns]
        (empty? new-matches) [matches patterns]
        (empty? sentences) [matches patterns]
        (< 3000 (count matches)) [#{} #{}]
        (empty? seeds) [#{} #{}]
        (empty? patterns) [matches patterns]))

(comment
  (def training-split-model (let [seed-frac 0.2]
                              (evaluation/frac-seeds training-model property seed-frac)))

  (def results (let [context-path-length-cap 100
                     params {:context-thresh    0.95
                             :cluster-thresh    0.95
                             :min-match-support 3
                             :min-seed-support  3
                             :min-match-matches 0}
                     context-match-fn (partial concept-context-match params)
                     pattern-update-fn (partial pattern-update context-match-fn params)]
                 (let [sentences (evaluation/context-path-filter context-path-length-cap (get-in training-split-model [0 :sentences]))
                       [matches patterns] (re/bootstrap (get-in training-split-model [1]) sentences {:terminate?        terminate?
                                                                                                     :context-match-fn  context-match-fn
                                                                                                     :pattern-update-fn pattern-update-fn})]
                   (log/info "Metrics:" (-> (get-in training-split-model [0])
                                            (assoc :predicted-true (evaluation/predicted-true matches))
                                            (math/calc-metrics)))
                   [matches patterns])))

  (apply evaluation/format-matches training-model results))

(log/set-level! :info)
(def results (evaluation/parameter-walk property training-model
                                        :context-path-length-cap [2 10 100] #_[2 3 5 10 20 35 100]
                                        :context-thresh #_[0.95] [0.975 0.95 0.925 0.9 0.85]
                                        :cluster-thresh #_[0.95] [0.95 0.9 0.75 0.5]
                                        :min-seed-support #_[3] [0 5 25]
                                        :min-match-support #_[0] [0 5 25]
                                        :min-match-matches #_[0] [0 5 25]
                                        :seed-frac #_[0.2] [0.05 0.25 0.5 0.75]
                                        :terminate? terminate?
                                        :context-match-fn concept-context-match
                                        :pattern-update-fn pattern-update))
(def results-dataset (incanter/to-dataset results))
(incanter/view results-dataset)
(spit (io/file training-dir "results" "results.edn") results)