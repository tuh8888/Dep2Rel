(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [incanter.core :as incanter]
            [incanter.charts :as charts]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [clojure.string :as str]
            [ubergraph.core :as uber]))

;; File naming patterns
(def sep "_")
(def training-prefix (apply str (interpose sep ["chemprot" "training"])))
(def training-pattern (apply str (interpose sep [training-prefix "%s"])))
(def testing-prefix (apply str (interpose sep ["chemprot" "test" "gs"])))
(def testing-pattern (apply str (interpose sep ["chemprot" "test" "%s" "gs"])))

;;; FILES ;;;
(def home-dir (io/file "/" "home" "harrison"))
#_(def home-dir (io/file "/" "media" "harrison" "Seagate Expansion Drive" "data"))
(def biocreative-dir (io/file home-dir "BioCreative" "BCVI-2017" "ChemProt_Corpus"))
(def training-dir (io/file biocreative-dir training-prefix))
(def testing-dir (io/file biocreative-dir testing-prefix))

(def word-vector-dir (io/file home-dir "WordVectors"))
(def word2vec-db (io/file word-vector-dir "bio-word-vectors-clj.vec"))

;;; MODELS ;;;
(defn make-model
  [v]
  (log/info "Making model")
  (let [model (as-> (k/simple-model v) model
                    (update model :structure-annotations #(util/pmap-kv sentence/assign-word-embedding %))
                    (update model :structure-annotations #(util/pmap-kv (partial sentence/assign-sent-id model) %))
                    (update model :concept-annotations #(util/pmap-kv (partial sentence/assign-tok model) %))
                    (assoc model :sentences (->> model
                                                 (sentence/concept-annotations->sentences)
                                                 (map #(evaluation/assign-property model %)))))]

    (log/info "Num sentences:" (count (:sentences model)))
    model))

(def testing-knowtator-view (k/view testing-dir))
(def training-knowtator-view (k/view training-dir))

(rdr/read-biocreative-files training-dir training-pattern training-knowtator-view)
(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator-view)


(def training-model (word2vec/with-word2vec word2vec-db
                      (make-model training-knowtator-view)))
(def testing-model (word2vec/with-word2vec word2vec-db
                     (make-model testing-knowtator-view)))
(log/info "Num sentences:" (count (keep :property (:sentences testing-model))))
(log/info "Num sentences:" (count (keep :property (:sentences training-model))))

#_(get-in training-model [:structure-annotations (sentence/ann-tok training-model (get-in training-model [:concept-annotations "23402364-T37"]))])
#_(get-in training-model [:structure-annotations "23402364-859768"])
#_(map #(:text (first (vals (get-in training-model [:structure-annotations % :spans])))) (keys (get-in training-model [:structure-graphs "23402364-Sentence 1" :node-map])))
;; #{"12871155-T7" "12871155-T20"} has a ridiculously long context due to the number of tokens in 4-amino-6,7,8,9-tetrahydro-2,3-diphenyl-5H-cyclohepta[e]thieno[2,3-b]pyridine
;;(filter #(= 35 (count (:context %))) (make-all-seeds model property (:sentences model) 100))

;;; CLUSTERING ;;;
(def property "INHIBITOR")

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


;;; RELATION EXTRACTION ;;;
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
  [model {:keys [iteration seeds new-matches matches patterns sentences]}]
  (let [success-model (assoc model :matches matches
                                   :patterns patterns
                                   :predicted-true (evaluation/predicted-true matches))]
    (cond (= 100 iteration) success-model
          (empty? new-matches) success-model
          (empty? sentences) success-model
          (< 3000 (count matches)) model
          (empty? seeds) model
          (empty? patterns) success-model)))

(comment
  (def split-training-model (let [seed-frac 0.2]
                              (evaluation/split-train-test training-model property seed-frac)))
  (def results (let [context-path-length-cap 100
                     params {:context-thresh    0.95
                             :cluster-thresh    0.95
                             :min-match-support 3
                             :min-seed-support  3
                             :min-match-matches 0}
                     context-match-fn (partial concept-context-match params)
                     pattern-update-fn (partial pattern-update context-match-fn params)]
                 (-> split-training-model
                     (update :sentences evaluation/context-path-filter context-path-length-cap)
                     (re/bootstrap {:terminate?        terminate?
                                    :context-match-fn  context-match-fn
                                    :pattern-update-fn pattern-update-fn})
                     (evaluation/calc-metrics))))

  (apply evaluation/format-matches training-model results))

(comment
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
  (def results-dataset (->> results
                            (map #(apply dissoc % (keys training-model)))
                            (incanter/to-dataset)))
  (incanter/view results-dataset)
  (spit (io/file training-dir "results" "results.edn") results))