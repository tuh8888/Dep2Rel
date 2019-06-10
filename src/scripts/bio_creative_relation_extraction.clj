(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [incanter.core :as incanter]
            [incanter.charts :as inc-charts]
            [incanter.svg :as inc-svg]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [uncomplicate-context-alg :as context]
            [uncomplicate.neanderthal.native :as thal-native]))

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
                    (update model :structure-annotations #(util/pmap-kv (fn [s]
                                                                          (->> s
                                                                               #_sentence/assign-word-embedding
                                                                               (sentence/assign-sent-id model)))
                                                                        %))
                    (update model :concept-annotations #(util/pmap-kv (partial sentence/assign-tok model) %)))]
    (log/info "Model" (util/map-kv count model))
    model))

(defn make-sentences
  [model]
  (let [sentences (->> model
                       (sentence/concept-annotations->sentences)
                       (pmap #(evaluation/assign-property model %)))]
    (log/info "Num sentences:" (count sentences))
    (log/info "Num sentences with property:" (util/map-kv count (group-by :property sentences)))
    sentences))

(def training-knowtator-view (k/view training-dir))
#_(def testing-knowtator-view (k/view testing-dir))

(rdr/read-biocreative-files training-dir training-pattern training-knowtator-view)
#_(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator-view)


(def training-model (word2vec/with-word2vec word2vec-db
                      (make-model training-knowtator-view)))

#_(def testing-model (word2vec/with-word2vec word2vec-db
                       (make-model testing-knowtator-view)))

(def training-sentences (make-sentences training-model))

(log/info "Num sentences with property:" (count (keep :property training-sentences))
          (util/map-kv count (group-by :property training-sentences)))

#_(get-in training-model
          [:structure-annotations (sentence/ann-tok training-model
                                                    (get-in training-model [:concept-annotations "23402364-T37"]))])
#_(get-in training-model [:structure-annotations "23402364-859768"])
#_(map #(:text (first (vals (get-in training-model [:structure-annotations % :spans]))))
       (keys (get-in training-model [:structure-graphs "23402364-Sentence 1" :node-map])))
;; #{"12871155-T7" "12871155-T20"} has a ridiculously long context due to the number of tokens in
;; 4-amino-6,7,8,9-tetrahydro-2,3-diphenyl-5H-cyclohepta[e]thieno[2,3-b]pyridine
;;(filter #(= 35 (count (:context %))) (make-all-seeds model property (:sentences model) 100))

;;; CLUSTERING ;;;
(def properties #{"INHIBITOR"} #_#{"PART-OF"
                                   "REGULATOR" "DIRECT-REGULATOR" "INDIRECT-REGULATOR"
                                   "UPREGULATOR" "ACTIVATOR" "INDIRECT-UPREGULATOR"
                                   "DOWNREGULATOR" "INHIBITOR" "INDIRECT-DOWNREGULATOR"
                                   "AGONIST" "AGONIST-ACTIVATOR" "AGONIST-INHIBITOR"
                                   "ANTAGONIST"
                                   "MODULATOR" "MODULATOR‐ACTIVATOR" "MODULATOR‐INHIBITOR"
                                   "COFACTOR"
                                   "SUBSTRATE" "PRODUCT-OF" "SUBSTRATE_PRODUCT-OF"
                                   "NOT"})

;;; PCA ;;;
(comment
  (def triples-dataset (->> training-sentences
                            (filter #(or (nil? (:property %))
                                         (properties (:property %))))
                            (evaluation/sentences->dataset)))

  (def groups (incanter/sel triples-dataset :cols :property))
  (def y (incanter/sel triples-dataset :cols (range 0 200)))
  (def x (evaluation/pca-2 y))
  (-> (inc-charts/scatter-plot (get x 0) (get x 1)
                               :group-by groups
                               :legend true
                               :x-label "PC1"
                               :y-label "PC2"
                               :title "PCA")
      (incanter/view)
      #_(inc-svg/save-svg "pca-all.svg ")))

(comment
  (def clusters (-> training-sentences
                    (default-cluster #{} 0.75)))
  (def clust-sent-dataset (evaluation/sentences->dataset clusters))

  (def x3 (evaluation/pca-2 clust-sent-dataset))
  (incanter/view (inc-charts/scatter-plot (get x3 0) (get x3 1)
                                          :legend true
                                          :x-label "PC1"
                                          :y-label "PC2"
                                          :title "PCA")))



;;; RELATION EXTRACTION ;;;
(defn concept-context-match
  [{:keys [context-thresh] :as params} samples patterns]
  (when (and (seq samples) (seq patterns))
    (->> patterns (math/find-best-row-matches params samples)
         (filter (fn [{:keys [score]}] (< context-thresh score)))
         (filter (fn [{:keys [sample match]}] (re/sent-pattern-concepts-match? sample match)))
         (map (fn [{:keys [sample match]}]
                (assoc sample :predicted (:predicted match)))))))

(defn pattern-update
  [{:keys [min-match-support reclustering?] :as params}
   new-matches patterns property]
  (let [samples (->> new-matches
                     (filter #(= (:predicted %) property))
                     (set))
        patterns (->> patterns
                      (filter #(= (:predicted %) property))
                      (set))
        patterns (->> (cluster-tools/single-pass-cluster samples patterns
                        (merge params {:cluster-merge-fn re/add-to-pattern
                                       :cluster-sim-fn   context/context-vector-cosine-sim}))
                      (map #(assoc % :predicted property)))
        filt #(or (empty? new-matches) (< min-match-support (count (:support %))))]
    [(filter filt patterns)
     (when reclustering?
       (->> patterns
            (remove filt)
            (mapcat :support)))]))

(defn terminate?
  [{:keys [max-iterations max-matches]} model
   {:keys [iteration seeds new-matches matches patterns samples last-new-matches]}]
  (let [success-model (assoc model :matches matches
                                   :patterns patterns)]
    (cond (<= max-iterations iteration) (do (log/info "Max iteration reached")
                                            success-model)
          (= last-new-matches new-matches) (do (log/info "No new matches")
                                               success-model)
          (empty? samples) (do (log/info "No more samples")
                               success-model)
          (<= max-matches (count matches)) (do (log/info "Too many matches")
                                               model)
          (empty? seeds) (do (log/info "No seeds")
                             model))))

(def training-sentences (map #(sentence/map->Sentence %) training-sentences))

(def split-training-model (let [seed-frac 0.2]
                            (evaluation/split-train-test training-sentences training-model seed-frac properties)))
(log/set-level! :debug)

(def results (let [context-path-length-cap 100
                   params {:context-thresh    0.95
                           :cluster-thresh    0.95
                           :min-match-support 0
                           :max-iterations    10
                           :max-matches       3000
                           :reclustering?     true
                           :vector-fn         #(or (:VEC %) (context/context-vector % training-model))
                           :factory           thal-native/native-double}
                   context-match-fn (partial concept-context-match params)
                   pattern-update-fn (partial pattern-update params)
                   terminate? (partial terminate? params)]
               (word2vec/with-word2vec word2vec-db
                 (-> split-training-model
                     (assoc :properties properties)
                     (update :samples (fn [samples] (evaluation/context-path-filter context-path-length-cap samples)))
                     (update :samples (fn [samples] (map #(assoc % :VEC (context/context-vector % training-model)) samples)))
                     (re/bootstrap {:terminate?        terminate?
                                    :context-match-fn  context-match-fn
                                    :pattern-update-fn pattern-update-fn})
                     (evaluation/calc-metrics)
                     (doall)))))


#_(apply evaluation/format-matches training-model results)

(comment
  (log/set-level! :info)
  #_(def results (evaluation/parameter-walk properties training-sentences training-model
                                            {:context-path-length-cap [2 10 100] #_[2 3 5 10 20 35 100]
                                             :context-thresh #_[0.95] [0.975 0.95 0.925 0.9 0.85]
                                             :cluster-thresh #_[0.95] [0.95 0.9 0.75 0.5]
                                             :min-seed-support #_[3]  [0 5 25]
                                             :min-match-support #_[0] [0 5 25]
                                             :min-match-matches #_[0] [0 5 25]
                                             :seed-frac #_[0.2]       [0.05 0.25 0.5 0.75]
                                             :terminate?              terminate?
                                             :context-match-fn        concept-context-match
                                             :pattern-update-fn       pattern-update}))
  (def results-dataset (->> results
                            (map #(apply dissoc % (keys training-model)))
                            (incanter/to-dataset)))
  (incanter/view results-dataset)
  (spit (io/file training-dir "results" "results.edn") results))