(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [edu.ucdenver.ccp.nlp.seeds :as seeds]
            [taoensso.timbre :as log]
            [taoensso.timbre.appenders.core :as appenders]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [uncomplicate.neanderthal.native :as thal-native]
            [incanter.core :as incanter]
            [incanter.io :as inc-io]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [clojure.string :as str]))

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
(def results-dir (io/file biocreative-dir "results"))

(def word-vector-dir (io/file home-dir "WordVectors"))
(def word2vec-db (io/file word-vector-dir "bio-word-vectors-clj.vec"))

(log/set-level! :info)

(log/merge-config!
  {:appenders {:spit   (appenders/spit-appender
                         {:fname (->> "biocreative.log"
                                      (io/file home-dir)
                                      (.getAbsolutePath))})
               :postal {:enabled? false} #_(postal-appender/postal-appender
                                             {:from "me@draines.com" :to "pielkekid@gmail.com"})}})

(def factory thal-native/native-double)

(def property-map {
                   ;"PART-OF" "CPR:1"
                   ;"REGULATOR" "CPR:2" "DIRECT-REGULATOR" "CPR:2" "INDIRECT-REGULATOR" "CPR:2"
                   "UPREGULATOR"   "CPR:3" "ACTIVATOR" "CPR:3" "INDIRECT-UPREGULATOR" "CPR:3"
                   "DOWNREGULATOR" "CPR:4" "INHIBITOR" "CPR:4" "INDIRECT-DOWNREGULATOR" "CPR:4"
                   "AGONIST"       "CPR:5" "AGONIST-ACTIVATOR" "CPR:5" "AGONIST-INHIBITOR" "CPR:5"
                   "ANTAGONIST"    "CPR:6"
                   ;"MODULATOR" "CPR:7" "MODULATOR‐ACTIVATOR" "CPR:7" "MODULATOR‐INHIBITOR" "CPR:7"
                   ;"COFACTOR" "CPR:8"
                   "SUBSTRATE"     "CPR:9" "PRODUCT-OF" "CPR:9" "SUBSTRATE_PRODUCT-OF" "CPR:9"
                   ;"NOT" "CPR:10"
                   re-model/NONE   re-model/NONE})



(def properties (set (vals property-map)))

(def allowed-concept-pairs #{#{"CHEMICAL" "GENE-N"}
                             #{"CHEMICAL" "GENE-Y"}})

;;; MODELS ;;;
(defn biocreative-model
  [model sentences property-map]
  (assoc model
    :sentences (->> sentences
                    (filter (fn [{:keys [entities]}]
                              (->> entities
                                   (map #(get-in model [:concept-annotations % :concept]))
                                   (set)
                                   (allowed-concept-pairs))))
                    (map #(update % :property (fn [property] (or (get property-map property)
                                                                 re-model/NONE)))))
    :properties (set (vals property-map))))

(def training-knowtator (k/model training-dir nil))
(rdr/read-biocreative-files training-dir training-pattern training-knowtator)
(def base-training-model (re-model/make-model training-knowtator word2vec-db factory
                                              (io/file training-dir "concept-annotations.edn")
                                              (io/file training-dir "structure-annotations.edn")))
(def training-sentences (re-model/make-sentences base-training-model (io/file training-dir "sentences.edn")))
(def training-model (biocreative-model base-training-model training-sentences property-map))

(def testing-knowtator (k/model testing-dir nil))
(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator)
(def base-testing-model (re-model/make-model testing-knowtator word2vec-db factory
                                             (io/file testing-dir "concept-annotations.edn")
                                             (io/file testing-dir "structure-annotations.edn")))
(def testing-sentences (re-model/make-sentences base-testing-model (io/file testing-dir "sentences.edn")))
(def testing-model (biocreative-model base-testing-model testing-sentences property-map))

;;; SENTENCE STATS ;;;
(log/info "Model\n"
          (incanter/to-dataset [(assoc (->> (select-keys training-model re-model/MODEL-KEYs)
                                            (util/map-kv count))
                                  :model :training)
                                (assoc (->> (select-keys testing-model re-model/MODEL-KEYs)
                                            (util/map-kv count))
                                  :model :testing)]))
(log/info "Num sentences with property\n"
          (->> [(:sentences training-model)
                (:sentences testing-model)]
               (map #(group-by :property %))
               (map #(util/map-kv count %))
               (map #(assoc %2 :model %1)
                    [:training :testing])
               (incanter/to-dataset)))


(comment
  (def training-context-paths-plot (evaluation/plot-context-lengths training-model results-dir "Training %s"))
  (incanter/view training-context-paths-plot)
  (def testing-context-paths-plot (evaluation/plot-context-lengths testing-model results-dir "Test %s"))
  (def testing-context-paths-plot-pos (-> testing-model
                                          (update :sentences (fn [sentences]
                                                               (re-model/actual-positive sentences)))
                                          (evaluation/plot-context-lengths
                                            results-dir "Pos Test %s")))
  (def testing-context-paths-plot-neg (-> testing-model
                                          (update :sentences (fn [sentences]
                                                               (re-model/actual-negative sentences)))
                                          (evaluation/plot-context-lengths
                                            results-dir "Neg Test %s")))
  (incanter/view testing-context-paths-plot)
  (incanter/view testing-context-paths-plot-pos)
  (incanter/view testing-context-paths-plot-neg))

;;; CLUSTERING ;;;
(comment
  (def training-clusters (-> training-model
                             (assoc :seed-frac 1
                                    :rng 0.022894
                                    :negative-cap 2000
                                    :confidence-thresh 0
                                    :cluster-thresh 0.95
                                    :cluster-merge-fn re-model/add-to-pattern
                                    :vector-fn #(re-model/context-vector % training-model))
                             (re-model/split-train-test)
                             (re/pattern-update)))

  (def clusters-dataset (-> training-model
                            (assoc :sentences (map #(assoc % :property (:predicted %)) training-clusters))
                            (evaluation/sentences->dataset)))

  (incanter/save (:sentences-dataset clusters-dataset) (str (io/file training-dir " clusters-dataset.csv ")))
  (def pca-plots (evaluation/pca-plots clusters-dataset
                                       {:save {:file (io/file results-dir "Clustering %s")}})))
#_(incanter/view (get pca-plots "ALL"))

;;; PCA ;;;

#_(def sentences-dataset (evaluation/sentences->dataset training-model))
#_(incanter/save (:sentences-dataset sentences-dataset) (str (io/file training-dir " sentences-dataset.csv ")))
#_(def pca-plots (evaluation/pca-plots sentences-dataset
                                       {:save {:file (io/file results-dir "%s")}}))


;;; RELATION EXTRACTION ;;;

(comment
  ;; This allows me to reset sentences if they get reloaded
  (def training-model (update training-model
                              :sentences (fn [sentences]
                                           (map #(re-model/map->Sentence %) sentences))))
  (def testing-model (update testing-model
                             :sentences (fn [sentences]
                                          (map #(re-model/map->Sentence %) sentences)))))

(def split-model (-> training-model
                     (assoc :seed-frac 1
                            :rng 0.022894
                            :negative-cap 3000)
                     (re-model/split-train-test)
                     (re-model/train-test testing-model)))

(def prepared-model (assoc split-model
                      :all-seed-patterns (seeds/all-seed-patterns split-model
                                                                  {:cluster-thresh (range 0.9 0.5 -0.05)
                                                                   :match-thresh   (range 0.9 0.5 -0.05)})))

(->> (-> prepared-model
         :all-seed-patterns
         (seeds/seed-patterns-with-selectivity properties {:min-f1        0
                                                           :min-recall    0.9
                                                           :min-precision 0})
         #_(count))
     (group-by :predicted)
     (util/map-kv count))

(count (remove #(or (nil? (:property %))
                    (= re-model/NONE (:property %)))
               (:sentences testing-model)))
#_(def results (let [results (-> prepared-model
                                 (update :seeds (fn [seeds] nil #_(->> seeds
                                                                       (remove #(= (:predicted %) re-model/NONE))
                                                                       (take 500))))
                                 #_(update :matches (fn [matches]
                                                      (map #(assoc % :predicted (:property %)) (:all-samples prepared-model))))
                                 #_(update :all-samples (fn [all-samples] nil))
                                 (assoc :patterns (lazy-cat
                                                    #_(apply concat
                                                             (-> prepared-model
                                                                 :all-seed-patterns
                                                                 (seeds/seed-patterns-with-selectivity properties
                                                                                                       {:min-f1        0
                                                                                                        :min-recall    0.9
                                                                                                        :min-precision 0})))
                                                    (apply min-key count
                                                           (-> prepared-model
                                                               :all-seed-patterns
                                                               (seeds/seed-patterns-with-selectivity properties
                                                                                                     {:min-f1        0
                                                                                                      :min-recall    0.8
                                                                                                      :min-precision 0})))
                                                    (apply max-key count
                                                           (-> prepared-model
                                                               :all-seed-patterns
                                                               (seeds/seed-patterns-with-selectivity properties
                                                                                                     {:min-f1        0
                                                                                                      :min-recall    0
                                                                                                      :min-precision 0.7}))))
                                        :context-path-length-cap 10
                                        :context-path-length-min 0
                                        :match-thresh 0.25
                                        :cluster-thresh 0.7
                                        :confidence-thresh 0
                                        :min-pattern-support 0
                                        :max-iterations 0 :max-matches 5000
                                        :re-clustering? true
                                        :match-fn re/sim-to-support-in-pattern-match
                                        #_re/concept-context-match
                                        #_re/sim-to-support-in-pattern-match)

                                 (evaluation/run-model results-dir))]
                 (let [results-file (io/file results-dir "results.tsv")]
                   (when (.exists results-file) (io/delete-file results-file))
                   (doseq [m (remove #(= (:predicted %) re-model/NONE) (:matches results))]
                     (let [[e1 e2] (vec (:entities m))
                           p        (:predicted m)
                           doc      (get-in results [:concept-annotations e1 :doc])
                           [e1 e2] (sort [(Integer/parseInt (str/replace e1 (str doc "-T") ""))
                                          (Integer/parseInt (str/replace e2 (str doc "-T") ""))])
                           e1       (str "Arg1:T" e1)
                           e2       (str "Arg2:T" e2)
                           to-write (str (apply str (interpose "\t" [doc p e1 e2])) "\n")]
                       (spit results-file to-write :append true))))
                 results))

#_(evaluation/calc-metrics results)
#_(count (:matches results))
#_(incanter/view (:plot results))

(def param-walk-results (evaluation/parameter-walk nil #_training-model
                                                   prepared-model #_testing-model
                                                   results-dir
                                                   {:context-path-length-cap          [100 10] #_[2 3 5 10 20 35 100]
                                                    :context-path-length-min          [2 8]
                                                    :match-thresh          #_[0.95]   [0.5 0.7 0.8 0.9]
                                                    :cluster-thresh          #_[0.95] [0.7 0.8 0.9]
                                                    :confidence-thresh                [0.9]
                                                    :min-pattern-support              [1] #_[0 5 25]
                                                    :seed-frac                        [1] #_[0.05 0.25 0.5 0.75]
                                                    :rng                              0.022894
                                                    :negative-cap                     3000
                                                    :min-seed-pattern-precision       [0 0.25 0.5 0.7 0.9]
                                                    :min-seed-pattern-recall          [0 0.25 0.5 0.7 0.9]
                                                    :match-fn                         [re/support-weighted-sim-distribution-context-match
                                                                                       re/concept-context-match
                                                                                       re/sim-to-support-in-pattern-match]}))

#_(def baseline-results {:precision 0.4544
                         :recall    0.5387
                         :f1        0.3729})

#_(def results (with-open [rdr (clojure.java.io/reader (io/file results-dir "results.edn"))]
                 (->> (line-seq rdr)
                      (map read-string)
                      (map #(merge % (:overall-metrics %)))
                      (vec))))

#_(def dataset (incanter/to-dataset results))
#_(incanter/view dataset)
#_(count results)
