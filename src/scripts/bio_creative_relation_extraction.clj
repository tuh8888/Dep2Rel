(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [taoensso.timbre :as log]
            [taoensso.timbre.appenders.core :as appenders]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [uncomplicate.neanderthal.native :as thal-native]
            [incanter.core :as incanter]
            [incanter.io :as inc-io]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]))

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
(def results-dir (io/file training-dir "results"))

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

(def training-knowtator-view (k/model training-dir nil))
(rdr/read-biocreative-files training-dir training-pattern training-knowtator-view)
(def base-training-model (re-model/make-model training-knowtator-view factory word2vec-db))
(def training-sentences (re-model/make-sentences base-training-model))
(spit (io/file training-dir "sentences.edn") (pr-str training-sentences))
(def training-sentences-filtered (->> training-sentences-filtered
                                      (filter (fn [s]
                                                (->> s
                                                     :entities
                                                     (map #(get-in base-training-model [:concept-annotations % :concept]))
                                                     (set)
                                                     (allowed-concept-pairs))))
                                      (map #(update % :property (fn [property] (or (get property-map property)
                                                                                   re-model/NONE))))))
(def training-model (assoc base-training-model :sentences training-sentences-filtered
                                               :properties properties))

(def testing-knowtator-view (k/model testing-dir nil))
(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator-view)
(def base-testing-model (re-model/make-model testing-knowtator-view factory word2vec-db))
(def testing-sentences (re-model/make-sentences base-testing-model))
(spit (io/file testing-sentences "sentences.edn") (pr-str testing-sentences))
(def testing-sentences-filtered (->> testing-sentences
                                     (filter (fn [s]
                                               (->> s
                                                    :entities
                                                    (map #(get-in base-testing-model [:concept-annotations % :concept]))
                                                    (set)
                                                    (allowed-concept-pairs))))
                                     (map #(update % :property (fn [property] (or (get property-map property)
                                                                                  re-model/NONE))))))
(def testing-model (-> base-testing-model
                       (assoc :sentences testing-sentences-filtered)
                       (assoc :properties properties)))

;;; SENTENCE STATS ;;;

(log/info "Model\n"
          (incanter/to-dataset [(assoc (->> training-model
                                            (re-model/model-params)
                                            (util/map-kv count))
                                  :model :training)
                                (assoc (->> testing-model
                                            (re-model/model-params)
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

#_(def training-context-paths-plot (evaluation/plot-context-lengths training-model results-dir "Training %s"))
#_(incanter/view training-context-paths-plot)
#_(def testing-context-paths-plot (evaluation/plot-context-lengths testing-model results-dir "Test %s"))
#_(def testing-context-paths-plot-pos (-> testing-model
                                          (update :sentences (fn [sentences])
                                                  (re-model/actual-positive sentences))
                                          (evaluation/plot-context-lengths
                                            results-dir "Pos Test %s")))
#_(def testing-context-paths-plot-neg (-> testing-model
                                          (update :sentences (fn [sentences]
                                                               (re-model/actual-negative sentences)))
                                          (evaluation/plot-context-lengths
                                            results-dir "Neg Test %s")))
#_(incanter/view testing-context-paths-plot)
#_(incanter/view testing-context-paths-plot-pos)
#_(incanter/view testing-context-paths-plot-neg)

;;; CLUSTERING ;;;

(def clusters-dataset (-> training-model
                          (assoc :sentences (-> training-model
                                                (assoc :cluster-thresh 0.95
                                                       :new-matches (:sentences training-model))
                                                (re/pattern-update)))
                          (evaluation/sentences->dataset)))
(incanter/save (:sentences-dataset clusters-dataset) (str (io/file training-dir " clusters-dataset.csv ")))
(def pca-plots (evaluation/pca-plots clusters-dataset
                                     {:save {:file (io/file results-dir "%s")}}))

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

(def prepared-model (-> training-model
                        (assoc :seed-frac 1
                               :rng 0.022894
                               :negative-cap 2000)
                        (re-model/split-train-test)
                        (re-model/train-test testing-model)))

(def results (-> prepared-model
                 (assoc :context-path-length-cap 100
                        :match-thresh 0.95
                        :cluster-thresh 0.9
                        :confidence-thresh 0
                        :min-match-support 0
                        :max-iterations 100
                        :max-matches 3000
                        :re-clustering? true)
                 (evaluation/run-model results-dir)))

#_(incanter/view (:plot results))

#_(def param-walk-results (evaluation/parameter-walk training-model testing-model results-dir
                                                     {:context-path-length-cap          [10 100] #_[2 3 5 10 20 35 100]
                                                      :match-thresh          #_[0.95]   [0.95 0.9 0.85]
                                                      :cluster-thresh          #_[0.95] [0.95 0.9 0.85]
                                                      :confidence-thresh                [0.95 0.9 0]
                                                      :min-match-support                [0 2] #_[0 5 25]
                                                      :seed-frac                        [1] #_[0.05 0.25 0.5 0.75]
                                                      :rng                              0.022894
                                                      :negative-cap                     3000}))

(def baseline-results {:precision 0.4544
                       :recall    0.5387
                       :f1        0.3729})