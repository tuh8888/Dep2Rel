(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [uncomplicate.neanderthal.native :as thal-native]
            [incanter.core :as incanter]
            [incanter.io :as inc-io]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]))

(log/set-level! :info)

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

(def factory thal-native/native-double)
(def properties #{"PART-OF"
                  "REGULATOR" "DIRECT-REGULATOR" "INDIRECT-REGULATOR"
                  "UPREGULATOR" "ACTIVATOR" "INDIRECT-UPREGULATOR"
                  "DOWNREGULATOR" "INHIBITOR" "INDIRECT-DOWNREGULATOR"
                  "AGONIST" "AGONIST-ACTIVATOR" "AGONIST-INHIBITOR"
                  "ANTAGONIST"
                  "MODULATOR" "MODULATOR‐ACTIVATOR" "MODULATOR‐INHIBITOR"
                  "COFACTOR"
                  "SUBSTRATE" "PRODUCT-OF" "SUBSTRATE_PRODUCT-OF"
                  "NOT"
                  re-model/NONE}
  #_#{"INHIBITOR" #_re-model/NONE})

;;; MODELS ;;;

(def training-knowtator-view (k/model training-dir nil))
(rdr/read-biocreative-files training-dir training-pattern training-knowtator-view)
(def base-training-model (re-model/make-model training-knowtator-view factory word2vec-db))
(def training-model-with-sentences (assoc base-training-model :sentences (re-model/make-sentences base-training-model)))
(def training-model (assoc training-model-with-sentences :properties properties))

(def testing-knowtator-view (k/model testing-dir nil))
(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator-view)
(def base-testing-model (re-model/make-model testing-knowtator-view factory word2vec-db))
(def testing-model-with-sentences (assoc base-testing-model :sentences (re-model/make-sentences base-testing-model)))
(def testing-model (assoc testing-model-with-sentences :properties properties))

;; This allows me to reset sentences if they get reloaded
#_(def training-model (update training-model
                              :sentences (fn [sentences]
                                           (map #(re-model/map->Sentence %) sentences))))

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

(def training-context-paths-plot (evaluation/plot-context-lengths training-model results-dir "Training %s"))
#_(incanter/view training-context-paths-plot)
(def testing-context-paths-plot(evaluation/plot-context-lengths testing-model results-dir "Test %s"))
#_(incanter/view testing-context-paths-plot)

;;; CLUSTERING ;;;

;;; PCA ;;;

#_(def model-with-sentences-dataset (evaluation/sentences->dataset training-model))
#_(incanter/save (:sentences-dataset model-with-sentences-dataset) (str (io/file training-dir " sentences-dataset.csv ")))
#_(def pca-plots (evaluation/pca-plots model-with-sentences-dataset
                                       {:save {:file (io/file results-dir " %s ")}}))


;;; RELATION EXTRACTION ;;;

(def prepared-model (-> training-model
                        (assoc :seed-frac 0.2
                               :rng 0.022894)
                        (re-model/split-train-test)))

#_(def prepared-model (re-model/test-train testing-model prepared-model))


(def results (-> prepared-model
                 (assoc :context-path-length-cap 100
                        :context-thresh 0.95
                        :cluster-thresh 0.95
                        :min-match-support 0
                        :max-iterations 100
                        :max-matches 3000
                        :re-clustering? true)
                 (evaluation/run-model results-dir)))
(incanter/view (:plot results))

(count (:samples results))

#_(apply evaluation/format-matches base-training-model results)

(def param-walk-results (evaluation/parameter-walk results-dir
                                                   base-training-model
                                                   {:context-path-length-cap [100] #_[2 3 5 10 20 35 100]
                                                    :context-thresh          [0.95] #_[0.975 0.95 0.925 0.9 0.85]
                                                    :cluster-thresh          [0.95] #_[0.95 0.9 0.75 0.5]
                                                    :min-match-support       [0] #_[0 5 25]
                                                    :seed-frac #_[0.2]       [0.05 0.25 0.5 0.75]
                                                    :rng                     0.022894}))