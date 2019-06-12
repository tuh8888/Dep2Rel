(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [uncomplicate.neanderthal.native :as thal-native]))

(log/set-level! :debug)

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
(word2vec/with-word2vec word2vec-db
  (word2vec/word-embedding "low"))
(def factory thal-native/native-double)

;;; MODELS ;;;
(def training-knowtator-view (k/model training-dir nil))
(rdr/read-biocreative-files training-dir training-pattern training-knowtator-view)
(def training-model (word2vec/with-word2vec word2vec-db
                      (re-model/make-model training-knowtator-view factory)))
(def training-sentences (re-model/make-sentences training-model))

#_(def testing-knowtator-view (k/view testing-dir))
#_(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator-view)
#_(def testing-model (word2vec/with-word2vec word2vec-db
                       (make-model testing-knowtator-view)))

;;; CLUSTERING ;;;
(def properties #_#{"INHIBITOR" #_re-model/NONE} #{"PART-OF"
                                                   "REGULATOR" "DIRECT-REGULATOR" "INDIRECT-REGULATOR"
                                                   "UPREGULATOR" "ACTIVATOR" "INDIRECT-UPREGULATOR"
                                                   "DOWNREGULATOR" "INHIBITOR" "INDIRECT-DOWNREGULATOR"
                                                   "AGONIST" "AGONIST-ACTIVATOR" "AGONIST-INHIBITOR"
                                                   "ANTAGONIST"
                                                   "MODULATOR" "MODULATOR‐ACTIVATOR" "MODULATOR‐INHIBITOR"
                                                   "COFACTOR"
                                                   "SUBSTRATE" "PRODUCT-OF" "SUBSTRATE_PRODUCT-OF"
                                                   "NOT"
                                                   re-model/NONE})

;;; PCA ;;;

(comment
  (def sentences-dataset (word2vec/with-word2vec word2vec-db
                           (->> training-sentences
                                (filter #(contains? properties (:property %)))
                                (evaluation/sentences->dataset training-model))))

  (def pca-pot (evaluation/pca-plot properties sentences-dataset (count (re-model/context-vector (first training-sentences) training-model))
                                    {:save {:file (io/file results-dir "pca-all.svg")}
                                     :view true})))

;;; RELATION EXTRACTION ;;;

;; This allows me to reset sentences if they get reloaded
#_(def training-sentences (map #(re-model/map->Sentence %) training-sentences))



(def split-training-model (word2vec/with-word2vec word2vec-db
                            (let [seed-frac 0.75
                                  rng 0.022894]
                              (re-model/split-train-test training-sentences training-model
                                                         seed-frac properties rng))))
(comment
  (def results (evaluation/run-model {:seed-frac               0.2
                                      :rng                     0.022894
                                      :context-path-length-cap 100
                                      :context-thresh          0.95
                                      :cluster-thresh          0.95
                                      :min-match-support       0
                                      :max-iterations          100
                                      :max-matches             3000
                                      :re-clustering?          true}
                                     training-model word2vec-db
                                     training-sentences results-dir
                                     split-training-model))

  (evaluation/plot-metrics (get results 1) properties
                           {:view true
                            :save {:file (io/file results-dir "metrics-0.75.svg")}})


  #_(apply evaluation/format-matches training-model results)

  (def param-walk-results (evaluation/parameter-walk word2vec-db results-dir
                                                     properties training-sentences training-model
                                                     {:context-path-length-cap [100] #_[2 3 5 10 20 35 100]
                                                      :context-thresh          [0.95] #_[0.975 0.95 0.925 0.9 0.85]
                                                      :cluster-thresh          [0.95] #_[0.95 0.9 0.75 0.5]
                                                      :min-match-support       [0] #_[0 5 25]
                                                      :seed-frac #_[0.2]       [0.05 0.25 0.5 0.75]
                                                      :rng                     0.022894})))