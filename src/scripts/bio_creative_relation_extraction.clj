(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]
            [incanter.core :as incanter]
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

(def factory thal-native/native-double)

;;; MODELS ;;;
(def training-knowtator-view (k/view training-dir))
(rdr/read-biocreative-files training-dir training-pattern training-knowtator-view)
(def training-model (word2vec/with-word2vec word2vec-db
                      (re-model/make-model training-knowtator-view factory)))
(def training-sentences (re-model/make-sentences training-model))

#_(def testing-knowtator-view (k/view testing-dir))
#_(rdr/read-biocreative-files testing-dir testing-pattern testing-knowtator-view)
#_(def testing-model (word2vec/with-word2vec word2vec-db
                       (make-model testing-knowtator-view)))

;;; CLUSTERING ;;;
(def properties #_#{"INHIBITOR"} #{"PART-OF"
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
(def triples-dataset (->> training-sentences
                          (filter #(or (= "NONE" (:property %))
                                       (properties (:property %))))
                          (evaluation/sentences->dataset training-model)))

(def groups (incanter/sel triples-dataset :cols :property))
(def y (incanter/sel triples-dataset :cols (range 0 200)))
(def x (evaluation/pca-2 y))

(evaluation/pca-plot x groups {:save {:file "pca-all.svg"}
                               :view true})

;;; RELATION EXTRACTION ;;;
(log/set-level! :info)

(def split-training-model (word2vec/with-word2vec word2vec-db
                            (let [seed-frac 0.2
                                  rng 0.022894]
                              (re-model/split-train-test training-sentences training-model seed-frac properties rng))))


(def results (let [context-path-length-cap 100
                   params {:context-thresh    0.95
                           :cluster-thresh    0.95
                           :min-match-support 1
                           :max-iterations    100
                           :max-matches       3000
                           :re-clustering?    true
                           :factory           (:factory split-training-model)
                           :vector-fn         #(context/context-vector % split-training-model)}
                   context-match-fn (partial re/concept-context-match params)
                   pattern-update-fn (partial re/pattern-update params)
                   terminate? (partial re/terminate? params)
                   support-filter (partial re/support-filter params)
                   decluster (partial re/decluster params support-filter)]
               (-> split-training-model
                   (update :samples (fn [samples] (evaluation/context-path-filter context-path-length-cap samples)))
                   (re/bootstrap {:terminate?        terminate?
                                  :context-match-fn  context-match-fn
                                  :pattern-update-fn pattern-update-fn
                                  :support-filter    support-filter
                                  :decluster         decluster})

                   (evaluation/calc-metrics)
                   (doall))))

(incanter/to-dataset (map #(assoc (second %) :property (first %)) results))


(comment
  #_(apply evaluation/format-matches training-model results)
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