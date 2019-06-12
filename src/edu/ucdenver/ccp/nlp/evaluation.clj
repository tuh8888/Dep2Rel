(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [cluster-tools]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [incanter.stats :as inc-stats]
            [com.climate.claypoole :as cp]
            [uncomplicate-context-alg :as context]
            [incanter.charts :as inc-charts]
            [incanter.svg :as inc-svg]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [clojure.java.io :as io]))

(defn format-matches
  [model matches _]
  (map (fn [match]
         (let [[e1 _ :as entities] (map #(get-in model [:concept-annotations %]) (:entities match))

               doc (:doc e1)
               sent (->> (get-in model [:structure-graphs (:sent e1) :node-map])
                         keys
                         (re-model/pprint-sent model))
               context (->> match
                            :context
                            (re-model/pprint-sent model))
               [e1-concept e2-concept] (->> entities
                                            (sort-by :concept)
                                            (map :concept)
                                            (map str))
               [e1-tok e2-tok] (->> entities
                                    (map :tok)
                                    (map #(get-in model [:structure-annotations %]))
                                    (map (comp :text first vals :spans)))
               seed (->> (get match :seed)
                         :concepts
                         (mapcat identity)
                         (interpose ", "))]
           {:doc        doc
            :context    context
            :e1-concept e1-concept
            :e1-tok     e1-tok
            :e2-concept e2-concept
            :e2-tok     e2-tok
            :seed       (apply str seed)
            :sentence   (str "\"" sent "\"")}))

       matches))

(defn ->csv
  [f model matches patterns]
  (let [formatted (format-matches model matches patterns)
        cols [:doc :e1-concept :e1-tok :e2-concept :e2-tok :seed :sentence]
        csv-form (str (apply str (interpose "," cols)) "\n" (apply str (map #(str (apply str (interpose "," ((apply juxt cols) %))) "\n") formatted)))]
    (spit f csv-form)))

(defn cluster-sentences
  [sentences]
  (map
    #(map :entities %)
    (map :support
         (filter
           #(when (< 1 (count (:support %)))
              %)
           (cluster-tools/single-pass-cluster sentences #{}
             {:cluster-merge-fn re/add-to-pattern})))))

(defn context-path-filter
  [dep-filter coll]
  (filter #(<= (count (:context %)) dep-filter) coll))

(defn sentences->entities
  [sentences]
  (->> sentences
       (map :entities)
       (map set)
       (set)))

(defn pca-2
  [data]
  (let [X (incanter/to-matrix data)
        pca (inc-stats/principal-components X)
        components (:rotation pca)
        pc1 (incanter/sel components :cols 0)
        pc2 (incanter/sel components :cols 1)
        x1 (incanter/mmult X pc1)
        x2 (incanter/mmult X pc2)]
    [x1 x2]))


(defn actual-true
  [property samples]
  (->> samples
       (filter #(= property (:property %)))
       (sentences->entities)))

(defn predicted-true
  [property matches]
  (->> matches
       (filter #(= property (:predicted %)))
       (sentences->entities)))

(defn calc-metrics
  [{:keys [matches properties samples]}]
  (let [all (sentences->entities samples)
        metrics (map (fn [property]
                       (let [actual-true (actual-true property samples)
                             predicted-true (predicted-true property matches)]
                         #_(log/info property "ALL" (count all) "AT" (count actual-true) "PT" (count predicted-true))
                         (-> (try
                               (math/calc-metrics {:actual-true    actual-true
                                                   :predicted-true predicted-true
                                                   :all            all})
                               (catch ArithmeticException _ {}))
                             (assoc :property property))))
                     properties)]
    (->> metrics
         (incanter/to-dataset)
         (log/info))
    metrics))

(defn add-property-series
  [plot dataset x y properties]
  (let [groups (map-indexed vector (incanter/sel dataset :cols :property))]
    (doseq [property properties]
      (let [x (vec (keep (fn [[i p]] (when (= p property)
                                       (get x i)))
                         groups))
            y (vec (keep (fn [[j p]] (when (= p property)
                                       (get y j)))
                         groups))]
        (when (and x y)
          (inc-charts/add-points plot x y :series-label property))))))

(defn pca-plot
  [properties sentences-dataset cols {{:as save :keys [file]} :save :keys [view]}]
  (let [numerical-data (incanter/sel sentences-dataset :cols (range 0 cols))
        pca-components (pca-2 numerical-data)
        plot (inc-charts/scatter-plot [] []
                                      :legend true
                                      :x-label "PC1"
                                      :y-label "PC2"
                                      :title "PCA")
        x (get pca-components 0)
        y (get pca-components 1)]
    (add-property-series plot sentences-dataset x y properties)
    (when save (inc-svg/save-svg plot (str file)))
    (when view (incanter/view plot))
    plot))

(defn plot-metrics
  [metrics-dataset properties {{:as save :keys [file]} :save :keys [view]}]
  (let [plot (inc-charts/scatter-plot [] []
                                      :legend true
                                      :x-label "Precision"
                                      :y-label "Recall"
                                      :title "Relation Extraction Results")
        x (vec (incanter/sel metrics-dataset :cols :precision))
        y (vec (incanter/sel metrics-dataset :cols :recall))]
    (add-property-series plot metrics-dataset x y properties)
    (when save (inc-svg/save-svg plot (str file)))
    (when view (incanter/view plot))
    plot))


(defn run-model
  "Run model with parameters"
  [{:keys [seed-frac rng context-path-length-cap] :as params} {:keys [properties] :as model}
   word2vec-db sentences results-dir split-model]

  (let [split-model (or split-model
                        (word2vec/with-word2vec word2vec-db
                          (re-model/split-train-test sentences model
                                                     seed-frac properties rng)))
        results (let [params (merge params {:factory   (:factory split-model)
                                            :vector-fn #(context/context-vector % split-model)})
                      context-match-fn (partial re/concept-context-match params)
                      pattern-update-fn (partial re/pattern-update params model)
                      terminate? (partial re/terminate? params)
                      support-filter (partial re/support-filter params)
                      decluster (partial re/decluster params support-filter)]
                  (-> split-model
                      (update :samples (fn [samples] (context-path-filter context-path-length-cap samples)))
                      (re/bootstrap {:terminate?        terminate?
                                     :context-match-fn  context-match-fn
                                     :pattern-update-fn pattern-update-fn
                                     :support-filter    support-filter
                                     :decluster         decluster})
                      (doall)))
        metrics (calc-metrics results)
        plot (plot-metrics (incanter/to-dataset metrics) properties
                           {:save {:file (->> params
                                              (format "metrics-%s.svg")
                                              (io/file results-dir))}})]
    [results metrics plot]))

(defn parameter-walk
  [word2vec-db results-dir properties sentences model
   {:keys [context-path-length-cap
           context-thresh
           cluster-thresh
           min-match-support
           seed-frac
           rng]}]
  ;; parallelize with
  #_(cp/upfor (dec (cp/ncpus)))
  (for [seed-frac seed-frac
        :let [split-model (re-model/split-train-test sentences model seed-frac properties rng)]
        context-path-length-cap context-path-length-cap
        context-thresh context-thresh
        cluster-thresh cluster-thresh
        min-match-support min-match-support]
    (let [params {:context-thresh          context-thresh
                  :cluster-thresh          cluster-thresh
                  :min-match-support       min-match-support
                  :max-iterations          100
                  :max-matches             3000
                  :re-clustering?          true
                  :context-path-length-cap context-path-length-cap
                  :seed-frac               seed-frac
                  :rng                     rng}]
      (log/warn params)
      (run-model params model word2vec-db sentences results-dir split-model))))

(defn flatten-context-vector
  [s model]
  (let [v (vec (seq (context/context-vector s model)))]
    (apply assoc s (interleave (range (count v)) v))))

(defn sentences->dataset
  [model sentences]
  (->> sentences
       (filter #(context/context-vector % model))
       (pmap #(flatten-context-vector % model))
       (map #(dissoc % :entities :concepts :context))
       (vec)
       (incanter/to-dataset)))
