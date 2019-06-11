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
            [edu.ucdenver.ccp.nlp.re-model :as re-model]))

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


(defn calc-metrics
  [{:keys [matches properties samples]}]
  (let [all (sentences->entities samples)
        metrics (map (fn [property]
                       (let [actual-true (->> samples
                                              (filter #(= property (:property %)))
                                              (sentences->entities))
                             predicted-true (->> matches
                                                 (filter #(= property (:predicted %)))
                                                 (sentences->entities))]
                         #_(log/debug "ALL" (count all) "AT" (count actual-true) "PT" (count predicted-true))
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

(defn parameter-walk
  [properties sentences model {:keys [context-match-fn pattern-update-fn terminate?
                                      support-filter decluster
                                      context-path-length-cap
                                      context-thresh cluster-thresh
                                      min-match-support
                                      seed-frac rng]}]
  (cp/upfor (dec (cp/ncpus)) [seed-frac seed-frac
                              :let [split-model (re-model/split-train-test sentences model seed-frac properties rng)]
                              context-path-length-cap context-path-length-cap
                              context-thresh context-thresh
                              cluster-thresh cluster-thresh
                              min-match-support min-match-support]
            (let [context-path-length-cap context-path-length-cap
                  params {:context-thresh    context-thresh
                          :cluster-thresh    cluster-thresh
                          :min-match-support min-match-support
                          :max-iterations    100
                          :max-matches       3000
                          :re-clustering?    true
                          :context-path-length-cap context-path-length-cap
                          :seed-frac seed-frac
                          :factory           (:factory split-model)
                          :vector-fn         #(context/context-vector % split-model)}
                  context-match-fn (partial context-match-fn params)
                  pattern-update-fn (partial pattern-update-fn params model)
                  terminate? (partial terminate? params)
                  support-filter (partial support-filter params)
                  decluster (partial decluster params support-filter)]
              (-> split-model
                  (update :samples (fn [samples] (context-path-filter context-path-length-cap samples)))
                  (re/bootstrap {:terminate?        terminate?
                                 :context-match-fn  context-match-fn
                                 :pattern-update-fn pattern-update-fn
                                 :support-filter    support-filter
                                 :decluster         decluster})
                  (doall)
                  (merge params)))))

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

(defn pca-plot
  [sentences-dataset groups {{:as save :keys [file]} :save :keys [view]}]
  (let [y (incanter/sel sentences-dataset :cols (range 0 200))
        x (pca-2 y)
        plot (inc-charts/scatter-plot (get x 0) (get x 1)
                                      :group-by groups
                                      :legend true
                                      :x-label "PC1"
                                      :y-label "PC2"
                                      :title "PCA")]
    (when save (inc-svg/save-svg plot file))
    (when view (incanter/view plot))))

(defn plot-metrics
  [metrics-dataset groups {{:as save :keys [file]} :save :keys [view]}]
  (let [plot (inc-charts/scatter-plot (incanter/sel metrics-dataset :cols :precision)
                                      (incanter/sel metrics-dataset :cols :recall)
                                      :group-by groups
                                      :legend true
                                      :x-label "Precision"
                                      :y-label "Recall"
                                      :title "Relation Extraction Results")]
    (when save (inc-svg/save-svg plot file))
    (when view (incanter/view plot))))
