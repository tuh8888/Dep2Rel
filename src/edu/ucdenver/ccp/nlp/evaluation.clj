(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [cluster-tools]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [incanter.stats :as inc-stats]
            [incanter.charts :as inc-charts]
            [incanter.svg :as inc-svg]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.seeds :as seeds]))

(def EVAL-KEYS #{:fn :tp :fp :tn :precision :recall :f1 :metrics :overall-metrics})

(defn format-matches
  [model matches _]
  (map (fn [match]
         (let [[e1 _ :as entities] (map #(get-in model [:concept-annotations %]) (:entities match))

               doc          (:doc e1)
               sent-text    (->> (:sent-id e1)
                                 (keys)
                                 (re-model/pprint-sent-text model))
               context-text (->> match
                                 :context
                                 (re-model/pprint-toks-text model))
               [e1-concept e2-concept] (->> entities
                                            (sort-by :concept)
                                            (map :concept)
                                            (map str))
               [e1-tok e2-tok] (->> entities
                                    (map :tok)
                                    (map #(get-in model [:structure-annotations %]))
                                    (map (comp :text first vals :spans)))
               seed         (->> (get match :seed)
                                 :concepts
                                 (mapcat identity)
                                 (interpose ", "))]
           {:doc        doc
            :context    context-text
            :e1-concept e1-concept
            :e1-tok     e1-tok
            :e2-concept e2-concept
            :e2-tok     e2-tok
            :seed       (apply str seed)
            :sentence   (str "\"" sent-text "\"")}))

       matches))

(defn ->csv
  [f model matches patterns]
  (let [formatted (format-matches model matches patterns)
        cols      [:doc :e1-concept :e1-tok :e2-concept :e2-tok :seed :sentence]]
    (->> formatted
         (map #(str (->> %
                         ((apply juxt cols))
                         (interpose ",")
                         (apply str))
                    "\n"))
         (apply str)
         (str (->> cols
                   (interpose ",")
                   (apply str))
              "\n")
         (spit f))))


(defn cluster-sentences
  [sentences]
  (map
    #(map :entities %)
    (map :support
         (filter
           #(when (< 1 (count (:support %)))
              %)
           (cluster-tools/single-pass-cluster sentences #{}
             {:cluster-merge-fn re-model/add-to-pattern})))))

(defn pca-2
  [data]
  (let [X          (incanter/to-matrix data)
        pca        (inc-stats/principal-components X)
        components (:rotation pca)
        pc1        (incanter/sel components :cols 0)
        pc2        (incanter/sel components :cols 1)
        x1         (incanter/mmult X pc1)
        x2         (incanter/mmult X pc2)]
    [x1 x2]))

(defn calc-metrics
  [{:keys [matches properties]}]
  (->> properties
       (map (fn [p]
              (let [tp        (->> matches
                                   (filter #(= (:property %) p))
                                   (filter #(= (:predicted %) p))
                                   (count))
                    fp        (->> matches
                                   (filter #(not= p (:property %)))
                                   (filter #(= p (:predicted %)))
                                   (count))
                    tn        (->> matches
                                   (filter #(not= (:property %) p))
                                   (filter #(not= (:predicted %) p))
                                   (count))
                    fn        (->> matches
                                   (filter #(= (:property %) p))
                                   (filter #(not= (:predicted %) p))
                                   (count))
                    precision (/ (float tp) (+ fp tp))
                    recall    (/ (float tp) (+ tp fn))
                    f1        (/ (* 2 precision recall)
                                 (+ precision recall))
                    metrics   {:property  p
                               :tp        tp
                               :tn        tn
                               :fp        fp
                               :fn        fn
                               :precision precision
                               :recall    recall
                               :f1        f1}]
                metrics)))))

(defn calc-overall-metrics
  [{:keys [matches properties]}]
  (try
    (let [properties (disj properties re-model/NONE)
          tp         (->> properties
                          (mapcat (fn [p]
                                    (->> matches
                                         (filter #(= (:property %) p))
                                         (filter #(= (:predicted %) p)))))
                          (count))
          fp         (->> properties
                          (mapcat (fn [p]
                                    (->> matches
                                         (filter #(not= p (:property %)))
                                         (filter #(= p (:predicted %))))))
                          (count))
          tn         (->> properties
                          (mapcat (fn [p]
                                    (->> matches
                                         (filter #(not= (:property %) p))
                                         (filter #(not= (:predicted %) p)))))
                          (count))
          fn         (->> properties
                          (mapcat (fn [p]
                                    (->> matches
                                         (filter #(= (:property %) p))
                                         (filter #(not= (:predicted %) p)))))
                          (count))
          precision  (/ (float tp) (+ fp tp))
          recall     (/ (float tp) (+ tp fn))
          f1         (/ (* 2 precision recall)
                        (+ precision recall))
          metrics    {:tp        tp
                      :tn        tn
                      :fp        fp
                      :fn        fn
                      :precision precision
                      :recall    recall
                      :f1        f1}]
      metrics)
    (catch Exception _ nil)))

(defn make-property-plot
  [{:keys [x-label y-label title]} property groups x y]
  (let [xy (->> groups
                (filter #(= (second %) property))
                (keep #(vector (get x (first %)) (get y (first %))))
                (vec))
        x  (vec (map first xy))
        y  (vec (map second xy))]
    (inc-charts/scatter-plot x y
                             :x-label x-label
                             :y-label y-label
                             :series-label property
                             :title title
                             :legend true)))

(defn add-property-series
  [plot p groups x y {{:as save :keys [file]} :save :keys [title]}]
  (let [p2-xy (->> groups
                   (filter #(= (second %) p))
                   (keep #(vector (get x (first %)) (get y (first %))))
                   (vec))]
    (if (seq p2-xy)
      (let [p2-x (vec (map first p2-xy))
            p2-y (vec (map second p2-xy))]
        (inc-charts/add-points plot p2-x p2-y :series-label p)
        (when save (inc-svg/save-svg plot (format (str file ".svg") title)))
        plot)
      (log/warn "No points found for" p))))

(defn property-plot
  ([{:keys [properties]} dataset x y {{:as save :keys [file]} :save :as params}]
   (let [groups (map-indexed vector (incanter/sel dataset :cols :property))]
     (let [property (first properties)
           plot     (make-property-plot params property groups x y)]
       (doseq [property (rest properties)] (add-property-series plot property groups x y (assoc params :save false)))
       (when save (inc-svg/save-svg plot (str file)))
       plot)))
  ([{:keys [properties]} dataset p1 x y params]
   (let [properties (disj properties p1)
         groups     (map-indexed vector (incanter/sel dataset :cols :property))]
     (->> properties
          (keep (fn [p2]
                  (let [params (update params :title (fn [title] (format "%s for %s and %s" title p1 p2)))]
                    [[p1 p2] (-> params
                                 (make-property-plot p1 groups x y)
                                 (add-property-series p2 groups x y params))])))
          (into {})))))

(defn pca-plots
  [{:keys [sentences-dataset sentences] :as model} params]
  (let [cols           (->> model
                            (re-model/context-vector (first sentences))
                            (count)
                            (range 0))
        numerical-data (incanter/sel sentences-dataset :cols cols)
        pca-components (pca-2 numerical-data)
        x              (vec (get pca-components 0))
        y              (vec (get pca-components 1))
        plots          (property-plot model sentences-dataset re-model/NONE x y (assoc params :x-label "PC1"
                                                                                              :y-label "PC2"
                                                                                              :title "PCA"))]
    (assoc plots "ALL" (property-plot model sentences-dataset x y (assoc params :x-label "PC1"
                                                                                :y-label "PC2"
                                                                                :title "PCA")))))

(defn plot-metrics
  [{:keys [metrics] :as model} {:keys [view] :as params}]
  (let [metrics-dataset (incanter/to-dataset (or metrics (calc-metrics model)))
        x               (vec (incanter/sel metrics-dataset :cols :precision))
        y               (vec (incanter/sel metrics-dataset :cols :recall))
        plot            (property-plot model metrics-dataset x y (assoc params :x-label "Precision"
                                                                               :y-label "Recall"
                                                                               :title "Relation Extraction Results"))]
    (when view (incanter/view plot))
    plot))


(defn run-model
  "Run model with parameters"
  [model results-dir]
  (let [model           (if (contains? model :all-samples)
                          model
                          (re-model/split-train-test model))
        results         (-> model
                            (re/bootstrap)
                            (doall))
        metrics         (calc-metrics results)
        overall-metrics (calc-overall-metrics results)
        results         (merge results overall-metrics {:property-metrics metrics})]
    (->> metrics
         (incanter/to-dataset)
         (log/info))
    (log/info "Overall Metrics" overall-metrics)
    (doto (io/file results-dir "results.edn")
      (spit (select-keys results (lazy-cat EVAL-KEYS re/PARAM-KEYS)) :append true)
      (spit "\n" :append true))
    (assoc results :plot (plot-metrics results
                                       {:save {:file (->> (select-keys results (disj re/PARAM-KEYS :match-fn))
                                                          (format "metrics-%s.svg")
                                                          (io/file results-dir))}}))))

(defn parameter-walk
  [{:keys [properties] :as training-model} testing-model results-dir {:keys [context-path-length-cap
                                                                             match-thresh
                                                                             cluster-thresh
                                                                             confidence-thresh
                                                                             min-pattern-support
                                                                             seed-frac
                                                                             rng negative-cap
                                                                             match-fn
                                                                             min-seed-pattern-precision
                                                                             min-seed-pattern-recall]}]
  (doall
    ;; parallelize with
    #_(cp/upfor (dec (cp/ncpus)))
    (for [seed-frac                  seed-frac
          :let [prepared-model (let [split-model (re-model/split-train-test (assoc training-model :seed-frac seed-frac
                                                                                                  :rng rng
                                                                                                  :negative-cap negative-cap))]
                                 (cond (contains? testing-model :all-seed-patterns) testing-model
                                      (seq testing-model) (re-model/train-test split-model testing-model)
                                      :else split-model))]
          context-path-length-cap    context-path-length-cap
          context-thresh             match-thresh
          cluster-thresh             cluster-thresh
          confidence-thresh          confidence-thresh
          min-match-support          min-pattern-support
          min-seed-pattern-precision min-seed-pattern-precision
          min-seed-pattern-recall    min-seed-pattern-recall
          match-fn                   match-fn]

      (-> prepared-model
          (update :seeds (fn [seeds] (if (and (zero? min-seed-pattern-recall)
                                              (zero? min-seed-pattern-precision))
                                       seeds
                                       nil)))
          (assoc :patterns (lazy-cat
                             (apply min-key count
                                    (-> prepared-model
                                        :all-seed-patterns
                                        (seeds/seed-patterns-with-selectivity properties
                                                                              {:min-recall    min-seed-pattern-recall
                                                                               :min-precision min-seed-pattern-precision})))
                             #_(apply max-key count
                                      (-> prepared-model
                                          :all-seed-patterns
                                          (seeds/seed-patterns-with-selectivity properties
                                                                                {:min-f1        0
                                                                                 :min-recall    0
                                                                                 :min-precision 0.7}))))
                 :match-thresh context-thresh
                 :cluster-thresh cluster-thresh
                 :confidence-thresh confidence-thresh
                 :min-pattern-support min-match-support
                 :min-seed-pattern-precision min-seed-pattern-precision
                 :min-seed-pattern-recall min-seed-pattern-recall
                 :max-iterations 0
                 :max-matches 5000
                 :re-clustering? true
                 :context-path-length-cap context-path-length-cap
                 :match-fn match-fn)
          (run-model results-dir)))))

(defn flatten-context-vector
  [s model]
  (let [v (vec (seq (re-model/context-vector s model)))]
    (apply assoc s (interleave (range (count v)) v))))

(defn sentences->dataset
  [{:keys [properties sentences word2vec-db] :as model}]
  (word2vec/with-word2vec word2vec-db
    (assoc model
      :sentences-dataset (->> sentences
                              (filter #(contains? properties (:property %)))
                              (filter #(re-model/context-vector % model))
                              (pmap #(flatten-context-vector % model))
                              (map #(dissoc % :VEC :entities :concepts :context :support :predicted))
                              (vec)
                              (incanter/to-dataset)))))

(defn plot-context-lengths
  [{:keys [sentences]} results-dir fmt]
  (incanter/with-data (incanter/$order :count :asc
                                       (incanter/to-dataset (->> sentences
                                                                 (map :context)
                                                                 (map count)
                                                                 (frequencies)
                                                                 (map (fn [[cnt n]]
                                                                        {:count cnt
                                                                         :num   n})))))
                      (let [title (format fmt "Context Path Lengths")
                            plot  (inc-charts/bar-chart :count :num
                                                        :title title
                                                        :x-label "Context Path Length"
                                                        :y-label "Frequency")]
                        (inc-svg/save-svg plot (str (io/file results-dir title) ".svg") :width 1000)
                        plot)))
