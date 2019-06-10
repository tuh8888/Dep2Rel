(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [cluster-tools]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [incanter.stats :as inc-stats]
            [com.climate.claypoole :as cp]
            [ubergraph.core :as uber]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [uncomplicate-context-alg :as context]))

(defn format-matches
  [model matches _]
  (map (fn [match]
         (let [[e1 _ :as entities] (map #(get-in model [:concept-annotations %]) (:entities match))

               doc (:doc e1)
               sent (->> (get-in model [:structure-graphs (:sent e1) :node-map])
                         keys
                         (sentence/pprint-sent model))
               context (->> match
                            :context
                            (sentence/pprint-sent model))
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

(defn frac-seeds
  [property sentences frac]
  (let [pot (->> sentences
                 (filter #(= (:property %) property))
                 (shuffle))]
    (-> pot
        (count)
        (* frac)
        (take pot)
        (set))))

(defn split-train-test
  "Splits model into train and test sets"
  [sentences model frac properties]
  (let [seeds (->> (group-by :property sentences)
                   (filter #(properties (first %)))
                   (map (fn [[property sentences]] (frac-seeds property sentences frac)))
                   (apply clojure.set/union))]
    (assoc model :samples (remove seeds sentences)
                 :seeds (->> seeds
                             (map #(assoc % :predicted (:property %)))
                             (set)))))

(defn sentences->entities
  [sentences]
  (->> sentences
       (map :entities)
       (map set)
       (set)))


(defn calc-metrics
  [{:keys [matches properties samples]}]
  (let [all (sentences->entities samples)
        metrics (zipmap properties
                  (map (fn [property]
                         (let [actual-true (->> samples
                                                (filter #(= property (:property %)))
                                                (sentences->entities))
                               predicted-true (->> matches
                                                   (filter #(= property (:predicted %)))
                                                   (sentences->entities))]

                           (log/debug "ALL" (count all) "AT" (count actual-true) "PT" (count predicted-true))
                           (try
                             (math/calc-metrics {:actual-true    actual-true
                                                 :predicted-true predicted-true
                                                 :all            all})
                             (catch ArithmeticException _ {}))))
                       properties))]
    (log/info "Metrics:" (seq metrics))
    metrics))

(defn parameter-walk
  [properties sentences model {:keys [context-match-fn pattern-update-fn terminate?
                                      context-path-length-cap
                                      context-thresh cluster-thresh
                                      min-seed-support min-match-support min-match-matches
                                      seed-frac]}]
  (cp/upfor (dec (cp/ncpus)) [seed-frac seed-frac
                              :let [split-model (split-train-test sentences model seed-frac properties)]
                              context-path-length-cap context-path-length-cap
                              context-thresh context-thresh
                              cluster-thresh cluster-thresh
                              min-seed-support min-seed-support
                              min-match-support min-match-support
                              min-match-matches min-match-matches]
            (let [params {:seed-frac               seed-frac
                          :context-path-length-cap context-path-length-cap
                          :context-thresh          context-thresh
                          :cluster-thresh          cluster-thresh
                          :min-match-support       min-match-support
                          :min-seed-support        min-seed-support
                          :min-match-matches       min-match-matches}
                  context-match-fn (partial context-match-fn params)
                  pattern-update-fn (partial pattern-update-fn context-match-fn params)]
              (-> split-model
                  (update :sentences context-path-filter context-path-length-cap)
                  (re/bootstrap {:context-match-fn  context-match-fn
                                 :pattern-update-fn pattern-update-fn
                                 :terminate?        terminate?})
                  (calc-metrics)
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
  [sentences model]
  (->> sentences
       (filter #(context/context-vector % model))
       (pmap flatten-context-vector)
       (map #(dissoc % :entities :concepts :context))
       (vec)
       (incanter/to-dataset)))

(defn sent-property
  [{:keys [concept-graphs]} [id1 id2]]
  (some
    (fn [g]
      (when-let [e (or (uber/find-edge g id2 id1) (uber/find-edge g id1 id2))]
        (:value (uber/attrs g e))))
    (vals concept-graphs)))

(defn assign-property
  "Assign the associated property with the sentence"
  [model s]
  (assoc s :property (sent-property model (vec (:entities s)))))
