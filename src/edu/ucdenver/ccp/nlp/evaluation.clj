(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [cluster-tools]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [taoensso.timbre :as log]
            [com.climate.claypoole :as cp]))

(defn count-seed-matches
  [matches]
  (->> matches (group-by :seed)
       (map
         #(vector (->>
                    (first %)
                    (:entities)
                    (map :concept))
                  (count (second %))))
       (into {})))

(defn pprint-sent
  [model sent]

  (->> sent
       (map #(get-in model [:structure-annotations %]))
       (map (comp first vals :spans))
       (sort-by :start)
       (map :text)
       (interpose " ")
       (apply str)))

(defn format-matches
  [model matches]
  (map (fn [match]
         (let [[e1 _ :as entities] (map #(get-in model [:concept-annotations %]) (:entities match))

               doc (:doc e1)
               sent (->> (get-in model [:structure-graphs (:sent e1) :node-map])
                         keys
                         (pprint-sent model))
               context (->> match
                            :context
                            (pprint-sent model))
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
            :sentence   (str "\"" sent "\"")
            }))
       matches))

(defn ->csv
  [f model matches]
  (let [formatted (format-matches model matches)
        cols [:doc :e1-concept :e1-tok :e2-concept :e2-tok :seed :sentence]
        csv-form (str (apply str (interpose "," cols)) "\n" (apply str (map #(str (apply str (interpose "," ((apply juxt cols) %))) "\n") formatted)))]
    (spit f csv-form)))

(defn sent->triple
  [match]
  (set (:entities match)))

(defn edge->triple
  [edge]
  #{(:src edge) (:dest edge)})

(defn cluster-sentences
  [sentences]
  (map
    #(map :entities %)
    (map :support
         (filter
           #(when (< 1 (count (:support %)))
              %)
           (cluster-tools/single-pass-cluster sentences #{}
                                              {:cluster-merge-fn re/add-to-pattern
                                               :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                                    (and (< (or %3 0.75) score)
                                                                         score))})))))

(defn make-seeds
  [sentences e1 e2]
  (clojure.set/intersection
    (set (sentence/sentences-with-ann sentences e1))
    (set (sentence/sentences-with-ann sentences e2))))

(defn predicted-true
  [matches]
  (set (map sent->triple matches)))

(defn actual-true
  [model property]
  (->> property
       (k/edges-for-property (vals (:concept-graphs model)))
       (map edge->triple)
       (set)))

(defn all-triples
  [model]
  (->> model
       :sentences
       (map sent->triple)
       (set)))

(defn potential-seeds
  [sentences actual-true]
  (filter (fn [t] (some #(= t (:entities %)) sentences)) actual-true))

(defn make-all-seeds
  [model property sentences]
  (->> (actual-true model property)
       (potential-seeds sentences)
       (mapcat #(apply make-seeds sentences %))))

(defn context-path-filter
  [dep-filt coll]
  (filter #(<= (count (:context %)) dep-filt) coll))

(defn frac-seeds
  [model sentences property frac]
  (let [num-seeds (-> (actual-true model property)
                      (count)
                      (* frac))]
    (set (take num-seeds (make-all-seeds model property sentences)))))


(defn parameter-walk
  [property model & [{:keys [cluster-thresh-min cluster-thresh-max cluster-thresh-step
                             context-thresh-min context-thresh-max context-thresh-step
                             min-support-min min-support-max min-support-step
                             context-path-length-cap-min context-path-length-cap-max context-path-length-cap-step
                             seed-frac-min seed-frac-max seed-frac-step]
                      :or   {cluster-thresh-min           0.5
                             cluster-thresh-max           0.95
                             cluster-thresh-step          0.1

                             context-thresh-min           0.85
                             context-thresh-max           0.99
                             context-thresh-step          0.02

                             min-support-min              1
                             min-support-max              30
                             min-support-step             5

                             context-path-length-cap-min  2
                             context-path-length-cap-max  35
                             context-path-length-cap-step 5

                             seed-frac-min                0.05
                             seed-frac-max                0.85
                             seed-frac-step               0.2}}]]
  (for [context-path-length-cap [35] #_[2 3 4 5 10 20 35] #_(range context-path-length-cap-min context-path-length-cap-max context-path-length-cap-step)
        context-thresh [0.95] #_ [0.975 0.95 0.925 0.9 0.85]
        cluster-thresh [0.95] #_[0.95 0.9 0.8 0.7 0.6 0.5] #_(range cluster-thresh-min cluster-thresh-max cluster-thresh-step)
        min-support [1] #_[1 3 5 10 20 30] #_(range min-support-min min-support-max min-support-step)
        seed-frac [0.05] #_[0.05 0.25 0.45 0.65 0.75] #_(range seed-frac-min seed-frac-max seed-frac-step)]
    (let [params {:sentence-filter-fn #(context-path-filter context-path-length-cap %)
                  :seed-fn            #(frac-seeds %1 %2 property seed-frac)
                  #_:context-match-fn #_#(< context-thresh (re/context-vector-cosine-sim %1 %2))
                  :context-match-fn   (fn [s p]
                                        (and (re/sent-pattern-concepts-match? s p)
                                             (< context-thresh (re/context-vector-cosine-sim s p))))
                  :cluster-merge-fn   re/add-to-pattern
                  :cluster-match-fn   #(let [score (re/context-vector-cosine-sim %1 %2)]
                                         (and (< (or %3 cluster-thresh) score)
                                              score))
                  :pattern-filter-fn  #(filter (fn [p] (<= min-support (count (:support p)))) %)
                  :pattern-update-fn  #(filter (fn [p] (<= min-support (count (:support p)))) %)}
          [model matches _] (re/init-bootstrap-persistent-patterns re/cluster-bootstrap-extract-relations-persistent-patterns model params)
          metrics (-> (try
                        (math/calc-metrics {:predicted-true (predicted-true matches)
                                            :actual-true    (actual-true model property)
                                            :all            (all-triples model)})

                        (catch ArithmeticException _ {}))
                      (assoc :context-path-length-cap context-path-length-cap
                             :context-thresh context-thresh
                             :cluster-thresh cluster-thresh
                             :min-support min-support
                             :seed-frac seed-frac))]
      (log/info "Metrics:" metrics)
      metrics)))
