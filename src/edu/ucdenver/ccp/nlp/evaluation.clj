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
  [model matches patterns]
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
  [f model matches patterns]
  (let [formatted (format-matches model matches patterns)
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
  [model property]
  (->> (actual-true model property)
       (potential-seeds (:sentences model))
       (mapcat #(apply make-seeds (model :sentences) %))))

(defn context-path-filter
  [dep-filt coll]
  (filter #(<= (count (:context %)) dep-filt) coll))

(defn frac-seeds
  [model property frac]
  (let [actual-true (actual-true model property)
        num-seeds (-> actual-true
                      (count)
                      (* frac))
        seeds (set (take num-seeds (make-all-seeds model property)))
        model (update model :sentences #(remove seeds %))
        model (assoc model :actual-true actual-true
                           :all (all-triples model))]
    [model seeds]))


(defn parameter-walk
  [property model & {:keys [context-path-length-cap context-thresh cluster-thresh min-support seed-frac]}]
  (for [seed-frac seed-frac]
    (let [split-model (frac-seeds model property seed-frac)]
      (for [context-path-length-cap context-path-length-cap
            context-thresh context-thresh
            cluster-thresh cluster-thresh
            min-support min-support]
        (let [sentences (context-path-filter context-path-length-cap (get-in split-model [0 :sentences]))
              params {:context-match-fn  (fn [s p]
                                           (and (re/sent-pattern-concepts-match? s p)
                                                (< context-thresh (re/context-vector-cosine-sim s p))))
                      :cluster-merge-fn  re/add-to-pattern
                      :cluster-match-fn  #(let [score (re/context-vector-cosine-sim %1 %2)]
                                            (and (< (or %3 cluster-thresh) score)
                                                 score))
                      :pattern-update-fn (fn [patterns _]
                                           (filter (fn [{:keys [support]}]
                                                     (<= min-support (count support)))
                                                   patterns))}
              [matches _] (re/cluster-bootstrap-extract-relations-persistent-patterns (get-in split-model [1]) sentences params)
              model (assoc model :predicted-true (predicted-true matches))
              metrics (let [metrics (try
                                      (math/calc-metrics model)
                                      (catch ArithmeticException _ {}))]
                        (assoc metrics :context-path-length-cap context-path-length-cap
                                       :context-thresh context-thresh
                                       :cluster-thresh cluster-thresh
                                       :min-support min-support
                                       :seed-frac seed-frac))]
          (log/info "Metrics:" metrics)
          metrics)))))
