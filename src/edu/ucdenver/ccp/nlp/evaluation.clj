(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [cluster-tools]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [taoensso.timbre :as log]
            [incanter.core :as incanter]
            [incanter.stats :as stats]))

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
  [model matches _]
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
  [dep-filter coll]
  (filter #(<= (count (:context %)) dep-filter) coll))

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
  [property model & {:keys [context-match-fn
                            pattern-update-fn
                            context-path-length-cap
                            context-thresh
                            cluster-thresh
                            min-seed-support
                            min-match-support
                            min-match-matches
                            seed-frac]}]
  (for [seed-frac seed-frac]
    (let [split-model (frac-seeds model property seed-frac)]
      (for [context-path-length-cap context-path-length-cap
            context-thresh context-thresh
            cluster-thresh cluster-thresh
            min-support min-seed-support
            min-match-support min-match-support
            min-match-matches min-match-matches
            context-match-fn (partial context-match-fn context-thresh)
            pattern-update-fn (partial pattern-update-fn context-match-fn {:cluster-thresh cluster-thresh
                                                                           :min-match-support min-match-support
                                                                           :min-seed-support min-seed-support
                                                                           :min-match-matches min-match-matches})]
        (let [sentences (context-path-filter context-path-length-cap (get-in split-model [0 :sentences]))
              params {:context-match-fn  context-match-fn
                      :pattern-update-fn pattern-update-fn}
              [matches _] (re/bootstrap (get-in split-model [1]) sentences params)
              model (assoc (get-in split-model [0]) :predicted-true (predicted-true matches))
              metrics (let [metrics (try
                                      (math/calc-metrics model)
                                      (catch ArithmeticException _ {}))]
                        (assoc metrics :context-path-length-cap context-path-length-cap
                                       :context-thresh context-thresh
                                       :cluster-thresh cluster-thresh
                                       :min-seed-support min-support
                                       :seed-frac seed-frac))]
          (log/info "Metrics:" metrics)
          metrics)))))

(defn pca-2
  [data]
  (let [X (incanter/to-matrix data)
        pca (stats/principal-components X)
        components (:rotation pca)
        pc1 (incanter/sel components :cols 0)
        pc2 (incanter/sel components :cols 1)
        x1 (incanter/mmult X pc1)
        x2 (incanter/mmult X pc2)]
    [x1 x2]))

(defn triple->sent
  [t sentences]
  (->> sentences
       (filter #(= t (sent->triple %)))
       (first)))
(defn edge->sent
  [g e sentences]
  (let [t (edge->triple e)
        property (:value (ubergraph.core/attrs g e))]
    (assoc (triple->sent t sentences) :property property)))

(defn flatten-context-vector
  [s]
  (let [v (vec (seq (:context-vector s)))]
    (apply assoc s (interleave (range (count v)) v))))

(defn sentences->dataset
  [sentences]
  (->> sentences
       (filter #(identity (:context-vector %)))
       (pmap flatten-context-vector)
       (map #(dissoc % :context-vector :entities :concepts :context))
       (vec)
       (incanter/to-dataset)))

(defn triples->dataset
  [model]
  (->> model
       :concept-graphs
       (vals)
       (mapcat
         (fn [g]
           (map #(edge->sent g % (:sentences model))
                (ubergraph.core/find-edges g {}))))
       (sentences->dataset)))
