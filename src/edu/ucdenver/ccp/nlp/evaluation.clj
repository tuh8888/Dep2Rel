(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [cluster-tools]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.knowtator-clj :as k]))

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
  [model property frac]
  (let [num-seeds (-> (actual-true model property)
                      (count)
                      (* frac))]
    (set (take num-seeds (make-all-seeds model property (:sentences model))))))

;
;(defn parameter-walk
;  [seeds sentences & [{:keys [seed-thresh-min seed-thresh-max seed-thresh-step
;                                                   cluster-thresh-min cluster-thresh-max cluster-thresh-step
;                                                   context-thresh-min context-thresh-max context-thresh-step
;                                                   min-support-min min-support-max min-support-step]
;                                            :or   {seed-thresh-min     0.5
;                                                   seed-thresh-max     0.99
;                                                   seed-thresh-step    0.1
;
;                                                   cluster-thresh-min  0.5
;                                                   cluster-thresh-max  0.99
;                                                   cluster-thresh-step 0.1
;
;                                                   context-thresh-min  0.5
;                                                   context-thresh-max  0.99
;                                                   context-thresh-step 0.1
;
;                                                   min-support-min     0.01
;                                                   min-support-max     0.05
;                                                   min-support-step    0.01}}]]
;  (mapcat
;    (fn [seed-thresh]
;      (mapcat
;        (fn [context-thresh]
;          (mapcat
;            (fn [cluster-thresh]
;              (mapcat
;                (fn [min-support]
;                  (let [params {:seed             (first seeds)
;                                :seed-thresh      seed-thresh
;                                :context-thresh   context-thresh
;                                :seed-match-fn    #(and (concepts-match? %1 %2)
;                                                        (< seed-thresh (context-vector-cosine-sim %1 %2)))
;                                :context-match-fn #(< context-thresh (context-vector-cosine-sim %1 %2))
;                                :cluster-merge-fn add-to-pattern
;                                :cluster-match-fn #(let [score (context-vector-cosine-sim %1 %2)]
;                                                     (and (< (or %3 cluster-thresh) score)
;                                                          score))
;                                :min-support      min-support}]
;                    (->> (cluster-bootstrap-extract-relations seeds sentences params)
;                         (map #(merge % params)))))
;                (range min-support-min min-support-max min-support-step)))
;            (range cluster-thresh-min cluster-thresh-max cluster-thresh-step)))
;        (range context-thresh-min context-thresh-max context-thresh-step)))
;    (range seed-thresh-min seed-thresh-max seed-thresh-step)))
