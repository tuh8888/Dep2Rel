(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [edu.ucdenver.ccp.nlp.relation-extraction :refer :all])
  (:import (org.semanticweb.owlapi.model HasIRI)))

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
  (let [structure-annotations (->> model
                                   vals
                                   (mapcat :structure-annotations)
                                   (into {}))]
    (->> sent
         (map #(get structure-annotations %))
         (map (comp first vals :spans))
         (sort-by :start)
         (map :text)
         (interpose " ")
         (apply str))))

(defn format-matches
  [model matches]
  (map (fn [match]
         (let [[e1 _ :as entities] (:entities match)

               doc (:doc e1)
               sent (->> (:sent e1)
                         :node-map
                         keys
                         (pprint-sent model))
               context (->> match
                            :context
                            (pprint-sent model))
               [e1-concept e2-concept] (->> entities
                                            (sort-by :concept)
                                            (map :concept)
                                            (map #(.getShortForm (.getIRI ^HasIRI %))))
               [e1-tok e2-tok] (map (comp :text first vals :spans :tok) entities)
               seed (->> (get-in match [:seed :entities])
                         (map :concept)
                         (map #(.getShortForm (.getIRI ^HasIRI %)))
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
  (set (map :id (:entities match))))

(defn edge->triple
  [model edge]
  (let [concept-annotations (:concept-annotations model)
        source (->> edge
                    :src
                    (get concept-annotations)
                    :id)
        target (->> edge
                    :dest
                    (get concept-annotations)
                    :id)]
    #{source target}))

(defn predicted-false
  [& [{:keys [predicted-true all]}]]
  (clojure.set/difference all predicted-true))

(defn actual-false
  [& [{:keys [actual-true all]}]]
  (clojure.set/difference all actual-true))

(defn tp
  [& [{:keys [predicted-true actual-true]}]]
  (clojure.set/intersection predicted-true actual-true))
(defn tn
  [& [params]]
  (clojure.set/intersection (predicted-false params)
                            (actual-false params)))
(defn fp
  [& [{:keys [predicted-true] :as params}]]
  (clojure.set/intersection predicted-true
                            (actual-false params)))
(defn fn
  [& [{:keys [actual-true] :as params}]]
  (clojure.set/intersection (predicted-false params)
                            actual-true))

(defn precision
  [& [params]]
  (float (/ (count (tp params)) (+ (count (tp params)) (count (fp params))))))
(defn recall
  [& [params]]
  (float (/ (count (tp params)) (+ (count (tp params)) (count (fn params))))))
(defn f1
  [& [params]]
  (float (/ (* 2 (precision params) (recall params))
            (+ (precision params) (recall params)))))

(defn calc-metrics
  [& [params]]
  {:tp        (count (tp params))
   :tn        (count (tn params))
   :fp        (count (fp params))
   :fn        (count (fn params))
   :precision (precision params)
   :recall    (recall params)
   :f1        (params f1)})
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
