(ns edu.ucdenver.ccp.nlp.evaluation
  (:require [edu.ucdenver.ccp.nlp.relation-extraction :refer :all]
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
  [sent]
  (apply str (interpose " " (map #(get % :FORM) sent))))

(defn pprint-context
  [context]
  (pprint-sent
    (loop [tok (or (some #(and (= -1 (:HEAD %))
                               %) context)
                   (first context))
           path [tok]
           rest-context (disj context tok)]
      (if (seq rest-context)
        (let [next-tok (some #(and (= (dec (:ID tok)) (:HEAD %))
                                   %)
                             rest-context)
              next-tok (if next-tok next-tok (first rest-context))]
          (recur next-tok (conj path tok) (disj rest-context next-tok)))
        path))))

(defn format-matches
  [matches]
  (map (fn [match]
         (let [doc (:doc match)
               sent (:sent match)
               context (->> (:context match)
                            (pprint-sent))
               entities (sort-by :concept (:entities match))
               [e1-concept e2-concept] (->> entities
                                            (map :concept)
                                            (map #(.getShortForm (.getIRI %))))
               [e1-ann e2-ann] (map :id (map :ann entities))
               [e1-tok e2-tok] (map :tok entities)
               [e1-tok e2-tok] (map :FORM [e1-tok e2-tok])
               source-sent (pprint-sent sent)
               seed (map :concept (get-in match [:seed :entities]))]
           {:doc        doc
            ;;:e1-ann        e1-ann
            :context    context
            :e1-concept e1-concept
            :e1-tok     e1-tok
            ;;:e2-ann        e2-ann
            :e2-concept e2-concept
            :e2-tok     e2-tok
            :seed       (apply str seed)
            :sentence   (str "\"" source-sent "\"")
            }))
       matches))

(defn to-csv
  [f matches]
  (let [formatted (format-matches matches)
        cols [:doc :e1-concept :e1-tok :e2-concept :e2-tok :seed :sentence]
        csv-form (str (apply str (interpose "," cols)) "\n" (apply str (map #(str (apply str (interpose "," ((apply juxt cols) %))) "\n") formatted)))]
    (spit f csv-form)))

(defn matched-triples
  [match annotations property]
  (let [triples (k/triples-for-property annotations property)]
    (filter (fn [triple]
              (let [source (->> triple
                                (:source)
                                (bean))
                    source-ann (->> source
                                    (:conceptAnnotation)
                                    (bean)
                                    (:id))
                    target (->> triple
                                (:target)
                                (bean))
                    target-ann (->> target
                                    (:conceptAnnotation)
                                    (bean)
                                    (:id))
                    triple-anns #{source-ann target-ann}]
                (= triple-anns (set (map :id (map :ann (:entities match)))))))
            triples)))

(defn parameter-walk
  [annotations property seeds sentences & [{:keys [seed-thresh-min seed-thresh-max seed-thresh-step
                                                   cluster-thresh-min cluster-thresh-max cluster-thresh-step
                                                   context-thresh-min context-thresh-max context-thresh-step
                                                   min-support-min min-support-max min-support-step]
                                            :or   {seed-thresh-min     0.5
                                                   seed-thresh-max     0.99
                                                   seed-thresh-step    0.1

                                                   cluster-thresh-min  0.5
                                                   cluster-thresh-max  0.99
                                                   cluster-thresh-step 0.1

                                                   context-thresh-min  0.5
                                                   context-thresh-max  0.99
                                                   context-thresh-step 0.1

                                                   min-support-min     0.01
                                                   min-support-max     0.05
                                                   min-support-step    0.01}}]]
  (mapcat
    (fn [seed-thresh]
      (mapcat
        (fn [context-thresh]
          (mapcat
            (fn [cluster-thresh]
              (mapcat
                (fn [min-support]
                  (let [params {:seed             (first seeds)
                                :seed-thresh      seed-thresh
                                :context-thresh   context-thresh
                                :seed-match-fn    #(and (concepts-match? %1 %2)
                                                        (< seed-thresh (context-vector-cosine-sim %1 %2)))
                                :context-match-fn #(< context-thresh (context-vector-cosine-sim %1 %2))
                                :cluster-merge-fn add-to-pattern
                                :cluster-match-fn #(let [score (context-vector-cosine-sim %1 %2)]
                                                     (and (< (or %3 cluster-thresh) score)
                                                          score))
                                :min-support      min-support}]
                    (->> (cluster-bootstrap-extract-relations seeds sentences params)
                         (map #(merge % params))
                         (map #(let [t (matched-triples % annotations property)]
                                 (assoc % :num-matches (count t) :triples t))))))
                (range min-support-min min-support-max min-support-step)))
            (range cluster-thresh-min cluster-thresh-max cluster-thresh-step)))
        (range context-thresh-min context-thresh-max context-thresh-step)))
    (range seed-thresh-min seed-thresh-max seed-thresh-step)))
