(ns edu.ucdenver.ccp.nlp.seeds
  (:require [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [edu.ucdenver.ccp.nlp.re-model :as re-model]
            [taoensso.timbre :as log]))

(defn pattern-seed-match-scores
  [{:keys [seeds patterns factory] :as params}]
  (let [seeds           (vec seeds)
        pattern-vectors (map #(re-model/context-vector % params) patterns)
        sample-vectors  (map #(re-model/context-vector % params) seeds)]
    (log/info "Calculating seed matches")
    (->> pattern-vectors
         (linear-algebra/mdot factory sample-vectors)
         (map vector patterns)
         (pmap (fn [[pattern scores]]
                 (assoc pattern :seed-match-scores scores))))))

(defn pattern-seed-match-ratio
  [{:keys [seeds match-thresh] :as model}]
  (let [patterns (pattern-seed-match-scores model)
        seeds-m  (->> seeds
                      (map-indexed vector)
                      (group-by #(:predicted (second %)))
                      (util/map-kv #(map first %)))]
    (->> patterns
         (map (fn [p]
                (let [predicted-positive (->> p
                                              :seed-match-scores
                                              (filter #(< match-thresh %))
                                              (count))]
                  (if (= 0 predicted-positive)
                    (assoc p :recall 0
                             :precision 0
                             :f1 0)
                    (let [tp              (->> p
                                               :predicted
                                               (get seeds-m)
                                               (select-keys (:seed-match-scores p))
                                               (vals)
                                               (filter #(< match-thresh %))
                                               (count))
                          actual-positive (count (get seeds-m (:predicted p)))
                          fp              (- predicted-positive tp)
                          fn              (- actual-positive tp)
                          precision       (/ tp predicted-positive)
                          recall          (/ tp actual-positive)]
                      (assoc p
                        :tp tp
                        :fp fp
                        :fn fn
                        :fn (- (count seeds) tp fp fn)
                        :recall recall
                        :precision precision
                        :f1 (if (or (= 0 precision) (= 0 recall))
                              0
                              (/ (* 2 precision recall)
                                 (+ precision recall))))))))))))


(defn all-seed-patterns
  [model {:keys [cluster-thresh match-thresh]}]
  (for [cluster-thresh cluster-thresh
        :let [ps (re/pattern-update (assoc model :cluster-thresh cluster-thresh))]
        match-thresh   match-thresh]
    (-> model
        (assoc :patterns ps
               :match-thresh match-thresh)
        (pattern-seed-match-ratio))))

(defn seed-patterns-with-selectivity
  [seed-patterns properties
   {:keys [min-f1 min-recall min-precision]
    :or   {min-f1        0
           min-recall    0
           min-precision 0}}]
  (->> seed-patterns
       (map #(into [] (comp
                        (filter (fn [{:keys [f1]}] (<= min-f1 f1)))
                        (filter (fn [{:keys [recall]}] (<= min-recall recall)))
                        (filter (fn [{:keys [precision]}] (<= min-precision precision)))) %))
       (filter (fn [ps]
                 (->> ps
                      (group-by :predicted)
                      (util/map-kv count)
                      (vals)
                      (filter #(<= 0 %))
                      (count)
                      (= (count properties)))))))


