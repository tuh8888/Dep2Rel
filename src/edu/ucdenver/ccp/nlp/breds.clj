(ns edu.ucdenver.ccp.nlp.breds
  (:require [clojure.test :refer :all]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [edu.ucdenver.nlp.util :refer :all]
            [edu.ucdenver.nlp.word2vec :refer :all]
            [uncomplicate.neanderthal.core :refer [axpy dot nrm2]]))

(defrecord Sentence [text start end toks e1 e2 pos-tags rels])
(defrecord Relationship [text bef, e1, tween, e2, aft])
(defrecord Entity [e-type, text, start, end])
(defrecord Tuple [text e1 e2 bef_vec tween_vec aft_vec])

(def reverb nil)                                            ;TODO
(def stop-words #{})                                        ;TODO
(def pos-to-exclude #{"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"}) ;TODO

(defn entry->entity
  [entry text start label]
  (let [e-start ((keyword (str label "_start")) entry)
        e-end   ((keyword (str label "_end")) entry)]
    (->Entity ((keyword (str label "_type")) entry)
              (subs text (- e-start start 1) (- e-end start 1))
              e-start
              e-end)))

(defn ->relations
  [text e1 e2 pos-tags]
  ;TODO
  (let [e1_tok_idx      (tok-idx pos-tags e1)
        e1_last_tok_idx (last-tok-idx pos-tags e1)
        e2_tok_idx      (tok-idx pos-tags e2)
        e2_last_tok_idx (last-tok-idx pos-tags e2)
        bef             (subvec pos-tags 0 e1_tok_idx)
        tween           (subvec pos-tags e1_last_tok_idx e2_tok_idx)
        aft             (subvec pos-tags e2_last_tok_idx)]
    [(->Relationship text bef e1 tween e2 aft)]))

(defn extract-reverb-patterns-tagged-ptb [tagged-text]
  ;TODO
  tagged-text)

(defn ->word-vectors
  [before between after]
  (let [between-vector (->> between
                            (extract-reverb-patterns-tagged-ptb)
                            (filter #(not (or (contains? stop-words %)
                                              (contains? pos-to-exclude %))))
                            (pattern2vector-sum))
        before_vector  (pattern2vector-sum before)
        after_vector   (pattern2vector-sum after)]
    [before_vector between-vector after_vector]))

(defn relation->tuple
  [rel]
  (let [[bef_vec bet_vec aft_vec] (->word-vectors (:bef rel) (:tween rel) (:aft rel))]
    (->Tuple (:text rel) (:e1 rel) (:e2 rel) bef_vec bet_vec aft_vec)))

(defn entry->sentence
  [entry tok-fn pos-fn]
  (let [text     (:sentence_text entry)
        start    (:sentence_start entry)
        end      (:sentence_end entry)
        toks     (tok-fn text start)
        pos-tags (pos-fn toks)
        [e1 e2] (sort-by :start [(entry->entity entry text start "concept1")
                                 (entry->entity entry text start "concept2")])
        rels     (->relations text e1 e2 pos-tags)]
    (->Sentence text start end toks e1 e2 pos-tags rels)))

(defn parse-query-result
  [query-result max-tokens min-tokens tok-fn pos-fn]
  (->> (json/read query-result
                  :key-fn keyword
                  :value-fn #(let [int-result-fields #{:sentence_start :sentence_end :concept1_end :concept1_start :concept2_start :concept2_end}]
                               (if (contains? int-result-fields %1)
                                 (read-string %2)
                                 (string/replace %2 "'" ""))))
       (map #(entry->sentence % tok-fn pos-fn))
       (filter #(let [dist (token-distance (:toks %) (:e1 %) (:e2 %))]
                  (and (>= max-tokens dist)
                       (<= min-tokens dist))))))

(deftest test-parse-query
  (let [sample-result (io/reader (io/resource "sample_query_result.json"))
        query-result  (parse-query-result sample-result 10 0 tokenize pos-tag)]
    (info query-result)))

(defrecord Pattern [positive negative unknown confidence tuples bet_vec_set bet_word_set])
(def alpha 1)
(def beta 1)
(def gamma 1)
(def thresh-sim 0)
(def w-unk 1)
(def w-neg 1)
(def min-pattern-support 0)
(def num-iterations 100)
(def processed-tuples [])

(defn match-seeds
  [pos-seeds]
  (filter (fn [t]
            (some (fn [s]
                    (and (= (:e1 t) (:e1 s))
                         (= (:e2 t) (:e2 s))))
                  pos-seeds))
          processed-tuples))

(defn sim-3-contexts
  [t1 t2]
  (reduce + (->> [:bef-vec :tween-vec :aft-vec]
                 (map #([(% t1) (% t2)]))
                 (map #(map nrm2 %))
                 (map (fn [[v1 v2]] (dot v1 v2)))
                 (map #(* %1 %2) [alpha beta gamma]))))

(defn similarity-all
  [tuple pattern]
  (let [[good bad max-sim] (reduce (fn [[good bad max-sim] t]
                                     (let [score   (sim-3-contexts tuple t)
                                           max-sim (if (> score max-sim) score max-sim)]
                                       (if (>= score thresh-sim)
                                         [(inc good) bad max-sim]
                                         [good (inc bad) max-sim])))
                                   [0 0 0]
                                   (:tuples pattern))]
    (if (>= good bad)
      [true max-sim]
      [false 0.0])))

(defn find-max-sim
  [tuple patterns]
  (reduce (fn [[max-sim max-sim-cluster-idx] [i extraction-pattern]]
            (let [[accept score] (similarity-all tuple extraction-pattern)]
              (if (and accept (> score max-sim))
                [score i]
                [max-sim max-sim-cluster-idx])))
          [0 0]
          (map-indexed vector patterns)))

(defn cluster-tuples->patterns
  [matched-tuples patterns]
  (as-> patterns patterns
        (if (empty? patterns)
          (conj patterns (->Pattern 0 0 0 0 #{(first matched-tuples)} #{} #{}))
          patterns)
        (reduce (fn [patterns t]
                  (let [[max-sim idx] (find-max-sim t patterns)]
                    (if (< max-sim thresh-sim)
                      (conj patterns (->Pattern 0 0 0 0 #{t} #{} #{}))
                      (update-in patterns [idx :tuples] conj t))))
                patterns
                matched-tuples)
        (filter #(> (:tuples %) min-pattern-support) patterns)))

(def pos-seed-tuples [])
(def neg-seed-tuples [])

(defn update-selectivity [pattern t]
  (let [matched-e1 (some #(= (:e1 t) (:e1 %)) pos-seed-tuples)
        matched-e2 (some #(= (:e2 t) (:e2 %)) pos-seed-tuples)
        pattern    (cond (and matched-e1 matched-e2) (update pattern :positive inc)
                         matched-e1 (update pattern :negative inc)
                         :else pattern)
        matched-e1 (or matched-e1 (some #(= (:e1 t) (:e1 %)) neg-seed-tuples))
        matched-e2 (or matched-e2 (some #(= (:e2 t) (:e2 %)) neg-seed-tuples))
        pattern    (if (and matched-e1 matched-e2)
                     pattern
                     (update pattern :unknown inc))]
    pattern))

(defn find-best-pattern
  [tuple patterns]
  (reduce (fn [[best-pattern sim-best] extraction-pattern]
            (let [[accept score] (similarity-all tuple extraction-pattern)
                  extraction-pattern (if accept
                                       (update-selectivity extraction-pattern tuple)
                                       extraction-pattern)]
              (if (> score sim-best)
                [extraction-pattern score]
                [best-pattern sim-best])))
          [nil 0]
          patterns))

(defn match-patterns
  [candidate-tuples patterns tuple]
  (let [[best-pattern sim-best] (find-best-pattern tuple patterns)]
    (if (>= sim-best thresh-sim)
      (update candidate-tuples tuple #(conj % [best-pattern sim-best]))
      candidate-tuples)))

(defn update-pattern-confidence
  [p]
  (cond-> p
          (> (:positive p) 0) (assoc :confidence (/ (:positive p)
                                                    (+ (:positive p)
                                                       (* (:unknown p) w-unk)
                                                       (* (:negative p) w-neg))))
          (= (:positive p) 0) (assoc :confidence 0)))

(def instance-confidence 0)
(defrecord Seed [e1 e2])

(defn update-seeds
  [pos-seeds candidate-tuples]
  (reduce (fn [pos-seeds t]
            (cond-> pos-seeds
                    (>= (:confidence t) instance-confidence) (conj pos-seeds (->Seed (:e1 t) (:e2 t)))))
          pos-seeds
          candidate-tuples))

(defn update-candidate-tuple-confidence
  [patterns t t-patterns]
  (update t :confidence
          (fn [confidence-old]
            (- 1
               (reduce (fn [confidence p]
                         (* confidence
                            (- 1 (* (:confidence (aget p 0))
                                    (aget p 1)))))
                       1
                       t-patterns)))))

(defn init-bootstrap
  [pos-seeds]
  (reduce (fn [[candidate-tuples patterns pos-seeds] _]
            (let [matched-tuples (match-seeds pos-seeds)]
              (if (empty? matched-tuples)
                [candidate-tuples patterns pos-seeds]
                (let [patterns         (cluster-tuples->patterns matched-tuples patterns)
                      candidate-tuples (reduce #(match-patterns %1 patterns %2) candidate-tuples processed-tuples)
                      patterns         (map update-pattern-confidence patterns)
                      candidate-tuples (map (fn [[t t-patterns]]
                                              (update-candidate-tuple-confidence patterns t t-patterns))
                                            candidate-tuples)
                      pos-seeds        (update-seeds pos-seeds candidate-tuples)]
                  [candidate-tuples patterns pos-seeds]))))
          [nil '() pos-seeds]
          (range num-iterations)))


(deftest test-bootstrap
  (init-bootstrap []))










