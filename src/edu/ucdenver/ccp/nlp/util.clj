(ns edu.ucdenver.ccp.nlp.util
  (:require [clojure.test :refer :all]
            [clojure.string :as string]))

(defrecord Token [text, start, end])

(defn tokenize
  [text sent_start]
  (let [toks (string/split text #" ")]
    (loop [tokens []
           toks toks
           start 0]
      (if (empty? toks)
        tokens
        (let [tok (first toks)
              end (+ start (count tok))]
          (recur (conj tokens (->Token tok (+ start sent_start) (+ end sent_start)))
                 (rest toks)
                 (inc end)))))))

(defn tok-idx
  [tokens entity]
  (some (fn [[idx tok]] (when (>= (:end tok)
                                  (:start entity))
                          idx))
        (map-indexed vector tokens)))

(defn last-tok-idx
  [tokens entity]
  (some (fn [[idx tok]] (when (>= (:start tok)
                                  (:end entity))
                          idx))
        (map-indexed vector tokens)))

(defn token-distance
  [tokens entity1 entity2]
  (let [[first_e second_e] (sort-by :start [entity1 entity2])
        first_e_tok_idx (tok-idx tokens first_e)
        second_e_tok_idx (tok-idx tokens second_e)]
    (- second_e_tok_idx first_e_tok_idx)))

(deftest test_tokens
         (let [text "The quick brown fox jumps over the lazy dog"
               tokens (tokenize text 0)
               entity1 {:start 10 :end 15}
               entity2 {:start 26 :end 30}]
           (is (= (count tokens)) 9)
           (is (= 3 (token-distance tokens entity1 entity2)))))

(defn pos-tag
  [tokens]
  ;TODO
  tokens
  )
