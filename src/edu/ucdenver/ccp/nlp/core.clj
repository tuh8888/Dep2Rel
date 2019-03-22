(ns edu.ucdenver.ccp.nlp.core
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.word2vec :as word2vec]
            [taoensso.timbre :refer [info debug warn]]
            [edu.ucdenver.ccp.conll :as conll]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :refer :all]
            [clojure.math.combinatorics :as combo]
            [edu.ucdenver.ccp.knowtator-clj :as k]))



(defn read-references
  [articles references-dir]
  (->> articles
       (pmap
         #(->> (str % ".txt")
               (io/file references-dir)
               (slurp)))
       (into [])))

(defn read-dependency
  [word2vec-db articles references dependency-dir]
  (word2vec/with-word2vec word2vec-db
    (->> articles
         (pmap
           #(do
              (warn %2)
              [%2 (->>
                    (str %2 ".tree.conllu")
                    (io/file dependency-dir)
                    (conll/read-conll %1 true)
                    (map
                      (fn [toks]
                        (into [] (map
                                   (fn [tok]
                                     (assoc tok :VEC (word2vec/get-word-vector (:LEMMA tok))))
                                   toks))))
                    (into []))])
           references)
         (into {}))))

(defn read-sentences
  [annotations dependency articles]
  (doall (vec (sentence/make-sentences annotations dependency articles))))

#_(defn extract-all-relations
  [seed-thresh cluster-thresh min-support sources]
  (let [all-concepts (map
                       #(sentence/->Entity % nil nil nil nil)
                       (knowtator/all-concepts (:annotations sources)))]
    (info "All concepts" (count all-concepts))
    (->> (combo/combinations all-concepts 2)
         (pmap set)
         (pmap #(sentence/->Sentence % nil nil))
         (pmap
           #(map
              (fn [r]
                (assoc r :seed %))
              (extract-relations (list %) (:sentences sources)
                                 :seed-thresh seed-thresh
                                 :cluster-thresh cluster-thresh
                                 :min-support min-support
                                 :max-iter 100)))
         (mapcat identity))))
