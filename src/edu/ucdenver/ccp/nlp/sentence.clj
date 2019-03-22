(ns edu.ucdenver.ccp.nlp.sentence
  (:require [uncomplicate.neanderthal.core :refer [xpy]]
            [util :refer [unit-vec-sum]]
            [edu.ucdenver.ccp.conll :as conll]
            [clojure.math.combinatorics :as combo]
            [taoensso.timbre :refer [info warn debug]]
            [clojure.set :refer [difference union intersection]]
            [edu.ucdenver.ccp.knowtator-clj :as k]))

(defrecord Sentence [concepts entities doc sent context context-vector])

(defrecord Entity [concept ann sent tok dep])

(defn overlap
  [tok concept-start concept-end]
  (or (<= (:START tok) concept-start concept-end (:END tok))
      (<= concept-start (:START tok) (:END tok) concept-end)))

(defn annotations->entities
  [concept-annotations dep-doc]
  (map
    (fn [ann]
      (let [concept-start (get-in ann [:spans 0 :start])
            concept-end (get-in ann [:spans 0 :end])
            concept (get ann :owlClass)]
        (some
          (fn [sent]
            (some
              (fn [tok]
                (when (overlap tok concept-start concept-end)
                  (->Entity concept ann sent tok (conll/walk-dep sent tok))))
              sent))
          dep-doc)))
    concept-annotations))

(defn make-sentences
  [annotations dependency articles]
  (let [mem-get-descs (memoize #(k/get-owl-descendants (k/reasoner annotations) %))]
    (mapcat
      (fn [doc-id]
        (->>
          (annotations->entities (get-in (k/simple-model annotations) [doc-id :conceptAnnotations])
                                 (get dependency doc-id))
          (group-by :sent)
          (mapcat
            (fn [[sent sent-entities]]
              (map
                (fn [[i [e1 e2]]]
                  (warn i)
                  (let [entities #{e1 e2}
                        concepts (->> [e1 e2]
                                      (map :concept)
                                      (map #(-> (mem-get-descs %)
                                                (set)
                                                (conj %)))
                                      (set))
                        deps (map (comp set :dep) [e1 e2])
                        context (conj (difference (apply union deps)
                                                  (let [same (apply intersection deps)]
                                                    (if (= (count same) 1)
                                                      (remove
                                                        #(= -1 (get % :HEAD))
                                                        same)
                                                      same)))
                                      (get e1 :tok)
                                      (get e2 :tok))
                        context-vector (when-let [vectors (->> context
                                                               (keep :VEC)
                                                               (seq))]
                                         (apply unit-vec-sum vectors))]
                    (->Sentence concepts entities doc-id sent context context-vector)))
                (map-indexed vector (combo/combinations sent-entities 2)))))))
      articles)))
