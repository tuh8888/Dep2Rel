(ns scripts.biocreative.readers
  (:require [clojure.string :as s]))

(defn read-abstracts
  [f]
  (->> (line-seq f)
       (map #(s/split % "\t"))
       (map (fn [line] [(first line) (apply str (interpose "\n" (rest line)))]))
       (into {})))

(defn read-relations
  [f]
  (->> (line-seq f)
       (map #(s/split % "\t"))
       (map
         (fn [[doc id has-relevant-relation? property source target]]
           {:doc           doc
            :id            id
            :has-relation? (= has-relevant-relation? "Y")
            :property      property
            :source        (second (s/split source ":"))
            :target        (second (s/split target ":"))}))
       (vec)))

(defn read-entities
  [f reference]
  (->> (line-seq f)
       (map #(s/split % "\t"))
       (map
         (fn [[doc id concept start end spanned-text]]
           (let [start (Integer/parseInt start)
                 end (Integer/parseInt end)]
             {:doc     doc
              :id      id
              :concept concept
              :start   start
              :end     end
              :tok     (subs (doc reference) start end)})))
       (vec)))


