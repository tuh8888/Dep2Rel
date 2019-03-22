(ns edu.ucdenver.ccp.nlp.readers
  (:require [clojure.string :as s]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.word2vec :as word2vec]
            [taoensso.timbre :as t]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.conll :as conll]))

(defn biocreative-read-abstracts
  [f]
  (->> (line-seq f)
       (map #(s/split % "\t"))
       (map (fn [line] [(first line) (apply str (interpose "\n" (rest line)))]))
       (into {})))

(defn biocreative-read-relations
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

(defn biocreative-read-entities
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
                                    (t/warn %2)
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


