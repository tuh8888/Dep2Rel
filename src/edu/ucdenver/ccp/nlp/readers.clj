(ns edu.ucdenver.ccp.nlp.readers
  (:require [clojure.string :as s]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.word2vec :as word2vec]
            [taoensso.timbre :as t]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.conll :as conll])
  (:import (java.io File)))

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

(defn article-names-in-dir
  [dir ext]
  (->> (file-seq dir)
       (filter #(.isFile ^File %))
       (map #(.getName %))
       (filter #(s/ends-with? % (str "." ext)))
       (map #(s/replace % (re-pattern (str "\\." ext)) ""))))

(defn read-references
  [articles references-dir]
  (->> articles
       (pmap
         #(->> (str % ".txt")
               (io/file references-dir)
               (slurp)))
       (into [])))

(defn assign-embedding
  [m v embedding-fn]
  (assoc m :VEC (embedding-fn v)))

(defn toks-with-embeddings
  [toks k embedding-fn]
  (mapv
    (fn [{lemma k :as tok}]
      (assign-embedding tok lemma embedding-fn))
    toks))

(defn conll-with-embeddings
  [reference f]
  (mapv
    #(toks-with-embeddings % :LEMMA word2vec/word-embedding)
    (conll/read-conll reference true f)))

(defn read-dependency
  [word2vec-db articles references dependency-dir]
  (word2vec/with-word2vec word2vec-db
    (zipmap articles
            (->> articles
                 (map
                   #(str % ".tree.conllu"))
                 (map
                   #(io/file dependency-dir %))
                 (pmap
                   conll-with-embeddings
                   references)))))



(defn read-sentences
  [annotations dependency articles]
  (doall (vec (sentence/make-sentences annotations dependency articles))))


