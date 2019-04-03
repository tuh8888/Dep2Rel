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
  (let [lines (->> (io/reader f)
                   (line-seq)
                   (map #(s/split % #"\t")))]
    (zipmap (->> lines
                 (map first)
                 (map keyword))
            (->> lines
                 (map
                   (fn [[id title abstract]]
                     {:id       id
                      :title    title
                      :abstract abstract
                      :full     (str title "\n" abstract)}))
                 (map #(assoc % :sentences (s/split (:full %) #"[.\n]")))))))

(defn biocreative-read-relations
  [f]
  (->> (io/reader f)
       (line-seq)
       (map #(s/split % #"\t"))
       (map
         (fn [[doc id has-relevant-relation? property source target]]
           {:doc           doc
            :id            id
            :has-relation? (= has-relevant-relation? "Y")
            :property      property
            :source        (second (s/split source #":"))
            :target        (second (s/split target #":"))}))))

(defn biocreative-read-entities
  [f reference]
  (->> (io/reader f)
       (line-seq)
       (map #(s/split % #"\t"))
       (map
         (fn [[doc id concept start end spanned-text]]
           (let [start (Integer/parseInt start)
                 end (Integer/parseInt end)
                 doc (keyword doc)]
             {:doc     doc
              :id      id
              :concept concept
              :start   start
              :end     end
              :tok     (subs (get-in reference [doc :full]) start end)})))))

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


