(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [clojure.string :as s]
            [edu.ucdenver.ccp.knowtator-clj :as k])
  (:import (edu.ucdenver.ccp.knowtator.model KnowtatorModel)))

(def home-dir
  (io/file "/" "media" "tuh8888" "Seagate Expansion Drive" "data"))

(def biocreative-dir
  (io/file home-dir "BioCreative" "BCVI-2017" "ChemProt_Corpus"))

(def training-dir
  (io/file biocreative-dir "chemprot_training"))
(def word-vector-dir
  (io/file home-dir "WordVectors"))

(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def annotations (KnowtatorModel. training-dir nil))
(.load annotations)

(def references-dir (io/file training-dir "Articles"))
(def articles
  (rdr/article-names-in-dir references-dir "txt"))

(def references (rdr/read-references articles references-dir))

(def dependency-dir (io/file training-dir "chemprot_training_sentences"))
(def dependency (rdr/read-dependency word2vec-db articles references dependency-dir :ext "conll" :tok-key :FORM))


(comment
  (def sentences (rdr/read-sentences annotations dependency articles)))

(comment
  (def abstracts-f (io/file training-dir "chemprot_training_abstracts.tsv"))
  (rdr/biocreative-read-abstracts annotations abstracts-f)

  (def entities-f (io/file training-dir "chemprot_training_entities.tsv"))
  (rdr/biocreative-read-entities annotations entities-f)

  (def relations-f (io/file training-dir "chemprot_training_relations.tsv"))
  (rdr/biocreative-read-relations annotations relations-f))

(comment
  (def sentences (rdr/sentenize annotations))
  (first sentences)

  (def sentences-dir dependency-dir)
  (doall
    (map
      (fn [[k v]]
        (let [sentence-f (io/file sentences-dir (str (name k) ".txt"))
              content (apply str (interpose "\n" (map :text v)))]
          (spit sentence-f content)))
      sentences)))