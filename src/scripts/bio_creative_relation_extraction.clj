(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [clojure.string :as s]))

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

(def abstracts-f (io/file training-dir "chemprot_training_abstracts.tsv"))
(def abstracts (rdr/biocreative-read-abstracts abstracts-f))


(def entities-f (io/file training-dir "chemprot_training_entities.tsv"))
(def entities (rdr/biocreative-read-entities entities-f abstracts))

(def relations-f (io/file training-dir "chemprot_training_relations.tsv"))
(def relations (rdr/biocreative-read-relations relations-f))