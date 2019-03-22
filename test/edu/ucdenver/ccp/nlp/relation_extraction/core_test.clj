(ns edu.ucdenver.nlp.relation_extraction.core-test
  (:require [clojure.test :refer :all]
            [edu.ucdenver.nlp.relation_extraction.relation-extraction :refer :all]
            [uncomplicate.neanderthal.native :refer [dv]]
            [taoensso.nippy :as nippy])
  (:import (edu.ucdenver.nlp.relation_extraction.relation-extraction Sentence)))

(def dim 5)
(def sentences (list (Sentence. "Harrison lives in Denver"
                                "Harrison"
                                "Denver"
                                (dv (repeat 1 dim))
                                (dv (repeat 1 dim))
                                (dv (repeat 1 dim))
                                0)
                     (Sentence. "Tiffany lives in Boulder"
                                 "Tiffany"
                                 "Boulder"
                                 (dv (repeat 1 dim))
                                 (dv (repeat 1 dim))
                                 (dv (repeat 1 dim))
                                 0)))
(def patterns (list (Sentence. "Harrison is in Denver"
                                "Harrison"
                                "Denver"
                                (dv (repeat 1 dim))
                                (dv (repeat 1 dim))
                                (dv (repeat 1 dim))
                                0)))
(def matches (mapcat #(find-matches % sentences) patterns))

;(defn load-word2vec
;  []
;  (let [file (io/file "C:/Users/pielkelh/Downloads/word-vecs.bin")]
;    (Word2VecModel/fromBinFile file)))
;
;(def WORD2VEC (load-word2vec))

