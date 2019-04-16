(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [clojure.string :as s]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [taoensso.timbre :as log])
  (:import (edu.ucdenver.ccp.knowtator.model KnowtatorModel)
           (edu.ucdenver.ccp.knowtator.model.object GraphSpace TextSource ConceptAnnotation Span AnnotationNode Quantifier)))

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

(def annotations (k/view training-dir))

(def references-dir (io/file training-dir "Articles"))
(def articles
  (rdr/article-names-in-dir references-dir "txt"))

(def references (rdr/read-references articles references-dir))

(def dependency-dir (io/file training-dir "chemprot_training_sentences"))
(def dependency (rdr/read-dependency word2vec-db articles references dependency-dir :ext "conll" :tok-key :FORM))

(do
  (doall
    (map
     (fn [[doc-id doc]]
       (let [text-source ^TextSource (.get (.get (.getTextSources annotations) doc-id))]
         (.removeModelListener annotations text-source)
         (doall
           (map
            (fn [[sent-idx sent]]
              (let [graph-space (GraphSpace. text-source (str "Sentence " sent-idx))]
                (.add (.getStructureGraphSpaces text-source) graph-space)
                (let [nodes (mapv
                              (fn [tok]
                                (let [ann (ConceptAnnotation. text-source nil nil (.getDefaultProfile annotations) (:DEPREL tok) "")
                                      span (Span. ann nil (:START tok) (:END tok))
                                      ann-node (AnnotationNode. nil ann 20 20 graph-space)]
                                  (.add ann span)
                                  (.add (.getStructureAnnotations text-source) ann)
                                  (.addCellToGraph graph-space ann-node)
                                  [ann-node tok]))
                              sent)]
                  (doall
                    (map
                     (fn [[source tok]]
                       (let [target-idx (:HEAD tok)]
                         (cond (<= 0 target-idx) (let [target (first (get nodes target-idx))
                                                       property nil]
                                                   (.addTriple graph-space source target nil (.getDefaultProfile annotations)
                                                               property Quantifier/some "", false, ""))
                               (not (= "ROOT" (:DEPREL tok))) (throw (Throwable. "Excluding root"))
                               :else nil)))
                     nodes)))))
            (group-by :SENT doc)))
         (.addModelListener annotations text-source)))
     dependency))
  nil)

(def model (k/simple-model annotations))

(def structures-annotations-with-embeddings
  (zipmap (keys (:structure-annotations model))
          (word2vec/with-word2vec word2vec-db
            (doall
              (pmap sentence/assign-word-embedding
                    (vals (:structure-annotations model)))))))

(def concepts-with-toks
  (zipmap (keys (:concept-annotations model))
          (pmap
            #(let [tok-id (sentence/annotation-tok-id model %)
                   sent-id (sentence/tok-sent-id model tok-id)]
               (assoc % :tok tok-id
                        :sent sent-id))
            (vals (:concept-annotations model)))))

(def reasoner (k/reasoner annotations))

(def mem-descs
  (memoize
    (fn [c]
      (log/info c)
      (k/get-owl-descendants reasoner c))))

(def model (assoc model
             :concept-annotations concepts-with-toks
             :structure-annotations structures-annotations-with-embeddings))


(def sentences (->>
                 (sentence/concept-annotations->sentences model)
                 (map
                   #(update % :concepts
                            (fn [concepts]
                              (map
                                (fn [concept-set]
                                  (into concept-set (mem-descs (first concept-set))))
                                concepts))))))

(log/info "Num sentences:" (count sentences))

(comment
  (def sentences (rdr/read-sentences annotations dependency articles)))

(comment
  (def abstracts-f (io/file training-dir "chemprot_training_abstracts.tsv"))
  (rdr/biocreative-read-abstracts annotations abstracts-f)

  (def entities-f (io/file training-dir "chemprot_training_entities.tsv"))
  (rdr/biocreative-read-entities annotations entities-f)

  (def relations-f (io/file training-dir "chemprot_training_relations.tsv"))
  (rdr/biocreative-read-relations annotations relations-f))



;(comment
;  (def sentences (rdr/sentenize annotations))
;  (first sentences)
;
;  (def sentences-dir dependency-dir)
;  (doall
;    (map
;      (fn [[k v]]
;        (let [sentence-f (io/file sentences-dir (str (name k) ".txt"))
;              content (apply str (interpose "\n" (map :text v)))]
;          (spit sentence-f content)))
;      sentences)))