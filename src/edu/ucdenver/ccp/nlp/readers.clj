(ns edu.ucdenver.ccp.nlp.readers
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [org.clojurenlp.core :as corenlp]
            [word2vec]
            [taoensso.timbre :as log])
  (:import (edu.ucdenver.ccp.knowtator.model KnowtatorModel)
           (edu.ucdenver.ccp.knowtator.model.object TextSource ConceptAnnotation Span GraphSpace AnnotationNode Quantifier)))

(defn biocreative-read-abstracts
  [^KnowtatorModel annotations f]
  (let [lines (->> (io/reader f)
                   (line-seq)
                   (map #(s/split % #"\t")))]
    (doall
      (map
        (fn [[id title abstract]]
          (let [article-f (io/file (.getArticlesLocation annotations) (str id ".txt"))]
            (spit article-f (str title "\n" abstract))
            (let [text-sources (.getTextSources annotations)
                  text-source (TextSource. annotations
                                           (io/file (.getAnnotationsLocation annotations)
                                                    (str id ".xml"))
                                           (.getName article-f))]
              (.add text-sources
                    text-source))))
        lines))
    (log/info "Done")))

(defn sentenize
  [^KnowtatorModel annotations]
  (let [text-sources (.getTextSources annotations)]
    (zipmap (map #(.getId %) text-sources)
            (->> text-sources
                 (map #(.getContent %))
                 (map #(corenlp/sentenize %))))))

(defn biocreative-read-relations
  [^KnowtatorModel annotations f]
  (->> (io/reader f)
       (line-seq)
       (map #(s/split % #"\t"))
       (map
         (fn [[doc id _ property source target]]
           (let [text-source ^TextSource (.get (.get (.getTextSources annotations) doc))
                 graph-space (GraphSpace. text-source nil)
                 source (str doc "-" (second (s/split source #":")))

                 source (AnnotationNode. (str "node_" source)
                                         (.get (.get (.getConceptAnnotations text-source)
                                                     source))
                                         0
                                         0
                                         graph-space)
                 target (str doc "-" (second (s/split target #":")))
                 target (AnnotationNode. (str "node_" target)
                                         (.get (.get (.getConceptAnnotations text-source)
                                                     target))
                                         0
                                         0
                                         graph-space)]
             (.removeModelListener annotations text-source)
             (.add text-source graph-space)
             (.addCellToGraph graph-space source)
             (.addCellToGraph graph-space target)
             (.addTriple graph-space
                         source
                         target
                         (str doc "-" id)
                         (.getDefaultProfile annotations)
                         property
                         (Quantifier/some)
                         ""
                         false
                         "")
             (log/info property)
             (.addModelListener annotations text-source))))))

(defn biocreative-read-entities
  [^KnowtatorModel annotations f]
  (doall
    (->> (io/reader f)
         (line-seq)
         (map #(s/split % #"\t"))
         (map
           (fn [[doc id concept start end _]]
             (log/debug doc)
             (let [start (Integer/parseInt start)
                   end (Integer/parseInt end)
                   text-source ^TextSource (.get (.get (.getTextSources annotations) doc))
                   concept-annotation (ConceptAnnotation. text-source (str doc "-" id) nil (.getDefaultProfile annotations) concept nil)
                   span (Span. concept-annotation nil start end)]
               (.removeModelListener annotations text-source)
               (.add ^ConceptAnnotation concept-annotation span)
               (.add (.getConceptAnnotations text-source) concept-annotation)
               (.addModelListener annotations text-source))))))
  (log/info "Done"))