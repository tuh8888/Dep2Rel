(ns edu.ucdenver.ccp.nlp.readers
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [org.clojurenlp.core :as corenlp]
            [word2vec]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [clojure.string :as str])
  (:import (edu.ucdenver.ccp.knowtator.model KnowtatorModel)
           (edu.ucdenver.ccp.knowtator.model.object TextSource ConceptAnnotation Span GraphSpace AnnotationNode Quantifier)
           (edu.ucdenver.ccp.knowtator.io.conll ConllUtil)))

(defn biocreative-read-abstracts
  [^KnowtatorModel annotations f]
  (let [lines (->> (io/reader f)
                   (line-seq)
                   (map #(s/split % #"\t")))]
    (map
      (fn [[id title abstract]]
        (log/debug "Article" id title)
        (let [article-f (io/file (.getArticlesLocation annotations) (str id ".txt"))]
          (spit article-f (str title "\n" abstract))
          (let [text-sources (.getTextSources annotations)
                text-source (TextSource. annotations
                                         (io/file (.getAnnotationsLocation annotations)
                                                  (str id ".xml"))
                                         (.getName article-f))]
            (.add text-sources
                  text-source)
            text-source)))
      lines)))

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
                 src-ent (str doc "-" (second (s/split source #":")))
                 src-id (str "node_" src-ent)
                 source (AnnotationNode. src-id
                                         (.get (.get (.getConceptAnnotations text-source)
                                                     src-ent))
                                         0
                                         0
                                         graph-space)
                 dest-ent (str doc "-" (second (s/split target #":")))
                 dest-id (str "node_" dest-ent)
                 dest (AnnotationNode. dest-id
                                       (.get (.get (.getConceptAnnotations text-source)
                                                   dest-ent))
                                       0
                                       0
                                       graph-space)
                 id (str doc "-" id)]
             (log/debug "Relation" id property src-ent dest-ent)
             (.removeModelListener annotations text-source)
             (.add text-source graph-space)
             (.addCellToGraph graph-space source)
             (.addCellToGraph graph-space dest)
             (.addTriple graph-space
                         source
                         dest
                         id
                         (.getDefaultProfile annotations)
                         property
                         (Quantifier/some)
                         ""
                         false
                         "")
             (.addModelListener annotations text-source)
             graph-space)))))

(defn biocreative-read-entities
  [^KnowtatorModel annotations f]
  (->> (io/reader f)
       (line-seq)
       (map #(s/split % #"\t"))
       (map
         (fn [[doc id concept start end _]]
           (let [id (str doc "-" id)
                 start (Integer/parseInt start)
                 end (Integer/parseInt end)
                 text-source ^TextSource (.get (.get (.getTextSources annotations) doc))
                 concept-annotation (ConceptAnnotation. text-source id concept (.getDefaultProfile annotations) nil nil)
                 span (Span. concept-annotation nil start end)]
             (log/debug "Entity" id concept)
             (.removeModelListener annotations text-source)
             (.add ^ConceptAnnotation concept-annotation span)
             (.add (.getConceptAnnotations text-source) concept-annotation)
             (.addModelListener annotations text-source)
             concept-annotation)))))

(defn read-biocreative-files
  "Read biocreative files to model. Caches them by saving the model.
  Only reads in files if the cache is empty."
  [dir pat v]
  (let [sent-dir (->> "sentences"
                      (format pat)
                      (io/file dir))]
    (.setLoading (k/model v) true)

    (when (empty? (rest (file-seq (io/file dir "Articles"))))
      (log/info "Reading BioCreative abstracts")
      (->> "abstracts"
           (format (str pat ".tsv"))
           (io/file dir)
           (biocreative-read-abstracts (k/model v))
           (doall)))
    (when (empty? (rest (file-seq sent-dir)))
      (log/info "Writing abstract sentences to" (str sent-dir))
      (doseq [[id content] (sentenize (k/model v))]
        (let [content (->> content
                           (map :text)
                           (interpose "\n")
                           (apply str))
              sentence-f (io/file sent-dir (str id ".txt"))]
          (spit sentence-f content)))
      (println "Run dependency parser, then continue")
      (read-line))
    (when (empty? (rest (file-seq (io/file dir "Annotations"))))
      (log/info "Reading BioCreative entities")
      (->> "entities"
           (format (str pat ".tsv"))
           (io/file dir)
           (biocreative-read-entities (k/model v))
           (doall))
      (log/info "Reading BioCreative relations")
      (->> "relations"
           (format (str pat ".tsv"))
           (io/file dir)
           (biocreative-read-relations (k/model v))
           (doall)))
    (when (empty? (rest (file-seq (io/file dir "Structures"))))
      (log/info "Reading BioCreative structures")
      (let [conll-util (ConllUtil.)]
        (doseq [f (->> sent-dir
                       (file-seq)
                       (filter #(str/ends-with? % ".conll")))]
          (.readToStructureAnnotations conll-util (k/model v) f))))

    (.setLoading (k/model v) false)

    (log/info "Saving Knowtator model")
    (.save ^KnowtatorModel (k/model v))))