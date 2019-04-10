(ns edu.ucdenver.ccp.nlp.readers
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.conll :as conll]
            [org.clojurenlp.core :as corenlp]
            [word2vec :as word2vec]
            [clojure.string :as str])
  (:import (java.io File)
           (edu.ucdenver.ccp.knowtator.model KnowtatorModel)
           (edu.ucdenver.ccp.knowtator.model.object TextSource ConceptAnnotation Span GraphSpace AnnotationNode Quantifier RelationAnnotation)))

(defn biocreative-read-abstracts
  [^KnowtatorModel annotations f]
  (let [lines (->> (io/reader f)
                   (line-seq)
                   (map #(s/split % #"\t")))]
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
      lines)))

(defn sentenize
  [^KnowtatorModel annotations]
  (into {}
        (map
          #(vector (.getId %) (corenlp/sentenize (.getContent %)))
          (.getTextSources annotations))))

(defn biocreative-read-relations
  [^KnowtatorModel annotations f]
  (->> (io/reader f)
       (line-seq)
       (map #(s/split % #"\t"))
       (map
         (fn [[doc id _ property source target]]
           (let [text-source ^TextSource (.get (.get (.getTextSources annotations) doc))
                 graph-space (GraphSpace. text-source nil)
                 source (second (s/split source #":"))

                 source (AnnotationNode. (str source "Node")
                                         (.get (.get (.getConceptAnnotations text-source)
                                                     source))
                                         0
                                         0
                                         graph-space)
                 target (second (s/split target #":"))
                 target (AnnotationNode. (str target "Node")
                                         (.get (.get (.getConceptAnnotations text-source)
                                                     target))
                                         0
                                         0
                                         graph-space)]
             (.addCellToGraph graph-space source)
             (.addCellToGraph graph-space target)
             (.addTriple graph-space
                         source
                         target
                         id
                         (.getDefaultProfile annotations)
                         nil
                         (Quantifier/some)
                         ""
                         false
                         "")
             (.setValue ^RelationAnnotation (first (filter #(= (.getId %) id) (.getRelationAnnotations graph-space)))
                        property))))))

(defn biocreative-read-entities
  [^KnowtatorModel annotations f]
  (->> (io/reader f)
       (line-seq)
       (map #(s/split % #"\t"))
       (map
         (fn [[doc id concept start end _]]
           (let [start (Integer/parseInt start)
                 end (Integer/parseInt end)
                 text-source ^TextSource (.get (.get (.getTextSources annotations) doc))
                 concept-annotation (ConceptAnnotation. text-source id nil (.getDefaultProfile annotations) concept nil)
                 span (Span. concept-annotation nil start end)]
             (.add ^ConceptAnnotation concept-annotation span)
             (.add (.getConceptAnnotations text-source) concept-annotation))))))

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


(defn conll-with-embeddings
  [k reference f]
  (mapv
    (fn [{lemma k :as tok}]
      (assign-embedding tok
                        (str/lower-case lemma)
                        word2vec/word-embedding))
    (try
      (conll/read-conll reference true f)
      (catch Throwable e
        (println f)
        (throw e)))))

(defn read-dependency
  [word2vec-db articles references dependency-dir & {:keys [ext tok-key] :or {tok-key :LEMMA}}]
  (word2vec/with-word2vec word2vec-db
    (zipmap articles
            (->> articles
                 (map
                   #(str % "." ext))
                 (map
                   #(io/file dependency-dir %))
                 (pmap
                   (partial conll-with-embeddings tok-key)
                   references)))))


