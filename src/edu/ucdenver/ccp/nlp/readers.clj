(ns edu.ucdenver.ccp.nlp.readers
  (:require [clojure.string :as s]
            [clojure.java.io :as io]
            [edu.ucdenver.ccp.conll :as conll]
            [org.clojurenlp.core :as corenlp]
            [word2vec]
            [taoensso.timbre :as log])
  (:import (java.io File)
           (edu.ucdenver.ccp.knowtator.model KnowtatorModel)
           (edu.ucdenver.ccp.knowtator.model.object TextSource ConceptAnnotation Span GraphSpace AnnotationNode Quantifier RelationAnnotation)))

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
           (log/info id)
           (let [text-source ^TextSource (.get (.get (.getTextSources annotations) doc))
                 graph-space (GraphSpace. text-source nil)
                 source (second (s/split source #":"))

                 source (AnnotationNode. (str "node_" source)
                                         (.get (.get (.getConceptAnnotations text-source)
                                                     source))
                                         0
                                         0
                                         graph-space)
                 target (second (s/split target #":"))
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
                         id
                         (.getDefaultProfile annotations)
                         nil
                         (Quantifier/some)
                         ""
                         false
                         "")
             (.setValue ^RelationAnnotation (first (filter #(= (.getId %) id) (.getRelationAnnotations graph-space)))
                        property)
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
                  concept-annotation (ConceptAnnotation. text-source id nil (.getDefaultProfile annotations) concept nil)
                  span (Span. concept-annotation nil start end)]
              (.removeModelListener annotations text-source)
              (.add ^ConceptAnnotation concept-annotation span)
              (.add (.getConceptAnnotations text-source) concept-annotation)
              (.addModelListener annotations text-source))))))
  (log/info "Done"))

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
                        (s/lower-case lemma)
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

(defn biocreative-read-dependency
  [annotations dir word2vec-db]
  (let [
        references-dir (io/file dir "Articles")
        articles
        (article-names-in-dir references-dir "txt")

        references (read-references articles references-dir)

        dependency-dir (io/file dir "chemprot_training_sentences")
        dependency (read-dependency word2vec-db articles references dependency-dir :ext "conll" :tok-key :FORM)]

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
      nil)))


