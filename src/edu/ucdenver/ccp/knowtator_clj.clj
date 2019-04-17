(ns edu.ucdenver.ccp.knowtator-clj
  (:require [ubergraph.core :as uber])
  (:import (edu.ucdenver.ccp.knowtator.view KnowtatorView)
           (javax.swing JFrame)
           (org.semanticweb.HermiT ReasonerFactory)
           (org.semanticweb.owlapi.reasoner OWLReasoner)
           (edu.ucdenver.ccp.knowtator.model KnowtatorModel)))

(defn display
  [v]
  (doto (JFrame.)
    (.setContentPane v)
    (.pack)
    (.setVisible true)))

(defn view
  [f]
  (let [v (KnowtatorView.)]
    (.loadProject v f nil)
    v))

(defn model
  ([v]
   (if (instance? KnowtatorModel v)
     v
     (.get (.getModel v))))
  ([f owl-workspace]
   (let [k (KnowtatorModel. f owl-workspace)]
     (.load k)
     k)))

(defn simple-span
  [span]
  (-> span
      (assoc :text (:spannedText span))
      (select-keys [:start :end :text])))

(defn simple-concept
  [annotation]
  (or (:owlClass annotation) (:annotationType annotation)))

(defn simple-collection
  [collection simplify-fn & {:keys [key-fn] :or {key-fn :id}}]
  (let [collection (map bean collection)]
    (zipmap (map key-fn collection)
            (map simplify-fn collection))))

(defn simple-concept-annotation
  [annotation]
  {:id      (:id annotation)
   :spans   (simple-collection (:collection annotation)
                               simple-span)
   :concept (simple-concept annotation)})

(defn simple-annotation-node
  [annotation-node]
  {:concept (-> annotation-node
                :conceptAnnotation
                simple-concept-annotation)
   :graph   (-> annotation-node
                :graph-space
                bean
                :id)})


(defn simple-triple
  [triple]
  (let [doc-id (-> triple :textSource (bean) :id)
        source (->> triple :source (bean) :conceptAnnotation (bean) :id (str doc-id "-"))
        target (->> triple :target (bean) :conceptAnnotation (bean) :id (str doc-id "-"))
        value {:value (or (:property triple) (:value triple))}]
    [source target value]))

(defn simple-graph-space
  [graph-space]
  (apply uber/multidigraph
         (vals
           (simple-collection (:relationAnnotations graph-space)
                              simple-triple))))

(defn simple-model
  [v]
  (let [text-sources (map bean (:textSources (bean (model v))))
        text-sources (zipmap (map :id text-sources)
                             text-sources)]
    {:structure-annotations (apply merge
                                   (map
                                     (fn [[id doc]]
                                       (util/map-kv
                                         #(assoc % :doc id)
                                         (simple-collection (:structureAnnotations doc)
                                                            simple-concept-annotation)))
                                     text-sources))
     :concept-annotations   (apply merge
                                   (map
                                     (fn [[id doc]]
                                       (util/map-kv
                                         #(assoc % :doc id)
                                         (simple-collection (:conceptAnnotations doc)
                                                            simple-concept-annotation
                                                            :key-fn #(str id "-" (:id %)))))
                                     text-sources))
     :concept-graphs        (apply merge
                                   (map
                                     (fn [[id doc]]
                                       (util/map-kv
                                         #(assoc % :doc id)
                                         (simple-collection (:graphSpaces doc)
                                                            simple-graph-space
                                                            :key-fn #(str id "-" (:id %)))))
                                     text-sources))
     :structure-graphs      (apply merge
                                   (map
                                     (fn [[id doc]]
                                       (util/map-kv
                                         #(assoc % :doc id)
                                         (simple-collection (:structureGraphSpaces doc)
                                                            simple-graph-space
                                                            :key-fn #(str id "-" (:id %)))))
                                     text-sources))}))

(defn selected-annotation
  [v]
  (bean (.get (:selectedConceptAnnotation (bean (model v))))))

(defn reasoner
  [v]
  (let [ont-man (:owlOntologyManager (bean (model v)))
        ont (first (:ontologies (bean ont-man)))]
    (when ont
      (.createReasoner (ReasonerFactory.) ont))))

(defn get-owl-descendants
  [^OWLReasoner r owlClass]
  (-> r
      (.getSubClasses owlClass false)
      (.getFlattened)
      (set)))

(defn edges-for-property
  [model property]
  (->> model
       :concept-graphs
       vals
       (mapcat #(ubergraph.core/find-edges % {:value property}))
       (keep identity)))