(ns edu.ucdenver.ccp.knowtator-clj
  (:import (edu.ucdenver.ccp.knowtator.view KnowtatorView)
           (javax.swing JFrame)
           (org.semanticweb.HermiT ReasonerFactory)
           (org.semanticweb.owlapi.reasoner OWLReasoner)
           (org.semanticweb.owlapi.model OWLObjectProperty)
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
    (KnowtatorModel. f owl-workspace)))

(defn simple-model
  [v]
  (->> (:textSources (bean (model v)))
       (map (fn [t]
              (let [t (-> (bean t)
                          (update :conceptAnnotations #(vec (map (fn [a]
                                                                   (assoc (bean a) :spans (vec (map bean a))))
                                                                 %)))
                          (update :graphSpaces #(vec (map (fn [g]
                                                            (let [g (bean g)]
                                                              (-> g
                                                                  (assoc :triples (vec (map bean (:relationAnnotations g))))
                                                                  (assoc :vertices (vec (map bean (:annotationNodes g)))))))
                                                          %))))]
                [(:id t) t])))
       (into {})))

(defn selected-annotation
  [v]
  (bean (.get (:selectedConceptAnnotation (bean (model v))))))

(defn reasoner
  [v]
  (let [ont-man (:owlOntologyManager (bean (model v)))
        ont (first (:ontologies (bean ont-man)))]
    (.createReasoner (ReasonerFactory.) ont)))

(defn get-owl-descendants
  [^OWLReasoner r owlClass]
  (-> r
      (.getSubClasses owlClass false)
      (.getFlattened)))

(defn triples-for-property
  [annotations property]
  (->> (simple-model annotations)
       (vals)
       (mapcat :graphSpaces)
       (mapcat :triples)
       (filter #(= (-> %
                       ^OWLObjectProperty (:property)
                       (.getIRI)
                       (.getShortForm))
                   property))))