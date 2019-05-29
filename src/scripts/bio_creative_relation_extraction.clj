(ns scripts.bio-creative-relation-extraction
  (:require [clojure.java.io :as io]
            [edu.ucdenver.ccp.nlp.readers :as rdr]
            [edu.ucdenver.ccp.knowtator-clj :as k]
            [edu.ucdenver.ccp.nlp.sentence :as sentence]
            [edu.ucdenver.ccp.nlp.relation-extraction :as re]
            [taoensso.timbre :as log]
            [edu.ucdenver.ccp.nlp.evaluation :as evaluation]))

(def home-dir (io/file "/" "home" "harrison"))

#_(def home-dir
  (io/file "/" "media" "harrison" "Seagate Expansion Drive" "data"))

(def biocreative-dir
  (io/file home-dir "BioCreative" "BCVI-2017" "ChemProt_Corpus"))

(def training-dir
  (io/file biocreative-dir "chemprot_training"))
(def word-vector-dir
  (io/file home-dir "WordVectors"))

(def word2vec-db
  (.getAbsolutePath
    (io/file word-vector-dir "bio-word-vectors-clj.vec")))

(def annotations (k/model training-dir nil))

(def abstracts-f (io/file training-dir "chemprot_training_abstracts.tsv"))
(rdr/biocreative-read-abstracts (k/model annotations) abstracts-f)

(def entities-f (io/file training-dir "chemprot_training_entities.tsv"))
(rdr/biocreative-read-entities (k/model annotations) entities-f)

(def relations-f (io/file training-dir "chemprot_training_relations.tsv"))
(rdr/biocreative-read-relations (k/model annotations) relations-f)
#_(.save (k/model annotations))

(comment
  ;; Read from conll files
  (rdr/biocreative-read-dependency annotations training-dir word2vec-db))


(def model1 (k/simple-model annotations))
(count (get model1 :structure-annotations))
(def structures-annotations-with-embeddings (word2vec/with-word2vec word2vec-db
                                              (sentence/structures-annotations-with-embeddings model1)))

(def concept-annotations-with-toks (sentence/concept-annotations-with-toks model1))

(def model (assoc model1
             :concept-annotations concept-annotations-with-toks
             :structure-annotations structures-annotations-with-embeddings))

(def sentences (sentence/concept-annotations->sentences model))
(log/info "Num sentences:" (count sentences))

(def property "INHIBITOR")

(def actual-true (set (map evaluation/edge->triple
                           (k/edges-for-property model property))))

(ubergraph.core/find-edges (first (vals (:concept-graphs model))) {:value property})
(def all-triples (set (map evaluation/sent->triple sentences)))

(log/info "Num actual true:" (count actual-true))

(first actual-true)
(first sentences)

(get-in model [:concept-annotations "17429625-T32"])
(get-in model [:structure-annotations "17429625-1291885"])
(map #(get-in model [:structure-annotations %]) (keys (:node-map (get-in model [:structure-graphs "17429625-Sentence 0"]))))
(sentence/tok-sent-id model "17429625-1291885")
(sentence/sentences-with-ann sentences "17429625-T32")
(count (get (group-by :sent (vals (:concept-annotations model))) nil))
(defn c-metrics
  [matches]
  (math/calc-metrics {:predicted-true (evaluation/predicted-true matches)
                      :actual-true    actual-true
                      :all            all-triples}))

(comment
  (def matches (let [seeds (clojure.set/union
                             (evaluation/make-seeds sentences
                               "17429625-T19" "17429625-T32")
                             #_(evaluation/make-seeds sentences
                               "CRAFT_aggregate_ontology_Instance_21365"
                               "CRAFT_aggregate_ontology_Instance_22495"))
                     seed-thresh 0.95
                     context-thresh 0.95
                     cluster-thresh 0.7
                     min-support 10
                     params {:seed             (first seeds)
                             :seed-thresh      seed-thresh
                             :context-thresh   context-thresh
                             :seed-match-fn    #(and (re/concepts-match? %1 %2)
                                                     (< seed-thresh (re/context-vector-cosine-sim %1 %2)))
                             :context-match-fn #(< context-thresh (re/context-vector-cosine-sim %1 %2))
                             :cluster-merge-fn re/add-to-pattern
                             :cluster-match-fn #(let [score (re/context-vector-cosine-sim %1 %2)]
                                                  (and (< (or %3 cluster-thresh) score)
                                                       score))
                             :min-support      min-support}
                     matches (->> (re/cluster-bootstrap-extract-relations seeds sentences params)
                                  (map #(merge % params)))]
                 (log/info "Metrics" (c-metrics matches))
                 matches)))





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