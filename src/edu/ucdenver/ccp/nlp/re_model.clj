(ns edu.ucdenver.ccp.nlp.re-model
  (:require [clojure.string :as str]
            [ubergraph.alg :as uber-alg]
            [ubergraph.core :as uber]
            [graph :as graph]
            [linear-algebra :as linear-algebra]
            [util :as util]
            [word2vec :as word2vec]
            [taoensso.timbre :as log]
            [clojure.math.combinatorics :as combo]
            [edu.ucdenver.ccp.knowtator-clj :as k])
  (:import (clojure.lang PersistentArrayMap ExceptionInfo)
           (java.io File)))

(def NONE "NONE")

(def MODEL-KEYs #{:concept-annotations
                  :concept-graphs
                  :structure-annotations
                  :structure-graphs
                  :sentences})

(defn word-embedding-catch
  [word]
  (try
    (word2vec/word-embedding word)

    (catch ExceptionInfo e
      (if (= :word-not-found (-> e (ex-data) :cause))
        (log/debug (-> e (ex-message)))
        (throw e)))))

(defprotocol ContextVector
  (context-vector [self model]))

(extend-type PersistentArrayMap
  ContextVector
  (context-vector [self _]
    (->> self
         :spans
         vals
         (keep :text)
         (mapcat #(str/split % #" |-"))
         (map str/lower-case)
         (map word-embedding-catch)
         (doall)
         (apply linear-algebra/vec-sum))))

(defrecord Sentence [concepts entities context sent-id full-sentence-text context-text]
  ContextVector
  (context-vector [self {:keys [structure-annotations concept-annotations factory] :as model}]
    (or (:VEC self)
        (let [context-toks (->> self
                                :context
                                (map #(get structure-annotations %)))
              v            (->> self
                                :entities
                                (map (fn [e] (get concept-annotations e)))
                                (lazy-cat context-toks)
                                (map #(context-vector % model))
                                (doall)
                                (apply linear-algebra/vec-sum))]
          (when v (linear-algebra/unit-vec factory v))))))


(defrecord Pattern [support]
  ContextVector
  (context-vector [self {:keys [factory] :as model}]
    (or (:VEC self)
        (->> self
             :support
             (map #(context-vector % model))
             (apply linear-algebra/vec-sum)
             (linear-algebra/unit-vec factory)))))


(defn add-to-pattern
  [_ p s]
  (->Pattern (conj (set (:support p)) s)))

(defn pprint-toks-text
  [{:keys [structure-annotations]} toks]
  (->> toks
       (map #(get structure-annotations %))
       (map (comp first vals :spans))
       (sort-by :start)
       (map :text)
       (interpose " ")
       (apply str)))

(defn pprint-sent-text
  [{:keys [structure-graphs] :as model} sent-id]
  (->> sent-id
       (get structure-graphs)
       :node-map
       (keys)
       (pprint-toks-text model)))

(defn predicted-positive
  [samples]
  (remove #(= (:predicted %) NONE) samples))

(defn actual-positive
  ([samples]
   (remove #(= NONE (:property %)) samples))
  ([property samples]
   (filter #(= property (:property %)) samples)))

(defn actual-negative
  "The samples labelled negative"
  [samples]
  (filter #(= NONE (:property %)) samples))

(defn predicted-positive
  ([samples]
   (remove #(= NONE (:predicted %)) samples))
  ([property matches]
   (filter #(= property (:predicted %)) matches)))

(defn predicted-negative
  [samples]
  (filter #(= (:predicted %) NONE) samples))

(defn min-start
  [a]
  (->> a
       :spans
       (vals)
       (map :start)
       (reduce min)))

(defn max-end
  [a]
  (->> a
       :spans
       (vals)
       (map :end)
       (reduce max)))

(defn ann-toks
  [model {:keys [doc] :as ann}]
  (let [concept-start (min-start ann)
        concept-end   (max-end ann)
        tok-ids       (into [] (comp (filter #(= (:doc %) doc))
                                     (filter #(let [tok-start (min-start %)
                                                    tok-end   (max-end %)]
                                                (or (<= tok-start concept-start tok-end)
                                                    (<= tok-start concept-end tok-end))
                                                #_(<= concept-start tok-start tok-end concept-end))))
                            (vals (:structure-annotations model)))]
    (when-not (seq tok-ids) (log/warn "No token found for" ann)
                            (throw (ex-info (str "No token found for " ann)
                                            {:type :tok-assignment, :cause :tok-not-found})))
    tok-ids))

(defn tok-sent-id
  [{:keys [structure-graphs]} {:keys [id]}]
  (some
    (fn [[sent-id sent]]
      (when (get-in sent [:node-map id])
        sent-id))
    structure-graphs))

(defn overlap
  "Finds overlap between two annotations"
  [a1 a2]
  (let [min-a1-start (min-start a1)
        min-a2-start (min-start a2)
        max-a1-end   (max-end a1)
        max-a2-end   (max-end a2)]
    (<= min-a1-start min-a2-start max-a2-end max-a1-end)))

(defn make-context-path
  [{:keys [structure-annotations concept-annotations]} undirected-sent sent-id tok1 tok2]
  (let [sent-anns (filter #(= (:sent-id %) sent-id) (vals concept-annotations))]
    (->> (uber-alg/shortest-path undirected-sent tok1 tok2)
         (uber-alg/nodes-in-path)
         (map #(get structure-annotations %))
         (reduce
           (fn [[toks ann :as x] tok]
             (if (and ann (overlap ann tok))
               x
               (let [ann (first (filter #(overlap % tok) sent-anns))]
                 [(conj toks (if ann
                               (assoc tok :spans (:spans ann))
                               tok))
                  ann])))
           nil)
         (first)
         (remove #(empty? (str/replace (->> %
                                            :spans
                                            (vals)
                                            (map :text)
                                            (str/join))
                                       #"\)|\(" "")))
         (map :id))))

(defn make-sentence
  "Make a sentence using the sentence graph and entities"
  [model undirected-sent sent-id [ann1 ann2 :as anns]]
  ;; TODO: Remove context toks that are part of the entities
  (let [concepts           (->> anns
                                (map :concept)
                                (map #(conj #{} %))
                                (set))
        entities           (->> anns
                                (map :id)
                                (set))
        context            (apply min-key count (for [tok1 (:toks ann1)
                                                      tok2 (:toks ann2)]
                                                  (make-context-path model undirected-sent sent-id tok1 tok2)))
        full-sentence-text (pprint-sent-text model sent-id)
        context-text       (pprint-toks-text model context)]
    (->Sentence concepts entities context sent-id full-sentence-text context-text)))

(defn combination-sentences
  [model undirected-sent sent-id sent-annotations]
  (->> (combo/combinations sent-annotations 2)
       (map #(make-sentence model undirected-sent sent-id %))))

(defn concept-annotations->sentences
  [{:keys [concept-annotations structure-graphs] :as model}]
  (let [undirected-sents (util/map-kv graph/undirected-graph structure-graphs)]
    (->> concept-annotations
         (vals)
         (group-by :sent-id)
         (pmap (fn [[sent-id sent-annotations]]
                 (combination-sentences model (get undirected-sents sent-id) sent-id sent-annotations)))
         (apply concat))))

(defn assign-embedding
  [model tok]
  (assoc tok :VEC (context-vector tok model)))

(defn assign-sent-id
  [model tok]
  (assoc tok :sent-id (tok-sent-id model tok)))

(defn assign-toks
  [model ann]
  (let [toks (ann-toks model ann)
        {:keys [sent-id]} (first toks)]
    (assoc ann :toks (map :id toks)
               :sent-id sent-id)))

(defn sent-property
  [{:keys [concept-graphs]} [id1 id2]]
  (some
    (fn [g]
      (when-let [e (or (uber/find-edge g id2 id1) (uber/find-edge g id1 id2))]
        (:value (uber/attrs g e))))
    (vals concept-graphs)))

(defn assign-property
  "Assign the associated property with the sentence"
  [model s]
  (assoc s :property (or (sent-property model (vec (:entities s)))
                         NONE)))

(defn sentences-with-ann
  [sentences id]
  (filter
    (fn [{:keys [entities]}]
      (some #(= id %) entities))
    sentences))

(defn make-model
  [v word2vec-db factory concept-annotations-file structure-annotations-file]
  (word2vec/with-word2vec word2vec-db
    (log/info "Making model")
    (let [model               (k/simple-model v)
          model               (->> (let [structure-annotations (when (and structure-annotations-file (.exists ^File structure-annotations-file))
                                                                 (read-string (slurp structure-annotations-file)))]
                                     (if (seq structure-annotations)
                                       structure-annotations
                                       (let [structure-annotations (do
                                                                     (log/info "Making structure annotations")
                                                                     (util/pmap-kv (fn [s]
                                                                                     (assign-sent-id model s))
                                                                                   (:structure-annotations model)))]
                                         (when structure-annotations-file (spit structure-annotations-file (pr-str structure-annotations)))
                                         structure-annotations)))
                                   (assoc model :structure-annotations))
          concept-annotations (let [concept-annotations (when (and concept-annotations-file (.exists ^File concept-annotations-file))
                                                          (read-string (slurp concept-annotations-file)))]
                                (if (seq concept-annotations)
                                  concept-annotations
                                  (let [concept-annotations (do
                                                              (log/info "Making concept annotations")
                                                              (util/pmap-kv (fn [s]
                                                                              (assign-toks model s))
                                                                            (:concept-annotations model)))]
                                    (when concept-annotations-file (spit concept-annotations-file (pr-str concept-annotations)))
                                    concept-annotations)))]

      (assoc model :factory factory
                   :word2vec-db word2vec-db
                   :concept-annotations concept-annotations))))

(defn make-sentences
  [{:keys [word2vec-db] :as model} sentences-file]
  (let [sentences (when (.exists ^File sentences-file)
                    (read-string (slurp sentences-file)))]
    (if (seq sentences)
      sentences
      (word2vec/with-word2vec word2vec-db
        (log/info "Making sentences")
        (let [sentences (->> model
                             (concept-annotations->sentences)
                             (map #(assign-embedding model %))
                             (pmap #(assign-property model %))
                             (doall))]
          (spit sentences-file (pr-str sentences))
          sentences)))))

(defn frac-seeds
  [property {:keys [sentences seed-frac rng]}]
  (let [
        pot (->> sentences
                 (filter #(= (:property %) property))
                 (filter #(= (:property %) property))
                 (util/deterministic-shuffle rng))]
    (-> pot
        (count)
        (* seed-frac)
        (take pot)
        (set))))

(defn train-test
  [{:keys [seeds seed-frac rng]} {:keys [word2vec-db] :as testing-model}]
  (word2vec/with-word2vec word2vec-db
    (assoc testing-model :seed-frac seed-frac
                         :rng rng
                         :seeds seeds
                         :all-samples (->> testing-model
                                           :sentences
                                           (map #(assign-embedding testing-model %))
                                           (doall)))))

(defn split-train-test
  "Splits model into train and test sets"
  [{:keys [sentences properties word2vec-db negative-cap seed-frac] :as model}]
  (word2vec/with-word2vec word2vec-db
    (let [seeds (->> (disj properties NONE)
                     (map (fn [property] (frac-seeds property model)))
                     (apply clojure.set/union))
          seeds (->> sentences
                     (actual-negative)
                     (count)
                     (/ negative-cap)
                     (min seed-frac)
                     (assoc model :seed-frac)
                     (frac-seeds NONE)
                     (clojure.set/union seeds))]
      (-> model
          (assoc :all-samples (remove seeds sentences)
                 :seeds (->> seeds
                             (map #(assoc % :predicted (:property %)
                                            :confidence 1))
                             (set)))))))

