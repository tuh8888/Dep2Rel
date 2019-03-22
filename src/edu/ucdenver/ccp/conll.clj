(ns edu.ucdenver.ccp.conll
  (:require [clojure.string :as s]
            [taoensso.timbre :refer [warn]]
            [util :refer [parse-int]]))

(def conllu-names [:ID :FORM :LEMMA :UPOS :XPOS :FEATS :HEAD :DEPREL :DEPS :MISC])
(def conll-names [:ID :FORM :LEMMA :POS :FEAT :HEAD :DEPREL])

(defn walk-dep
  ([sent tok]
   (try (let [path (walk-dep sent tok [])]
          (assert ((complement zero?) (count path)))
          path)
        (catch StackOverflowError _
          (warn sent tok)
          (throw (StackOverflowError. (str "While walking dependency for " tok))))))
  ([sent tok path]
   (let [tok-idx (get tok :HEAD)
         path (conj path tok)
         tok (get sent tok-idx)]
     (if (= tok-idx -1)
       path
       (walk-dep sent tok path)))))

(defn form-to-string
  [tok]
  (-> (get tok :FORM "")
      (s/replace #"\'\'" "\"")
      (s/replace #"\`\`" "\"")))

(defn find-span
  [tok form start end reference]
  (let [span [start end]
        check (partial subs reference)]
    (try
      (or (when (= form (apply check span)) span)
          (when (zero? start) (throw (Throwable. (str "Can't match token token " tok start end form " to reference"))))
          (let [span (map dec span)] (when (= form (apply check span)) span))
          (let [span (map inc span)] (when (= form (apply check span)) span))
          (let [span (map (partial - 2) span)] (when (= form (apply check span)) span))
          (let [span (map (partial + 2) span)] (when (= form (apply check span)) span))
          (do
            (warn "Can't match token token" tok start end form " to reference")
            span)
          #_(throw (Throwable. (str "Can't match token token " [(.getName f) sent tok start end form] " to reference"))))
      (catch StringIndexOutOfBoundsException _ (map dec span)))))

(defn match-toks-to-reference
  [reference c]
  (first (reduce
           (fn [[c start] [tok-idx tok]]
             (let [form (form-to-string tok)
                   end (+ start (count form))
                   [start end] (find-span tok form start end reference)
                   c (update c tok-idx assoc :START start :END end)]
               [c (inc end)]))
           [c 0]
           (map-indexed vector c))))

(defn read-conll
  [reference conllu? f]
  (->>
    (slurp f)
    (s/split-lines)
    (map #(s/split % #"\t"))
    (map #(interleave (if conllu? conllu-names conll-names) %))
    (mapv #(apply assoc {} %))
    (match-toks-to-reference reference)
    (map #(assoc % :DOC (.getName f)))
    (map #(update % :HEAD (fn [x] (when-let [x (parse-int x)] (dec x)))))
    (map #(update % :ID parse-int))
    (reduce #(if (and (string? (:ID %2)) (empty? (:ID %2)))
               (conj (conj %1 []) [])
               (conj (rest %1) (conj (first %1) %2)))
            [[]])))