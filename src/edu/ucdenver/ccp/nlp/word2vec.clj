(ns edu.ucdenver.ccp.nlp.word2vec
  (:require [clojure.java.io :as io]
            [clojure.string :as s]
            [uncomplicate.neanderthal.native :refer [dv]]
            [spicerack.core :as spice]
            [taoensso.timbre :refer [debug info warn]]
            [com.climate.claypoole :as cp]))

(def ^:dynamic *word-vectors*)

(defmacro with-word2vec
  [db-f & body]
  `(with-open [db# (spice/open-database ~db-f)]
     (binding [*word-vectors* (spice/open-hashmap db# "word-vectors")]
       ~@body)))

(defn string-seq->vec
  [s]
  (->> s
       (map read-string)
       (into [])))

(defn load-word2vec
  [f]
  (->> f
       (io/reader)
       (line-seq)
       (rest)
       (map #(s/split % #"\ "))
       (map #(vector (first %) (string-seq->vec (rest %))))))

(defn make-word2vec-database
  [model-f]
  (let [word-vectors (->> (load-word2vec model-f)
                          (map-indexed vector)
                          (partition-all 1000))]
    (cp/pdoseq (inc (cp/ncpus)) [vec-part word-vectors]
               (doseq [[i [k v]] vec-part]
                 (when (zero? (rem i 1000)) (info i))
                 (spice/assoc! @*word-vectors* k v)))))

(defn zero-vec []
  (dv (take (count (get *word-vectors* "the"))
            (repeat 0))))

(defn word-embedding
  [word]
  (if-let [vec (get *word-vectors* word)]
    (dv vec)
    (debug "Word vector not found" word) #_(throw (Throwable. (str "Word not found " word)))))

;(comment
;  (let [word2vec-dir (io/file  "/home" "harrison" "word-vectors") #_(io/file "E:" "data" "WordVectors")
;        model-f (io/file word2vec-dir "bio-word-vectors.vec")
;        db-f (.getAbsolutePath (io/file word2vec-dir "bio-word-vectors-clj.vec"))]
;    (with-word2vec db-f
;      (make-word2vec-database model-f))))
;(comment
;  (let [word-vector-dir (io/file "E:" "data" "WordVectors")
;        db-f (.getAbsolutePath (io/file word-vector-dir "bio-word-vectors-clj.vec"))]
;    (with-word2vec db-f
;      (get-word-vector "the"))))