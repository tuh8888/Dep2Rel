(ns edu.ucdenver.ccp.biostacks
  (:import (edu.ucdenver.ccp GoogleCloudHelper)))

(defn run-query
  [query]
  (seq (GoogleCloudHelper/runQuery query)))