(defproject relation-extraction "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url  "http://www.eclipse.org/legal/epl-v10.html"}
  :main ^:skip-aot edu.ucdenver.ccp.nlp.relation-extraction
  ;;:local-repo "C:/Users/pielkelh/.m2/repository"
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [org.clojure/data.json "0.2.6"]
                 [com.google.cloud/google-cloud-bigquery "1.56.0"]
                 [uncomplicate/neanderthal "0.21.0"]
                 [org.slf4j/slf4j-simple "1.7.25"]
                 [com.taoensso/nippy "2.14.0"]
                 [com.climate/claypoole "1.1.4"]
                 [com.taoensso/timbre "4.10.0"]
                 ;;[net.sourceforge.owlapi/org.semanticweb.hermit "1.4.3.517"]
                 [edu.ucdenver.ccp/knowtator "2.1.6"]
                 [org.clojure/math.combinatorics "0.1.4"]
                 [spicerack "0.1.6"]]
  :profiles {:uberjar {:aot :all}
             :cluster {:jvm-opts ["-Xmx90g"]}})
