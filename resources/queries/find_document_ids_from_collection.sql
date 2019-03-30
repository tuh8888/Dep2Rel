-- Find documents by collection
-- Author: Harrison Pielke-Lombardo
-- Email: harrison.pielke-lombardo@ucdenver.edu
SELECT
  document_id
FROM
  biostacks_requested.document_collections
WHERE
  collection_name = 'HARRISON'
  