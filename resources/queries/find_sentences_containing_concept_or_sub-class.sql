-- Find sentences containing concept or sub-class
-- Author: Harrison Pielke-Lombardo
-- Email: harrison.pielke-lombardo@ucdenver.edu

WITH constants AS (
SELECT
  'http://purl.obolibrary.org/obo/GO_0008150' AS CONCEPT1  --biological process
)
SELECT
  covered_text AS sentence_text,
  span_start AS sentence_start,
  span_end AS sentence_end,
  concept1_type,
  concept1_ancestor,
  concept1_start,
  concept1_end
FROM biostacks_multitable.annotation
JOIN (
  SELECT
    concept1_table.encompassing_annotation_id,
    concept1_type,
    concept1_ancestor,
    concept1_start,
    concept1_end
  FROM (
      -- Find encompassing annotation id for sentences containing annotations with type
      SELECT
        encompassing_annotation_id,
        concept1_type,
        concept1_ancestor,
        concept1_start,
        concept1_end
      FROM biostacks_multitable.in_sentence
      JOIN (
        -- Find annotations with type
        SELECT
          annotation_id,
          annotation_type AS concept1_type,
          CONCEPT1 AS concept1_ancestor,
          span_start AS concept1_start,
          span_end AS concept1_end
        FROM biostacks_multitable.annotation, constants
        WHERE
          -- Any concept whose ancestor is CONCEPT1
          annotation_type = CONCEPT1
          OR annotation_type IN (SELECT concept_id FROM biostacks_multitable.concept_ancestor WHERE ancestor_id = CONCEPT1)
      ) concept1
      ON concept1.annotation_id = in_sentence.annotation_id
  ) concept1_table
) concepts_table
ON concepts_table.encompassing_annotation_id = annotation_id