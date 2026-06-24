# TRUSTWORTHINESS.md — TrialSight Quality and Reliability

## Core Principle
Every output from TrialSight must be traceable, honest,
and verifiable. No hallucinated trial data. No silent failures.
No confident answers without evidence.

## RAG Trustworthiness Rules

### Citation Required
- Every RAG response must reference at least one NCT ID
- If no NCT ID can be cited, the response must say
  "I don't have enough information to answer this confidently"
- Source trials must be shown in an expandable section
  below every response

### Confidence Transparency
- Qdrant relevance scores must be shown alongside citations
- Score below 0.45: trigger low-confidence fallback response
- Score above 0.45: proceed with normal response
- Score above 0.75: mark as high-confidence response

### Hallucination Prevention
- System prompt explicitly instructs model to use
  only provided context
- Temperature set to 0.3 to reduce creative deviation
- Context window capped at 3000 tokens to prevent
  model from filling gaps with training data

## Pipeline Trustworthiness Rules

### Bronze Layer Checks
- Record count must be > 0 — abort if API returns empty
- nctId must be present on every record — drop if missing
- Log raw record count before any filtering

### Silver Layer Checks
- Null check: nctId, briefTitle, condition — drop if null
- Date range check: startDate must be between 2000-2026
- Phase label normalization: map all variants to
  standard labels (Phase 1, Phase 2, Phase 3, Phase 4, N/A)
- Log: records dropped at each check with reason

### Gold Layer Checks
- Referential integrity: all foreign keys must resolve
- No duplicate nctId in fact_trials
- is_single_sponsor flag must be explicitly set
  (True or False, never NULL)

### Supabase Load Checks
- Verify row count after load matches Gold record count
- Log any insertion failures with nctId for debugging

### Qdrant Embedding Checks
- Verify upsert count matches Gold record count
- Log any failed upserts with nctId
- Confirm collection exists before upserting

## Monitoring
- Every pipeline run logs: timestamp, stage,
  input count, output count, records dropped, reason
- Every RAG query logs: timestamp, query,
  retrieval count, top score, response length
