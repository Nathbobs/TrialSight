# DESIGN.md — TrialSight Architecture

## System Overview
```
ClinicalTrials.gov API
        ↓
   [BRONZE LAYER]
   Raw XML/JSON → Parquet
        ↓
   [SILVER LAYER]
   Cleaned, filtered, typed Parquet
        ↓
   [GOLD LAYER]
   Star schema, quality checked, analytics-ready
        ↓
      ┌─────────────────┐
      │  Supabase       │  ← structured queries, dashboard metrics
      │  PostgreSQL     │
      └─────────────────┘
        ↓
   [EMBEDDING LAYER]
   HuggingFace all-MiniLM-L6-v2
        ↓
      ┌─────────────────┐
      │  Qdrant Cloud   │  ← semantic search for RAG
      │  Vector Store   │
      └─────────────────┘
        ↓
   [RAG LAYER]
   Groq Llama 3.3 70B
        ↓
   Streamlit App
   ├── Dashboard Tab
   └── RAG Chat Tab
```

## Medallion Layer Definitions

### Bronze
- Source: ClinicalTrials.gov API v2
- Action: Fetch raw JSON, minimal filtering, save as Parquet
- Output fields: nctId, briefTitle, officialTitle, overallStatus,
  phase, startDate, completionDate, condition, intervention,
  sponsorName, briefSummary, eligibilityCriteria
- Record count: ~500-1000 trials (configurable by condition keyword)

### Silver
- Source: Bronze Parquet
- Action: Drop nulls on critical fields, normalize dates,
  standardize phase labels, filter incomplete records
- Quality checks: null check on nctId/briefTitle/condition,
  date range validation (2000-2026), phase label normalization
- Output: cleaned Parquet with record count logged

### Gold
- Source: Silver Parquet
- Action: Build star schema, add surrogate keys,
  flag single-sponsor trials as supply risk equivalent
- Tables: dim_condition, dim_sponsor, dim_phase, fact_trials
- Output: Gold Parquet ready for Supabase load

## Supabase Schema
```sql
CREATE TABLE dim_condition (
    condition_id SERIAL PRIMARY KEY,
    condition_name TEXT UNIQUE NOT NULL
);

CREATE TABLE dim_sponsor (
    sponsor_id SERIAL PRIMARY KEY,
    sponsor_name TEXT UNIQUE NOT NULL,
    sponsor_type TEXT
);

CREATE TABLE dim_phase (
    phase_id SERIAL PRIMARY KEY,
    phase_label TEXT UNIQUE NOT NULL
);

CREATE TABLE fact_trials (
    trial_id SERIAL PRIMARY KEY,
    nct_id TEXT UNIQUE NOT NULL,
    brief_title TEXT NOT NULL,
    official_title TEXT,
    overall_status TEXT,
    phase_id INTEGER REFERENCES dim_phase(phase_id),
    condition_id INTEGER REFERENCES dim_condition(condition_id),
    sponsor_id INTEGER REFERENCES dim_sponsor(sponsor_id),
    start_date DATE,
    completion_date DATE,
    brief_summary TEXT,
    eligibility_criteria TEXT,
    is_single_sponsor BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Qdrant Collection Structure
- Collection name: `clinical_trials`
- Vector size: 384 (all-MiniLM-L6-v2 output dimension)
- Distance metric: Cosine
- Payload fields: nct_id, brief_title, condition, phase, sponsor, brief_summary

## Architecture Decisions
- **Qdrant over ChromaDB**: cloud persistence, no local storage,
  open source and self-hostable (important for clinical data privacy)
- **Groq over OpenAI**: free tier, no billing risk on public demo,
  Llama 3.3 70B is sufficiently capable for clinical Q&A
- **Medallion over flat ETL**: mirrors production data engineering
  standards, each layer is independently testable and auditable
- **HuggingFace all-MiniLM-L6-v2**: lightweight (90MB), no auth
  required, proven for semantic similarity on domain text
- **Supabase over raw PostgreSQL**: managed, free tier,
  Seoul region available, built-in REST API
