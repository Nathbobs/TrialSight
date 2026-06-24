# TrialSight — Portfolio Case Study

---

## 1. Project Overview

TrialSight is an end-to-end clinical trial data intelligence platform that ingests real study records from ClinicalTrials.gov, processes them through a production-grade ETL pipeline, and serves a Streamlit application with structured analytics and a retrieval-augmented generation (RAG) chat interface. The system was designed to demonstrate how modern data engineering and applied NLP can make clinical trial data more accessible and queryable — moving from raw API JSON to a semantically searchable, analyst-ready knowledge base.

---

## 2. Problem Statement

ClinicalTrials.gov is the world's largest clinical study registry, containing over 500,000 registered trials. Despite being publicly accessible, the data is difficult to work with at scale: it is deeply nested, inconsistently structured, uses non-standard phase and status codes, and offers no semantic search capability. A researcher asking "what hypertension trials are currently recruiting in Phase 3?" must either manually browse the registry or write custom API queries — there is no infrastructure to answer natural-language questions grounded in up-to-date trial records.

TrialSight closes that gap by building a complete data pipeline from raw API extraction through transformation, structured storage, vector embedding, and a chat interface that cites its sources by NCT ID on every response.

---

## 3. System Architecture

```
ClinicalTrials.gov API (v2)
            │
            ▼
    ┌───────────────┐
    │  BRONZE LAYER │  Raw JSON → Parquet  (pipeline/extract.py)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  SILVER LAYER │  Cleaning, validation, normalisation  (pipeline/transform.py)
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   GOLD LAYER  │  Star schema — fact + 3 dimension tables
    └───────┬───────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐  ┌──────────┐
│Supabase │  │  Qdrant  │  Two specialised stores for two query types
│PostgreSQL│  │  Cloud   │
└────┬────┘  └────┬─────┘
     │             │
     └──────┬──────┘
            ▼
    ┌───────────────┐
    │ Streamlit App │  Dashboard + floating RAG chat panel
    └───────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
Dashboard       RAG Chat
(SQL joins)   (Qdrant → Groq)
```

**Component roles:**

| Component | Role |
|-----------|------|
| `pipeline/extract.py` | Fetches studies from the ClinicalTrials.gov v2 REST API. Paginates with cursor tokens, deduplicates by NCT ID, persists as timestamped Parquet. |
| `pipeline/transform.py` | Runs Bronze → Silver (null drops, date validation, phase normalisation) then Silver → Gold (star schema construction with surrogate keys). |
| `pipeline/load.py` | Upserts Gold tables into Supabase via the PostgREST REST API. Handles Gold-to-Supabase ID remapping after server-assigned primary keys are returned. |
| `pipeline/embed.py` | Encodes trial text with `all-MiniLM-L6-v2` and upserts 384-dimensional vectors into Qdrant Cloud. Uses stable NCT-digit IDs to ensure idempotent upserts. |
| `rag/retriever.py` | Embeds user queries and searches Qdrant with a configurable score threshold. Supports concurrent in-memory PDF search via cosine similarity. |
| `rag/generator.py` | Assembles retrieved context (capped at 12,000 characters) and calls Groq Llama 3.3 70B with a citation-enforcing system prompt. |
| `app/dashboard.py` | Reads from Supabase via embedded select joins. Falls back to local Gold Parquet if Supabase is unavailable. Renders metrics, charts, and a filterable trial table. |
| `app/chat.py` | Floating RAG chat panel. Handles PDF upload, in-session chunking and embedding, and renders responses with source citations. |

---

## 4. Tech Stack

| Tool | Purpose | Why Chosen |
|------|----------|-----------|
| **Python 3.11** | Primary language | Strong data ecosystem; type hints for pipeline safety |
| **pandas + pyarrow** | Data transformation and Parquet I/O | Columnar format is efficient for tabular data with repeated strings; pyarrow enables strongly typed schemas |
| **Supabase (PostgreSQL)** | Structured storage and SQL queries | Managed Postgres with a REST API usable from a `anon` JWT — no server to operate; PostgREST enables relational joins without writing SQL in application code |
| **Qdrant Cloud** | Vector storage and semantic search | Purpose-built for vector workloads; persistent cloud-hosted collections; `query_points` API is clean and well-documented; free tier sufficient for this dataset size |
| **sentence-transformers / all-MiniLM-L6-v2** | Text embedding | Runs locally (no API cost); 384-dimensional output balances quality and storage; best-in-class for its size on STS benchmarks |
| **Groq / Llama 3.3 70B** | LLM generation | Groq's inference hardware delivers ~500 tokens/second — near-instant responses for a chat interface; Llama 3.3 70B is competitive with GPT-4o on instruction-following |
| **Streamlit** | Front-end application | Rapid iteration for data applications; `st.cache_resource` prevents model reloads on every interaction; `streamlit-float` enables the floating chat panel without custom frontend code |
| **pdfplumber** | PDF text extraction | Reliable text extraction from standard PDFs; exposes page numbers for citation metadata; pure Python with no system dependencies |
| **python-dotenv** | Secret management | `.env` locally, `st.secrets` on Streamlit Cloud — single code path handles both environments |

---

## 5. Key Technical Decisions

### 1. Medallion Architecture Over a Flat ETL

A single-step pipeline (API → database) would have been faster to build but brittle. By separating extraction (Bronze), cleaning (Silver), and schema construction (Gold) into distinct layers — each persisting its output as Parquet — any stage can be re-run independently. During development, Silver and Gold were iterated dozens of times without re-fetching the API. Each layer has a single, auditable responsibility: Bronze owns fidelity to the source, Silver owns data quality, Gold owns the analytical schema.

### 2. Qdrant over ChromaDB

ChromaDB runs in-process as a local file-based store. It is well-suited for development but requires an infrastructure change to move to production (the persistent client writes to disk; the HTTP client requires a separate server process). Qdrant Cloud is a managed service with a stable cloud API — the same code that runs locally points at the cloud collection with no modification. For a system targeting deployment on Streamlit Cloud, keeping all stateful stores managed (Supabase, Qdrant Cloud) eliminates the file system dependency.

### 3. Groq over OpenAI

Latency is the primary differentiator for a chat interface. OpenAI's GPT-4o typically responds in 3–8 seconds for a 500-token generation. Groq's LPU hardware returns the same output in under 1 second. In a floating chat panel where perceived responsiveness is critical, that difference is substantial. Groq's API also uses the same `client.chat.completions.create` interface as OpenAI, making it a near-zero-cost swap. Llama 3.3 70B was chosen specifically for its strong instruction-following, which is required to enforce citation behaviour.

### 4. Session-Scoped PDF Storage

PDF embeddings are stored in `st.session_state` and are never written to Qdrant, disk, or any database. This was a deliberate design constraint: clinical trial protocols and regulatory documents may be confidential. Session state is the only storage mechanism in Streamlit that is guaranteed to be isolated per user and cleared on session end. Every other approach — temporary files, a database table, a separate Qdrant collection — would persist the document beyond the user's session, which is unacceptable for private clinical data.

### 5. Stable NCT-Digit IDs for Qdrant

Qdrant point IDs must be integers. The initial implementation used Python's `hash()` function on the NCT ID string. This caused a critical data integrity failure: Python enables hash randomisation by default (a security feature — `PYTHONHASHSEED` is random per process), so each pipeline run generated different integer IDs for the same NCT IDs. Rather than updating existing Qdrant points, every run inserted new ones. The collection accumulated to 1,962 vectors after two runs and 2,943 after three. The fix was to extract digits directly from the NCT ID string: `NCT04280705` → `04280705` → `4280705`. This is deterministic, stable, and produces IDs that respect Qdrant's integer constraint without collisions across valid NCT IDs.

### 6. Two Stores for Two Query Types

Supabase and Qdrant serve fundamentally different query patterns and cannot substitute for each other. Supabase answers structured, filterable questions: "how many trials are in Phase 3?", "who are the top 10 sponsors?", "show me all RECRUITING trials for diabetes." These are SQL aggregations. Qdrant answers semantic questions: "find trials related to this question about immunotherapy." This requires vector similarity search, which SQL cannot perform. Maintaining both stores doubles the storage and load complexity but avoids forcing either system to do something it was not designed for.

---

## 6. Pipeline Results

| Metric | Value |
|--------|-------|
| Conditions searched | 5 (cancer, diabetes, Alzheimer's, hypertension, COVID-19) |
| Target trials per condition | 200 |
| Raw records fetched | ~1,000 |
| Records after deduplication (Bronze) | ~990 |
| Records after Silver cleaning | **981** |
| Records dropped in Silver | ~9 (null critical fields or out-of-range dates) |
| Fact table rows in Supabase | 981 |
| Dimension: `dim_phase` | 5 rows |
| Dimension: `dim_condition` | ~400–600 rows (unique primary conditions) |
| Dimension: `dim_sponsor` | ~700+ rows (unique lead sponsors) |
| Vectors in Qdrant | **981** |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| Total pipeline runtime | **~43 seconds** |

The 43-second runtime breaks down approximately as: 25–30 seconds for API extraction (5 conditions × 2 paginated requests with rate-limit delays), 2–3 seconds for Silver + Gold transformation, 5–7 seconds for Supabase load (batched upserts), and 8–10 seconds for embedding (model load + encoding 981 texts).

---

## 7. Data Quality Framework

Quality is enforced at each Medallion layer with explicit checks and logged record counts.

### Bronze Layer
- **Minimum viability check:** Abort the entire pipeline if the API returns zero records.
- **Non-null NCT ID:** Drop any record missing `nctId` — an identifier-less record cannot be keyed, cited, or deduplicated.
- **Deduplication:** The same trial can appear under multiple search conditions. Duplicates are removed by `nctId` before persisting.
- **Audit trail:** Every extraction saves a timestamped Parquet file. The raw API response is never overwritten.

### Silver Layer
- **Critical field nulls:** Drop records missing `nctId`, `briefTitle`, or `condition`. These are the minimum fields required to build the Gold schema and a usable UI row.
- **Date range validation:** `startDate` must be between 2000 and 2026, or null. Pre-2000 trials have poor data quality in the registry; post-2026 dates are likely data entry errors.
- **Phase standardisation:** Raw API phase enums (`PHASE1`, `EARLY_PHASE1`, `NA`) are mapped to five standard labels. Multi-phase trials take the highest phase. Every output value is one of: Phase 1, Phase 2, Phase 3, Phase 4, N/A.
- **Sponsor null fill:** Null sponsors become `"Unknown"` rather than being dropped — the trial is valid even if the sponsor field is absent.
- **Text normalisation:** String `"nan"` artifacts (produced by Pandas' `astype(str)` on null values) are converted back to true nulls.

### Gold Layer
- **Referential integrity assertions:** Three hard assertions verify that every `phase_id`, `condition_id`, and `sponsor_id` in `fact_trials` exists in its corresponding dimension table. The pipeline crashes rather than loading corrupt data.
- **Duplicate NCT IDs:** A final deduplication step removes any NCT IDs that survived to this layer.
- **`is_single_sponsor` null prohibition:** The boolean flag is explicitly set and null-filled to `False`. A nullable boolean in a fact table is a data quality defect.

### Post-Load Verification
- Both Supabase and Qdrant are queried for row/vector counts after loading.
- A warning is logged if the count is below 90% of the expected Gold output. This catches silent partial failures without aborting a partially successful load.

---

## 8. RAG Architecture

The RAG pipeline runs in six steps on every user query:

```
User query
    │
    ▼
[1] Validation — empty or whitespace query returns fallback immediately
    │
    ▼
[2] Embedding — query encoded with all-MiniLM-L6-v2 → 384-dim vector
    │
    ▼
[3] Retrieval — Qdrant query_points(limit=5, score_threshold=0.45)
              + optional in-memory cosine search over session PDF embeddings
    │
    ▼
[4] Confidence gate — if no results above 0.45, return fallback message.
                      LLM is never called without evidence.
    │
    ▼
[5] Context assembly — up to 5 trials formatted as text, capped at 12,000
                       characters (~3,000 tokens). PDF chunks appended if present.
    │
    ▼
[6] Generation — Groq Llama 3.3 70B with system prompt, context, and question.
                 temperature=0.3, max_tokens=1024
    │
    ▼
Structured response: { answer, qdrant_results, pdf_results, top_score,
                        is_high_confidence, fallback }
```

**Confidence scoring:**

| Score range | UI label | Meaning |
|-------------|----------|---------|
| < 0.45 | — | Fallback response. No answer generated. |
| 0.45 – 0.74 | Relevance score: X.XXX | Reasonable match. Proceed with generation. |
| ≥ 0.75 | High-confidence response | Strong semantic match. Answer well-grounded in retrieved trials. |

**Citation requirement:** Every response is accompanied by the raw `ScoredPoint` objects from Qdrant rendered in a "Source Trials" expander. The NCT ID, title, condition, phase, and similarity score are shown for each retrieved trial. The user can verify any claim in the answer against the source record. This is non-optional — the expander is rendered unconditionally on every non-fallback response.

---

## 9. Trustworthiness Design

Every design decision in TrialSight prioritises verifiability over convenience.

**The system does not answer if it cannot cite its source.** If no trial in the Qdrant collection achieves a cosine similarity of 0.45 against the user's query, the system returns a fixed fallback message rather than calling the LLM. An LLM asked to answer with no relevant context will improvise — it will generate plausible-sounding but unverifiable trial information. The score threshold is the primary hallucination prevention mechanism.

**The LLM is explicitly constrained to provided context.** The system prompt instructs Llama 3.3 70B to answer using only the retrieved trial data and to explicitly state when it lacks sufficient information. Temperature 0.3 further reduces the probability of creative departures from the evidence.

**Context is deliberately capped.** The 12,000-character context window is not a technical limitation — it is a design choice. A larger context increases the risk that the model fills gaps with its training data rather than the retrieved documents. Keeping context tight and relevant ensures the model reads what was retrieved.

**Every number in the pipeline is logged.** Input and output record counts are printed at every stage. A developer can audit the path of every record from the 990 raw API responses to the 981 Supabase rows and 981 Qdrant vectors without inspecting the data itself. Drops are explained — not silently absorbed.

**PDF data is never persisted.** The explicit privacy constraint — in-session only, never written to Qdrant or any database — ensures that uploaded documents (which may be confidential trial protocols) cannot be accessed by any other user or any future session. This was a design constraint specified before implementation and enforced at the storage layer, not enforced by policy alone.

---

## 10. Challenges and Solutions

### Challenge 1: Supabase Row-Level Security Blocking All Inserts

After the schema migration SQL was run in the Supabase SQL Editor, every INSERT attempt returned:

```
postgrest.exceptions.APIError: {'message': 'new row violates row-level security
policy for table "dim_phase"', 'code': '42501'}
```

**Root cause:** Supabase enables Row-Level Security by default on all new tables. When RLS is enabled with no policies defined, the default behaviour is deny-all — no rows can be read or written by any non-superuser role, including `anon`. The migration script contained both the `CREATE TABLE` statements and the `ALTER TABLE DISABLE ROW LEVEL SECURITY` statements, but the latter did not execute because the user ran only part of the script.

**Solution:** The pipeline was updated to detect the 42501 error code specifically and print a targeted remediation message — the exact SQL commands needed to disable RLS and grant table access — rather than a generic failure. The load step now also catches RLS errors at the dimension upsert stage separately from fact table errors, making it possible to identify exactly which table was blocked. The Supabase migration SQL was also restructured so the RLS and GRANT statements are visually prominent and clearly explained.

**Lesson:** Supabase's default-deny RLS is correct security behaviour for applications with user-specific data. For a system serving fully public data, it adds friction with zero benefit. The production-ready approach is to include explicit GRANT statements in the initial migration and verify they executed before attempting data loads.

---

### Challenge 2: Qdrant Vector Accumulation from Non-Deterministic IDs

After the embedding pipeline was first run successfully with 981 vectors, a second pipeline run produced 1,962 vectors. A third produced 2,943. The collection was accumulating duplicates on every run rather than updating existing records.

**Root cause:** Qdrant's `upsert` operation updates an existing point if the provided ID already exists, or inserts a new point if it does not. The initial implementation used Python's `hash()` function to generate integer IDs from NCT ID strings. Python enables hash randomisation by default for security reasons — `PYTHONHASHSEED` is randomised per process, so `hash("NCT04280705")` produces a different integer on every run. This meant every execution generated 981 entirely new IDs, and `upsert` inserted 981 new vectors rather than updating the 981 existing ones.

**Solution:** The ID derivation was changed to extract the digit sequence from the NCT ID directly: `"NCT04280705"` → `"04280705"` → integer `4280705`. NCT IDs always follow the format `NCT` followed by exactly 8 digits, so this produces a unique, stable, deterministic integer for every valid NCT ID. The Qdrant collection was deleted and recreated cleanly, and the embedding pipeline was re-run. The collection reached exactly 981 vectors and remained at 981 on subsequent runs.

**Lesson:** Any system that relies on stable identifiers across process boundaries must not use Python's `hash()`. The fix is obvious in retrospect but easy to miss during initial implementation when the system is tested with a single pipeline run.

---

## 11. Links

| | |
|---|---|
| **GitHub Repository** | https://github.com/Nathbobs/TrialSight |
| **Live Demo** | *(URL to be added upon Streamlit Cloud deployment)* |
