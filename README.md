# TrialSight

End-to-end clinical trial data intelligence platform. Fetches real trial data from ClinicalTrials.gov, processes it through a Medallion Architecture ETL pipeline, stores it in Supabase PostgreSQL, embeds trial descriptions into Qdrant Cloud for semantic search, and serves a Streamlit app with an analytics dashboard and RAG-powered chat interface.

---

## Architecture

```
ClinicalTrials.gov API (no key required)
         │
         ▼
   [BRONZE LAYER]  pipeline/extract.py
   Raw JSON → Parquet (data/bronze/)
         │
         ▼
   [SILVER LAYER]  pipeline/transform.py
   Cleaned, filtered, date-normalised
         │
         ▼
   [GOLD LAYER]    pipeline/transform.py
   Star schema: fact_trials + dim_condition/sponsor/phase
         │
    ┌────┴────┐
    ▼         ▼
Supabase   Qdrant Cloud
PostgreSQL  Vector Store
(dashboard) (semantic search)
    └────┬────┘
         ▼
   [RAG LAYER]
   HuggingFace all-MiniLM-L6-v2
   + Groq Llama 3.3 70B
         │
         ▼
   Streamlit App
   ├── Dashboard Tab  (metrics, charts, trial browser)
   └── RAG Chat Tab   (Q&A with citations + PDF upload)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data source | ClinicalTrials.gov API v2 |
| ETL | Python + pandas + pyarrow |
| Storage | Supabase PostgreSQL (managed) |
| Vector DB | Qdrant Cloud |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384 dims) |
| LLM | Groq Llama 3.3 70B Versatile |
| App | Streamlit |
| PDF parsing | pdfplumber |

---

## Project Structure

```
trialsight/
├── run_pipeline.py          ← full ETL + embedding runner
├── requirements.txt
├── .env                     ← secrets (never commit)
├── data/
│   ├── bronze/              ← raw Parquet from API
│   ├── silver/              ← cleaned Parquet
│   ├── gold/                ← star-schema Parquet
│   └── supabase_migration.sql  ← one-time schema setup
├── pipeline/
│   ├── extract.py           ← Bronze: fetch from ClinicalTrials.gov
│   ├── transform.py         ← Silver + Gold transformation
│   ├── load.py              ← Supabase loader
│   └── embed.py             ← Qdrant embedding
├── rag/
│   ├── retriever.py         ← Qdrant semantic search
│   ├── generator.py         ← Groq LLM call
│   └── pipeline.py          ← end-to-end RAG orchestration
├── app/
│   ├── dashboard.py         ← Streamlit dashboard tab
│   ├── chat.py              ← Streamlit chat tab (+ PDF upload)
│   └── main.py              ← Streamlit entry point
├── tests/
│   ├── test_pipeline.py     ← ETL unit tests (27 tests)
│   └── test_rag.py          ← RAG unit tests (19 tests + 3 integration)
└── docs/
    ├── DESIGN.md
    ├── SKILLS.md
    ├── AGENTS.md
    └── TRUSTWORTHINESS.md
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/TrialSight.git
cd TrialSight
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
QDRANT_URL=https://<cluster-id>.us-east-1-1.aws.cloud.qdrant.io
QDRANT_API_KEY=eyJ...
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=eyJ...
```

### 3. Create the Supabase schema (one-time)

The pipeline auto-generates `data/supabase_migration.sql`. Apply it once:

1. Open your [Supabase dashboard](https://app.supabase.com)
2. Go to **SQL Editor → New query**
3. Paste the contents of `data/supabase_migration.sql`
4. Click **Run**

This creates the four tables (`dim_phase`, `dim_condition`, `dim_sponsor`, `fact_trials`) with RLS disabled so the anon API key can read and write.

---

## Running the Pipeline

```bash
# Full pipeline: extract → transform → load → embed (~5–10 min)
python run_pipeline.py

# Skip re-fetching (reuse latest Bronze files)
python run_pipeline.py --skip-extract

# Skip Supabase load (if tables not yet created)
python run_pipeline.py --skip-extract --skip-load
```

Pipeline output example:

```
TrialSight Pipeline — started at 2026-06-23 11:26:13
[BRONZE] Starting extraction at ...
[BRONZE] 'cancer': 200 records fetched
...
[SILVER] Input: 990 → Output: 981 records
[GOLD]   Input: 981 → Output: 981 fact records
[LOAD]   981 rows upserted to fact_trials
[EMBED]  981 vectors upserted to Qdrant
TrialSight Pipeline — completed in ~300s
```

---

## Running the App

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

### Dashboard tab
- Top-line metrics: total trials, unique conditions, sponsors, single-sponsor %
- Status breakdown, phase distribution, trial starts per year
- Filterable/searchable trial table
- Falls back to Gold Parquet files if Supabase is unavailable

### RAG Chat tab
- Ask natural language questions about clinical trials
- Every response cites NCT IDs and shows relevance scores
- Optional PDF upload for session-scoped document search (privacy-safe: in-memory only)

---

## Running Tests

```bash
# All unit tests (46 tests, no network calls)
python -m pytest tests/ -v -m "not integration"

# Include live integration tests (requires Qdrant + Groq)
python -m pytest tests/ -v -m integration
```

---

## Deploying to Streamlit Cloud

1. Push your repository to GitHub (`.env` and `data/` are gitignored)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch `main`, entrypoint `app/main.py`
4. Under **Advanced settings → Secrets**, add the same five variables as your `.env`:

```toml
GROQ_API_KEY = "gsk_..."
QDRANT_URL = "https://..."
QDRANT_API_KEY = "eyJ..."
SUPABASE_URL = "https://..."
SUPABASE_KEY = "eyJ..."
```

5. Deploy — the app loads the embedding model on first run (may take 1–2 min to warm up)

---

## Data Coverage

- ~981 unique clinical trials across 5 conditions
- Conditions: cancer, diabetes, alzheimer, hypertension, covid
- Source: ClinicalTrials.gov API v2 (no key required)
- Refresh by re-running `python run_pipeline.py`

---

## Trustworthiness

TrialSight follows strict trustworthiness rules (see `docs/TRUSTWORTHINESS.md`):

- Every RAG response cites at least one NCT ID
- Relevance scores are shown alongside every citation
- Responses with top score < 0.45 trigger a fallback message
- Responses with top score > 0.75 are marked as high-confidence
- LLM temperature is 0.3 to minimise hallucination
- System prompt explicitly prohibits speculation

---

*Data from ClinicalTrials.gov. Not medical advice.*
