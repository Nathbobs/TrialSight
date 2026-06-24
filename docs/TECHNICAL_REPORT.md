# TrialSight — Technical Report

**Audience:** The developer who built this system and wants to deeply understand every decision and tradeoff.  
**Pipeline run referenced throughout:** ~43 seconds, 990 raw → 981 clean → 981 vectors.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [ClinicalTrials.gov API](#2-clinicaltrialgov-api)
3. [Bronze Layer](#3-bronze-layer)
4. [Silver Layer](#4-silver-layer)
5. [Gold Layer](#5-gold-layer)
6. [Supabase Load](#6-supabase-load)
7. [Embedding Layer](#7-embedding-layer)
8. [RAG Pipeline](#8-rag-pipeline)
9. [PDF Upload Feature](#9-pdf-upload-feature)
10. [Trustworthiness Layer](#10-trustworthiness-layer)
11. [Testing](#11-testing)
12. [Known Limitations](#12-known-limitations)

---

## 1. System Architecture Overview

TrialSight is composed of two independent subsystems that share no code but share the same data: the **ETL pipeline** and the **Streamlit application**. They connect through two persistent stores — Supabase PostgreSQL and Qdrant Cloud.

```
ClinicalTrials.gov API
        │
        ▼
  [BRONZE LAYER]          pipeline/extract.py
  Raw JSON → Parquet
        │
        ▼
  [SILVER LAYER]          pipeline/transform.py
  Cleaned Parquet
        │
        ▼
  [GOLD LAYER]            pipeline/transform.py
  Star Schema Parquet
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
  [SUPABASE LOAD]                    [QDRANT EMBED]
  pipeline/load.py                   pipeline/embed.py
  PostgreSQL REST API                384-dim vectors
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
              [STREAMLIT APP]
              app/main.py
               │         │
        dashboard.py   chat.py
        (SQL queries)  (RAG pipeline)
                            │
                    rag/retriever.py   ←── Qdrant
                    rag/generator.py   ←── Groq API
                    rag/pipeline.py    ←── orchestrator
```

**Data flows in one direction only.** The pipeline writes; the app reads. No component reaches back into the pipeline at runtime. This is intentional — it means the app can run even while the pipeline is broken, as long as data was loaded at some prior point.

**Why two stores?** Supabase handles structured queries — "how many trials are in Phase 3?", "who are the top 10 sponsors?" — efficiently via SQL. Qdrant handles semantic search — "find trials relevant to this question" — which SQL cannot do. The dashboard reads Supabase. The chat reads Qdrant. Neither can substitute for the other.

**Why Parquet at every stage?** Each layer saves its output as Apache Parquet files in `data/bronze/`, `data/silver/`, or `data/gold/`. This means you can re-run any downstream stage without re-fetching the API. The `--skip-extract` flag in `run_pipeline.py` exists because of this — it loads the latest Bronze Parquet and skips the 30-second API fetch. Parquet is also strongly typed, columnar, and compresses well for tabular data with repeated strings (condition names, statuses).

---

## 2. ClinicalTrials.gov API

### Endpoint

```
GET https://clinicaltrials.gov/api/v2/studies
```

This is the v2 API. No API key is required. The v2 API differs substantially from v1: it uses a JSON structure with a `protocolSection` top-level key that contains nested module objects rather than a flat key-value schema.

### Parameters Used

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `query.cond` | e.g. `"cancer"` | Free-text search on condition field |
| `pageSize` | `100` | Maximum studies per request (API cap is 1000, but 100 is safer for stability) |
| `format` | `"json"` | Response format |
| `fields` | (see below) | Whitelist of fields to return |
| `pageToken` | (pagination cursor) | Next-page token returned by prior response |

The `fields` parameter is critical for performance. Without it, the API returns the entire study record which can be hundreds of kilobytes per study. By whitelisting exactly what is needed:

```
NCTId, BriefTitle, OfficialTitle, OverallStatus,
Phase, StartDate, CompletionDate, Condition,
InterventionName, LeadSponsorName,
BriefSummary, EligibilityCriteria
```

...the response size is kept manageable. Each page of 100 studies with these fields is typically 300–800KB.

### Pagination

The API uses cursor-based pagination. When a response contains more results, it includes a `nextPageToken` string. The extractor passes this token as `pageToken` on the next request. The loop continues until either:
- `nextPageToken` is absent (no more results exist for this query), or
- The collected record count reaches `TARGET_PER_CONDITION` (200).

After hitting the target, the loop breaks and results are sliced to exactly 200 with `studies[:target]`.

### Response Structure

Each study in `data["studies"]` has this nested shape:

```json
{
  "protocolSection": {
    "identificationModule": { "nctId": "NCT...", "briefTitle": "..." },
    "statusModule": { "overallStatus": "RECRUITING", "startDateStruct": {"date": "2020-01"} },
    "descriptionModule": { "briefSummary": "..." },
    "designModule": { "phases": ["PHASE2"] },
    "sponsorCollaboratorsModule": { "leadSponsor": {"name": "..."} },
    "eligibilityModule": { "eligibilityCriteria": "Inclusion: ..." },
    "conditionsModule": { "conditions": ["Lung Cancer", "NSCLC"] },
    "armsInterventionsModule": { "interventions": [{"name": "Drug A"}] }
  }
}
```

Each module is accessed with `.get("moduleName", {})` to safely handle missing modules — some studies omit entire modules, particularly `armsInterventionsModule` for observational studies.

### Why These 5 Conditions?

Cancer, diabetes, Alzheimer's, hypertension, and COVID-19 were selected to produce a diverse, high-volume dataset that covers:
- Oncology (largest category on ClinicalTrials.gov, >100k trials)
- Chronic metabolic disease (diabetes)
- Neurodegeneration (Alzheimer's)
- Cardiovascular (hypertension)
- Infectious disease (COVID-19)

These conditions together ensure all major therapy areas are represented and that the embedding space covers diverse medical vocabulary.

### Why These Fields?

- **nctId**: The globally unique identifier for every clinical trial. Used as the primary key throughout the system and as the citation in RAG responses.
- **briefTitle / officialTitle**: briefTitle is short and always present. officialTitle is often more precise but can be null.
- **overallStatus**: The trial lifecycle state — RECRUITING, COMPLETED, TERMINATED, etc. Used for dashboard filtering.
- **phase**: The drug development stage. Clinically meaningful — Phase 1 is safety, Phase 3 is efficacy at scale.
- **startDate / completionDate**: Used for timeline analysis in the dashboard.
- **condition**: The condition(s) the trial targets. Note: the API returns a list here — a single trial may list "Lung Cancer; NSCLC; Stage IV". Only the first condition is used in the Gold layer.
- **briefSummary**: 200–500 word plain-language description. This is the most important text for semantic embedding — it captures what the trial actually does.
- **eligibilityCriteria**: Who can participate. Long (often 1000–3000 words). Stored but truncated for embedding.
- **sponsorName** (leadSponsor): The entity running the trial. Used for sponsor dimension and `is_single_sponsor` flag.
- **intervention**: Drug or device being tested. Stored as a semicolon-joined string when multiple interventions exist.

### Retry Logic

If any `requests.get()` call fails (timeout, 5xx, network error), the extractor sleeps 2 seconds and retries once. If the retry also fails, it raises a `RuntimeError` and the pipeline aborts. There is no exponential backoff or circuit breaker — for a data pipeline that runs once, a single retry is sufficient.

---

## 3. Bronze Layer

### What It Does

The Bronze layer's job is to fetch raw data and persist it. Nothing more. It performs the minimum transformation necessary to handle a data structure (nested API JSON → flat dict) but applies virtually no quality filtering.

### What "Minimal Filtering" Means

Two things happen in Bronze that could be called filtering:

1. **nctId null drop**: Any record missing `nctId` is dropped immediately. This is not a quality filter — it's an impossibility check. A study without an NCT ID is not a study. It cannot be deduplicated, keyed, or cited. There's nothing to do with it.

2. **Deduplication by nctId**: The extractor fetches 5 conditions × 200 = up to 1,000 records. Because the same trial can appear under multiple search conditions (a "cancer AND diabetes" trial would appear in both cancer and diabetes results), the combined list can have duplicates. `drop_duplicates(subset=["nctId"])` removes these. In the actual pipeline run, this step reduced 1,000 fetched records to approximately 990 unique records.

Everything else is preserved exactly as received. Null briefSummary? Keep it. Weird date format? Keep it. Multiple phases in one field? Keep the raw string. The principle is that Bronze is the permanent record of what the API returned. If something goes wrong later, you can re-run Silver and Gold from Bronze without hitting the API again.

### Output Format: Parquet

Files are saved as `data/bronze/trials_raw_{YYYYMMDD_HHMMSS}.parquet`. The timestamp in the filename means each pipeline run creates a new file rather than overwriting the previous one. This is intentional — it provides a full audit trail of every extraction.

The Parquet schema for Bronze is exactly the API field mapping:

```
nctId           object (string)
briefTitle      object
officialTitle   object
overallStatus   object
phase           object  ← semicolon-separated, e.g. "PHASE1; PHASE3"
startDate       object  ← raw string, e.g. "2020-01" or "January 2020"
completionDate  object
condition       object  ← semicolon-separated list
searchCondition object  ← the condition keyword used to search
intervention    object  ← semicolon-separated
sponsorName     object
briefSummary    object
eligibilityCriteria object
```

All fields are `object` dtype in Bronze because nothing has been parsed yet. Dates are still strings. Phases are still raw API enums. This is deliberate — Silver is responsible for type coercion.

---

## 4. Silver Layer

### Purpose

Silver applies every quality filter that determines whether a record is usable. A record that passes Silver can be trusted to have the minimum fields required for the Gold schema.

### Step 1: Null Drop on Critical Fields

```python
critical = ["nctId", "briefTitle", "condition"]
df = df.dropna(subset=critical)
```

Three fields are treated as non-negotiable:

- **nctId**: Needed for the primary key and citation. A trial without an NCT ID cannot be cited in a RAG response.
- **briefTitle**: Every trial must have a human-readable name. The dashboard table would show a blank row without it.
- **condition**: Needed to build the `dim_condition` dimension table and assign `condition_id`. Without a condition, referential integrity in Gold would fail.

Fields like `briefSummary`, `eligibilityCriteria`, and `officialTitle` are allowed to be null — they'll be handled gracefully later with `_safe_text()` in the loader.

In the actual run, this step dropped approximately 9 records, reducing from ~990 to ~981.

### Step 2: Date Parsing and Range Validation

```python
df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
```

`errors="coerce"` converts unparseable date strings to `NaT` (Not a Time) rather than raising an exception. The ClinicalTrials.gov API returns dates in several formats: `"2020-01"`, `"January 2020"`, `"2020-01-15"`. Pandas handles all of these correctly.

After parsing, trials with `startDate` before 2000 or after 2026 are dropped:

```python
valid_start = df["startDate"].isna() | (
    (df["startDate"].dt.year >= 2000) & (df["startDate"].dt.year <= 2026)
)
```

**Why 2000?** Trials from before 2000 are typically historical and their data quality on ClinicalTrials.gov is poor — many fields are absent or truncated. They also skew the timeline chart in the dashboard.

**Why allow null startDate through?** Not every trial has a confirmed start date, especially APPROVED or NOT_YET_RECRUITING trials. Dropping nulls here would remove valid trials that are simply pending. The null is preserved and handled gracefully in the loader (`start = None if pd.isna(...)`) and in the dashboard's time-series chart (which drops NaT explicitly before plotting).

**Why 2026?** This is the practical future bound given the data was collected in 2025. A trial listed as starting in 2030 is suspicious and possibly a data quality issue.

In the actual run, this validation dropped approximately 0–2 records (very few trials had out-of-range dates).

### Step 3: Phase Normalization

```python
PHASE_MAP = {
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "EARLY_PHASE1": "Phase 1",
    "NA": "N/A",
    "N/A": "N/A",
}
```

The API returns phases as SCREAMING_SNAKE_CASE: `"PHASE1"`, `"EARLY_PHASE1"`, `"NA"`. These are mapped to readable labels. `EARLY_PHASE1` — used for First-in-Human dosing studies — collapses to `"Phase 1"` because from a data analysis perspective, it belongs in that tier.

**Multi-phase handling:** Some trials span phases: `"PHASE1; PHASE3"` (rare but real, typically adaptive trial designs). The normalizer splits on semicolons and picks the highest phase:

```python
for label in ["Phase 4", "Phase 3", "Phase 2", "Phase 1"]:
    if label in normalized:
        return label
```

The logic is that if a trial is in both Phase 1 and Phase 3 simultaneously, classifying it as Phase 3 is more informative for users filtering by phase.

**Unknown phases:** Any raw string not in `PHASE_MAP` returns `"N/A"`. This is a safe default — better to say "unknown phase" than to create junk categories.

### Step 4: Text Field Whitespace Normalization

```python
for col in ["briefTitle", "officialTitle", "briefSummary", "eligibilityCriteria"]:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace("nan", np.nan)
```

The `astype(str)` converts actual Python `None` values to the string `"nan"` (because `str(None)` is `"None"` but `str(NaN)` is `"nan"` — Pandas' behavior). The `.replace("nan", np.nan)` converts those string artifacts back to real nulls. Without this step, null summaries would appear in the UI as the literal text "nan".

### Step 5: Sponsor Null Fill

```python
df["sponsorName"] = df["sponsorName"].fillna("Unknown")
```

Sponsor is required to build `dim_sponsor`. Rather than dropping trials with no sponsor (which would be aggressive — some valid trials simply lack this field), they're assigned `"Unknown"` as a catch-all sponsor. In Gold, multiple trials without a sponsor all share this one `"Unknown"` sponsor entry.

### Silver Output

~981 records, each with:
- Guaranteed non-null: nctId, briefTitle, condition, sponsorName
- Typed dates: startDate and completionDate as datetime64
- Standardized phases: one of {Phase 1, Phase 2, Phase 3, Phase 4, N/A}
- Clean text: no "nan" strings, whitespace stripped

---

## 5. Gold Layer

### What a Star Schema Is

A star schema organizes data into one central fact table and multiple surrounding dimension tables. The fact table holds the events (in this case, clinical trials), and the dimension tables hold the descriptive attributes (conditions, sponsors, phases).

The name "star schema" comes from the diagram shape: the fact table in the center, dimensions radiating outward like star points.

### Why Use a Star Schema Here?

The alternative is a single flat table: one row per trial with every field denormalized into columns. That works for 981 records. The star schema was chosen because:

1. **It matches Supabase's relational model** — foreign key relationships can be declared and enforced.
2. **It reduces redundancy** — "Phase 2" appears as one row in `dim_phase` rather than in 300+ rows of a flat table.
3. **It enables Supabase's embedded select syntax** — `client.table("fact_trials").select("*, dim_phase(phase_label)")` joins the dimension in one REST call.
4. **It aligns with how real data warehouses work** — the Medallion Architecture is a well-established pattern and implementing it completely is the point of this project.

### The Four Tables

#### `dim_phase`

```
phase_id    INTEGER  (surrogate key, 1-based)
phase_label TEXT     (Phase 1, Phase 2, Phase 3, Phase 4, N/A)
```

5–6 rows total (one per distinct phase label). In the actual run there were 5 unique phases.

#### `dim_condition`

```
condition_id    INTEGER  (surrogate key, 1-based)
condition_name  TEXT     (e.g. "Lung Cancer", "Type 2 Diabetes Mellitus")
```

This dimension can have many rows — the primary condition extracted from each trial's condition list is often highly specific. A trial under the "cancer" search might have a primary condition of "Non-small Cell Lung Carcinoma Stage IIIB". In the actual run there were ~400–600 unique conditions.

The primary condition is extracted by splitting on semicolons and taking the first element:

```python
df["primary_condition"] = df["condition"].apply(
    lambda x: str(x).split(";")[0].strip() if pd.notna(x) else "Unknown"
)
```

This is a deliberate simplification. The full condition string (e.g., "Lung Cancer; NSCLC; Metastatic") is too granular for a foreign key. Taking the first condition preserves the most specific primary classification.

#### `dim_sponsor`

```
sponsor_id    INTEGER  (surrogate key, 1-based)
sponsor_name  TEXT     (e.g. "Pfizer", "National Cancer Institute")
sponsor_type  TEXT     (always "Unknown" — see note below)
```

**Why is sponsor_type always "Unknown"?** The ClinicalTrials.gov API v2 does return a sponsor class field (`leadSponsorClass`) in the `sponsorCollaboratorsModule`, but it wasn't included in the field whitelist during extraction. The `sponsor_type` column exists in the schema for forward compatibility — it's in the Supabase table and could be populated in a future extraction run. For now, "Unknown" is honest.

#### `fact_trials`

```
nct_id              TEXT     (natural key, unique)
brief_title         TEXT
official_title      TEXT     (nullable)
overall_status      TEXT     (nullable)
phase_id            INTEGER  → dim_phase
condition_id        INTEGER  → dim_condition
sponsor_id          INTEGER  → dim_sponsor
start_date          DATE     (nullable)
completion_date     DATE     (nullable)
brief_summary       TEXT     (nullable)
eligibility_criteria TEXT    (nullable)
is_single_sponsor   BOOLEAN
```

### What Surrogate Keys Are

A surrogate key is an ID that the system invents. It has no meaning in the real world — it's just a sequential integer (1, 2, 3...) assigned to each unique value in a dimension table. The Gold layer assigns these locally using:

```python
dim_condition = pd.DataFrame({
    "condition_id": range(1, len(conditions) + 1),
    "condition_name": conditions,
})
```

**The critical complication:** When these local Gold IDs (1, 2, 3...) are loaded into Supabase, Supabase assigns its own IDs using a SERIAL sequence. The Supabase-assigned IDs for the same rows may be completely different numbers. The load.py file handles this by:

1. Upserting the dimension tables first
2. Reading back the server-returned IDs from the API response
3. Building a `{name → server_id}` map
4. Using that map to set foreign keys in fact_trials before upserting

This remapping is the trickiest part of the entire pipeline. If it were skipped, fact_trials rows would have Gold-local IDs pointing at the wrong Supabase dimension rows.

### What `is_single_sponsor` Means

```python
sponsor_counts = df["sponsorName"].value_counts()
fact_trials["is_single_sponsor"] = df["sponsorName"].map(
    lambda s: bool(sponsor_counts.get(s, 0) == 1)
).values
```

For each trial, this flag is True if that trial's sponsor appears exactly once in the entire dataset — meaning they sponsor only one trial in the 981-record set. A sponsor who runs 50 trials gets `is_single_sponsor = False` for all their trials.

**Why this is interesting:** Sponsors who run only one trial are typically academic institutions, small biotech companies, or individual researchers doing proof-of-concept work. Large pharmaceutical companies always have `is_single_sponsor = False`. This flag creates a simple proxy for "is this a commercial trial or an academic/investigator-initiated trial?" It appears in the dashboard as a percentage metric.

Note: this is calculated against the dataset (981 trials), not against the full ClinicalTrials.gov database. A sponsor who has 1 trial in TrialSight but 50 on ClinicalTrials.gov will incorrectly get `is_single_sponsor = True`.

### Referential Integrity Assertions

After building the fact table, three assertions run:

```python
assert fact_trials["phase_id"].notna().all()
assert fact_trials["condition_id"].notna().all()
assert fact_trials["sponsor_id"].notna().all()
```

If any foreign key is null, the pipeline crashes with a clear error. This cannot happen in normal operation (Silver guarantees non-null condition and sponsor; phase is always set to at least "N/A"), but the assertions act as a safety net against logic bugs.

---

## 6. Supabase Load

### Why Direct PostgreSQL Failed

The SUPABASE_KEY in `.env` is a JWT (JSON Web Token) with the claim `"role": "anon"`. This is the anonymous API key meant for client-side read/write operations through the PostgREST REST API. It is not a PostgreSQL password.

The pipeline attempts direct PostgreSQL connection first via SQLAlchemy:

```python
db_url = (
    f"postgresql+psycopg2://postgres.{PROJECT_REF}:{SUPABASE_KEY}"
    f"@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
)
```

This fails because the transaction pooler requires the actual PostgreSQL `postgres` user password, which is a different secret from the API key. Additionally, the Supabase project's direct connection hostname (`db.{ref}.supabase.co`) resolves via IPv6 only, and port 5432 was firewalled in the test environment. All of this is logged and the pipeline falls through to the REST approach gracefully.

**Why SQLAlchemy was tried at all:** DDL (CREATE TABLE, ALTER TABLE) requires database-level access. The anon key cannot execute DDL through PostgREST — PostgREST only exposes table-level CRUD operations. The SQLAlchemy attempt was the only way to auto-create tables programmatically.

### The Migration SQL Workaround

Since DDL cannot run programmatically, `load.py` auto-generates `data/supabase_migration.sql` and instructs the user to run it once in the Supabase SQL Editor. This file contains:

```sql
CREATE TABLE IF NOT EXISTS fact_trials (...);
CREATE TABLE IF NOT EXISTS dim_phase (...);
-- etc.
ALTER TABLE dim_phase DISABLE ROW LEVEL SECURITY;
-- etc.
GRANT ALL ON dim_phase, dim_condition, dim_sponsor, fact_trials TO anon, authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
```

This is not a limitation of the architecture — it's the expected pattern for Supabase projects that use the anon key for data operations. The migration runs once and never needs to run again unless the schema changes.

### What Upsert Means

`UPSERT` is an "insert or update" operation. In Supabase/PostgREST:

```python
client.table("fact_trials").upsert(records, on_conflict="nct_id").execute()
```

This means: for each record, if a row with the same `nct_id` already exists, update it with the new values; if it doesn't exist, insert it. The alternative would be pure INSERT, which fails if a row already exists.

**Why upsert instead of insert?** The pipeline is designed to be re-run safely. If you run `run_pipeline.py` twice, the second run should not create 1962 rows — it should update the 981 existing rows with fresh data. Upsert achieves this idempotency. The downside is a small performance overhead (the database must check for existing rows before deciding what to do), but at 981 records this is negligible.

### How RLS Was Blocking Inserts

Row-Level Security (RLS) is a PostgreSQL feature that lets you define policies controlling which rows a given database role can access. Supabase enables RLS by default on all new tables.

When RLS is enabled with no policies defined, the default is **deny all** — no rows can be read or written by any non-superuser role, including `anon`. The CREATE TABLE statements in the migration ran successfully, but the `ALTER TABLE DISABLE ROW LEVEL SECURITY` statements at the end of the same script were not applied (the user only ran part of the migration). This left tables with RLS enabled and no policies, causing every INSERT to fail with:

```
new row violates row-level security policy for table "dim_phase"
PostgreSQL error code: 42501
```

**Why disabling RLS is safe here:** TrialSight serves public data. ClinicalTrials.gov is a US government database of public records. There is no user-specific data, no PII, no access control requirement. RLS would only add complexity with zero security benefit for this use case. In a production system with user accounts or private data, RLS policies would be essential.

### Batch Upsert Strategy

fact_trials rows are upserted in batches of 100:

```python
for i in range(0, len(records), batch_size):
    batch = records[i: i + batch_size]
    res = client.table("fact_trials").upsert(batch, on_conflict="nct_id").execute()
    time.sleep(0.05)
```

The 50ms sleep between batches is a courtesy rate limit. The Supabase free tier enforces rate limits, and slamming 981 rows in one HTTP call risks hitting them. With batches of 100 and 50ms delays, the load completes in under 2 seconds.

### Post-Load Verification

After all batches are upserted, the pipeline queries the row count:

```python
verify = client.table("fact_trials").select("trial_id", count="exact").execute()
db_count = verify.count or 0
if db_count < len(fact_df) * 0.9:
    print(f"WARNING: expected ~{len(fact_df)}, found {db_count}")
```

The 90% threshold (not 100%) accounts for the possibility of a few records failing due to network hiccups. If more than 10% of records are missing, something is seriously wrong and it deserves investigation. In the successful run, `db_count` matched `len(fact_df)` exactly at 981.

---

## 7. Embedding Layer

### What Sentence Transformers Do (Plain Language)

A sentence transformer is a neural network that reads text and outputs a list of numbers — a **vector** or **embedding**. The key property: texts with similar meanings produce numerically similar vectors, even if the words are completely different. "Cancer immunotherapy clinical trial" and "oncology immune checkpoint study" would produce very similar vectors, even though they share almost no words.

This similarity is measured geometrically. Each embedding is a point in a high-dimensional space. Two points that are close together represent texts with similar meaning. This is what makes semantic search possible — instead of looking for exact word matches, you look for nearby points.

### Why all-MiniLM-L6-v2

The model `sentence-transformers/all-MiniLM-L6-v2` was chosen for four reasons:

1. **Speed:** The "MiniLM" architecture is a compressed (distilled) version of larger BERT models. The L6 variant has 6 transformer layers. It encodes 100 texts in ~0.5 seconds on CPU — critical for embedding 981 trials in a reasonable pipeline runtime.

2. **No auth required:** Unlike OpenAI embeddings, this model runs locally. No API key, no cost, no rate limit, no network dependency during the embedding phase.

3. **384 dimensions:** Competitive quality in a compact representation. OpenAI's `text-embedding-ada-002` uses 1536 dimensions, which is more expressive but 4× larger — overkill for a dataset of 981 trials.

4. **Well-benchmarked:** This model is #1 on several semantic textual similarity (STS) benchmarks for its size class. It was specifically trained on diverse text pairs to understand semantic similarity, not just syntactic overlap.

### What 384 Dimensions Means

Each trial's text is converted to a list of 384 floating-point numbers. You can think of these 384 numbers as coordinates locating the trial's meaning in a 384-dimensional space. No individual dimension has an interpretable label — they're learned abstractions that emerge from training. What matters is that the geometric distance between any two points in this space approximates the semantic distance between the corresponding texts.

For 981 trials, the embedding matrix is `981 × 384 = 376,704` numbers, which at 4 bytes per float32 is about 1.5MB. Trivial to hold in memory.

### What the Embedded Text Looks Like

For each trial, the embedder builds a text string:

```python
parts = [
    f"Title: {row['brief_title']}",
    f"Condition: {row['condition']}",
    f"Phase: {row['phase']}",
    f"Status: {row['overall_status']}",
    f"Summary: {row['brief_summary'] or ''}",
]
text = " | ".join(p for p in parts if p.split(": ", 1)[1].strip())
```

A typical embedded text looks like:
```
Title: A Phase 2 Study of Drug X in Patients With Advanced NSCLC | 
Condition: Non-small Cell Lung Carcinoma | Phase: Phase 2 | 
Status: RECRUITING | Summary: This study evaluates the safety...
```

The fields are concatenated with ` | ` separators so the model sees them as a continuous sequence. The brief_summary is the most important field for semantic similarity — it contains the most clinical vocabulary. However, title and condition are prepended so that searches for "Phase 3 cancer" match immediately without depending on the summary containing those words.

### How Qdrant Stores Vectors

Each vector in Qdrant is a **Point** with three parts:
- `id`: An integer uniquely identifying this point in the collection
- `vector`: The 384-dimensional float list
- `payload`: A JSON object with arbitrary metadata

The payload stored for each trial:
```python
{
    "nct_id": "NCT04280705",
    "brief_title": "A Phase 2 Study...",
    "condition": "Non-small Cell Lung Carcinoma",
    "phase": "Phase 2",
    "sponsor": "Pfizer",
    "brief_summary": "This study evaluates...",  # truncated to 1000 chars
    "overall_status": "RECRUITING",
}
```

The payload is what gets returned in search results and displayed in the chat citations.

### Stable Integer IDs

Qdrant requires integer IDs for points. A simple approach would be to use Python's `hash()` function on the NCT ID string. This was actually tried initially and caused a critical bug: Python's `hash()` is non-deterministic by default (it uses a random seed per process, a security feature called hash randomization). Running the pipeline twice would produce different hash values for the same NCT ID, causing the second run to insert duplicate vectors instead of updating existing ones. After two runs the collection had 1,962 vectors; after three, 2,943.

The fix:
```python
nct_str = str(row["nct_id"])
digits = "".join(filter(str.isdigit, nct_str))
point_id = int(digits) if digits else abs(int.from_bytes(nct_str.encode(), "big")) % (2**53)
```

NCT IDs follow the format `NCT` followed by 8 digits. `NCT04280705` → strip non-digits → `"04280705"` → `int("04280705")` → `4280705`. This is deterministic across all runs, all Python versions, all machines.

The fallback (`int.from_bytes(...)`) handles any hypothetical non-standard ID that contains no digits. The modulo `2**53` keeps the ID within JavaScript's safe integer range (Qdrant's API is HTTP-based; large integers can lose precision in JSON).

### Cosine Distance in Qdrant

The collection was created with `Distance.COSINE`:

```python
VectorParams(size=384, distance=Distance.COSINE)
```

Cosine distance measures the angle between two vectors, ignoring their magnitudes. Two vectors pointing in the same direction have cosine similarity of 1.0 (identical meaning). Perpendicular vectors have similarity 0.0 (unrelated). This is preferable to Euclidean distance for text embeddings because the magnitude of an embedding can vary based on text length — cosine distance normalizes this out.

---

## 8. RAG Pipeline

### What Retrieval Augmented Generation Means (Plain Language)

Retrieval Augmented Generation (RAG) solves a fundamental problem with large language models: they are trained on data up to a cutoff date and have no knowledge of your specific database. If you ask Groq's Llama 3.3 "What clinical trials are recruiting for diabetes right now?", it cannot answer from its training data — it doesn't know what's in your Qdrant collection.

RAG solves this in three steps:
1. **Retrieve:** Find the most relevant documents from your database for the user's question.
2. **Augment:** Paste those documents into the prompt sent to the LLM, as context.
3. **Generate:** Ask the LLM to answer the question *using only the provided context*.

The LLM becomes a reading comprehension engine rather than a knowledge base. It reads the retrieved trials and synthesizes an answer. The retrieved trials are the ground truth; the LLM just explains them.

### Step-by-Step: What Happens When a User Asks a Question

**Step 1 — Query Intake**  
`run_rag()` in `rag/pipeline.py` receives the user's text. If it's empty or only whitespace, it returns immediately with `{"fallback": True, "answer": "Please enter a question."}` — no embedding is computed, no API is called.

**Step 2 — Query Embedding**  
The user's question text is passed through the same `all-MiniLM-L6-v2` model that embedded the trials. This produces a 384-dimensional vector representing the question's meaning.

```python
query_vector = model.encode(query).tolist()
```

This is fast — single-text encoding takes ~2ms on CPU. The model is loaded once at app startup via `@st.cache_resource` and reused for every query.

**Step 3 — Qdrant Retrieval**  
Qdrant searches its `clinical_trials` collection for the 5 points (vectors) nearest to the query vector, with a minimum score threshold of 0.45:

```python
response = client.query_points(
    collection_name="clinical_trials",
    query=query_vector,
    limit=5,
    score_threshold=0.45,
    with_payload=True,
)
results = response.points  # list of ScoredPoint objects
```

Qdrant performs an Approximate Nearest Neighbor (ANN) search — it doesn't compare the query against all 981 vectors exhaustively. It uses an HNSW (Hierarchical Navigable Small World) graph index to find near-optimal neighbors in O(log n) time. For 981 vectors, this is effectively instantaneous (~1ms), but ANN matters at scale.

**Step 4 — Score Threshold Check**  
If no results exceed the 0.45 threshold, the pipeline returns the fallback message immediately without calling Groq. This is critical — calling Groq with zero relevant context would result in a hallucinated answer. The threshold acts as a gate: only proceed with generation if there is genuine relevant evidence.

**Step 5 — Context Assembly**  
`_build_context()` in `rag/generator.py` formats the retrieved trials as text:

```python
"Relevant clinical trials from the database:\n"
"Trial NCT04280705: A Phase 2 Study...\nSummary: This study evaluates..."
# ... up to 5 trials
```

The context is capped at 12,000 characters, which is approximately 3,000 tokens. This cap prevents the LLM from having to read more context than it needs and keeps costs predictable.

**Step 6 — LLM Generation**  
The assembled context + user question are sent to Groq as a chat completion:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Context:\n{context}"},
    {"role": "user", "content": f"Question: {query}"},
]
```

The system prompt is the key behavioural constraint:
> "Answer questions using only the provided clinical trial context. Always cite the NCT ID of trials you reference. If the context does not contain enough information, say so honestly rather than speculating."

**Step 7 — Response Return**  
`run_rag()` returns a structured dict containing the answer, the raw retrieved points (for citation display), the top score, and confidence flags. The UI layer reads this dict — it never sees the intermediate steps.

### Why Temperature 0.3

Temperature controls how creative (or random) the LLM is when generating text. At temperature 1.0, the model samples from its probability distribution with full randomness. At temperature 0.0, it always picks the highest-probability token — fully deterministic but sometimes repetitive.

Temperature 0.3 is a low-but-not-zero setting. It means:
- The model closely follows the retrieved context rather than improvising
- It won't hallucinate creative extensions of what the trials say
- It still produces natural, varied phrasing rather than robotic repetition

For a system where factual accuracy matters (clinical trials are medical information), keeping creativity low is the right tradeoff even at the cost of slightly drier responses.

### What the Score Threshold Means

The retrieval score is cosine similarity between the query vector and a trial's vector, on a 0–1 scale:

- **0.0–0.44:** Below threshold. No results returned. Fallback message shown. The query is semantically unrelated to any trial in the database.
- **0.45–0.74:** Normal confidence. Results are relevant enough to proceed with generation. The "Relevance score" label is shown in citations.
- **0.75+:** High confidence. The query closely matches at least one trial. The "High-confidence response" label is shown in citations.

The 0.45 threshold was set empirically. Scores below 0.45 were found to produce retrieval results that were tangentially related at best — for example, asking about "climate change" might retrieve a trial about respiratory disease at 0.3 similarity. These results would cause the LLM to generate a misleading answer. The threshold prevents this.

---

## 9. PDF Upload Feature

### How pdfplumber Extracts Text

`pdfplumber` is a Python library that extracts text from PDFs by parsing the PDF's internal content stream. PDFs store text as sequences of positioned characters — not as flowing paragraphs. pdfplumber reconstructs the character sequences into lines and pages.

```python
with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
```

`uploaded_file.read()` reads the raw PDF bytes into memory. `io.BytesIO()` wraps them as a file-like object so pdfplumber can open it without writing to disk — this is important for the session-scoped privacy model.

**What pdfplumber handles well:** Standard text PDFs — research papers, trial protocols, regulatory documents. It preserves page numbers, which are tracked and shown in citations.

**What it handles poorly:** Scanned PDFs (images of text) — these require OCR, which pdfplumber does not do. A scanned PDF returns empty text per page and the system correctly shows "Could not extract text from PDF."

### What Chunking Is and Why Overlap Matters

The full text of a PDF might be 10,000–50,000 characters — far too large to embed as one unit (the embedding model has a 256-token limit, beyond which it truncates). The text must be broken into chunks small enough to embed meaningfully.

TrialSight uses 500-character chunks with 50-character overlap:

```python
while start < len(text):
    end = min(start + chunk_size, len(text))
    chunk = text[start:end].strip()
    chunks.append({"text": chunk, "page": page, "chunk_index": idx})
    start += chunk_size - overlap  # advance by 450, not 500
```

**Why 500 characters?** Roughly 80–100 words — enough to contain a complete thought but not so long that the embedding averages over multiple unrelated topics.

**Why overlap?** Without overlap, a sentence that happens to fall on a chunk boundary would be split in half. The first half would be in chunk N and the second half in chunk N+1. Neither chunk would contain the complete meaning. The 50-character overlap means the last 50 characters of chunk N are the first 50 of chunk N+1, so boundary-crossing sentences appear fully in at least one chunk.

### Why Session-Scoped Storage

Uploaded PDF data is stored in `st.session_state`:

```python
st.session_state["pdf_embeddings"] = [(text, vector), ...]
st.session_state["pdf_chunks_meta"] = [{"text": ..., "page": ..., "chunk_index": ...}, ...]
```

`st.session_state` lives for the duration of a single browser session and is cleared when the tab closes or the browser refreshes. This is intentional for clinical data privacy: a user uploading a private trial protocol should not have their document accessible to anyone else, ever.

The alternatives were:
- **Write to Qdrant:** Would persist the document across sessions. Privacy violation.
- **Write to disk:** Same issue, plus complicates multi-user deployment.
- **Write to a database:** Same issue.

Session state is the only option that guarantees the document never persists beyond the user's current session.

### How In-Memory Search Differs from Qdrant Search

Qdrant uses an HNSW graph index for approximate nearest neighbor search. In-memory search uses brute-force cosine similarity:

```python
query_vec = np.array(model.encode(query))
for chunk_text, chunk_vec in session_embeddings:
    vec = np.array(chunk_vec)
    score = float(
        np.dot(query_vec, vec)
        / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10)
    )
    scored.append((score, idx, chunk_text))
scored.sort(reverse=True)
```

This is O(n) in the number of chunks. For a PDF of 100 pages, that might be 200–500 chunks — a brute-force search over 500 vectors takes under 1ms on any modern machine. The HNSW index would be overkill and would require loading Qdrant libraries in-session. Brute force is the right choice at this scale.

The `+ 1e-10` in the denominator prevents division by zero if either vector is all-zeros.

**When PDF is uploaded:** `retrieve_with_session_embeddings()` runs both Qdrant search and in-memory search, combining results. The generator receives both lists and assembles context from both.

**When no PDF is uploaded:** `session_embeddings` is `None`, and only Qdrant search runs. The PDF code path is completely bypassed.

---

## 10. Trustworthiness Layer

TrialSight's `docs/TRUSTWORTHINESS.md` defines quality and reliability rules that apply at every layer. This section explains what each rule does and why it exists.

### Hallucination in LLM Context

"Hallucination" refers to an LLM generating confident, fluent text that is factually false. LLMs don't "know" facts — they predict likely continuations of text. If asked about a trial that doesn't exist, a hallucinating model will invent plausible-sounding NCT IDs, sponsor names, and results. This is dangerous in a clinical context where a user might act on false information.

TrialSight prevents hallucination through three overlapping mechanisms:

1. **Retrieval gate:** The model only generates if `score >= 0.45`. Below this threshold, the fallback message is returned immediately and the model is never called. The model cannot hallucinate if it's not asked.

2. **Context-only instruction:** The system prompt says "Answer using only the provided clinical trial context." LLMs are instruction-following — they generally respect this constraint. The model is told to say "I don't have enough information" if the context is insufficient.

3. **Temperature 0.3:** Reduces the probability of creative departures from the provided context.

No single mechanism is sufficient alone. The combination creates defense in depth.

### Citation Requirement

Every response is accompanied by the raw `ScoredPoint` objects from Qdrant, shown in a "Source Trials" expander. The user can verify every NCT ID mentioned in the answer by expanding this section. This is the key accountability mechanism — the system cannot answer without providing its evidence.

### Confidence Tiers

```
score < 0.45:  ❌ Fallback — no answer generated
score 0.45–0.74: ℹ️ Normal — "Relevance score: X.XXX"
score ≥ 0.75:  ✅ High confidence — "High-confidence response"
```

The tier is shown in the UI so users understand how much to trust the answer. A score of 0.46 means the retrieved trial is only weakly related to the question — the answer might be partially relevant but should be verified. A score of 0.82 means the retrieved trial is a close semantic match.

### Pipeline Audit Trail

Every stage logs:
- Timestamp
- Input record count
- Output record count
- Number of records dropped
- Reason for drops

This logging pattern means that if something is wrong (e.g., 50 records unexpectedly dropped in Silver), you can see exactly where and why without re-running the pipeline.

Example Silver output:
```
[SILVER] Starting transformation at 20250623_143201
[SILVER] Input: 990 records
[SILVER] Dropped 2 records with null critical fields
[SILVER] Dropped 7 records with out-of-range startDate
[SILVER] Input: 990 → Output: 981 records
```

### Post-Load Verification

Both Supabase and Qdrant run a count check after loading:

```python
# Supabase
verify = client.table("fact_trials").select("trial_id", count="exact").execute()
if db_count < len(fact_df) * 0.9:
    print(f"WARNING: expected ~{len(fact_df)}, found {db_count}")

# Qdrant
info = client.get_collection(COLLECTION_NAME)
if info.points_count < len(fact_df) * 0.9:
    print(f"WARNING: expected ~{len(fact_df)}, found {db_count}")
```

The 10% tolerance accounts for recoverable partial failures. The check is informational — it logs a warning but doesn't abort the pipeline, because partial data is better than no data for the application to function.

---

## 11. Testing

### test_pipeline.py — 27 Tests

**`TestNormalizePhase` (8 tests)**  
Tests every branch of `_normalize_phase()`: standard phases, EARLY_PHASE1 collapsing to Phase 1, None → N/A, NaN → N/A, multi-phase highest-wins logic, unknown strings → N/A. Phase normalization is a pure function with well-defined inputs and outputs, making it ideal for exhaustive unit testing.

**`TestSilverLayer` (7 tests)**  
Tests Silver with a synthetic Bronze DataFrame created by `_make_bronze_df(n)`. Key tests:
- `test_drops_null_nct_id`: verifies that a row with `nctId=None` doesn't appear in Silver output
- `test_drops_out_of_range_dates`: inserts a "1999-01" date (guaranteed to be dropped) and checks the output
- `test_phase_normalised`: verifies the output phase column only contains the 5 valid labels
- `test_sponsor_filled`: verifies no null sponsors survive

The `_make_bronze_df` fixture uses modulo cycling (`phases_cycle[i % len(phases_cycle)]`) to generate phase values. This was a bug fix — an earlier version used `* (n//5+1)` which produces a list longer than `n`, causing a shape mismatch when assigned to the DataFrame column.

**`TestGoldLayer` (8 tests)**  
Tests run the full Silver + Gold pipeline on synthetic data. Key tests:
- Three referential integrity tests: checks that every `phase_id`, `condition_id`, `sponsor_id` in `fact_trials` exists in the corresponding dimension table. These mirror the assertions in the production code.
- `test_is_single_sponsor_no_nulls`: the flag must be explicitly set — never null.
- `test_is_single_sponsor_is_bool`: dtype must be Python bool, not int or object.

**`TestBronzeParquetFiles` (4 tests)**  
Integration-style tests that read the actual Bronze Parquet files from disk. These verify that `extract.py` was actually run and produced valid output. They are not unit tests in the strict sense — they require the pipeline to have been run at least once.

### test_rag.py — 19 Unit Tests + 3 Integration Tests

**`TestBuildContext` (5 tests)**  
Tests the context assembly function in isolation. Key test: `test_context_length_bounded` — injects trials with 5,000-character summaries and verifies the output is capped near `MAX_CONTEXT_CHARS` (12,000). Without this bound, a pathological input could generate a context too large for the LLM.

**`TestGenerate` (6 tests)**  
Uses `MagicMock` to simulate the Groq client. Key tests:
- `test_returns_fallback_when_no_results`: the function must return `FALLBACK_MESSAGE` without calling Groq if the retrieval list is empty.
- `test_calls_groq_with_system_prompt`: verifies the system prompt is included in the messages list (not skipped or replaced).
- `test_uses_low_temperature`: asserts `temperature <= 0.35` — a slight tolerance above 0.3 so the test doesn't fail if someone bumps it to 0.32.
- `test_returns_fallback_on_groq_error`: simulates a Groq API exception and verifies the system returns a user-friendly error message rather than crashing.

**`TestRunRAG` (8 tests)**  
End-to-end pipeline tests with all three clients mocked. The mock setup:
```python
model = MagicMock()
model.encode.return_value = np.zeros(384)

qdrant_client = MagicMock()
mock_response = MagicMock()
mock_response.points = _make_results(n_results, score)
qdrant_client.query_points.return_value = mock_response
```

Key test: `test_pdf_session_embeddings_used` — passes in synthetic session embeddings and verifies the result contains a `pdf_results` list. The cosine similarity computation runs against real numpy arrays (the mock model returns `np.ones(384) * 0.5`), so this tests the actual in-memory search logic, not just the interface.

**`TestIntegration` (3 tests, `@pytest.mark.integration`)**  
These hit live Qdrant and Groq. They're marked so they're excluded from the default test run (`pytest -m "not integration"`) and only run when explicitly requested (`pytest -m integration`). They verify:
- That a "cancer clinical trial" query returns real results with score ≥ 0.45
- That a full diabetes pipeline run produces a non-fallback answer
- That a nonsense query ("asdfqwerzxcv") correctly returns `fallback: True`

---

## 12. Known Limitations and What Would Change in Production

### Dataset Size

981 trials is sufficient for demonstrating the system but tiny compared to ClinicalTrials.gov's 500,000+ registered studies. The pipeline is designed to scale — `TARGET_PER_CONDITION` is a constant that can be increased — but at 10,000+ trials, the embedding step would take 10+ minutes and Qdrant's free tier (1GB RAM) would need to be upgraded.

**Production change:** Increase to 10,000–50,000 trials. Add incremental extraction (fetch only trials updated since last run using the `lastUpdatePostDateStruct` API field). Run embedding on a GPU-enabled machine to reduce from ~30 minutes to ~2 minutes.

### Single-Condition Assignment

The Gold layer assigns each trial one condition (the first in the semicolon-separated list). A trial for "Lung Cancer; Stage IV NSCLC" is assigned "Lung Cancer" and Stage IV NSCLC is lost. This means a user asking about "Stage IV NSCLC" might miss trials that list it as a secondary condition.

**Production change:** Normalize condition to a canonical vocabulary (e.g., MeSH terms — Medical Subject Headings, the NLM's standardized medical taxonomy). Create a many-to-many `fact_trial_conditions` bridge table instead of a single `condition_id`.

### Sponsor Type Missing

`sponsor_type` is always "Unknown" because `LeadSponsorClass` wasn't included in the initial field extraction. The API does provide this — it returns values like `INDUSTRY`, `NIH`, `NETWORK`, `OTHER_GOV`, `FED`. This would significantly improve the dashboard's sponsor analysis.

**Production change:** Add `LeadSponsorClass` to the `fields` parameter in extract.py. Map it to readable labels in Silver. Populate `sponsor_type` in the Gold dimension table.

### No Incremental Pipeline

Every pipeline run re-fetches all 981 trials, re-transforms, re-loads all rows (via upsert), and re-embeds all vectors. This is fine for a 43-second pipeline but would be unacceptable at scale.

**Production change:** Track last extraction timestamp. Fetch only studies updated since then using `filter.advanced=AREA[LastUpdatePostDate]RANGE[{last_run},{today}]`. Only upsert changed rows. Only re-embed changed trials (delete old vector, insert new one by NCT ID).

### Qdrant is Not Truly Persistent During Development

The collection `clinical_trials` on Qdrant Cloud is persistent — it survives process restarts. However, re-running the pipeline with the same NCT IDs and the same stable integer IDs correctly upserts (replaces) existing vectors. The system handles this well. The earlier bug (using Python `hash()`) would cause accumulation — each pipeline run added 981 new vectors instead of replacing existing ones, reaching 2,943 after three runs.

**Production change:** Add an explicit count verification at pipeline start — if `qdrant.count() > expected_count * 1.1`, log a warning that vectors may have accumulated.

### No Authentication on the Streamlit App

The app has no login. Anyone with the URL can see all 981 trials and use the RAG chat. For public clinical trial data, this is acceptable. For private institutional data (private trial protocols, internal research), this would be a critical gap.

**Production change:** Add Streamlit's native authentication, or put the app behind an identity provider (e.g., Auth0, Google OAuth) using a reverse proxy.

### Groq Rate Limits

Groq's free tier has rate limits (typically 30 requests/minute on the free plan). Under concurrent user load, the chat would fail with rate limit errors. The current error handling returns a fallback message rather than crashing, but users would see degraded service.

**Production change:** Implement a simple request queue or add exponential backoff with jitter. Cache common queries — a LRU cache keyed on `(query_text, top_nct_ids)` would serve repeated questions from memory without hitting Groq.

### Temperature is a Fixed Constant

Temperature is hardcoded at 0.3. Different use cases might want different temperatures — exploratory "tell me about this area of research" questions might benefit from higher creativity (0.6), while "what are the exact inclusion criteria for NCT04280705" should be as deterministic as possible (0.1).

**Production change:** Classify the query type (exact lookup vs. exploratory summary) and adjust temperature dynamically.

### No Monitoring or Alerting

The pipeline logs to stdout. If the Qdrant upload silently fails for 50 trials, you'd only know by reading the log. There's no automated alerting.

**Production change:** Emit structured JSON logs. Send pipeline completion metrics (duration, record counts, error counts) to a monitoring system (Datadog, Grafana, or even a Slack webhook). Alert if `errors > 0` or `output_count < expected * 0.95`.

---

*Report generated from codebase as of June 2026. All code references are accurate to the versions in the repository at time of writing.*
