"""
Supabase loader — Gold → PostgreSQL via supabase-py REST API.

Schema creation strategy (in order):
1. Try direct PostgreSQL via SQLAlchemy (requires SUPABASE_DB_URL env var
   or derives connection from project ref + SUPABASE_KEY as password).
2. If direct connection fails, write migration SQL to data/supabase_migration.sql
   and attempt to apply via management API (requires management PAT — will
   gracefully skip if unavailable).
3. Check whether tables exist via REST API and proceed with data loading.
4. If tables still don't exist, raise an error with clear instructions.
"""

import os
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("[LOAD] SUPABASE_URL and SUPABASE_KEY must be set in .env")

# Normalize URL — strip /rest/v1/ suffix if present
_base_url = re.sub(r"/rest/v1/?$", "", SUPABASE_URL.rstrip("/"))

# Extract project reference from project URL
_match = re.search(r"https://([a-zA-Z0-9]+)\.supabase\.co", _base_url)
if not _match:
    raise RuntimeError(f"[LOAD] Cannot parse project ref from SUPABASE_URL: {SUPABASE_URL}")
PROJECT_REF = _match.group(1)

DATA_DIR = Path(__file__).parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"

SCHEMA_SQL = """\
-- TrialSight Supabase Migration
-- Run once in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- RLS is disabled so the anon API key can read and write data.

CREATE TABLE IF NOT EXISTS dim_phase (
    phase_id SERIAL PRIMARY KEY,
    phase_label TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_condition (
    condition_id SERIAL PRIMARY KEY,
    condition_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_sponsor (
    sponsor_id SERIAL PRIMARY KEY,
    sponsor_name TEXT UNIQUE NOT NULL,
    sponsor_type TEXT
);

CREATE TABLE IF NOT EXISTS fact_trials (
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

-- Disable RLS so the anon/service-role API key can read and write
ALTER TABLE dim_phase DISABLE ROW LEVEL SECURITY;
ALTER TABLE dim_condition DISABLE ROW LEVEL SECURITY;
ALTER TABLE dim_sponsor DISABLE ROW LEVEL SECURITY;
ALTER TABLE fact_trials DISABLE ROW LEVEL SECURITY;

-- Grant full access to anon and authenticated roles
GRANT ALL ON dim_phase, dim_condition, dim_sponsor, fact_trials TO anon, authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
"""


def _write_migration_file() -> Path:
    """Write the schema SQL to data/supabase_migration.sql."""
    path = DATA_DIR / "supabase_migration.sql"
    path.write_text(SCHEMA_SQL)
    return path


def _try_sqlalchemy_schema() -> bool:
    """
    Attempt to create schema via SQLAlchemy.
    Tries SUPABASE_DB_URL env var first, then derives from project ref.
    Returns True if schema was applied successfully.
    """
    from sqlalchemy import create_engine, text

    db_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if not db_url:
        # Derive connection URL — try direct connection with API key as password
        db_url = (
            f"postgresql+psycopg2://postgres.{PROJECT_REF}:{SUPABASE_KEY}"
            f"@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
        )

    try:
        engine = create_engine(db_url, connect_args={"connect_timeout": 10}, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
            # Apply each table creation statement separately
            for stmt in SCHEMA_SQL.split(";"):
                s = stmt.strip()
                if s and not s.startswith("--"):
                    conn.execute(text(s))
        print("[LOAD] Schema created/verified via SQLAlchemy")
        return True
    except Exception as e:
        print(f"[LOAD] SQLAlchemy schema creation failed: {type(e).__name__}: {str(e)[:200]}")
        return False


def _tables_exist(client) -> bool:
    """Return True if all four tables are accessible via the REST API."""
    for table in ["dim_phase", "dim_condition", "dim_sponsor", "fact_trials"]:
        try:
            client.table(table).select("*").limit(1).execute()
        except Exception:
            return False
    return True


_RLS_HINT = (
    "\n[LOAD] RLS is blocking inserts. Run this SQL in Supabase SQL Editor:\n"
    "  ALTER TABLE dim_phase DISABLE ROW LEVEL SECURITY;\n"
    "  ALTER TABLE dim_condition DISABLE ROW LEVEL SECURITY;\n"
    "  ALTER TABLE dim_sponsor DISABLE ROW LEVEL SECURITY;\n"
    "  ALTER TABLE fact_trials DISABLE ROW LEVEL SECURITY;\n"
    "  GRANT ALL ON dim_phase, dim_condition, dim_sponsor, fact_trials TO anon, authenticated;\n"
    "  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;\n"
    "Then re-run: python run_pipeline.py --skip-extract\n"
)


def _upsert_dim(client, table: str, records: list[dict], conflict: str) -> list[dict]:
    """Upsert dimension records; return rows with server-assigned IDs."""
    if not records:
        return []
    try:
        result = client.table(table).upsert(records, on_conflict=conflict).execute()
        return result.data
    except Exception as e:
        if "row-level security" in str(e).lower() or "42501" in str(e):
            raise RuntimeError(_RLS_HINT) from e
        raise


def _upsert_facts(client, records: list[dict]) -> int:
    """Upsert fact_trials in batches of 100; return count upserted."""
    batch_size = 100
    total = 0
    failed_ids = []
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        try:
            res = client.table("fact_trials").upsert(batch, on_conflict="nct_id").execute()
            total += len(res.data)
        except Exception as e:
            ids = [r.get("nct_id", "?") for r in batch]
            failed_ids.extend(ids)
            print(f"[LOAD] Batch upsert failed (batch {i//batch_size}): {e}")
        time.sleep(0.05)
    if failed_ids:
        print(f"[LOAD] Failed NCT IDs ({len(failed_ids)}): {failed_ids[:5]}...")
    return total


def _resolve_pk(lookup: dict[str, int], key: str, fallback_key: str) -> int:
    """Return PK from lookup map; fall back to fallback_key if key missing."""
    if key in lookup:
        return lookup[key]
    if fallback_key in lookup:
        return lookup[fallback_key]
    return next(iter(lookup.values()))


def run_load(gold_tables: dict[str, pd.DataFrame]) -> None:
    """Load Gold tables into Supabase PostgreSQL."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[LOAD] Starting Supabase load at {timestamp}")

    fact_df = gold_tables["fact_trials"]
    dim_condition_df = gold_tables["dim_condition"]
    dim_sponsor_df = gold_tables["dim_sponsor"]
    dim_phase_df = gold_tables["dim_phase"]

    print(
        f"[LOAD] Input: {len(fact_df)} facts | "
        f"{len(dim_condition_df)} conditions | "
        f"{len(dim_sponsor_df)} sponsors | "
        f"{len(dim_phase_df)} phases"
    )

    # Step 1 — try SQLAlchemy DDL
    ddl_ok = _try_sqlalchemy_schema()

    # Step 2 — if DDL failed, write migration file and guide user
    if not ddl_ok:
        migration_path = _write_migration_file()
        print(
            f"\n[LOAD] Schema creation via direct connection failed.\n"
            f"[LOAD] Migration SQL written to: {migration_path}\n"
            f"[LOAD] To create tables, open Supabase → SQL Editor → paste and run that file.\n"
            f"[LOAD] Continuing — will check if tables already exist via REST API...\n"
        )

    # Step 3 — connect via supabase-py REST client
    from supabase import create_client
    client = create_client(_base_url, SUPABASE_KEY)

    if not _tables_exist(client):
        raise RuntimeError(
            "[LOAD] Tables do not exist in Supabase. "
            f"Please run the migration at data/supabase_migration.sql "
            "in the Supabase SQL Editor, then re-run the pipeline."
        )

    # Step 4 — upsert dimension tables
    phase_rows = _upsert_dim(
        client, "dim_phase",
        dim_phase_df[["phase_label"]].to_dict(orient="records"),
        "phase_label"
    )
    cond_rows = _upsert_dim(
        client, "dim_condition",
        dim_condition_df[["condition_name"]].to_dict(orient="records"),
        "condition_name"
    )
    sponsor_rows = _upsert_dim(
        client, "dim_sponsor",
        dim_sponsor_df[["sponsor_name", "sponsor_type"]].to_dict(orient="records"),
        "sponsor_name"
    )
    print(
        f"[LOAD] Dimensions: {len(phase_rows)} phases | "
        f"{len(cond_rows)} conditions | {len(sponsor_rows)} sponsors"
    )

    # Build name → server-assigned-PK maps
    phase_map: dict[str, int] = {r["phase_label"]: r["phase_id"] for r in phase_rows}
    cond_map: dict[str, int] = {r["condition_name"]: r["condition_id"] for r in cond_rows}
    sponsor_map: dict[str, int] = {r["sponsor_name"]: r["sponsor_id"] for r in sponsor_rows}

    # Build reverse lookup: local integer ID → label (from Gold dim tables)
    phase_id_to_label = dict(zip(dim_phase_df["phase_id"], dim_phase_df["phase_label"]))
    cond_id_to_name = dict(zip(dim_condition_df["condition_id"], dim_condition_df["condition_name"]))
    sponsor_id_to_name = dict(zip(dim_sponsor_df["sponsor_id"], dim_sponsor_df["sponsor_name"]))

    # Step 5 — prepare and upsert fact_trials
    fact_records = []
    for _, row in fact_df.iterrows():
        phase_label = phase_id_to_label.get(int(row["phase_id"]), "N/A")
        cond_name = cond_id_to_name.get(int(row["condition_id"]), "Unknown")
        sponsor_name = sponsor_id_to_name.get(int(row["sponsor_id"]), "Unknown")

        start = None
        if pd.notna(row["start_date"]):
            try:
                start = pd.Timestamp(row["start_date"]).date().isoformat()
            except Exception:
                pass

        end = None
        if pd.notna(row["completion_date"]):
            try:
                end = pd.Timestamp(row["completion_date"]).date().isoformat()
            except Exception:
                pass

        def _safe_text(val, max_len: int = 5000) -> str | None:
            if pd.isna(val) or str(val) in ("nan", "None", ""):
                return None
            return str(val)[:max_len]

        fact_records.append({
            "nct_id": str(row["nct_id"]),
            "brief_title": str(row["brief_title"])[:500],
            "official_title": _safe_text(row.get("official_title"), 500),
            "overall_status": _safe_text(row.get("overall_status")),
            "phase_id": _resolve_pk(phase_map, phase_label, "N/A"),
            "condition_id": _resolve_pk(cond_map, cond_name, next(iter(cond_map))),
            "sponsor_id": _resolve_pk(sponsor_map, sponsor_name, next(iter(sponsor_map))),
            "start_date": start,
            "completion_date": end,
            "brief_summary": _safe_text(row.get("brief_summary")),
            "eligibility_criteria": _safe_text(row.get("eligibility_criteria")),
            "is_single_sponsor": bool(row["is_single_sponsor"]),
        })

    loaded = _upsert_facts(client, fact_records)

    # Step 6 — verify row count
    try:
        verify = client.table("fact_trials").select("trial_id", count="exact").execute()
        db_count = verify.count or 0
        print(f"[LOAD] Verification: {db_count} total rows in fact_trials")
        if db_count < len(fact_df) * 0.9:
            print(f"[LOAD] WARNING: expected ~{len(fact_df)}, found {db_count}")
    except Exception as e:
        print(f"[LOAD] Row count verification skipped: {e}")

    print(f"[LOAD] Input: {len(fact_df)} → Output: {loaded} rows upserted to fact_trials")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.extract import BRONZE_DIR
    from pipeline.transform import run_transform

    bronze_files = sorted(BRONZE_DIR.glob("*.parquet"))
    if not bronze_files:
        raise FileNotFoundError("No Bronze Parquet files found. Run extract.py first.")
    df_bronze = pd.read_parquet(bronze_files[-1])
    gold = run_transform(df_bronze)
    run_load(gold)
