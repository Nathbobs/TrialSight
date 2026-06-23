"""
run_pipeline.py — TrialSight end-to-end ETL + embedding pipeline.

Runs the full sequence:
  1. Bronze extraction from ClinicalTrials.gov
  2. Silver + Gold transformation (Medallion Architecture)
  3. Supabase load (requires tables created via data/supabase_migration.sql)
  4. Qdrant embedding

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-extract   # reuse latest Bronze files
    python run_pipeline.py --skip-load      # skip Supabase load
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="TrialSight pipeline runner")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip Bronze extraction and reuse the latest Bronze Parquet file",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip Supabase load step",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip Qdrant embedding step",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    print("=" * 60)
    print(f"TrialSight Pipeline — started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    import pandas as pd
    from pipeline.extract import BRONZE_DIR, run_bronze
    from pipeline.transform import run_transform
    from pipeline.load import run_load, _write_migration_file
    from pipeline.embed import run_embed

    # ─── Step 1: Bronze ─────────────────────────────────────────────
    if args.skip_extract:
        bronze_files = sorted(BRONZE_DIR.glob("*.parquet"))
        if not bronze_files:
            print("[PIPELINE] No Bronze files found. Cannot skip extraction.")
            sys.exit(1)
        df_bronze = pd.read_parquet(bronze_files[-1])
        print(f"\n[PIPELINE] Reusing Bronze file: {bronze_files[-1].name} ({len(df_bronze):,} records)")
    else:
        df_bronze = run_bronze()

    # ─── Step 2: Silver + Gold ───────────────────────────────────────
    gold_tables = run_transform(df_bronze)

    # ─── Step 3: Supabase load ───────────────────────────────────────
    if not args.skip_load:
        # Write migration file in case tables don't exist yet
        migration_path = _write_migration_file()
        print(f"\n[PIPELINE] Migration SQL available at: {migration_path}")

        try:
            run_load(gold_tables)
        except RuntimeError as e:
            err = str(e)
            if "do not exist" in err.lower() or "schema cache" in err.lower():
                print(
                    f"\n[PIPELINE] Supabase tables not found. "
                    f"Please run the migration SQL in Supabase SQL Editor:\n"
                    f"  Dashboard → SQL Editor → New query → paste {migration_path}\n"
                    f"Then re-run: python run_pipeline.py --skip-extract\n"
                )
            elif "row-level security" in err.lower() or "RLS" in err:
                print(err)  # already contains the fix instructions
            else:
                print(f"\n[PIPELINE] Supabase load failed: {e}")
            print("[PIPELINE] Continuing to embedding step...")
    else:
        print("\n[PIPELINE] Skipping Supabase load (--skip-load)")

    # ─── Step 4: Qdrant embedding ────────────────────────────────────
    if not args.skip_embed:
        run_embed(gold_tables)
    else:
        print("\n[PIPELINE] Skipping Qdrant embedding (--skip-embed)")

    # ─── Summary ─────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(f"TrialSight Pipeline — completed in {elapsed:.1f}s")
    print(f"  Bronze: {len(df_bronze):,} raw records")
    print(f"  Gold:   {len(gold_tables['fact_trials']):,} clean trials")
    print("=" * 60)


if __name__ == "__main__":
    main()
