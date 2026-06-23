import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BRONZE_DIR = Path(__file__).parent.parent / "data" / "bronze"
SILVER_DIR = Path(__file__).parent.parent / "data" / "silver"
GOLD_DIR = Path(__file__).parent.parent / "data" / "gold"

SILVER_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

PHASE_MAP = {
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "EARLY_PHASE1": "Phase 1",
    "NA": "N/A",
    "N/A": "N/A",
}


def _normalize_phase(phase_str) -> str:
    """Map raw phase strings to standard labels."""
    if pd.isna(phase_str) or phase_str is None:
        return "N/A"
    # Handle semicolon-separated phases (e.g., "PHASE1; PHASE2")
    parts = [p.strip().upper().replace(" ", "_") for p in str(phase_str).split(";")]
    normalized = [PHASE_MAP.get(p, "N/A") for p in parts]
    # Return highest phase if multiple
    for label in ["Phase 4", "Phase 3", "Phase 2", "Phase 1"]:
        if label in normalized:
            return label
    return "N/A"


def run_silver(df_bronze: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter Bronze data into Silver layer."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[SILVER] Starting transformation at {timestamp}")
    print(f"[SILVER] Input: {len(df_bronze)} records")

    df = df_bronze.copy()

    # Drop nulls on critical fields
    critical = ["nctId", "briefTitle", "condition"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped_nulls = before - len(df)
    if dropped_nulls:
        print(f"[SILVER] Dropped {dropped_nulls} records with null critical fields")

    # Normalize dates
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["completionDate"] = pd.to_datetime(df["completionDate"], errors="coerce")

    # Date range validation: startDate must be between 2000-2026
    before = len(df)
    valid_start = df["startDate"].isna() | (
        (df["startDate"].dt.year >= 2000) & (df["startDate"].dt.year <= 2026)
    )
    df = df[valid_start]
    dropped_date = before - len(df)
    if dropped_date:
        print(f"[SILVER] Dropped {dropped_date} records with out-of-range startDate")

    # Normalize phase labels
    df["phase"] = df["phase"].apply(_normalize_phase)

    # Clean text fields — strip whitespace
    for col in ["briefTitle", "officialTitle", "briefSummary", "eligibilityCriteria"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)

    # Ensure sponsorName is a string (not null will be handled at Gold)
    df["sponsorName"] = df["sponsorName"].fillna("Unknown")

    output_path = SILVER_DIR / f"trials_silver_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)

    print(f"[SILVER] Input: {len(df_bronze)} → Output: {len(df)} records")
    print(f"[SILVER] Saved to {output_path}")
    return df


def run_gold(df_silver: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build star schema Gold layer from Silver data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[GOLD] Starting star schema build at {timestamp}")
    print(f"[GOLD] Input: {len(df_silver)} records")

    df = df_silver.copy()

    # === dim_condition ===
    # Use the first condition listed (primary condition)
    df["primary_condition"] = df["condition"].apply(
        lambda x: str(x).split(";")[0].strip() if pd.notna(x) else "Unknown"
    )
    conditions = df["primary_condition"].unique()
    dim_condition = pd.DataFrame({
        "condition_id": range(1, len(conditions) + 1),
        "condition_name": conditions,
    })

    # === dim_sponsor ===
    sponsors = df["sponsorName"].unique()
    dim_sponsor = pd.DataFrame({
        "sponsor_id": range(1, len(sponsors) + 1),
        "sponsor_name": sponsors,
        "sponsor_type": "Unknown",  # API v2 doesn't reliably expose sponsor type
    })

    # === dim_phase ===
    phases = df["phase"].unique()
    dim_phase = pd.DataFrame({
        "phase_id": range(1, len(phases) + 1),
        "phase_label": phases,
    })

    # === fact_trials ===
    cond_map = dict(zip(dim_condition["condition_name"], dim_condition["condition_id"]))
    sponsor_map = dict(zip(dim_sponsor["sponsor_name"], dim_sponsor["sponsor_id"]))
    phase_map = dict(zip(dim_phase["phase_label"], dim_phase["phase_id"]))

    # Count trials per sponsor to flag single-sponsor trials
    sponsor_counts = df["sponsorName"].value_counts()

    fact_trials = pd.DataFrame({
        "nct_id": df["nctId"].values,
        "brief_title": df["briefTitle"].values,
        "official_title": df["officialTitle"].values,
        "overall_status": df["overallStatus"].values,
        "phase_id": df["phase"].map(phase_map).values,
        "condition_id": df["primary_condition"].map(cond_map).values,
        "sponsor_id": df["sponsorName"].map(sponsor_map).values,
        "start_date": df["startDate"].values,
        "completion_date": df["completionDate"].values,
        "brief_summary": df["briefSummary"].values,
        "eligibility_criteria": df["eligibilityCriteria"].values,
        "is_single_sponsor": df["sponsorName"].map(lambda s: bool(sponsor_counts.get(s, 0) == 1)).values,
    })

    # Referential integrity check
    assert fact_trials["phase_id"].notna().all(), "[GOLD] phase_id has nulls"
    assert fact_trials["condition_id"].notna().all(), "[GOLD] condition_id has nulls"
    assert fact_trials["sponsor_id"].notna().all(), "[GOLD] sponsor_id has nulls"

    # No duplicate nctId
    before = len(fact_trials)
    fact_trials = fact_trials.drop_duplicates(subset=["nct_id"])
    if len(fact_trials) < before:
        print(f"[GOLD] Deduplicated {before - len(fact_trials)} nctIds")

    # is_single_sponsor must never be NULL
    fact_trials["is_single_sponsor"] = fact_trials["is_single_sponsor"].fillna(False).astype(bool)

    # Save all tables
    fact_trials.to_parquet(GOLD_DIR / f"fact_trials_{timestamp}.parquet", index=False)
    dim_condition.to_parquet(GOLD_DIR / f"dim_condition_{timestamp}.parquet", index=False)
    dim_sponsor.to_parquet(GOLD_DIR / f"dim_sponsor_{timestamp}.parquet", index=False)
    dim_phase.to_parquet(GOLD_DIR / f"dim_phase_{timestamp}.parquet", index=False)

    print(f"[GOLD] Input: {len(df_silver)} → Output: {len(fact_trials)} fact records")
    print(f"[GOLD] Dimensions: {len(dim_condition)} conditions, {len(dim_sponsor)} sponsors, {len(dim_phase)} phases")
    print(f"[GOLD] Saved to {GOLD_DIR}/")

    return {
        "fact_trials": fact_trials,
        "dim_condition": dim_condition,
        "dim_sponsor": dim_sponsor,
        "dim_phase": dim_phase,
    }


def run_transform(df_bronze: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run Silver then Gold transformation. Returns Gold tables."""
    df_silver = run_silver(df_bronze)
    gold_tables = run_gold(df_silver)
    return gold_tables


if __name__ == "__main__":
    # Load the latest bronze file
    bronze_files = sorted(BRONZE_DIR.glob("*.parquet"))
    if not bronze_files:
        raise FileNotFoundError("No Bronze Parquet files found. Run extract.py first.")
    df_bronze = pd.read_parquet(bronze_files[-1])
    gold = run_transform(df_bronze)
    print("\nFact trials sample:")
    print(gold["fact_trials"].head())
