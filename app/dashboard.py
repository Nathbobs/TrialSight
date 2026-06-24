"""
Dashboard tab — shows aggregate metrics and charts from Supabase Gold tables.
Falls back gracefully to Gold Parquet files if Supabase is unavailable.
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path

GOLD_DIR = Path(__file__).parent.parent / "data" / "gold"


def _load_secrets() -> tuple[str, str]:
    """Return (SUPABASE_URL, SUPABASE_KEY) from st.secrets or env."""
    try:
        url = st.secrets["SUPABASE_URL"].strip()
        key = st.secrets["SUPABASE_KEY"].strip()
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
    return url, key


@st.cache_data(ttl=300)
def _load_from_supabase() -> tuple[pd.DataFrame | None, str | None]:
    """
    Load fact_trials joined with dimensions from Supabase.
    Returns (DataFrame, error_message). If error, DataFrame is None.
    """
    import re
    url, key = _load_secrets()
    if not url or not key:
        return None, "Supabase credentials not configured."

    base_url = re.sub(r"/rest/v1/?$", "", url.rstrip("/"))

    try:
        from supabase import create_client
        client = create_client(base_url, key)

        result = (
            client.table("fact_trials")
            .select(
                "nct_id, brief_title, overall_status, start_date, completion_date,"
                "is_single_sponsor,"
                "dim_phase(phase_label),"
                "dim_condition(condition_name),"
                "dim_sponsor(sponsor_name)"
            )
            .limit(2000)
            .execute()
        )

        if not result.data:
            return None, "No data returned from Supabase."

        rows = []
        for r in result.data:
            rows.append({
                "nct_id": r.get("nct_id"),
                "brief_title": r.get("brief_title"),
                "overall_status": r.get("overall_status"),
                "start_date": r.get("start_date"),
                "completion_date": r.get("completion_date"),
                "is_single_sponsor": r.get("is_single_sponsor"),
                "phase": (r.get("dim_phase") or {}).get("phase_label", "N/A"),
                "condition": (r.get("dim_condition") or {}).get("condition_name", "Unknown"),
                "sponsor": (r.get("dim_sponsor") or {}).get("sponsor_name", "Unknown"),
            })

        df = pd.DataFrame(rows)
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        return df, None

    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600)
def _load_from_parquet() -> pd.DataFrame | None:
    """Load Gold parquet files as fallback."""
    fact_files = sorted(GOLD_DIR.glob("fact_trials_*.parquet"))
    cond_files = sorted(GOLD_DIR.glob("dim_condition_*.parquet"))
    phase_files = sorted(GOLD_DIR.glob("dim_phase_*.parquet"))
    sponsor_files = sorted(GOLD_DIR.glob("dim_sponsor_*.parquet"))

    if not fact_files:
        return None

    fact = pd.read_parquet(fact_files[-1])
    cond = pd.read_parquet(cond_files[-1]) if cond_files else pd.DataFrame()
    phase = pd.read_parquet(phase_files[-1]) if phase_files else pd.DataFrame()
    sponsor = pd.read_parquet(sponsor_files[-1]) if sponsor_files else pd.DataFrame()

    if not cond.empty:
        fact = fact.merge(cond, on="condition_id", how="left")
    if not phase.empty:
        fact = fact.merge(phase, on="phase_id", how="left")
    if not sponsor.empty:
        fact = fact.merge(sponsor, on="sponsor_id", how="left")

    fact = fact.rename(columns={
        "condition_name": "condition",
        "phase_label": "phase",
        "sponsor_name": "sponsor",
    })
    return fact


def render_dashboard() -> None:
    """Render the dashboard tab in the Streamlit app."""
    st.title("TrialSight — Clinical Trials Dashboard")
    st.caption("Data sourced from ClinicalTrials.gov via the TrialSight ETL pipeline")

    # Load data
    with st.spinner("Loading trial data..."):
        df, error = _load_from_supabase()
        source_label = "Supabase"

        if df is None:
            df = _load_from_parquet()
            source_label = "Local Gold (Parquet)"
            if error:
                st.warning(
                    f"Supabase unavailable ({error[:80]}). "
                    "Showing data from local Gold files."
                )

    if df is None or df.empty:
        st.error(
            "No trial data available. "
            "Run the pipeline first: `python run_pipeline.py`"
        )
        return

    st.success(f"Loaded **{len(df):,}** trials from {source_label}")

    # ─── Top-line metrics ────────────────────────────────────────────
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trials", f"{len(df):,}")
    c2.metric("Unique Conditions", f"{df['condition'].nunique():,}" if "condition" in df else "N/A")
    c3.metric("Unique Sponsors", f"{df['sponsor'].nunique():,}" if "sponsor" in df else "N/A")

    if "is_single_sponsor" in df.columns:
        single_pct = df["is_single_sponsor"].mean() * 100
        c4.metric("Single-Sponsor Trials", f"{single_pct:.1f}%")

    st.divider()

    # ─── Status breakdown ─────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Trial Status")
        if "overall_status" in df.columns:
            status_counts = (
                df["overall_status"]
                .value_counts()
                .reset_index()
                .rename(columns={"overall_status": "Status", "count": "Count"})
            )
            st.bar_chart(status_counts.set_index("Status")["Count"])

    with col_r:
        st.subheader("Phase Distribution")
        if "phase" in df.columns:
            phase_counts = (
                df["phase"]
                .value_counts()
                .reset_index()
                .rename(columns={"phase": "Phase", "count": "Count"})
            )
            st.bar_chart(phase_counts.set_index("Phase")["Count"])

    st.divider()

    # ─── Top conditions ───────────────────────────────────────────────
    st.subheader("Top 15 Conditions")
    if "condition" in df.columns:
        top_conditions = (
            df["condition"]
            .value_counts()
            .head(15)
            .reset_index()
            .rename(columns={"condition": "Condition", "count": "Count"})
        )
        st.bar_chart(top_conditions.set_index("Condition")["Count"])

    st.divider()

    # ─── Top sponsors ─────────────────────────────────────────────────
    st.subheader("Top 10 Sponsors")
    if "sponsor" in df.columns:
        top_sponsors = (
            df["sponsor"]
            .value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"sponsor": "Sponsor", "count": "Count"})
        )
        st.bar_chart(top_sponsors.set_index("Sponsor")["Count"])

    st.divider()

    # ─── Trial starts over time ────────────────────────────────────────
    st.subheader("Trial Starts per Year")
    if "start_date" in df.columns:
        start_data = df.copy()
        start_data["start_date"] = pd.to_datetime(start_data["start_date"], errors="coerce")
        yearly = (
            start_data.dropna(subset=["start_date"])
            .assign(year=lambda x: x["start_date"].dt.year)
            .query("year >= 2000 and year <= 2026")
            .groupby("year")
            .size()
            .reset_index(name="count")
        )
        if not yearly.empty:
            st.line_chart(yearly.set_index("year")["count"])

    st.divider()

    # ─── Searchable trial table ────────────────────────────────────────
    st.subheader("Browse Trials")

    search = st.text_input("Search by title or NCT ID", placeholder="e.g. NCT01234567 or 'diabetes'")
    filter_cols = ["condition", "phase", "overall_status"]
    available = [c for c in filter_cols if c in df.columns]

    filter_vals = {}
    if available:
        fcols = st.columns(len(available))
        for i, col in enumerate(available):
            unique_vals = sorted(df[col].dropna().unique())
            filter_vals[col] = fcols[i].selectbox(
                f"Filter by {col.replace('_', ' ').title()}",
                ["All"] + list(unique_vals),
                key=f"filter_{col}",
            )

    display_df = df.copy()
    if search:
        mask = (
            display_df["brief_title"].str.contains(search, case=False, na=False)
            | display_df["nct_id"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    for col, val in filter_vals.items():
        if val != "All":
            display_df = display_df[display_df[col] == val]

    show_cols = [c for c in ["nct_id", "brief_title", "overall_status", "phase", "condition", "sponsor", "start_date"] if c in display_df.columns]
    st.dataframe(display_df[show_cols].reset_index(drop=True), use_container_width=True)
    st.caption(f"Showing {len(display_df):,} of {len(df):,} trials")
