import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    CollectionStatus,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("[EMBED] QDRANT_URL and QDRANT_API_KEY must be set in .env")

COLLECTION_NAME = "clinical_trials"
VECTOR_SIZE = 384
BATCH_SIZE = 100
GOLD_DIR = Path(__file__).parent.parent / "data" / "gold"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def _ensure_collection(client: QdrantClient) -> None:
    """Create or verify the Qdrant collection."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[EMBED] Collection '{COLLECTION_NAME}' created")
    else:
        info = client.get_collection(COLLECTION_NAME)
        if info.status != CollectionStatus.GREEN:
            print(f"[EMBED] WARNING: collection status is {info.status}")
        else:
            print(f"[EMBED] Collection '{COLLECTION_NAME}' exists and is GREEN")


def _build_text(row: pd.Series) -> str:
    """Build the text to embed for a trial record."""
    parts = [
        f"Title: {row.get('brief_title', '')}",
        f"Condition: {row.get('condition', '')}",
        f"Phase: {row.get('phase', '')}",
        f"Status: {row.get('overall_status', '')}",
        f"Summary: {row.get('brief_summary', '') or ''}",
    ]
    return " | ".join(p for p in parts if p.split(": ", 1)[1].strip())


def run_embed(gold_tables: dict[str, pd.DataFrame]) -> int:
    """Embed fact_trials and upsert to Qdrant. Returns count of vectors upserted."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[EMBED] Starting embedding at {timestamp}")

    fact_df = gold_tables["fact_trials"]
    dim_condition = gold_tables["dim_condition"]
    dim_sponsor = gold_tables["dim_sponsor"]
    dim_phase = gold_tables["dim_phase"]

    print(f"[EMBED] Input: {len(fact_df)} trials to embed")

    # Build dimension lookup maps (local Gold IDs → labels)
    phase_map = dict(zip(dim_phase["phase_id"], dim_phase["phase_label"]))
    cond_map = dict(zip(dim_condition["condition_id"], dim_condition["condition_name"]))
    sponsor_map = dict(zip(dim_sponsor["sponsor_id"], dim_sponsor["sponsor_name"]))

    # Merge dimension labels into fact table for embedding context
    df = fact_df.copy()
    df["condition"] = df["condition_id"].map(cond_map)
    df["phase"] = df["phase_id"].map(phase_map)
    df["sponsor"] = df["sponsor_id"].map(sponsor_map)

    # Load model and client
    print(f"[EMBED] Loading model {MODEL_NAME}...")
    model = _get_model()
    client = _get_client()

    # Ensure collection exists (Gold check)
    _ensure_collection(client)

    # Build texts for embedding
    texts = [_build_text(row) for _, row in df.iterrows()]

    # Batch-encode and upsert
    total_upserted = 0
    failed_ids = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]
        batch_rows = df.iloc[i: i + BATCH_SIZE]

        embeddings = model.encode(batch_texts, show_progress_bar=False)

        points = []
        for j, (_, row) in enumerate(batch_rows.iterrows()):
            summary = str(row.get("brief_summary", "") or "")
            if summary in ("nan", "None"):
                summary = ""

            # Derive a stable integer ID from the NCT number (e.g., NCT01234567 → 1234567)
            nct_str = str(row["nct_id"])
            digits = "".join(filter(str.isdigit, nct_str))
            point_id = int(digits) if digits else abs(int.from_bytes(nct_str.encode(), "big")) % (2**53)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[j].tolist(),
                    payload={
                        "nct_id": str(row["nct_id"]),
                        "brief_title": str(row.get("brief_title", "")),
                        "condition": str(row.get("condition", "")),
                        "phase": str(row.get("phase", "")),
                        "sponsor": str(row.get("sponsor", "")),
                        "brief_summary": summary[:1000],
                        "overall_status": str(row.get("overall_status", "")),
                    },
                )
            )

        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_upserted += len(points)
        except Exception as e:
            ids = [p.payload.get("nct_id", "?") for p in points]
            failed_ids.extend(ids)
            print(f"[EMBED] Batch upsert failed (batch {i//BATCH_SIZE}): {e}")

        time.sleep(0.05)

    if failed_ids:
        print(f"[EMBED] Failed NCT IDs ({len(failed_ids)}): {failed_ids[:5]}...")

    # Verify upsert count
    try:
        info = client.get_collection(COLLECTION_NAME)
        db_count = info.points_count
        print(f"[EMBED] Qdrant verification: {db_count} vectors in collection")
        if db_count < len(fact_df) * 0.9:
            print(f"[EMBED] WARNING: expected ~{len(fact_df)}, found {db_count}")
    except Exception as e:
        print(f"[EMBED] Qdrant count check failed: {e}")

    print(f"[EMBED] Input: {len(fact_df)} → Output: {total_upserted} vectors upserted")
    return total_upserted


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
    run_embed(gold)
