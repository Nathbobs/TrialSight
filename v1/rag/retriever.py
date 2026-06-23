import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("[RETRIEVER] QDRANT_URL and QDRANT_API_KEY must be set in .env")

COLLECTION_NAME = "clinical_trials"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCORE_THRESHOLD = 0.45
DEFAULT_LIMIT = 5


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client (safe for st.cache_resource)."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def get_embedding_model() -> SentenceTransformer:
    """Return the embedding model (safe for st.cache_resource)."""
    return SentenceTransformer(MODEL_NAME)


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    limit: int = DEFAULT_LIMIT,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[ScoredPoint]:
    """
    Embed query and search Qdrant for relevant clinical trials.

    Returns up to `limit` results with score >= score_threshold.
    Returns empty list if no results meet the threshold.
    """
    if not query or not query.strip():
        return []

    print(f"[RAG] Query received: {query[:50]}...")

    # Step 2 — embed query
    query_vector = model.encode(query).tolist()

    # Step 3 — Qdrant retrieval (uses query_points API in qdrant-client >= 1.9)
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results: list[ScoredPoint] = response.points
    except Exception as e:
        print(f"[RETRIEVER] Qdrant search failed: {e}")
        return []

    if results:
        print(
            f"[RAG] Retrieved {len(results)} chunks, "
            f"top score: {results[0].score:.3f}"
        )
    else:
        print("[RAG] No results above score threshold")

    return results


def retrieve_with_session_embeddings(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    session_embeddings: list[tuple[str, list[float]]] | None = None,
    limit: int = DEFAULT_LIMIT,
    score_threshold: float = SCORE_THRESHOLD,
) -> tuple[list[ScoredPoint], list[dict]]:
    """
    Retrieve from Qdrant and optionally from in-session PDF embeddings.

    Returns (qdrant_results, pdf_results) where pdf_results are dicts with
    keys: text, score, page, chunk_index.
    """
    import numpy as np

    qdrant_results = retrieve(query, model, client, limit, score_threshold)
    pdf_results: list[dict] = []

    if session_embeddings:
        query_vec = np.array(model.encode(query))

        scored = []
        for i, (chunk_text, chunk_vec) in enumerate(session_embeddings):
            vec = np.array(chunk_vec)
            score = float(
                np.dot(query_vec, vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-10)
            )
            scored.append((score, i, chunk_text))

        scored.sort(reverse=True, key=lambda x: x[0])
        for score, idx, text in scored[:limit]:
            if score >= score_threshold:
                pdf_results.append({
                    "text": text,
                    "score": score,
                    "chunk_index": idx,
                })

    return qdrant_results, pdf_results


if __name__ == "__main__":
    model = get_embedding_model()
    client = get_qdrant_client()
    results = retrieve("cancer immunotherapy clinical trials phase 3", model, client)
    for r in results:
        print(f"  [{r.score:.3f}] {r.payload.get('nct_id')} — {r.payload.get('brief_title', '')[:60]}")
