"""
End-to-end RAG pipeline.

Steps (per AGENTS.md):
  1. Query intake and validation
  2. Query embedding
  3. Qdrant retrieval (+ optional session PDF retrieval)
  4. Context assembly
  5. Groq LLM generation
  6. Response + citations (returned as structured dict for UI layer)
"""

from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

from rag.retriever import (
    retrieve_with_session_embeddings,
    SCORE_THRESHOLD,
)
from rag.generator import generate, FALLBACK_MESSAGE

HIGH_CONFIDENCE_THRESHOLD = 0.75


def run_rag(
    query: str,
    model: SentenceTransformer,
    qdrant_client: QdrantClient,
    groq_client: Groq,
    session_embeddings: list[tuple[str, list[float]]] | None = None,
    limit: int = 5,
) -> dict:
    """
    Run the full RAG pipeline for a user query.

    Returns a dict with keys:
      - answer: str — the LLM response
      - qdrant_results: list[ScoredPoint] — retrieved Qdrant trials
      - pdf_results: list[dict] — retrieved PDF chunks (may be empty)
      - top_score: float — highest retrieval score (0 if no results)
      - is_high_confidence: bool — True if top score >= 0.75
      - fallback: bool — True if no results found
    """
    # Step 1 — Query intake
    if not query or not query.strip():
        return {
            "answer": "Please enter a question.",
            "qdrant_results": [],
            "pdf_results": [],
            "top_score": 0.0,
            "is_high_confidence": False,
            "fallback": True,
        }

    # Steps 2 & 3 — Embed + retrieve
    qdrant_results, pdf_results = retrieve_with_session_embeddings(
        query=query,
        model=model,
        client=qdrant_client,
        session_embeddings=session_embeddings,
        limit=limit,
        score_threshold=SCORE_THRESHOLD,
    )

    # Determine confidence
    all_scores = [r.score for r in qdrant_results] + [r["score"] for r in pdf_results]
    top_score = max(all_scores) if all_scores else 0.0
    is_fallback = len(qdrant_results) == 0 and len(pdf_results) == 0

    if is_fallback:
        return {
            "answer": FALLBACK_MESSAGE,
            "qdrant_results": [],
            "pdf_results": [],
            "top_score": 0.0,
            "is_high_confidence": False,
            "fallback": True,
        }

    # Steps 4 & 5 — Assemble context + generate
    answer = generate(
        query=query,
        qdrant_results=qdrant_results,
        client=groq_client,
        pdf_results=pdf_results if pdf_results else None,
    )

    return {
        "answer": answer,
        "qdrant_results": qdrant_results,
        "pdf_results": pdf_results,
        "top_score": top_score,
        "is_high_confidence": top_score >= HIGH_CONFIDENCE_THRESHOLD,
        "fallback": False,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag.retriever import get_embedding_model, get_qdrant_client
    from rag.generator import get_groq_client

    model = get_embedding_model()
    qdrant = get_qdrant_client()
    groq = get_groq_client()

    test_queries = [
        "What diabetes trials are in Phase 2?",
        "Tell me about Alzheimer's prevention studies.",
        "xyzzy not a real condition",
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        result = run_rag(q, model, qdrant, groq)
        print(f"Top score: {result['top_score']:.3f} | High-conf: {result['is_high_confidence']} | Fallback: {result['fallback']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {[r.payload.get('nct_id') for r in result['qdrant_results']]}")
