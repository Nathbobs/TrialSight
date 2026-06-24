import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client.http.models import ScoredPoint

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError("[GENERATOR] GROQ_API_KEY must be set in .env")

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.3
MAX_CONTEXT_CHARS = 12000  # rough upper bound to stay under 3000 tokens

SYSTEM_PROMPT = (
    "You are a clinical trial intelligence assistant. "
    "Answer questions using only the provided clinical trial context. "
    "Always cite the NCT ID of trials you reference. "
    "If the context does not contain enough information, "
    "say so honestly rather than speculating."
)

FALLBACK_MESSAGE = (
    "I couldn't find relevant clinical trials for that query. "
    "Try rephrasing or asking about a specific condition or drug."
)


def _build_context(
    qdrant_results: list[ScoredPoint],
    pdf_results: list[dict] | None = None,
) -> str:
    """Assemble retrieval context from Qdrant results and optional PDF chunks."""
    parts = []

    if qdrant_results:
        parts.append("Relevant clinical trials from the database:")
        for r in qdrant_results:
            p = r.payload
            nct_id = p.get("nct_id", "N/A")
            title = p.get("brief_title", "N/A")
            summary = p.get("brief_summary", "") or ""
            parts.append(
                f"Trial {nct_id}: {title}\nSummary: {summary[:400]}"
            )

    if pdf_results:
        parts.append("\nRelevant sections from uploaded document:")
        for i, chunk in enumerate(pdf_results):
            parts.append(
                f"[PDF Chunk {i+1}]: {chunk['text'][:400]}"
            )

    context = "\n\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]


def get_groq_client() -> Groq:
    """Return a Groq client (safe for st.cache_resource)."""
    return Groq(api_key=GROQ_API_KEY)


def generate(
    query: str,
    qdrant_results: list[ScoredPoint],
    client: Groq,
    pdf_results: list[dict] | None = None,
) -> str:
    """
    Generate a response to `query` using Groq Llama 3.3 70B.

    Returns FALLBACK_MESSAGE if retrieval results are empty.
    On Groq API failure, returns a fallback message rather than crashing.
    """
    if not qdrant_results and not pdf_results:
        return FALLBACK_MESSAGE

    context = _build_context(qdrant_results, pdf_results)
    if not context.strip():
        return FALLBACK_MESSAGE

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Question: {query}"},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        response_text = completion.choices[0].message.content or FALLBACK_MESSAGE
        return response_text
    except Exception as e:
        print(f"[GENERATOR] Groq API failed: {e}")
        return (
            "I encountered an error generating a response. "
            "Please try again in a moment."
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag.retriever import get_embedding_model, get_qdrant_client, retrieve

    model = get_embedding_model()
    qdrant_client = get_qdrant_client()
    groq_client = get_groq_client()

    query = "What are the latest cancer immunotherapy trials in Phase 3?"
    results = retrieve(query, model, qdrant_client)
    answer = generate(query, results, groq_client)
    print("\n=== Response ===")
    print(answer)
