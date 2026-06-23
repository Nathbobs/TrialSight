"""
RAG Chat tab — clinical trial Q&A powered by Qdrant + Groq Llama 3.3 70B.
Supports optional PDF upload for session-scoped document search.
"""

import io
import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _load_secrets_chat() -> tuple[str, str, str]:
    """Return (QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY) from st.secrets or env."""
    try:
        qdrant_url = st.secrets["QDRANT_URL"].strip()
        qdrant_key = st.secrets["QDRANT_API_KEY"].strip()
        groq_key = st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
    return qdrant_url, qdrant_key, groq_key


@st.cache_resource
def _get_embedding_model() -> SentenceTransformer:
    from rag.retriever import get_embedding_model
    return get_embedding_model()


@st.cache_resource
def _get_qdrant_client() -> QdrantClient:
    from rag.retriever import get_qdrant_client
    return get_qdrant_client()


@st.cache_resource
def _get_groq_client() -> Groq:
    from rag.generator import get_groq_client
    return get_groq_client()


def _parse_pdf(uploaded_file) -> list[dict]:
    """Parse uploaded PDF into pages. Returns list of {page, text}."""
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def _chunk_text(text: str, page: int, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping chunks. Returns list of {text, page, chunk_index}."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "page": page, "chunk_index": idx})
            idx += 1
        start += chunk_size - overlap
    return chunks


def _embed_pdf(pages: list[dict], model: SentenceTransformer) -> list[tuple[str, list[float], int, int]]:
    """
    Embed all PDF chunks.
    Returns list of (chunk_text, embedding, page_number, chunk_index).
    """
    all_chunks = []
    for page_data in pages:
        chunks = _chunk_text(page_data["text"], page_data["page"])
        all_chunks.extend(chunks)

    if not all_chunks:
        return []

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    return [
        (c["text"], emb.tolist(), c["page"], c["chunk_index"])
        for c, emb in zip(all_chunks, embeddings)
    ]


def _handle_pdf_upload(model: SentenceTransformer) -> None:
    """Render the PDF uploader and embed uploaded document into session state."""
    st.subheader("Upload a Clinical Trial Document (Optional)")
    uploaded = st.file_uploader(
        "Upload clinical trial PDF",
        type=["pdf"],
        key="pdf_uploader",
    )
    st.caption(
        "Uploaded documents are processed in-session only and are not stored "
        "permanently. Your data is cleared when the session ends."
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("pdf_file_id") != file_id:
            with st.spinner(f"Processing {uploaded.name}..."):
                pages = _parse_pdf(uploaded)
                if not pages:
                    st.error("Could not extract text from PDF.")
                    return

                embedded = _embed_pdf(pages, model)
                st.session_state["pdf_embeddings"] = [
                    (text, vec) for text, vec, _, _ in embedded
                ]
                st.session_state["pdf_chunks_meta"] = [
                    {"text": text, "page": pg, "chunk_index": ci}
                    for text, _, pg, ci in embedded
                ]
                st.session_state["pdf_file_id"] = file_id
                st.session_state["pdf_name"] = uploaded.name

            st.success(
                f"Processed **{uploaded.name}**: "
                f"{len(pages)} pages, {len(embedded)} chunks embedded."
            )
    elif "pdf_file_id" in st.session_state:
        # File removed — clear session state
        for k in ["pdf_embeddings", "pdf_chunks_meta", "pdf_file_id", "pdf_name"]:
            st.session_state.pop(k, None)


def _render_citations(qdrant_results, pdf_results: list[dict], top_score: float) -> None:
    """Render the source trials expander below the response."""
    with st.expander("Source Trials"):
        if top_score >= 0.75:
            st.success(f"High-confidence response (top score: {top_score:.3f})")
        elif top_score >= 0.45:
            st.info(f"Relevance score: {top_score:.3f}")
        else:
            st.warning(f"Low-confidence results (top score: {top_score:.3f})")

        if qdrant_results:
            st.write("**Clinical Trials Database:**")
            for r in qdrant_results:
                p = r.payload
                st.markdown(
                    f"- **{p.get('nct_id', 'N/A')}** — {p.get('brief_title', 'N/A')[:80]}  \n"
                    f"  Score: `{r.score:.3f}` | Condition: {p.get('condition', 'N/A')[:40]} | Phase: {p.get('phase', 'N/A')}"
                )

        if pdf_results:
            pdf_name = st.session_state.get("pdf_name", "Uploaded PDF")
            st.write(f"**From {pdf_name}:**")
            for chunk in pdf_results:
                st.markdown(
                    f"- Uploaded PDF — Page {chunk.get('page', '?')}, "
                    f"Chunk {chunk.get('chunk_index', '?')}  \n"
                    f"  Score: `{chunk['score']:.3f}`"
                )


def render_chat() -> None:
    """Render the RAG chat tab."""
    st.title("TrialSight — Clinical Trials Chat")
    st.caption("Ask questions about clinical trials. Answers cite NCT IDs from the database.")

    # Load resources
    model = _get_embedding_model()
    qdrant_client = _get_qdrant_client()
    groq_client = _get_groq_client()

    # PDF upload section
    with st.sidebar:
        _handle_pdf_upload(model)
        if "pdf_name" in st.session_state:
            st.info(f"PDF active: **{st.session_state['pdf_name']}**")

    # Initialise message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Replay chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations(
                    msg["citations"]["qdrant"],
                    msg["citations"]["pdf"],
                    msg["citations"]["top_score"],
                )

    # New user input
    user_input = st.chat_input("Ask about clinical trials...")
    if not user_input:
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Run RAG pipeline
    from rag.pipeline import run_rag

    session_embeddings = st.session_state.get("pdf_embeddings") or None

    with st.chat_message("assistant"):
        with st.spinner("Searching clinical trials..."):
            result = run_rag(
                query=user_input,
                model=model,
                qdrant_client=qdrant_client,
                groq_client=groq_client,
                session_embeddings=session_embeddings,
            )

        st.markdown(result["answer"])
        _render_citations(
            result["qdrant_results"],
            result["pdf_results"],
            result["top_score"],
        )

    # Save assistant message with citation metadata
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "citations": {
            "qdrant": result["qdrant_results"],
            "pdf": result["pdf_results"],
            "top_score": result["top_score"],
        },
    })
