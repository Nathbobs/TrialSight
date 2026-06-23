"""
Unit and integration tests for the RAG pipeline.
Unit tests use mocks. Integration tests hit live Qdrant and Groq.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from rag.generator import (
    _build_context,
    generate,
    FALLBACK_MESSAGE,
    SYSTEM_PROMPT,
)
from rag.pipeline import run_rag, HIGH_CONFIDENCE_THRESHOLD


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_scored_point(nct_id: str, title: str, summary: str, score: float):
    """Create a mock ScoredPoint-like object."""
    point = MagicMock()
    point.score = score
    point.payload = {
        "nct_id": nct_id,
        "brief_title": title,
        "condition": "Cancer",
        "phase": "Phase 2",
        "sponsor": "Test Corp",
        "brief_summary": summary,
        "overall_status": "RECRUITING",
    }
    return point


def _make_results(n: int = 3, score: float = 0.6) -> list:
    return [
        _make_scored_point(f"NCT{i:08d}", f"Trial {i}", f"Summary {i}", score)
        for i in range(n)
    ]


# ─── Generator tests ──────────────────────────────────────────────────────────

class TestBuildContext:
    def test_returns_string(self):
        results = _make_results(2)
        ctx = _build_context(results)
        assert isinstance(ctx, str)

    def test_contains_nct_ids(self):
        results = _make_results(2)
        ctx = _build_context(results)
        assert "NCT00000000" in ctx
        assert "NCT00000001" in ctx

    def test_empty_results_returns_empty_or_fallback(self):
        ctx = _build_context([])
        assert ctx.strip() == "" or "database" in ctx.lower()

    def test_pdf_results_included(self):
        results = _make_results(1)
        pdf = [{"text": "important drug info", "score": 0.7, "chunk_index": 0}]
        ctx = _build_context(results, pdf_results=pdf)
        assert "important drug info" in ctx

    def test_context_length_bounded(self):
        # Large summaries should be truncated
        results = _make_results(5, score=0.6)
        for r in results:
            r.payload["brief_summary"] = "x" * 5000
        ctx = _build_context(results)
        assert len(ctx) <= 13000  # slightly above MAX_CONTEXT_CHARS for tolerance


class TestGenerate:
    def test_returns_fallback_when_no_results(self):
        client = MagicMock()
        result = generate("any query", [], client)
        assert result == FALLBACK_MESSAGE

    def test_returns_fallback_when_pdf_and_qdrant_empty(self):
        client = MagicMock()
        result = generate("any query", [], client, pdf_results=[])
        assert result == FALLBACK_MESSAGE

    def test_calls_groq_with_system_prompt(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test answer"
        client.chat.completions.create.return_value = mock_response

        results = _make_results(2)
        answer = generate("What trials exist?", results, client)

        assert answer == "Test answer"
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[1]
        assert any(m["content"] == SYSTEM_PROMPT for m in messages)

    def test_returns_fallback_on_groq_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API Error")
        results = _make_results(2)
        answer = generate("What trials exist?", results, client)
        assert "error" in answer.lower() or "again" in answer.lower()

    def test_uses_correct_model(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_response

        results = _make_results(1)
        generate("test", results, client)

        call_args = client.chat.completions.create.call_args
        model = call_args.kwargs.get("model") or call_args.args[0]
        assert model == "llama-3.3-70b-versatile"

    def test_uses_low_temperature(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        client.chat.completions.create.return_value = mock_response

        results = _make_results(1)
        generate("test", results, client)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature", 1.0) <= 0.35


# ─── RAG pipeline tests ───────────────────────────────────────────────────────

class TestRunRAG:
    def _make_mocks(self, qdrant_score: float = 0.6, n_results: int = 3):
        model = MagicMock()
        model.encode.return_value = np.zeros(384)

        qdrant_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = _make_results(n_results, qdrant_score)
        qdrant_client.query_points.return_value = mock_response

        groq_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Generated answer"
        groq_client.chat.completions.create.return_value = mock_completion

        return model, qdrant_client, groq_client

    def test_returns_dict_with_required_keys(self):
        model, qdrant, groq = self._make_mocks()
        result = run_rag("cancer trials", model, qdrant, groq)
        for key in ["answer", "qdrant_results", "pdf_results", "top_score", "is_high_confidence", "fallback"]:
            assert key in result, f"Missing key: {key}"

    def test_empty_query_returns_fallback(self):
        model, qdrant, groq = self._make_mocks()
        result = run_rag("", model, qdrant, groq)
        assert result["fallback"] is True

    def test_whitespace_only_query_returns_fallback(self):
        model, qdrant, groq = self._make_mocks()
        result = run_rag("   ", model, qdrant, groq)
        assert result["fallback"] is True

    def test_no_results_returns_fallback(self):
        model, qdrant, groq = self._make_mocks(n_results=0)
        mock_response = MagicMock()
        mock_response.points = []
        qdrant.query_points.return_value = mock_response
        result = run_rag("xyzzy", model, qdrant, groq)
        assert result["fallback"] is True
        assert result["answer"] == FALLBACK_MESSAGE

    def test_high_score_sets_high_confidence(self):
        model, qdrant, groq = self._make_mocks(qdrant_score=0.8)
        result = run_rag("test query", model, qdrant, groq)
        assert result["is_high_confidence"] is True
        assert result["top_score"] >= HIGH_CONFIDENCE_THRESHOLD

    def test_low_score_not_high_confidence(self):
        model, qdrant, groq = self._make_mocks(qdrant_score=0.5)
        result = run_rag("test query", model, qdrant, groq)
        assert result["is_high_confidence"] is False

    def test_answer_is_string(self):
        model, qdrant, groq = self._make_mocks()
        result = run_rag("cancer phase 2", model, qdrant, groq)
        assert isinstance(result["answer"], str)

    def test_pdf_session_embeddings_used(self):
        model, qdrant, groq = self._make_mocks()
        model.encode.return_value = np.ones(384) * 0.5

        session_embs = [
            ("relevant diabetes text", (np.ones(384) * 0.5).tolist()),
            ("unrelated content", (np.zeros(384)).tolist()),
        ]
        result = run_rag("diabetes", model, qdrant, groq, session_embeddings=session_embs)
        assert isinstance(result["pdf_results"], list)


# ─── Integration tests (live Qdrant + Groq) ───────────────────────────────────

@pytest.mark.integration
class TestIntegration:
    """These tests call live services. Mark with -m integration to run."""

    @pytest.fixture(scope="class")
    def live_clients(self):
        from rag.retriever import get_embedding_model, get_qdrant_client
        from rag.generator import get_groq_client

        return {
            "model": get_embedding_model(),
            "qdrant": get_qdrant_client(),
            "groq": get_groq_client(),
        }

    def test_retrieve_cancer_trials(self, live_clients):
        from rag.retriever import retrieve
        results = retrieve(
            "cancer clinical trial",
            live_clients["model"],
            live_clients["qdrant"],
        )
        assert len(results) > 0
        assert results[0].score >= 0.45

    def test_full_pipeline_diabetes(self, live_clients):
        result = run_rag(
            "What diabetes trials are recruiting?",
            live_clients["model"],
            live_clients["qdrant"],
            live_clients["groq"],
        )
        assert not result["fallback"]
        assert result["top_score"] > 0
        assert len(result["answer"]) > 20

    def test_pipeline_fallback_for_nonsense(self, live_clients):
        result = run_rag(
            "asdfqwerzxcv not a real thing",
            live_clients["model"],
            live_clients["qdrant"],
            live_clients["groq"],
        )
        assert result["fallback"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
