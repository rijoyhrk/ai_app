"""
Tests for the LLM reranker.
Claude API is mocked — verifies scoring, filtering, and merging logic.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.reranker import rerank, _get_code
from conftest import make_claude_rerank_response


def _make_hit(chunk_id, hybrid_score=0.7, file_path="test.sql",
              file_type="sql", line_start=1, line_end=10,
              embedding_text="[CODE]\nSELECT * FROM cust_tab\n[ENTITIES]\nTables: customer"):
    return {
        "chunk_id": chunk_id,
        "hybrid_score": hybrid_score,
        "embedding_text": embedding_text,
        "metadata": {
            "file_path": file_path,
            "file_type": file_type,
            "line_start": line_start,
            "line_end": line_end,
        },
    }


class TestRerank:

    def test_returns_list(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 0.9, "operation": "READ",
             "explanation": "Reads from customer via alias.", "matched_via": "alias"},
        ])
        hits = [_make_hit("c1")]
        results = rerank(mock_anthropic_client, "customer", hits)
        assert isinstance(results, list)

    def test_final_score_is_weighted_combination(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 1.0, "operation": "READ",
             "explanation": "Direct match.", "matched_via": "direct"},
        ])
        hits = [_make_hit("c1", hybrid_score=0.5)]
        results = rerank(mock_anthropic_client, "customer", hits)
        assert len(results) == 1
        expected = round(0.4 * 0.5 + 0.6 * 1.0, 4)
        assert results[0]["final_score"] == pytest.approx(expected, abs=0.001)

    def test_filters_low_score_hits(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 0.10, "operation": "READ",
             "explanation": "Barely relevant.", "matched_via": "direct"},
            {"chunk_id": "c2", "relevance_score": 0.85, "operation": "READ",
             "explanation": "Strong match.", "matched_via": "direct"},
        ])
        hits = [_make_hit("c1"), _make_hit("c2")]
        results = rerank(mock_anthropic_client, "customer", hits, min_score=0.25)
        ids = [r["chunk_id"] for r in results]
        assert "c1" not in ids
        assert "c2" in ids

    def test_sorted_by_final_score_descending(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 0.5, "operation": "READ",
             "explanation": "ok", "matched_via": "direct"},
            {"chunk_id": "c2", "relevance_score": 0.9, "operation": "READ",
             "explanation": "great", "matched_via": "direct"},
        ])
        hits = [_make_hit("c1", hybrid_score=0.5), _make_hit("c2", hybrid_score=0.8)]
        results = rerank(mock_anthropic_client, "customer", hits)
        assert results[0]["chunk_id"] == "c2"
        assert results[1]["chunk_id"] == "c1"

    def test_operation_attached_to_result(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 0.8, "operation": "WRITE",
             "explanation": "Inserts into customer.", "matched_via": "direct"},
        ])
        results = rerank(mock_anthropic_client, "customer", [_make_hit("c1")])
        assert results[0]["operation"] == "WRITE"

    def test_explanation_attached_to_result(self, mock_anthropic_client):
        explanation = "Uses cust_tab alias to read customer data."
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 0.8, "operation": "READ",
             "explanation": explanation, "matched_via": "alias"},
        ])
        results = rerank(mock_anthropic_client, "customer", [_make_hit("c1")])
        assert results[0]["explanation"] == explanation

    def test_matched_via_attached(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "c1", "relevance_score": 0.8, "operation": "READ",
             "explanation": "Via alias.", "matched_via": "alias"},
        ])
        results = rerank(mock_anthropic_client, "customer", [_make_hit("c1")])
        assert results[0]["matched_via"] == "alias"

    def test_empty_hits_returns_empty(self, mock_anthropic_client):
        results = rerank(mock_anthropic_client, "customer", [])
        assert results == []
        mock_anthropic_client.messages.create.assert_not_called()

    def test_handles_malformed_json_response(self, mock_anthropic_client):
        msg = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = "not json"
        msg.content = [block]
        mock_anthropic_client.messages.create.return_value = msg

        hits = [_make_hit("c1", hybrid_score=0.7)]
        results = rerank(mock_anthropic_client, "customer", hits)
        # No results survive since LLM response is unparseable (score stays 0.0, filtered)
        assert isinstance(results, list)

    def test_handles_markdown_fenced_rerank_response(self, mock_anthropic_client):
        import json
        raw = [{"chunk_id": "c1", "relevance_score": 0.85, "operation": "READ",
                "explanation": "Reads customer.", "matched_via": "direct"}]
        msg = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = f"```json\n{json.dumps(raw)}\n```"
        msg.content = [block]
        mock_anthropic_client.messages.create.return_value = msg

        results = rerank(mock_anthropic_client, "customer", [_make_hit("c1")])
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_caps_at_15_snippets(self, mock_anthropic_client):
        """Reranker should only send max 15 snippets to Claude regardless of input size."""
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([])
        # pass 25 hits; reranker should only forward 15 to the LLM
        hits = [_make_hit(f"c{i}", hybrid_score=0.7) for i in range(25)]
        rerank(mock_anthropic_client, "customer", hits)
        # verify Claude was called exactly once (batched)
        assert mock_anthropic_client.messages.create.call_count == 1
        # inspect the user message content for snippet count
        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        # the reranker embeds snippets as JSON in the prompt; each snippet has one "chunk_id"
        # plus the prompt template example adds 1 → expect exactly 15 + 1 = 16 at most
        import re
        chunk_id_matches = re.findall(r'"chunk_id"', user_content)
        assert len(chunk_id_matches) <= 16  # 15 snippets + 1 from template example


class TestGetCode:

    def test_strips_entities_section(self):
        text = "[CODE]\nSELECT * FROM customer\n\n[ENTITIES]\nTables: customer"
        code = _get_code({"embedding_text": text})
        assert "[ENTITIES]" not in code
        assert "SELECT * FROM customer" in code

    def test_strips_code_marker(self):
        text = "[CODE]\nSELECT 1"
        code = _get_code({"embedding_text": text})
        assert "[CODE]" not in code

    def test_missing_embedding_text_returns_empty(self):
        code = _get_code({})
        assert code == ""

    def test_no_entities_section_returns_full_code(self):
        text = "[CODE]\nSELECT * FROM orders"
        code = _get_code({"embedding_text": text})
        assert "orders" in code
