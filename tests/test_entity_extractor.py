"""
Tests for LLM entity extraction.
Claude API is mocked — no real API calls are made.
"""
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.base import CodeChunk
from src.extractors.entity_extractor import (
    extract_entities, batch_extract_entities, EntityRef, ChunkEntities,
)
from conftest import make_claude_entity_response


def _make_chunk(raw_text, file_type="sql", chunk_type="statement",
                file_path="test.sql", line_start=1, line_end=10):
    return CodeChunk(
        chunk_id=f"{file_path}::{line_start}-{line_end}",
        file_path=file_path,
        file_type=file_type,
        chunk_type=chunk_type,
        raw_text=raw_text,
        line_start=line_start,
        line_end=line_end,
    )


class TestExtractEntities:

    def test_returns_chunk_entities_object(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([
            {"raw": "cust_tab", "canonical": "customer", "type": "READ", "confidence": "high"},
        ])
        chunk = _make_chunk("SELECT * FROM cust_tab")
        result = extract_entities(mock_anthropic_client, chunk)
        assert isinstance(result, ChunkEntities)

    def test_parses_entity_refs_correctly(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([
            {"raw": "cust_tab",  "canonical": "customer", "type": "READ",  "confidence": "high"},
            {"raw": "orders",    "canonical": "orders",   "type": "READ",  "confidence": "high"},
        ])
        chunk = _make_chunk("SELECT * FROM cust_tab JOIN orders")
        result = extract_entities(mock_anthropic_client, chunk)
        assert len(result.refs) == 2
        raws = [r.raw for r in result.refs]
        assert "cust_tab" in raws
        assert "orders"   in raws

    def test_canonical_name_mapped_correctly(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([
            {"raw": "cust_tab", "canonical": "customer", "type": "READ", "confidence": "high"},
        ])
        chunk = _make_chunk("SELECT * FROM cust_tab")
        result = extract_entities(mock_anthropic_client, chunk)
        assert result.refs[0].canonical == "customer"
        assert result.refs[0].raw       == "cust_tab"
        assert result.refs[0].ref_type  == "READ"

    def test_chunk_id_preserved(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([])
        chunk = _make_chunk("SELECT 1", file_path="my.sql", line_start=5, line_end=5)
        result = extract_entities(mock_anthropic_client, chunk)
        assert result.chunk_id == "my.sql::5-5"

    def test_empty_response_returns_no_refs(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([])
        chunk = _make_chunk("-- just a comment")
        result = extract_entities(mock_anthropic_client, chunk)
        assert result.refs == []

    def test_handles_malformed_json_gracefully(self, mock_anthropic_client):
        msg = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = "this is not json at all"
        msg.content = [block]
        mock_anthropic_client.messages.create.return_value = msg

        chunk = _make_chunk("SELECT * FROM orders")
        result = extract_entities(mock_anthropic_client, chunk)
        assert result.refs == []

    def test_handles_markdown_fenced_json(self, mock_anthropic_client):
        """Claude sometimes wraps JSON in ```json ... ``` fences."""
        refs = [{"raw": "orders", "canonical": "orders", "type": "READ", "confidence": "high"}]
        msg = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = f"```json\n{json.dumps(refs)}\n```"
        msg.content = [block]
        mock_anthropic_client.messages.create.return_value = msg

        chunk = _make_chunk("SELECT * FROM orders")
        result = extract_entities(mock_anthropic_client, chunk)
        assert len(result.refs) == 1
        assert result.refs[0].raw == "orders"

    def test_write_operation_captured(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([
            {"raw": "customer", "canonical": "customer", "type": "WRITE", "confidence": "high"},
        ])
        chunk = _make_chunk("INSERT INTO customer SELECT * FROM staging")
        result = extract_entities(mock_anthropic_client, chunk)
        assert result.refs[0].ref_type == "WRITE"

    def test_column_ref_type_preserved(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([
            {"raw": "cust_id", "canonical": None, "type": "COLUMN", "confidence": "high"},
        ])
        chunk = _make_chunk("SELECT cust_id FROM customer")
        result = extract_entities(mock_anthropic_client, chunk)
        assert result.refs[0].ref_type == "COLUMN"
        assert result.refs[0].canonical is None


class TestBatchExtractEntities:

    def test_processes_all_chunks(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([
            {"raw": "orders", "canonical": "orders", "type": "READ", "confidence": "high"},
        ])
        chunks = [
            _make_chunk("SELECT * FROM orders", file_path="f1.sql", line_start=1, line_end=1),
            _make_chunk("SELECT * FROM orders", file_path="f2.sql", line_start=1, line_end=1),
            _make_chunk("SELECT * FROM orders", file_path="f3.sql", line_start=1, line_end=1),
        ]
        results = batch_extract_entities(mock_anthropic_client, chunks)
        assert len(results) == 3

    def test_skips_trivially_small_chunks(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([])
        chunks = [
            _make_chunk("SELECT * FROM orders WHERE status = 'COMPLETED'"),  # valid (>10 chars)
            _make_chunk("   "),                                               # whitespace — skipped
            _make_chunk("--"),                                                # too short — skipped
        ]
        results = batch_extract_entities(mock_anthropic_client, chunks)
        # Only the valid chunk should be processed
        assert len(results) == 1

    def test_returns_list_of_chunk_entities(self, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_claude_entity_response([])
        chunk = _make_chunk("SELECT * FROM customer")
        results = batch_extract_entities(mock_anthropic_client, [chunk])
        assert isinstance(results, list)
        assert all(isinstance(r, ChunkEntities) for r in results)

    def test_empty_chunk_list_returns_empty(self, mock_anthropic_client):
        results = batch_extract_entities(mock_anthropic_client, [])
        assert results == []
