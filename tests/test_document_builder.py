"""
Tests for the enriched document builder.
Verifies that embedding text correctly weaves raw code + canonical entity metadata
so semantic search can bridge alias→canonical gaps.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.base import CodeChunk
from src.extractors.entity_extractor import ChunkEntities, EntityRef
from src.ingestion.document_builder import (
    build_enriched_document, build_enriched_documents, EnrichedDocument,
)


def _chunk(raw_text="SELECT * FROM cust_tab", file_type="sql",
           file_path="test.sql", line_start=1, line_end=5):
    return CodeChunk(
        chunk_id=f"{file_path}::{line_start}-{line_end}",
        file_path=file_path,
        file_type=file_type,
        chunk_type="select",
        raw_text=raw_text,
        line_start=line_start,
        line_end=line_end,
    )


def _entities(chunk_id, file_path, refs):
    return ChunkEntities(
        chunk_id=chunk_id,
        file_path=file_path,
        refs=[EntityRef(**r) for r in refs],
    )


class TestBuildEnrichedDocument:

    def test_returns_enriched_document(self):
        chunk = _chunk()
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert isinstance(doc, EnrichedDocument)

    def test_canonical_tables_extracted(self):
        chunk = _chunk()
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
            {"raw": "orders",   "canonical": "orders",   "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "customer" in doc.canonical_tables
        assert "orders"   in doc.canonical_tables

    def test_aliases_extracted(self):
        chunk = _chunk()
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "cust_tab" in doc.aliases_used

    def test_canonical_not_in_aliases_when_same_as_raw(self):
        chunk = _chunk("SELECT * FROM customer")
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "customer", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        # when raw == canonical, it should NOT appear in aliases_used
        assert "customer" not in doc.aliases_used

    def test_operations_captured(self):
        chunk = _chunk("INSERT INTO customer SELECT * FROM staging")
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "customer", "canonical": "customer", "ref_type": "WRITE", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "WRITE" in doc.operations

    def test_embedding_text_contains_code_section(self):
        chunk = _chunk("SELECT * FROM cust_tab")
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "[CODE]" in doc.embedding_text
        assert "cust_tab" in doc.embedding_text

    def test_embedding_text_contains_entities_section(self):
        chunk = _chunk()
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "[ENTITIES]" in doc.embedding_text
        # canonical name must appear in embedding text to bridge alias gap
        assert "customer" in doc.embedding_text

    def test_alias_in_embedding_text(self):
        """Key invariant: the alias must appear in embedding text for BM25 to find it."""
        chunk = _chunk("SELECT * FROM cust_data WHERE region = 'US'")
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_data", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "cust_data" in doc.embedding_text

    def test_metadata_fields_pipe_delimited(self):
        chunk = _chunk()
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ",  "confidence": "high"},
            {"raw": "ord_tab",  "canonical": "orders",   "ref_type": "WRITE", "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        # ChromaDB doesn't support lists — must be pipe-delimited strings
        assert isinstance(doc.metadata["canonical_tables"], str)
        assert "|" in doc.metadata["canonical_tables"] or len(doc.canonical_tables) == 1

    def test_metadata_contains_file_path(self):
        chunk = _chunk(file_path="etl/load.sql")
        entities = _entities(chunk.chunk_id, chunk.file_path, [])
        doc = build_enriched_document(chunk, entities)
        assert doc.metadata["file_path"] == "etl/load.sql"

    def test_column_refs_excluded_from_canonical(self):
        chunk = _chunk()
        entities = _entities(chunk.chunk_id, chunk.file_path, [
            {"raw": "cust_id", "canonical": None,       "ref_type": "COLUMN", "confidence": "high"},
            {"raw": "cust_tab","canonical": "customer",  "ref_type": "READ",   "confidence": "high"},
        ])
        doc = build_enriched_document(chunk, entities)
        assert "cust_id" not in doc.canonical_tables
        assert "customer" in doc.canonical_tables

    def test_chunk_with_no_entities_builds_correctly(self):
        chunk = _chunk("-- empty comment block")
        entities = _entities(chunk.chunk_id, chunk.file_path, [])
        doc = build_enriched_document(chunk, entities)
        assert doc.canonical_tables == []
        assert doc.aliases_used     == []
        assert "[CODE]" in doc.embedding_text


class TestBuildEnrichedDocuments:

    def test_processes_all_chunks(self):
        chunks = [
            _chunk("SELECT * FROM orders",   file_path="a.sql", line_start=1, line_end=1),
            _chunk("SELECT * FROM customer", file_path="b.sql", line_start=1, line_end=1),
        ]
        all_entities = [
            _entities(chunks[0].chunk_id, "a.sql", [
                {"raw": "orders",   "canonical": "orders",   "ref_type": "READ", "confidence": "high"},
            ]),
            _entities(chunks[1].chunk_id, "b.sql", [
                {"raw": "customer", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
            ]),
        ]
        docs = build_enriched_documents(chunks, all_entities)
        assert len(docs) == 2

    def test_unmatched_chunk_gets_empty_entities(self):
        chunk = _chunk("SELECT * FROM orders", file_path="c.sql", line_start=1, line_end=5)
        # No matching entity entry for this chunk
        docs = build_enriched_documents([chunk], [])
        assert len(docs) == 1
        assert docs[0].canonical_tables == []
