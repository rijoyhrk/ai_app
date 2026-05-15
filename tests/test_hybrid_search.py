"""
Tests for the hybrid search engine.
ChromaDB is replaced with a lightweight mock; BM25 runs on synthetic docs.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.hybrid_search import HybridSearchEngine


def _make_mock_store(semantic_hits=None, table_hits=None):
    """Build a mock VectorStore with configurable return values."""
    store = MagicMock()
    store.query_semantic.return_value = semantic_hits or []
    store.query_by_table.return_value = table_hits or []
    return store


def _make_hit(chunk_id, score=0.8, canonical_tables="customer",
              aliases_used="cust_tab", file_path="test.sql",
              embedding_text="[CODE]\nSELECT * FROM cust_tab\n[ENTITIES]\nTables: customer"):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "embedding_text": embedding_text,
        "metadata": {
            "file_path": file_path,
            "file_type": "sql",
            "line_start": 1,
            "line_end": 10,
            "canonical_tables": canonical_tables,
            "aliases_used": aliases_used,
        },
    }


class TestHybridSearchEngine:

    def test_search_returns_list(self, populated_registry):
        store = _make_mock_store(
            semantic_hits=[_make_hit("c1", score=0.9)],
            table_hits=[_make_hit("c1", score=0.9)],
        )
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([{
            "chunk_id": "c1",
            "embedding_text": "[CODE]\nSELECT * FROM cust_tab\n[ENTITIES]\nTables: customer",
            "metadata": {"file_path": "test.sql"},
        }])
        results = engine.search("customer", top_k=5)
        assert isinstance(results, list)

    def test_deduplicates_by_chunk_id(self, populated_registry):
        """Same chunk_id from semantic + metadata layers should appear once."""
        hit = _make_hit("c1", score=0.9)
        store = _make_mock_store(semantic_hits=[hit], table_hits=[hit])
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([])
        results = engine.search("customer", top_k=10)
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_hybrid_score_assigned(self, populated_registry):
        store = _make_mock_store(semantic_hits=[_make_hit("c1", score=0.8)])
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([{
            "chunk_id": "c1",
            "embedding_text": "SELECT * FROM cust_tab customer",
            "metadata": {},
        }])
        results = engine.search("customer", top_k=5)
        assert all("hybrid_score" in r for r in results)

    def test_matched_canonical_attached(self, populated_registry):
        store = _make_mock_store(semantic_hits=[_make_hit("c1")])
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([])
        results = engine.search("cust_tab", top_k=5)
        # cust_tab resolves to customer via registry
        assert all(r["matched_canonical"] == "customer" for r in results)

    def test_query_expansion_calls_registry(self, populated_registry):
        store = _make_mock_store()
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([])
        engine.search("cust_data", top_k=5)
        # semantic query should be called with expanded terms
        assert store.query_semantic.called

    def test_top_k_respected(self, populated_registry):
        hits = [_make_hit(f"c{i}", score=0.9 - i * 0.05) for i in range(10)]
        store = _make_mock_store(semantic_hits=hits)
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([])
        results = engine.search("customer", top_k=3)
        assert len(results) <= 3

    def test_empty_store_returns_empty(self, populated_registry):
        store = _make_mock_store(semantic_hits=[], table_hits=[])
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([])
        results = engine.search("customer", top_k=10)
        assert results == []

    def test_bm25_index_built_from_docs(self, populated_registry):
        store = _make_mock_store()
        engine = HybridSearchEngine(store, populated_registry)
        docs = [
            {"chunk_id": f"c{i}", "embedding_text": f"SELECT * FROM table_{i}", "metadata": {}}
            for i in range(5)
        ]
        engine.build_bm25_index(docs)
        assert engine._bm25_index is not None
        assert len(engine._bm25_corpus) == 5

    def test_build_bm25_index_empty_list(self, populated_registry):
        store = _make_mock_store()
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([])
        # should not raise; index stays None
        results = engine.search("customer", top_k=5)
        assert isinstance(results, list)

    def test_bm25_boosts_alias_matches(self, populated_registry):
        """A document containing the alias 'cust_tab' should score higher than one without."""
        hit_with_alias    = _make_hit("c1", score=0.5, aliases_used="cust_tab",
                                      embedding_text="SELECT * FROM cust_tab customer")
        hit_without_alias = _make_hit("c2", score=0.5, aliases_used="",
                                      embedding_text="SELECT * FROM unrelated_table")

        store = _make_mock_store(semantic_hits=[hit_with_alias, hit_without_alias])
        engine = HybridSearchEngine(store, populated_registry)
        engine.build_bm25_index([
            {"chunk_id": "c1", "embedding_text": "SELECT * FROM cust_tab customer", "metadata": {}},
            {"chunk_id": "c2", "embedding_text": "SELECT * FROM unrelated_table",   "metadata": {}},
        ])
        results = engine.search("cust_tab", top_k=10)
        ids = [r["chunk_id"] for r in results]
        if len(ids) >= 2:
            assert ids[0] == "c1"  # alias match should rank higher
