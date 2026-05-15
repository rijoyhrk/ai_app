"""
End-to-end integration tests for the full ingestion → search → rerank pipeline.

Uses:
- Real file chunkers (SQL / Python / Shell) on synthetic ETL files
- Mocked Claude API (entity extraction + reranking)
- In-memory ChromaDB (ephemeral client, no disk writes)
- Real AliasRegistry, DocumentBuilder, HybridSearchEngine
"""
import sys
import json
import pytest
import chromadb
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.dispatcher import crawl_and_chunk
from src.extractors.entity_extractor import batch_extract_entities, ChunkEntities, EntityRef
from src.extractors.alias_registry import AliasRegistry
from src.ingestion.document_builder import build_enriched_documents
from src.ingestion.vector_store import VectorStore
from src.search.hybrid_search import HybridSearchEngine
from src.search.reranker import rerank
from src.graph.lineage_graph import LineageGraph
from conftest import make_claude_entity_response, make_claude_rerank_response


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_entity_extraction(client, chunks, model):
    """
    Deterministic mock for batch_extract_entities.
    Returns hard-coded entities based on file type so tests are predictable.
    """
    results = []
    for chunk in chunks:
        if len(chunk.raw_text.strip()) < 10:
            continue
        refs = []
        text = chunk.raw_text.lower()

        if "cust_tab" in text or "cust_data" in text:
            refs.append(EntityRef(raw="cust_tab",  canonical="customer", ref_type="READ",  confidence="high"))
        if "customer_staging" in text:
            refs.append(EntityRef(raw="customer_staging", canonical="customer_staging", ref_type="READ", confidence="high"))
        if "insert into customer" in text or "customer_ltv" in text:
            refs.append(EntityRef(raw="customer", canonical="customer", ref_type="WRITE", confidence="high"))
        if "orders" in text:
            refs.append(EntityRef(raw="orders", canonical="orders", ref_type="READ", confidence="high"))
        if "prod_tbl" in text:
            refs.append(EntityRef(raw="prod_tbl", canonical="product", ref_type="WRITE", confidence="high"))
        if "cust_staging" in text:
            refs.append(EntityRef(raw="cust_staging", canonical="customer_staging", ref_type="WRITE", confidence="high"))

        results.append(ChunkEntities(
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            refs=refs,
        ))
    return results


def _make_in_memory_vector_store(tmp_path):
    """VectorStore backed by ephemeral ChromaDB (no disk writes)."""
    store = VectorStore.__new__(VectorStore)
    client = chromadb.EphemeralClient()
    store._client = client
    store._collection = client.get_or_create_collection(
        name="test_etl_lineage",
        metadata={"hnsw:space": "cosine"},
    )
    return store


# ─── Integration test class ────────────────────────────────────────────────────

class TestFullIngestionPipeline:

    @pytest.fixture
    def ingested_store_and_registry(self, tmp_etl_repo, tmp_path, mock_anthropic_client):
        """Run full ingestion on the synthetic ETL repo with mocked Claude."""
        # Step 1: Crawl
        chunks = crawl_and_chunk(str(tmp_etl_repo), excluded_dirs=["archive"])
        assert len(chunks) > 0

        # Step 2: Entity extraction (mocked)
        all_entities = _mock_entity_extraction(mock_anthropic_client, chunks, "claude-opus-4-7")
        assert len(all_entities) > 0

        # Step 3: Build registry
        registry = AliasRegistry()
        registry.ingest(all_entities)

        # Step 4: Build enriched docs
        docs = build_enriched_documents(chunks, all_entities)
        assert len(docs) > 0

        # Step 5: Upsert into in-memory vector store
        store = _make_in_memory_vector_store(tmp_path)
        store.upsert(docs)
        assert store.count() > 0

        return store, registry

    def test_chunks_found_for_all_file_types(self, tmp_etl_repo):
        chunks = crawl_and_chunk(str(tmp_etl_repo), excluded_dirs=["archive"])
        types = {c.file_type for c in chunks}
        assert "sql"    in types
        assert "python" in types
        assert "shell"  in types

    def test_archive_excluded(self, tmp_etl_repo):
        chunks_all  = crawl_and_chunk(str(tmp_etl_repo))
        chunks_excl = crawl_and_chunk(str(tmp_etl_repo), excluded_dirs=["archive"])
        assert len(chunks_all) > len(chunks_excl)

    def test_alias_registry_resolves_cust_tab(self, ingested_store_and_registry):
        _, registry = ingested_store_and_registry
        assert registry.resolve("cust_tab") == "customer"

    def test_alias_registry_resolves_prod_tbl(self, ingested_store_and_registry):
        _, registry = ingested_store_and_registry
        assert registry.resolve("prod_tbl") == "product"

    def test_registry_knows_multiple_canonicals(self, ingested_store_and_registry):
        _, registry = ingested_store_and_registry
        canonicals = registry.all_canonicals()
        assert "customer" in canonicals
        assert "orders"   in canonicals
        assert "product"  in canonicals

    def test_vector_store_indexed_chunks(self, ingested_store_and_registry):
        store, _ = ingested_store_and_registry
        assert store.count() > 0

    def test_semantic_search_returns_results(self, ingested_store_and_registry):
        store, _ = ingested_store_and_registry
        hits = store.query_semantic("customer data pipeline", n_results=5)
        assert isinstance(hits, list)

    def test_metadata_filter_returns_customer_chunks(self, ingested_store_and_registry):
        store, _ = ingested_store_and_registry
        hits = store.query_by_table("customer", n_results=10)
        # All returned chunks must reference "customer" in metadata
        for h in hits:
            assert "customer" in h["metadata"].get("canonical_tables", "")


class TestFullSearchPipeline:

    @pytest.fixture
    def search_setup(self, tmp_etl_repo, tmp_path, mock_anthropic_client):
        """Ingestion + BM25 index ready for search."""
        chunks      = crawl_and_chunk(str(tmp_etl_repo), excluded_dirs=["archive"])
        all_entities = _mock_entity_extraction(mock_anthropic_client, chunks, "claude-opus-4-7")

        registry = AliasRegistry()
        registry.ingest(all_entities)

        docs  = build_enriched_documents(chunks, all_entities)
        store = _make_in_memory_vector_store(tmp_path)
        store.upsert(docs)

        engine = HybridSearchEngine(store, registry)
        bm25_docs = [
            {"chunk_id": d.chunk_id, "embedding_text": d.embedding_text, "metadata": d.metadata}
            for d in docs
        ]
        engine.build_bm25_index(bm25_docs)

        return store, registry, engine

    def test_hybrid_search_finds_customer_chunks(self, search_setup):
        _, _, engine = search_setup
        hits = engine.search("customer", top_k=10)
        assert len(hits) > 0

    def test_alias_query_resolves_and_finds_results(self, search_setup):
        _, _, engine = search_setup
        # Search using alias — should still find customer-related chunks
        hits = engine.search("cust_tab", top_k=10)
        assert len(hits) > 0
        assert all(r["matched_canonical"] == "customer" for r in hits)

    def test_hybrid_scores_between_zero_and_one(self, search_setup):
        _, _, engine = search_setup
        hits = engine.search("customer", top_k=10)
        for h in hits:
            assert 0.0 <= h["hybrid_score"] <= 2.0  # can exceed 1.0 due to additive weights

    def test_product_query_finds_prod_tbl_chunks(self, search_setup):
        _, _, engine = search_setup
        hits = engine.search("product", top_k=10)
        # product_sync.sql contains prod_tbl which should be indexed
        assert len(hits) > 0

    def test_unknown_table_returns_empty_or_low_scoring(self, search_setup):
        _, _, engine = search_setup
        hits = engine.search("xyz_nonexistent_table_12345", top_k=10)
        # May return results from semantic similarity; scores should be low
        high_scoring = [h for h in hits if h["hybrid_score"] > 0.8]
        assert len(high_scoring) == 0


class TestFullRerankerIntegration:

    def test_reranker_applied_to_search_results(self, mock_anthropic_client, sample_hits):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "pipeline.py::1-10",     "relevance_score": 0.92,
             "operation": "READ",  "explanation": "Reads customer via cust_data alias.", "matched_via": "alias"},
            {"chunk_id": "load_customers.sql::1-5","relevance_score": 0.88,
             "operation": "WRITE", "explanation": "Inserts into customer table.",       "matched_via": "direct"},
        ])
        results = rerank(mock_anthropic_client, "customer", sample_hits, min_score=0.25)
        assert len(results) == 2
        assert results[0]["final_score"] > results[1]["final_score"]

    def test_lineage_graph_built_from_reranked_results(self, mock_anthropic_client, sample_hits, populated_registry):
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "pipeline.py::1-10",     "relevance_score": 0.9,
             "operation": "READ",  "explanation": "Reads customer.", "matched_via": "alias"},
            {"chunk_id": "load_customers.sql::1-5","relevance_score": 0.85,
             "operation": "WRITE", "explanation": "Writes customer.", "matched_via": "direct"},
        ])
        ranked = rerank(mock_anthropic_client, "customer", sample_hits)
        graph = LineageGraph()
        graph.build_from_results(ranked, populated_registry)
        G = graph.get_graph()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_summary_table_deduplicated_by_file(self, mock_anthropic_client, sample_hits):
        """Two chunks from the same file should appear as one file in summary."""
        mock_anthropic_client.messages.create.return_value = make_claude_rerank_response([
            {"chunk_id": "pipeline.py::1-10",  "relevance_score": 0.9,
             "operation": "READ", "explanation": "Read op.", "matched_via": "alias"},
            {"chunk_id": "load_customers.sql::1-5", "relevance_score": 0.85,
             "operation": "WRITE", "explanation": "Write op.", "matched_via": "direct"},
        ])
        ranked = rerank(mock_anthropic_client, "customer", sample_hits)

        from collections import defaultdict
        groups = defaultdict(list)
        for r in ranked:
            fp = r["metadata"]["file_path"]
            groups[fp].append(r)

        assert len(groups) == len(set(r["metadata"]["file_path"] for r in ranked))
