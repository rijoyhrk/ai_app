"""
Full ingestion pipeline:
  1. Crawl ETL files → AST-aware chunks
  2. LLM entity extraction + alias registry
  3. Enriched document building
  4. ChromaDB upsert
"""
import anthropic
from pathlib import Path
from typing import Optional
from src.chunkers.dispatcher import crawl_and_chunk
from src.extractors.entity_extractor import batch_extract_entities
from src.extractors.alias_registry import AliasRegistry
from src.ingestion.document_builder import build_enriched_documents
from src.ingestion.vector_store import VectorStore


def run_ingestion(
    repo_path: str,
    chroma_persist_dir: str,
    alias_registry_path: str,
    anthropic_client: anthropic.Anthropic,
    llm_model: str = "claude-opus-4-7",
    verbose: bool = True,
) -> tuple[VectorStore, AliasRegistry]:
    """
    Full ingestion pipeline. Returns the vector store and alias registry.
    """
    log = print if verbose else lambda *a, **k: None

    # Step 1: chunk all ETL files
    log(f"[1/4] Crawling and chunking ETL files in: {repo_path}")
    chunks = crawl_and_chunk(repo_path)
    log(f"      Found {len(chunks)} chunks across all files")

    if not chunks:
        log("      No files found. Check ETL_REPO_PATH.")
        registry = AliasRegistry()
        store = VectorStore(chroma_persist_dir)
        return store, registry

    # Step 2: LLM entity extraction
    log(f"[2/4] Extracting entities with LLM ({llm_model})...")
    all_entities = batch_extract_entities(anthropic_client, chunks, model=llm_model)
    log(f"      Processed {len(all_entities)} chunks")

    # Step 3: build alias registry
    log("[3/4] Building alias registry...")
    registry = AliasRegistry.load(alias_registry_path)
    registry.ingest(all_entities)
    registry.save(alias_registry_path)
    log(f"\n{registry.summary()}\n")

    # Step 4: enrich + upsert into ChromaDB
    log("[4/4] Building enriched documents and upserting into ChromaDB...")
    docs = build_enriched_documents(chunks, all_entities)
    store = VectorStore(chroma_persist_dir)
    store.upsert(docs)
    log(f"      Vector store now contains {store.count()} chunks")

    return store, registry
