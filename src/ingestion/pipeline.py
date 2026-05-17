"""
Full ingestion pipeline:
  1. Crawl ETL files → AST-aware chunks
  2. Entity extraction (LLM or rule-based fallback)
  3. Alias registry build
  4. ChromaDB upsert
"""
import anthropic
from pathlib import Path
from typing import Optional, List
from src.chunkers.dispatcher import crawl_and_chunk
from src.extractors.entity_extractor import batch_extract_entities
from src.extractors.rule_based_extractor import batch_extract_entities_rule_based
from src.extractors.alias_registry import AliasRegistry
from src.ingestion.document_builder import build_enriched_documents
from src.ingestion.vector_store import VectorStore


def run_ingestion(
    repo_path: str,
    chroma_persist_dir: str,
    alias_registry_path: str,
    anthropic_client: anthropic.Anthropic,
    llm_model: str = "claude-opus-4-7",
    excluded_dirs: Optional[list] = None,
    skip_llm: bool = False,
    verbose: bool = True,
) -> tuple[VectorStore, AliasRegistry]:
    """
    Full ingestion pipeline. Returns the vector store and alias registry.

    skip_llm=True uses regex-based entity extraction instead of Claude.
    No API calls are made in that mode — alias resolution is unavailable
    but exact table name search still works.
    """
    log = print if verbose else lambda *a, **k: None

    # Step 1: chunk all ETL files
    log(f"[1/4] Crawling and chunking ETL files in: {repo_path}")
    if excluded_dirs:
        log(f"      Skipping dirs: {excluded_dirs}")
    chunks = crawl_and_chunk(repo_path, excluded_dirs=excluded_dirs)
    log(f"      Found {len(chunks)} chunks across all files")

    if not chunks:
        log("      No files found. Check ETL_REPO_PATH.")
        return VectorStore(chroma_persist_dir), AliasRegistry()

    # Step 2: entity extraction
    if skip_llm:
        log("[2/4] Extracting entities with rule-based extractor (LLM skipped)...")
        all_entities = batch_extract_entities_rule_based(chunks)
    else:
        log(f"[2/4] Extracting entities with LLM ({llm_model})...")
        try:
            all_entities = batch_extract_entities(anthropic_client, chunks, model=llm_model)
        except (anthropic.BadRequestError, anthropic.APIStatusError,
                anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            log(f"      API unavailable ({e}). Falling back to rule-based extraction.")
            all_entities = batch_extract_entities_rule_based(chunks)
            skip_llm = True   # propagate so caller knows
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
    store.reset()          # clear stale chunks from previous runs before re-indexing
    store.upsert(docs)
    log(f"      Vector store now contains {store.count()} chunks")

    return store, registry
