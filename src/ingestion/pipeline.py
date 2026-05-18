"""
Full ingestion pipeline:
  1. Crawl ETL files → AST-aware chunks
  2. Entity extraction (LLM parallel or rule-based fallback)
  3. Alias registry build
  4. ChromaDB upsert via staging (rollback-safe)
  5. Registry persisted only after vector store commit succeeds
"""
import logging
import anthropic
from typing import Optional
from src.chunkers.dispatcher import crawl_and_chunk
from src.extractors.entity_extractor import batch_extract_entities
from src.extractors.rule_based_extractor import batch_extract_entities_rule_based
from src.extractors.alias_registry import AliasRegistry
from src.ingestion.document_builder import build_enriched_documents
from src.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


def run_ingestion(
    repo_path: str,
    chroma_persist_dir: str,
    alias_registry_path: str,
    anthropic_client: anthropic.Anthropic,
    llm_model: str = "claude-opus-4-7",
    excluded_dirs: Optional[list] = None,
    skip_llm: bool = False,
    verbose: bool = True,
    max_extraction_workers: int = 8,
) -> tuple[VectorStore, AliasRegistry]:
    """
    Full ingestion pipeline. Returns the vector store and alias registry.

    skip_llm=True uses regex-based entity extraction instead of Claude.
    No API calls are made in that mode — alias resolution is unavailable
    but exact table name search still works.

    Safe re-indexing: new data is written to a staging collection and
    promoted atomically only after the full upsert succeeds. The previous
    index is preserved if upsert crashes mid-way.
    """
    log = print if verbose else lambda *a, **k: None
    logger.info(
        "Ingestion started: repo=%s skip_llm=%s excluded=%s",
        repo_path, skip_llm, excluded_dirs,
    )

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
        log(f"[2/4] Extracting entities with LLM ({llm_model}) using {max_extraction_workers} workers...")
        try:
            all_entities = batch_extract_entities(
                anthropic_client, chunks, model=llm_model,
                max_workers=max_extraction_workers,
            )
        except (anthropic.BadRequestError, anthropic.APIStatusError,
                anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            log(f"      API unavailable ({e}). Falling back to rule-based extraction.")
            logger.warning("LLM extraction failed, falling back to rule-based: %s", e)
            all_entities = batch_extract_entities_rule_based(chunks)
            skip_llm = True
    log(f"      Processed {len(all_entities)} chunks")

    # Step 3: build alias registry in memory (persisted after commit in step 4)
    log("[3/4] Building alias registry...")
    registry = AliasRegistry.load(alias_registry_path)
    registry.ingest(all_entities)
    log(f"\n{registry.summary()}\n")

    # Step 4: enrich + upsert into ChromaDB via staging (rollback-safe)
    log("[4/4] Building enriched documents and upserting into ChromaDB...")
    docs = build_enriched_documents(chunks, all_entities)
    store = VectorStore(chroma_persist_dir)
    store.reset()       # creates staging collection; old data preserved until commit
    store.upsert(docs)  # writes to staging
    store.commit()      # atomically promotes staging → main collection
    log(f"      Vector store now contains {store.count()} chunks")

    # Persist registry only after the vector store commit succeeds
    registry.save(alias_registry_path)

    logger.info(
        "Ingestion complete: repo=%s chunks=%d tables=%d skip_llm=%s",
        repo_path, store.count(), len(registry.all_canonicals()), skip_llm,
    )

    return store, registry
