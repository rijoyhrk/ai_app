import json
import anthropic
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from .document_builder import EnrichedDocument, parse_meta_list


class VectorStore:
    """
    ChromaDB-backed vector store for enriched ETL code chunks.

    Safe re-indexing via staging collection:
      1. store.reset()  — creates a staging collection; old data in COLLECTION stays intact
      2. store.upsert() — writes to staging
      3. store.commit() — atomically promotes staging → COLLECTION, then deletes staging
    If upsert crashes, commit() is never called and the old COLLECTION is preserved.
    """

    COLLECTION = "etl_lineage"
    _STAGING = "etl_lineage_staging"

    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._staging: Optional[Any] = None
        # Clean up any leftover staging from a previous crashed run
        try:
            self._client.delete_collection(self._STAGING)
        except Exception:
            pass

    def reset(self) -> None:
        """Prepare staging collection. Old COLLECTION data is preserved until commit()."""
        try:
            self._client.delete_collection(self._STAGING)
        except Exception:
            pass
        self._staging = self._client.create_collection(
            name=self._STAGING,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, docs: List[EnrichedDocument]) -> None:
        """Write to staging (if reset() was called) or directly to main collection."""
        if not docs:
            return
        target = getattr(self, "_staging", None) or self._collection
        ids = [d.chunk_id for d in docs]
        texts = [d.embedding_text for d in docs]
        metadatas = [d.metadata for d in docs]
        target.upsert(ids=ids, documents=texts, metadatas=metadatas)

    def commit(self) -> None:
        """
        Promote staging → main collection. Copies embeddings so re-embedding is skipped.
        No-op if reset() was not called (direct-upsert mode used by tests).
        """
        if getattr(self, "_staging", None) is None:
            return

        # Pull everything from staging (pre-computed embeddings included)
        data = self._staging.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids", [])

        # Delete old main and recreate
        try:
            self._client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        new_col = self._client.create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Bulk insert in batches to avoid memory spikes
        batch_size = 500
        embs = data.get("embeddings") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for i in range(0, len(ids), batch_size):
            new_col.upsert(
                ids=ids[i:i + batch_size],
                documents=docs[i:i + batch_size] if docs else None,
                metadatas=metas[i:i + batch_size] if metas else None,
                embeddings=embs[i:i + batch_size] if embs else None,
            )

        self._collection = new_col

        # Clean up staging
        try:
            self._client.delete_collection(self._STAGING)
        except Exception:
            pass
        self._staging = None

    def query_semantic(
        self,
        query_text: str,
        n_results: int = 20,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        kwargs = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            hits.append({
                "chunk_id": chunk_id,
                "embedding_text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else 1.0,
                "score": 1.0 - (dists[i] if i < len(dists) else 1.0),
            })
        return hits

    def query_by_table(self, canonical_table: str, n_results: int = 50) -> List[Dict[str, Any]]:
        """Filter chunks that reference a specific canonical table name."""
        try:
            result = self._collection.query(
                query_texts=[canonical_table],
                n_results=n_results,
                where={"canonical_tables": {"$contains": canonical_table}},
            )
        except Exception:
            result = self._collection.query(
                query_texts=[canonical_table],
                n_results=n_results,
            )

        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            # Parse JSON or legacy pipe-delimited canonical_tables and exact-match filter
            tables_list = parse_meta_list(meta.get("canonical_tables", "[]"))
            if canonical_table not in tables_list:
                continue
            hits.append({
                "chunk_id": chunk_id,
                "embedding_text": docs[i] if i < len(docs) else "",
                "metadata": meta,
                "distance": dists[i] if i < len(dists) else 1.0,
                "score": 1.0 - (dists[i] if i < len(dists) else 1.0),
            })
        return hits

    def count(self) -> int:
        return self._collection.count()
