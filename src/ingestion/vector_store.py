import anthropic
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from .document_builder import EnrichedDocument


def _get_embedding(client: anthropic.Anthropic, text: str, model: str) -> List[float]:
    """
    Generate an embedding for text.
    Claude API does not expose a direct embedding endpoint — we use the
    messages API with a structured extraction approach via a lightweight
    sentence-transformer fallback using chromadb's default embedding fn.
    ChromaDB by default uses sentence-transformers/all-MiniLM-L6-v2 locally.
    """
    # ChromaDB handles embedding internally via its default embedding function.
    # We return None here and let ChromaDB embed for us.
    return None


class VectorStore:
    """
    ChromaDB-backed vector store for enriched ETL code chunks.
    Uses ChromaDB's built-in sentence-transformer embeddings (local, no API cost).
    """

    COLLECTION = "etl_lineage"

    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, docs: List[EnrichedDocument]) -> None:
        if not docs:
            return
        ids = [d.chunk_id for d in docs]
        texts = [d.embedding_text for d in docs]
        metadatas = [d.metadata for d in docs]
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

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
            # fallback: pure semantic if metadata filter fails
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
            # post-filter: only keep chunks that actually mention the table
            tables_str = meta.get("canonical_tables", "")
            if canonical_table not in tables_str:
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

    def reset(self) -> None:
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
