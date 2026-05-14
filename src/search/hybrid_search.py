"""
Hybrid search engine combining:
  1. Semantic vector search (ChromaDB cosine similarity)
  2. Metadata filter search (canonical_tables exact match)
  3. BM25 keyword search across all aliases
Results are merged and deduplicated by chunk_id.
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.ingestion.vector_store import VectorStore
from src.extractors.alias_registry import AliasRegistry


class HybridSearchEngine:

    def __init__(self, store: VectorStore, registry: AliasRegistry):
        self._store = store
        self._registry = registry
        self._bm25_corpus: List[Dict[str, Any]] = []
        self._bm25_index: BM25Okapi | None = None

    def build_bm25_index(self, docs: List[Dict[str, Any]]) -> None:
        """Build BM25 index from a list of enriched document dicts."""
        self._bm25_corpus = docs
        tokenized = [self._tokenize(d.get("embedding_text", "")) for d in docs]
        if tokenized:
            self._bm25_index = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return re.findall(r"\w+", text.lower())

    def search(
        self,
        query: str,
        top_k: int = 20,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.25,
        metadata_weight: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and return ranked results.
        """
        canonical = self._registry.resolve(query)
        aliases = self._registry.get_all_aliases(canonical)
        # include original query tokens too
        all_terms = list({query.lower(), canonical} | set(aliases))

        scores: Dict[str, float] = {}
        sources: Dict[str, Dict] = {}

        # --- 1. Semantic search ---
        # expand query with canonical name to bridge alias gap
        semantic_query = f"{query} {canonical} table references"
        semantic_hits = self._store.query_semantic(semantic_query, n_results=top_k * 2)
        for hit in semantic_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + semantic_weight * hit["score"]
            sources[cid] = hit

        # --- 2. Metadata filter search (exact canonical match) ---
        meta_hits = self._store.query_by_table(canonical, n_results=top_k * 2)
        for hit in meta_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + metadata_weight
            if cid not in sources:
                sources[cid] = hit

        # --- 3. BM25 keyword search across aliases ---
        if self._bm25_index and self._bm25_corpus:
            bm25_query_tokens = []
            for term in all_terms:
                bm25_query_tokens.extend(self._tokenize(term))

            bm25_scores = self._bm25_index.get_scores(bm25_query_tokens)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            for i, doc in enumerate(self._bm25_corpus):
                norm_score = bm25_scores[i] / max_bm25
                if norm_score > 0.01:
                    cid = doc.get("chunk_id", f"bm25_{i}")
                    scores[cid] = scores.get(cid, 0) + bm25_weight * norm_score
                    if cid not in sources:
                        sources[cid] = doc

        # --- Rank and return ---
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in ranked[:top_k]:
            hit = dict(sources.get(cid, {}))
            hit["hybrid_score"] = round(score, 4)
            hit["matched_canonical"] = canonical
            hit["matched_aliases"] = aliases
            results.append(hit)

        return results
