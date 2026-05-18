"""
Hybrid search engine combining:
  1. Semantic vector search (ChromaDB cosine similarity)
  2. Metadata filter search (canonical_tables exact match)
  3. BM25 keyword search across all aliases
Results are merged and deduplicated by chunk_id.

Weights are configurable via constructor and can be overridden per-search call.
"""
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.ingestion.vector_store import VectorStore
from src.extractors.alias_registry import AliasRegistry


class HybridSearchEngine:

    def __init__(
        self,
        store: VectorStore,
        registry: AliasRegistry,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.25,
        metadata_weight: float = 0.15,
    ):
        self._store = store
        self._registry = registry
        self._semantic_weight = semantic_weight
        self._bm25_weight = bm25_weight
        self._metadata_weight = metadata_weight
        self._bm25_corpus: List[Dict[str, Any]] = []
        self._bm25_index: Optional[BM25Okapi] = None
        # Pre-tokenized corpus for faster BM25 queries
        self._bm25_tokenized: List[List[str]] = []

    def build_bm25_index(self, docs: List[Dict[str, Any]]) -> None:
        """Build BM25 index from a list of enriched document dicts."""
        self._bm25_corpus = docs
        self._bm25_tokenized = [self._tokenize(d.get("embedding_text", "")) for d in docs]
        if self._bm25_tokenized:
            self._bm25_index = BM25Okapi(self._bm25_tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(
        self,
        query: str,
        top_k: int = 20,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        metadata_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and return ranked results.
        Per-call weight overrides take precedence over constructor defaults.
        """
        sw = semantic_weight if semantic_weight is not None else self._semantic_weight
        bw = bm25_weight if bm25_weight is not None else self._bm25_weight
        mw = metadata_weight if metadata_weight is not None else self._metadata_weight

        canonical = self._registry.resolve(query)
        aliases = self._registry.get_all_aliases(canonical)
        all_terms = list({query.lower(), canonical} | set(aliases))

        scores: Dict[str, float] = {}
        sources: Dict[str, Dict] = {}

        # --- 1. Semantic search ---
        semantic_query = f"{query} {canonical} table references"
        semantic_hits = self._store.query_semantic(semantic_query, n_results=top_k * 2)
        for hit in semantic_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + sw * hit["score"]
            sources[cid] = hit

        # --- 2. Metadata filter search (exact canonical match) ---
        meta_hits = self._store.query_by_table(canonical, n_results=top_k * 2)
        for hit in meta_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + mw
            if cid not in sources:
                sources[cid] = hit

        # --- 3. BM25 keyword search across aliases ---
        if self._bm25_index and self._bm25_corpus:
            # Tokenize all query terms once; reuse pre-tokenized corpus
            bm25_query_tokens: List[str] = []
            for term in all_terms:
                bm25_query_tokens.extend(self._tokenize(term))

            bm25_scores = self._bm25_index.get_scores(bm25_query_tokens)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            for i, doc in enumerate(self._bm25_corpus):
                norm_score = bm25_scores[i] / max_bm25
                if norm_score > 0.01:
                    cid = doc.get("chunk_id", f"bm25_{i}")
                    scores[cid] = scores.get(cid, 0) + bw * norm_score
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
