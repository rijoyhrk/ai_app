"""
LLM reranker using Claude API.
Takes top-K hybrid search results and asks Claude to:
  1. Confirm relevance to the entity query
  2. Explain WHY this file/chunk references the entity
  3. Identify the operation type (READ/WRITE/CREATE/etc.)
"""
import json
import anthropic
from typing import List, Dict, Any

_RERANK_SYSTEM = """You are a data lineage expert. Given a user query about a database table
and a set of code snippets, evaluate each snippet for relevance.

For each snippet, determine:
- relevance_score: 0.0 to 1.0 (how confidently does this code reference the queried table?)
- operation: READ | WRITE | CREATE | DROP | UNKNOWN
- explanation: one sentence explaining how/why this code references the table
- matched_via: "direct" if the exact table name appears, "alias" if an alias like cust_tab is used

Respond ONLY with a JSON array — one object per snippet, in the same order as provided."""

_RERANK_USER_TMPL = """Query: Find all scripts referencing table "{entity}"

Snippets to evaluate (JSON array):
{snippets_json}

Respond with a JSON array of objects in the same order:
[
  {{
    "chunk_id": "...",
    "relevance_score": 0.95,
    "operation": "READ",
    "explanation": "...",
    "matched_via": "alias"
  }},
  ...
]"""


def rerank(
    client: anthropic.Anthropic,
    entity: str,
    hits: List[Dict[str, Any]],
    model: str = "claude-opus-4-7",
    min_score: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Rerank search hits using Claude. Returns hits sorted by relevance, filtered
    by min_score, each augmented with explanation and operation fields.
    """
    if not hits:
        return []

    # Build snippets for LLM (limit to top 15, cap text length)
    snippets = []
    for h in hits[:15]:
        meta = h.get("metadata", {})
        snippets.append({
            "chunk_id": h.get("chunk_id", ""),
            "file": meta.get("file_path", ""),
            "lines": f"{meta.get('line_start', '?')}-{meta.get('line_end', '?')}",
            "type": meta.get("file_type", ""),
            "code_excerpt": _get_code(h)[:800],
        })

    snippets_json = json.dumps(snippets, indent=2)
    user_prompt = _RERANK_USER_TMPL.format(
        entity=entity,
        snippets_json=snippets_json,
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": _RERANK_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "[]").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        rerank_results = json.loads(text)
    except json.JSONDecodeError:
        rerank_results = []

    # Merge LLM scores back into hits
    rerank_map: Dict[str, Dict] = {}
    for r in rerank_results:
        if isinstance(r, dict) and "chunk_id" in r:
            rerank_map[r["chunk_id"]] = r

    enriched = []
    for h in hits[:15]:
        cid = h.get("chunk_id", "")
        llm_info = rerank_map.get(cid, {})
        llm_score = float(llm_info.get("relevance_score", 0.0))
        if llm_score < min_score:
            continue
        merged = dict(h)
        merged["llm_score"] = llm_score
        merged["operation"] = llm_info.get("operation", "UNKNOWN")
        merged["explanation"] = llm_info.get("explanation", "")
        merged["matched_via"] = llm_info.get("matched_via", "direct")
        # combined score
        merged["final_score"] = round(
            0.4 * merged.get("hybrid_score", 0) + 0.6 * llm_score, 4
        )
        enriched.append(merged)

    enriched.sort(key=lambda x: x["final_score"], reverse=True)
    return enriched


def _get_code(hit: Dict[str, Any]) -> str:
    """Extract raw code from an embedding_text hit (strip the [ENTITIES] section)."""
    text = hit.get("embedding_text", "")
    if "[ENTITIES]" in text:
        text = text.split("[ENTITIES]")[0]
    return text.replace("[CODE]", "").strip()
