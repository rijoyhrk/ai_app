import json
import anthropic
from typing import List, Dict, Any
from dataclasses import dataclass, field
from src.chunkers.base import CodeChunk

_SYSTEM_PROMPT = """You are an expert data lineage analyst specializing in ETL code.
Your job is to extract all table, dataset, and view references from code chunks and
determine their canonical (real) names even when aliases or abbreviations are used.

Rules:
- Look for table names in SQL FROM, JOIN, INTO, UPDATE, CREATE, DROP clauses
- Look for table names in Python (pandas, SQLAlchemy, PySpark, psycopg2 queries)
- Look for table names in shell scripts (psql -c, bq query, spark-submit args)
- Infer canonical names from abbreviations using domain knowledge:
  cust_tab, cust_tbl, CUST → customer
  ord_tab, orders_tbl → orders
  prod_tab, prd_tbl → product
  txn_tab, trans_tab → transaction
- If the alias clearly maps to a known pattern, mark confidence "high"
- If it is a reasonable guess, mark confidence "medium"
- If uncertain, mark confidence "low"
- For COLUMN references (not tables), set canonical to null and type to "COLUMN"

Respond ONLY with valid JSON. No markdown, no explanation."""

_EXTRACTION_PROMPT = """Extract all table/dataset references from this code chunk.

File: {file_path}
Type: {file_type}
Lines: {line_start}-{line_end}

Code:
{code}

Respond with a JSON array of objects, each with:
  raw        : string  — exact name as it appears in code
  canonical  : string|null — inferred real table name (null for columns)
  type       : "READ"|"WRITE"|"CREATE"|"DROP"|"COLUMN"|"UNKNOWN"
  confidence : "high"|"medium"|"low"

Example:
[
  {{"raw": "cust_tab", "canonical": "customer", "type": "READ", "confidence": "high"}},
  {{"raw": "orders",   "canonical": "orders",   "type": "READ", "confidence": "high"}}
]

If no table references found, return an empty array: []"""


@dataclass
class EntityRef:
    raw: str
    canonical: str | None
    ref_type: str
    confidence: str


@dataclass
class ChunkEntities:
    chunk_id: str
    file_path: str
    refs: List[EntityRef] = field(default_factory=list)


def extract_entities(
    client: anthropic.Anthropic,
    chunk: CodeChunk,
    model: str = "claude-opus-4-7",
) -> ChunkEntities:
    """Extract entity references from a single code chunk using Claude API."""
    user_prompt = _EXTRACTION_PROMPT.format(
        file_path=chunk.file_path,
        file_type=chunk.file_type,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        code=chunk.raw_text[:4000],  # cap to avoid token bloat
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": _SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "[]")
    text = text.strip()
    # strip markdown code fences if Claude wraps the JSON
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        raw_refs = json.loads(text)
    except json.JSONDecodeError:
        raw_refs = []

    refs = []
    for r in raw_refs:
        if not isinstance(r, dict) or "raw" not in r:
            continue
        refs.append(EntityRef(
            raw=r.get("raw", ""),
            canonical=r.get("canonical"),
            ref_type=r.get("type", "UNKNOWN"),
            confidence=r.get("confidence", "low"),
        ))

    return ChunkEntities(chunk_id=chunk.chunk_id, file_path=chunk.file_path, refs=refs)


def batch_extract_entities(
    client: anthropic.Anthropic,
    chunks: List[CodeChunk],
    model: str = "claude-opus-4-7",
) -> List[ChunkEntities]:
    """Extract entities from a list of chunks sequentially."""
    results = []
    for chunk in chunks:
        # skip empty or trivially small chunks
        if len(chunk.raw_text.strip()) < 10:
            continue
        result = extract_entities(client, chunk, model)
        results.append(result)
    return results
