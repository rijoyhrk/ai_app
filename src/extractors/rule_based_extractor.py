"""
Rule-based table name extractor — no LLM required.

Extracts table references from SQL, Python, and Shell code using regex patterns.
Does not perform alias resolution (cust_tab stays as cust_tab, not customer).
Used as a fallback when the Claude API is unavailable.
"""
import re
from typing import List
from src.chunkers.base import CodeChunk
from src.extractors.entity_extractor import ChunkEntities, EntityRef

# ── SQL patterns ───────────────────────────────────────────────────────────────
_SQL_READ_PATTERNS = [
    r'\bFROM\s+([`"\[]?[\w.]+[`"\]]?)',
    r'\bJOIN\s+([`"\[]?[\w.]+[`"\]]?)',
]
_SQL_WRITE_PATTERNS = [
    r'\bINSERT\s+INTO\s+([`"\[]?[\w.]+[`"\]]?)',
    r'\bUPDATE\s+([`"\[]?[\w.]+[`"\]]?)',
    r'\bMERGE\s+INTO\s+([`"\[]?[\w.]+[`"\]]?)',
]
_SQL_CREATE_PATTERNS = [
    r'\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?([`"\[]?[\w.]+[`"\]]?)',
]
_SQL_DROP_PATTERNS = [
    r'\bDROP\s+(?:TABLE|VIEW)\s+(?:IF\s+EXISTS\s+)?([`"\[]?[\w.]+[`"\]]?)',
]
# COPY tablename FROM / INTO tablename
_SQL_COPY_PATTERNS = [
    r'\bCOPY\s+([`"\[]?[\w.]+[`"\]]?)\s+\(',
    r'\bINTO\s+([`"\[]?[\w.]+[`"\]]?)',
]

# ── Python patterns ────────────────────────────────────────────────────────────
_PY_READ_PATTERNS = [
    r'\bFROM\s+([`"\']?[\w.]+[`"\']?)',             # SQL strings embedded in Python
    r'\bJOIN\s+([`"\']?[\w.]+[`"\']?)',
    r'read_sql\s*\(\s*["\'].*?FROM\s+([\w.]+)',      # pd.read_sql("SELECT ... FROM tbl")
    r'\.read_table\s*\(\s*["\']([^"\']+)["\']',      # spark.read.table("tbl")
]
_PY_WRITE_PATTERNS = [
    r'\.to_sql\s*\(\s*["\']([^"\']+)["\']',          # df.to_sql("tbl", ...)
    r'\.insertInto\s*\(\s*["\']([^"\']+)["\']',      # spark df.write.insertInto("tbl")
    r'\.saveAsTable\s*\(\s*["\']([^"\']+)["\']',
    r'\bINSERT\s+INTO\s+([\w.]+)',
]

# ── Shell patterns ─────────────────────────────────────────────────────────────
_SH_PATTERNS = [
    r'\bCOPY\s+([\w.]+)\s*\(',                        # COPY tbl (cols) FROM ...
    r'\bFROM\s+([\w.]+)',
    r'\bINTO\s+([\w.]+)',
    r'\bSELECT\s+COUNT\(\*\)\s+FROM\s+([\w.]+)',
]

# Words that look like table names but are SQL keywords / noise
_STOP_WORDS = {
    "select", "from", "where", "join", "on", "as", "and", "or", "not",
    "in", "is", "null", "true", "false", "by", "group", "order", "having",
    "limit", "offset", "union", "all", "distinct", "case", "when", "then",
    "else", "end", "cast", "coalesce", "count", "sum", "avg", "max", "min",
    "date", "timestamp", "interval", "varchar", "int", "integer", "bigint",
    "float", "double", "boolean", "string", "array", "map", "struct",
    "current_date", "current_timestamp", "now", "values", "set", "with",
    "recursive", "lateral", "cross", "inner", "left", "right", "full",
    "outer", "natural", "using", "exists", "between", "like", "ilike",
    "replace", "engine", "index", "key", "primary", "default",
}


def _clean(name: str) -> str:
    """Strip quotes, backticks, brackets, schema prefix."""
    name = re.sub(r'[`"\[\]]', '', name).strip()
    # strip schema prefix (schema.table → table)
    if "." in name:
        name = name.split(".")[-1]
    return name.lower()


def _extract_from_text(text: str, patterns: list, ref_type: str) -> List[EntityRef]:
    refs = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = _clean(match.group(1))
            if not raw or raw in _STOP_WORDS or len(raw) < 2 or raw in seen:
                continue
            # skip pure numbers or very short tokens that are likely not table names
            if re.match(r'^\d+$', raw):
                continue
            seen.add(raw)
            refs.append(EntityRef(
                raw=raw,
                canonical=raw,   # no alias resolution in rule-based mode
                ref_type=ref_type,
                confidence="medium",
            ))
    return refs


def extract_entities_rule_based(chunk: CodeChunk) -> ChunkEntities:
    """Extract table references using regex — no LLM, no API calls."""
    text = chunk.raw_text
    ft   = chunk.file_type
    refs: List[EntityRef] = []

    if ft == "sql":
        refs += _extract_from_text(text, _SQL_READ_PATTERNS,   "READ")
        refs += _extract_from_text(text, _SQL_WRITE_PATTERNS,  "WRITE")
        refs += _extract_from_text(text, _SQL_CREATE_PATTERNS, "CREATE")
        refs += _extract_from_text(text, _SQL_DROP_PATTERNS,   "DROP")
        refs += _extract_from_text(text, _SQL_COPY_PATTERNS,   "WRITE")
    elif ft == "python":
        refs += _extract_from_text(text, _PY_READ_PATTERNS,    "READ")
        refs += _extract_from_text(text, _PY_WRITE_PATTERNS,   "WRITE")
    elif ft == "shell":
        refs += _extract_from_text(text, _SH_PATTERNS,         "READ")

    # deduplicate: if same raw name appears as both READ and WRITE, keep WRITE
    seen: dict = {}
    for ref in refs:
        if ref.raw not in seen or ref.ref_type in ("WRITE", "CREATE", "DROP"):
            seen[ref.raw] = ref
    deduped = list(seen.values())

    return ChunkEntities(
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        refs=deduped,
    )


def batch_extract_entities_rule_based(chunks: List[CodeChunk]) -> List[ChunkEntities]:
    return [
        extract_entities_rule_based(chunk)
        for chunk in chunks
        if len(chunk.raw_text.strip()) >= 10
    ]
