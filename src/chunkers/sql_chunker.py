import uuid
import sqlglot
from sqlglot import exp
from typing import List
from .base import CodeChunk


def _line_of_offset(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def chunk_sql(file_path: str, source: str) -> List[CodeChunk]:
    """
    Parse SQL source into per-statement chunks using sqlglot AST.
    Each SELECT / INSERT / CREATE / UPDATE / DELETE / CTE block = one chunk.
    Falls back to semicolon splitting when sqlglot cannot parse.
    """
    chunks: List[CodeChunk] = []
    lines = source.splitlines()

    try:
        statements = sqlglot.parse(source, error_level=sqlglot.ErrorLevel.WARN)
    except Exception:
        statements = []

    if statements:
        for stmt in statements:
            if stmt is None:
                continue
            sql_text = stmt.sql(pretty=True)
            # determine line range via string search in original source
            stmt_raw = stmt.sql()
            try:
                start_idx = source.index(stmt_raw.split()[0])
                start_line = _line_of_offset(source, start_idx)
            except (ValueError, IndexError):
                start_line = 1
            end_line = start_line + sql_text.count("\n")

            chunk_type = _classify_sql_stmt(stmt)
            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                file_path=file_path,
                file_type="sql",
                chunk_type=chunk_type,
                raw_text=sql_text,
                line_start=start_line,
                line_end=min(end_line, len(lines)),
            ))
    else:
        # Fallback: split on semicolons
        parts = [p.strip() for p in source.split(";") if p.strip()]
        current_line = 1
        for part in parts:
            end_line = current_line + part.count("\n")
            chunks.append(CodeChunk(
                chunk_id=str(uuid.uuid4()),
                file_path=file_path,
                file_type="sql",
                chunk_type="statement",
                raw_text=part,
                line_start=current_line,
                line_end=end_line,
            ))
            current_line = end_line + 2

    return chunks


def _classify_sql_stmt(stmt) -> str:
    if isinstance(stmt, exp.Select):
        return "select"
    if isinstance(stmt, (exp.Insert, exp.Create)):
        return "write"
    if isinstance(stmt, exp.Update):
        return "update"
    if isinstance(stmt, exp.Delete):
        return "delete"
    if isinstance(stmt, exp.With):
        return "cte"
    return "statement"
