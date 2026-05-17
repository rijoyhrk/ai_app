import ast
import uuid
import textwrap
from typing import List
from .base import CodeChunk


def chunk_python(file_path: str, source: str) -> List[CodeChunk]:
    """
    Parse Python source into logical chunks via AST:
    - Each top-level function/class = one chunk
    - Top-level code blocks (imports, assignments, bare calls) grouped together
    - Nested functions/methods included within their parent chunk
    Falls back to line-block splitting if AST parse fails.
    """
    chunks: List[CodeChunk] = []
    lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _fallback_python_chunks(file_path, source)

    top_level_nodes = [n for n in ast.iter_child_nodes(tree)]
    preamble_lines: List[str] = []
    preamble_start = 1

    for node in top_level_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if preamble_lines:
                chunks.append(_make_chunk(file_path, "\n".join(preamble_lines),
                                          preamble_start, node.lineno - 1, "module_preamble"))
                preamble_lines = []

            chunk_text = _extract_lines(lines, node.lineno, node.end_lineno)
            chunks.append(_make_chunk(file_path, chunk_text,
                                      node.lineno, node.end_lineno, "function"))

        elif isinstance(node, ast.ClassDef):
            if preamble_lines:
                chunks.append(_make_chunk(file_path, "\n".join(preamble_lines),
                                          preamble_start, node.lineno - 1, "module_preamble"))
                preamble_lines = []

            chunk_text = _extract_lines(lines, node.lineno, node.end_lineno)
            chunks.append(_make_chunk(file_path, chunk_text,
                                      node.lineno, node.end_lineno, "class"))
        else:
            node_lines = _extract_lines(lines, node.lineno, getattr(node, "end_lineno", node.lineno))
            if not preamble_lines:
                preamble_start = node.lineno
            preamble_lines.append(node_lines)

    if preamble_lines:
        chunks.append(_make_chunk(file_path, "\n".join(preamble_lines),
                                  preamble_start, len(lines), "module_preamble"))

    return chunks or _fallback_python_chunks(file_path, source)


def _extract_lines(lines: List[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1:end])


def _make_chunk(file_path: str, text: str, start: int, end: int, chunk_type: str) -> CodeChunk:
    return CodeChunk(
        chunk_id=str(uuid.uuid4()),
        file_path=file_path,
        file_type="python",
        chunk_type=chunk_type,
        raw_text=textwrap.dedent(text),
        line_start=start,
        line_end=end,
    )


def _fallback_python_chunks(file_path: str, source: str) -> List[CodeChunk]:
    """Group lines into ~30-line blocks when AST parse fails."""
    lines = source.splitlines()
    block_size = 30
    chunks = []
    for i in range(0, len(lines), block_size):
        block = lines[i:i + block_size]
        chunks.append(CodeChunk(
            chunk_id=str(uuid.uuid4()),
            file_path=file_path,
            file_type="python",
            chunk_type="block",
            raw_text="\n".join(block),
            line_start=i + 1,
            line_end=min(i + block_size, len(lines)),
        ))
    return chunks
