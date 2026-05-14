import re
import uuid
from typing import List
from .base import CodeChunk

# Patterns that signal a new logical pipeline/command group
_BOUNDARY_PATTERNS = [
    re.compile(r"^\s*#.*$"),                    # comment lines as section separators
    re.compile(r"^\s*function\s+\w+\s*\(?\)?"), # function definitions
    re.compile(r"^\s*(if|for|while|until|case)\s"),
    re.compile(r"^\s*(psql|bq|spark-submit|hive|sqoop|airflow)\s"),
    re.compile(r"^\s*\w+="),                     # variable assignments
]

_FUNCTION_DEF = re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\s*\)\s*\{?")


def chunk_shell(file_path: str, source: str) -> List[CodeChunk]:
    """
    Split shell scripts into logical command groups:
    - Function definitions become individual chunks
    - Comment-separated sections become chunks
    - Consecutive pipeline commands grouped together
    """
    lines = source.splitlines()
    chunks: List[CodeChunk] = []
    current_block: List[str] = []
    block_start = 1
    in_function = False
    brace_depth = 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip blank lines between blocks
        if not stripped:
            if current_block:
                current_block.append(line)
            continue

        # Detect function start
        if _FUNCTION_DEF.match(stripped):
            if current_block:
                chunks.append(_make_chunk(file_path, current_block, block_start, i - 1, "command_group"))
                current_block = []
            block_start = i
            in_function = True
            brace_depth = stripped.count("{") - stripped.count("}")
            current_block.append(line)
            continue

        if in_function:
            current_block.append(line)
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                chunks.append(_make_chunk(file_path, current_block, block_start, i, "function"))
                current_block = []
                block_start = i + 1
                in_function = False
            continue

        # New section if we hit a strong boundary and already have content
        is_boundary = any(p.match(stripped) for p in _BOUNDARY_PATTERNS)
        if is_boundary and current_block and stripped.startswith("#"):
            chunks.append(_make_chunk(file_path, current_block, block_start, i - 1, "command_group"))
            current_block = []
            block_start = i

        current_block.append(line)

    if current_block:
        chunks.append(_make_chunk(file_path, current_block, block_start, len(lines), "command_group"))

    return chunks or [CodeChunk(
        chunk_id=str(uuid.uuid4()),
        file_path=file_path,
        file_type="shell",
        chunk_type="script",
        raw_text=source,
        line_start=1,
        line_end=len(lines),
    )]


def _make_chunk(file_path: str, block: List[str], start: int, end: int, chunk_type: str) -> CodeChunk:
    text = "\n".join(block).strip()
    return CodeChunk(
        chunk_id=str(uuid.uuid4()),
        file_path=file_path,
        file_type="shell",
        chunk_type=chunk_type,
        raw_text=text,
        line_start=start,
        line_end=end,
    )
