from pathlib import Path
from typing import List
from .base import CodeChunk
from .sql_chunker import chunk_sql
from .python_chunker import chunk_python
from .shell_chunker import chunk_shell

_EXTENSION_MAP = {
    ".sql": ("sql", chunk_sql),
    ".py":  ("python", chunk_python),
    ".sh":  ("shell", chunk_shell),
}


def chunk_file(file_path: str) -> List[CodeChunk]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in _EXTENSION_MAP:
        return []

    _, chunker_fn = _EXTENSION_MAP[ext]
    source = path.read_text(encoding="utf-8", errors="replace")
    return chunker_fn(file_path, source)


def crawl_and_chunk(repo_path: str) -> List[CodeChunk]:
    root = Path(repo_path)
    all_chunks: List[CodeChunk] = []
    for ext in _EXTENSION_MAP:
        for file_path in root.rglob(f"*{ext}"):
            all_chunks.extend(chunk_file(str(file_path)))
    return all_chunks
