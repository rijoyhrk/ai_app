from pathlib import Path
from typing import List, Set, Optional
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


def _is_excluded(file_path: Path, root: Path, excluded_dirs: Set[str]) -> bool:
    """Return True if any part of file_path (relative to root) matches an excluded dir name."""
    try:
        relative = file_path.relative_to(root)
    except ValueError:
        relative = file_path
    return any(part in excluded_dirs for part in relative.parts)


def crawl_and_chunk(
    repo_path: str,
    excluded_dirs: Optional[List[str]] = None,
) -> List[CodeChunk]:
    """
    Walk repo_path recursively, chunk every .sql/.py/.sh file found.
    Any file whose path contains a directory name listed in excluded_dirs is skipped.
    """
    root = Path(repo_path)
    exclude_set: Set[str] = set(excluded_dirs or [])
    all_chunks: List[CodeChunk] = []

    for ext in _EXTENSION_MAP:
        for file_path in root.rglob(f"*{ext}"):
            if exclude_set and _is_excluded(file_path, root, exclude_set):
                continue
            all_chunks.extend(chunk_file(str(file_path)))

    return all_chunks
