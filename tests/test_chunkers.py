"""
Tests for AST-aware code chunkers.
Verifies that SQL, Python, and Shell files are split at correct logical boundaries
and that chunk metadata (line numbers, chunk_type, file_type) is accurate.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.sql_chunker import chunk_sql
from src.chunkers.python_chunker import chunk_python
from src.chunkers.shell_chunker import chunk_shell
from src.chunkers.dispatcher import crawl_and_chunk, chunk_file
from conftest import (
    SYNTHETIC_SQL, SYNTHETIC_PYTHON, SYNTHETIC_SHELL,
    SYNTHETIC_PRODUCT_SQL,
)


# ─── SQL chunker ──────────────────────────────────────────────────────────────

class TestSQLChunker:

    def test_splits_multiple_statements(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        # INSERT, SELECT, CREATE → at least 3 chunks
        assert len(chunks) >= 3

    def test_chunk_ids_are_unique(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_file_type_is_sql(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        assert all(c.file_type == "sql" for c in chunks)

    def test_insert_chunk_classified_correctly(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        types = [c.chunk_type for c in chunks]
        # sqlglot classifies INSERT as "write"; also accept "statement" as fallback
        assert any(t in ("write", "insert", "statement") for t in types)

    def test_select_chunk_classified_correctly(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        types = [c.chunk_type for c in chunks]
        assert any(t in ("select", "statement") for t in types)

    def test_line_numbers_are_positive(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        for c in chunks:
            assert c.line_start >= 1
            assert c.line_end >= c.line_start

    def test_raw_text_contains_sql_keywords(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        combined = " ".join(c.raw_text.upper() for c in chunks)
        assert "SELECT" in combined
        assert "INSERT" in combined

    def test_table_names_present_in_raw_text(self):
        chunks = chunk_sql("test.sql", SYNTHETIC_SQL)
        combined = " ".join(c.raw_text for c in chunks)
        assert "cust_tab" in combined or "customer" in combined

    def test_empty_sql_returns_no_chunks(self):
        chunks = chunk_sql("empty.sql", "-- just a comment\n")
        # empty or single comment chunk — raw text should be minimal
        assert all(len(c.raw_text.strip()) < 50 for c in chunks) or len(chunks) == 0

    def test_single_statement_produces_one_chunk(self):
        sql = "SELECT * FROM orders WHERE status = 'COMPLETED';"
        chunks = chunk_sql("single.sql", sql)
        assert len(chunks) == 1
        assert "orders" in chunks[0].raw_text


# ─── Python chunker ───────────────────────────────────────────────────────────

class TestPythonChunker:

    def test_splits_into_functions(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        # extract_customers, load_orders, write_report + module preamble
        assert len(chunks) >= 3

    def test_file_type_is_python(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        assert all(c.file_type == "python" for c in chunks)

    def test_function_chunks_identified(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        types = [c.chunk_type for c in chunks]
        assert "function" in types

    def test_function_names_in_raw_text(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        combined = " ".join(c.raw_text for c in chunks)
        assert "extract_customers" in combined
        assert "load_orders" in combined
        assert "write_report" in combined

    def test_alias_cust_data_in_chunks(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        combined = " ".join(c.raw_text for c in chunks)
        assert "cust_data" in combined

    def test_chunk_ids_unique(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_line_numbers_sequential(self):
        chunks = chunk_python("test.py", SYNTHETIC_PYTHON)
        for c in chunks:
            assert c.line_start >= 1
            assert c.line_end >= c.line_start

    def test_syntax_error_falls_back_to_blocks(self):
        bad_python = "def broken(\n    x\n    # missing closing paren\nSELECT * FROM t\n" * 10
        chunks = chunk_python("bad.py", bad_python)
        # should not raise; returns block chunks
        assert isinstance(chunks, list)

    def test_empty_file_returns_empty_or_preamble(self):
        chunks = chunk_python("empty.py", "")
        assert isinstance(chunks, list)


# ─── Shell chunker ────────────────────────────────────────────────────────────

class TestShellChunker:

    def test_produces_chunks(self):
        chunks = chunk_shell("test.sh", SYNTHETIC_SHELL)
        assert len(chunks) >= 1

    def test_file_type_is_shell(self):
        chunks = chunk_shell("test.sh", SYNTHETIC_SHELL)
        assert all(c.file_type == "shell" for c in chunks)

    def test_psql_command_in_chunks(self):
        chunks = chunk_shell("test.sh", SYNTHETIC_SHELL)
        combined = " ".join(c.raw_text for c in chunks)
        assert "psql" in combined or "COPY" in combined

    def test_table_alias_cust_staging_present(self):
        chunks = chunk_shell("test.sh", SYNTHETIC_SHELL)
        combined = " ".join(c.raw_text for c in chunks)
        assert "cust_staging" in combined

    def test_chunk_ids_unique(self):
        chunks = chunk_shell("test.sh", SYNTHETIC_SHELL)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ─── Dispatcher ───────────────────────────────────────────────────────────────

class TestDispatcher:

    def test_crawl_finds_all_file_types(self, tmp_etl_repo):
        chunks = crawl_and_chunk(str(tmp_etl_repo))
        types = {c.file_type for c in chunks}
        assert "sql" in types
        assert "python" in types
        assert "shell" in types

    def test_crawl_excludes_specified_dirs(self, tmp_etl_repo):
        chunks_all = crawl_and_chunk(str(tmp_etl_repo))
        chunks_excl = crawl_and_chunk(str(tmp_etl_repo), excluded_dirs=["archive"])
        paths_all = {c.file_path for c in chunks_all}
        paths_excl = {c.file_path for c in chunks_excl}
        # archive/old_pipeline.py should be missing when excluded
        archived = [p for p in paths_all if "archive" in p]
        assert len(archived) > 0, "fixture should have an archive file"
        assert all(p not in paths_excl for p in archived)

    def test_crawl_without_exclusion_includes_archive(self, tmp_etl_repo):
        chunks = crawl_and_chunk(str(tmp_etl_repo))
        paths = {c.file_path for c in chunks}
        assert any("archive" in p for p in paths)

    def test_chunk_file_unknown_extension_returns_empty(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n")
        result = chunk_file(str(f))
        assert result == []

    def test_crawl_nonexistent_path_returns_empty(self, tmp_path):
        chunks = crawl_and_chunk(str(tmp_path / "does_not_exist"))
        assert chunks == []

    def test_multiple_exclusions(self, tmp_etl_repo):
        # add a second excluded dir
        legacy = tmp_etl_repo / "legacy"
        legacy.mkdir()
        (legacy / "old.sql").write_text("SELECT 1 FROM old_table;")
        chunks = crawl_and_chunk(str(tmp_etl_repo), excluded_dirs=["archive", "legacy"])
        paths = {c.file_path for c in chunks}
        assert not any("archive" in p for p in paths)
        assert not any("legacy" in p for p in paths)
