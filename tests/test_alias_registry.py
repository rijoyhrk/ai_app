"""
Tests for AliasRegistry — the bidirectional alias→canonical map.
All tests are pure in-memory with no external dependencies.
"""
import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.alias_registry import AliasRegistry
from src.extractors.entity_extractor import ChunkEntities, EntityRef


def _make_entities(chunk_id, file_path, refs):
    return ChunkEntities(
        chunk_id=chunk_id,
        file_path=file_path,
        refs=[EntityRef(**r) for r in refs],
    )


class TestAliasRegistryIngest:

    def test_ingest_single_alias(self):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])])
        assert registry.resolve("cust_tab") == "customer"

    def test_ingest_canonical_maps_to_itself(self):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "customer", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])])
        assert registry.resolve("customer") == "customer"

    def test_ingest_multiple_aliases_same_canonical(self):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.py", [
            {"raw": "cust_tab",  "canonical": "customer", "ref_type": "READ", "confidence": "high"},
            {"raw": "cust_data", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
            {"raw": "CUST",      "canonical": "customer", "ref_type": "READ", "confidence": "medium"},
        ])])
        aliases = registry.get_all_aliases("customer")
        assert "cust_tab"  in aliases
        assert "cust_data" in aliases
        assert "cust"      in aliases  # lowercased

    def test_ingest_skips_column_refs(self):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "cust_id", "canonical": None, "ref_type": "COLUMN", "confidence": "high"},
        ])])
        assert registry.all_canonicals() == []

    def test_ingest_skips_null_canonical(self):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "something", "canonical": None, "ref_type": "READ", "confidence": "low"},
        ])])
        assert registry.all_canonicals() == []

    def test_ingest_multiple_files_accumulates(self):
        registry = AliasRegistry()
        registry.ingest([
            _make_entities("c1", "file_a.py",  [{"raw": "cust_tab",  "canonical": "customer", "ref_type": "READ",  "confidence": "high"}]),
            _make_entities("c2", "file_b.sql", [{"raw": "cust_data", "canonical": "customer", "ref_type": "WRITE", "confidence": "high"}]),
        ])
        files = registry.get_files_for("customer")
        assert "file_a.py"  in files
        assert "file_b.sql" in files

    def test_ingest_different_canonicals(self):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ",  "confidence": "high"},
            {"raw": "ord_tab",  "canonical": "orders",   "ref_type": "READ",  "confidence": "high"},
            {"raw": "prod_tbl", "canonical": "product",  "ref_type": "WRITE", "confidence": "high"},
        ])])
        assert set(registry.all_canonicals()) == {"customer", "orders", "product"}


class TestAliasRegistryResolve:

    def test_resolve_known_alias(self, populated_registry):
        assert populated_registry.resolve("cust_data") == "customer"
        assert populated_registry.resolve("cust_tab")  == "customer"

    def test_resolve_canonical_returns_itself(self, populated_registry):
        assert populated_registry.resolve("customer") == "customer"

    def test_resolve_unknown_returns_lowercased_input(self, populated_registry):
        assert populated_registry.resolve("UNKNOWN_TABLE") == "unknown_table"

    def test_resolve_case_insensitive(self, populated_registry):
        assert populated_registry.resolve("CUST_TAB") == "customer"
        assert populated_registry.resolve("Customer")  == "customer"

    def test_get_all_aliases_includes_canonical(self, populated_registry):
        aliases = populated_registry.get_all_aliases("customer")
        assert "customer" in aliases

    def test_get_all_aliases_includes_all_variants(self, populated_registry):
        aliases = populated_registry.get_all_aliases("customer")
        assert "cust_tab"  in aliases
        assert "cust_data" in aliases

    def test_get_files_for_returns_correct_files(self, populated_registry):
        files = populated_registry.get_files_for("customer")
        assert "pipeline.py" in files

    def test_get_files_for_unknown_returns_empty(self, populated_registry):
        assert populated_registry.get_files_for("nonexistent_table") == []

    def test_all_canonicals_sorted(self, populated_registry):
        canonicals = populated_registry.all_canonicals()
        assert canonicals == sorted(canonicals)


class TestAliasRegistryPersistence:

    def test_save_and_load_roundtrip(self, tmp_registry_path):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "cust_tab", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
            {"raw": "ord_tab",  "canonical": "orders",   "ref_type": "READ", "confidence": "high"},
        ])])
        registry.save(tmp_registry_path)

        loaded = AliasRegistry.load(tmp_registry_path)
        assert loaded.resolve("cust_tab") == "customer"
        assert loaded.resolve("ord_tab")  == "orders"
        assert "customer" in loaded.all_canonicals()

    def test_load_nonexistent_returns_empty_registry(self, tmp_path):
        registry = AliasRegistry.load(str(tmp_path / "missing.json"))
        assert registry.all_canonicals() == []

    def test_save_creates_parent_dirs(self, tmp_path):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "customer", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])])
        deep_path = str(tmp_path / "a" / "b" / "c" / "registry.json")
        registry.save(deep_path)
        assert Path(deep_path).exists()

    def test_saved_json_is_valid(self, tmp_registry_path):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.sql", [
            {"raw": "prod_tbl", "canonical": "product", "ref_type": "WRITE", "confidence": "high"},
        ])])
        registry.save(tmp_registry_path)
        data = json.loads(Path(tmp_registry_path).read_text())
        assert "product" in data
        assert "prod_tbl" in data["product"]["aliases"]

    def test_load_preserves_alias_to_canonical_map(self, tmp_registry_path):
        registry = AliasRegistry()
        registry.ingest([_make_entities("c1", "f.py", [
            {"raw": "cust_data", "canonical": "customer", "ref_type": "READ", "confidence": "high"},
        ])])
        registry.save(tmp_registry_path)

        loaded = AliasRegistry.load(tmp_registry_path)
        assert loaded.resolve("cust_data") == "customer"

    def test_summary_contains_canonical_names(self, populated_registry):
        summary = populated_registry.summary()
        assert "customer" in summary
        assert "orders"   in summary
