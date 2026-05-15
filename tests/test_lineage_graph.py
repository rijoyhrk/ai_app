"""
Tests for lineage graph construction and PyVis rendering.
No external dependencies — uses NetworkX in-memory.
"""
import sys
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.lineage_graph import LineageGraph


def _result(chunk_id, file_path, operation, canonical="customer",
            line_start=1, line_end=10, file_type="sql"):
    return {
        "chunk_id": chunk_id,
        "operation": operation,
        "final_score": 0.85,
        "matched_canonical": canonical,
        "explanation": f"{operation} operation on {canonical}",
        "matched_via": "direct",
        "metadata": {
            "file_path": file_path,
            "file_type": file_type,
            "line_start": line_start,
            "line_end": line_end,
        },
    }


class TestLineageGraphBuild:

    def test_empty_results_produce_empty_graph(self):
        graph = LineageGraph()
        graph.build_from_results([])
        assert graph.get_graph().number_of_nodes() == 0

    def test_read_result_adds_table_and_file_nodes(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "pipeline.py", "READ"))
        G = graph.get_graph()
        assert "table:customer" in G.nodes
        assert "file:pipeline.py" in G.nodes

    def test_read_creates_table_to_file_edge(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "pipeline.py", "READ"))
        G = graph.get_graph()
        assert G.has_edge("table:customer", "file:pipeline.py")

    def test_write_creates_file_to_table_edge(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "load.sql", "WRITE"))
        G = graph.get_graph()
        assert G.has_edge("file:load.sql", "table:customer")

    def test_create_creates_file_to_table_edge(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "setup.sql", "CREATE"))
        G = graph.get_graph()
        assert G.has_edge("file:setup.sql", "table:customer")

    def test_insert_creates_file_to_table_edge(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "etl.sql", "INSERT"))
        G = graph.get_graph()
        assert G.has_edge("file:etl.sql", "table:customer")

    def test_multiple_results_build_correctly(self, sample_hits):
        graph = LineageGraph()
        graph.build_from_results([
            _result("c1", "read.py",  "READ",  canonical="customer"),
            _result("c2", "write.sql","WRITE", canonical="customer"),
            _result("c3", "other.py", "READ",  canonical="orders"),
        ])
        G = graph.get_graph()
        assert G.number_of_nodes() >= 4  # customer, orders, read.py, write.sql, other.py
        assert G.number_of_edges() >= 3

    def test_duplicate_file_node_not_duplicated(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "pipeline.py", "READ",  line_start=1,  line_end=10))
        graph.add_result(_result("c2", "pipeline.py", "WRITE", line_start=20, line_end=30))
        G = graph.get_graph()
        file_nodes = [n for n in G.nodes if n.startswith("file:pipeline.py")]
        assert len(file_nodes) == 1

    def test_node_attributes_set(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "pipeline.py", "READ"))
        G = graph.get_graph()
        table_attrs = G.nodes["table:customer"]
        file_attrs  = G.nodes["file:pipeline.py"]
        assert table_attrs["node_type"] == "table"
        assert file_attrs["node_type"]  == "file"
        assert table_attrs["label"]     == "customer"
        assert file_attrs["label"]      == "pipeline.py"

    def test_edge_attributes_include_operation(self):
        graph = LineageGraph()
        graph.add_result(_result("c1", "pipeline.py", "READ"))
        G = graph.get_graph()
        edge_data = G.get_edge_data("table:customer", "file:pipeline.py")
        assert edge_data["operation"] == "READ"


class TestLineageGraphSummary:

    def test_summary_groups_by_operation(self):
        graph = LineageGraph()
        graph.build_from_results([
            _result("c1", "reader.py",  "READ",  canonical="customer"),
            _result("c2", "writer.sql", "WRITE", canonical="customer"),
        ])
        summary = graph.get_lineage_summary("customer")
        assert "READ"  in summary
        assert "WRITE" in summary

    def test_summary_lists_correct_files(self):
        graph = LineageGraph()
        graph.build_from_results([
            _result("c1", "a.py",  "READ",  canonical="orders"),
            _result("c2", "b.sql", "READ",  canonical="orders"),
            _result("c3", "c.sh",  "WRITE", canonical="orders"),
        ])
        summary = graph.get_lineage_summary("orders")
        read_files = summary.get("READ", [])
        assert any("a.py"  in f for f in read_files)
        assert any("b.sql" in f for f in read_files)

    def test_summary_empty_for_unknown_table(self):
        graph = LineageGraph()
        graph.build_from_results([_result("c1", "a.py", "READ", canonical="customer")])
        summary = graph.get_lineage_summary("nonexistent_table")
        assert summary == {}

    def test_summary_excludes_empty_operation_lists(self):
        graph = LineageGraph()
        graph.build_from_results([_result("c1", "a.py", "READ", canonical="customer")])
        summary = graph.get_lineage_summary("customer")
        # WRITE, CREATE, OTHER should not appear if no such edges exist
        for op, files in summary.items():
            assert len(files) > 0


class TestLineageGraphPyVis:

    def test_pyvis_html_creates_file(self, tmp_path):
        graph = LineageGraph()
        graph.build_from_results([
            _result("c1", "pipeline.py", "READ",  canonical="customer"),
            _result("c2", "load.sql",    "WRITE", canonical="customer"),
        ])
        html_path = str(tmp_path / "lineage.html")
        graph.to_pyvis_html(html_path, title="Test Lineage")
        assert Path(html_path).exists()

    def test_pyvis_html_contains_node_labels(self, tmp_path):
        graph = LineageGraph()
        graph.build_from_results([_result("c1", "pipeline.py", "READ")])
        html_path = str(tmp_path / "lineage.html")
        graph.to_pyvis_html(html_path)
        html = Path(html_path).read_text()
        assert "customer"    in html
        assert "pipeline.py" in html

    def test_pyvis_empty_graph_still_creates_file(self, tmp_path):
        graph = LineageGraph()
        html_path = str(tmp_path / "empty.html")
        graph.to_pyvis_html(html_path)
        assert Path(html_path).exists()
