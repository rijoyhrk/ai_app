"""
Lineage graph builder using NetworkX.
Builds a DAG: table → script relationships with operation edge labels.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import networkx as nx
from src.extractors.alias_registry import AliasRegistry


class LineageGraph:

    def __init__(self):
        self._G = nx.DiGraph()

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a single search result (reranked hit) to the graph."""
        meta = result.get("metadata", {})
        file_path = meta.get("file_path", "unknown")
        file_name = Path(file_path).name
        canonical = result.get("matched_canonical", "unknown_table")
        operation = result.get("operation", "UNKNOWN")
        line_start = meta.get("line_start", "?")
        line_end = meta.get("line_end", "?")

        table_node = f"table:{canonical}"
        file_node = f"file:{file_name}"

        # add nodes with attributes
        if not self._G.has_node(table_node):
            self._G.add_node(table_node, node_type="table", label=canonical)

        if not self._G.has_node(file_node):
            self._G.add_node(file_node, node_type="file", label=file_name,
                             full_path=file_path, file_type=meta.get("file_type", ""))

        # add edge: direction depends on operation
        edge_key = (table_node, file_node, operation)
        if operation in ("WRITE", "CREATE", "INSERT", "UPDATE", "DELETE", "DROP"):
            # file writes to table
            self._G.add_edge(
                file_node, table_node,
                operation=operation,
                lines=f"{line_start}-{line_end}",
                score=result.get("final_score", 0),
            )
        else:
            # file reads from table (READ, UNKNOWN)
            self._G.add_edge(
                table_node, file_node,
                operation=operation,
                lines=f"{line_start}-{line_end}",
                score=result.get("final_score", 0),
            )

    def build_from_results(
        self,
        results: List[Dict[str, Any]],
        registry: Optional[AliasRegistry] = None,
    ) -> None:
        for result in results:
            self.add_result(result)

    def get_graph(self) -> nx.DiGraph:
        return self._G

    def get_lineage_summary(self, canonical_table: str) -> Dict[str, List[str]]:
        """Return dict of operation type → list of files for a given table."""
        table_node = f"table:{canonical_table}"
        summary: Dict[str, List[str]] = {"READ": [], "WRITE": [], "CREATE": [], "OTHER": []}

        for u, v, data in self._G.edges(data=True):
            op = data.get("operation", "OTHER")
            lines = data.get("lines", "")
            if u == table_node:
                # table → file (READ)
                label = Path(v.replace("file:", "")).name
                summary.setdefault("READ", []).append(f"{label} (lines {lines})")
            elif v == table_node:
                # file → table (WRITE/CREATE/etc.)
                label = Path(u.replace("file:", "")).name
                summary.setdefault(op, []).append(f"{label} (lines {lines})")

        return {k: v for k, v in summary.items() if v}

    def to_pyvis_html(self, output_path: str, title: str = "Data Lineage Graph") -> str:
        """Render the graph as an interactive PyVis HTML file."""
        try:
            from pyvis.network import Network
        except ImportError:
            return ""

        net = Network(
            height="600px",
            width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="white",
            notebook=False,
        )
        net.set_options("""
        {
          "physics": {"enabled": true, "stabilization": {"iterations": 100}},
          "edges": {"arrows": {"to": {"enabled": true}}},
          "nodes": {"font": {"size": 14}}
        }
        """)

        color_map = {"table": "#e94560", "file": "#0f3460"}
        shape_map = {"table": "diamond", "file": "box"}

        for node, attrs in self._G.nodes(data=True):
            ntype = attrs.get("node_type", "file")
            label = attrs.get("label", node)
            net.add_node(
                node,
                label=label,
                color=color_map.get(ntype, "#888"),
                shape=shape_map.get(ntype, "box"),
                title=f"{ntype}: {attrs.get('full_path', label)}",
            )

        op_colors = {"READ": "#53d8fb", "WRITE": "#ff6b6b",
                     "CREATE": "#ffd166", "DROP": "#ef233c"}
        for u, v, data in self._G.edges(data=True):
            op = data.get("operation", "")
            lines = data.get("lines", "")
            net.add_edge(
                u, v,
                label=op,
                title=f"{op} (lines {lines})",
                color=op_colors.get(op, "#aaa"),
            )

        net.save_graph(output_path)
        return output_path
