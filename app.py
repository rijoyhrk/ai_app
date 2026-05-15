"""
Data Lineage Discovery Dashboard
RAG-powered entity resolver: enter any table name → see every script that references it.
"""
import os
import sys
import tempfile
import streamlit as st
import anthropic
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.ingestion.pipeline import run_ingestion
from src.ingestion.vector_store import VectorStore
from src.extractors.alias_registry import AliasRegistry
from src.search.hybrid_search import HybridSearchEngine
from src.search.reranker import rerank
from src.graph.lineage_graph import LineageGraph

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Lineage Discovery",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
  .result-card {
      background: #1e1e2e; border-radius: 8px; padding: 16px;
      margin-bottom: 12px; border-left: 4px solid #7c3aed;
  }
  .op-badge {
      display: inline-block; padding: 2px 10px; border-radius: 12px;
      font-size: 0.75rem; font-weight: bold; margin-right: 4px;
  }
  .READ   { background: #1e40af; color: #93c5fd; }
  .WRITE  { background: #7f1d1d; color: #fca5a5; }
  .CREATE { background: #713f12; color: #fde68a; }
  .UNKNOWN{ background: #374151; color: #d1d5db; }
  .summary-box {
      background: #12122a; border-radius: 10px; padding: 18px 22px;
      margin-bottom: 18px; border: 1px solid #7c3aed33;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    for k, v in [
        ("store", None), ("registry", None), ("engine", None),
        ("ingested", False), ("results", []), ("query", ""),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


@st.cache_resource(show_spinner=False)
def get_anthropic_client():
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _load_existing() -> bool:
    try:
        store = VectorStore(settings.chroma_persist_dir)
        registry = AliasRegistry.load(settings.alias_registry_path)
        if store.count() == 0:
            return False
        st.session_state.store = store
        st.session_state.registry = registry
        st.session_state.ingested = True
        return True
    except Exception:
        return False


def _do_search(query: str) -> list:
    store: VectorStore = st.session_state.store
    registry: AliasRegistry = st.session_state.registry
    client = get_anthropic_client()

    if st.session_state.engine is None:
        engine = HybridSearchEngine(store, registry)
        raw = store._collection.get(limit=200, include=["documents", "metadatas"])
        bm25_docs = [
            {
                "chunk_id": doc_id,
                "embedding_text": raw["documents"][i] if raw.get("documents") else "",
                "metadata": raw["metadatas"][i] if raw.get("metadatas") else {},
            }
            for i, doc_id in enumerate(raw.get("ids", []))
        ]
        engine.build_bm25_index(bm25_docs)
        st.session_state.engine = engine

    engine: HybridSearchEngine = st.session_state.engine
    hits = engine.search(query, top_k=15)
    ranked = rerank(client, query, hits, model=settings.llm_model, min_score=0.25)
    return ranked


def _group_by_file(results: list) -> dict:
    """
    Collapse chunk-level results into file-level groups.
    Each group keeps the highest-scoring chunk per operation type.
    Returns: {file_path: {ops, best_score, aliases_used, matched_via, chunks}}
    """
    groups: dict = defaultdict(lambda: {
        "ops": set(),
        "best_score": 0.0,
        "aliases_used": set(),
        "matched_via": "direct",
        "chunks": [],
        "file_type": "",
        "file_name": "",
    })

    for r in results:
        meta = r.get("metadata", {})
        fp = meta.get("file_path", "unknown")
        g = groups[fp]
        g["file_name"] = Path(fp).name
        g["file_type"] = meta.get("file_type", "")
        g["chunks"].append(r)
        op = r.get("operation", "UNKNOWN")
        g["ops"].add(op)
        score = r.get("final_score", 0.0)
        if score > g["best_score"]:
            g["best_score"] = score
        if r.get("matched_via") == "alias":
            g["matched_via"] = "alias"
        raw_aliases = meta.get("aliases_used", "")
        if raw_aliases:
            g["aliases_used"].update(a.strip() for a in raw_aliases.split("|") if a.strip())

    # sort chunks within each file by line_start
    for fp, g in groups.items():
        g["chunks"].sort(key=lambda r: r.get("metadata", {}).get("line_start", 0))

    return dict(sorted(groups.items(), key=lambda x: x[1]["best_score"], reverse=True))


def _op_badge(op: str) -> str:
    cls = op if op in ("READ", "WRITE", "CREATE") else "UNKNOWN"
    return f'<span class="op-badge {cls}">{op}</span>'


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    etl_path = st.text_input("ETL Repo Path", value=settings.etl_repo_path)
    if st.button("🔄 Re-ingest ETL Files", use_container_width=True):
        with st.spinner("Ingesting ETL files — this may take a minute…"):
            client = get_anthropic_client()
            store, registry = run_ingestion(
                repo_path=etl_path,
                chroma_persist_dir=settings.chroma_persist_dir,
                alias_registry_path=settings.alias_registry_path,
                anthropic_client=client,
                llm_model=settings.llm_model,
                verbose=False,
            )
            st.session_state.store = store
            st.session_state.registry = registry
            st.session_state.engine = None
            st.session_state.ingested = True
        st.success(f"Ingested {store.count()} chunks!")

    st.divider()

    if st.session_state.ingested and st.session_state.registry:
        registry: AliasRegistry = st.session_state.registry
        st.subheader("📋 Known Tables & Aliases")
        canonicals = registry.all_canonicals()
        if canonicals:
            for c in sorted(canonicals)[:30]:
                aliases = [a for a in registry.get_all_aliases(c) if a != c]
                tooltip = f"→ {', '.join(aliases)}" if aliases else ""
                st.markdown(f"**`{c}`** {tooltip}")
        else:
            st.info("No tables indexed yet.")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🔍 Data Lineage Discovery Dashboard")
st.markdown("*Enter any table name — find every script that references it, even via aliases.*")

if not st.session_state.ingested:
    _load_existing()

if not st.session_state.ingested:
    st.info("👈 Click **Re-ingest ETL Files** in the sidebar to index your ETL repository.")
    st.stop()

# ─── Search bar ───────────────────────────────────────────────────────────────
st.markdown(f"**{st.session_state.store.count()} chunks indexed** across your ETL repo.")

col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input(
        "Table name",
        placeholder='e.g.  customer   or   orders   or   cust_tab',
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

if search_clicked and query.strip():
    with st.spinner(f"Searching for **{query.strip()}**…"):
        results = _do_search(query.strip())
        st.session_state.results = results
        st.session_state.query = query.strip()

# ─── Results ──────────────────────────────────────────────────────────────────
if not st.session_state.results:
    st.stop()

results = st.session_state.results
q = st.session_state.query
registry: AliasRegistry = st.session_state.registry
canonical = registry.resolve(q)
aliases = [a for a in registry.get_all_aliases(canonical) if a != canonical]
file_groups = _group_by_file(results)
n_files = len(file_groups)

# ── Summary banner ────────────────────────────────────────────────────────────
alias_note = f" (also known as: **{', '.join(aliases)}**)" if aliases else ""
st.markdown(f"""
<div class="summary-box">
  <h3 style="margin:0 0 6px 0;">Table: <code>{canonical}</code>{alias_note if not aliases else ""}</h3>
  {"<p style='margin:4px 0; color:#a78bfa;'>Also known as: <b>" + ", ".join(f"<code>{a}</code>" for a in aliases) + "</b></p>" if aliases else ""}
  <p style="margin:6px 0 0 0; font-size:1.1rem;">
    Found in <b>{n_files} script{"s" if n_files != 1 else ""}</b>
  </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📋 Summary Table", "📄 Script Details", "🕸️ Lineage Graph"])

# ── Tab 1: Summary table ──────────────────────────────────────────────────────
with tabs[0]:
    rows = []
    for fp, g in file_groups.items():
        ops_str = " + ".join(sorted(g["ops"]))
        aliases_str = ", ".join(sorted(g["aliases_used"])) if g["aliases_used"] else "—"
        match_type = "alias" if g["matched_via"] == "alias" else "direct"
        rows.append({
            "Script": g["file_name"],
            "Operations": ops_str,
            "Matched Via": match_type,
            "Aliases Used": aliases_str,
            "Relevance Score": round(g["best_score"], 3),
            "Full Path": fp,
        })

    df = pd.DataFrame(rows)

    st.markdown(f"**`{canonical}` is referenced in the following {n_files} script(s):**")
    st.dataframe(
        df[["Script", "Operations", "Matched Via", "Aliases Used", "Relevance Score"]],
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name=f"lineage_{canonical}.csv",
        mime="text/csv",
    )

# ── Tab 2: Per-file detail cards ─────────────────────────────────────────────
with tabs[1]:
    for fp, g in file_groups.items():
        file_name = g["file_name"]
        ops_html = " ".join(_op_badge(op) for op in sorted(g["ops"]))
        alias_html = (
            f"<br><span style='color:#a78bfa; font-size:0.85rem;'>via alias: "
            f"<b>{', '.join(sorted(g['aliases_used']))}</b></span>"
            if g["aliases_used"] else ""
        )
        score_str = f"{g['best_score']:.3f}"
        n_chunks = len(g["chunks"])

        st.markdown(f"""
<div class="result-card">
  <b style="font-size:1.05rem;">{file_name}</b>&nbsp;&nbsp;
  {ops_html}
  <span style="color:#888; font-size:0.8rem; float:right;">score {score_str}</span>
  {alias_html}
  <br><code style="color:#666; font-size:0.75rem;">{fp}</code>
</div>
""", unsafe_allow_html=True)

        for chunk in g["chunks"]:
            meta = chunk.get("metadata", {})
            chunk_op = chunk.get("operation", "UNKNOWN")
            lines = f"lines {meta.get('line_start','?')}–{meta.get('line_end','?')}"
            explanation = chunk.get("explanation", "")
            emb = chunk.get("embedding_text", "")
            code = emb.split("[ENTITIES]")[0].replace("[CODE]", "").strip()
            ft = meta.get("file_type", "text")
            lang = {"python": "python", "sql": "sql", "shell": "bash"}.get(ft, "text")

            with st.expander(f"{_op_badge(chunk_op)} {lines}  —  {explanation[:90]}", expanded=False):
                st.code(code[:2000], language=lang)

        st.markdown("---")

# ── Tab 3: Lineage Graph ──────────────────────────────────────────────────────
with tabs[2]:
    graph = LineageGraph()
    graph.build_from_results(results, registry)
    G = graph.get_graph()

    if G.number_of_nodes() == 0:
        st.warning("No graph data to display.")
    else:
        summary = graph.get_lineage_summary(canonical)
        if summary:
            cols = st.columns(len(summary))
            emoji_map = {"READ": "📖", "WRITE": "✍️", "CREATE": "🏗️", "OTHER": "🔗"}
            for col, (op_type, files) in zip(cols, summary.items()):
                with col:
                    st.markdown(f"**{emoji_map.get(op_type, '🔗')} {op_type}**")
                    for f in files:
                        st.markdown(f"- `{f}`")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            html_path = tmp.name
        graph.to_pyvis_html(html_path, title=f"Lineage: {canonical}")
        if Path(html_path).exists():
            html_content = Path(html_path).read_text()
            st.components.v1.html(html_content, height=620, scrolling=True)
