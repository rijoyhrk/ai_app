"""
Data Lineage Discovery Dashboard
RAG-powered unknown entity resolver with semantic alias matching.
"""
import os
import sys
import tempfile
import streamlit as st
import anthropic
from pathlib import Path
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
      background: #1e1e2e; border-radius: 8px; padding: 14px;
      margin-bottom: 10px; border-left: 4px solid #7c3aed;
  }
  .op-badge {
      display: inline-block; padding: 2px 8px; border-radius: 12px;
      font-size: 0.75rem; font-weight: bold; margin-right: 6px;
  }
  .READ  { background: #1e40af; color: #93c5fd; }
  .WRITE { background: #7f1d1d; color: #fca5a5; }
  .CREATE { background: #713f12; color: #fde68a; }
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
    """Try loading an already-ingested store + registry."""
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
        # build BM25 from stored collection (sample 200 docs)
        raw = store._collection.get(limit=200, include=["documents", "metadatas"])
        bm25_docs = []
        for i, doc_id in enumerate(raw.get("ids", [])):
            bm25_docs.append({
                "chunk_id": doc_id,
                "embedding_text": raw["documents"][i] if raw.get("documents") else "",
                "metadata": raw["metadatas"][i] if raw.get("metadatas") else {},
            })
        engine.build_bm25_index(bm25_docs)
        st.session_state.engine = engine

    engine: HybridSearchEngine = st.session_state.engine
    hits = engine.search(query, top_k=15)
    ranked = rerank(client, query, hits, model=settings.llm_model, min_score=0.25)
    return ranked


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    etl_path = st.text_input("ETL Repo Path", value=settings.etl_repo_path)
    if st.button("🔄 Re-ingest ETL Files", use_container_width=True):
        with st.spinner("Ingesting ETL files..."):
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
        st.subheader("📋 Alias Registry")
        canonicals = registry.all_canonicals()
        if canonicals:
            for c in canonicals[:20]:
                aliases = [a for a in registry.get_all_aliases(c) if a != c]
                tooltip = f"aliases: {', '.join(aliases)}" if aliases else "no aliases"
                st.markdown(f"**{c}** — _{tooltip}_")
        else:
            st.info("No tables indexed yet.")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🔍 Data Lineage Discovery Dashboard")
st.markdown("*RAG-powered entity resolver · semantic alias matching · hybrid search*")

# auto-load if available
if not st.session_state.ingested:
    _load_existing()

if not st.session_state.ingested:
    st.info("👈 Click **Re-ingest ETL Files** in the sidebar to index your ETL repository.")
    st.stop()

st.success(f"✅ {st.session_state.store.count()} chunks indexed")

col_search, col_btn = st.columns([5, 1])
with col_search:
    query = st.text_input(
        "Search for a table / entity",
        placeholder='e.g. "customer" or "cust_tab"',
        key="query_input",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Search", use_container_width=True, type="primary")

if search_clicked and query:
    with st.spinner(f"Searching for '{query}'..."):
        results = _do_search(query)
        st.session_state.results = results
        st.session_state.query = query

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    q = st.session_state.query
    registry: AliasRegistry = st.session_state.registry
    canonical = registry.resolve(q)
    aliases = [a for a in registry.get_all_aliases(canonical) if a != canonical]

    st.subheader(f"Results for '{q}'")
    if aliases:
        st.caption(f"Resolved to canonical: **{canonical}** · known aliases: {', '.join(aliases)}")

    tabs = st.tabs(["📄 File References", "🕸️ Lineage Graph"])

    # ── Tab 1: Results list ──────────────────────────────────────────────────
    with tabs[0]:
        if not results:
            st.warning("No matching scripts found.")
        else:
            st.markdown(f"**{len(results)} script(s)** reference `{canonical}`:")
            for r in results:
                meta = r.get("metadata", {})
                file_name = Path(meta.get("file_path", "unknown")).name
                file_path = meta.get("file_path", "")
                op = r.get("operation", "UNKNOWN")
                score = r.get("final_score", 0)
                lines = f"{meta.get('line_start','?')}-{meta.get('line_end','?')}"
                explanation = r.get("explanation", "")
                matched_via = r.get("matched_via", "direct")
                aliases_used = meta.get("aliases_used", "")

                op_class = op if op in ("READ", "WRITE", "CREATE") else "READ"
                with st.container():
                    st.markdown(f"""
<div class="result-card">
  <b>{file_name}</b>&nbsp;
  <span class="op-badge {op_class}">{op}</span>
  <span style="color:#888; font-size:0.8rem;">lines {lines} · score {score:.2f}</span>
  {"<br><i style='color:#a78bfa; font-size:0.85rem;'>via alias: " + aliases_used + "</i>" if matched_via == "alias" and aliases_used else ""}
  <br><span style='color:#ccc; font-size:0.9rem;'>{explanation}</span>
  <br><code style='font-size:0.75rem; color:#888;'>{file_path}</code>
</div>
""", unsafe_allow_html=True)

                    with st.expander("View code excerpt"):
                        emb = r.get("embedding_text", "")
                        code = emb.split("[ENTITIES]")[0].replace("[CODE]", "").strip()
                        ft = meta.get("file_type", "text")
                        lang = {"python": "python", "sql": "sql", "shell": "bash"}.get(ft, "text")
                        st.code(code[:2000], language=lang)

    # ── Tab 2: Lineage Graph ─────────────────────────────────────────────────
    with tabs[1]:
        graph = LineageGraph()
        graph.build_from_results(results, registry)
        G = graph.get_graph()

        if G.number_of_nodes() == 0:
            st.warning("No graph data to display.")
        else:
            summary = graph.get_lineage_summary(canonical)
            if summary:
                for op_type, files in summary.items():
                    emoji = {"READ": "📖", "WRITE": "✍️", "CREATE": "🏗️"}.get(op_type, "🔗")
                    st.markdown(f"**{emoji} {op_type}**")
                    for f in files:
                        st.markdown(f"  - `{f}`")

            # render pyvis graph
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                html_path = tmp.name
            graph.to_pyvis_html(html_path, title=f"Lineage: {canonical}")
            if Path(html_path).exists():
                html_content = Path(html_path).read_text()
                st.components.v1.html(html_content, height=620, scrolling=True)
