"""
Data Lineage Discovery Dashboard
Enter a root ETL folder + optional exclusions, then search any table name.
"""
import json
import logging
import os
import re
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
from src.ingestion.document_builder import parse_meta_list
from src.extractors.alias_registry import AliasRegistry
from src.search.hybrid_search import HybridSearchEngine
from src.search.reranker import rerank
from src.graph.lineage_graph import LineageGraph

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
  .READ    { background: #1e40af; color: #93c5fd; }
  .WRITE   { background: #7f1d1d; color: #fca5a5; }
  .CREATE  { background: #713f12; color: #fde68a; }
  .UNKNOWN { background: #374151; color: #d1d5db; }
  .summary-box {
      background: #12122a; border-radius: 10px; padding: 18px 22px;
      margin-bottom: 18px; border: 1px solid #7c3aed33;
  }
  .config-pill {
      display: inline-block; background: #1e1e2e; border: 1px solid #7c3aed66;
      border-radius: 6px; padding: 2px 10px; font-size: 0.8rem;
      color: #a78bfa; margin: 2px 4px 2px 0;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "store": None,
        "registry": None,
        "engine": None,
        "engine_chunk_count": -1,   # fingerprint for BM25 cache invalidation
        "ingested": False,
        "results": [],
        "rerank_skipped": False,
        "query": "",
        "etl_path": settings.etl_repo_path,
        "excluded_dirs": "",
        "indexed_path": "",
        "indexed_excluded": [],
        "skip_rerank": False,
    }
    for k, v in defaults.items():
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
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.warning("Could not load existing vector store: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error loading vector store: %s", exc, exc_info=True)
        return False


def _parse_excluded(raw: str) -> list[str]:
    """Split comma/newline-separated folder names into a clean list."""
    parts = re.split(r"[,\n]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _sanitize_query(q: str) -> str:
    """Allow only alphanumeric, underscore, hyphen, dot, space — cap at 200 chars."""
    return re.sub(r"[^\w\s\-\.]", "", q)[:200].strip()


def _fallback_format(hits: list) -> list:
    """
    Format hybrid search hits to match reranker output shape when the API
    is unavailable. Infers operation from metadata, sets score = hybrid_score.
    """
    results = []
    for h in hits:
        meta = h.get("metadata", {})
        ops_list = parse_meta_list(meta.get("operations", "[]"))
        operation = ops_list[0].strip().upper() if ops_list else "UNKNOWN"
        result = dict(h)
        result["operation"]    = operation
        result["explanation"]  = "LLM reranking skipped — showing hybrid search score."
        result["matched_via"]  = "alias" if parse_meta_list(meta.get("aliases_used", "[]")) else "direct"
        result["llm_score"]    = 0.0
        result["final_score"]  = round(h.get("hybrid_score", 0.0), 4)
        results.append(result)
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def _get_or_build_engine(store: VectorStore, registry: AliasRegistry) -> HybridSearchEngine:
    """Return cached BM25 engine, rebuilding only if the collection has changed."""
    current_count = store.count()
    if (
        st.session_state.engine is not None
        and st.session_state.engine_chunk_count == current_count
    ):
        return st.session_state.engine

    engine = HybridSearchEngine(
        store, registry,
        semantic_weight=settings.hybrid_semantic_weight,
        bm25_weight=settings.hybrid_bm25_weight,
        metadata_weight=settings.hybrid_metadata_weight,
    )
    raw = store._collection.get(include=["documents", "metadatas"])
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
    st.session_state.engine_chunk_count = current_count
    return engine


def _do_search(query: str) -> tuple[list, bool]:
    """
    Returns (results, rerank_skipped).
    rerank_skipped=True means the API was unavailable and hybrid results are returned directly.
    """
    store: VectorStore = st.session_state.store
    registry: AliasRegistry = st.session_state.registry
    client = get_anthropic_client()

    engine = _get_or_build_engine(store, registry)
    hits = engine.search(query, top_k=15)

    skip_rerank = st.session_state.get("skip_rerank", False)
    if skip_rerank:
        return _fallback_format(hits), True

    try:
        ranked = rerank(client, query, hits, model=settings.llm_model, min_score=0.25)
        return ranked, False
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower() or "billing" in str(e).lower():
            return _fallback_format(hits), True
        raise
    except (anthropic.APIStatusError, anthropic.APIConnectionError,
            anthropic.RateLimitError, anthropic.APITimeoutError):
        return _fallback_format(hits), True


def _group_by_file(results: list) -> dict:
    groups: dict = defaultdict(lambda: {
        "ops": set(), "best_score": 0.0, "aliases_used": set(),
        "matched_via": "direct", "chunks": [], "file_type": "", "file_name": "",
    })
    for r in results:
        meta = r.get("metadata", {})
        fp = meta.get("file_path", "unknown")
        g = groups[fp]
        g["file_name"] = Path(fp).name
        g["file_type"] = meta.get("file_type", "")
        g["chunks"].append(r)
        g["ops"].add(r.get("operation", "UNKNOWN"))
        score = r.get("final_score", 0.0)
        if score > g["best_score"]:
            g["best_score"] = score
        if r.get("matched_via") == "alias":
            g["matched_via"] = "alias"
        for a in parse_meta_list(meta.get("aliases_used", "[]")):
            if a.strip():
                g["aliases_used"].add(a.strip())
    for g in groups.values():
        g["chunks"].sort(key=lambda r: r.get("metadata", {}).get("line_start", 0))
    return dict(sorted(groups.items(), key=lambda x: x[1]["best_score"], reverse=True))


def _op_badge(op: str) -> str:
    cls = op if op in ("READ", "WRITE", "CREATE") else "UNKNOWN"
    return f'<span class="op-badge {cls}">{op}</span>'


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # ── Input 1: ETL root folder ────────────────────────────────────────────
    etl_path = st.text_input(
        "📁 ETL Root Folder",
        value=st.session_state.etl_path,
        placeholder="/path/to/your/etl/repo",
        help="Absolute or relative path to the root of your ETL repository.",
        key="etl_path_input",
    )
    st.session_state.etl_path = etl_path

    # ── Input 2: Excluded folders ───────────────────────────────────────────
    excluded_raw = st.text_area(
        "🚫 Excluded Folders (optional)",
        value=st.session_state.excluded_dirs,
        placeholder="archive\ndeprecated\ntests, legacy",
        height=90,
        help="Folder names to skip during crawl. One per line or comma-separated. Matches any directory at any depth.",
        key="excluded_dirs_input",
    )
    st.session_state.excluded_dirs = excluded_raw
    excluded_list = _parse_excluded(excluded_raw)
    if excluded_list:
        st.caption(f"Will skip: {', '.join(f'`{d}`' for d in excluded_list)}")

    skip_llm = st.toggle(
        "⚡ Skip LLM entity extraction",
        value=st.session_state.get("skip_llm_ingest", False),
        help="Use regex-based extraction instead of Claude. No API credits needed. "
             "Finds exact table names but cannot resolve aliases (cust_tab → customer).",
        key="skip_llm_toggle",
    )
    st.session_state.skip_llm_ingest = skip_llm

    if st.button("🔄 Index ETL Files", use_container_width=True, type="primary"):
        if not etl_path.strip():
            st.error("Please enter an ETL root folder path.")
        elif not Path(etl_path.strip()).exists():
            st.error(f"Path not found: `{etl_path.strip()}`")
        else:
            label = "Indexing (offline mode)…" if skip_llm else "Indexing ETL files — this may take a minute…"
            with st.spinner(label):
                client = get_anthropic_client()
                store, registry = run_ingestion(
                    repo_path=etl_path.strip(),
                    chroma_persist_dir=settings.chroma_persist_dir,
                    alias_registry_path=settings.alias_registry_path,
                    anthropic_client=client,
                    llm_model=settings.llm_model,
                    excluded_dirs=excluded_list or None,
                    skip_llm=skip_llm,
                    verbose=False,
                    max_extraction_workers=settings.max_extraction_workers,
                )
                st.session_state.store = store
                st.session_state.registry = registry
                st.session_state.engine = None
                st.session_state.engine_chunk_count = -1  # force BM25 rebuild
                st.session_state.ingested = True
                st.session_state.indexed_path = etl_path.strip()
                st.session_state.indexed_excluded = excluded_list
                st.session_state.indexed_skip_llm = skip_llm
            st.success(f"Indexed {store.count()} chunks!")

    st.divider()

    # ── Input 3: Table name search ──────────────────────────────────────────
    st.markdown("### 🔎 Search Table")
    table_query = st.text_input(
        "Table Name",
        placeholder="e.g.  customer   orders   product",
        help="Enter any table name — including aliases. The system resolves them automatically.",
        key="query_input",
    )
    st.session_state.skip_rerank = st.toggle(
        "⚡ Skip LLM reranking",
        value=st.session_state.skip_rerank,
        help="Returns hybrid search results directly without calling the Claude API. "
             "Use this when credits are low or to save costs.",
    )

    search_clicked = st.button(
        "Search",
        use_container_width=True,
        disabled=not st.session_state.ingested,
        help="Index ETL files first, then search for a table name.",
    )

    if search_clicked and table_query.strip():
        if not st.session_state.ingested:
            st.warning("Please index ETL files first.")
        else:
            sanitized = _sanitize_query(table_query.strip())
            if sanitized != table_query.strip():
                st.info(f"Query sanitized to: `{sanitized}`")
            with st.spinner(f"Searching for **{sanitized}**…"):
                results, rerank_skipped = _do_search(sanitized)
                st.session_state.results = results
                st.session_state.rerank_skipped = rerank_skipped
                st.session_state.query = sanitized

    st.divider()

    if st.session_state.ingested and st.session_state.registry:
        registry_ref: AliasRegistry = st.session_state.registry
        st.subheader("📋 Known Tables & Aliases")
        canonicals = registry_ref.all_canonicals()
        if canonicals:
            for c in sorted(canonicals)[:30]:
                aliases = [a for a in registry_ref.get_all_aliases(c) if a != c]
                tooltip = f"→ {', '.join(aliases)}" if aliases else ""
                st.markdown(f"**`{c}`** {tooltip}")
        else:
            st.info("No tables indexed yet.")


# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🔍 Data Lineage Discovery Dashboard")
st.markdown("*Use the sidebar to set your ETL folder and search for any table name.*")

if not st.session_state.ingested:
    _load_existing()

# ─── Active configuration banner ──────────────────────────────────────────────
if st.session_state.ingested:
    idx_path = st.session_state.get("indexed_path") or st.session_state.etl_path
    idx_excl = st.session_state.get("indexed_excluded") or []
    idx_offline = st.session_state.get("indexed_skip_llm", False)
    excl_html = (
        "  ·  Excluded: " + " ".join(f'<span class="config-pill">{d}</span>' for d in idx_excl)
        if idx_excl else ""
    )
    mode_html = '  ·  <span class="config-pill">⚡ offline mode</span>' if idx_offline else ""
    st.markdown(
        f'<p style="font-size:0.88rem; color:#888;">Indexed: '
        f'<code style="color:#a78bfa">{idx_path}</code>'
        f'{excl_html}{mode_html} &nbsp;·&nbsp; {st.session_state.store.count()} chunks</p>',
        unsafe_allow_html=True,
    )
else:
    st.info("👈 **Step 1:** Set the ETL Root Folder in the sidebar and click **Index ETL Files**.")
    st.stop()

# ─── Results ──────────────────────────────────────────────────────────────────
if not st.session_state.results:
    if st.session_state.query:
        st.warning(f"No matching scripts found for **{st.session_state.query}**.")
    else:
        st.info("👈 **Step 2:** Enter a table name in the sidebar and click **Search**.")
    st.stop()

results = st.session_state.results
q = st.session_state.query
registry: AliasRegistry = st.session_state.registry
canonical = registry.resolve(q)
aliases = [a for a in registry.get_all_aliases(canonical) if a != canonical]
file_groups = _group_by_file(results)
n_files = len(file_groups)

# ── Summary banner ────────────────────────────────────────────────────────────
alias_pills = (
    "  ·  Also known as: " + " ".join(f'<span class="config-pill">{a}</span>' for a in aliases)
    if aliases else ""
)
st.markdown(f"""
<div class="summary-box">
  <h3 style="margin:0 0 4px 0;">Table: <code>{canonical}</code>{alias_pills}</h3>
  <p style="margin:6px 0 0 0; font-size:1.1rem;">
    Referenced in <b>{n_files} script{"s" if n_files != 1 else ""}</b>
  </p>
</div>
""", unsafe_allow_html=True)

if st.session_state.rerank_skipped:
    st.warning(
        "⚡ **LLM reranking was skipped** — results are ranked by hybrid search score "
        "(semantic + BM25) without Claude's relevance scoring. "
        "To enable full reranking, add API credits at [console.anthropic.com](https://console.anthropic.com) "
        "or turn off the **Skip LLM reranking** toggle if credits are available.",
        icon="⚠️",
    )

tabs = st.tabs(["📋 Summary Table", "📄 Script Details", "🕸️ Lineage Graph"])

# ── Tab 1: Summary table ──────────────────────────────────────────────────────
with tabs[0]:
    rows = []
    for fp, g in file_groups.items():
        rows.append({
            "Script": g["file_name"],
            "Operations": " + ".join(sorted(g["ops"])),
            "Matched Via": "alias" if g["matched_via"] == "alias" else "direct",
            "Aliases Used": ", ".join(sorted(g["aliases_used"])) or "—",
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
    st.download_button(
        label="⬇️ Download as CSV",
        data=df.to_csv(index=False),
        file_name=f"lineage_{canonical}.csv",
        mime="text/csv",
    )

# ── Tab 2: Per-file detail cards ─────────────────────────────────────────────
with tabs[1]:
    for fp, g in file_groups.items():
        ops_html = " ".join(_op_badge(op) for op in sorted(g["ops"]))
        alias_html = (
            f"<br><span style='color:#a78bfa; font-size:0.85rem;'>via alias: "
            f"<b>{', '.join(sorted(g['aliases_used']))}</b></span>"
            if g["aliases_used"] else ""
        )
        st.markdown(f"""
<div class="result-card">
  <b style="font-size:1.05rem;">{g['file_name']}</b>&nbsp;&nbsp;
  {ops_html}
  <span style="color:#888; font-size:0.8rem; float:right;">score {g['best_score']:.3f}</span>
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
            emoji_map = {"READ": "📖", "WRITE": "✍️", "CREATE": "🏗️", "OTHER": "🔗"}
            cols = st.columns(len(summary))
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
