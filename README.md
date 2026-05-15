# Data Lineage Discovery Dashboard

An intelligent RAG-powered dashboard: enter **any table name** and instantly see every ETL script that references it — even when scripts use aliases like `cust_tab`, `CUST`, or `prod_tbl` instead of the canonical name.

## How It Works

```
User query: "customer"
       │
       ▼
  Alias Registry ──► expands to: customer, cust_tab, cust_data, CUST, cust_tbl
       │
       ▼
  Hybrid Search (ChromaDB semantic + BM25 keyword + metadata filter)
       │
       ▼
  LLM Reranker (Claude scores 0–1, identifies READ / WRITE / CREATE)
       │
       ▼
  Lineage Graph (NetworkX DiGraph → PyVis interactive HTML)
       │
       ▼
  Dashboard (3 tabs):
    ├── Summary Table  — one row per script, with operations + aliases used + CSV download
    ├── Script Details — expandable per-chunk code excerpts with operation badges
    └── Lineage Graph  — interactive directed graph (table ↔ scripts)
```

## Architecture

The system is built in six layers:

| Layer | What it does |
|---|---|
| **AST-aware chunking** | Splits `.sql` (sqlglot), `.py` (Python ast), `.sh` (regex) at logical code boundaries — functions, statements, command groups |
| **LLM entity extraction** | Claude reads each chunk and emits `{raw, canonical, type, confidence}` — e.g. `cust_tab → customer` |
| **Alias registry** | Persistent JSON mapping every alias → canonical table name; used to expand queries at search time |
| **Enriched embeddings** | Embedding text = raw code + extracted canonical entity metadata; bridges the alias→canonical semantic gap |
| **Hybrid search** | Semantic (ChromaDB cosine, 60%) + BM25 keyword across all alias variants (25%) + metadata filter (15%) |
| **LLM reranker** | Claude scores each hit 0–1, tags operation type and match reason; `final_score = 0.4×hybrid + 0.6×llm` |

## Project Structure

```
ai_app/
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt
├── .env.example                    # Copy to .env and add your API key
├── src/
│   ├── config.py                   # pydantic-settings config (reads .env)
│   ├── chunkers/
│   │   ├── base.py                 # CodeChunk dataclass
│   │   ├── sql_chunker.py          # sqlglot-based SQL chunker
│   │   ├── python_chunker.py       # ast-based Python chunker
│   │   ├── shell_chunker.py        # regex-based shell chunker
│   │   └── dispatcher.py           # crawl_and_chunk() entry point
│   ├── extractors/
│   │   ├── entity_extractor.py     # LLM entity extraction with prompt caching
│   │   └── alias_registry.py       # Alias ↔ canonical table registry
│   ├── ingestion/
│   │   ├── document_builder.py     # Builds enriched ChromaDB documents
│   │   ├── vector_store.py         # ChromaDB wrapper (cosine similarity)
│   │   └── pipeline.py             # run_ingestion() orchestrator
│   ├── search/
│   │   ├── hybrid_search.py        # 3-layer hybrid search engine
│   │   └── reranker.py             # LLM reranker (Claude)
│   └── graph/
│       └── lineage_graph.py        # NetworkX DiGraph + PyVis HTML renderer
└── data/
    └── sample_etl/                 # Example ETL scripts for demo
        ├── customer_data_pipeline.py
        ├── customer_transformation.sql
        ├── load_customers.sh
        ├── orders_pipeline.py
        └── product_sync.sql
```

## Quick Start

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/rijoyhrk/ai_app.git
cd ai_app
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

Key settings in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
ETL_REPO_PATH=data/sample_etl     # path to your ETL scripts
CHROMA_PERSIST_DIR=chroma_db
ALIAS_REGISTRY_PATH=alias_registry.json
LLM_MODEL=claude-opus-4-7
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

On first run, click **Re-ingest ETL Files** in the sidebar to crawl your scripts, extract entities, and build the vector index.

## Using the Dashboard

1. **Type any table name** in the search bar (e.g. `customer`, `orders`, `prod_tbl`) and press **Search**.
2. The **Summary Table** tab shows a deduplicated list of every script that references the table, the operation type (READ / WRITE / CREATE), and which alias was matched.
3. The **Script Details** tab shows each matching script as a card with expandable code excerpts per chunk.
4. The **Lineage Graph** tab renders an interactive directed graph — arrows show whether a script reads from or writes to the table.
5. Use the **⬇️ Download as CSV** button in the Summary Table to export the results.

## Example: Alias Resolution in Action

Querying `customer` finds all five sample scripts. The **Summary Table** output looks like:

| Script | Operations | Matched Via | Aliases Used | Relevance Score |
|---|---|---|---|---|
| `customer_data_pipeline.py` | READ + WRITE | alias | `cust_tab` | 0.91 |
| `customer_transformation.sql` | WRITE | alias | `CUST`, `customer_staging` | 0.88 |
| `load_customers.sh` | WRITE | alias | `customer_staging`, `cust_tab` | 0.85 |
| `orders_pipeline.py` | READ | alias | `cust_data` | 0.79 |
| `product_sync.sql` | READ | alias | `CUSTOMER` | 0.74 |

None of these scripts use the exact word `customer` everywhere — alias resolution bridges the gap automatically.

## Using Your Own ETL Repository

1. Set `ETL_REPO_PATH` in `.env` to the root of your ETL repo (absolute or relative path).
2. The crawler picks up all `.py`, `.sql`, and `.sh` files recursively.
3. Click **Re-ingest ETL Repo** to rebuild the index.

The system works best on repos where table aliases are consistent within a script (e.g., `cust_tab` always means `customer`). The LLM entity extractor is prompted to infer these mappings automatically.

## Tech Stack

| Component | Library |
|---|---|
| Vector store | ChromaDB (local persistent) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| LLM (extraction + reranking) | Anthropic Claude (claude-opus-4-7) |
| SQL parsing | sqlglot |
| Python AST | stdlib `ast` |
| BM25 keyword search | rank-bm25 |
| Lineage graph | NetworkX + PyVis |
| Dashboard | Streamlit |
| Config | pydantic-settings + python-dotenv |

## Security Notes

- The dashboard has **no authentication**. Do not expose it on a public IP with a real API key.
- `.env` and `chroma_db/` are gitignored — never commit your API key.
- The Streamlit server binds to `localhost` by default (see `.streamlit/config.toml`).
