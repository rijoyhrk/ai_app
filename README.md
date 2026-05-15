# Data Lineage Discovery Dashboard

An intelligent RAG-powered dashboard: enter **any table name** and instantly see every ETL script that references it — even when scripts use aliases like `cust_tab`, `CUST`, or `prod_tbl` instead of the canonical name.

## How It Works

```
[Input 1] ETL Root Folder  +  [Input 2] Excluded Folders
       │
       ▼
  Crawl & Chunk (.py / .sql / .sh) — skipping excluded dirs
       │
       ▼
  LLM Entity Extraction → Alias Registry (alias → canonical)
       │
[Input 3] Table name: "customer"
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

The sidebar and main panel expose three input fields:

### Input 1 — ETL Root Folder (sidebar)

Enter the absolute or relative path to the root of your ETL repository.  
The crawler walks it recursively and picks up all `.py`, `.sql`, and `.sh` files.

```
📁 ETL Root Folder
/path/to/your/etl/repo
```

### Input 2 — Excluded Folders (sidebar)

Optionally list folder names to skip during indexing — one per line or comma-separated.  
Matches any directory at any depth inside the ETL root (e.g. `archive` will skip `/repo/archive/` and `/repo/jobs/archive/`).

```
🚫 Excluded Folders
archive
deprecated
tests, legacy
```

Click **Index ETL Files** to crawl the folder (respecting exclusions), extract entities with Claude, and build the vector index. A status bar shows the active path, excluded folders, and chunk count.

### Input 3 — Table Name (main panel)

Type any table name in the search bar and press **Search**.  
The system expands the query to all known aliases and runs hybrid search + LLM reranking.

```
🔎 Search Table
customer
```

### Results (3 tabs)

| Tab | What it shows |
|---|---|
| **Summary Table** | One row per script — Script, Operations, Matched Via, Aliases Used, Score. Downloadable as CSV. |
| **Script Details** | Per-file cards with expandable code excerpts per matching chunk, annotated with operation badges. |
| **Lineage Graph** | Interactive directed graph: table → file (READ) or file → table (WRITE/CREATE). |

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

## Pointing at Your Own ETL Repository

You can switch repositories at any time without restarting the app:

1. Enter the new root path in **ETL Root Folder**.
2. List any folders to skip in **Excluded Folders**.
3. Click **Index ETL Files** — the index is rebuilt from scratch for the new path.

`ETL_REPO_PATH` in `.env` sets the default path shown on startup, but it can be overridden live in the sidebar.

The system works best on repos where table aliases are consistent within a script (e.g., `cust_tab` always means `customer`). The LLM entity extractor infers these mappings automatically.

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
