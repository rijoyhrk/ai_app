"""
Shared fixtures and synthetic test data for all test modules.
All external dependencies (Claude API, GCP, ChromaDB on disk) are mocked or
replaced with in-memory equivalents so tests run offline with no credentials.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List


# ─── Synthetic ETL source code ────────────────────────────────────────────────

SYNTHETIC_SQL = """\
-- Load customers from staging into main table
INSERT INTO customer (cust_id, cust_name, region, signup_date)
SELECT cust_id, cust_name, region, signup_date
FROM customer_staging
WHERE updated_at > (SELECT MAX(updated_at) FROM customer);

-- Report: orders per customer using legacy alias
SELECT c.cust_name, COUNT(o.order_id) AS order_count
FROM cust_tab c
JOIN orders o ON c.cust_id = o.cust_id
GROUP BY c.cust_name
ORDER BY order_count DESC;

-- Create summary table
CREATE TABLE customer_ltv_report AS
SELECT cust_id, SUM(total_amount) AS ltv
FROM orders
GROUP BY cust_id;
"""

SYNTHETIC_PYTHON = """\
import pandas as pd
import sqlalchemy as sa

engine = sa.create_engine("postgresql://user:pass@localhost/dwh")

def extract_customers():
    \"\"\"Pull active customers using legacy alias cust_data.\"\"\"
    query = \"SELECT cust_id, cust_name, region FROM cust_data WHERE active = true\"
    return pd.read_sql(query, engine)

def load_orders():
    \"\"\"Load orders joined to customer dimension.\"\"\"
    query = \"\"\"
        SELECT o.order_id, o.total_amount, cd.cust_name
        FROM orders o
        INNER JOIN cust_data cd ON o.cust_id = cd.cust_id
    \"\"\"
    return pd.read_sql(query, engine)

def write_report(df):
    df.to_sql("customer_report", engine, if_exists="replace", index=False)
"""

SYNTHETIC_SHELL = """\
#!/usr/bin/env bash
# Load customer data from S3 into Redshift

S3_PATH="s3://datalake/raw/customers/"
CONN="host=redshift.company.com dbname=dwh user=etl"

# Step 1: Copy raw data into staging
psql "$CONN" -c "
COPY cust_staging (cust_id, cust_name, region)
FROM '$S3_PATH'
IAM_ROLE 'arn:aws:iam::123:role/RedshiftS3Role'
FORMAT AS CSV IGNOREHEADER 1;
"

# Step 2: Validate row counts
COUNT=$(psql "$CONN" -t -c "SELECT COUNT(*) FROM customer;")
echo "customer rows: $COUNT"
"""

SYNTHETIC_PRODUCT_SQL = """\
INSERT INTO prod_tbl (product_id, product_name, price)
SELECT product_id, product_name, price
FROM product_staging
WHERE updated_at > (SELECT MAX(updated_at) FROM prod_tbl);
"""


# ─── Mock Claude API response builders ────────────────────────────────────────

def make_claude_entity_response(refs: list) -> MagicMock:
    """Build a mock anthropic.messages.create response for entity extraction."""
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(refs)
    msg.content = [block]
    return msg


def make_claude_rerank_response(results: list) -> MagicMock:
    """Build a mock anthropic.messages.create response for reranking."""
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(results)
    msg.content = [block]
    return msg


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_etl_repo(tmp_path):
    """Create a temporary ETL repository with synthetic source files."""
    (tmp_path / "load_customers.sql").write_text(SYNTHETIC_SQL)
    (tmp_path / "pipeline.py").write_text(SYNTHETIC_PYTHON)
    (tmp_path / "load_customers.sh").write_text(SYNTHETIC_SHELL)
    (tmp_path / "product_sync.sql").write_text(SYNTHETIC_PRODUCT_SQL)

    # An excluded subfolder with an extra file
    excluded = tmp_path / "archive"
    excluded.mkdir()
    (excluded / "old_pipeline.py").write_text("SELECT * FROM deprecated_table;")

    return tmp_path


@pytest.fixture
def tmp_registry_path(tmp_path):
    return str(tmp_path / "alias_registry.json")


@pytest.fixture
def mock_anthropic_client():
    """Return a MagicMock standing in for anthropic.Anthropic."""
    return MagicMock()


@pytest.fixture
def populated_registry():
    """Return an AliasRegistry already loaded with customer/orders/product aliases."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.extractors.alias_registry import AliasRegistry
    from src.extractors.entity_extractor import ChunkEntities, EntityRef

    registry = AliasRegistry()
    registry.ingest([
        ChunkEntities(
            chunk_id="f1::1-10",
            file_path="pipeline.py",
            refs=[
                EntityRef(raw="cust_data",  canonical="customer", ref_type="READ",  confidence="high"),
                EntityRef(raw="cust_tab",   canonical="customer", ref_type="READ",  confidence="high"),
                EntityRef(raw="customer",   canonical="customer", ref_type="WRITE", confidence="high"),
                EntityRef(raw="orders",     canonical="orders",   ref_type="READ",  confidence="high"),
            ]
        ),
        ChunkEntities(
            chunk_id="f2::1-5",
            file_path="product_sync.sql",
            refs=[
                EntityRef(raw="prod_tbl",  canonical="product", ref_type="WRITE", confidence="high"),
                EntityRef(raw="product_staging", canonical="product_staging", ref_type="READ", confidence="high"),
            ]
        ),
    ])
    return registry


@pytest.fixture
def sample_hits():
    """Synthetic hybrid search hits for reranker / graph tests."""
    return [
        {
            "chunk_id": "pipeline.py::1-10",
            "hybrid_score": 0.85,
            "matched_canonical": "customer",
            "embedding_text": "[CODE]\nSELECT * FROM cust_data\n\n[ENTITIES]\nTables referenced: customer",
            "metadata": {
                "file_path": "pipeline.py",
                "file_type": "python",
                "chunk_type": "function",
                "line_start": 1,
                "line_end": 10,
                "canonical_tables": "customer",
                "aliases_used": "cust_data",
                "operations": "READ",
            },
        },
        {
            "chunk_id": "load_customers.sql::1-5",
            "hybrid_score": 0.78,
            "matched_canonical": "customer",
            "embedding_text": "[CODE]\nINSERT INTO customer ...\n\n[ENTITIES]\nTables referenced: customer",
            "metadata": {
                "file_path": "load_customers.sql",
                "file_type": "sql",
                "chunk_type": "insert",
                "line_start": 1,
                "line_end": 5,
                "canonical_tables": "customer|customer_staging",
                "aliases_used": "",
                "operations": "WRITE",
            },
        },
    ]
