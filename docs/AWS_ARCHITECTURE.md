# Data Lineage Discovery Dashboard — AWS Architecture

> Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). This document describes how the same
> system would be re-implemented as a production-grade, cloud-native service on AWS —
> service mapping, end-to-end data flow, LLM orchestration, and AI governance/guardrails.

---

## 1. Purpose

The current implementation (ChromaDB + local Streamlit + direct Anthropic SDK calls) is a
single-process prototype. This document specifies the equivalent architecture built on
managed AWS services — suitable for multi-user access, elastic scaling, audit requirements,
and AI governance controls that a local prototype cannot provide.

---

## 2. Component Mapping

| Capability | Prototype (current) | AWS (production) |
|---|---|---|
| Vector store | ChromaDB (local persistent) | Amazon OpenSearch Serverless (k-NN index) |
| Keyword search | `rank-bm25` (in-memory) | OpenSearch native BM25 full-text scoring |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Amazon Bedrock — Titan Embeddings v2 |
| LLM (extraction + rerank) | Anthropic SDK → Claude | Amazon Bedrock → Claude (Haiku/Sonnet) |
| ETL script storage | Local filesystem | Amazon S3 (versioned bucket) |
| Alias registry persistence | Local JSON file | Amazon S3 object (or DynamoDB) |
| Secrets | GCP Secret Manager | AWS Secrets Manager |
| Config | `pydantic-settings` + `.env` | AWS Systems Manager Parameter Store / env vars |
| Ingestion orchestration | Single Python function | AWS Step Functions state machine |
| Compute | Local process / single container | AWS Lambda (event-driven, auto-scaling) |
| Dashboard | Streamlit (localhost) | Streamlit on AWS App Runner, or React SPA on CloudFront |
| Lineage graph rendering | PyVis → temp HTML file | PyVis HTML → S3 → served via CloudFront |
| Job state tracking | None (synchronous, blocking) | DynamoDB (ingestion job status) |
| Async messaging | None (in-process) | Amazon SQS (decouples pipeline stages) |
| Observability | Python `logging` | CloudWatch Logs, Metrics, Alarms |
| Audit trail | None | CloudTrail + Bedrock Model Invocation Logging |
| AI safety controls | None | Amazon Bedrock Guardrails |

---

## 3. End-to-End Architecture Diagram

```
                          ┌──────────────────┐
                          │   S3 Bucket      │
                          │  etl-scripts/    │◄── Developer / CI uploads ETL files
                          │  alias_registry/ │
                          │  lineage_graphs/ │
                          └────────┬─────────┘
                                   │ S3 Event Notification
                                   ▼
                          ┌──────────────────┐
                          │   SQS Queues     │   (chunk → extract → embed,
                          │  (3 stages)      │    each with a DLQ)
                          └────────┬─────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │                 INGESTION (Step Functions Map state)     │
        │                                                          │
        │  [Chunker Lambda] → [Entity Extractor Lambda] → [Embedder]│
        │   sqlglot / ast /        Bedrock — Claude            Bedrock │
        │   regex (unchanged)      (w/ Guardrails)              Titan │
        │                                                          │
        │  MaxConcurrency: 8   |  Catch → rule-based fallback Lambda │
        └──────────────────────────┬──────────────────────────────┘
                                   │ vectors + metadata
                                   ▼
                        ┌──────────────────────┐
                        │  OpenSearch          │
                        │  Serverless          │
                        │  (k-NN + BM25 +      │
                        │   metadata filter)   │
                        └──────────┬───────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────────┐
        │           SEARCH (Step Functions Express / single Lambda)│
        │                                                          │
        │  [Resolve aliases] → [Embed query] → [Hybrid query]      │
        │       S3 registry        Bedrock           OpenSearch    │
        │                Titan                                     │
        │                          → [Rerank] (Bedrock Claude,     │
        │                              w/ Guardrails grounding)    │
        └──────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   API Gateway    │
                          │  /search /ingest │
                          │  /status/{id}    │
                          └────────┬─────────┘
                                   │ HTTPS
                                   ▼
                          ┌──────────────────┐
                          │  App Runner      │
                          │  (Streamlit) or  │◄── Browser
                          │  CloudFront      │
                          │  (React SPA)     │
                          └──────────────────┘

   ───────────────────────────────────────────────────────────────
   Cross-cutting: Secrets Manager · IAM · CloudWatch · CloudTrail ·
                  DynamoDB (job state) · Bedrock Guardrails
   ───────────────────────────────────────────────────────────────
```

---

## 4. Ingestion Pipeline

1. **Trigger** — ETL scripts land in `s3://<bucket>/etl-scripts/`. An S3 event notification
   pushes a message to the `etl-ingest-queue` SQS queue.
2. **Chunker Lambda** — downloads the file from S3, runs the existing chunker
   (`sqlglot` for `.sql`, `ast` for `.py`, regex for `.sh` — unchanged from the prototype),
   and publishes one message per chunk to `chunk-extract-queue`.
3. **Entity Extractor Lambda** (concurrency capped at 8, mirroring the prototype's
   `ThreadPoolExecutor(max_workers=8)`) — calls **Bedrock → Claude** through a
   **Bedrock Guardrail** (PII/secrets redaction on the way in, grounding check on the way
   out), then publishes the enriched chunk to `embed-queue`.
4. **Embedder + Indexer Lambda** — calls **Bedrock → Titan Embeddings** on the enriched
   `embedding_text` (code + entities section, same bridging technique as the prototype),
   writes the vector + metadata into the OpenSearch k-NN index, and updates
   `alias_registry.json` in S3.
5. **Orchestration** — a **Step Functions `Map` state** owns the fan-out across chunks,
   with per-state retry/backoff on `ThrottlingException` and a `Catch` clause routing to a
   rule-based regex Lambda (port of `rule_based_extractor.py`) if Bedrock is unavailable.
6. **Job tracking** — DynamoDB holds `{job_id, status, chunks_total, chunks_done, errors}`
   so the dashboard can poll ingestion progress instead of blocking on a spinner.

For large initial ingestions (10,000+ chunks), the bulk extraction step can use
**Bedrock Batch Inference** (async, ~50% cheaper) instead of the real-time fan-out;
incremental updates (one new file) stay on the real-time path.

---

## 5. Search Pipeline

1. **API Gateway** receives `POST /search {query: "customer"}`.
2. **Resolve aliases** — read `alias_registry.json` from S3 (cached in Lambda `/tmp`,
   invalidated on ETag change), expand `customer` → `[cust_tab, cust_data, CUST, ...]`.
3. **Embed query** — Bedrock Titan Embeddings on the augmented query string.
4. **Hybrid query** — a single OpenSearch `hybrid` query combining:
   - k-NN vector search (weight 0.6)
   - BM25 full-text across all alias terms (weight 0.25)
   - metadata filter on `canonical_tables` (weight 0.15)
5. **Rerank** — top-15 hits sent to Bedrock Claude (same `0.4×hybrid + 0.6×llm` scoring as
   the prototype), passed through a Guardrail grounding check so the LLM's explanation
   field can't assert a relevance reason unsupported by the actual code snippet.
6. **Response** returned through API Gateway to the dashboard.

This path uses **Step Functions Express Workflows** (sub-second, sync, billed per request)
or — given the steps are tightly sequential with no need for independent retry
visibility — a single Lambda function chaining the calls directly.

---

## 6. LLM Orchestration

| Concern | Pattern | AWS Mechanism |
|---|---|---|
| Parallel chunk extraction | Fan-out, bounded concurrency | Step Functions `Map` state, `MaxConcurrency: 8` |
| Search → rerank sequence | Sequential, low-latency | Step Functions Express, or single Lambda |
| Model selection per task | Static routing by task complexity | Haiku for high-volume extraction, Sonnet for rerank judgment — same Bedrock API, different model ID |
| Prompt cost reduction | Prompt caching | Bedrock `cache_control: ephemeral` on the static system prompt |
| LLM unavailability | Catch + fallback state | Step Functions `Catch` → rule-based extractor Lambda |
| Sustained outage | Circuit breaker | CloudWatch Alarm flips an SSM flag; Lambdas check it before invoking Bedrock |
| Large bulk ingestion | Async batch | Bedrock Batch Inference job, polled by Step Functions |
| Execution visibility | Audit/debug | Step Functions execution history graph (replaces buried `logger.warning` calls) |

Model routing is a deliberate cost/quality decision: extraction runs at high volume with a
narrow, structured task (good fit for a cheaper, faster model), while reranking runs at low
volume but requires nuanced relevance judgment (justifies a more capable model). Both share
the identical `InvokeModel` request/response shape on Bedrock, so routing is a config change,
not a code change.

---

## 7. AI Governance & Guardrails

| Governance Concern | AWS Mechanism |
|---|---|
| PII / secrets leaking into LLM prompts | **Bedrock Guardrails** — sensitive info filter, applied pre-invoke on every chunk sent to Claude |
| Hallucinated lineage relationships | **Bedrock Guardrails** — contextual grounding check, applied post-invoke on extraction and rerank output |
| Prompt injection from malicious code comments | **Bedrock Guardrails** — denied topics policy |
| Over-privileged access between pipeline stages | **IAM** — least-privilege execution role per Lambda (Entity Extractor cannot write to S3; Search Handler cannot delete OpenSearch indices) |
| Bypassing guardrails via direct API calls | IAM resource policy condition requiring `bedrock:GuardrailIdentifier` on every `InvokeModel` call |
| Audit trail of what the LLM saw and produced | **Bedrock Model Invocation Logging** (full prompt/response to S3, KMS-encrypted) + **CloudTrail** (who/what/when) |
| Data encryption at rest / in transit | KMS + TLS across S3, OpenSearch, Bedrock; VPC endpoints to keep traffic off the public internet |
| Training-data leakage to the model provider | Bedrock's no-training-on-customer-data policy |
| Runaway cost from retry storms | **Service Quotas** (TPS cap) + **AWS Budgets** alerts at 80% threshold |
| Low-confidence alias mappings corrupting the registry | DynamoDB review queue + SNS notification — human approval required before a `confidence: "low"` extraction is written to `alias_registry.json` |

Governance here is layered, not a single control: guardrails filter what the model sees and
says, IAM constrains what each component can touch, invocation logging plus CloudTrail make
every call auditable after the fact, and a human-in-the-loop queue catches the cases the
model itself flags as uncertain.

---

## 8. Code Reuse vs Rewrite

| Component | Change Required |
|---|---|
| `chunkers/` (SQL / Python / Shell) | **None** — pure Python, no AWS dependency |
| `document_builder.py` | **None** |
| `lineage_graph.py` | **None** — HTML output target changes from temp file to S3 |
| `alias_registry.py` | **Minimal** — `save()`/`load()` point at S3 instead of local path |
| `entity_extractor.py`, `reranker.py` | **Low** — swap `anthropic.Anthropic()` client for `boto3` Bedrock client; same prompts |
| `rule_based_extractor.py` | **None** — becomes the Step Functions fallback Lambda as-is |
| `vector_store.py` | **High** — rewritten against `opensearch-py` (k-NN + BM25 + filter) |
| `hybrid_search.py` | **High** — rewritten as a single OpenSearch hybrid query |
| `pipeline.py` | **High** — becomes a Step Functions state machine definition |
| `app.py` | **Medium** — calls API Gateway endpoints instead of in-process functions |
| `config.py` | **Low** — AWS Secrets Manager / Parameter Store instead of GCP Secret Manager |

The chunking, document-building, and graph logic — the parts with no external service
dependency — move to AWS untouched. The rewrite effort concentrates entirely in the
search/storage layer (ChromaDB → OpenSearch) and the orchestration layer
(single Python function → Step Functions).

---

## 9. Migration Path (Incremental)

1. **Swap secrets only** — AWS Secrets Manager replaces GCP Secret Manager; no other change.
   Validates AWS credentials/IAM setup with minimal risk.
2. **Swap LLM client** — route `entity_extractor.py` and `reranker.py` through Bedrock
   `boto3` instead of the Anthropic SDK, keeping ChromaDB and Streamlit as-is. Validates
   Bedrock access, model availability, and Guardrail configuration in isolation.
3. **Swap vector store** — replace ChromaDB with OpenSearch Serverless. This is the highest
   -effort step; validate hybrid query parity against existing integration tests
   (`tests/test_integration.py`) before deprecating ChromaDB.
4. **Swap orchestration** — move `pipeline.py`'s sequential logic into a Step Functions
   state machine. Keep Lambda functions thin wrappers around existing, unchanged Python
   modules.
5. **Swap presentation** — containerize Streamlit for App Runner, or build the React SPA
   against the now-stable API Gateway endpoints.

Each step is independently testable and reversible — the system remains functional at every
stage of the migration, rather than requiring a single big-bang cutover.
