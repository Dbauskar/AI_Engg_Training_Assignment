# unreliable-assistant — project rules

Enterprise-grade, MCP-enabled hybrid RAG agent. Python 3.11, managed with uv.

## Non-negotiable rules
1. **No implementation code until `docs/design.md` exists and is marked `STATUS: LOCKED`.**
   Scaffolding, config, and docs are allowed before then.
2. **Every tool call goes through `ToolExecutionHarness`** (validation, auto-repair,
   RBAC tier check, idempotency). No direct tool invocation.
3. **`tenant_id` is required in every DB query and cache key.** Reject calls without it.
4. **Never log raw PII.** Redact before logging, tracing, or persisting.
5. **Every feature ships with a pytest test and an OTel span.**

## Layout
app/ (API, harness, cache) · mcp_servers/ (hybrid-rag, state-store, graphify-vault)
plugins/ (pluggy lifecycle hooks) · agents/ (triage, ReAct, subagents)
evals/ (RAGAS, red team, chaos) · load_tests/ (k6, JMeter) · docs/ · knowledge_vault/ · tests/

## Commands
- `uv sync` — install deps
- `uv run pytest` — run tests
- `docker compose up -d` — Postgres (pgvector), Redis, OTel collector
