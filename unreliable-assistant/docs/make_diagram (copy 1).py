"""Generate docs/workflow_orchestration.png (300 DPI) from the locked design (docs/design.md v1.1).

Usage:  uv run python docs/make_diagram.py [output_dir]
Requires the Graphviz `dot` binary (apt install graphviz).
"""
import sys
from pathlib import Path

from graphviz import Digraph

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

# Colour coding: security / retrieval / control-plane / egress
SEC = {"fillcolor": "#FDE2E2", "color": "#C0392B"}
RAG = {"fillcolor": "#DCEBFA", "color": "#2471A3"}
CTL = {"fillcolor": "#FFF1CC", "color": "#B7950B"}
EGR = {"fillcolor": "#DDF3E4", "color": "#1E8449"}
NEU = {"fillcolor": "#EEEEEE", "color": "#555555"}

g = Digraph("workflow", format="png")
g.attr(
    rankdir="TB", dpi="300", splines="spline", nodesep="0.35", ranksep="0.55",
    fontname="Helvetica", labelloc="t", fontsize="22", pad="0.3",
    label="The Unreliable Assistant: Enterprise Workflow Orchestration  (design.md v1.1)",
)
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11", penwidth="1.4")
g.attr("edge", fontname="Helvetica", fontsize="9", color="#444444")


def cluster(name, label, colors, build):
    with g.subgraph(name=f"cluster_{name}") as c:
        c.attr(label=label, style="rounded,dashed", color=colors["color"], fontcolor=colors["color"],
               fontname="Helvetica-Bold", fontsize="14", penwidth="2")
        build(c)


# --- Client -----------------------------------------------------------------
g.node("client", "Client / Gateway\n(TLS, rate limit, Envoy/NGINX)", **NEU)
g.node("ingress", "Ingress\nFastAPI async worker\n(W3C traceparent started)", **NEU)

# --- 1. Ingress & Security Layer -------------------------------------------
def sec(c):
    c.node("pii", "PII Redactor\n(priority 10, fail-closed)\nspan: hook.pre", **SEC)
    c.node("inj", "Injection Classifier\n(priority 20, fail-closed)\ndirect + indirect", **SEC)
    c.node("cache", "Redis Semantic Cache Check\nkey = SHA256(tenant:user:query)\ncosine >= 0.92, tenant-scoped\nspan: cache.lookup", **SEC)
    c.edge("pii", "inj")
    c.edge("inj", "cache")

cluster("security", "1. Ingress & Security Layer (Pre-Execution Plugins)", SEC, sec)

# --- 2. Triage & Subagent Routing ------------------------------------------
def triage(c):
    c.node("triage", "Triage Agent\ntemperature = 0.0, <= 25 ms\n(intent, domain, risk_tier, complexity)\nspan: triage.classify", **CTL)
    c.node("direct", "Direct Response\n(DIRECT_ANSWER)", **CTL)
    c.node("oos", "Reject Out-of-Scope\n(REJECT_OOS, fixed refusal)", **CTL)
    c.node("retr", "Retrieval Subagent\nread-only", **RAG)
    c.node("exec", "Execution Subagent\nno RAG / no web", **CTL)
    c.node("audit", "Audit / Verification Subagent\nindependent context, NLI veto", **CTL)

cluster("triage", "2. Triage & Subagent Routing", CTL, triage)

# --- 3. MCP & 3-Stage RAG Pipeline -----------------------------------------
def mcp(c):
    c.node("rag_in", "MCP JSON-RPC: hybrid_retrieve\n(mcp-hybrid-rag, stdio, <= 4 ms overhead)", **RAG)
    c.node("bm25", "Stage 1a: BM25Okapi\nk1=1.25, b=0.75, top 50\n< 12 ms", **RAG)
    c.node("dense", "Stage 1b: Bi-Encoder (bge-large ONNX)\n+ FAISS HNSW, top 50\n< 25 ms", **RAG)
    c.node("rrf", "Stage 2: Reciprocal Rank Fusion\nk=60, dedupe by chunk_hash, pool 40\n< 3 ms", **RAG)
    c.node("rerank", "Stage 3: Cross-Encoder (bge-reranker-v2-m3, ONNX FP16)\ntop 4, score > 0.42\n< 45 ms", **RAG)
    c.node("abstain", "Deterministic Abstention\n(top score < 0.42, no LLM, < 90 ms)", **SEC)
    c.node("graph", "MCP: mcp-graphify-vault\nObsidian [[wikilinks]] + code graph\nsubgraph depth <= 2 (~320 tokens)", **RAG)
    c.node("state", "MCP: mcp-state-store\nPostgres (pgvector): session_facts,\naudit_log, approvals", **RAG)
    c.edge("rag_in", "bm25")
    c.edge("rag_in", "dense")
    c.edge("bm25", "rrf")
    c.edge("dense", "rrf")
    c.edge("rrf", "rerank")
    c.edge("rerank", "abstain", label="score < 0.42", style="dashed", color="#C0392B")

cluster("mcp", "3. MCP & 3-Stage Hybrid RAG Pipeline", RAG, mcp)

# --- 4. Control Plane & Egress ---------------------------------------------
def ctl(c):
    c.node("react", "Bounded ReAct Loop\nmax_steps = 5, 120 ms/step, 450 ms total\nGoalAnchorBlock at index 0, obs <= 180 tokens\nspan: agent.react.step", **CTL)
    c.node("llm", "LLM Generate (stream)\nvLLM, continuous batching\nspan: llm.generate", **CTL)
    c.node("breaker", "Circuit Breaker\n(tool, args-hash) failed x2\n-> HALTED_BREAKER", **CTL)
    c.node("hitl", "Tier 3 HITL Gate\nPENDING_APPROVAL, execution frozen\nhuman approver required", **CTL)
    c.node("verify", "Verification\nclaim-to-chunk NLI, veto", **CTL)
    c.node("post", "Post-Execution (async)\nRAGAS scoring + OTel export\naudit-log hash-chain append\nspan: hook.post.ragas", **EGR)
    c.node("resp", "Response to Client\n(streamed, cited)", **EGR)
    c.node("obs", "OTel Collector\n-> Prometheus + Tempo + Grafana", **EGR)

cluster("control", "4. Control Plane & Egress", CTL, ctl)

# --- Edges -----------------------------------------------------------------
g.edge("client", "ingress")
g.edge("ingress", "pii")
g.edge("cache", "resp", label="hit", style="dashed", color="#1E8449")
g.edge("cache", "triage", label="miss")

g.edge("triage", "direct", label="DIRECT_ANSWER")
g.edge("triage", "oos", label="REJECT_OOS")
g.edge("triage", "retr", label="HYBRID_RAG")
g.edge("triage", "exec", label="TRANSACTION_EXEC")
g.edge("triage", "audit", label="audit / verify")

g.edge("direct", "llm")
g.edge("oos", "resp", label="fixed refusal")
g.edge("retr", "rag_in")
g.edge("retr", "graph")
g.edge("rerank", "react", label="top-4 chunks")
g.edge("graph", "react", label="subgraph")
g.edge("abstain", "resp", label="abstain", color="#C0392B")

g.edge("exec", "state", label="harness: validate, repair,\nRBAC, idempotency")
g.edge("exec", "hitl", label="Tier 3")
g.edge("hitl", "state", label="PENDING_APPROVAL", style="dashed")
g.edge("hitl", "resp", label="awaiting approval ID", style="dashed")
g.edge("audit", "state", label="audit log only")

g.edge("react", "breaker", label="tool failure", dir="both")
g.edge("breaker", "resp", label="HALTED_BREAKER", style="dashed", color="#B7950B")
g.edge("react", "llm")
g.edge("llm", "verify")
g.edge("verify", "resp", label="grounded")
g.edge("verify", "react", label="veto: re-answer / abstain", style="dashed", color="#C0392B", constraint="false")
g.edge("resp", "post", label="async")
g.edge("post", "obs")

# --- Legend ----------------------------------------------------------------
with g.subgraph(name="cluster_legend") as lg:
    lg.attr(label="Legend", style="rounded", color="#999999", fontname="Helvetica-Bold", fontsize="12")
    lg.node("l1", "Security / guardrail", **SEC)
    lg.node("l2", "Retrieval / MCP", **RAG)
    lg.node("l3", "Control plane / agents", **CTL)
    lg.node("l4", "Egress / observability", **EGR)
    lg.edge("l1", "l2", style="invis")
    lg.edge("l2", "l3", style="invis")
    lg.edge("l3", "l4", style="invis")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out = g.render(filename="workflow_orchestration", directory=str(OUT_DIR), cleanup=True)
print("wrote", out)
