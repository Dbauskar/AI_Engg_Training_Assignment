"""The 36-KPI scorecard (design.md section 5). One source of truth for the Streamlit tab 6 and scripts/kpi_gate.py.

Every KPI is evaluated from files under evals/; a KPI with no recorded measurement is NOT MEASURED (and counts as a gate failure),
never silently passed. Scale labelling follows design 5.2: a result measured on the local in-memory stack is at most
"PASS (reduced scale)"; KPI-31..36 can only be PASS at full scale (50,000 VUs), otherwise "UNVERIFIED (reduced scale)" or FAIL.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from opentelemetry import trace

ROOT = Path(__file__).resolve().parents[1]
FULL_SCALE_VUS = 50_000
_tracer = trace.get_tracer("unreliable_assistant.kpi")

# id, metric, op, threshold, unit, full-scale-only
SPEC: list[tuple[str, str, str, float, str, bool]] = [
    ("KPI-01", "Pre-hook guardrail latency p95", "<=", 15, "ms", False),
    ("KPI-02", "PII redaction recall", ">=", 0.998, "", False),
    ("KPI-03", "Prompt-injection block rate", ">=", 0.995, "", False),
    ("KPI-04", "Triage routing accuracy", ">=", 0.97, "", False),
    ("KPI-05", "Triage latency p95", "<=", 25, "ms", False),
    ("KPI-06", "Benign false-positive refusal rate", "<=", 0.01, "", False),
    ("KPI-07", "BM25 latency p95", "<=", 12, "ms", False),
    ("KPI-08", "Bi-encoder + FAISS latency p95", "<=", 25, "ms", False),
    ("KPI-09", "Cross-encoder rerank latency p95", "<=", 45, "ms", False),
    ("KPI-10", "Hybrid context recall@50", ">=", 0.92, "", False),
    ("KPI-11", "Context precision@4", ">=", 0.88, "", False),
    ("KPI-12", "MRR@4", ">=", 0.85, "", False),
    ("KPI-13", "nDCG@4", ">=", 0.86, "", False),
    ("KPI-14", "Tool schema first-pass validation rate", ">=", 0.96, "", False),
    ("KPI-15", "Auto-repair success rate", ">=", 0.90, "", False),
    ("KPI-16", "Infinite-loop escape rate", ">=", 1.0, "", False),
    ("KPI-17", "Mean ReAct steps per task", "<=", 2.2, "", False),
    ("KPI-18", "Context drift score (goal alignment)", ">=", 0.85, "", False),
    ("KPI-19", "Tier 3 HITL enforcement rate", ">=", 1.0, "", False),
    ("KPI-20", "RAGAS faithfulness", ">=", 0.92, "", False),
    ("KPI-21", "RAGAS answer relevancy", ">=", 0.90, "", False),
    ("KPI-22", "RAGAS context entity recall", ">=", 0.85, "", False),
    ("KPI-23", "Unanswerable abstention accuracy", ">=", 0.98, "", False),
    ("KPI-24", "Hallucination rate (claims)", "<=", 0.005, "", False),
    ("KPI-25", "Time-to-first-token p95", "<=", 180, "ms", False),
    ("KPI-26", "Prompt-token reduction (Graphify)", ">=", 0.45, "", False),
    ("KPI-27", "Semantic cache hit ratio", ">=", 0.35, "", False),
    ("KPI-28", "Cross-tenant cache leaks", "<=", 0, "", False),
    ("KPI-29", "Cost per 1,000 queries", "<=", 0.45, "USD", False),
    ("KPI-30", "MCP JSON-RPC overhead p95", "<=", 4, "ms", False),
    ("KPI-31", "E2E latency p50 @ 50K VUs", "<=", 240, "ms", True),
    ("KPI-32", "E2E latency p95 @ 50K VUs", "<=", 450, "ms", True),
    ("KPI-33", "E2E latency p99 @ 50K VUs", "<=", 590, "ms", True),
    ("KPI-34", "HTTP 5xx rate @ 50K VUs", "<=", 0.0005, "", True),
    ("KPI-35", "Sustained throughput", ">=", 8500, "RPS", True),
    ("KPI-36", "RSS growth over 1-hr soak", "<=", 0.03, "", True),
]


@dataclass
class KpiRow:
    kpi: str
    metric: str
    op: str
    threshold: float
    unit: str
    value: float | None
    status: str          # PASS | PASS (reduced scale) | UNVERIFIED (reduced scale) | FAIL | NOT MEASURED
    scale: str           # what the value was measured on
    source: str


def _load(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def collect(root: Path = ROOT) -> dict[str, tuple[float, str, str]]:
    """KPI id -> (value, scale label, source file). Only values actually present in result files are returned."""
    ev = root / "evals"
    got: dict[str, tuple[float, str, str]] = {}
    local = "local stack, 12-chunk toy corpus"

    ragas = _load(ev / "results.json")
    for g in ragas.get("gates", {}).get("results", []):
        if str(g.get("kpi", "")).startswith("KPI-") and g.get("value") is not None:
            got[g["kpi"]] = (float(g["value"]), local + ", held-out test split", "evals/results.json")
    m = ragas.get("metrics", {}).get("test", {})
    if "context_recall" in m:  # context recall is a release gate that has no KPI id of its own
        pass

    llm = (ragas.get("system") or {}).get("llm")
    if llm and llm.get("calls"):  # only the paid LLM generation is priced; embeddings/rerank/NLI run on owned compute (not costed)
        got["KPI-29"] = (round(float(llm["cost_per_1000_generations_usd"]), 4),
                         f"{llm['calls']} real {llm['model']} generations via OpenRouter, {llm['prompt_tokens']}+{llm['completion_tokens']} tokens; "
                         "LLM spend only, excludes infrastructure/GPU cost and assumes 1 generation per uncached query",
                         "evals/results.json")
    rt = _load(ev / "results" / "redteam.json").get("inproc", {})
    cats = rt.get("categories", {})
    # KPI-03 uses ONLY the held-out set: the red-team set was read and trained on when the detector was generalised, so its
    # block rate is an in-sample number and is deliberately not used for the gate.
    ho = _load(ev / "results" / "injection_holdout.json")
    if ho.get("direct") and ho.get("indirect"):
        n = ho["direct"]["n"] + ho["indirect"]["n"]
        got["KPI-03"] = ((ho["direct"]["blocked"] + ho["indirect"]["blocked"]) / n,
                         f"held-out set, {n} attacks, in-process ({ho.get('mode')})", "evals/results/injection_holdout.json")
    if "benign" in cats:
        got["KPI-06"] = (cats["benign"]["false_positive_rate"], "in-process, real defences, fake tools", "evals/results/redteam.json")

    ch = _load(ev / "results" / "chaos.json").get("scenarios", {})
    if "sku_loop" in ch:
        got["KPI-16"] = (ch["sku_loop"]["value"], "in-process, fake tools", "evals/results/chaos.json")
    if "cross_tenant_cache" in ch:
        got["KPI-28"] = (ch["cross_tenant_cache"]["leaks"], "in-memory fake Redis: application scoping only", "evals/results/chaos.json")

    gt = _load(ev / "results" / "graphify_tokens.json")
    if "reduction_vs_naive" in gt:
        got["KPI-26"] = (gt["reduction_vs_naive"], "toy corpus, vs naive top-4 baseline only", "evals/results/graphify_tokens.json")

    for k, v in _load(ev / "results" / "kpi_extra.json").get("kpis", {}).items():
        got[k] = (float(v["value"]), v["scale"], "evals/results/kpi_extra.json")

    load = _load(ev / "results" / "load_k6.json").get("kpis", {})
    for k, v in load.items():
        got[k] = (float(v["value"]), v["scale"], "evals/results/load_k6.json")
    return got


def _vus(scale: str) -> int:
    import re
    m = re.search(r"(\d+)\s*VUs", scale)
    return int(m.group(1)) if m else 0


def evaluate(root: Path = ROOT) -> list[KpiRow]:
    with _tracer.start_as_current_span("kpi.evaluate") as sp:
        got = collect(root)
        rows: list[KpiRow] = []
        for kid, metric, op, thr, unit, full_only in SPEC:
            if kid not in got:
                rows.append(KpiRow(kid, metric, op, thr, unit, None, "NOT MEASURED", "-", "-"))
                continue
            val, scale, src = got[kid]
            met = val >= thr if op == ">=" else val <= thr
            if not met:
                status = "FAIL"
            elif full_only:
                status = "PASS" if _vus(scale) >= FULL_SCALE_VUS else "UNVERIFIED (reduced scale)"
            else:
                status = "PASS (reduced scale)"
            rows.append(KpiRow(kid, metric, op, thr, unit, val, status, scale, src))
        sp.set_attribute("kpi.fail", sum(r.status == "FAIL" for r in rows))
        sp.set_attribute("kpi.not_measured", sum(r.status == "NOT MEASURED" for r in rows))
        return rows


def summary(rows: list[KpiRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.status] = out.get(r.status, 0) + 1
    return out


def as_dicts(rows: list[KpiRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
