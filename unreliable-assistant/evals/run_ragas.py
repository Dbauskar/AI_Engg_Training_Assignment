"""Run the real system over the 250-question golden set and check the release gates.

  uv run python -m evals.run_ragas [--threshold T | --calibrate safety|balanced] [--gates all|ragas] [--limit-per-stratum N] [--out evals/results.json]

Process
  A  retrieval records for every question (threshold-free top-4 + stage candidate lists), triage bypassed
  B  calibrate the abstention threshold on the DEV split only (or take --threshold)
  C  run every question end to end through triage -> retrieval -> generation -> verification at that threshold
  D  metrics on the TEST split (dev is reported too, but gates are judged on test, which calibration never saw)
Exit status 1 if any gate fails. results.json is written either way.

These are deterministic stand-ins for the LLM-judged RAGAS metrics (no LLM judge exists in this project); every metric below says
exactly how it is computed. The generator is the extractive answerer, so faithfulness is structurally easy to satisfy, and the
questions were written by the system's author: both limits are recorded in results.json["caveats"].
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

from agents.nli import OnnxNli  # noqa: E402
from agents.subagents import ExtractiveReasoner, Orchestrator, rewrite_query  # noqa: E402
from agents.triage import TriageAgent  # noqa: E402
from agents.verification import VerificationSubagent, _candidates, _numbers, _sentences, near_verbatim, split_claims  # noqa: E402
from app.bulkhead import Pools  # noqa: E402
from app.harness import Caller, Tier, ToolExecutionHarness  # noqa: E402
from evals.golden import PATH as GOLDEN_PATH, GoldenItem, load_golden, load_vault  # noqa: E402
from evals.metrics import (context_entity_recall, context_precision, context_recall, mean, mrr_at_k, ndcg_at_k,  # noqa: E402
                           recall_at_k)
from mcp_servers.rag_pipeline import DEFAULT_EMBED_DIR, HybridRetriever, OnnxEmbedder, build_index  # noqa: E402
from mcp_servers.rag_server import register_rag_tools  # noqa: E402

TENANT = "acme-corp"
EVAL_NLI_DIR = "models/nli-MiniLM2-L6-H768"  # deliberately NOT the model the runtime verifier uses
DESIGN_THRESHOLD = 0.42
GATES = [  # (name, KPI, metric key, op, threshold, group)
    ("faithfulness", "KPI-20", "faithfulness", ">=", 0.92, "ragas"),
    ("answer_relevancy", "KPI-21", "answer_relevancy", ">=", 0.90, "ragas"),
    ("context_precision", "KPI-11", "context_precision", ">=", 0.88, "ragas"),
    ("context_recall", "gate", "context_recall", ">=", 0.85, "ragas"),
    ("context_entity_recall", "KPI-22", "context_entity_recall", ">=", 0.85, "ragas"),
    ("recall_at_50", "KPI-10", "recall_at_50", ">=", 0.92, "kpi"),
    ("mrr_at_4", "KPI-12", "mrr_at_4", ">=", 0.85, "kpi"),
    ("ndcg_at_4", "KPI-13", "ndcg_at_4", ">=", 0.86, "kpi"),
    ("abstention_accuracy", "KPI-23", "abstention_accuracy", ">=", 0.98, "kpi"),
    ("hallucination_rate", "KPI-24", "hallucination_rate_claims", "<=", 0.005, "kpi"),
]
CAVEATS = [
    "The golden questions were written by the same author as the system, from the same 40 facts: a regression/calibration set, not an independent benchmark.",
    "The knowledge base has 12 chunks, so Recall@50 is trivially 1.0 (50 candidates cover the whole index); Recall@3 is reported as the informative variant.",
    "No LLM judge exists: faithfulness = independent-NLI claim support; answer relevancy = cross-encoder relevance of the answer to the question; context metrics use gold evidence from the vault.",
    "The answerer is deterministic and extractive (quotes the best sentences), so faithfulness and hallucination are structurally easy; they become informative only with a real LLM.",
    "Multi-hop items are two-fact questions (parallel), not sequential bridge reasoning; complex items need a rule applied, which the extractive answerer cannot do, so they test retrieval only.",
]


class NoMemory:
    async def load(self, *a: Any) -> list:
        return []

    async def save(self, *a: Any) -> bool:
        return True


DUMP_UNSUPPORTED: list[dict[str, Any]] = []  # set non-empty (a sentinel) by --dump-unsupported


def clean_answer(text: str) -> str:
    return re.sub(r"\[[a-z0-9]{2,16}-\d{4}-[0-9a-f]{8}\]", "", re.sub(r"^#+ .*\n+", "", text)).strip()


def build_stack(tmp: Path) -> dict[str, Any]:
    embedder = OnnxEmbedder(DEFAULT_EMBED_DIR, threads=2)
    retriever = HybridRetriever(tmp, embedder=embedder)
    tok = Tokenizer.from_file(f"{DEFAULT_EMBED_DIR}/tokenizer.json")
    tok.no_truncation()
    tok.no_padding()
    build_index(TENANT, ROOT / "knowledge_vault", tmp, embedder, tok)
    return {"embedder": embedder, "retriever": retriever, "triage": TriageAgent(embedder), "nli_runtime": OnnxNli(),
            "nli_eval": OnnxNli(ROOT / EVAL_NLI_DIR), "pools": Pools.from_env()}


def retrieval_record(retriever: HybridRetriever, q: str) -> dict[str, Any]:
    """Phase A: threshold-free top-4 plus the candidate lists of every stage."""
    rq = rewrite_query(q)
    out = retriever.retrieve(TENANT, rq, score_threshold=0.0, rerank_k=4)
    cand = retriever.candidate_docs(TENANT, rq, 50)
    return {"ranked_docs": [c.chunk_id.split("-")[0] for c in out.chunks], "scores": [round(c.score, 4) for c in out.chunks],
            "top_score": out.top_score, "union": cand["union"], "pool": cand["pool"]}


NEG_MARGIN = 0.02  # how far above the highest dev-negative reranker score the abstention threshold must sit


def calibrate(dev: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Pick the abstention threshold on DEV. Rule: answer iff top reranker score >= tau."""
    pos = np.array([r["top_score"] for r in dev if r["expect"] == "answer"])
    neg = np.array([r["top_score"] for r in dev if r["expect"] == "abstain"])
    curve = []
    for tau in np.round(np.arange(0.02, 0.99, 0.02), 2):
        tpr, tnr = float((pos >= tau).mean()), float((neg < tau).mean())
        curve.append({"tau": float(tau), "answerable_answered": round(tpr, 4), "negative_abstained": round(tnr, 4), "balanced": round((tpr + tnr) / 2, 4)})
    safe = [c for c in curve if c["negative_abstained"] >= 0.98]
    best_bal = max(c["balanced"] for c in curve)
    balanced = [c for c in curve if c["balanced"] == best_bal]
    chosen = (min(safe, key=lambda c: c["tau"]) if safe else balanced[-1]) if mode == "safety" else balanced[len(balanced) // 2]
    if mode == "safety" and safe and len(neg):
        # Margin rule (DEV only): the rule is "answer iff score >= tau", so a threshold sitting ON the highest dev-negative score lets an
        # equal-scoring unseen negative through. Require tau to clear the highest dev negative by MARGIN.
        want = float(np.round(float(neg.max()) + NEG_MARGIN, 2))
        cand = [c for c in safe if c["tau"] >= want]
        chosen = min(cand, key=lambda c: c["tau"]) if cand else max(safe, key=lambda c: c["tau"])
    at_design = min(curve, key=lambda c: abs(c["tau"] - DESIGN_THRESHOLD))
    return {"mode": mode, "chosen": chosen, "safety_threshold_exists": bool(safe), "design_threshold_row": at_design,
            "dev_answerable": int(len(pos)), "dev_negatives": int(len(neg)), "curve": curve}


_QUOTED = re.compile(r"[\"“]([^\"”]{20,})[\"”]")


def _norm(t: str) -> str:
    return " ".join(re.findall(r"\w+", t.lower()))


def verbatim_quote_supported(claim: str, ctx: str) -> bool:
    """Deterministic grounding: the claim is a short wrapper ('The policy states that ...') around a span copied word for word
    from the retrieved text. A verbatim span cannot be an invention; the NLI model is unreliable on quote-wrapped sentences."""
    m = _QUOTED.search(claim)
    if not m:
        return False
    wrapper = _norm(claim.replace(m.group(0), " "))
    return len(wrapper.split()) <= 6 and _norm(m.group(1)) in _norm(ctx)


def judge(nli: OnnxNli, claims: list[str], chunk_texts: list[str], question: str = "") -> list[bool]:
    """Independent claim check: a claim is supported iff some sentence/window of some retrieved chunk entails it (P >= 0.5)
    and every number in the claim occurs in the retrieved text."""
    ctx_numbers = _numbers(" ".join(chunk_texts)) | _numbers(question)  # figures the user typed may be restated
    out = []
    for claim in claims:
        if _numbers(claim) - ctx_numbers:
            out.append(False)
            continue
        if verbatim_quote_supported(claim, " ".join(chunk_texts)) or near_verbatim(claim, [x for t in chunk_texts for x in _sentences(t)]):
            out.append(True)
            continue
        premises, hyps = [], []
        for t in chunk_texts:
            for cand in _candidates(claim, t):
                premises.append(cand)
                hyps.append(claim)
        ent = nli.probs(premises, hyps)[0] if premises else np.zeros(1)
        out.append(bool(float(ent.max()) >= 0.5))
    return out


async def run_system(stack: dict[str, Any], items: list[GoldenItem], tau: float, progress: bool = True, llm: Any = None) -> dict[str, Any]:
    from agents.llm_reasoner import LLMReasoner, ResilientReasoner
    from agents.react import ReActConfig
    harness = ToolExecutionHarness()
    register_rag_tools(harness, stack["retriever"], stack["pools"])
    verifier = VerificationSubagent(stack["nli_runtime"], None, stack["pools"])
    orch = Orchestrator(harness, stack["triage"], verifier,
                        (lambda q, data: (ResilientReasoner(LLMReasoner(llm, data["chunks"], q), ExtractiveReasoner(data["chunks"], q), llm.usage)
                                          if llm is not None else ExtractiveReasoner(data["chunks"], q)) if data else _Hello()),
                        memory=NoMemory(), pools=stack["pools"], retrieval_threshold=tau,
                        config=ReActConfig(per_step_timeout_s=15.0, total_budget_s=25.0) if llm is not None else None)
    caller, results = Caller(TENANT, "eval", Tier.TIER_1_READ), {}
    t0 = time.perf_counter()
    for n, it in enumerate(items, 1):
        results[it.id] = await orch.answer(caller, it.question, f"{n:08d}-0000-4000-8000-000000000000")
        if progress and n % 25 == 0:
            print(f"  ... {n}/{len(items)} questions ({time.perf_counter() - t0:.0f}s)", flush=True)
    return results


class _Hello:
    async def next_step(self, prompt: list) -> Any:
        from agents.react import Final
        return Final("Hello!")


def score_items(stack: dict[str, Any], items: list[GoldenItem], ret: dict[str, dict], sysres: dict[str, Any], chunk_text: dict[str, str]) -> list[dict[str, Any]]:
    notes, _ = load_vault()
    recs = []
    for it in items:
        r, res = ret[it.id], sysres[it.id]
        rec: dict[str, Any] = {"id": it.id, "stratum": it.stratum, "kind": it.kind, "split": it.split, "expect": it.expect,
                               "outcome": res.outcome, "route": res.route, "reason": res.reason, "vetoes": res.verifier_vetoes, "halt": res.halt_reason, "top_score": r["top_score"], "answered": res.outcome == "ANSWERED"}
        if it.expect == "answer":
            gold = set(it.gold_docs)
            rec.update(mrr_at_4=mrr_at_k(r["ranked_docs"], gold, 4), ndcg_at_4=ndcg_at_k(r["ranked_docs"], gold, 4),
                       recall_at_3=recall_at_k(r["ranked_docs"], gold, 3), recall_at_50=recall_at_k(r["union"], gold, 50),
                       pool_recall=recall_at_k(r["pool"], gold, 40))
            sent = [c.split("-")[0] for c in res.retrieved_chunk_ids]  # the contexts the generator really received, in rank order
            ctx = "\n".join(chunk_text[c] for c in res.retrieved_chunk_ids)
            facts = [notes[g]["text"] for g in it.gold_notes]
            rec.update(context_precision=context_precision(sent, gold, 4) if sent else 0.0,
                       context_recall=context_recall(ctx, facts) if sent else 0.0,
                       context_entity_recall=context_entity_recall(" ".join(facts), ctx) if sent else 0.0)
            if res.outcome == "ANSWERED" and res.route == "HYBRID_RAG":
                claims = split_claims(res.answer)
                ok = judge(stack["nli_eval"], claims, [chunk_text[c] for c in res.retrieved_chunk_ids], it.question) if claims else []
                if DUMP_UNSUPPORTED and not all(ok):  # diagnostics only: synthetic golden data, never real user text
                    DUMP_UNSUPPORTED.append({"id": it.id, "stratum": it.stratum, "question": it.question, "answer": res.answer,
                                             "unsupported_claims": [c for c, g in zip(claims, ok) if not g], "chunks": list(res.retrieved_chunk_ids)})
                rec.update(claims=len(claims), unsupported=sum(not x for x in ok), faithfulness=(sum(ok) / len(ok)) if ok else 1.0,
                           answer_relevancy=float(stack["retriever"].reranker.score(it.question, [clean_answer(res.answer)])[0]))
                if DUMP_UNSUPPORTED and rec["answer_relevancy"] < 0.8:  # diagnostics only (synthetic golden data)
                    DUMP_UNSUPPORTED.append({"id": it.id, "low_relevancy": round(rec["answer_relevancy"], 3), "question": it.question, "answer": res.answer})
        else:
            rec["abstained_correctly"] = res.outcome in ("ABSTAINED", "REFUSED")
        recs.append(rec)
    return recs


def aggregate(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in recs if r["expect"] == "answer"]
    neg = [r for r in recs if r["expect"] == "abstain"]
    gen = [r for r in ans if "faithfulness" in r]
    claims = sum(r["claims"] for r in gen)
    m = lambda key, rows: mean([r[key] for r in rows if key in r])  # noqa: E731
    return {
        "n": len(recs), "n_answerable": len(ans), "n_negative": len(neg), "n_generated": len(gen),
        "faithfulness": m("faithfulness", gen), "answer_relevancy": m("answer_relevancy", gen),
        "hallucination_rate_claims": (sum(r["unsupported"] for r in gen) / claims) if claims else 0.0,
        "hallucination_rate_responses": mean([float(r["unsupported"] > 0) for r in gen]) if gen else 0.0,
        "context_precision": m("context_precision", ans), "context_recall": m("context_recall", ans),
        "context_entity_recall": m("context_entity_recall", ans),
        "mrr_at_4": m("mrr_at_4", ans), "ndcg_at_4": m("ndcg_at_4", ans), "recall_at_3": m("recall_at_3", ans),
        "recall_at_50": m("recall_at_50", ans), "pool_recall_at_40": m("pool_recall", ans),
        "abstention_accuracy": mean([float(r["abstained_correctly"]) for r in neg]) if neg else float("nan"),
        "abstention_accuracy_oos": mean([float(r["abstained_correctly"]) for r in neg if r["kind"] == "negative-oos"]) if neg else float("nan"),
        "abstention_accuracy_uncovered": mean([float(r["abstained_correctly"]) for r in neg if r["kind"] == "negative-uncovered"]) if neg else float("nan"),
        "answerable_answered_rate": mean([float(r["answered"]) for r in ans]) if ans else float("nan"),
        "answerable_abstained_rate": mean([float(r["outcome"] == "ABSTAINED") for r in ans]) if ans else float("nan"),
        "answerable_refused_rate": mean([float(r["outcome"] == "REFUSED") for r in ans]) if ans else float("nan"),
        "outcomes": dict(Counter(r["outcome"] for r in recs)),
    }


def check_gates(metrics: dict[str, Any], which: str) -> list[dict[str, Any]]:
    out = []
    for name, kpi, key, op, thr, group in GATES:
        if which == "ragas" and group != "ragas":
            continue
        v = metrics[key]
        ok = (v >= thr) if op == ">=" else (v <= thr)
        out.append({"gate": name, "kpi": kpi, "metric": key, "value": round(float(v), 4), "op": op, "threshold": thr, "status": "PASS" if ok else "FAIL"})
    return out


async def amain(a: argparse.Namespace) -> int:
    items = load_golden()
    if a.limit_per_stratum:
        seen: Counter = Counter()
        keep = []
        for it in items:
            if seen[it.stratum] < a.limit_per_stratum:
                keep.append(it)
                seen[it.stratum] += 1
        items = keep
    print(f"golden set: {len(items)} questions {dict(Counter(i.stratum for i in items))}", flush=True)
    tmp = Path(tempfile.mkdtemp())
    stack = build_stack(tmp)
    retriever = stack["retriever"]
    retrieval_record(retriever, "warm up")
    chunk_text = {c.chunk_id: c.text for c in retriever.index_for(TENANT).chunks}

    t = time.perf_counter()
    ret = {it.id: retrieval_record(retriever, it.question) for it in items}
    print(f"A. retrieval records for {len(items)} questions ({time.perf_counter() - t:.0f}s)", flush=True)
    rows = [{"expect": it.expect, **ret[it.id]} for it in items if it.split == "dev"]
    cal = calibrate(rows, a.calibrate)
    tau = a.threshold if a.threshold is not None else cal["chosen"]["tau"]
    print(f"B. abstention threshold: {'--threshold ' + str(tau) if a.threshold is not None else f'calibrated on dev ({a.calibrate}) = {tau}'} (design default {DESIGN_THRESHOLD}; "
          f"on dev the design value answers {cal['design_threshold_row']['answerable_answered']:.0%} of answerable and abstains on {cal['design_threshold_row']['negative_abstained']:.0%} of negatives)", flush=True)

    t = time.perf_counter()
    llm = None
    if a.llm:
        from agents.llm_reasoner import OpenRouterClient
        from app.config import load_env
        load_env()
        llm = OpenRouterClient()
    sysres = await run_system(stack, items, tau, llm=llm)
    llm_report = None
    if llm is not None:
        u = llm.usage
        llm_report = {"model": llm.model, "calls": u.calls, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                      "cost_usd": round(u.cost_usd, 5), "retries": u.retries, "fallbacks": u.fallbacks, "errors": u.errors,
                      "cost_per_1000_generations_usd": round(u.cost_usd / max(u.calls, 1) * 1000, 4),
                      "latency_ms_p50_p95": [round(float(np.percentile(u.latencies_ms, 50)), 0), round(float(np.percentile(u.latencies_ms, 95)), 0)] if u.latencies_ms else None}
        await llm.close()
    print(f"C. end-to-end run done ({time.perf_counter() - t:.0f}s)", flush=True)
    recs = score_items(stack, items, ret, sysres, chunk_text)
    splits = {s: aggregate([r for r in recs if r["split"] == s]) for s in ("dev", "test")}
    splits["all"] = aggregate(recs)
    by_stratum = {s: aggregate([r for r in recs if r["split"] == "test" and r["stratum"] == s]) for s in dict.fromkeys(i.stratum for i in items)}
    gates = check_gates(splits["test"], a.gates)
    failures = {
        "answerable_not_answered": [{"id": r["id"], "outcome": r["outcome"], "reason": r["reason"], "vetoes": r["vetoes"], "halt": r["halt"], "route": r["route"], "top_score": round(r["top_score"], 3)} for r in recs if r["split"] == "test" and r["expect"] == "answer" and not r["answered"]],
        "negatives_answered": [{"id": r["id"], "kind": r["kind"], "outcome": r["outcome"], "top_score": round(r["top_score"], 3)} for r in recs if r["split"] == "test" and r["expect"] == "abstain" and not r["abstained_correctly"]],
        "retrieval_misses": [{"id": r["id"], "mrr": r["mrr_at_4"]} for r in recs if r["split"] == "test" and r["expect"] == "answer" and r["mrr_at_4"] < 1.0],
    }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden": {"path": str(GOLDEN_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest(), "items": len(items),
                   "strata": dict(Counter(i.stratum for i in items)), "split": dict(Counter(i.split for i in items))},
        "system": {"retrieval_threshold": tau, "design_threshold": DESIGN_THRESHOLD, "answerer": (f"llm:{llm.model}" if llm is not None else "extractive"), "llm": llm_report, "nli_runtime": "nli-deberta-v3-xsmall (+guards)",
                   "nli_eval": "nli-MiniLM2-L6-H768 (independent)", "reranker": "ms-marco-MiniLM-L-6-v2", "embedder": "bge-small-en-v1.5"},
        "calibration": cal, "metrics": splits, "by_stratum_test": by_stratum, "gates": {"judged_on": "test", "which": a.gates, "results": gates,
        "all_pass": all(g["status"] == "PASS" for g in gates)}, "failures": failures, "caveats": CAVEATS,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    print_report(result)
    stack["pools"].shutdown()
    return 0 if result["gates"]["all_pass"] else 1


def print_report(r: dict[str, Any]) -> None:
    t = r["metrics"]["test"]
    print(f"\n=== TEST split: {t['n']} questions ({t['n_answerable']} answerable, {t['n_negative']} negative) | threshold {r['system']['retrieval_threshold']} ===")
    print(f"outcomes: {t['outcomes']}")
    print(f"answerable: answered {t['answerable_answered_rate']:.1%} | abstained {t['answerable_abstained_rate']:.1%} | refused {t['answerable_refused_rate']:.1%}")
    print(f"negatives : abstention accuracy {t['abstention_accuracy']:.1%} (out-of-scope {t['abstention_accuracy_oos']:.1%}, uncovered {t['abstention_accuracy_uncovered']:.1%})")
    print(f"retrieval : MRR@4 {t['mrr_at_4']:.3f} | nDCG@4 {t['ndcg_at_4']:.3f} | Recall@3 {t['recall_at_3']:.3f} | Recall@50 {t['recall_at_50']:.3f} (trivial: 12 chunks)")
    print("\nGATES (judged on test):")
    for g in r["gates"]["results"]:
        print(f"  [{g['status']}] {g['gate']:22s} {g['kpi']:7s} {g['value']:.4f} {g['op']} {g['threshold']}")
    print("\nby stratum (test): " + " | ".join(f"{s}: recall {m['context_recall']:.2f}, prec {m['context_precision']:.2f}" for s, m in r["by_stratum_test"].items() if m["n_answerable"]))
    f = r["failures"]
    print(f"failures: {len(f['answerable_not_answered'])} answerable not answered, {len(f['negatives_answered'])} negatives answered, {len(f['retrieval_misses'])} retrieval misses")
    print(f"\nresult: {'ALL GATES PASS' if r['gates']['all_pass'] else 'GATE FAILURE'}  (full detail: evals/results.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None, help="fix the abstention threshold instead of calibrating on dev")
    ap.add_argument("--calibrate", choices=["safety", "balanced"], default="safety")
    ap.add_argument("--llm", action="store_true", help="answer with the OpenRouter LLM (needs OPENROUTER_API_KEY; costs real money, ~$0.03)")
    ap.add_argument("--dump-unsupported", default="", help="write answers with judge-unsupported claims to this JSON file (diagnostics)")
    ap.add_argument("--gates", choices=["all", "ragas"], default="all")
    ap.add_argument("--limit-per-stratum", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "evals/results.json"))
    args = ap.parse_args()
    if args.dump_unsupported:
        DUMP_UNSUPPORTED.append({"_sentinel": True})
    rc = asyncio.run(amain(args))
    if args.dump_unsupported:
        Path(args.dump_unsupported).write_text(json.dumps(DUMP_UNSUPPORTED[1:], indent=1))
    sys.exit(rc)
