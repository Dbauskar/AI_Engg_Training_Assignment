"""Convert the policy notes in knowledge_vault/*.md into ATOMIC notes (one rule each) under knowledge_vault/atomic/.

Each atomic note: YAML frontmatter (id, title, domain, risk_tier, version, part_of) + the rule verbatim from its policy
(so the policy chunk the RAG index stores always contains it: citations and verification keep working) + [[wikilinks]]
to its policy ("part_of") and to curated related notes. Deterministic: re-running rewrites identical files.

Usage: uv run python scripts/build_vault.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp_servers.rag_pipeline import parse_note  # noqa: E402

STOP = set("a an and are as at be by for from has have in is it of on or that the this to was with must can may per".split())

# curated cross-policy links, by (policy, sentence index) -> list of (policy, sentence index)
RELATED = {
    ("refunds", 2): [("wires", 0)],            # manager approval for big refunds <-> two approvers for big wires
    ("wires", 0): [("refunds", 2), ("wires", 2)],
    ("wires", 1): [("wires", 0)],
    ("wires", 2): [("wires", 0), ("changemgmt", 0)],
    ("retention", 2): [("vendorrisk", 1), ("incident", 2)],   # personal data handling
    ("incident", 2): [("retention", 2)],
    ("vendorrisk", 1): [("retention", 2)],
    ("passwords", 0): [("accessctl", 0)],
    ("accessctl", 0): [("passwords", 0)],
    ("changemgmt", 1): [("changemgmt", 2), ("incident", 0)],  # freeze exceptions <-> emergency response
    ("incident", 0): [("changemgmt", 1)],
    ("expenses", 2): [("refunds", 2)],
    ("pto", 1): [("remote", 1)],
}
# which code implements a rule: atomic (policy, sentence index) -> node ids in the code graph
ENFORCED_BY = {
    ("wires", 0): ["schema:SubmitTransactionIn", "module:app.state_store"],            # approvals are stored as PENDING_APPROVAL
    ("wires", 2): ["schema:SubmitTransactionOut", "module:plugins.builtin", "module:agents.react", "module:app.db"],
    # ^ HitlGate vetoes claims of execution; the loop freezes at approval; the DB CHECK forbids approver == requester
    ("accessctl", 2): ["module:app.state_store"],                                      # hash-chained audit log
}


def sentences(body: str) -> list[str]:
    body = re.sub(r"^# .*\n+", "", body.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]


def slug(sentence: str, n: int = 3) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if w not in STOP]
    return "-".join(words[:n])


def main() -> None:
    policies = {}
    for p in sorted((ROOT / "knowledge_vault").glob("*.md")):
        meta, body = parse_note(p.read_text())
        policies[meta["id"]] = (meta, sentences(body))
    ids = {(pid, i): f"{pid}-{slug(s)}" for pid, (_, ss) in policies.items() for i, s in enumerate(ss)}
    assert len(set(ids.values())) == len(ids), "slug collision"
    out = ROOT / "knowledge_vault" / "atomic"
    for old in out.glob("*.md"):
        old.unlink()
    for (pid, i), nid in ids.items():
        meta, ss = policies[pid]
        title = ss[i].rstrip(".")
        fm = [f"id: {nid}", f"title: {title}", f"domain: {meta['domain']}", f"risk_tier: {meta['risk_tier']}",
              f"version: {meta['version']}", f"part_of: {pid}"]
        if (pid, i) in ENFORCED_BY:
            fm.append("enforced_by: [" + ", ".join(ENFORCED_BY[(pid, i)]) + "]")
        links = [f"[[{pid}]]"] + [f"[[{ids[k]}]]" for k in RELATED.get((pid, i), [])]
        body = f"{ss[i]}\n\nPart of {links[0]}." + (f" See also {', '.join(links[1:])}." if len(links) > 1 else "")
        (out / f"{nid}.md").write_text("---\n" + "\n".join(fm) + "\n---\n" + body + "\n")
    print(f"wrote {len(ids)} atomic notes from {len(policies)} policies")


if __name__ == "__main__":
    main()
