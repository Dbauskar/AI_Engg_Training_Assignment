"""Graphify: a unified structural graph of policy notes, Python modules and Pydantic schemas, plus minimal-subgraph
extraction for token-efficient prompt context (docs/design.md Gate 17).

The blueprint's `graphify` CLI is not a real public package, so this is a small self-contained implementation.

Nodes   policy:<id>  hub policy notes (knowledge_vault/*.md)         note:<id>   atomic notes (knowledge_vault/atomic)
        module:<dotted>  Python modules (AST-parsed)                 schema:<Class>  Pydantic models (AST-parsed)
Edges   part_of (note->policy) · links_to ([[wikilinks]]) · enforced_by (note->code) · imports (module->module)
        defines (module->schema) · inherits (schema->schema)

Subgraph(query): resolve the ANCHOR note (BM25 over notes), walk at most `depth` (<=2) hops, then keep only nodes that
are relevant to the query, within a token budget, adding a bridge node only when needed for connectivity.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.tokens import count_tokens, truncate_tokens
from mcp_servers.rag_pipeline import bm25_tokens

PROJECT_PACKAGES = ("app", "agents", "mcp_servers", "plugins")
SCHEMA_ROOTS = {"BaseModel", "ToolInputBase"}
WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
GRAPH_VERSION = 1


# ---- parsing ---------------------------------------------------------------------------------------------------------
def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return {}, raw
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            meta[k.strip()] = [x.strip() for x in v[1:-1].split(",") if x.strip()] if v.startswith("[") and v.endswith("]") else v
    return meta, m.group(2).strip()


def _first_line(doc: str | None, limit: int = 160) -> str:
    return (doc or "").strip().split("\n")[0][:limit]


def _py_files(root: Path, packages: tuple[str, ...]) -> list[Path]:
    out = []
    for pkg in packages:
        out += sorted((root / pkg).glob("*.py"))
    return out


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def build_graph(root: str | Path, vault: str = "knowledge_vault", packages: tuple[str, ...] = PROJECT_PACKAGES) -> dict[str, Any]:
    root = Path(root)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    dangling: list[str] = []

    # --- policies and atomic notes ---
    note_files = {"policy": sorted((root / vault).glob("*.md")), "note": sorted((root / vault / "atomic").glob("*.md"))}
    pending_links: list[tuple[str, str, str]] = []
    for kind, files in note_files.items():
        for p in files:
            meta, body = parse_frontmatter(p.read_text())
            nid = f"{kind}:{meta['id']}"
            nodes[nid] = {"id": nid, "type": kind, "title": meta.get("title", meta["id"]), "text": re.sub(r"^# .*\n+", "", body).strip(),
                          "domain": meta.get("domain"), "risk_tier": int(meta.get("risk_tier", 1)), "version": meta.get("version"),
                          "hub": meta.get("part_of") if kind == "note" else meta["id"], "path": str(p.relative_to(root))}
            if kind == "note":
                pending_links.append((nid, "part_of", f"policy:{meta['part_of']}"))
                for target in meta.get("enforced_by", []):
                    pending_links.append((nid, "enforced_by", target))
            for link in WIKILINK.findall(body):
                pending_links.append((nid, "links_to", link.strip()))

    # --- python modules and pydantic schemas ---
    class_bases: dict[str, list[str]] = {}
    class_info: dict[str, dict[str, Any]] = {}
    mod_imports: dict[str, set[str]] = {}
    for path in _py_files(root, packages):
        mod = _module_name(root, path)
        tree = ast.parse(path.read_text())
        publics = [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]
        nodes[f"module:{mod}"] = {"id": f"module:{mod}", "type": "module", "title": mod, "text": _first_line(ast.get_docstring(tree)),
                                  "domain": None, "risk_tier": 1, "version": None, "hub": None, "path": str(path.relative_to(root)),
                                  "public": publics[:12]}
        imps: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                imps.add(n.module)
                imps |= {f"{n.module}.{a.name}" for a in n.names}
            elif isinstance(n, ast.Import):
                imps |= {a.name for a in n.names}
        mod_imports[mod] = imps
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "" for b in n.bases]
                class_bases[n.name] = bases
                fields = [s.target.id for s in n.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
                class_info[n.name] = {"module": mod, "fields": fields, "doc": _first_line(ast.get_docstring(n))}
    schema_names = set()
    changed = True
    while changed:  # a class is a schema if it derives (transitively) from BaseModel / ToolInputBase
        changed = False
        for name, bases in class_bases.items():
            if name not in schema_names and any(b in SCHEMA_ROOTS or b in schema_names for b in bases):
                schema_names.add(name)
                changed = True
    for name in sorted(schema_names):
        info = class_info[name]
        nodes[f"schema:{name}"] = {"id": f"schema:{name}", "type": "schema", "title": name,
                                   "text": (info["doc"] + " fields: " + ", ".join(info["fields"][:10])).strip(),
                                   "domain": None, "risk_tier": 1, "version": None, "hub": None, "path": info["module"]}
        edges.append({"src": f"module:{info['module']}", "dst": f"schema:{name}", "type": "defines"})
        for b in class_bases[name]:
            if b in schema_names:
                edges.append({"src": f"schema:{name}", "dst": f"schema:{b}", "type": "inherits"})
    for mod, imps in mod_imports.items():
        for target in sorted(i for i in imps if i.split(".")[0] in packages):
            tgt = f"module:{target}"
            if tgt in nodes and tgt != f"module:{mod}":
                edges.append({"src": f"module:{mod}", "dst": tgt, "type": "imports"})

    # --- resolve note links (wikilinks name a note or a policy; enforced_by already names a node) ---
    for src, etype, raw in pending_links:
        dst = raw if ":" in raw else (f"note:{raw}" if f"note:{raw}" in nodes else f"policy:{raw}")
        if dst in nodes:
            edges.append({"src": src, "dst": dst, "type": etype})
        else:
            dangling.append(f"{src} -> {raw}")
    seen, uniq = set(), []
    for e in edges:
        k = (e["src"], e["dst"], e["type"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return {"version": GRAPH_VERSION, "nodes": list(nodes.values()), "edges": uniq, "dangling": sorted(set(dangling))}


def save_graph(graph: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(graph, indent=1, sort_keys=True))


# ---- query-time graph ----------------------------------------------------------------------------------------------------
@dataclass
class SubNode:
    id: str
    type: str
    title: str
    text: str
    hub: str | None
    domain: str | None
    risk_tier: int
    distance: int
    relevance: float
    bridge: bool = False


@dataclass
class Subgraph:
    anchor: str | None
    anchor_score: float
    anchors: list[str] = field(default_factory=list)  # more than one when two notes match the query about equally
    nodes: list[SubNode] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    rendered: str = ""
    tokens: int = 0


class Graph:
    def __init__(self, data: dict[str, Any], embedder: Any | None = None) -> None:
        if data.get("version") != GRAPH_VERSION:
            raise ValueError("unsupported graph version")
        self.nodes: dict[str, dict[str, Any]] = {n["id"]: n for n in data["nodes"]}
        self.edges: list[dict[str, str]] = data["edges"]
        self.adj: dict[str, list[tuple[str, str]]] = {n: [] for n in self.nodes}
        for e in self.edges:  # traversal is undirected; direction is kept in the edge records
            self.adj[e["src"]].append((e["dst"], e["type"]))
            self.adj[e["dst"]].append((e["src"], e["type"]))
        self._ids = list(self.nodes)
        self._bm25 = BM25Okapi([self._doc_tokens(self.nodes[i]) for i in self._ids])
        self._emb = embedder  # optional: dense similarity fixes vocabulary mismatch (MFA vs multi-factor)
        self._note_ids = [i for i in self._ids if self.nodes[i]["type"] == "note"]
        self._note_vecs = None
        if embedder is not None and self._note_ids:
            m = np.asarray(embedder.encode([f"{self.nodes[i]['title']}. {self.nodes[i]['text']}" for i in self._note_ids], False), dtype=np.float32)
            self._note_vecs = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    @classmethod
    def load(cls, path: str | Path, embedder: Any | None = None) -> "Graph":
        return cls(json.loads(Path(path).read_text()), embedder)

    @staticmethod
    def _doc_tokens(n: dict[str, Any]) -> list[str]:
        return bm25_tokens(" ".join([n["title"], n["title"], n["text"], n["id"].split(":", 1)[1].replace("-", " ").replace(".", " ")]))

    def relevance(self, query: str) -> dict[str, float]:
        scores = self._bm25.get_scores(bm25_tokens(query))
        top = float(max(scores)) if len(scores) and max(scores) > 0 else 0.0
        return {i: (float(s) / top if top else 0.0) for i, s in zip(self._ids, scores)}

    def anchors(self, query: str, k: int = 1, hub_ids: list[str] | None = None, ratio: float = 0.7) -> list[tuple[str, float]]:
        """Best-matching atomic notes (optionally restricted to given policies). Score = BM25 alone, or BM25 and dense
        similarity averaged after min-max scaling. Returns the best note plus, up to k, others scoring >= ratio x best."""
        rel = self.relevance(query)
        pool = [i for i in self._note_ids if (hub_ids is None or self.nodes[i]["hub"] in hub_ids)]
        if not pool:
            return []
        score = {i: rel[i] for i in pool}
        if self._note_vecs is not None:
            q = np.asarray(self._emb.encode([query], False), dtype=np.float32)[0]
            q = q / max(float(np.linalg.norm(q)), 1e-9)
            cos = dict(zip(self._note_ids, (self._note_vecs @ q).tolist()))
            lo, hi = min(cos[i] for i in pool), max(cos[i] for i in pool)
            score = {i: 0.5 * rel[i] + 0.5 * ((cos[i] - lo) / (hi - lo) if hi > lo else 1.0) for i in pool}
        ranked = sorted(((i, s) for i, s in score.items() if s > 0), key=lambda p: (-p[1], p[0]))
        if not ranked:
            return []
        best = ranked[0][1]
        return [r for j, r in enumerate(ranked[:max(1, k)]) if j == 0 or r[1] >= ratio * best]

    @staticmethod
    def _line(n: dict[str, Any], bridge: bool = False) -> str:
        nid = n["id"].split(":", 1)[1]
        if n["type"] == "note":
            return f"[{nid}] {n['text'].split(chr(10))[0]}"
        if n["type"] == "policy":
            return f"[{nid}] (policy) {n['title']}"
        return f"[{n['id']}] ({n['type']}) {n['text'] or n['title']}"[:200]

    def subgraph(self, query: str, depth: int = 2, token_budget: int = 320, hub_ids: list[str] | None = None,
                 min_rel: float = 0.25, max_nodes: int = 8, max_anchors: int = 1) -> Subgraph:
        depth = max(1, min(depth, 2))
        a = self.anchors(query, max_anchors, hub_ids)
        if not a:
            return Subgraph(None, 0.0)
        (anchor, anchor_score), extra = a[0], [x for x, _ in a[1:]]
        rel = self.relevance(query)
        dist, parent, frontier = {anchor: 0, **{x: 0 for x in extra}}, {}, [anchor] + extra
        for d in range(1, depth + 1):
            nxt = []
            for u in frontier:
                for v, _ in self.adj[u]:
                    if v not in dist:
                        dist[v], parent[v] = d, u
                        nxt.append(v)
            frontier = nxt
        decay = {1: 1.0, 2: 0.7}
        cands = sorted((v for v in dist if v not in [anchor] + extra and self.nodes[v]["type"] != "policy" and rel[v] >= min_rel),
                       key=lambda v: (-(rel[v] * decay[dist[v]]), v))
        chosen: list[str] = [anchor] + extra
        bridges: set[str] = set()
        used = sum(count_tokens(self._line(self.nodes[x])) for x in chosen)
        for v in cands:
            if len(chosen) >= max_nodes:
                break
            need = [v] + ([parent[v]] if dist[v] == 2 and parent[v] not in chosen and parent[v] not in bridges else [])
            cost = sum(count_tokens(self._line(self.nodes[x])) for x in need)
            if used + cost > token_budget:
                continue
            used += cost
            if len(need) == 2:
                bridges.add(need[1])  # a connecting node, kept only because it links a selected node to the anchor
            chosen.append(v)
        order = list(chosen) + sorted(bridges)
        sel = set(order)
        edges = [e for e in self.edges if e["src"] in sel and e["dst"] in sel and e["type"] != "part_of" or
                 (e["src"] in sel and e["dst"] in sel and e["type"] == "part_of" and e["dst"] in bridges)]
        nodes = [SubNode(i, self.nodes[i]["type"], self.nodes[i]["title"], self.nodes[i]["text"], self.nodes[i]["hub"],
                         self.nodes[i]["domain"], self.nodes[i]["risk_tier"], dist[i], round(rel[i], 3), i in bridges) for i in order]
        rendered = truncate_tokens("\n".join(self._line(self.nodes[i]) for i in order), token_budget)
        return Subgraph(anchor, round(anchor_score, 3), [anchor] + extra, nodes, edges, rendered, count_tokens(rendered))
