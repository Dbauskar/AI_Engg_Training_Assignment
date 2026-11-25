"""Create a signing key (first run) and sign manifests for the servers in mcp_config.json.

Usage: uv run python scripts/sign_manifests.py
The PRIVATE key stays in ./keys (gitignored, mode 600). Commit keys/manifest_signing.pub and manifests/*.json.
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.harness import Tier  # noqa: E402
from mcp_servers import manifest as M  # noqa: E402
from mcp_servers.graphify_server import build_server as build_graph  # noqa: E402
from mcp_servers.rag_server import build_server as build_rag  # noqa: E402
from mcp_servers.state_server import build_server as build_state  # noqa: E402

TIERS = {"hybrid-rag": {"hybrid_retrieve": int(Tier.TIER_1_READ)},
         "state-store": {"get_session_facts": 1, "write_session_fact": 2, "submit_transaction": 3},
         "graphify-vault": {"graphify_subgraph": 1}}


async def main() -> None:
    keys = ROOT / "keys"
    if not (keys / "manifest_signing.key").exists():
        M.generate_keypair(keys)
        print("generated new signing key pair in", keys)
    servers = {"hybrid-rag": (build_rag(object()), "stdio"), "state-store": (build_state(object()), "streamable-http"),
               "graphify-vault": (build_graph(object()), "stdio")}  # tool lists need no live backends
    for name, (server, transport) in servers.items():
        advertised = await M.describe_server(server)
        manifest = M.build_manifest(name, "1.0.0", transport, advertised, TIERS[name])
        signed = M.sign_manifest(manifest, keys / "manifest_signing.key")
        out = ROOT / "manifests" / f"{name}.manifest.json"
        out.write_text(json.dumps(signed, indent=2, sort_keys=True))
        print("signed", out.relative_to(ROOT), "tools:", sorted(manifest["tools"]))


asyncio.run(main())
