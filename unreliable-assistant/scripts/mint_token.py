"""Mint a development API token. Usage: uv run python scripts/mint_token.py --tenant acme-corp --user alice [--tier 1] [--ttl 3600]"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.auth import mint_token  # noqa: E402
from app.config import _req, load_env  # noqa: E402
from app.harness import Tier  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tenant", required=True); ap.add_argument("--user", required=True)
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2, 3]); ap.add_argument("--ttl", type=int, default=3600); a = ap.parse_args()
    load_env()
    print(mint_token(_req("API_TOKEN_SECRET").encode(), a.user, a.tenant, Tier(a.tier), a.ttl))
