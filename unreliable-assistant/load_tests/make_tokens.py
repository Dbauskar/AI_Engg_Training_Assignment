"""Mint a pool of tokens for the load test. Usage: uv run python -m load_tests.make_tokens [--n 200] [--ttl 7200]
Writes load_tests/tokens.json (git-ignored; tokens are credentials). Users are spread over virtual users, so each VU's
requests carry a real per-user identity. All use tenant acme-corp, the only tenant with an index in this repo."""
import argparse
import json
from pathlib import Path

from app.auth import mint_token
from app.config import _req, load_env
from app.harness import Tier

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ttl", type=int, default=7200)
    ap.add_argument("--tenant", default="acme-corp")
    ap.add_argument("--csv", action="store_true", help="also write tokens.csv for JMeter")
    a = ap.parse_args()
    load_env()
    secret = _req("API_TOKEN_SECRET").encode()
    toks = [mint_token(secret, f"load{i}", a.tenant, Tier.TIER_1_READ, a.ttl) for i in range(a.n)]
    Path(__file__).with_name("tokens.json").write_text(json.dumps(toks))
    if a.csv:
        Path(__file__).with_name("tokens.csv").write_text("\n".join(toks) + "\n")
    print(f"wrote {len(toks)} tokens")
