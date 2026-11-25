import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mcp_servers import manifest as M
from mcp_servers.graphify_server import build_server as build_graph
from mcp_servers.rag_server import build_server as build_rag
from mcp_servers.state_server import build_server as build_state
from tests.test_rag_pipeline import ROOT

TIERS = {"hybrid_retrieve": 1}
NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)


@pytest.fixture
def keys(tmp_path):
    priv, pub = M.generate_keypair(tmp_path / "keys")
    other_priv, other_pub = M.generate_keypair(tmp_path / "other", "attacker")
    return priv, pub, other_priv, other_pub


@pytest.fixture
async def signed(keys):
    adv = await M.describe_server(build_rag(object()))
    manifest = M.build_manifest("hybrid-rag", "1.0.0", "stdio", adv, TIERS, now=NOW)
    return M.sign_manifest(manifest, keys[0]), adv


def test_private_key_is_owner_only(keys):
    assert (keys[0].stat().st_mode & 0o777) == 0o600


async def test_valid_signed_manifest_verifies(signed, keys):
    s, _ = signed
    m = M.verify_signed(s, keys[1], now=NOW + timedelta(days=1))
    assert m["server"] == "hybrid-rag" and m["tools"]["hybrid_retrieve"]["tier"] == 1


async def test_any_edit_to_the_manifest_breaks_the_signature(signed, keys):
    s, _ = signed
    for mutate in (lambda m: m["tools"]["hybrid_retrieve"].update(tier=3), lambda m: m.update(server="evil"),
                   lambda m: m["tools"].update(extra={"input_schema_sha256": "0" * 64, "tier": 1}),
                   lambda m: m.update(expires_at="2999-01-01T00:00:00+00:00")):
        bad = copy.deepcopy(s)
        mutate(bad["manifest"])
        with pytest.raises(M.ManifestError, match="signature is invalid"):
            M.verify_signed(bad, keys[1], now=NOW)


async def test_signature_from_an_untrusted_key_is_rejected(signed, keys):
    s, adv = signed
    forged = M.sign_manifest(s["manifest"], keys[2])  # attacker signs the same content with their own key
    with pytest.raises(M.ManifestError, match="untrusted key"):
        M.verify_signed(forged, keys[1], now=NOW)
    spoof = {**forged, "key_id": s["key_id"]}  # ...and claims to be the trusted key id
    with pytest.raises(M.ManifestError, match="signature is invalid"):
        M.verify_signed(spoof, keys[1], now=NOW)


async def test_expired_and_malformed_manifests_are_rejected(signed, keys):
    s, _ = signed
    with pytest.raises(M.ManifestError, match="expired"):
        M.verify_signed(s, keys[1], now=NOW + timedelta(days=400))
    for broken in ({}, {"manifest": {}}, {**s, "signature": "!!not-base64!!"}, {**s, "alg": "none"}):
        with pytest.raises(M.ManifestError):
            M.verify_signed(broken, keys[1], now=NOW)


async def test_live_server_matching_the_manifest_has_no_drift(signed):
    s, adv = signed
    assert M.verify_live(s["manifest"], adv) == []


async def test_live_drift_is_detected_for_added_removed_and_changed_tools(signed):
    s, adv = signed
    added = {**adv, "tools": {**adv["tools"], "drop_database": "f" * 64}}
    assert M.verify_live(s["manifest"], added) == ["unsigned tool exposed: drop_database"]
    assert M.verify_live(s["manifest"], {**adv, "tools": {}}) == ["signed tool missing: hybrid_retrieve"]
    changed = {**adv, "tools": {"hybrid_retrieve": "e" * 64}}
    assert M.verify_live(s["manifest"], changed) == ["tool contract changed: hybrid_retrieve"]
    assert M.verify_live(s["manifest"], {**adv, "resource_templates": []}) == ["resource templates differ from the signed manifest"]


def test_manifest_requires_a_tier_for_exactly_the_advertised_tools():
    adv = {"tools": {"a": "0" * 64, "b": "1" * 64}, "resource_templates": []}
    with pytest.raises(M.ManifestError):
        M.build_manifest("s", "1", "stdio", adv, {"a": 1})
    with pytest.raises(M.ManifestError):
        M.build_manifest("s", "1", "stdio", adv, {"a": 1, "b": 1, "c": 1})


# ---- config loading: fail closed ----
def _cfg(tmp_path, signed_rag, pub, **over):
    (tmp_path / "manifests").mkdir(exist_ok=True)
    (tmp_path / "manifests/hybrid-rag.manifest.json").write_text(json.dumps(signed_rag))
    cfg = {"mcpServers": {"hybrid-rag": {"transport": "stdio", "required": True, "manifest": "manifests/hybrid-rag.manifest.json"},
                          "optional-one": {"transport": "stdio", "required": False, "manifest": "manifests/missing.json"},
                          "off": {"transport": "stdio", "enabled": False}},
           "manifest_policy": {"public_key": str(pub), "require_signed": True}}
    cfg.update(over)
    p = tmp_path / "mcp_config.json"
    p.write_text(json.dumps(cfg))
    return p


async def test_config_loader_accepts_valid_skips_optional_and_disabled(signed, keys, tmp_path):
    s, _ = signed
    out = M.load_verified_config(_cfg(tmp_path, s, keys[1]), tmp_path, now=NOW)
    assert set(out) == {"hybrid-rag"}  # 'optional-one' has no manifest -> skipped; 'off' is disabled


async def test_config_loader_refuses_to_start_when_a_required_manifest_is_invalid(signed, keys, tmp_path):
    s, _ = signed
    tampered = copy.deepcopy(s)
    tampered["manifest"]["version"] = "9.9.9"
    with pytest.raises(M.ManifestError, match="required server 'hybrid-rag'"):
        M.load_verified_config(_cfg(tmp_path, tampered, keys[1]), tmp_path, now=NOW)
    with pytest.raises(M.ManifestError, match="required server 'hybrid-rag'"):  # signed by the wrong key
        M.load_verified_config(_cfg(tmp_path, M.sign_manifest(s["manifest"], keys[2]), keys[1]), tmp_path, now=NOW)


async def test_a_manifest_cannot_be_reused_for_a_different_server_or_transport(signed, keys, tmp_path):
    s, _ = signed
    p = _cfg(tmp_path, s, keys[1])
    cfg = json.loads(p.read_text())
    cfg["mcpServers"]["hybrid-rag"]["transport"] = "streamable-http"
    p.write_text(json.dumps(cfg))
    with pytest.raises(M.ManifestError):
        M.load_verified_config(p, tmp_path, now=NOW)


# ---- the committed manifests ----
async def test_committed_manifests_verify_and_match_the_servers_in_this_repo():
    """Fails if someone changes a tool contract without re-running scripts/sign_manifests.py."""
    pub = ROOT / "keys/manifest_signing.pub"
    if not pub.exists():
        pytest.skip("public key not present")
    servers = M.load_verified_config(ROOT / "mcp_config.json", ROOT)
    assert set(servers) == {"hybrid-rag", "state-store", "graphify-vault"}
    for name, builder in (("hybrid-rag", build_rag), ("state-store", build_state), ("graphify-vault", build_graph)):
        live = await M.describe_server(builder(object()))
        assert M.verify_live(servers[name].manifest, live) == [], name
    assert servers["state-store"].manifest["tools"]["submit_transaction"]["tier"] == 3
