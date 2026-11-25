"""Signed MCP server manifests (docs/design.md Gate 13: 'signed MCP server manifests').

A manifest pins what a server is allowed to be: its name, version, transport, and the exact tool contracts it exposes
(name + SHA-256 of each input JSON schema) and resource templates. It is signed with an Ed25519 PRIVATE key held by the
publisher; hosts hold only the PUBLIC key, so a compromised host cannot forge a manifest.

Host side, fail closed:
  verify_signed()  rejects bad signatures, unknown keys, malformed or expired manifests;
  verify_live()    compares what a running server advertises (tools/list, resource templates) with the signed manifest and
                   reports every drift: an added, removed or altered tool means "do not connect".
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ALG = "ed25519"


class ManifestError(Exception):
    pass


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def schema_hash(schema: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(schema)).hexdigest()


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_keypair(directory: str | Path, name: str = "manifest_signing") -> tuple[Path, Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    priv_path, pub_path = d / f"{name}.key", d / f"{name}.pub"
    priv_path.write_bytes(priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                             serialization.NoEncryption()))
    priv_path.chmod(0o600)
    pub_path.write_bytes(priv.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv_path, pub_path


def load_private(path: str | Path) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)  # type: ignore[return-value]


def load_public(path: str | Path) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(Path(path).read_bytes())  # type: ignore[return-value]


async def describe_server(server: Any) -> dict[str, Any]:
    """What a server really advertises right now (in-process; for a remote server pass an MCP Client instead)."""
    from mcp.client import Client

    async with Client(server) as c:
        return describe_from_lists((await c.list_tools()).tools, (await c.list_resource_templates()).resource_templates)


def describe_from_lists(tools: list[Any], templates: list[Any]) -> dict[str, Any]:
    return {"tools": {t.name: schema_hash(t.input_schema) for t in tools},
            "resource_templates": sorted(t.uri_template for t in templates)}


def build_manifest(name: str, version: str, transport: str, advertised: dict[str, Any], tiers: dict[str, int],
                   valid_days: int = 365, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if set(tiers) != set(advertised["tools"]):
        raise ManifestError("every advertised tool needs a declared tier (and nothing else)")
    return {"server": name, "version": version, "transport": transport,
            "tools": {t: {"input_schema_sha256": h, "tier": tiers[t]} for t, h in sorted(advertised["tools"].items())},
            "resource_templates": advertised["resource_templates"],
            "issued_at": now.isoformat(), "expires_at": (now + timedelta(days=valid_days)).isoformat()}


def sign_manifest(manifest: dict[str, Any], private_key_path: str | Path) -> dict[str, Any]:
    priv = load_private(private_key_path)
    return {"manifest": manifest, "alg": ALG, "key_id": key_id(priv.public_key()),
            "signature": base64.b64encode(priv.sign(canonical(manifest))).decode()}


def verify_signed(signed: dict[str, Any], public_key_path: str | Path, now: datetime | None = None) -> dict[str, Any]:
    """Return the manifest if (and only if) it is validly signed by the trusted key and unexpired."""
    try:
        manifest, sig, alg, kid = signed["manifest"], base64.b64decode(signed["signature"], validate=True), signed["alg"], signed["key_id"]
    except (KeyError, ValueError, TypeError) as e:
        raise ManifestError("malformed signed manifest") from e
    pub = load_public(public_key_path)
    if alg != ALG:
        raise ManifestError("unsupported signature algorithm")
    if kid != key_id(pub):
        raise ManifestError("manifest was signed by an untrusted key")
    try:
        pub.verify(sig, canonical(manifest))
    except InvalidSignature as e:
        raise ManifestError("manifest signature is invalid (content was altered or not signed by the trusted key)") from e
    now = now or datetime.now(timezone.utc)
    try:
        if now >= datetime.fromisoformat(manifest["expires_at"]):
            raise ManifestError("manifest has expired")
    except (KeyError, ValueError) as e:
        raise ManifestError("manifest has no valid expiry") from e
    return manifest


def verify_live(manifest: dict[str, Any], advertised: dict[str, Any]) -> list[str]:
    """Differences between the signed manifest and what the server advertises. Empty list = matches exactly."""
    problems = []
    signed_tools = {t: v["input_schema_sha256"] for t, v in manifest["tools"].items()}
    for t in sorted(set(advertised["tools"]) - set(signed_tools)):
        problems.append(f"unsigned tool exposed: {t}")
    for t in sorted(set(signed_tools) - set(advertised["tools"])):
        problems.append(f"signed tool missing: {t}")
    for t in sorted(set(signed_tools) & set(advertised["tools"])):
        if signed_tools[t] != advertised["tools"][t]:
            problems.append(f"tool contract changed: {t}")
    if sorted(manifest.get("resource_templates", [])) != sorted(advertised["resource_templates"]):
        problems.append("resource templates differ from the signed manifest")
    return problems


@dataclass(frozen=True)
class VerifiedServer:
    name: str
    config: dict[str, Any]
    manifest: dict[str, Any] | None


def load_verified_config(config_path: str | Path, root: str | Path = ".", now: datetime | None = None) -> dict[str, VerifiedServer]:
    """Load mcp_config.json and verify every enabled server's manifest. A required server without a valid signed
    manifest aborts startup (fail closed); an optional one is skipped."""
    root = Path(root)
    cfg = json.loads(Path(config_path).read_text())
    policy = cfg.get("manifest_policy", {})
    pub = root / policy.get("public_key", "keys/manifest_signing.pub")
    out: dict[str, VerifiedServer] = {}
    for name, entry in cfg["mcpServers"].items():
        if entry.get("enabled", True) is False:
            continue
        try:
            if policy.get("require_signed", True):
                path = root / entry["manifest"]
                manifest = verify_signed(json.loads(path.read_text()), pub, now)
                if manifest["server"] != name or manifest["transport"] != entry["transport"]:
                    raise ManifestError(f"manifest does not describe server '{name}' on transport '{entry['transport']}'")
            else:
                manifest = None
        except (ManifestError, KeyError, FileNotFoundError, json.JSONDecodeError) as e:
            if entry.get("required", True):
                raise ManifestError(f"required server '{name}' has no valid signed manifest: {e}") from e
            continue
        out[name] = VerifiedServer(name, entry, manifest)
    return out
