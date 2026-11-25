"""Caller identity at the API edge: signed bearer tokens carrying tenant, user and a tier CEILING.

The API takes tenant/user/tier ONLY from a verified token, never from the request body, so a client cannot claim another
tenant or a higher tier. Format: ua1.<base64url(JSON claims)>.<base64url(HMAC-SHA256)>. HS256-style shared secret:
fine for one service that both mints and verifies; use an asymmetric scheme (as in the signed manifests) if a separate
identity provider issues tokens.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass

from app.contracts import TENANT_RE, USER_RE
from app.harness import Tier

PREFIX = "ua1"
MIN_SECRET_BYTES = 32


class TokenError(Exception):
    """Deliberately vague: callers learn only that the token was not accepted."""


@dataclass(frozen=True)
class Claims:
    user_id: str
    tenant_id: str
    max_tier: Tier
    exp: int


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(secret: bytes, body: str) -> str:
    return _b64(hmac.new(secret, f"{PREFIX}.{body}".encode(), hashlib.sha256).digest())


def mint_token(secret: bytes, user_id: str, tenant_id: str, max_tier: Tier = Tier.TIER_1_READ, ttl_s: int = 3600,
               now: float | None = None) -> str:
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(f"signing secret must be at least {MIN_SECRET_BYTES} bytes")
    if not re.fullmatch(TENANT_RE, tenant_id) or not re.fullmatch(USER_RE, user_id):
        raise ValueError("malformed tenant or user id")
    t = int(now if now is not None else time.time())
    body = _b64(json.dumps({"sub": user_id, "tid": tenant_id, "tier": int(max_tier), "iat": t, "exp": t + ttl_s, "v": 1},
                           sort_keys=True, separators=(",", ":")).encode())
    return f"{PREFIX}.{body}.{_sign(secret, body)}"


def verify_token(secret: bytes, token: str, now: float | None = None) -> Claims:
    try:
        prefix, body, sig = token.split(".")
        if prefix != PREFIX or not hmac.compare_digest(sig, _sign(secret, body)):
            raise TokenError("bad signature")
        c = json.loads(_unb64(body))
        if c.get("v") != 1 or int(c["exp"]) <= (now if now is not None else time.time()):
            raise TokenError("expired")
        if not re.fullmatch(TENANT_RE, c["tid"]) or not re.fullmatch(USER_RE, c["sub"]) or int(c["tier"]) not in (1, 2, 3):
            raise TokenError("malformed claims")
        return Claims(c["sub"], c["tid"], Tier(int(c["tier"])), int(c["exp"]))
    except TokenError:
        raise
    except Exception as e:  # noqa: BLE001 - any parse failure is just 'not accepted'
        raise TokenError("malformed token") from e
