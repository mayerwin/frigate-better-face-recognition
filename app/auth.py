"""Optional authentication that piggybacks on Frigate's own user database.

When AUTH=frigate, the companion serves a login page that validates the
submitted credentials against Frigate's `POST /api/login` (the same users and
password as Frigate). On success it issues its own stateless signed-cookie
session (an HMAC over an expiry timestamp), so no Frigate token is stored and
nothing extra has to be managed. AUTH=none disables it (front the app with your
own reverse-proxy auth / VPN instead).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

COOKIE = "bfr_session"
TTL = 7 * 24 * 3600  # 7 days


def load_secret(data_dir: str) -> bytes:
    """Load (or create + persist) the cookie-signing secret in the data dir."""
    path = os.path.join(data_dir, ".session_secret")
    try:
        with open(path, "rb") as f:
            s = f.read()
        if len(s) >= 16:
            return s
    except Exception:
        pass
    s = os.urandom(32)
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(s)
        os.chmod(path, 0o600)
    except Exception:
        pass
    return s


def make_token(secret: bytes, ttl: int = TTL) -> str:
    msg = str(int(time.time()) + ttl).encode()
    sig = hmac.new(secret, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(msg + b"." + sig).decode()


def valid_token(secret: bytes, token: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        msg, sig = raw.rsplit(b".", 1)
        expected = hmac.new(secret, msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return False
        return int(msg) > time.time()
    except Exception:
        return False
