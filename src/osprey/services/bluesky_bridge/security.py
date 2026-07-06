"""The Bluesky bridge's promote-token gate.

``POST /runs/{id}/promote`` (task 1.5's ``app.py``) is the only HTTP route that
starts a real scan; this module is its sole guard. Fail-closed by design: if
``BLUESKY_PROMOTE_TOKEN`` isn't set in the bridge process's environment, the
promote path is simply not armed, regardless of what header a caller sends.
This is arming/network protection only — the authoritative safety check is the
agent-side ``launch_scan`` tool's in-tool ``writes_enabled`` re-read (task 1.8),
which runs before this token is ever sent. Mirrors BELLA's
``runs.require_armed`` / ``promote_intent`` token check.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException

_PROMOTE_TOKEN_ENV = "BLUESKY_PROMOTE_TOKEN"


def require_armed() -> str:
    """Return the configured promote token, or raise 503 if none is set.

    Unset means the operator has not armed promotion for this bridge instance
    at all — the failure detail names the missing env var but never a token
    value (there isn't one to leak).
    """
    expected = os.environ.get(_PROMOTE_TOKEN_ENV)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"promotion disabled: {_PROMOTE_TOKEN_ENV} is not configured",
        )
    return expected


def verify_promote_token(header: str | None) -> None:
    """Raise 503 (unarmed) or 403 (mismatch) unless ``header`` matches the armed token.

    Compares in constant time via :func:`secrets.compare_digest` so a mismatch
    can't be distinguished by timing. The error details never echo the
    expected or received token.
    """
    expected = require_armed()
    if not secrets.compare_digest(header or "", expected):
        raise HTTPException(status_code=403, detail="invalid or missing promote token")
