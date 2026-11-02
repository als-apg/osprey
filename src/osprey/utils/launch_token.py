"""The launch-token check every arming service applies to its callers.

A service that can set hardware moving guards its arming routes with a launch
token read from its own environment: the caller that was handed the token
presents it in a header, and every other process that can reach the port is
refused. One check, parameterised by the environment variable that holds the
token, so each service names its own token while all of them refuse the same
way and under the same codes.

Fail-closed by design. A service whose variable is unset — or set to a blank
value, which a deploy renders when it has nothing to put there — arms nothing,
whatever header a caller sends. The variable is read on every request, so a
service never answers from a token it cached before its environment changed.

Two coded refusals, each an :class:`~fastapi.HTTPException` whose ``detail`` is
``{"code", "detail"}`` so a client branches on ``detail.code`` and never on
prose or on the status alone:

- ``503`` :data:`LAUNCH_TOKEN_REQUIRED` — this service has no launch token
  configured, so no caller can arm it; the remedy is the operator's.
- ``403`` :data:`LAUNCH_TOKEN_INVALID` — a token is configured and the caller
  presented none, or a different one.

Neither refusal echoes the configured or the presented token. The comparison is
constant-time over :func:`osprey.utils.bearer.credential_bytes`, so it is total
over every header a caller can send.
"""

from __future__ import annotations

import hmac
import os

from osprey.utils.bearer import credential_bytes

__all__ = [
    "LAUNCH_TOKEN_HEADER",
    "LAUNCH_TOKEN_INVALID",
    "LAUNCH_TOKEN_REQUIRED",
    "require_launch_token",
    "verify_launch_token",
]

#: The request header a caller presents its launch token in.
LAUNCH_TOKEN_HEADER = "X-Launch-Token"  # noqa: S105 -- names a header, holds no secret

#: The service has no launch token configured: nothing it guards can be armed.
LAUNCH_TOKEN_REQUIRED = "launch_token_required"  # noqa: S105 -- an error code, holds no secret

#: A token is configured and the caller's is missing or does not match it.
LAUNCH_TOKEN_INVALID = "launch_token_invalid"  # noqa: S105 -- an error code, holds no secret


def require_launch_token(env_name: str) -> str:
    """Return the launch token configured in ``env_name``, or refuse 503.

    A blank value is no token: it is refused exactly as an unset variable is.
    The refusal names the variable but never a value.

    Args:
        env_name: The environment variable that holds this service's token.

    Returns:
        The configured token, as the environment holds it.

    Raises:
        HTTPException: ``503`` with code :data:`LAUNCH_TOKEN_REQUIRED`.
    """
    expected = os.environ.get(env_name, "")
    if not expected.strip():
        # Readers of the codes import this module without the web framework.
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail={
                "code": LAUNCH_TOKEN_REQUIRED,
                "detail": f"launch disabled: {env_name} is not configured",
            },
        )
    return expected


def verify_launch_token(presented: str | None, env_name: str) -> None:
    """Refuse unless ``presented`` matches the token configured in ``env_name``.

    The configured side is checked first, so an unconfigured service answers
    503 whatever the caller presented.

    Args:
        presented: The caller's header value; ``None`` or ``""`` when absent.
        env_name: The environment variable that holds this service's token.

    Raises:
        HTTPException: ``503`` :data:`LAUNCH_TOKEN_REQUIRED` when no token is
            configured; ``403`` :data:`LAUNCH_TOKEN_INVALID` when the presented
            token is missing or wrong.
    """
    expected = require_launch_token(env_name)
    if not presented or not hmac.compare_digest(
        credential_bytes(presented), credential_bytes(expected)
    ):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=403,
            detail={"code": LAUNCH_TOKEN_INVALID, "detail": "invalid or missing launch token"},
        )
