"""The deployment's two gates on what a caller may make the optimizer do.

**The launch token.** The routes that move hardware (start, resume, the
results restore, ``PUT /environments/{name}/variables``, ``POST
/environments/{name}/evaluate``) and the one that accepts code (a pickled
surrogate model is executed when it is loaded, so ``POST /models``) require the
``X-Launch-Token`` header to match ``OPTIMIZER_LAUNCH_TOKEN``. The gate fails
closed: with no token configured those routes answer 503 rather than run for
anyone. Routes that only read, and the ones that stop motion (pause, cancel),
are not gated: stopping a run must never wait on a credential.

**Drafts are open.** Creating a draft (``POST /optimizations``), editing it
(``PUT /jobs/{id}/config``) and pointing the cockpit at one (``PUT
/jobs/current``) need no token. A draft write reads the target's setpoints, to
centre the draft's bounds on them, as the open read routes do, and never writes
to the machine or loads a model. What a draft may not do is change under a
review: ``GET /jobs/{id}/config`` sends the draft's revision in
``X-Config-Revision``, and a start must name it (``config_revision``), so a
draft written after it was shown is refused with 409 instead of run.

**The allowed targets.** ``OPTIMIZER_TARGETS`` lists the control targets this
deployment's optimizer may drive (``va,standin``). A request or a run naming
any other target is refused with 403, before an environment is loaded. Unset,
every target is allowed: that is a process run by hand, not a deployment, whose
compose file always sets it.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status

LAUNCH_TOKEN_ENV = "OPTIMIZER_LAUNCH_TOKEN"  # noqa: S105 -- names an environment variable, holds no secret
LAUNCH_TOKEN_HEADER = "X-Launch-Token"  # noqa: S105 -- names a header, holds no secret
TARGETS_ENV = "OPTIMIZER_TARGETS"


def launch_token_refusal(presented: str) -> HTTPException | None:
    """Why *presented* does not arm a gated route, or ``None`` when it does.

    503 with no token configured, 403 unless it matches. Compared as bytes in
    constant time: a header is decoded as latin-1, and ``compare_digest``
    refuses a non-ASCII ``str``. Neither detail ever echoes a token.
    """
    expected = os.environ.get(LAUNCH_TOKEN_ENV)
    if not expected:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"launch disabled: {LAUNCH_TOKEN_ENV} is not configured",
        )
    if not secrets.compare_digest(presented.encode(), expected.encode()):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="invalid or missing launch token"
        )
    return None


def require_launch_token(
    x_launch_token: str = Header(default="", alias=LAUNCH_TOKEN_HEADER),
) -> None:
    """The route dependency over :func:`launch_token_refusal`."""
    refusal = launch_token_refusal(x_launch_token)
    if refusal is not None:
        raise refusal


def allowed_targets() -> frozenset[str] | None:
    """The targets this deployment may drive, or ``None`` when unrestricted."""
    raw = os.environ.get(TARGETS_ENV)
    if raw is None:
        return None
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def check_target_allowed(target: str) -> None:
    """403 unless this deployment's optimizer may drive *target*."""
    allowed = allowed_targets()
    if allowed is not None and target not in allowed:
        listed = ", ".join(sorted(allowed)) or "none"
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"This deployment's optimizer does not drive the {target!r} target "
                f"(allowed: {listed})."
            ),
        )
