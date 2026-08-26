"""The session posture as the ledger records it — one reader for every surface.

Three in-process recorders — the MCP audit middleware, the executor's gates and
the protected-set funnel — file records that carry the same three facts about
the process they run in: which posture it is under, how that posture was
established, and which posture-store key it belongs to. All three read them
from the environment the Web Terminal's spawn sites stamp, and all three spell
them the same way, so the answers live here once and each recorder imports
them. This module is a stdlib-only leaf below all three: the middleware and the
gates already depend on the audit package, and nothing here depends on them.

The spellings are the wire contract with the spawn sites
(``interfaces/web_terminal``), which stamp the variables by these names and are
pinned against them by test rather than by import — the interfaces package must
not be dragged behind every MCP server.
"""

from __future__ import annotations

import os

from osprey.audit.envelope import POSTURE_SOURCE_PROCESS, POSTURE_SOURCES

__all__ = [
    "POSTURE_ENV_VAR",
    "POSTURE_SANDBOX",
    "POSTURE_SESSION_ENV_VAR",
    "POSTURE_SOURCE_ENV_VAR",
    "POSTURE_WRITES",
    "SANDBOX_MODE",
    "posture",
    "posture_session",
    "posture_source",
]

#: The session posture, carried into every child of a Web Terminal session,
#: MCP servers included.
POSTURE_ENV_VAR = "OSPREY_EXECUTION_MODE"

#: How the posture in :data:`POSTURE_ENV_VAR` was established, and the
#: posture-store key it belongs to. Stamped by the Web Terminal spawn sites and
#: absent everywhere else (a dispatch worker, a CLI run, a container-level
#: execution mode), which is what
#: :data:`~osprey.audit.envelope.POSTURE_SOURCE_PROCESS` is for.
POSTURE_SOURCE_ENV_VAR = "OSPREY_POSTURE_SOURCE"
POSTURE_SESSION_ENV_VAR = "OSPREY_POSTURE_SESSION"

#: The one value of :data:`POSTURE_ENV_VAR` that sandboxes a session. A *value*
#: comparison, never a presence check: the writes posture sets the same
#: variable.
SANDBOX_MODE = "readonly"

#: How the ledger spells the two postures — the Web Terminal's vocabulary, not
#: the environment variable's. Records from surfaces that never see
#: ``OSPREY_EXECUTION_MODE`` join on these.
POSTURE_SANDBOX = "sandbox"
POSTURE_WRITES = "writes"


def posture() -> str:
    """This process's posture, spelled the way the ledger spells it.

    The state at decision time, not the reason for any decision: the protected
    set and the static executor layers refuse in the writes posture too.
    """
    return POSTURE_SANDBOX if os.environ.get(POSTURE_ENV_VAR) == SANDBOX_MODE else POSTURE_WRITES


def posture_source(declared: str | None = None) -> str:
    """The provenance of the posture a decision was made under.

    *declared* is the call site's own answer, for a surface that knows it —
    a web request belongs to no session and stamps ``app`` whatever the server
    process inherited. With no answer the environment ladder is read: a session
    child carries the marker its spawn stamped, and anything else is
    ``process``.

    An unrecognised value degrades rather than being carried through, from
    either source: :data:`~osprey.audit.envelope.POSTURE_SOURCES` is closed, a
    record whose provenance is unrecognised reads as authoritative while
    meaning nothing, and the envelope would reject it — which would cost the
    record entirely rather than one field of it.
    """
    stamped = declared if declared is not None else os.environ.get(POSTURE_SOURCE_ENV_VAR)
    return stamped if stamped in POSTURE_SOURCES else POSTURE_SOURCE_PROCESS


def posture_session() -> str | None:
    """The posture-store key this process's posture belongs to, if it was stamped."""
    return (os.environ.get(POSTURE_SESSION_ENV_VAR) or "").strip() or None
