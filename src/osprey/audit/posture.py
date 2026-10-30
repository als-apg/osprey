"""The posture as the agent record records it — one reader for every surface.

Three in-process recorders — the MCP audit middleware, the executor's gates and
the protected-set funnel — file records that carry the same three facts about
the process they run in: which posture it is under, how that posture was
established, and which audit session it belongs to. All three read them
from the environment the Web Terminal's spawn sites stamp, and all three spell
them the same way, so the answers live here once and each recorder imports
them. This module is a leaf below all three: the middleware and the gates
already depend on the audit package, and nothing here depends on them.

The posture *value* is the write verdict's, read for the control target this
process's writes are about: :func:`verdict` asks
:func:`osprey_connectors.posture_store.write_verdict` about
:func:`recorded_control_target`, and :func:`posture` spells its answer the way
the record does — ``writes`` when the verdict allows, ``sandbox`` when it
refuses. So the record reads ``sandbox`` whenever a write to that target would
be refused: a read-only run, a tier that allows no writes, an unknown target or
owner, an unreadable record, a switch that is off. The connector package is
imported lazily so that the leaf stays a leaf.

The verdict belongs to the DEPLOYMENT's record rather than to a process tree
inside it, so :data:`POSTURE_SESSION_ENV_VAR` does not index it and does not
gate it: a dispatch worker, an ``agent_runner`` child and a CLI run are answered
from the same record as a Web Terminal session's child. That variable is the
audit session id, which is what the record joins entries on.

:func:`verdict` and :func:`posture` never raise. Every recorder calls
:func:`posture` on every record it files, and an exception would cost the call
rather than answer it. A config that cannot be read is a tier that allows
nothing (``tier_never``), and a verdict that cannot be reached at all is a
refusal; neither is ever a grant. This module decides nothing: the refusals are
the surfaces' own, through the same verdict.

The spellings are the wire contract with the spawn sites
(``interfaces/web_terminal``), which stamp the variables by these names and are
pinned against them by test rather than by import — the interfaces package must
not be dragged behind every MCP server.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from osprey.audit.envelope import POSTURE_SOURCE_PROCESS, POSTURE_SOURCES
from osprey_connectors.posture_env import POSTURE_ENV_VAR, SANDBOX_MODE, is_readonly_run

if TYPE_CHECKING:
    from osprey_connectors.posture_store import WriteVerdict

logger = logging.getLogger("osprey.audit.posture")

__all__ = [
    "CONTROL_TARGET_ENV_VAR",
    "OSPREY_AGENT_DATA_ROOT",
    "POSTURE_ENV_VAR",
    "POSTURE_SANDBOX",
    "POSTURE_SESSION_ENV_VAR",
    "POSTURE_SOURCE_ENV_VAR",
    "POSTURE_WRITES",
    "SANDBOX_MODE",
    "control_system_section",
    "is_readonly_run",
    "posture",
    "posture_session",
    "posture_source",
    "recorded_control_target",
    "verdict",
]

#: :data:`POSTURE_ENV_VAR`, :data:`SANDBOX_MODE` and :func:`is_readonly_run` are
#: :mod:`osprey_connectors.posture_env`'s, re-exported so the recorders keep one
#: import: the session posture's variable, the one value of it that sandboxes,
#: and the first term of the write verdict. A standard-library leaf, so importing
#: it keeps this module a leaf.

#: How the posture in :data:`POSTURE_ENV_VAR` was established, and the audit
#: session id it belongs to. Stamped by the Web Terminal spawn sites and
#: absent everywhere else (a dispatch worker, a CLI run, a container-level
#: execution mode), which is what
#: :data:`~osprey.audit.envelope.POSTURE_SOURCE_PROCESS` is for.
POSTURE_SOURCE_ENV_VAR = "OSPREY_POSTURE_SOURCE"
POSTURE_SESSION_ENV_VAR = "OSPREY_POSTURE_SESSION"

#: The agent-data root the spawning surface resolved, stamped as a PAIR with
#: :data:`POSTURE_SESSION_ENV_VAR` — never one without the other. A session
#: child, the MCP servers below it and the stdlib-only hooks all have to agree
#: on ONE directory for the control-target state file and the control-context
#: record; each deriving it for itself means a deployment that moves
#: ``agent_data.base_dir``, or a hook that can only guess the repo root, reads a
#: different directory from the one the server writes. The spawn sites resolve
#: it once and say so here, so every reader below has an authoritative anchor
#: and only a process with no stamp at all falls back to its own derivation.
#:
#: The name is the value: this is an environment variable's spelling, and the
#: wire contract is the string, not the constant.
OSPREY_AGENT_DATA_ROOT = "OSPREY_AGENT_DATA_ROOT"

#: The control target a RUN was pinned to, stamped into the executor's sandbox
#: subprocess beside its generation and state PID (see
#: :data:`osprey.runtime.ENV_CONTROL_TARGET`, which spells the same string for
#: the reading side of the write pin). Where it is present it is the authority
#: on which target this process's writes are about: the session may have
#: switched since the run started, and the run's writes are still checked
#: against the target it was pinned to. Absent everywhere else — a stamp is
#: never inherited stale, because the executor pops all three variables when it
#: has no target to name.
CONTROL_TARGET_ENV_VAR = "OSPREY_CONTROL_TARGET"

#: How the record spells the two postures — the Web Terminal's vocabulary, not
#: the environment variable's. Records from surfaces that never see
#: ``OSPREY_EXECUTION_MODE`` join on these.
POSTURE_SANDBOX = "sandbox"
POSTURE_WRITES = "writes"


def posture() -> str:
    """This process's posture, spelled the way the record spells it.

    :func:`verdict` mapped to the record's two words: ``writes`` when it allows
    a write to this process's control target, ``sandbox`` when it refuses one,
    for whichever reason. The state at decision time, not the reason for any
    decision — the reason is the refusing surface's to record.

    :data:`POSTURE_SESSION_ENV_VAR` is not consulted: the record is the
    deployment's, so a process that belongs to no session is answered from the
    same record as one that does.

    Never raises: anything that goes wrong on the way is ``sandbox``.
    """
    try:
        allowed = verdict().allowed
    except Exception:  # verdict() never raises by contract; answer it if it does
        logger.debug("Write verdict unavailable; the posture reads sandbox", exc_info=True)
        allowed = False
    return POSTURE_WRITES if allowed else POSTURE_SANDBOX


def control_system_section() -> Any:
    """The deployment's ``control_system`` section, or ``None`` when it cannot be read.

    The one reader the audit package uses for the section a verdict is asked
    about: :func:`verdict` here, and the audit middleware's backstop. ``None``
    is not an empty grant — every verdict reads a section that is not a mapping
    as a tier that allows nothing (``tier_never``).

    Never raises.
    """
    try:
        # Imported inside the function: this module is a leaf.
        from osprey_connectors.config import get_config_value

        return get_config_value("control_system", None)
    except Exception:  # an unreadable config is a tier that allows nothing
        logger.debug("The control_system section could not be read", exc_info=True)
        return None


def verdict() -> WriteVerdict:
    """The write verdict for this process's recorded control target.

    :func:`osprey_connectors.posture_store.write_verdict` for
    :func:`control_system_section` and :func:`recorded_control_target`, with the
    owner left to that function's own ladder. ``None`` for a target that cannot
    be named is passed through, and the verdict answers it as it answers any
    target it cannot resolve.

    Never raises. A config that cannot be read answers ``tier_never``, through
    the verdict itself; a verdict that cannot be reached refuses as
    ``record_unavailable``.

    ``osprey-connectors`` is a declared dependency of this package, so the lazy
    import below is a leaf's import-cost rule, not an optional feature.
    """
    from osprey_connectors.posture_store import WriteVerdict, write_verdict

    try:
        return write_verdict(control_system_section(), recorded_control_target())
    except Exception:  # write_verdict never raises by contract; refuse if it does
        logger.debug("Write verdict could not be reached; refusing", exc_info=True)
        return WriteVerdict(allowed=False, reason="record_unavailable", target=None, kind=None)


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
    """The audit session id this process's records belong to, if it was stamped.

    An identifier the record joins on, and since the narrowing moved into the
    control-context record it is nothing more: no reader indexes a posture by
    it, and a process without one is narrowed exactly like a process with one.
    """
    return (os.environ.get(POSTURE_SESSION_ENV_VAR) or "").strip() or None


# -- the deployment's control target ----------------------------------------


def recorded_control_target() -> str | None:
    """The control target this process's writes are about, or ``None``.

    Public because the refusals are per target now: a gate that names the
    narrowed target in its message has to resolve the same target
    :func:`posture` resolved, and resolving it a second way would let the two
    disagree — the refusal would name a machine the clamp did not fire for.

    Two readers, in this order:

    1. :data:`CONTROL_TARGET_ENV_VAR`, when it names a target readers know.
       Only the executor's sandbox subprocess carries it, and there it is the
       authority: that run is pinned to that target and its writes are refused
       if the session moves off it, so its posture is the posture of the target
       it was pinned to, not of the one the deployment has since switched to.
       The value is checked against
       :data:`~osprey_connectors.types.CONTROL_TARGETS` exactly as the record's
       is — a stamp naming something no reader knows can only index the posture
       to a key nothing writes, so it is dropped in favour of the record rather
       than answered with.
    2. Otherwise the deployment's control-context record, through
       :func:`osprey_connectors.control_context.read_record`. There is one
       record per deployment and one reader for it, so this answer and the
       chip's cannot differ. This is the reader for the MCP servers, which is
       where :func:`posture` is called from on every tool call.

    That reader re-stats the record on every call and re-parses only when its
    signature moves, so the per-tool-call cost is a ``stat``. Nothing rests on
    the freshness of the answer either: the guarantee that a write cannot land
    on a machine nobody selected is the runtime's generation pin, not this
    lookup.

    ``None`` is an honest "not knowable here". :func:`verdict` passes it on, and
    the write verdict refuses a target it cannot resolve; nothing here invents
    one.

    Never raises: an unimportable ``osprey_connectors`` (a deployment where the
    connector package is not installed) and an unreadable record are both "not
    knowable here" — for the stamp too, which is validated against that
    package's vocabulary and so cannot be answered without it.
    """
    stamped = (os.environ.get(CONTROL_TARGET_ENV_VAR) or "").strip()

    try:
        # Imported inside the function: this module is a leaf, and every MCP
        # server imports it while only a session child ever reaches this line.
        from osprey_connectors import control_context
        from osprey_connectors.types import CONTROL_TARGETS

        if stamped:
            if stamped in CONTROL_TARGETS:
                return stamped
            logger.debug(
                "%s names an unknown control target %r; reading the record instead",
                CONTROL_TARGET_ENV_VAR,
                stamped,
            )

        record = control_context.read_record()
    except Exception:  # an unreadable record is "unknown"
        logger.debug(
            "Control-context record unavailable; the control target is unknown", exc_info=True
        )
        return None

    return None if record is None else record.target
