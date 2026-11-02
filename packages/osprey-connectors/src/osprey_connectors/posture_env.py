"""The read-only run, as the environment carries it.

A deployment started read-only, and the python executor's sandbox subprocess
for a run submitted with ``execution_mode="readonly"``, both set
:data:`POSTURE_ENV_VAR` to :data:`SANDBOX_MODE` on every process they own. This
module spells that variable and the one value that sandboxes, and answers the
question every write path asks first: :func:`is_readonly_run`.

It is a leaf: it imports nothing from this package or the framework, so the
control-system connectors, the posture store, the connector host and the
framework's audit and executor layers can all ask it without importing each
other. The stdlib-only PreToolUse hooks cannot import it at all and restate it
in ``osprey_target_state``; a parity test holds that copy equal to this one.

It is not :mod:`osprey_connectors.posture_store`, although that module's
:func:`~osprey_connectors.posture_store.write_verdict` asks it first: the
posture store reads the deployment's record from the agent-data root, while
this answers from the environment alone, so it can be asked before any config
or file is read, and from processes (the connector host, the executor's
sandbox) that must not load the record reader to know they are read-only. An
operator's narrowing of ONE control target is deliberately not here: it never
lived in the environment.
"""

from __future__ import annotations

import os

__all__ = ["POSTURE_ENV_VAR", "SANDBOX_MODE", "is_readonly_run"]

#: The variable that carries the session posture into every child process of a
#: run, MCP servers included.
POSTURE_ENV_VAR = "OSPREY_EXECUTION_MODE"

#: The one value of :data:`POSTURE_ENV_VAR` that sandboxes a process. A *value*
#: comparison, never a presence check: the writes posture (``readwrite``) sets
#: the same variable.
SANDBOX_MODE = "readonly"


def is_readonly_run() -> bool:
    """True when this process must refuse control-system writes, whatever it is.

    Two things set ``OSPREY_EXECUTION_MODE=readonly``, and this answers yes to
    both:

    * the **deployment**, when the whole run is started read-only — the
      variable is on every process the deployment owns, the MCP servers
      included, so a channel write is refused by the connector itself rather
      than only by the (best-effort) hook chain;
    * the **python executor**, onto the sandbox subprocess of a run submitted
      with ``execution_mode="readonly"`` — so the per-run claim is enforceable
      at runtime, independent of the pre-execution pattern scan.

    Both are refusals; only the operator-facing wording differs, and the
    control-system base class decides that by which process is asking, because
    the two are the same variable at the same value.

    This is the first term of
    :func:`osprey_connectors.posture_store.write_verdict` (``readonly_run``).
    Outside these two the variable is unset or ``readwrite``, and the answer is
    still not "writes" by itself: the deployment's ``control_system`` section
    and the control-context record both apply, through the rest of that verdict.

    The comparison is by VALUE, never presence: ``readwrite`` is the writes
    posture, and a stale or misspelled value must not silently sandbox
    anything.
    """
    return os.environ.get(POSTURE_ENV_VAR) == SANDBOX_MODE
