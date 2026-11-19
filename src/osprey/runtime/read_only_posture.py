"""The read-only posture of a process that opens a control session only to read.

A process that serves reads and nothing else (the Channels panel's values child,
the server fleet refresher) takes two steps before its session opens:

* :func:`enter_sandbox_mode` says so in
  :data:`~osprey_connectors.posture_env.POSTURE_ENV_VAR`. The connector then
  stays on the read gateway and refuses every write ``readonly_run``, a switch
  flip is not a move, and the audit records read ``sandbox``.
* :func:`arm_raw_put_block` refuses every raw client put in the process, rpc and
  Tango commands with them, so nothing in it can put around the connector.

Both live here, outside any feature package, so every such process imports the
one posture under the import-linter contracts instead of reaching into another
feature for it. The block cannot be taken off again once armed, so a caller arms
it once, in the process that only reads; a failure to arm is the caller's reason
to refuse to start.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

from osprey_connectors.posture_env import POSTURE_ENV_VAR, SANDBOX_MODE

__all__ = ["arm_raw_put_block", "enter_sandbox_mode"]


def enter_sandbox_mode(environ: MutableMapping[str, str] | None = None) -> None:
    """Mark this process (or *environ*) a read-only run, whatever the operator's switch says."""
    target = os.environ if environ is None else environ
    target[POSTURE_ENV_VAR] = SANDBOX_MODE


def arm_raw_put_block() -> None:
    """Refuse every raw client put in this process, and rpc and Tango commands with them.

    A process in this posture only reads, so unlike the CANOPY backend it has no
    use for them. The executor's contract leaves PVAccess puts to its limits
    check, as a readwrite run's one PVAccess write route; here there is neither
    that check nor any write, so they are refused like the rest.

    The imports stay inside: callers patch :func:`osprey.runtime.raw_put_block.install`
    by attribute, and a process that never arms does not load the executor.
    """
    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import armed_contract
    from osprey.services.python_executor.write_surface import _ARMED_CHECKED

    contract = armed_contract(refuse_rpc=True)
    contract["blocked_targets"] = (
        *contract["blocked_targets"],
        *((dotted, (attr,)) for dotted, attr in _ARMED_CHECKED),
    )
    contract["ca_provider_targets"] = ()
    raw_put_block.install("armed", **contract)
