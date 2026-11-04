"""Who a run is for: the owner login that rides the start request into the worker.

On a deployment whose optimizer container holds the control-context tree, the
connector governs a write by the switch of the login the write is for
(:func:`osprey_connectors.posture_store.current_owner`). The container's own
account names nobody, so with no owner stamped every write is refused
``owner_unknown``. A run therefore carries its owner the way a dispatch job
does: the worker process exports it as ``OSPREY_CONTROL_OWNER`` before anything
opens a connector, and every connector-host child it spawns inherits it.

**Where the owner comes from.** The launch token is a deployment secret, not a
principal: it says the caller may start a run, not for whom. The person is
named by the ``X-Osprey-Owner`` request header, the one the deployment already
mints for the Bluesky bridge and the event dispatcher (``osprey.utils.owner_header``):
the terminal proxy and the auth sidecar strip any inbound value and set their
own, and an MCP client in a terminal container stamps
``current_owner()``. It is read on ``start`` only, next to the launch token, so
the caller trusted to start a run is the caller trusted to say whose it is.

**The accepted shape** is restated from ``osprey.utils.owner_header``, which
this package cannot import (it never imports the framework): the roster
username charset ``[A-Za-z0-9._-]``, at most :data:`MAX_OWNER_LENGTH`
characters, no path fragment, no unexpanded ``${...}``. A parity test holds the
two to the same answers. A refused value is owner-less, never an error: the
write verdict then refuses the run's writes, which is the fail-closed answer.
"""

from __future__ import annotations

import logging
import os
import re

from osprey_connectors.posture_store import CONTROL_OWNER_ENV_VAR

logger = logging.getLogger(__name__)

#: The header's wire spelling, as ``osprey.utils.owner_header.OWNER_HEADER``.
OWNER_HEADER = "X-Osprey-Owner"

#: The longest accepted name, as ``osprey.utils.owner_header.MAX_OWNER_LENGTH``.
MAX_OWNER_LENGTH = 64

_OWNER_RE = re.compile(rf"\A[A-Za-z0-9._-]{{1,{MAX_OWNER_LENGTH}}}\Z")
_PATH_SEPARATORS = ("/", "\\", "\0")
_RESERVED_NAMES = (".", "..")
_PLACEHOLDER_MARKER = "${"


def _refuse(shape: str) -> None:
    logger.warning(
        "Ignoring %s: the value is %s, so this run is started owner-less and its writes "
        "are refused unless this host's account is the owner. The value itself is not logged.",
        OWNER_HEADER,
        shape,
    )


def owner_from_header(value: str | None) -> str | None:
    """The owner *value* names, or ``None`` when it names nobody. Never raises."""
    if value is None:
        return None
    if not isinstance(value, str):
        _refuse("not a string")
        return None
    if not value.strip():
        _refuse("empty or whitespace-only")
        return None
    if _PLACEHOLDER_MARKER in value:
        _refuse("an unexpanded ${...} placeholder rather than a name")
        return None
    if len(value) > MAX_OWNER_LENGTH:
        _refuse(f"longer than the {MAX_OWNER_LENGTH}-character limit")
        return None
    if value in _RESERVED_NAMES or any(sep in value for sep in _PATH_SEPARATORS):
        _refuse("a path fragment rather than a single name")
        return None
    if not _OWNER_RE.match(value):
        _refuse("outside the accepted charset [A-Za-z0-9._-]")
        return None
    return value


def stamp_owner(owner: str | None) -> None:
    """Make *owner* the owner of every write this process makes from here on.

    Called first thing in a job worker, before any connector is opened, so the
    connector-host children it spawns inherit the stamp and the verdict this
    process takes over them names the same owner. ``None`` clears any stamp the
    worker inherited: a run started for nobody must not borrow the owner of
    whatever process spawned it.
    """
    if owner:
        os.environ[CONTROL_OWNER_ENV_VAR] = owner
    else:
        os.environ.pop(CONTROL_OWNER_ENV_VAR, None)
