"""The EPICS client environment a process inherits, and taking it out.

A connector-host child configures Channel Access and PVAccess from its target's
config alone, so an ``EPICS_CA_*`` or ``EPICS_PVA_*`` variable it inherited
would silently widen what ``connect()`` decided. Its supervisor launches it
without them (:func:`without_epics_env`) and the child scrubs them again on its
first line (:func:`scrub_epics_env`); any other process that must not reach an
ambient gateway takes them out the same way.

Standard library only, so a process takes it without loading anything else of
the package.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping

__all__ = ["EPICS_ENV_PREFIXES", "scrub_epics_env", "without_epics_env"]

#: Every variable with one of these prefixes configures an EPICS client.
EPICS_ENV_PREFIXES = ("EPICS_CA_", "EPICS_PVA_")


def scrub_epics_env(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Remove every inherited ``EPICS_CA_*``/``EPICS_PVA_*`` variable.

    Args:
        environ: Mapping to scrub, defaulting to :data:`os.environ`.

    Returns:
        The removed name/value pairs, for logging. Nothing is put back: an
        endpoint this child was not configured with must not be reachable from
        it, and a variable that survived the scrub would silently widen what
        ``connect()`` decided.
    """
    env = os.environ if environ is None else environ
    removed = {}
    for name in [name for name in env if name.startswith(EPICS_ENV_PREFIXES)]:
        removed[name] = env.pop(name)
    return removed


def without_epics_env(env: Mapping[str, str]) -> dict[str, str]:
    """A copy of *env* with every ``EPICS_CA_*``/``EPICS_PVA_*`` variable left out."""
    return {name: value for name, value in env.items() if not name.startswith(EPICS_ENV_PREFIXES)}
