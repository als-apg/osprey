"""Every kind of import an application may make, at module and at function scope."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from osprey.apps import sdk
from osprey.apps.sdk import serve_tools
from osprey_connectors.types import tier_writes

from . import helpers

if TYPE_CHECKING:
    from osprey_connectors.posture_store import WriteVerdict


def describe(section: dict[str, Any], verdict: WriteVerdict) -> str:
    """Report what the fixture reached, so every import above is used."""
    from .helpers import label

    return json.dumps([label(tier_writes(section)), label(verdict), helpers.__name__, sdk.__name__])


def door_name() -> str:
    """Name a public module of a connector subpackage."""
    from osprey_connectors.control_system import write_door

    return write_door.__name__


def main(register: Any) -> None:
    """Hand the server to the SDK: the fixture is read, not run."""
    serve_tools(register)
