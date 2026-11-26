"""The contract number every test manifest declares.

Tests build manifests through :func:`with_contract`, so raising
:data:`~osprey.apps.contract.APP_CONTRACT` edits no test's base dictionary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.apps.contract import APP_CONTRACT

#: The number this OSPREY implements, as the tests spell it.
CONTRACT = APP_CONTRACT


def with_contract(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of *mapping* that declares the current contract first."""
    return {"contract": CONTRACT, **mapping}
