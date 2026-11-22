"""The VA finds engines under the group the engine contract names, and only there.

The composite reads its engine plug-ins through
:data:`osprey_connectors.simulation.engines.ENGINE_GROUP`; a plug-in still
registered under the retired ``osprey.simulation.engines`` group is not found.
"""

from __future__ import annotations

import sys
from importlib import metadata
from types import SimpleNamespace
from typing import Any

import pytest

from osprey_connectors.simulation import composite as composite_module
from osprey_connectors.simulation.engines import ENGINE_GROUP

STUB_MODULE = "osprey_connectors_tests._engine_group_stub"
OLD_GROUP = "osprey.simulation.engines"


def _serve(monkeypatch: pytest.MonkeyPatch, *entries: metadata.EntryPoint) -> None:
    """Make the installed entry points exactly ``entries``."""
    monkeypatch.setitem(sys.modules, STUB_MODULE, SimpleNamespace(build=lambda *a, **k: None))
    found = metadata.EntryPoints(entries)

    def entry_points(**selection: Any) -> metadata.EntryPoints:
        return found.select(**selection) if selection else found

    monkeypatch.setattr(metadata, "entry_points", entry_points)


def test_the_group_is_named_after_the_contract_module() -> None:
    assert ENGINE_GROUP == "osprey_connectors.simulation.engines"


def test_a_stub_under_the_engine_group_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, metadata.EntryPoint(name="stub", value=STUB_MODULE, group=ENGINE_GROUP))

    assert composite_module.Composite._engine("stub") is sys.modules[STUB_MODULE]


def test_a_stub_under_the_old_group_only_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, metadata.EntryPoint(name="stub", value=STUB_MODULE, group=OLD_GROUP))

    with pytest.raises(LookupError, match=ENGINE_GROUP):
        composite_module.Composite._engine("stub")
