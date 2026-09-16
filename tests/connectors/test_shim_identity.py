"""Module-identity guarantees for the osprey-connectors extraction.

Main osprey re-exports the moved modules under their historical paths via
sys.modules-aliasing shims; these tests pin the contract that both names
resolve to the SAME module object (patching/isinstance safe).

A shim is read two ways: at runtime it is the module it substitutes, and to a
type checker it is the file as written. So it re-exports its target's public
names in the file itself, and these tests pin that the two readings agree.
"""

from __future__ import annotations

import ast
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"

#: `sys` is the shim's own machinery, and it is subtracted from both sides of
#: the parity comparison because a target module that imports `sys` itself would
#: otherwise make the comparison depend on which side happened to name it.
_SHIM_OWN_NAMES = frozenset({"sys"})

pytestmark = pytest.mark.skipif(
    not _SRC.is_dir(), reason="shim discovery needs the source tree; none in an installed wheel"
)


@dataclass(frozen=True)
class _Shim:
    """One compatibility shim, as the file says it forwards."""

    path: Path
    dotted: str
    target: str
    star_target: str | None


def _discover_shims() -> list[_Shim]:
    """Every shim under ``src/osprey``, with the targets its imports name."""
    if not _SRC.is_dir():
        return []

    shims: list[_Shim] = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "sys.modules[__name__] = _mod" not in text:
            continue

        target: str | None = None
        star_target: str | None = None
        for node in ast.parse(text).body:
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            for alias in node.names:
                if alias.asname == "_mod":
                    target = f"{node.module}.{alias.name}"
                elif alias.name == "*":
                    star_target = node.module
        assert target is not None, f"{path} substitutes a module it never imports as `_mod`"

        dotted = ".".join(("osprey", *path.relative_to(_SRC).with_suffix("").parts))
        shims.append(_Shim(path=path, dotted=dotted, target=target, star_target=star_target))
    return shims


_SHIMS = _discover_shims()


def test_the_shim_discovery_finds_the_shim_modules():
    assert _SHIMS, f"no shim modules found under {_SRC}"


@pytest.mark.parametrize("shim", _SHIMS, ids=lambda shim: shim.dotted)
def test_a_shim_stars_from_the_module_it_substitutes(shim: _Shim):
    # Typing against one module while substituting another is the one way this
    # file can lie about what it forwards, and neither half fails on its own.
    assert shim.star_target == shim.target


@pytest.mark.parametrize("shim", _SHIMS, ids=lambda shim: shim.dotted)
def test_a_shim_exposes_its_targets_public_names(shim: _Shim):
    # A throwaway probe name, because executing the file writes
    # sys.modules[__name__]; the `finally` keeps that write out of the session.
    namespace: dict[str, object] = {"__name__": f"_shim_parity_probe_{shim.dotted}"}
    try:
        exec(compile(shim.path.read_text(encoding="utf-8"), str(shim.path), "exec"), namespace)
    finally:
        sys.modules.pop(str(namespace["__name__"]), None)

    module = importlib.import_module(shim.target)
    target_public = set(
        getattr(module, "__all__", None)
        or [name for name in vars(module) if not name.startswith("_")]
    )
    exposed = {name for name in namespace if not name.startswith("_")}
    assert exposed - _SHIM_OWN_NAMES == target_public - _SHIM_OWN_NAMES


def test_osprey_connectors_is_installed():
    import osprey_connectors

    # The package versions with the framework's calendar stream (one checkout,
    # one number for both wheels), so the pin is the stream itself: a year-led
    # version. The retired independent 0.x line — and the "0.0.0+unbuilt"
    # fallback of a source tree with no installed dist — both fail here.
    major = osprey_connectors.__version__.split(".")[0]
    assert major.isdigit() and int(major) >= 2026, osprey_connectors.__version__


def test_errors_shim_preserves_module_identity():
    import osprey.errors
    import osprey_connectors.errors

    assert osprey.errors is osprey_connectors.errors


def test_utils_shims_preserve_module_identity():
    import osprey.utils.config
    import osprey.utils.dotenv
    import osprey.utils.logger
    import osprey.utils.relative_time
    import osprey_connectors.config
    import osprey_connectors.dotenv
    import osprey_connectors.logger
    import osprey_connectors.relative_time

    assert osprey.utils.config is osprey_connectors.config
    assert osprey.utils.dotenv is osprey_connectors.dotenv
    assert osprey.utils.logger is osprey_connectors.logger
    assert osprey.utils.relative_time is osprey_connectors.relative_time


def test_patching_through_shim_reaches_real_module(monkeypatch):
    import osprey_connectors.config as real_config

    monkeypatch.setattr("osprey.utils.config.get_config_value", lambda *a, **k: "patched")
    assert real_config.get_config_value("anything") == "patched"


def test_simulation_core_shims_preserve_module_identity():
    import osprey.simulation
    import osprey.simulation.engine
    import osprey_connectors.simulation.engine

    assert osprey.simulation.engine is osprey_connectors.simulation.engine
    assert osprey.simulation.SimulationEngine is osprey_connectors.simulation.SimulationEngine


def test_connector_shims_preserve_module_identity():
    import osprey.connectors.archiver.base
    import osprey.connectors.control_system.epics_connector
    import osprey.connectors.control_system.tango_connector
    import osprey.connectors.factory
    import osprey_connectors.archiver.base
    import osprey_connectors.control_system.epics_connector
    import osprey_connectors.control_system.tango_connector
    import osprey_connectors.factory

    assert (
        osprey.connectors.control_system.epics_connector
        is osprey_connectors.control_system.epics_connector
    )
    assert (
        osprey.connectors.control_system.tango_connector
        is osprey_connectors.control_system.tango_connector
    )
    assert osprey.connectors.archiver.base is osprey_connectors.archiver.base
    assert osprey.connectors.factory is osprey_connectors.factory


def test_exception_identity_across_namespaces():
    from osprey.errors import ChannelWriteBlockedError as old
    from osprey_connectors.errors import ChannelWriteBlockedError as new

    assert old is new
