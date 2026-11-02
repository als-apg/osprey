"""The ``api.jev`` rows of the config-key manifest agree with ``JevSettings``.

``osprey config --defaults`` renders the manifest's ``default:`` column verbatim,
so a literal there that drifts from the settings model would tell an operator
the wrong thing about an omitted key. These tests read both sides and compare.

The block is never rendered by a template or spelled by a preset: a deployment
that wants a non-default route writes it into its own config. The last test
holds that line, because every row here is marked ``rendered: false``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from osprey.models.jev.config import JevSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_PATH = REPO_ROOT / "scripts" / "check_config_keys.py"
BLOCK = "api.jev"


def _load_guard_module() -> Any:
    name = "check_config_keys"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, GUARD_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # import-time required because scripts/ is not a package: check_config_keys.py
    # is loaded by path and registered in sys.modules before exec so @dataclass can
    # resolve annotations through cls.__module__; the manifest it loads feeds the
    # module-level key tables the parametrize decorators below read at import.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard_module = _load_guard_module()
MANIFEST = guard_module.load_manifest(guard_module.DEFAULT_MANIFEST)
SENTINELS = guard_module.DEFAULT_SENTINELS


def _jev_rows() -> dict[str, dict[str, Any]]:
    return {
        key: spec
        for key, spec in MANIFEST["keys"].items()
        if key == BLOCK or key.startswith(BLOCK + ".")
    }


def _settings_leaves(node: Any, prefix: str = BLOCK) -> dict[str, Any]:
    """Dotted leaf path -> default value, walked from the settings dump."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(_settings_leaves(value, f"{prefix}.{key}"))
        return out
    return {prefix: node}


SETTINGS_LEAVES = _settings_leaves(JevSettings().model_dump())
ROWS = _jev_rows()
LEAF_ROWS = {key: spec for key, spec in ROWS.items() if key != BLOCK}


def test_block_row_is_anchored_on_the_settings_reader():
    block = ROWS[BLOCK]
    assert block.get("rendered") is False
    evidence = block.get("evidence")
    assert evidence, "the api.jev block must carry anchored evidence"
    source = (REPO_ROOT / "src/osprey/models/jev/config.py").read_text()
    assert re.search(evidence, source), "evidence must match the JevSettings reader"


def test_every_settings_leaf_has_a_row_and_no_row_is_invented():
    assert set(LEAF_ROWS) == set(SETTINGS_LEAVES)


@pytest.mark.parametrize("key", sorted(SETTINGS_LEAVES))
def test_leaf_row_is_covered_by_the_block_and_not_rendered(key):
    spec = LEAF_ROWS[key]
    assert spec.get("covered-by") == BLOCK
    assert spec.get("rendered") is False
    assert "evidence" not in spec


@pytest.mark.parametrize("key", sorted(SETTINGS_LEAVES))
def test_literal_default_equals_the_settings_default(key):
    spec = LEAF_ROWS[key]
    default = spec["default"]
    if isinstance(default, str) and default in SENTINELS:
        pytest.skip(f"{key} records the sentinel {default!r}")
    assert default == SETTINGS_LEAVES[key]
    assert type(default) is type(SETTINGS_LEAVES[key]) or (
        isinstance(default, int | float) and isinstance(SETTINGS_LEAVES[key], int | float)
    )
    assert "default_note" not in spec, "a literal default carries no note"


def test_api_key_is_derived_and_names_the_route_variable():
    spec = LEAF_ROWS[f"{BLOCK}.routes.typesafe.api_key"]
    assert spec["default"] == "derived"
    assert "TYPESAFE_API_KEY" in spec["default_note"]


def test_no_rendered_source_carries_an_api_jev_leaf():
    guard = guard_module.ConfigKeyGuard(REPO_ROOT, MANIFEST)
    carried = {
        path: sorted(sources)
        for path, sources in guard.union().items()
        if path == BLOCK or path.startswith(BLOCK + ".")
    }
    assert carried == {}
