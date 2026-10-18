"""The service connector's call timeout, as the service and the compose render read it."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from osprey.services.trajectory import constants
from osprey.services.trajectory.constants import connector_call_timeout_s, stop_grace_period_s


def _config(timeout_s: object | None = None) -> dict:
    block: dict = {} if timeout_s is None else {"timeout_s": timeout_s}
    return {
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": block},
        }
    }


def test_a_declared_timeout_is_the_one_returned() -> None:
    assert connector_call_timeout_s(_config(12.5), "va") == 12.5


def test_a_block_without_timeout_gets_the_default() -> None:
    assert connector_call_timeout_s(_config(), "va") == 5.0


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"control_system": "not a mapping"},
        {"control_system": {"type": "virtual_accelerator", "connector": {}}},
    ],
)
def test_a_section_without_the_block_gets_the_default(config: dict) -> None:
    assert connector_call_timeout_s(config, "va") == 5.0


def test_a_target_that_does_not_resolve_gets_the_default() -> None:
    assert connector_call_timeout_s(_config(12.5), "nowhere") == 5.0


def test_live_without_a_derivable_machine_gets_the_default() -> None:
    assert connector_call_timeout_s(_config(12.5), "live") == 5.0


@pytest.mark.parametrize("bad", [0, -1, "soon", True, float("inf")])
def test_an_invalid_timeout_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        connector_call_timeout_s(_config(bad), "va")


def test_the_grace_period_follows_a_non_default_timeout() -> None:
    seconds = connector_call_timeout_s(_config(12.5), "va")
    assert stop_grace_period_s(seconds) > stop_grace_period_s(5.0)


def test_the_module_imports_no_connector_at_load_time() -> None:
    tree = ast.parse(Path(constants.__file__).read_text())
    top_level = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    modules = {
        alias.name if isinstance(node, ast.Import) else node.module or ""
        for node in top_level
        for alias in node.names
    }
    assert not any(name.startswith(("osprey_connectors", "fastapi")) for name in modules)


def test_the_service_has_no_private_copy() -> None:
    from osprey.services.trajectory import app

    assert not hasattr(app, "_call_timeout_s")
