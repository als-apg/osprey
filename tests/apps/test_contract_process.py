"""Every ``OSPREY_*`` variable handed to an application is in ``APP_ENVIRONMENT``.

An application's processes receive two environments: its container's, rendered
by the bundled ``app_runtime`` compose template, and its tool server's, written
by the projection. ``APP_ENVIRONMENT`` is the promised list of the ``OSPREY_*``
names in them: exact names, and two patterns, one per roster secret and one per
declared corpus's sidecar URL. A variable added to the template only fails here until the tuple names it, and a
tuple entry neither environment carries fails too, except the one name a tool
server inherits from the agent's session.
"""

from __future__ import annotations

import copy
from fnmatch import fnmatchcase
from typing import Any

import pytest

from osprey.apps.manifest import parse_manifest
from osprey.apps.projection import APP_ENVIRONMENT, _project_server
from osprey.runtime.session import SESSION_KEY_ENV_VAR
from tests.deployment.test_app_runtime_compose import QMD, _render, _roster_credential, _row

from ._contract import with_contract

#: The prefix the promised names share.
PREFIX = "OSPREY_"

#: The inherited name: the agent's session sets it, no projection writes it.
INHERITED = "OSPREY_SESSION_ID"

TOOLED: dict[str, Any] = with_contract(
    {
        "name": "knob",
        "containers": {"knob": {"runtime": "osprey", "health": "/health"}},
        "panel": {"label": "Knob", "container": "knob"},
        "tools": {"module": "knob_app.tools", "shown_when": "panel", "read": ["knob_read"]},
    }
)


def _every_branch_row() -> dict[str, Any]:
    """A container row turning on every conditional the template reads."""
    return _row(
        network="bridge",
        state="var/knob",
        launch_token_env="CORRECTOR_TRIM_LAUNCH_TOKEN",
        roster_grant={"panel_id": "corrector-trim"},
        drives_target=True,
        control_identity=True,
        corpora=["procedures"],
    )


def _container_environment(row: dict[str, Any]) -> dict[str, Any]:
    grant = {
        "credentials": [_roster_credential("ALICE", "alice"), _roster_credential("BOB", "bob")]
    }
    _, svc = _render(row, panel_grant=grant, qmd=QMD)
    return svc["environment"]


def _server_environment() -> dict[str, Any]:
    manifest = parse_manifest(copy.deepcopy(TOOLED), app=TOOLED["name"])
    assert manifest.tools is not None
    return _project_server(manifest, manifest.tools)["env"]


def _exact() -> set[str]:
    return {name for name in APP_ENVIRONMENT if "*" not in name}


def _patterns() -> list[str]:
    return sorted(name for name in APP_ENVIRONMENT if "*" in name)


def _promised(name: str) -> bool:
    return name in _exact() or any(fnmatchcase(name, pattern) for pattern in _patterns())


def _unpromised(environment: dict[str, Any], declared: set[str]) -> list[str]:
    return sorted(
        name
        for name in environment
        if name.startswith(PREFIX) and name not in declared and not _promised(name)
    )


@pytest.fixture(scope="module")
def container() -> dict[str, Any]:
    return _container_environment(_every_branch_row())


def test_the_tuple_holds_only_osprey_names_and_two_patterns():
    assert all(name.startswith(PREFIX) for name in APP_ENVIRONMENT)
    assert len(set(APP_ENVIRONMENT)) == len(APP_ENVIRONMENT)
    assert _patterns() == ["OSPREY_QMD_*_URL", "OSPREY_TERMINAL_SECRET_*"]


def test_the_every_branch_render_turns_on_each_branch(container):
    # Each guarded block of the template contributes at least one name.
    for name in (
        "OSPREY_APP_NAME",
        "OSPREY_APP_CONFIG",
        "OSPREY_APP_CORPORA",
        "OSPREY_APP_PORT",
        "OSPREY_APP_STATE_DIRS",
        "OSPREY_AUDIT_IDENTITY",
        "OSPREY_GUARDED_RUN_DIR",
        "OSPREY_QMD_PROCEDURES_URL",
        "OSPREY_TERMINAL_ROSTER_OWNERS",
        "OSPREY_TERMINAL_SECRET_ALICE",
        "OSPREY_TERMINAL_SECRET_BOB",
        "CORRECTOR_TRIM_LAUNCH_TOKEN",
    ):
        assert name in container, name


def test_every_osprey_variable_of_the_container_is_promised(container):
    declared = set(_every_branch_row()["env"])
    assert not _unpromised(container, declared)


def test_an_app_declared_osprey_name_is_the_apps_own():
    row = _every_branch_row()
    row["env"] = [*row["env"], "OSPREY_APP_OWN_KNOB"]
    environment = _container_environment(row)
    assert "OSPREY_APP_OWN_KNOB" in environment
    assert not _unpromised(environment, set(row["env"]))


def test_a_variable_added_to_the_template_only_is_refused(container):
    # The guard the tuple exists for: the template grows, the tuple does not.
    grown = {**container, "OSPREY_TEMPLATE_ONLY": "1"}
    declared = set(_every_branch_row()["env"])
    assert _unpromised(grown, declared) == ["OSPREY_TEMPLATE_ONLY"]


def test_every_osprey_variable_of_the_tool_server_is_promised():
    server = _server_environment()
    assert [name for name in server if name.startswith(PREFIX)]
    assert not _unpromised(server, set())


def test_every_exact_name_is_handed_over(container):
    handed = set(container) | set(_server_environment())
    missing = sorted(_exact() - handed - {INHERITED})
    assert not missing, missing


def test_the_inherited_name_is_the_sessions_variable():
    assert INHERITED in APP_ENVIRONMENT
    assert INHERITED == SESSION_KEY_ENV_VAR


def test_the_launch_token_variable_is_not_promised(container):
    assert "CORRECTOR_TRIM_LAUNCH_TOKEN" in container
    assert not any(name.endswith("_LAUNCH_TOKEN") for name in APP_ENVIRONMENT)
    assert "OSPREY_WORKER_IMAGE" not in APP_ENVIRONMENT
