"""An application container's declared ``state`` directory, from build to mount.

A projected ``services.<k>`` entry (``osprey.apps.projection``) carries the
owning application's ``app``, its ``audit_identity`` and, when the container
declares one, ``state``: a path relative to the container's project. The build
provisions the host side inside the identity's state zone, and the render
context carries the per-service fact ``services.<k>.state_mounts`` — the
``[{source, target}]`` list the application runtime template binds and names in
``OSPREY_APP_STATE_DIRS``. Nothing here mounts a zone root, and nothing spells
the zone other than through :mod:`osprey.utils.workspace`.

The projection that feeds these entries into a build config lands separately,
so these tests hand the config the projected shape directly.
"""

from __future__ import annotations

import copy
import stat
from pathlib import Path, PurePosixPath

import pytest

from osprey.deployment.compose_generator import (
    _CONTAINER_APP_ROOT,
    _ensure_agent_data_structure,
    _inject_project_metadata,
    app_state_dir,
    app_state_mounts,
    ensure_app_state_dirs,
)
from osprey.utils.workspace import STATE_DIR_NAME, STATE_DIR_RELPATH

PROJECT = "demo"
CONTAINER_PROJECT = PurePosixPath(_CONTAINER_APP_ROOT) / PROJECT


def _projected(**overrides) -> dict:
    """The block ``osprey.apps.projection`` writes for a runtime container."""
    block = {
        "app": "knob-demo",
        "port": 18100,
        "state": f"{STATE_DIR_NAME}/knob-demo",
        "drives_target": True,
        "control_identity": True,
        "audit_identity": "knob-demo",
        "config_file": "./services/knob-demo/app-config.yml",
    }
    block.update(overrides)
    return block


def _config(tmp_path: Path, services: dict) -> dict:
    return {
        "project_name": PROJECT,
        "project_root": str(tmp_path),
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": list(services),
        "services": copy.deepcopy(services),
    }


def _assert_group_shared(path: Path) -> None:
    assert path.is_dir(), path
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & stat.S_ISGID, oct(mode)
    assert mode & stat.S_IWGRP, oct(mode)


# ── where the directory lives ────────────────────────────────────────────────


def test_the_host_directory_is_inside_the_identity_state_zone(tmp_path):
    path = app_state_dir(tmp_path, "knob-demo", "knob-demo")

    assert path == tmp_path / STATE_DIR_RELPATH / "knob-demo" / "knob-demo"


@pytest.mark.parametrize("identity", ["../x", "Knob", "", "a/b"])
def test_an_identity_outside_the_charset_is_refused(tmp_path, identity):
    with pytest.raises(ValueError, match="identity"):
        app_state_dir(tmp_path, identity, "knob-demo")


@pytest.mark.parametrize("key", ["..", ".", "a/b", "", "a:b"])
def test_a_service_key_that_is_not_one_path_segment_is_refused(tmp_path, key):
    with pytest.raises(ValueError, match="service key"):
        app_state_dir(tmp_path, "knob-demo", key)


# ── the per-service fact ─────────────────────────────────────────────────────


def test_a_declared_state_becomes_one_mount_at_its_declared_path(tmp_path):
    config = _config(tmp_path, {"knob-demo": _projected()})

    mounts = app_state_mounts(config, CONTAINER_PROJECT)

    assert mounts == {
        "knob-demo": [
            {
                "source": f"./{STATE_DIR_RELPATH}/knob-demo/knob-demo",
                "target": f"{CONTAINER_PROJECT}/{STATE_DIR_NAME}/knob-demo",
            }
        ]
    }


def test_each_container_of_one_app_gets_its_own_directory(tmp_path):
    config = _config(
        tmp_path,
        {
            "knob-demo": _projected(),
            "knob-worker": _projected(state=f"{STATE_DIR_NAME}/worker"),
        },
    )

    mounts = app_state_mounts(config, CONTAINER_PROJECT)

    sources = {key: entry[0]["source"] for key, entry in mounts.items()}
    assert sources == {
        "knob-demo": f"./{STATE_DIR_RELPATH}/knob-demo/knob-demo",
        "knob-worker": f"./{STATE_DIR_RELPATH}/knob-demo/knob-worker",
    }


def test_only_projected_services_that_declare_state_get_a_mount(tmp_path):
    stateless = _projected()
    del stateless["state"]
    config = _config(
        tmp_path,
        {
            "knob-demo": _projected(),
            "knob-cache": stateless,
            # A framework service with a ``state`` key is not an application's.
            "bluesky": {"state": "var/whatever"},
            "dispatch_worker": None,
        },
    )

    assert set(app_state_mounts(config, CONTAINER_PROJECT)) == {"knob-demo"}


def test_no_mount_binds_a_zone_root(tmp_path):
    config = _config(tmp_path, {"knob-demo": _projected()})

    (entry,) = app_state_mounts(config, CONTAINER_PROJECT)["knob-demo"]

    source = entry["source"].removeprefix("./")
    assert source not in (STATE_DIR_RELPATH, f"{STATE_DIR_RELPATH}/knob-demo")
    assert source.startswith(f"{STATE_DIR_RELPATH}/knob-demo/")


@pytest.mark.parametrize("declared", ["/etc", "../outside", "var/../../x", "var/a:b", "."])
def test_a_declared_path_outside_the_container_project_is_refused(tmp_path, declared):
    config = _config(tmp_path, {"knob-demo": _projected(state=declared)})

    with pytest.raises(ValueError, match="state"):
        app_state_mounts(config, CONTAINER_PROJECT)


def test_the_render_context_carries_the_fact_on_the_service_block(tmp_path):
    config = _config(tmp_path, {"knob-demo": _projected()})

    context = _inject_project_metadata(config)

    assert context["services"]["knob-demo"]["state_mounts"] == [
        {
            "source": f"./{STATE_DIR_RELPATH}/knob-demo/knob-demo",
            "target": f"{_CONTAINER_APP_ROOT}/{PROJECT}/{STATE_DIR_NAME}/knob-demo",
        }
    ]
    assert "state_mounts" not in config["services"]["knob-demo"], "the input was mutated"


def test_a_render_without_app_services_adds_no_fact(tmp_path):
    config = _config(tmp_path, {"dispatch_worker": {"worker_count": 1}})

    context = _inject_project_metadata(config)

    assert "state_mounts" not in context["services"]["dispatch_worker"]


def test_an_undeployed_app_service_gets_no_mount_fact_or_directory(tmp_path):
    config = _config(tmp_path, {"knob-demo": _projected()})
    config["deployed_services"] = []

    assert app_state_mounts(config, CONTAINER_PROJECT) == {}
    assert ensure_app_state_dirs(config, tmp_path) == []
    assert not (tmp_path / STATE_DIR_RELPATH / "knob-demo" / "knob-demo").exists()
    context = _inject_project_metadata(config)
    assert "state_mounts" not in context["services"]["knob-demo"]


# ── provisioning ─────────────────────────────────────────────────────────────


def test_provisioning_creates_each_directory_group_shared(tmp_path):
    config = _config(
        tmp_path,
        {
            "knob-demo": _projected(),
            "knob-worker": _projected(state=f"{STATE_DIR_NAME}/worker"),
        },
    )

    provisioned = ensure_app_state_dirs(config, tmp_path)

    expected = [
        tmp_path / STATE_DIR_RELPATH / "knob-demo" / "knob-demo",
        tmp_path / STATE_DIR_RELPATH / "knob-demo" / "knob-worker",
    ]
    assert provisioned == expected
    _assert_group_shared(tmp_path / STATE_DIR_RELPATH / "knob-demo")
    for path in expected:
        _assert_group_shared(path)


def test_provisioning_skips_a_bad_entry_without_failing(tmp_path, caplog):
    config = _config(
        tmp_path,
        {
            "knob-demo": _projected(),
            "bad": _projected(audit_identity="../escape"),
        },
    )

    provisioned = ensure_app_state_dirs(config, tmp_path)

    assert provisioned == [tmp_path / STATE_DIR_RELPATH / "knob-demo" / "knob-demo"]
    assert not (tmp_path.parent / "escape").exists()
    assert "Skipping" in caplog.text


def test_the_build_path_provisions_declared_state(tmp_path):
    config = _config(tmp_path, {"knob-demo": _projected()})

    _ensure_agent_data_structure(config)

    _assert_group_shared(tmp_path / STATE_DIR_RELPATH / "knob-demo" / "knob-demo")
