"""Every framework port a build injects moves with ``deployment.port_base``.

The layout's one rule is that a port comes from the base the deployment
actually resolved, never from the layout's own default. A profile that names a
base and no port keys is the case where that rule is easiest to break and
hardest to see: the loader fills the unspelled keys, the injectors write what
the loader filled into ``config.yml``, and the compose templates' ``|
default(osprey_ports.<slot>)`` never fires because the key is *present*. The
result is a rendered deployment whose config half sits in one block and whose
compose half sits in another, which nothing downstream can detect — both files
are internally consistent.

So this module parses one profile that spells only ``deployment.port_base``,
runs the injectors the build runs, and reads the rendered ``config.yml`` back.
Two claims, deliberately separate: the LOADER fills from the resolved base, and
the INJECTORS carry what it filled all the way to the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_injectors import (
    _inject_bluesky,
    _inject_bluesky_web,
    _inject_dispatch,
    _inject_va,
    _inject_va_archiver,
)
from osprey.cli.build_profile_load import _parse_profile

#: A triggers file naming the bundled worker the way the shipped one does —
#: at the LAYOUT's default base, which is what makes it stale for this module's
#: deployment.
BUNDLED_TRIGGERS = """\
dispatcher:
  # The dispatcher forwards each fired trigger to this worker.
  dispatch_target: http://dispatch-worker-1:10011
  max_concurrent_runs: 2
  max_queue_depth: 50
triggers:
  - name: hello-dispatch
    source: webhook
    action:
      prompt: Say hello.
"""

#: The same file pointed at a target the facility owns. Nothing about it is the
#: framework's to move.
FOREIGN_TRIGGERS = BUNDLED_TRIGGERS.replace(
    "http://dispatch-worker-1:10011", "http://my-worker.example:9000"
)

#: The base this module's profile claims — far from the layout's own 10000, so
#: a number that failed to move is off by a round 10000 and unmistakable.
PORT_BASE = 20000

#: What the layout puts where, at :data:`PORT_BASE`. Spelled as literals rather
#: than as ``default_port`` calls: a test that derives its expectations the same
#: way the code does would pass even if the offsets themselves were wrong.
EXPECTED = {
    "dispatcher": 20010,
    "worker": 20011,
    "tiled": 20070,
    "bluesky_web": 20071,
    "bluesky": 20080,
    "bluesky_second_lane": 20081,
    "va_standin": 20090,
    "mongo": 20801,
}


def _profile(**blocks: Any):
    """Parse a profile whose only port statement is the base itself."""
    raw: dict[str, Any] = {
        "name": "moved",
        "config": {"deployment.port_base": PORT_BASE},
    }
    raw.update(blocks)
    return _parse_profile(raw)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """A project with a config.yml on a VA baseline, plus an empty profile dir.

    The VA baseline is what lets the two-lane bluesky assertion run: the second
    lane is named for the target it serves, so the injector needs a baseline
    that has a second target at all.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    yaml = YAML()
    config = {
        "control_system": {"type": "virtual_accelerator"},
        "deployed_services": [],
        "services": {},
    }
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)
    return project_path, profile_dir


def _services(project_path: Path) -> dict:
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)["services"]


# ── the loader fills from the resolved base ──────────────────────────────────


def test_load_moves_every_unspelled_framework_port_to_the_profiles_base() -> None:
    """No port key is spelled, so every one of them comes from the base."""
    profile = _profile(
        dispatch={"triggers": "t.yml"},
        bluesky={"second_lane": True, "tiled_enabled": True},
        bluesky_web={},
        virtual_accelerator={"live_standin": True},
        va_archiver={},
    )

    assert profile.dispatch is not None
    assert profile.dispatch.dispatcher_port == EXPECTED["dispatcher"]
    assert profile.dispatch.worker_port_base == EXPECTED["worker"]
    assert profile.bluesky is not None
    assert profile.bluesky.port == EXPECTED["bluesky"]
    assert profile.bluesky.tiled_port == EXPECTED["tiled"]
    assert profile.bluesky.second_lane_port(PORT_BASE) == EXPECTED["bluesky_second_lane"]
    assert profile.bluesky_web is not None
    assert profile.bluesky_web.port == EXPECTED["bluesky_web"]
    assert profile.virtual_accelerator is not None
    assert profile.virtual_accelerator.live_standin == EXPECTED["va_standin"]
    assert profile.va_archiver is not None
    assert profile.va_archiver.port_host == EXPECTED["mongo"]


def test_load_leaves_the_channel_access_port_where_the_protocol_wants_it() -> None:
    """The VA's own port is the one the base does not move.

    Instance 1 serves EPICS on 5064 so clients configured for a real facility
    reach it unchanged; only the *stand-in* is in the block.
    """
    profile = _profile(virtual_accelerator={"live_standin": True})

    assert profile.virtual_accelerator is not None
    assert profile.virtual_accelerator.port == 5064


def test_load_still_honours_a_port_the_profile_names_by_hand() -> None:
    """Filling an unspelled key must not overwrite a spelled one."""
    profile = _profile(
        dispatch={"triggers": "t.yml", "dispatcher_port": 31900},
        bluesky={"port": 31090},
        va_archiver={"port_host": 31017},
    )

    assert profile.dispatch is not None
    assert profile.dispatch.dispatcher_port == 31900
    # The key beside it is still unspelled, so it still comes from the base.
    assert profile.dispatch.worker_port_base == EXPECTED["worker"]
    assert profile.bluesky is not None
    assert profile.bluesky.port == 31090
    assert profile.va_archiver is not None
    assert profile.va_archiver.port_host == 31017


# ── the injectors carry it to the rendered config ────────────────────────────


def test_injected_dispatch_config_carries_the_moved_ports(tmp_path: Path) -> None:
    """``services.{event_dispatcher,dispatch_worker}`` land in the profile's block."""
    project_path, profile_dir = _project(tmp_path)
    profile = _profile(dispatch={"triggers": "tutorial_triggers.yml"})
    assert profile.dispatch is not None

    _inject_dispatch(profile.dispatch, profile_dir=profile_dir, project_path=project_path)

    services = _services(project_path)
    assert services["event_dispatcher"]["port"] == EXPECTED["dispatcher"]
    assert services["dispatch_worker"]["worker_port_base"] == EXPECTED["worker"]


def test_injected_bluesky_config_carries_both_lanes_in_the_moved_block(
    tmp_path: Path,
) -> None:
    """Lane 1 is the base's bluesky slot and lane 2 the slot directly above it."""
    project_path, _ = _project(tmp_path)
    profile = _profile(bluesky={"second_lane": True, "tiled_enabled": True})
    assert profile.bluesky is not None

    _inject_bluesky(profile.bluesky, project_path, None, base=PORT_BASE)

    services = _services(project_path)
    assert services["bluesky"]["port"] == EXPECTED["bluesky"]
    assert services["bluesky"]["tiled_port"] == EXPECTED["tiled"]
    # Lane 2 is named for the target it serves; on a VA baseline that is `live`.
    assert services["bluesky_live"]["port"] == EXPECTED["bluesky_second_lane"]


def test_injected_bluesky_web_config_carries_the_moved_port(tmp_path: Path) -> None:
    """The sidecar publishes in the deployment's own block."""
    project_path, _ = _project(tmp_path)
    profile = _profile(bluesky_web={})
    assert profile.bluesky_web is not None

    _inject_bluesky_web(profile.bluesky_web, project_path)

    assert _services(project_path)["bluesky_web"]["port"] == EXPECTED["bluesky_web"]


def test_injected_standin_carries_the_moved_port(tmp_path: Path) -> None:
    """``live_standin: true`` reaches config.yml as the base's stand-in slot."""
    project_path, _ = _project(tmp_path)
    profile = _profile(virtual_accelerator={"live_standin": True})
    assert profile.virtual_accelerator is not None

    _inject_va(profile.virtual_accelerator, project_path)

    services = _services(project_path)
    assert services["virtual_accelerator"]["port"] == 5064
    assert services["live_standin"]["port"] == EXPECTED["va_standin"]


def test_injected_archiver_store_carries_the_moved_host_port(tmp_path: Path) -> None:
    """The store the deployment stands up publishes in its own block."""
    project_path, _ = _project(tmp_path)
    profile = _profile(va_archiver={})
    assert profile.va_archiver is not None

    _inject_va_archiver(profile.va_archiver, project_path)

    assert _services(project_path)["mongodb"]["port_host"] == EXPECTED["mongo"]


# ── the refusal that keeps a bad base from being rendered at the default ─────


@pytest.mark.parametrize("bad", [1000, 65000])
def test_a_base_whose_block_cannot_exist_is_refused_at_load(bad: int) -> None:
    """Refused, never quietly rendered at the layout's own base instead."""
    with pytest.raises(ValueError, match=f"deployment.port_base is {bad}"):
        _parse_profile(
            {
                "name": "impossible",
                "config": {"deployment.port_base": bad},
                "dispatch": {"triggers": "t.yml"},
            }
        )


# ── the copied triggers file routes into the moved block ─────────────────────


def _write_triggers(profile_dir: Path, body: str) -> None:
    """Put a triggers file in the profile dir, so injection resolves it there."""
    (profile_dir / "triggers.yml").write_text(body, encoding="utf-8")


def _dispatcher_block(project_path: Path) -> dict:
    yaml = YAML()
    with open(project_path / "triggers.yml") as fh:
        return yaml.load(fh)["dispatcher"]


def test_copied_triggers_route_to_the_worker_port_this_deployment_binds(
    tmp_path: Path,
) -> None:
    """The shipped target names the default base; the build re-bases it.

    Left alone, the dispatcher would forward every fired trigger to 10011 while
    the worker container listens on 20011 — a closed port, and a failure that
    shows up only at the first dispatch.
    """
    project_path, profile_dir = _project(tmp_path)
    _write_triggers(profile_dir, BUNDLED_TRIGGERS)
    profile = _profile(dispatch={"triggers": "triggers.yml"})
    assert profile.dispatch is not None

    _inject_dispatch(profile.dispatch, profile_dir=profile_dir, project_path=project_path)

    assert _dispatcher_block(project_path)["dispatch_target"] == (
        f"http://dispatch-worker-1:{EXPECTED['worker']}"
    )


def test_copied_triggers_leave_a_facilitys_own_target_alone(tmp_path: Path) -> None:
    """A target the framework does not render is the facility's routing decision."""
    project_path, profile_dir = _project(tmp_path)
    _write_triggers(profile_dir, FOREIGN_TRIGGERS)
    profile = _profile(dispatch={"triggers": "triggers.yml"})
    assert profile.dispatch is not None

    _inject_dispatch(profile.dispatch, profile_dir=profile_dir, project_path=project_path)

    dispatcher = _dispatcher_block(project_path)
    assert dispatcher["dispatch_target"] == "http://my-worker.example:9000"
    # The pool-limit half of the same patch still ran.
    assert dispatcher["max_concurrent_runs"] == 2


def test_host_mode_still_replaces_the_whole_target_with_the_host_address(
    tmp_path: Path,
) -> None:
    """The compose DNS name does not resolve when the pair shares the host's namespace."""
    project_path, profile_dir = _project(tmp_path)
    _write_triggers(profile_dir, BUNDLED_TRIGGERS)
    profile = _profile(dispatch={"triggers": "triggers.yml", "network": "host"})
    assert profile.dispatch is not None

    _inject_dispatch(profile.dispatch, profile_dir=profile_dir, project_path=project_path)

    assert _dispatcher_block(project_path)["dispatch_target"] == (
        f"http://localhost:{EXPECTED['worker']}"
    )
