"""Network-axis behavior of the ``_inject_dispatch`` build step.

The dispatcher and its workers share ONE network knob (``dispatch.network``):
they talk to each other over addresses the build emits, so a half on the compose
bridge and a half on the host network could not reach each other. This module
covers what injection does with that single value:

* bridge (the default, set or omitted) — every artifact byte-for-byte what it
  was before the axis existed, pinned against a render captured from the
  pre-axis injector;
* host — the mode written into BOTH service blocks, the triggers file's
  ``dispatch_target`` rewritten off the compose DNS name onto ``localhost``, and
  the per-worker port rule recorded in the worker's service config for the
  compose render and the host-port preflight to derive from.

The bundled-triggers/template-copy/``deployed_services`` responsibilities of the
same step are covered in ``test_inject_dispatch.py``.
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_dispatch
from osprey.cli.build_profile import DispatchConfig

# The exact config.yml this injection renders for the profile built by
# ``_dispatch()`` below, captured from the injector as it stood before the
# network axis existed. Any byte of drift in the default (bridge) render is a
# regression, not a refresh: the axis is opt-in and unset means unchanged.
PRE_AXIS_CONFIG_YML = """\
deployed_services:
- postgresql
- event_dispatcher
- dispatch_worker
services:
  postgresql: {}
  event_dispatcher:
    path: ./services/event_dispatcher
    port: 10010
    facility_name: ALS
    pv_strip_prefix: 'ALS:'
    additional_dirs:
    - src: triggers.yml
      dst: triggers.yml
  dispatch_worker:
    path: ./services/dispatch_worker
    worker_count: 1
    worker_port_base: 10011
    workspace_mode: isolated
    timeout_sec: 300
    inactivity_sec: 120
web:
  panels:
    events:
      url: http://localhost:10010
      path: /dashboard
      label: EVENTS
      health_endpoint: /health
"""

# The dispatcher block of the copied triggers file, likewise captured from the
# pre-axis injector (the surrounding file is re-indented by the round-trip that
# patches the pool limits — that reshaping predates this axis and is not what
# these tests pin). Host mode changes the ``dispatch_target`` line and nothing
# else about this region.
PRE_AXIS_TRIGGERS_DISPATCHER = """\
dispatcher:
  # The dispatcher forwards each fired trigger to this worker. The compose
  # template names the single worker "dispatch-worker-1", one port above the
  # dispatcher itself — `deployment.port_base` + 11, so 10011 at the default
  # base. Moving the block moves both. Under `dispatch.network: host` the build
  # rewrites this line to the worker's host address instead.
  # (Multi-worker load distribution is not yet implemented — see docs.)
  dispatch_target: http://dispatch-worker-1:10011
  max_concurrent_runs: 2
  max_queue_depth: 50
"""


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """Create a project with a minimal config.yml, plus an empty profile dir.

    The empty profile directory forces the bundled-triggers resolution path, so
    the triggers assertions run against the shipped ``tutorial_triggers.yml``.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    yaml = YAML()
    config = {"deployed_services": ["postgresql"], "services": {"postgresql": {}}}
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)
    return project_path, profile_dir


def _dispatch(**overrides: object) -> DispatchConfig:
    """Build the DispatchConfig the pinned render above was captured from."""
    base: dict = {
        "triggers": "tutorial_triggers.yml",
        "facility_name": "ALS",
        "pv_strip_prefix": "ALS:",
    }
    base.update(overrides)
    return DispatchConfig(**base)  # type: ignore[arg-type]


def _services(project_path: Path) -> dict:
    """Reload the rendered config.yml's ``services`` block."""
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)["services"]


def _dispatcher_block(project_path: Path) -> dict:
    """Reload the copied triggers file's ``dispatcher`` block."""
    yaml = YAML()
    with open(project_path / "triggers.yml") as fh:
        return yaml.load(fh)["dispatcher"]


class TestBridgeDefault:
    """With the axis unset, nothing about the build changes."""

    def test_omitted_network_renders_the_pre_axis_config_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        """A profile with no ``dispatch.network`` renders the pinned bytes."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(_dispatch(), profile_dir=profile_dir, project_path=project_path)

        rendered = (project_path / "config.yml").read_text(encoding="utf-8")
        assert rendered == PRE_AXIS_CONFIG_YML

    def test_spelled_bridge_renders_the_same_bytes_as_omitting_it(self, tmp_path: Path) -> None:
        """Writing the default out explicitly is the same build as leaving it out.

        The knob's default is spelled on ``DispatchConfig`` and again at the
        injection read site; both spellings have to agree, or a facility that
        documents its (default) topology in the profile would get a different
        deployment than one that stays silent.
        """
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="bridge"), profile_dir=profile_dir, project_path=project_path
        )

        rendered = (project_path / "config.yml").read_text(encoding="utf-8")
        assert rendered == PRE_AXIS_CONFIG_YML

    def test_neither_half_carries_a_network_key(self, tmp_path: Path) -> None:
        """The default is left to the schema and the template, not written out."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(_dispatch(), profile_dir=profile_dir, project_path=project_path)

        services = _services(project_path)
        assert "network" not in services["event_dispatcher"]
        assert "network" not in services["dispatch_worker"]
        assert "worker_port_stride" not in services["dispatch_worker"]

    def test_dispatch_target_keeps_the_compose_service_name(self, tmp_path: Path) -> None:
        """On the compose bridge the worker is reachable by its service DNS name."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(_dispatch(), profile_dir=profile_dir, project_path=project_path)

        assert _dispatcher_block(project_path)["dispatch_target"] == (
            "http://dispatch-worker-1:10011"
        )

    def test_dispatcher_block_is_written_out_pre_axis_byte_for_byte(self, tmp_path: Path) -> None:
        """The patched region of the triggers file is unchanged, comments included.

        The host-mode rewrite rides on the same round-trip that patches the pool
        limits, so a quoting or comment-anchoring change there would land on
        every build, default or not.
        """
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(_dispatch(), profile_dir=profile_dir, project_path=project_path)

        copied = (project_path / "triggers.yml").read_text(encoding="utf-8")
        assert PRE_AXIS_TRIGGERS_DISPATCHER in copied


class TestHostMode:
    """``dispatch.network: host`` moves the pair and the addresses with it."""

    def test_both_halves_get_the_single_knobs_value(self, tmp_path: Path) -> None:
        """One profile knob, written into both service blocks."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host"), profile_dir=profile_dir, project_path=project_path
        )

        services = _services(project_path)
        assert services["event_dispatcher"]["network"] == "host"
        assert services["dispatch_worker"]["network"] == "host"

    def test_the_rest_of_each_service_block_is_untouched(self, tmp_path: Path) -> None:
        """The axis adds keys; it does not disturb what injection already wrote."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host"), profile_dir=profile_dir, project_path=project_path
        )

        services = _services(project_path)
        dispatcher = services["event_dispatcher"]
        assert dispatcher["path"] == "./services/event_dispatcher"
        assert dispatcher["port"] == 10010
        assert dispatcher["facility_name"] == "ALS"
        assert dispatcher["pv_strip_prefix"] == "ALS:"
        assert dispatcher["additional_dirs"] == [{"src": "triggers.yml", "dst": "triggers.yml"}]
        worker = services["dispatch_worker"]
        assert worker["path"] == "./services/dispatch_worker"
        assert worker["worker_count"] == 1
        assert worker["worker_port_base"] == 10011
        assert worker["workspace_mode"] == "isolated"
        assert worker["timeout_sec"] == 300
        assert worker["inactivity_sec"] == 120

    def test_dispatch_target_moves_to_localhost_on_the_base_port(self, tmp_path: Path) -> None:
        """The compose DNS name does not resolve on the host network.

        Sharing the host's namespace, the dispatcher reaches worker 1 at the
        port that worker binds — the base port.
        """
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host"), profile_dir=profile_dir, project_path=project_path
        )

        assert _dispatcher_block(project_path)["dispatch_target"] == "http://localhost:10011"

    def test_only_the_target_line_of_the_dispatcher_block_changes(self, tmp_path: Path) -> None:
        """The rewrite edits one value; the block's comments and keys stay put."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host"), profile_dir=profile_dir, project_path=project_path
        )

        expected = PRE_AXIS_TRIGGERS_DISPATCHER.replace(
            "dispatch_target: http://dispatch-worker-1:10011",
            "dispatch_target: http://localhost:10011",
        )
        assert expected in (project_path / "triggers.yml").read_text(encoding="utf-8")

    def test_dispatch_target_follows_a_custom_worker_port_base(self, tmp_path: Path) -> None:
        """The rewritten address is derived, never a hardcoded 9901."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host", worker_port_base=9500),
            profile_dir=profile_dir,
            project_path=project_path,
        )

        assert _dispatcher_block(project_path)["dispatch_target"] == "http://localhost:9500"

    def test_pool_limits_are_still_patched_alongside_the_rewrite(self, tmp_path: Path) -> None:
        """The address rewrite rides on the existing pool-limit patch, not over it."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host", max_concurrent_runs=7, max_queue_depth=33),
            profile_dir=profile_dir,
            project_path=project_path,
        )

        dispatcher = _dispatcher_block(project_path)
        assert dispatcher["max_concurrent_runs"] == 7
        assert dispatcher["max_queue_depth"] == 33
        assert dispatcher["dispatch_target"] == "http://localhost:10011"

    def test_triggers_themselves_survive_the_rewrite(self, tmp_path: Path) -> None:
        """Only the dispatcher block is patched — the trigger list is the facility's."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host"), profile_dir=profile_dir, project_path=project_path
        )

        yaml = YAML()
        with open(project_path / "triggers.yml") as fh:
            doc = yaml.load(fh)
        assert any(trigger["name"] == "hello-dispatch" for trigger in doc["triggers"])

    def test_deployed_services_registration_is_unaffected(self, tmp_path: Path) -> None:
        """Host mode changes addresses, not which services deploy."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host"), profile_dir=profile_dir, project_path=project_path
        )

        yaml = YAML()
        with open(project_path / "config.yml") as fh:
            deployed = [str(name) for name in yaml.load(fh)["deployed_services"]]
        assert deployed == ["postgresql", "event_dispatcher", "dispatch_worker"]


class TestPerWorkerPortDerivation:
    """The worker's port rule is recorded as data for the render to consume."""

    def test_stride_is_recorded_beside_base_and_count(self, tmp_path: Path) -> None:
        """The three inputs of ``worker_port_base + (i - 1) * stride`` are all present.

        On the compose bridge each worker has its own namespace and they all
        listen on the base port; on the host network they share one, so the
        compose render and the host-port preflight have to fan the ports out —
        from a declared rule rather than a constant hidden in each consumer.
        """
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host", worker_count=3),
            profile_dir=profile_dir,
            project_path=project_path,
        )

        worker = _services(project_path)["dispatch_worker"]
        assert worker["worker_count"] == 3
        assert worker["worker_port_base"] == 10011
        assert worker["worker_port_stride"] == 1

    def test_recorded_inputs_derive_one_port_per_worker(self, tmp_path: Path) -> None:
        """Reading the config back the way a consumer will yields distinct ports."""
        project_path, profile_dir = _project(tmp_path)

        _inject_dispatch(
            _dispatch(network="host", worker_count=3, worker_port_base=9500),
            profile_dir=profile_dir,
            project_path=project_path,
        )

        worker = _services(project_path)["dispatch_worker"]
        ports = [
            worker["worker_port_base"] + (i - 1) * worker["worker_port_stride"]
            for i in range(1, worker["worker_count"] + 1)
        ]
        assert ports == [9500, 9501, 9502]
        assert len(set(ports)) == len(ports)
