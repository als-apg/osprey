"""``dispatch.tier``: the worker runs the tier's render as built.

A worker at a tier runs the tier image, whose project is baked at
``/app/<p>-<tier>`` with its own ``build/config.yml``. The worker compose
therefore points ``CONFIG_FILE`` at that baked config, and derives every other
in-container path the worker writes or reads — ``OSPREY_PROJECT_DIR``, the
agent-data volume target, the audit root, the simulator-log and guarded-run
mount targets — from that one project directory. The one thing mounted over
the baked config is the worker's own staged copy of it,
``build/services/dispatch_worker/config.yml``: the container-rooted tier render
plus the ``control_system.login_tiers`` map, which ``osprey build`` writes after
the tier projects are rendered. The terminals on that tier keep the map-free
baked render; the worker keeps its tier's posture and gains the map.

With ``dispatch.tier`` unset the worker compose is byte-for-byte what it was
before the tier existed: every untiered shape the committed goldens pin
(rendered from the template as it stood before this change) renders the same
bytes now.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import _inject_project_metadata

_TEMPLATES_TESTS = Path(__file__).resolve().parents[1] / "templates"
_WORKER_TEMPLATE = "services/dispatch_worker/docker-compose.yml.j2"
_TIER_ROOT = "/app/demo-readonly"
_OWN_ROOT = "/app/demo"


def _renders() -> dict:
    return {
        name: {"kind": "tier", "project": f"demo-{name}", "project_path": f"build/demo-{name}"}
        for name in ("readonly", "readwrite")
    }


def _config(*, tier: str | None = "readonly", worker: dict | None = None) -> dict:
    worker_cfg = dict(worker or {})
    if tier is not None:
        worker_cfg["tier"] = tier
    return {
        "project_name": "demo",
        "project_root": "/r/demo",
        "registry": {"url": "reg.example/osprey"},
        "system": {"timezone": "UTC"},
        "deployed_services": ["event_dispatcher", "dispatch_worker"],
        "services": {"dispatch_worker": worker_cfg},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "image_tag": "v1",
                "default_tier": "readwrite",
                "users": [],
                "renders": _renders(),
            }
        },
    }


def _environment():
    from importlib import resources

    from jinja2 import Environment, FileSystemLoader

    return Environment(
        loader=FileSystemLoader(str(resources.files("osprey").joinpath("templates"))),
        autoescape=False,
        keep_trailing_newline=True,
    )


def _render(config: dict) -> str:
    context = _inject_project_metadata(config)
    context["osprey_env_chain"] = []
    context["osprey_env_present"] = False
    context["osprey_env_shared_present"] = False
    return _environment().get_template(_WORKER_TEMPLATE).render(**context)


def _worker_services(rendered: str) -> dict:
    services = yaml.safe_load(rendered)["services"]
    return {key: value for key, value in services.items() if key.startswith("dispatch-worker")}


def _mount_targets(service: dict) -> list[str]:
    return [str(volume).split(":")[1] for volume in service.get("volumes", [])]


# ── a tiered worker runs the tier's render ───────────────────────────────────


@pytest.mark.parametrize("mode", ["isolated", "shared"])
def test_tiered_worker_paths_share_the_tier_project_dir(mode: str) -> None:
    config = _config(worker={"worker_count": 2, "workspace_mode": mode})
    agent_data = _inject_project_metadata(config)["osprey_container_agent_data_dir"]
    assert agent_data.startswith(f"{_TIER_ROOT}/")
    workers = _worker_services(_render(config))
    assert sorted(workers) == ["dispatch-worker-1", "dispatch-worker-2"]
    for key, service in workers.items():
        index = key.rsplit("-", 1)[1]
        volume = "dispatch_workspace" if mode == "shared" else f"dispatch_workspace_{index}"
        env = service["environment"]
        assert service["image"].endswith("demo-readonly:local}")
        assert env["OSPREY_PROJECT_DIR"] == _TIER_ROOT
        assert env["CONFIG_FILE"] == f"{_TIER_ROOT}/build/config.yml"
        assert env["OSPREY_SPOOL_DIR"] == f"{_TIER_ROOT}/var/spool/{key}"
        assert env["OSPREY_GUARDED_RUN_DIR"].startswith(f"{_TIER_ROOT}/")
        assert f"{volume}:{agent_data}" in service["volumes"]
        assert f"{_TIER_ROOT}/var/spool/{key}" in _mount_targets(service)


def test_tiered_worker_agent_data_volume_lands_under_the_tier_project() -> None:
    context = _inject_project_metadata(_config())
    agent_data = context["osprey_container_agent_data_dir"]
    assert agent_data.startswith(f"{_TIER_ROOT}/")
    rendered = _render(_config())
    (service,) = _worker_services(rendered).values()
    assert f"dispatch_workspace_1:{agent_data}" in service["volumes"]


def test_tiered_worker_mounts_its_staged_copy_over_the_tier_config_and_names_no_own_project_path() -> (
    None
):
    """The staged tier copy is the one config mount, read-only, at the tier's
    own ``CONFIG_FILE`` — never at the deployment's own project path."""
    rendered = _render(_config())
    assert f"{_OWN_ROOT}/" not in rendered
    (service,) = _worker_services(rendered).values()
    config_mounts = [
        str(volume)
        for volume in service.get("volumes", [])
        if str(volume).split(":")[1].endswith("/build/config.yml")
    ]
    assert config_mounts == [
        f"./build/services/dispatch_worker/config.yml:{_TIER_ROOT}/build/config.yml:ro"
    ]
    assert service["environment"]["CONFIG_FILE"] == f"{_TIER_ROOT}/build/config.yml"


def test_tiered_worker_simulator_log_target_follows_the_tier_project() -> None:
    config = _config()
    config["control_system"] = {"type": "virtual_accelerator"}
    context = _inject_project_metadata(config)
    assert context["osprey_simulator_log_mount"]["target"].startswith(f"{_TIER_ROOT}/")
    assert context["osprey_guarded_run_mount"]["target"].startswith(f"{_TIER_ROOT}/")


def test_tier_moves_only_the_worker_zone_roots() -> None:
    """Other services (history) keep the deployment's own spool and state roots."""
    context = _inject_project_metadata(_config())
    for zone in ("spool", "state"):
        assert context[f"osprey_container_{zone}_dir"] == f"{_OWN_ROOT}/var/{zone}"
        assert context[f"osprey_worker_{zone}_dir"] == f"{_TIER_ROOT}/var/{zone}"


def test_tier_render_does_not_mutate_the_input_config() -> None:
    config = _config()
    before = yaml.safe_dump(config)
    _render(config)
    assert yaml.safe_dump(config) == before


def test_tier_without_a_render_is_refused() -> None:
    with pytest.raises(ValueError, match="dispatch.tier"):
        _inject_project_metadata(_config(tier="missing"))


# ── unset dispatch.tier: today's bytes ──────────────────────────────────────


def test_untiered_worker_keeps_the_deployment_project_dir() -> None:
    context = _inject_project_metadata(_config(tier=None))
    assert context["osprey_worker_project_dir"] == _OWN_ROOT
    assert context["osprey_worker_spool_dir"] == context["osprey_container_spool_dir"]
    assert context["osprey_worker_state_dir"] == context["osprey_container_state_dir"]
    assert context["osprey_worker_stages_config"] is True
    rendered = _render(_config(tier=None))
    (service,) = _worker_services(rendered).values()
    assert service["environment"]["OSPREY_PROJECT_DIR"] == _OWN_ROOT
    assert (
        f"./build/services/dispatch_worker/config.yml:{_OWN_ROOT}/build/config.yml:ro"
        in service["volumes"]
    )


@pytest.fixture(scope="module")
def golden_harnesses():
    """The two golden suites, whose committed worker renders predate the tier."""
    sys.path.insert(0, str(_TEMPLATES_TESTS))
    try:
        defaults = importlib.import_module("test_render_defaults_golden")
        axes = importlib.import_module("test_render_axis_shapes")
    finally:
        sys.path.remove(str(_TEMPLATES_TESTS))
    return defaults, axes


def test_untiered_default_worker_render_is_byte_identical(golden_harnesses) -> None:
    defaults, _ = golden_harnesses
    golden = (defaults._GOLDEN_DIR / "dispatch_worker.yml").read_text(encoding="utf-8")
    with defaults._probe_repo() as repo_root:
        context = defaults._default_context(repo_root)
        assert "tier" not in (context["services"].get("dispatch_worker") or {})
        now = defaults._render_templates(context, [_WORKER_TEMPLATE])["dispatch_worker.yml"]
    assert now == golden


def test_untiered_worker_axis_shapes_are_byte_identical(golden_harnesses) -> None:
    _, axes = golden_harnesses
    scenarios = [s for s in axes.SCENARIOS if "dispatch_worker" in s.templates]
    assert scenarios, "the axis-shape suite pins no worker render"
    for scenario in scenarios:
        assert "tier" not in scenario.services.get("dispatch_worker", {})
        now = axes._render_scenario(scenario)["dispatch_worker.yml"]
        assert now == axes._golden_text(scenario, "dispatch_worker"), scenario.name


# ── build staging ───────────────────────────────────────────────────────────


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict) -> Path:
    from osprey.deployment.compose_generator import setup_build_dir

    (tmp_path / "config.yml").write_text(yaml.safe_dump({"project_name": "demo"}))
    service_dir = tmp_path / "services" / "dispatch_worker"
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml.j2").write_text("services:\n  w:\n    image: test\n")
    monkeypatch.chdir(tmp_path)
    setup_build_dir(
        str(Path("services") / "dispatch_worker" / "docker-compose.yml.j2"),
        {**config, "project_root": str(tmp_path), "build_dir": "./build"},
        {"copy_src": False},
    )
    return tmp_path / "build" / "services" / "dispatch_worker" / "config.yml"


def test_setup_build_dir_stages_no_config_for_a_tiered_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _stage(tmp_path, monkeypatch, _config())
    assert staged.parent.is_dir()
    assert not staged.exists()


def test_setup_build_dir_stages_the_config_for_an_untiered_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _stage(tmp_path, monkeypatch, _config(tier=None))
    assert staged.is_file()


# ── a tiered worker binds what its tier is granted ──────────────────────────

_CORPUS_CONFIG = {
    "facility_knowledge": {"bundle_path": "data/facility/knowledge"},
    "ariel": {
        "enhancement_modules": {
            "qmd_export": {"enabled": True, "mirror_path": "data/ariel/mirror"},
        }
    },
}


def _corpus_repo(tmp_path: Path, *, entitled: bool) -> dict:
    """A repo whose readonly render does (or does not) configure bundle and mirror."""
    render_config = dict(_CORPUS_CONFIG) if entitled else {}
    for name in ("readonly", "readwrite"):
        project = tmp_path / "build" / f"demo-{name}"
        project.mkdir(parents=True)
        (project / "config.yml").write_text(
            yaml.safe_dump(render_config if name == "readonly" else {})
        )
    config = _config()
    config["project_root"] = str(tmp_path)
    config.update(_CORPUS_CONFIG)
    return config


def _render_worker(config: dict) -> dict:
    from osprey.deployment.compose_generator import _dispatch_worker_corpus_mounts

    context = _inject_project_metadata(config)
    context["osprey_env_chain"] = []
    context["osprey_env_present"] = False
    context["osprey_env_shared_present"] = False
    context["dispatch_worker_corpus_mounts"] = _dispatch_worker_corpus_mounts(
        config, "services/dispatch_worker"
    )
    rendered = _environment().get_template(_WORKER_TEMPLATE).render(**context)
    (service,) = _worker_services(rendered).values()
    return service


def test_tiered_worker_binds_the_bundle_and_mirror_its_tier_is_granted(tmp_path: Path) -> None:
    """The worker reads the grant sets the web terminals read, and binds in the
    web-terminal shape: the deployment's directory at the render's own path."""
    # Arrange
    config = _corpus_repo(tmp_path, entitled=True)

    # Act
    service = _render_worker(config)

    # Assert
    env = service["environment"]
    assert env["OSPREY_FACILITY_BUNDLE_DIR"] == f"{_TIER_ROOT}/data/facility/knowledge"
    assert env["OSPREY_ARIEL_MIRROR_DIR"] == f"{_TIER_ROOT}/data/ariel/mirror"
    assert f"./data/facility/knowledge:{_TIER_ROOT}/data/facility/knowledge" in service["volumes"]
    assert f"./data/ariel/mirror:{_TIER_ROOT}/data/ariel/mirror" in service["volumes"]


def test_tiered_worker_binds_nothing_its_tier_is_not_granted(tmp_path: Path) -> None:
    # Arrange
    config = _corpus_repo(tmp_path, entitled=False)

    # Act
    service = _render_worker(config)

    # Assert
    assert "OSPREY_FACILITY_BUNDLE_DIR" not in service["environment"]
    assert "OSPREY_ARIEL_MIRROR_DIR" not in service["environment"]
    assert not any("data/facility/knowledge" in v for v in service["volumes"])


def test_worker_corpus_mounts_are_the_workers_alone() -> None:
    from osprey.deployment.compose_generator import _dispatch_worker_corpus_mounts

    assert _dispatch_worker_corpus_mounts(_config(), "services/ariel_sync") is None
    assert _dispatch_worker_corpus_mounts(_config(tier=None), "services/dispatch_worker") is None
