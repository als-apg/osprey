"""``dispatch.tier``: the tier the dispatch workers run at.

Schema half: the key parses onto :class:`DispatchConfig`, names a tier file,
is refused when that tier holds something a headless run has no approver for,
and is written by the dispatch injector as ``services.dispatch_worker.tier`` —
a name, never an image string. Unset renders exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_dispatch
from osprey.cli.build_profile import DispatchConfig
from osprey.cli.build_profile_presets import _presets_dir
from osprey.cli.build_profile_schema import (
    dispatch_tier_errors,
    headless_refused_flags,
)
from osprey.deployment.web_terminals.roster import DEPLOYMENT_EDIT_DOORS
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import invoke_build_in

_READONLY = """\
name: Read-Only
config:
  control_system.writes_enabled: false
  web.ui_mode: simple
"""

_ADMIN = """\
name: Admin
config:
  claude_code.permissions.remove_deny:
    - mcp__osprey_workspace__setup_patch
  web.config_panel.enabled: true
  web.scaffold_gallery.write_enabled: true
"""


def _repo(tmp_path: Path, **tiers: str) -> Path:
    """A profile root with ``tiers/<name>.yml`` for each keyword."""
    root = tmp_path / "repo"
    (root / "tiers").mkdir(parents=True)
    (root / "profile.yml").write_text("name: test\n", encoding="utf-8")
    for name, text in tiers.items():
        (root / "tiers" / f"{name}.yml").write_text(text, encoding="utf-8")
    return root


def _dispatch(**overrides: object) -> DispatchConfig:
    base: dict = {"triggers": "tutorial_triggers.yml", "dispatcher_port": 8020}
    base.update(overrides)
    return DispatchConfig(**base)  # type: ignore[arg-type]


def _write_config(project_path: Path, config: dict | None = None) -> None:
    doc = {"deployed_services": [], "services": {}}
    doc.update(config or {})
    with open(project_path / "config.yml", "w") as fh:
        YAML().dump(doc, fh)


def _read_config(project_path: Path) -> dict:
    with open(project_path / "config.yml") as fh:
        return YAML().load(fh)


# ── the dataclass and the parser ─────────────────────────────────────────────


def test_schema_tier_defaults_to_unset() -> None:
    assert _dispatch().tier is None


def test_schema_parser_carries_dispatch_tier() -> None:
    from osprey.cli.build_profile_load import _KNOWN_DISPATCH_KEYS, _parse_profile

    assert "tier" in _KNOWN_DISPATCH_KEYS
    raw = {"name": "t", "dispatch": {"triggers": "tutorial_triggers.yml", "tier": "readonly"}}
    profile = _parse_profile(raw)
    assert profile.dispatch is not None
    assert profile.dispatch.tier == "readonly"


def test_schema_parser_leaves_tier_unset_when_unspelled() -> None:
    from osprey.cli.build_profile_load import _parse_profile

    profile = _parse_profile({"name": "t", "dispatch": {"triggers": "tutorial_triggers.yml"}})
    assert profile.dispatch is not None
    assert profile.dispatch.tier is None


# ── dispatch_tier_errors ─────────────────────────────────────────────────────


def test_schema_unset_tier_is_accepted(tmp_path: Path) -> None:
    assert dispatch_tier_errors(None, tmp_path) == []


def test_schema_readonly_tier_in_repo_is_accepted(tmp_path: Path) -> None:
    root = _repo(tmp_path, readonly=_READONLY)
    assert dispatch_tier_errors("readonly", root) == []


@pytest.mark.parametrize("value", ["", "   ", 3, ["readonly"], True])
def test_schema_tier_must_be_a_name(tmp_path: Path, value: object) -> None:
    errors = dispatch_tier_errors(value, _repo(tmp_path))
    assert len(errors) == 1
    assert "dispatch.tier" in errors[0]


def test_schema_tier_must_name_a_file_in_tiers(tmp_path: Path) -> None:
    root = _repo(tmp_path, readonly=_READONLY)
    errors = dispatch_tier_errors("operator", root)
    assert len(errors) == 1
    assert "'operator'" in errors[0]
    assert "tiers/operator.yml" in errors[0]
    assert "readonly" in errors[0]  # names what is available


def test_schema_admin_tier_is_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path, admin=_ADMIN)
    errors = dispatch_tier_errors("admin", root)
    assert len(errors) == 1
    message = errors[0]
    assert "a headless run has no approver" in message
    assert "lifts a base deny" in message
    for flag in headless_refused_flags():
        assert flag in message


def test_headless_refused_flags_are_the_switch_doors_of_the_door_table() -> None:
    """The headless refusal reads the one door table, never a list of its own."""
    switches = tuple(door.key for door in DEPLOYMENT_EDIT_DOORS if door.is_switch)
    assert headless_refused_flags() == switches
    assert headless_refused_flags() == (
        "web.config_panel.enabled",
        "web.scaffold_gallery.write_enabled",
    )


@pytest.mark.parametrize(
    ("body", "names"),
    [
        (
            "config:\n  claude_code.permissions.remove_deny: [mcp__x__y]\n",
            "lifts a base deny",
        ),
        ("config:\n  web.config_panel.enabled: true\n", "web.config_panel.enabled"),
        (
            "config:\n  web.scaffold_gallery.write_enabled: true\n",
            "web.scaffold_gallery.write_enabled",
        ),
        # Nested spelling names the same key.
        ("config:\n  web:\n    config_panel:\n      enabled: true\n", "web.config_panel.enabled"),
        (
            "config:\n  claude_code:\n    permissions:\n      remove_deny: [mcp__x__y]\n",
            "lifts a base deny",
        ),
    ],
)
def test_schema_each_headless_privilege_is_refused(tmp_path: Path, body: str, names: str) -> None:
    root = _repo(tmp_path, worker=f"name: worker\n{body}")
    errors = dispatch_tier_errors("worker", root)
    assert len(errors) == 1
    assert names in errors[0]
    assert "a headless run has no approver" in errors[0]


@pytest.mark.parametrize(
    "body",
    [
        "config:\n  claude_code.permissions.remove_deny: []\n",
        "config:\n  web.config_panel.enabled: false\n",
        "config:\n  web.scaffold_gallery.write_enabled: false\n",
        "name: no config block\n",
    ],
)
def test_schema_off_or_empty_privileges_are_accepted(tmp_path: Path, body: str) -> None:
    root = _repo(tmp_path, worker=body)
    assert dispatch_tier_errors("worker", root) == []


def test_schema_flag_inherited_from_the_deployment_is_refused(tmp_path: Path) -> None:
    """A tier that does not spell a flag runs the deployment's value of it."""
    root = _repo(tmp_path, readonly=_READONLY)
    base = {"web": {"config_panel": {"enabled": True}}}
    errors = dispatch_tier_errors("readonly", root, base)
    assert len(errors) == 1
    assert "web.config_panel.enabled" in errors[0]


def test_schema_tier_override_beats_the_deployment_flag(tmp_path: Path) -> None:
    root = _repo(tmp_path, worker="config:\n  web.config_panel.enabled: false\n")
    base = {"web": {"config_panel": {"enabled": True}}}
    assert dispatch_tier_errors("worker", root, base) == []


def test_schema_preset_mode_reads_bundled_tiers() -> None:
    presets = _presets_dir()
    assert dispatch_tier_errors("readonly", presets) == []
    admin = dispatch_tier_errors("admin", presets)
    assert len(admin) == 1
    assert "a headless run has no approver" in admin[0]
    assert dispatch_tier_errors("no-such-tier", presets) != []


def test_schema_repo_tiers_shadow_nothing_from_the_package(tmp_path: Path) -> None:
    """A repo resolves only its own tiers/: a bundled name it lacks is unknown."""
    root = _repo(tmp_path)
    errors = dispatch_tier_errors("readonly", root)
    assert len(errors) == 1
    assert "names no tier" in errors[0]


# ── _inject_dispatch ─────────────────────────────────────────────────────────


def _project(tmp_path: Path, config: dict | None = None) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    _write_config(project, config)
    return project


def test_schema_injector_writes_worker_tier_by_name(tmp_path: Path) -> None:
    root = _repo(tmp_path, readonly=_READONLY)
    project = _project(tmp_path)
    _inject_dispatch(_dispatch(tier="readonly"), profile_dir=root, project_path=project)

    services = _read_config(project)["services"]
    assert services["dispatch_worker"]["tier"] == "readonly"
    assert "image" not in services["dispatch_worker"]
    assert "tier" not in services["event_dispatcher"]


def test_schema_injector_unset_tier_writes_nothing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    project = _project(tmp_path)
    _inject_dispatch(_dispatch(), profile_dir=root, project_path=project)

    worker = _read_config(project)["services"]["dispatch_worker"]
    assert "tier" not in worker
    assert "image" not in worker


def test_schema_injector_refuses_admin_before_writing(tmp_path: Path) -> None:
    root = _repo(tmp_path, admin=_ADMIN)
    project = _project(tmp_path)
    before = (project / "config.yml").read_text(encoding="utf-8")

    with pytest.raises(BuildProfileError, match="a headless run has no approver"):
        _inject_dispatch(_dispatch(tier="admin"), profile_dir=root, project_path=project)

    assert (project / "config.yml").read_text(encoding="utf-8") == before
    assert not (project / "triggers.yml").exists()


def test_schema_injector_refuses_inherited_panel(tmp_path: Path) -> None:
    root = _repo(tmp_path, readonly=_READONLY)
    project = _project(tmp_path, {"web": {"config_panel": {"enabled": True}}})
    with pytest.raises(BuildProfileError, match="web.config_panel.enabled"):
        _inject_dispatch(_dispatch(tier="readonly"), profile_dir=root, project_path=project)


def test_schema_injector_refuses_unknown_tier(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    project = _project(tmp_path)
    with pytest.raises(BuildProfileError, match="names no tier"):
        _inject_dispatch(_dispatch(tier="readonly"), profile_dir=root, project_path=project)


# ── bridges gain nothing (A1) ────────────────────────────────────────────────


def test_schema_bridge_configs_have_no_tier() -> None:
    from dataclasses import fields

    from osprey.cli.build_profile_schema import (
        GChatBridgeProfileConfig,
        NextcloudBridgeProfileConfig,
        TeamsBridgeProfileConfig,
    )

    for cls in (NextcloudBridgeProfileConfig, GChatBridgeProfileConfig, TeamsBridgeProfileConfig):
        assert "tier" not in {f.name for f in fields(cls)}, cls.__name__


# ── the manifest registers the key ───────────────────────────────────────────


def test_schema_manifest_registers_worker_tier() -> None:
    import re

    import osprey

    manifest_path = Path(osprey.__file__).parent / "profiles" / "config_key_manifest.yml"
    manifest = YAML(typ="safe").load(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["keys"]["services.dispatch_worker.tier"]
    # control-assistant sets ``dispatch.tier``, so its render carries the key:
    # a ``rendered: false`` flag would be a false claim the guard refuses.
    assert entry.get("rendered", True) is True
    injectors = (Path(osprey.__file__).parent / "cli" / "build_injectors.py").read_text(
        encoding="utf-8"
    )
    assert re.search(entry["evidence"], injectors)


# ── the flagship preset and the exemplar ─────────────────────────────────────


def _emitted_profile(preset: str) -> dict:
    """The ``profile.yml`` that ``osprey init --preset <preset>`` writes, parsed."""
    from osprey.cli.build_profile_emit import emit_standalone_profile_yaml

    text = emit_standalone_profile_yaml(preset, (), "Demo", project_name="demo")
    return YAML(typ="safe").load(text)


@pytest.fixture(scope="module")
def _built_exemplar(tmp_path_factory) -> Path:
    """The exemplar repo after one real ``osprey build`` (no venv, no hooks)."""

    from tests.fixtures.lifecycle_repo import (
        EXEMPLAR_DIRNAME,
        build_exemplar_repo,
        preserved_environ,
    )

    base = tmp_path_factory.mktemp("flagship-dispatch-tier")
    repo = build_exemplar_repo(base / EXEMPLAR_DIRNAME, seed_env=True)
    with preserved_environ():
        result = invoke_build_in(repo)
    assert result.exit_code == 0, result.output
    return repo


class TestFlagshipExemplar:
    """The flagship preset runs its workers at ``readonly``; nothing else changes."""

    def test_control_assistant_profile_sets_dispatch_tier_readonly(self) -> None:
        assert _emitted_profile("control-assistant")["dispatch"]["tier"] == "readonly"

    def test_exemplar_profile_carries_the_preset_line(self) -> None:
        from tests.fixtures.lifecycle_repo import exemplar_source_files

        profile = YAML(typ="safe").load(exemplar_source_files()["profile.yml"])
        assert profile["dispatch"]["tier"] == "readonly"

    def test_hello_world_has_no_dispatch_tier(self) -> None:
        profile = _emitted_profile("hello-world")
        assert "tier" not in (profile.get("dispatch") or {})

    @pytest.mark.slow
    def test_built_worker_runs_the_readonly_render(self, _built_exemplar: Path) -> None:
        from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME

        compose = _built_exemplar / "build" / "services" / "dispatch_worker" / "docker-compose.yml"
        services = YAML(typ="safe").load(compose.read_text(encoding="utf-8"))["services"]
        workers = {k: v for k, v in services.items() if k.startswith("dispatch-worker")}
        assert workers, sorted(services)
        root = f"/app/{EXEMPLAR_DIRNAME}-readonly"
        for key, service in workers.items():
            env = service["environment"]
            assert service["image"].endswith(f"{EXEMPLAR_DIRNAME}-readonly:local}}")
            assert env["OSPREY_PROJECT_DIR"] == root
            assert env["CONFIG_FILE"] == f"{root}/build/config.yml"
            assert env["OSPREY_SPOOL_DIR"] == f"{root}/var/spool/{key}"
            workspace = [
                str(volume).split(":")[1]
                for volume in service.get("volumes", [])
                if str(volume).startswith("dispatch_workspace")
            ]
            assert workspace, service.get("volumes")
            assert all(target.startswith(f"{root}/") for target in workspace), workspace
