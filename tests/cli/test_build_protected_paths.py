"""The protected-path emitter, on real profile loads of temp deployments.

Every case writes a deployment — ``profile.yml``, its declared sources, login
files, host variants — resolves it the way ``osprey build`` does and reads the
entries the emitter computes. Where the product has a resolver for a key, the
emitted entry is compared with that resolver's answer rather than a literal.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.application_material import dependency_package_dirs
from osprey.cli.build_profile_load import LoadedProfile, load_profile_document
from osprey.cli.build_profile_merge import (
    ProfileInputRole,
    profile_inputs,
    profile_triggers_material,
)
from osprey.cli.build_profile_presets import resolve_service_template
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.cli.build_protected_paths import (
    CONFIG_PATH_KEYS,
    EXEMPT_CONFIG_KEYS,
    INPUT_ENUMERATION_KEYS,
    PROFILE_PATH_FIELDS,
    UndecidedPath,
    emit_protected_paths,
    is_path_shaped,
)
from osprey.registry.manager import resolve_registry_path
from osprey.repo_layout import display_path, rendered_sources
from osprey.services.channel_directory.demo import resolve_replay_file
from osprey.utils.config_paths import resolve_config_relative_path, resolve_render_relative_path
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.workspace import anchored_path

_AGENT_DATA = Path("var/agent_data")


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _deployment(root: Path, **body: Any) -> Path:
    """A deployment at *root* whose ``profile.yml`` states *body*."""
    profile: dict[str, Any] = {"name": "Facility"}
    profile.update(body)
    return _write_yaml(root / "profile.yml", profile)


def _load(path: Path, overlays: tuple[Path, ...] = ()) -> LoadedProfile:
    return resolve_build_document(path, None, overlays)


def _emit(root: Path, *profiles: LoadedProfile):
    return emit_protected_paths(profiles, deployment_root=root, agent_data_root=_AGENT_DATA)


def _entries(root: Path, **body: Any) -> tuple[str, ...]:
    return _emit(root, _load(_deployment(root, **body))).entries


def _at(root: Path, entry: str) -> Path:
    """Where an entry anchors at run time (``~`` expanded, root-relative)."""
    return anchored_path(entry, root).resolve()


def _render(root: Path) -> Path:
    return root / "build"


# ── the loaded profile carries its resolved document ─────────────────────────


def test_the_loaded_profile_carries_the_document_it_was_parsed_from(tmp_path: Path) -> None:
    parent = _write_yaml(tmp_path / "profiles" / "base.yml", {"config": {"a.b": 1}})
    path = _deployment(tmp_path, extends="profiles/base.yml", config={"c.d": 2})

    for loaded in (_load(path), load_profile_document(path)):
        assert loaded.document.root_dir == tmp_path.resolve()
        assert loaded.document.extends_files == (parent.resolve(),)
        assert loaded.document.raw["config"]["a.b"] == 1


def test_a_set_edit_reaches_the_carried_document(tmp_path: Path) -> None:
    path = _deployment(tmp_path)

    loaded = resolve_build_document(path, None, (), ("config.images.site_ca=/etc/ca.pem",))

    assert "/etc/ca.pem" in _emit(tmp_path, loaded).entries


def test_a_preset_carries_a_document_and_contributes_no_package_content(tmp_path: Path) -> None:
    loaded = resolve_build_document(None, "hello-world")

    assert loaded.document.raw is not None
    assert _emit(tmp_path, loaded).entries == ()


# ── feed 1: the build's input enumeration ────────────────────────────────────


def _enumerated(loaded: LoadedProfile) -> set[str]:
    return {
        item.key.rstrip("/")
        for item in profile_inputs(loaded.document)
        if item.role is not ProfileInputRole.INSTALLED
    }


def test_every_enumerated_input_is_emitted(tmp_path: Path) -> None:
    hooks = display_path("hooks")
    (tmp_path / hooks).mkdir(parents=True)
    loaded = _load(_deployment(tmp_path))

    entries = set(_emit(tmp_path, loaded).entries)

    assert _enumerated(loaded) <= entries
    assert {"profile.yml", hooks} <= entries


def test_an_extends_parent_is_emitted_and_its_values_are_read(tmp_path: Path) -> None:
    parent = _write_yaml(
        tmp_path / "shared" / "base.yml", {"config": {"images.site_ca": "/etc/parent-ca.pem"}}
    )
    loaded = _load(_deployment(tmp_path, extends="shared/base.yml"))

    entries = _emit(tmp_path, loaded).entries

    assert "shared/base.yml" in entries
    assert _at(tmp_path, "shared/base.yml") == parent.resolve()
    assert "/etc/parent-ca.pem" in entries


def test_every_rendered_source_is_emitted(tmp_path: Path) -> None:
    sources = [source.source_rel for source in rendered_sources()]
    assert sources
    for source in sources:
        (tmp_path / source).mkdir(parents=True)

    assert set(sources) <= set(_entries(tmp_path))


def test_a_machine_root_is_emitted(tmp_path: Path) -> None:
    (tmp_path / "machine-sub").mkdir()
    loaded = _load(_deployment(tmp_path, machine="machine-sub"))

    assert "machine-sub" in _emit(tmp_path, loaded).entries


def test_a_local_service_template_is_emitted_where_the_resolver_finds_it(tmp_path: Path) -> None:
    (tmp_path / "templates" / "mine").mkdir(parents=True)
    (tmp_path / "templates" / "mine" / "docker-compose.yml.j2").write_text("services: {}\n")
    loaded = _load(
        _deployment(tmp_path, services={"mine": {"template": "templates/mine", "config": {}}})
    )

    entries = _emit(tmp_path, loaded).entries

    expected = resolve_service_template(tmp_path, "templates/mine").path
    assert [e for e in entries if _at(tmp_path, e) == expected.resolve()] == ["templates/mine"]


def test_a_trigger_file_stated_only_in_the_host_variant_is_emitted(tmp_path: Path) -> None:
    (tmp_path / "triggers.yml").write_text("triggers: []\n")
    path = _deployment(tmp_path)
    variant = _write_yaml(
        tmp_path / "profiles" / "studio.yml", {"dispatch": {"triggers": "triggers.yml"}}
    )

    base = _emit(tmp_path, _load(path)).entries
    loaded = _load(path, (variant,))
    entries = _emit(tmp_path, loaded).entries

    assert "triggers.yml" not in base
    spelled, location = profile_triggers_material(loaded.document.raw, tmp_path)
    assert spelled in entries
    assert _at(tmp_path, spelled) == location.resolve()


def _package(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "0"\n')


def test_a_local_package_is_emitted_where_the_resolver_finds_it(tmp_path: Path) -> None:
    _package(tmp_path / "packages" / "pkg")
    loaded = _load(_deployment(tmp_path, dependencies=["./packages/pkg"]))

    entries = _emit(tmp_path, loaded).entries

    (package,) = dependency_package_dirs(["./packages/pkg"], tmp_path)
    assert [e for e in entries if _at(tmp_path, e) == package.resolve()] == ["packages/pkg"]


def test_a_local_package_at_the_root_contributes_its_pyproject_only(tmp_path: Path) -> None:
    _package(tmp_path)
    loaded = _load(_deployment(tmp_path, dependencies=["."]))

    emitted = _emit(tmp_path, loaded)

    assert "pyproject.toml" in emitted.entries
    assert "." not in emitted.entries
    assert any("pyproject.toml" in note for note in emitted.notes)


# ── feed 2: profile fields the enumeration does not list ─────────────────────


def test_the_interpreter_is_emitted_as_stated(tmp_path: Path) -> None:
    assert sys.executable in _entries(tmp_path, environment={"python": sys.executable})


@pytest.mark.parametrize("word", ["project", "build"])
def test_a_python_env_word_names_no_path(tmp_path: Path, word: str) -> None:
    assert _entries(tmp_path, python_env=word) == _entries(tmp_path / "plain")


def test_an_absolute_python_env_is_emitted(tmp_path: Path) -> None:
    assert sys.executable in _entries(tmp_path, python_env=sys.executable)


def test_a_pre_build_working_directory_is_emitted_relative_to_the_root(tmp_path: Path) -> None:
    lifecycle = {
        "pre_build": [
            {"name": "gen", "run": "true", "cwd": "tools/gen"},
            {"name": "gen", "run": "true", "cwd": "{project_root}/scripts"},
        ]
    }

    entries = _entries(tmp_path, lifecycle=lifecycle)

    assert {"tools/gen", "scripts"} <= set(entries)


def test_a_pre_build_working_directory_at_the_root_contributes_nothing(tmp_path: Path) -> None:
    lifecycle = {"pre_build": [{"name": "gen", "run": "true", "cwd": "{project_root}"}]}
    emitted = _emit(tmp_path, _load(_deployment(tmp_path, lifecycle=lifecycle)))

    assert not [e for e in emitted.entries if _at(tmp_path, e) == tmp_path.resolve()]
    assert any("lifecycle.pre_build[0].cwd" in note for note in emitted.notes)


def test_the_bluesky_plan_dir_and_parameter_schemas_are_emitted(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "abs.json"
    for schema in (tmp_path / "deployment" / "schemas" / "scan.json", outside):
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text("{}")
    bluesky = {
        "plan_dir": "plans/",
        "external": {
            "zmq_control_addr": "tcp://qserver:60615",
            "insecure_plaintext": True,
            "parameter_schemas": {
                "scan.motor": "schemas/scan.json",
                "abs.motor": str(outside),
            },
        },
    }

    entries = _entries(tmp_path / "deployment", bluesky=bluesky)

    assert {"plans/", "schemas/scan.json", str(outside)} <= set(entries)


def test_the_optimizer_environments_dir_is_emitted(tmp_path: Path) -> None:
    assert "optimizer-envs" in _entries(tmp_path, optimizer={"environments_dir": "optimizer-envs"})


def _deploy(host: str) -> dict[str, Any]:
    return {
        "ci": "gitlab",
        "image_source": "local",
        "host": {"name": host, "user": "ops", "project_path": "/srv/facility"},
    }


def test_the_deploy_project_path_is_emitted_on_the_deploy_host(tmp_path: Path) -> None:
    assert "/srv/facility" in _entries(tmp_path, deploy=_deploy(socket.gethostname()))


def test_the_deploy_project_path_is_ignored_elsewhere(tmp_path: Path) -> None:
    entries = _entries(tmp_path, deploy=_deploy("deploy-host-elsewhere.invalid"))

    assert "/srv/facility" not in entries


# ── feed 3: configuration keys ───────────────────────────────────────────────

#: (key, stated value, the entry it emits). One per protected config row.
_CONFIG_CASES: list[tuple[str, Any, list[str]]] = [
    ("services.qmd.models_dir", "/srv/qmd/models", ["/srv/qmd/models"]),
    (
        "services.qmd.corpora",
        [{"name": "docs", "source": "docs/"}, {"name": "pre", "index_dir": "/srv/idx"}],
        ["docs/", "/srv/idx"],
    ),
    ("services.bluesky.plan_dir", "./plans", ["./plans"]),
    (
        "services.bluesky.external.parameter_schema_mounts",
        [{"source": "/opt/s.json", "target": "/app/project/plan_schemas/s.json"}],
        ["/opt/s.json"],
    ),
    ("services.optimizer.environments_dir", "envs", ["envs"]),
    (
        "services.event_dispatcher.additional_dirs",
        ["docs", {"src": "scripts", "dst": "bin"}, {"dst": "nothing"}],
        ["docs", "scripts"],
    ),
    ("registry_path", "src/facility/registry.py", ["src/facility/registry.py"]),
    ("application.registry_path", "src/app/registry.py", ["src/app/registry.py"]),
    ("health.plugins", ["facility.health.plugin", "plugins/check.py"], ["plugins/check.py"]),
    ("facility_knowledge.bundle_path", "knowledge", ["knowledge"]),
    ("ariel.vocabulary.path", "vocab/terms.yaml", ["vocab/terms.yaml"]),
    ("channel_directory.alarm_state.replay_file", "replays/alarms.jsonl", ["replays/alarms.jsonl"]),
    ("ariel.ingestion.source_url", "logbook/export.json", ["logbook/export.json"]),
    (
        "phoebus.panels",
        {"main": "displays/main.bob", "rf": {"file": "displays/rf.bob", "macros": ["S"]}},
        ["displays/main.bob", "displays/rf.bob"],
    ),
    ("web.panels.canopy.theme_file", "theme/site.json", ["theme/site.json"]),
    ("images.site_ca", "/etc/site-ca.pem", ["/etc/site-ca.pem"]),
    ("archiver.epics_archiver.tls.ca_bundle", "/etc/epics-ca.pem", ["/etc/epics-ca.pem"]),
    ("archiver.mongodb_archiver.tls.ca_bundle", "/etc/mongo-ca.pem", ["/etc/mongo-ca.pem"]),
    ("channel_directory.settings.tls.ca_bundle", "/etc/cd-ca.pem", ["/etc/cd-ca.pem"]),
    (
        "channel_directory.alarm_state.settings.tls.ca_bundle",
        "/etc/alarm-ca.pem",
        ["/etc/alarm-ca.pem"],
    ),
    ("ariel.ingestion.ca_bundle", "/etc/ariel-ca.pem", ["/etc/ariel-ca.pem"]),
    ("modules.web_terminals.tls.host_cert_dir", "/etc/osprey/certs", ["/etc/osprey/certs"]),
    ("bluesky.plan_dirs", ["plans/a", "/opt/plans"], ["plans/a", "/opt/plans"]),
    ("bluesky.plan_module", "plans/legacy.py", ["plans/legacy.py"]),
    (
        "ariel.enhancement_modules.qmd_export.settings.mirror_path",
        "mirror/nested",
        ["mirror/nested"],
    ),
    ("ariel.enhancement_modules.qmd_export.mirror_path", "mirror/flat", ["mirror/flat"]),
    (
        "modules.web_terminals.landing.notices",
        ["notices/one.md", "notices/two.md"],
        ["notices/one.md", "notices/two.md"],
    ),
    ("services.graphdb.ttl_path", "/srv/seed/graph.ttl", ["/srv/seed/graph.ttl"]),
    ("services.postgresql.path", "/srv/compose/postgresql", ["/srv/compose/postgresql"]),
]


def _config_only(root: Path, config: dict[str, Any]) -> set[str]:
    return set(_entries(root, config=config)) - set(_entries(root / "bare"))


@pytest.mark.parametrize(
    ("key", "value", "expected"), _CONFIG_CASES, ids=[c[0] for c in _CONFIG_CASES]
)
def test_each_protected_config_key_is_emitted(
    tmp_path: Path, key: str, value: Any, expected: list[str]
) -> None:
    assert _config_only(tmp_path, {key: value}) == set(expected)


def test_every_protected_config_row_has_a_case() -> None:
    covered = {case[0] for case in _CONFIG_CASES}

    assert {row.key for row in CONFIG_PATH_KEYS} - covered == {"services.<name>.path"}
    assert "services.postgresql.path" in covered


def test_config_entries_anchor_where_the_products_resolvers_read(tmp_path: Path) -> None:
    config = {
        "registry_path": "src/facility/registry.py",
        "facility_knowledge.bundle_path": "knowledge",
        "channel_directory.alarm_state.replay_file": "replays/alarms.jsonl",
        "services.graphdb.ttl_path": "../seeds/graph.ttl",
    }
    entries = set(_entries(tmp_path, config=config))
    render = _render(tmp_path)

    registry = resolve_registry_path({"registry_path": config["registry_path"]}, base_path=tmp_path)
    assert any(_at(tmp_path, e) == Path(registry).resolve() for e in entries)
    knowledge = resolve_config_relative_path(config["facility_knowledge.bundle_path"], render)
    assert any(_at(tmp_path, e) == knowledge for e in entries)
    replay = resolve_replay_file(config["channel_directory.alarm_state.replay_file"], tmp_path)
    assert any(_at(tmp_path, e) == replay.resolve() for e in entries)
    seed = resolve_render_relative_path(config["services.graphdb.ttl_path"], render)
    assert any(_at(tmp_path, e) == seed for e in entries)


def test_a_service_key_is_read_in_both_spellings(tmp_path: Path) -> None:
    services = {"qmd": {"template": "osprey.qmd", "config": {"models_dir": "/srv/block/models"}}}
    config = {"services": {"qmd": {"models_dir": "/srv/config/models"}}}

    entries = _entries(tmp_path, services=services, config=config)

    assert {"/srv/block/models", "/srv/config/models"} <= set(entries)


def test_dotted_and_nested_config_spellings_are_one_key(tmp_path: Path) -> None:
    nested = _entries(tmp_path / "nested", config={"images": {"site_ca": "/etc/ca.pem"}})
    dotted = _entries(tmp_path / "dotted", config={"images.site_ca": "/etc/ca.pem"})

    assert "/etc/ca.pem" in nested
    assert "/etc/ca.pem" in dotted


# ── the filters ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("health.plugins", ["facility.health.module"]),
        ("channel_directory.alarm_state.replay_file", "demo:example-alarms"),
        ("ariel.ingestion.source_url", "https://logbook.example.org/api"),
        ("modules.web_terminals.landing.notices", []),
    ],
)
def test_a_filtered_value_names_no_path(tmp_path: Path, key: str, value: Any) -> None:
    assert _config_only(tmp_path, {key: value}) == set()


def test_a_packaged_replay_resolves_outside_the_deployment(tmp_path: Path) -> None:
    replay = resolve_replay_file("demo:example-alarms", tmp_path)

    assert not replay.resolve().is_relative_to(tmp_path.resolve())


# ── anchors ──────────────────────────────────────────────────────────────────


def test_a_graph_seed_inside_the_render_is_not_emitted(tmp_path: Path) -> None:
    assert _config_only(tmp_path, {"services.graphdb.ttl_path": "data/graph.ttl"}) == set()


def test_a_graph_seed_that_climbs_out_of_the_render_is_emitted_where_it_is(tmp_path: Path) -> None:
    emitted = _config_only(tmp_path, {"services.graphdb.ttl_path": "../seeds/graph.ttl"})

    (entry,) = emitted
    assert entry == "seeds/graph.ttl"
    expected = resolve_render_relative_path("../seeds/graph.ttl", _render(tmp_path))
    assert _at(tmp_path, entry) == expected


def test_an_absolute_graph_seed_is_emitted(tmp_path: Path) -> None:
    emitted = _config_only(tmp_path, {"services.graphdb.ttl_path": "/srv/graph.ttl"})

    assert emitted == {"/srv/graph.ttl"}
    assert _at(tmp_path, "/srv/graph.ttl") == resolve_render_relative_path(
        "/srv/graph.ttl", _render(tmp_path)
    )


def test_a_relative_service_path_stays_in_the_render(tmp_path: Path) -> None:
    assert _config_only(tmp_path, {"services.postgresql.path": "./services/postgresql"}) == set()


def test_home_and_placeholders_are_emitted_verbatim(tmp_path: Path) -> None:
    config = {
        "images.site_ca": "~/certs/site-ca.pem",
        "services.qmd.models_dir": "${QMD_MODELS_ROOT}/models",
    }

    assert _config_only(tmp_path, config) == {"~/certs/site-ca.pem", "${QMD_MODELS_ROOT}/models"}


def test_a_config_key_naming_the_root_is_noted_not_emitted(tmp_path: Path) -> None:
    emitted = _emit(tmp_path, _load(_deployment(tmp_path, config={"bluesky.plan_dirs": ["."]})))

    assert "." not in emitted.entries
    assert any("bluesky.plan_dirs" in note for note in emitted.notes)


# ── logins ───────────────────────────────────────────────────────────────────


def test_the_union_covers_a_login_file(tmp_path: Path) -> None:
    base = _deployment(tmp_path, config={"bluesky.plan_dirs": ["plans/base"]})
    login = _write_yaml(
        tmp_path / "tiers" / "ops.yml",
        {"name": "Ops", "config": {"images.site_ca": "/etc/login-ca.pem"}},
    )

    entries = _emit(tmp_path, _load(base), _load(login)).entries

    assert {"plans/base", "/etc/login-ca.pem", "tiers/ops.yml"} <= set(entries)


# ── decision 6: no protected path in the agent data zone ─────────────────────


def test_a_protected_path_inside_the_agent_data_zone_is_refused(tmp_path: Path) -> None:
    loaded = _load(_deployment(tmp_path, config={"images.site_ca": "var/agent_data/ca.pem"}))

    with pytest.raises(BuildProfileError, match=r"images\.site_ca.*var/agent_data/ca\.pem"):
        _emit(tmp_path, loaded)


def test_a_custom_agent_data_root_is_the_zone_refused(tmp_path: Path) -> None:
    loaded = _load(_deployment(tmp_path, config={"bluesky.plan_dirs": ["scratch/plans"]}))

    with pytest.raises(BuildProfileError, match="bluesky.plan_dirs"):
        emit_protected_paths([loaded], deployment_root=tmp_path, agent_data_root=Path("scratch"))


# ── decision 7: undecided path-shaped values ─────────────────────────────────


def test_an_undecided_path_shaped_value_is_reported_not_emitted(tmp_path: Path) -> None:
    services = {"mine": {"template": "osprey.qmd", "config": {"seed_file": "./seed/s.csv"}}}
    config = {
        "custom.store": "/srv/custom",
        "custom.label": "hello",
        "screen_capture.output_dir": "./caps",
    }

    emitted = _emit(tmp_path, _load(_deployment(tmp_path, config=config, services=services)))

    assert UndecidedPath("custom.store", "/srv/custom") in emitted.undecided
    assert UndecidedPath("services.mine.seed_file", "./seed/s.csv") in emitted.undecided
    assert {u.key for u in emitted.undecided} == {"custom.store", "services.mine.seed_file"}
    assert "/srv/custom" not in emitted.entries


@pytest.mark.parametrize(
    ("value", "shaped"),
    [
        ("/abs", True),
        ("./rel", True),
        ("../up", True),
        ("~/home", True),
        ("dir/file.txt", True),
        ("dir/sub", False),
        ("file.txt", False),
        ("n/a", False),
        ("https://x/y.json", False),
        ("/with space", False),
    ],
)
def test_path_shape(value: str, shaped: bool) -> None:
    assert is_path_shaped(value) is shaped


def test_the_feeds_and_tables_do_not_overlap() -> None:
    field_keys = {row.key for row in PROFILE_PATH_FIELDS}
    config_keys = {row.key for row in CONFIG_PATH_KEYS}

    assert not field_keys & INPUT_ENUMERATION_KEYS
    assert not config_keys & set(EXEMPT_CONFIG_KEYS)
    assert all(row.shape and row.anchor for row in (*PROFILE_PATH_FIELDS, *CONFIG_PATH_KEYS))
