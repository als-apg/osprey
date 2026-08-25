"""The ARIEL mirror and the per-user audit zone, bound into web terminals.

Two more host directories the per-user containers WRITE, each with the same
shape as the knowledge bundle (``test_bundle_mount.py``) and asserted the same
way:

* **The ARIEL qmd mirror.** A persona whose render enables ``qmd_export``
  runs that module inside its own container, and the module writes wherever
  ``mirror_path`` resolves — the persona's project root, in-container. Bound
  to the deployment's one mirror (the directory the host exporter fills and
  the sidecar indexes), the entry reaches the search corpus; unbound, it lands
  in the container's writable layer where nothing indexes it and the next
  recreate discards it. Entitlement is per persona, the source is shared, the
  target is per service.
* **The refusal audit zone.** The python executor in every container appends
  its refusal log under ``<project>/var/audit``. Nothing else backs that path
  — the agent-data volume is a sibling — so it is bound to a per-user host
  directory (``var/audit/<user>``): per user because every container writes
  the same fixed filename, a host bind because an audit log has to be readable
  without entering a container. Every user, no entitlement.

Both join the same ``group_add`` story as the bundle, and the three are
emitted as one list per service.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.utils.workspace import AUDIT_DIR_RELPATH

from .test_golden_render import EXAMPLE_CONFIG

MIRROR_PATH = "var/ariel_mirror"
MIRROR_SOURCE = f"./{MIRROR_PATH}"


def _config(*, mirror: bool = True, personas: bool = False, via_settings: bool = True) -> dict:
    """The reference roster, optionally persona-backed and running a qmd export."""
    config = copy.deepcopy(EXAMPLE_CONFIG)
    if mirror:
        export: dict = {"enabled": True}
        if via_settings:
            export["settings"] = {"mirror_path": MIRROR_PATH}
        else:
            export["mirror_path"] = MIRROR_PATH
        config["ariel"] = {"enhancement_modules": {"qmd_export": export}}
    if personas:
        web_terminals = config["modules"]["web_terminals"]
        web_terminals["personas"] = {
            "operator": {"project": "dls-operator", "project_path": "../dls-operator"},
            "physicist": {"project": "dls-physicist", "project_path": "../dls-physicist"},
        }
        web_terminals["users"] = [
            {"name": "alice", "index": 0, "persona": "operator"},
            {"name": "bob", "index": 1, "persona": "physicist"},
        ]
    return config


def _services(config: dict, **kwargs) -> dict[str, dict]:
    compose = yaml.safe_load(render_web_terminals(config, **kwargs)["docker-compose.web.yml"])
    return {name: service for name, service in compose["services"].items() if name != "nginx"}


def _volumes(config: dict, **kwargs) -> dict[str, list[str]]:
    return {name: svc.get("volumes", []) for name, svc in _services(config, **kwargs).items()}


def _mirror_mounts(config: dict, **kwargs) -> dict[str, list[str]]:
    return {
        name: [v for v in volumes if v.startswith(f"{MIRROR_SOURCE}:")]
        for name, volumes in _volumes(config, **kwargs).items()
    }


def _audit_mounts(config: dict, **kwargs) -> dict[str, list[str]]:
    return {
        name: [v for v in volumes if v.startswith(f"./{AUDIT_DIR_RELPATH}/")]
        for name, volumes in _volumes(config, **kwargs).items()
    }


# ---------------------------------------------------------------------------
# The mirror mount
# ---------------------------------------------------------------------------


def test_no_qmd_export_mounts_no_mirror():
    assert _mirror_mounts(_config(mirror=False)) == {"web-alice": [], "web-bob": []}


def test_every_persona_less_user_gets_the_mirror():
    """With no catalog the deploy config is every user's config, so its own
    export entitles everyone — answered with no disk read."""
    assert _mirror_mounts(_config()) == {
        "web-alice": [f"{MIRROR_SOURCE}:/app/dls-assistant/{MIRROR_PATH}"],
        "web-bob": [f"{MIRROR_SOURCE}:/app/dls-assistant/{MIRROR_PATH}"],
    }


def test_mirror_path_on_the_module_block_is_honoured_too():
    """``settings.mirror_path`` wins when present; the bare key is read otherwise —
    the exporter's own merge rule, followed here so the mount lands where it writes."""
    assert _mirror_mounts(_config(via_settings=False)) == {
        "web-alice": [f"{MIRROR_SOURCE}:/app/dls-assistant/{MIRROR_PATH}"],
        "web-bob": [f"{MIRROR_SOURCE}:/app/dls-assistant/{MIRROR_PATH}"],
    }


def test_target_is_computed_per_persona_from_its_own_project_dir():
    mounts = _mirror_mounts(_config(personas=True), ariel_mirror_personas={"operator", "physicist"})
    assert mounts == {
        "web-alice": [f"{MIRROR_SOURCE}:/app/dls-operator/{MIRROR_PATH}"],
        "web-bob": [f"{MIRROR_SOURCE}:/app/dls-physicist/{MIRROR_PATH}"],
    }


def test_only_entitled_personas_get_the_mirror():
    mounts = _mirror_mounts(_config(personas=True), ariel_mirror_personas={"operator"})
    assert mounts == {
        "web-alice": [f"{MIRROR_SOURCE}:/app/dls-operator/{MIRROR_PATH}"],
        "web-bob": [],
    }


def test_no_entitled_personas_mounts_no_mirror():
    assert _mirror_mounts(_config(personas=True)) == {"web-alice": [], "web-bob": []}


def test_disabled_export_mounts_nothing():
    config = _config()
    config["ariel"]["enhancement_modules"]["qmd_export"]["enabled"] = False
    assert _mirror_mounts(config) == {"web-alice": [], "web-bob": []}


def test_absolute_mirror_path_is_not_re_anchored():
    config = _config()
    config["ariel"]["enhancement_modules"]["qmd_export"]["settings"]["mirror_path"] = "/srv/mirror"
    volumes = _volumes(config)["web-alice"]
    assert "/srv/mirror:/srv/mirror" in volumes


def test_mirror_mount_is_read_write():
    """The container's exporter writes here; only the sidecar mounts it ``:ro``."""
    (mount,) = _mirror_mounts(_config())["web-alice"]
    assert not mount.endswith(":ro")


# ---------------------------------------------------------------------------
# The audit mount
# ---------------------------------------------------------------------------


def test_every_user_gets_a_per_user_audit_bind():
    assert _audit_mounts(_config(mirror=False)) == {
        "web-alice": [f"./{AUDIT_DIR_RELPATH}/alice:/app/dls-assistant/{AUDIT_DIR_RELPATH}"],
        "web-bob": [f"./{AUDIT_DIR_RELPATH}/bob:/app/dls-assistant/{AUDIT_DIR_RELPATH}"],
    }


def test_audit_target_follows_each_personas_project_dir():
    assert _audit_mounts(_config(mirror=False, personas=True)) == {
        "web-alice": [f"./{AUDIT_DIR_RELPATH}/alice:/app/dls-operator/{AUDIT_DIR_RELPATH}"],
        "web-bob": [f"./{AUDIT_DIR_RELPATH}/bob:/app/dls-physicist/{AUDIT_DIR_RELPATH}"],
    }


def test_audit_sources_are_distinct_per_user():
    """Two containers write the same filename; one shared directory would
    have them clobber each other."""
    sources = {
        name: mount.split(":")[0] for name, (mount,) in _audit_mounts(_config(mirror=False)).items()
    }
    assert len(set(sources.values())) == len(sources)


def test_audit_target_is_where_the_executor_writes():
    """The target is the writer's own derivation — the project root plus
    :data:`AUDIT_DIR_RELPATH` — never a literal that could drift from it."""
    from osprey.deployment.web_terminals.render import _container_audit_dir

    assert _container_audit_dir("/app/x") == f"/app/x/{AUDIT_DIR_RELPATH}"
    (mount,) = _audit_mounts(_config(mirror=False))["web-alice"]
    assert mount.endswith(f":{_container_audit_dir('/app/dls-assistant')}")


# ---------------------------------------------------------------------------
# group_add: one list per service, every shared directory's group once
# ---------------------------------------------------------------------------


def _group_add(config: dict, **kwargs) -> dict[str, list[str]]:
    return {name: svc.get("group_add", []) for name, svc in _services(config, **kwargs).items()}


def test_no_gids_emits_no_group_add():
    assert _group_add(_config()) == {"web-alice": [], "web-bob": []}


def test_audit_gid_is_joined_by_every_service():
    assert _group_add(_config(mirror=False), audit_gid=1234) == {
        "web-alice": ["1234"],
        "web-bob": ["1234"],
    }


def test_each_distinct_gid_is_listed_once():
    """The deploy creates every shared directory as one user, so they usually
    share a group; a list that repeats it would read like two groups."""
    config = _config()
    config["facility_knowledge"] = {"bundle_path": "data/facility_knowledge"}
    groups = _group_add(config, facility_bundle_gid=20, ariel_mirror_gid=20, audit_gid=20)
    assert groups == {"web-alice": ["20"], "web-bob": ["20"]}
    groups = _group_add(config, facility_bundle_gid=20, ariel_mirror_gid=21, audit_gid=22)
    assert groups == {"web-alice": ["20", "21", "22"], "web-bob": ["20", "21", "22"]}


def test_mirror_gid_reaches_only_entitled_services():
    groups = _group_add(
        _config(personas=True), ariel_mirror_personas={"operator"}, ariel_mirror_gid=21
    )
    assert groups == {"web-alice": ["21"], "web-bob": []}


# ---------------------------------------------------------------------------
# Entitlement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "ariel": {
                    "enhancement_modules": {"qmd_export": {"enabled": True, "mirror_path": "m"}}
                }
            },
            True,
            id="enabled-with-path",
        ),
        pytest.param(
            {
                "ariel": {
                    "enhancement_modules": {
                        "qmd_export": {"enabled": True, "settings": {"mirror_path": "m"}}
                    }
                }
            },
            True,
            id="enabled-with-settings-path",
        ),
        pytest.param(
            {
                "ariel": {
                    "enhancement_modules": {"qmd_export": {"enabled": False, "mirror_path": "m"}}
                }
            },
            False,
            id="disabled",
        ),
        pytest.param(
            {"ariel": {"enhancement_modules": {"qmd_export": {"enabled": True}}}},
            False,
            id="enabled-without-path",
        ),
        pytest.param({"ariel": {}}, False, id="no-modules"),
        pytest.param({}, False, id="absent"),
    ],
)
def test_config_needs_ariel_mirror(config, expected):
    from osprey.deployment.web_terminals.personas import config_needs_ariel_mirror

    assert config_needs_ariel_mirror(config) is expected


def test_personas_needing_ariel_mirror_reads_each_personas_config(tmp_path):
    from osprey.deployment.web_terminals.personas import personas_needing_ariel_mirror

    (tmp_path / "dls-operator").mkdir()
    (tmp_path / "dls-operator" / "config.yml").write_text(
        yaml.safe_dump(
            {
                "ariel": {
                    "enhancement_modules": {"qmd_export": {"enabled": True, "mirror_path": "m"}}
                }
            }
        )
    )
    (tmp_path / "dls-physicist").mkdir()
    (tmp_path / "dls-physicist" / "config.yml").write_text(yaml.safe_dump({"facility": {}}))
    config = _config(personas=True)
    catalog = config["modules"]["web_terminals"]["personas"]
    catalog["operator"]["project_path"] = "dls-operator"
    catalog["physicist"]["project_path"] = "dls-physicist"

    assert personas_needing_ariel_mirror(config, tmp_path) == {"operator"}


# ---------------------------------------------------------------------------
# The host side: one reader, one directory, provisioned before the render
# ---------------------------------------------------------------------------


def test_mirror_reader_is_shared_with_the_sidecar_corpus_list(tmp_path):
    """The mount source, the sidecar's corpus and the provisioned directory all
    come from ``configured_ariel_mirror_path`` — one string, three consumers."""
    from osprey.deployment.compose_generator import (
        _resolve_qmd_corpora,
        configured_ariel_mirror_path,
        resolve_ariel_mirror_dir,
    )

    config = _config()
    assert configured_ariel_mirror_path(config) == MIRROR_PATH
    corpora = {c["collection"]: c for c in _resolve_qmd_corpora(config, str(tmp_path))}
    assert corpora["ariel"]["source"] == MIRROR_SOURCE
    assert resolve_ariel_mirror_dir(config, tmp_path) == tmp_path / MIRROR_PATH
    (mount,) = _mirror_mounts(config)["web-alice"]
    assert mount.split(":")[0] == corpora["ariel"]["source"]


def test_user_audit_dir_is_the_bind_source(tmp_path):
    from osprey.deployment.compose_generator import resolve_user_audit_dir, user_audit_relpath

    assert user_audit_relpath("alice") == f"{AUDIT_DIR_RELPATH}/alice"
    assert resolve_user_audit_dir(tmp_path, "alice") == tmp_path / AUDIT_DIR_RELPATH / "alice"
    (mount,) = _audit_mounts(_config(mirror=False))["web-alice"]
    assert mount.split(":")[0] == f"./{user_audit_relpath('alice')}"


# ---------------------------------------------------------------------------
# The entitlement/source split, caught by lint
# ---------------------------------------------------------------------------


def _mirror_lintable(tmp_path, persona_mirror: str | None, deploy_mirror: str | None) -> dict:
    """A local-mode roster with one rendered persona, linted from *tmp_path*."""
    project = tmp_path / "dls-operator"
    project.mkdir(exist_ok=True)
    (project / "Dockerfile").write_text("FROM scratch")
    persona_config: dict = {"project_name": "dls-operator"}
    if persona_mirror is not None:
        persona_config["ariel"] = {
            "enhancement_modules": {"qmd_export": {"enabled": True, "mirror_path": persona_mirror}}
        }
    (project / "config.yml").write_text(yaml.safe_dump(persona_config))

    config = _config(mirror=deploy_mirror is not None, personas=True, via_settings=False)
    if deploy_mirror is not None:
        config["ariel"]["enhancement_modules"]["qmd_export"]["mirror_path"] = deploy_mirror
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["image_source"] = "local"
    web_terminals["personas"] = {
        "operator": {
            "project": "dls-operator",
            "project_path": "dls-operator",
            "build_profile": "personas/operator.yml",
        }
    }
    web_terminals["users"] = [{"name": "alice", "index": 0, "persona": "operator"}]
    return config


def _mirror_findings(config: dict) -> list:
    from osprey.deployment.web_terminals.lint import lint_web_terminals

    return [
        f
        for f in lint_web_terminals(config)
        if f.code == "web_terminals.persona_mirror_path_divergence"
    ]


def test_lint_flags_a_persona_exporting_where_the_deployment_writes_no_mirror(
    tmp_path, monkeypatch
):
    """The persona is entitled (its config runs a qmd export) but the
    deployment writes no mirror, so the overlay has no source to bind: the
    persona's exporter would write into its writable layer, unindexed and
    lost at the next recreate, with every layer reporting success."""
    monkeypatch.chdir(tmp_path)
    config = _mirror_lintable(tmp_path, persona_mirror=MIRROR_PATH, deploy_mirror=None)

    (finding,) = _mirror_findings(config)

    assert finding.severity == "error"
    assert "runs none" in finding.message


def test_lint_flags_a_persona_that_relocated_its_mirror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _mirror_lintable(tmp_path, persona_mirror="var/elsewhere", deploy_mirror=MIRROR_PATH)

    (finding,) = _mirror_findings(config)

    assert finding.severity == "error"
    assert "var/elsewhere" in finding.message and MIRROR_PATH in finding.message


def test_lint_is_quiet_when_persona_and_deployment_agree_on_the_mirror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _mirror_findings(_mirror_lintable(tmp_path, MIRROR_PATH, MIRROR_PATH)) == []
    assert _mirror_findings(_mirror_lintable(tmp_path, None, MIRROR_PATH)) == []
    assert _mirror_findings(_mirror_lintable(tmp_path, None, None)) == []
