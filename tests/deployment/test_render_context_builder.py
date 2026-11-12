"""One render context for every service render, on both setup paths.

``setup_build_dir`` and its busy-build-dir fallback ``_incremental_setup_build_dir``
render a service's compose template from the same context: the per-service key
(``service_key``, ``service``), the ``panel_grant`` the roster-grant table fills
for the panel's service — the roster users' secrets, or on a roster-less
deployment the single-user alias — and the staged login map for a
``drives_target`` container.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import (
    _incremental_setup_build_dir,
    service_render_context,
    setup_build_dir,
)
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from tests._renders import tier_render

_PARTIALS = Path(__file__).parents[2] / "src" / "osprey" / "templates" / "services"

LOGIN_TIERS = {"alice": "allowed", "bob": "asked", "carol": "never"}

#: Reads every name this builder injects, so a render shows them all.
_TEMPLATE = """\
{%- import "services/_panel_grant.j2" as panel -%}
services:
  {{ service_key }}:
    image: demo
    environment:
      PORT: "{{ service.port }}"
{{- panel.env(panel_grant.credentials) }}
"""

_ENTRY_POINTS = ("full", "incremental")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with a panel service and a plain one, entered as the build does."""
    repo = tmp_path / "repo"
    for key in ("knob-demo", "plain"):
        service = repo / "services" / key
        service.mkdir(parents=True)
        (service / "docker-compose.yml.j2").write_text(_TEMPLATE)
    shutil.copy(_PARTIALS / "_panel_grant.j2", repo / "services")
    for tier, panels in (("viewer", {}), ("operator", {"knob-demo": {"url": "http://x"}})):
        project = repo / "build" / f"demo-{tier}"
        project.mkdir(parents=True)
        (project / "config.yml").write_text(yaml.safe_dump({"web": {"panels": panels}}))
    (repo / "config.yml").write_text("project_name: demo\n")
    monkeypatch.chdir(repo)
    return repo


def _config(repo: Path) -> dict:
    return {
        "project_name": "demo",
        "project_root": str(repo),
        "config_dir": str(repo),
        "build_dir": "./build",
        "deployment": {},
        "system": {"timezone": "UTC"},
        "services": {
            "knob-demo": {
                "path": "services/knob-demo",
                "port": 18000,
                "drives_target": True,
                "roster_grant": {"panel_id": "knob-demo", "single_user_alias": False},
            },
            "plain": {"path": "services/plain", "port": 18001},
        },
        "deployed_services": ["knob-demo", "plain"],
        "modules": {
            "web_terminals": {
                "default_tier": "viewer",
                "renders": {
                    tier: tier_render(tier, f"demo-{tier}", project_path=f"build/demo-{tier}")
                    for tier in ("viewer", "operator")
                },
                "users": [
                    {"name": "alice", "index": 0, "tier": "operator"},
                    {"name": "bob", "index": 1, "tier": "viewer"},
                ],
            }
        },
    }


def _render(entry_point: str, key: str, config: dict) -> dict:
    template = f"services/{key}/docker-compose.yml.j2"
    out_dir = Path("build") / "services" / key
    if entry_point == "full":
        setup_build_dir(template, config, {}, login_tiers=LOGIN_TIERS, service_key=key)
    else:
        # A busy build dir: the previous render's files are still in place.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yml").write_text("project_name: demo\n")
        _incremental_setup_build_dir(
            template, config, {}, str(out_dir), login_tiers=LOGIN_TIERS, service_key=key
        )
    return yaml.safe_load((out_dir / "docker-compose.yml").read_text())


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_the_panel_service_renders_its_grant(repo: Path, entry_point: str) -> None:
    alice = terminal_secret_var("alice")
    suffix = alice.removeprefix("OSPREY_TERMINAL_SECRET_")
    rendered = _render(entry_point, "knob-demo", _config(repo))
    assert rendered["services"]["knob-demo"]["environment"] == {
        "PORT": "18000",
        "OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS": "1",
        alice: f"${{{alice}:-}}",
        "OSPREY_TERMINAL_ROSTER_OWNERS": f"{suffix}=alice",
    }


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_every_other_service_renders_its_own_key_and_no_grant(repo: Path, entry_point: str) -> None:
    rendered = _render(entry_point, "plain", _config(repo))
    assert rendered["services"]["plain"]["environment"] == {"PORT": "18001"}


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_both_paths_stage_the_login_map_into_a_drives_target_context(
    repo: Path, entry_point: str
) -> None:
    _render(entry_point, "knob-demo", _config(repo))
    staged = yaml.safe_load((Path("build") / "services" / "knob-demo" / "config.yml").read_text())
    assert staged["control_system"]["login_tiers"] == LOGIN_TIERS


def test_the_service_key_defaults_to_the_template_directory(repo: Path) -> None:
    setup_build_dir("services/plain/docker-compose.yml.j2", _config(repo), {})
    rendered = yaml.safe_load((Path("build/services/plain/docker-compose.yml")).read_text())
    assert rendered["services"]["plain"]["environment"] == {"PORT": "18001"}


def test_the_bluesky_sidecar_reads_its_grant_from_panel_grant(repo: Path) -> None:
    """The sidecar's grant is ``panel_grant``, under no older name of its own."""
    config = _config(repo)
    config["services"]["bluesky_web"] = {"path": "services/bluesky_web"}
    (repo / "build" / "demo-operator" / "config.yml").write_text(
        yaml.safe_dump({"web": {"panels": {"bluesky": {"url": "http://x"}}}})
    )
    out_dir = repo / "build" / "services" / "bluesky_web"
    out_dir.mkdir(parents=True)

    context = service_render_context(
        config, "services/bluesky_web", str(out_dir), service_key="bluesky_web"
    )
    alice = terminal_secret_var("alice")
    assert context["panel_grant"] == {
        "credentials": [{"var": alice, "source": alice, "owner": "alice", "required": False}]
    }
    assert "bluesky_panel_secret_env_vars" not in context
    assert "bluesky_panel_roster_owners" not in context
    assert context["service_key"] == "bluesky_web"
    assert context["service"] == config["services"]["bluesky_web"]

    plain = service_render_context(config, "services/plain", str(out_dir), service_key="plain")
    assert plain["panel_grant"] == {"credentials": []}


@pytest.mark.parametrize("name", ["bluesky_panel_secret_env_vars", "bluesky_panel_roster_owners"])
def test_a_template_reading_a_retired_grant_name_is_refused(repo: Path, name: str) -> None:
    """A hand-kept template still reading an old name fails the build, never renders empty."""
    template = Path("services/plain/docker-compose.yml.j2")
    template.write_text(
        _TEMPLATE + f"{{%- if {name} | default([]) %}}\n      X: '1'\n{{%- endif %}}\n"
    )
    with pytest.raises(DeploymentPreconditionError, match=name) as refused:
        setup_build_dir(str(template), _config(repo), {})
    assert "panel_grant.credentials" in refused.value.reason
    assert "services/bluesky_web/docker-compose.yml.j2" in refused.value.remedy


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_the_named_service_key_decides_the_staged_roles(repo: Path, entry_point: str) -> None:
    """A ``drives_target`` block keyed apart from its template directory still gets both stages.

    The key the render names is the one both role-keyed stagers read, never the
    directory's basename.
    """
    template_dir = repo / "services" / "knob-template"
    template_dir.mkdir(parents=True)
    (template_dir / "docker-compose.yml.j2").write_text(_TEMPLATE)
    config = _config(repo)
    config["services"]["knob-demo"]["path"] = "services/knob-template"
    config["services"]["knob-demo"]["control_identity"] = True

    template = "services/knob-template/docker-compose.yml.j2"
    out_dir = Path("build") / "services" / "knob-template"
    if entry_point == "full":
        setup_build_dir(template, config, {}, login_tiers=LOGIN_TIERS, service_key="knob-demo")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yml").write_text("project_name: demo\n")
        _incremental_setup_build_dir(
            template, config, {}, str(out_dir), login_tiers=LOGIN_TIERS, service_key="knob-demo"
        )

    staged = yaml.safe_load((out_dir / "config.yml").read_text())
    assert staged["control_system"]["login_tiers"] == LOGIN_TIERS
    assert (out_dir / "control_identity").is_dir()


def _roster_less(repo: Path, *, single_user_alias: bool) -> dict:
    config = _config(repo)
    del config["modules"]["web_terminals"]
    config["services"]["knob-demo"]["roster_grant"]["single_user_alias"] = single_user_alias
    return config


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_an_alias_panel_with_no_roster_accepts_the_deployment_secret_as_the_build_account(
    repo: Path, entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "ops")
    var = terminal_secret_var("ops")
    suffix = var.removeprefix("OSPREY_TERMINAL_SECRET_")
    rendered = _render(entry_point, "knob-demo", _roster_less(repo, single_user_alias=True))
    assert rendered["services"]["knob-demo"]["environment"] == {
        "PORT": "18000",
        "OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS": "1",
        var: "${OSPREY_TERMINAL_SECRET}",
        "OSPREY_TERMINAL_ROSTER_OWNERS": f"{suffix}=ops",
    }


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_no_alias_for_a_panel_that_does_not_ask_for_one(repo: Path, entry_point: str) -> None:
    rendered = _render(entry_point, "knob-demo", _roster_less(repo, single_user_alias=False))
    assert rendered["services"]["knob-demo"]["environment"] == {"PORT": "18000"}
    plain = _render(entry_point, "plain", _roster_less(repo, single_user_alias=True))
    assert plain["services"]["plain"]["environment"] == {"PORT": "18001"}


def _optimizer_web_single_user(repo: Path, **deployment: str) -> dict:
    """A roster-less deployment serving the OPTIMIZER tab through optimizer_web."""
    service = repo / "services" / "optimizer_web"
    service.mkdir(parents=True, exist_ok=True)
    (service / "docker-compose.yml.j2").write_text(_TEMPLATE)
    config = _roster_less(repo, single_user_alias=False)
    config["services"]["optimizer_web"] = {"path": "services/optimizer_web", "port": 18002}
    config["deployed_services"].append("optimizer_web")
    config["deployment"] = dict(deployment)
    return config


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_the_optimizer_sidecar_names_the_stamped_host_identity(
    repo: Path, entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The alias owner is who ``osprey web`` signs in as, not who ran the build.

    A Start from the panel files its write request under this name, and the
    terminal's chip lists only the requests its own login owns: two names for one
    person would leave the request unapprovable.
    """
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "someone-else")
    var = terminal_secret_var("operator")
    suffix = var.removeprefix("OSPREY_TERMINAL_SECRET_")
    config = _optimizer_web_single_user(repo, host_identity="operator")
    rendered = _render(entry_point, "optimizer_web", config)
    assert rendered["services"]["optimizer_web"]["environment"] == {
        "PORT": "18002",
        "OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS": "1",
        var: "${OSPREY_TERMINAL_SECRET}",
        "OSPREY_TERMINAL_ROSTER_OWNERS": f"{suffix}=operator",
    }


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_the_optimizer_sidecar_published_on_the_network_names_nobody(
    repo: Path, entry_point: str
) -> None:
    config = _optimizer_web_single_user(repo, host_identity="operator", bind_address="0.0.0.0")
    rendered = _render(entry_point, "optimizer_web", config)
    assert rendered["services"]["optimizer_web"]["environment"] == {"PORT": "18002"}
