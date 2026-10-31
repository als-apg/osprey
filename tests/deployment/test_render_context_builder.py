"""One render context for every service render, on both setup paths.

``setup_build_dir`` and its busy-build-dir fallback ``_incremental_setup_build_dir``
render a service's compose template from the same context: the per-service key
(``service_key``, ``service``), the ``panel_grant`` the roster-grant table fills
for the panel's service, and the staged login map for a ``drives_target``
container.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import (
    _incremental_setup_build_dir,
    service_render_context,
    setup_build_dir,
)
from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from tests._renders import tier_render

LOGIN_TIERS = {"alice": "allowed", "bob": "asked", "carol": "never"}

#: Reads every name this builder injects, so a render shows them all.
_TEMPLATE = """\
services:
  {{ service_key }}:
    image: demo
    environment:
      PORT: "{{ service.port }}"
{%- if panel_grant.secret_env_vars %}
      OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS: "1"
{%- for var in panel_grant.secret_env_vars %}
      {{ var }}: ${{ "{" }}{{ var }}:-}
{%- endfor %}
      OSPREY_TERMINAL_ROSTER_OWNERS: "{{ panel_grant.roster_owners }}"
{%- endif %}
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


def test_the_builder_aliases_the_bluesky_names_to_its_grant(repo: Path) -> None:
    """The two legacy bluesky names are the same values ``panel_grant`` holds."""
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
    assert context["panel_grant"]["secret_env_vars"] == [terminal_secret_var("alice")]
    assert context["bluesky_panel_secret_env_vars"] == context["panel_grant"]["secret_env_vars"]
    assert context["bluesky_panel_roster_owners"] == context["panel_grant"]["roster_owners"]
    assert context["service_key"] == "bluesky_web"
    assert context["service"] == config["services"]["bluesky_web"]

    plain = service_render_context(config, "services/plain", str(out_dir), service_key="plain")
    assert plain["panel_grant"] == {"secret_env_vars": [], "roster_owners": ""}
    assert plain["bluesky_panel_secret_env_vars"] == []
    assert plain["bluesky_panel_roster_owners"] == ""
