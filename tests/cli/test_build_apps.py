"""``osprey build`` over a deployment's applications.

Every test runs the real build over the exemplar repo. The refusals are read
before any artifact is written; the rest read the renders the build published.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.web_terminals.render_naming import render_project
from osprey.profiles.web_panels import panel_spec_enabled
from osprey_connectors.types import tier_writes
from tests.cli._lifecycle_build import invoke_build, rendered_config, stub_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

KNOB: dict[str, Any] = {
    "name": "knob",
    "containers": {
        "knob": {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_app.server"],
            "health": "/health",
            "drives_target": True,
        },
    },
    "panel": {
        "label": "Knob",
        "container": "knob",
        "launch_token": True,
        "agent_routes": [["GET", "/api/state"], ["POST", "/api/set"]],
    },
    "tools": {
        "module": "knob_app.tools",
        "shown_when": "panel",
        "read": ["knob_read"],
        "approval": [
            {
                "tool": "knob_zero",
                "prompt": "Zero the knob",
                "require": {"all_of": ["approved_target"]},
            }
        ],
    },
}

#: A requirement the framework's own pin of the package refuses.
MOVED_PIN = "lume-base==0.0.1"

GET_ROUTE = {"method": "GET", "path": "/api/state"}
POST_ROUTE = {"method": "POST", "path": "/api/set"}

WRITE_TOOL = "mcp__knob__knob_zero"

TIERS = ("asked", "allowed", "never")


def _quiet_login(tier: str) -> str:
    """A login file at *tier* showing nothing of the deployment's applications."""
    return (
        f"name: Quiet {tier}\n"
        f"config:\n  control_system.writes: {tier}\n"
        "apps:\n  knob:\n    shown: false\n"
    )


def _knob(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(KNOB), **changes}


def _write_app(repo: Path, manifest: dict[str, Any]) -> Path:
    app_dir = repo / "apps" / manifest["name"]
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.yml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return app_dir


def _append_to_profile(repo: Path, block: dict[str, Any]) -> None:
    with (repo / "profile.yml").open("a", encoding="utf-8") as profile:
        profile.write("\n" + yaml.safe_dump(block, sort_keys=False))


def _list_apps(repo: Path, apps: dict[str, Any]) -> None:
    _append_to_profile(repo, {"apps": apps})


def _with_knob(repo: Path, entry: dict[str, Any] | None = None, **changes: Any) -> Path:
    _write_app(repo, _knob(**changes))
    _list_apps(repo, {"knob": entry or {}})
    return repo


def _roster_less(repo: Path, tier: str = "allowed") -> Path:
    """The exemplar with no ``tiers/`` and no web terminals: the root is the one login."""
    shutil.rmtree(repo / "tiers")
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["config"].pop("modules.web_terminals")
    profile["config"]["modules.web_terminals.enabled"] = False
    profile["config"]["control_system.writes"] = tier
    profile["dispatch"].pop("tier", None)
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return repo


def _set_config(repo: Path, key: str, value: Any) -> None:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["config"][key] = value
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def _render_dir(repo: Path, login: str) -> Path:
    project, _ = render_project(EXEMPLAR_DIRNAME, login)
    return repo / "build" / project


def _render(repo: Path, login: str) -> dict[str, Any]:
    return yaml.safe_load((_render_dir(repo, login) / "config.yml").read_text(encoding="utf-8"))


def _denied(repo: Path, login: str) -> list[str]:
    settings = _render_dir(repo, login) / ".claude" / "settings.json"
    return json.loads(settings.read_text(encoding="utf-8"))["permissions"]["deny"]


def _gated(render_dir: Path) -> list[str]:
    path = render_dir / ".claude" / "hooks" / "hook_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["facility_write_tools"]


def _built(repo: Path) -> Path:
    result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    return repo


def _package(directory: Path, name: str, *dependencies: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\ndependencies = {list(dependencies)!r}\n',
        encoding="utf-8",
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def no_artifact(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Assert a refusal arrived before the deployment's packages, the first artifact step."""
    reached: list[bool] = []

    def stop(*_args: Any, **_kwargs: Any) -> None:
        reached.append(True)
        raise RuntimeError("build_deployment_packages was reached")

    monkeypatch.setattr("osprey.cli.build_cmd.build_deployment_packages", stop)

    def refused(repo: Path) -> str:
        build = stub_build(repo)
        before = _tree(build)
        result = invoke_build(repo)
        assert result.exit_code != 0
        assert not reached, caplog.text
        assert _tree(build) == before
        return result.output + caplog.text

    return refused


# --- refusals, before any artifact -------------------------------------------


def test_two_invalid_apps_report_both_violations_and_leave_build_untouched(
    lifecycle_repo: Path, no_artifact
) -> None:
    _write_app(lifecycle_repo, _knob(name="phoebus", panel=None, tools=None))
    _write_app(lifecycle_repo, _knob(name="ariel", panel=None, tools=None))
    _list_apps(lifecycle_repo, {"phoebus": {}, "ariel": {}})

    output = no_artifact(lifecycle_repo)

    assert "apps/phoebus/app.yml" in output
    assert "apps/ariel/app.yml" in output
    assert "a name OSPREY keeps for itself" in output


def test_a_pinned_package_moved_by_an_app_names_app_package_and_both_versions(
    lifecycle_repo: Path, no_artifact
) -> None:
    from osprey.apps.rules import framework_pins

    _package(_with_knob(lifecycle_repo) / "apps" / "knob", "knob-app", MOVED_PIN)

    output = no_artifact(lifecycle_repo)

    assert "knob" in output
    assert "lume-base" in output
    assert "0.0.1" in output
    assert framework_pins()["lume-base"] in output


def test_an_audit_identity_taken_by_a_roster_user_is_refused(
    lifecycle_repo: Path, no_artifact
) -> None:
    _with_knob(lifecycle_repo, audit_identity="alice")

    output = no_artifact(lifecycle_repo)

    assert "apps/knob/app.yml" in output
    assert "alice" in output
    assert "roster" in output


def test_a_shared_person_card_on_an_arming_login_is_refused_naming_the_card(
    lifecycle_repo: Path, no_artifact
) -> None:
    _with_knob(lifecycle_repo)
    text = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    (lifecycle_repo / "profile.yml").write_text(
        text.replace("      - name: alice\n", "      - name: alice\n        access: any\n"),
        encoding="utf-8",
    )

    output = no_artifact(lifecycle_repo)

    assert "alice" in output
    assert "knob" in output
    assert "is shared" in output


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("services.knob.port", 18999),
        ("claude_code.servers.knob", {"command": "x"}),
        ("apps.knob.approval", {}),
        ("dispatch.denied_app_tools", []),
    ],
)
def test_a_hand_written_key_inside_a_projected_namespace_is_refused(
    lifecycle_repo: Path, no_artifact, key: str, value: Any
) -> None:
    _with_knob(lifecycle_repo)
    profile = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    profile["config"][key] = value
    (lifecycle_repo / "profile.yml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )

    output = no_artifact(lifecycle_repo)

    assert key in output
    assert "projects" in output


@pytest.mark.parametrize(
    ("key", "value"),
    [("apps.knob.approval", {}), ("dispatch.denied_app_tools", [])],
)
def test_a_profile_listing_no_application_may_not_write_the_projected_keys(
    lifecycle_repo: Path, no_artifact, key: str, value: Any
) -> None:
    profile = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    profile["config"][key] = value
    (lifecycle_repo / "profile.yml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )

    output = no_artifact(lifecycle_repo)

    assert key in output
    assert "projects" in output


def test_a_login_listing_no_application_may_not_write_the_projected_keys(
    lifecycle_repo: Path, no_artifact
) -> None:
    (lifecycle_repo / "tiers" / "quiet.yml").write_text(
        "name: Quiet\nconfig:\n  dispatch.denied_app_tools: []\n", encoding="utf-8"
    )

    output = no_artifact(lifecycle_repo)

    assert "dispatch.denied_app_tools" in output
    assert "tiers/quiet.yml" in output


def test_a_panel_id_spelled_under_web_panels_is_refused(lifecycle_repo: Path, no_artifact) -> None:
    _with_knob(lifecycle_repo)
    profile = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    profile["config"]["web.panels.knob.label"] = "Mine"
    (lifecycle_repo / "profile.yml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )

    output = no_artifact(lifecycle_repo)

    assert "panel.id" in output
    assert "already a web panel" in output


def test_a_pinned_port_on_a_claimed_port_is_refused_naming_both(
    lifecycle_repo: Path, no_artifact
) -> None:
    from osprey.cli.build_profile import resolve_build_document

    claimed = resolve_build_document(
        lifecycle_repo / "profile.yml", None, ()
    ).profile._claimed_ports()
    owner, port = next(iter(claimed.items()))
    knob = _knob()
    knob["containers"]["knob"]["port"] = port
    _write_app(lifecycle_repo, knob)
    _list_apps(lifecycle_repo, {"knob": {}})

    output = no_artifact(lifecycle_repo)

    assert "apps.knob.services.knob" in output
    assert owner in output
    assert str(port) in output


def test_two_applications_pinning_one_port_are_refused(lifecycle_repo: Path, no_artifact) -> None:
    for name in ("knob", "dial"):
        manifest = _knob(name=name, panel=None, tools=None)
        manifest["containers"] = {f"{name}-c": {"runtime": "osprey", "port": 18950}}
        _write_app(lifecycle_repo, manifest)
    _list_apps(lifecycle_repo, {"knob": {}, "dial": {}})

    output = no_artifact(lifecycle_repo)

    assert "apps.dial.services.dial-c" in output
    assert "apps.knob.services.knob-c" in output


def test_an_image_container_with_no_fragment_directory_is_refused(
    lifecycle_repo: Path, no_artifact
) -> None:
    manifest = _knob(panel=None, tools=None)
    manifest["containers"]["cache"] = {"image": "redis:7.2"}
    _write_app(lifecycle_repo, manifest)
    _list_apps(lifecycle_repo, {"knob": {}})

    output = no_artifact(lifecycle_repo)

    assert "containers.cache" in output
    assert "apps/knob/compose/cache" in output


def test_a_failing_vendor_fetch_refuses_before_any_artifact(
    lifecycle_repo: Path, no_artifact, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("host unreachable")

    monkeypatch.setattr("osprey.interfaces.vendor._fetch_one", unreachable)
    _with_knob(
        lifecycle_repo,
        vendor=[
            {
                "url": "https://example.invalid/knob-1.2.3.js",
                "into": "panel/vendor/knob-1.2.3.js",
                "sha256": "0" * 64,
            }
        ],
    )

    output = no_artifact(lifecycle_repo)

    assert "apps/knob" in output
    assert "example.invalid" in output
    assert "host unreachable" in output


# --- what a roster deployment renders ------------------------------------------


@pytest.fixture(scope="module")
def roster_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo = build_exemplar_repo(tmp_path_factory.mktemp("roster") / EXEMPLAR_DIRNAME)
    manifest = _knob()
    manifest["containers"]["cache"] = {"image": "redis:7.2"}
    fragment = _write_app(repo, manifest) / "compose" / "cache"
    fragment.mkdir(parents=True)
    (fragment / "docker-compose.yml.j2").write_text(
        "services:\n  cache:\n    image: redis:7.2\n", encoding="utf-8"
    )
    _list_apps(
        repo,
        {
            "knob": {
                "config": {"gain": 3},
                "services": {"knob": {"env": ["KNOB_EXTRA"]}},
            }
        },
    )
    for tier in TIERS:
        (repo / "tiers" / f"quiet-{tier}.yml").write_text(_quiet_login(tier), encoding="utf-8")
    (repo / "tiers" / "asker.yml").write_text(
        "name: Asker\nconfig:\n  control_system.writes: asked\n", encoding="utf-8"
    )
    # A shared login showing the application, which it can never arm.
    with (repo / "tiers" / "logbook.yml").open("a", encoding="utf-8") as logbook:
        logbook.write("\napps:\n  knob:\n    shown: true\n")
    # A directory the profile does not list: reported, not built.
    _write_app(
        repo,
        _knob(
            name="unlisted", panel=None, tools=None, containers={"unlisted": {"runtime": "osprey"}}
        ),
    )
    result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    repo.joinpath(".build-output").write_text(result.output, encoding="utf-8")
    return repo


def _knob_panel(render: dict[str, Any]) -> dict[str, Any] | None:
    return (render.get("web", {}).get("panels") or {}).get("knob")


@pytest.mark.parametrize("login", ["asker", "readwrite", "admin"])
def test_an_arming_login_gets_routes_token_and_write_tool(roster_repo: Path, login: str) -> None:
    render = _render(roster_repo, login)
    panel = _knob_panel(render)

    assert panel_spec_enabled(panel)
    assert panel["agent_routes"] == [GET_ROUTE, POST_ROUTE]
    assert panel["launch_token_env"] == "KNOB_LAUNCH_TOKEN"
    server = render["claude_code"]["servers"]["knob"]
    assert server["writes"] == ["knob_zero"]
    assert server["condition"] == "panel:knob"
    assert render["apps"]["knob"]["approval"]["knob_zero"]["prompt"] == "Zero the knob"
    assert WRITE_TOOL not in _denied(roster_repo, login)


def test_a_never_login_gets_the_panel_and_read_routes_only(roster_repo: Path) -> None:
    panel = _knob_panel(_render(roster_repo, "readonly"))

    assert panel_spec_enabled(panel)
    assert panel["agent_routes"] == [GET_ROUTE]
    assert "launch_token_env" not in panel
    assert WRITE_TOOL in _denied(roster_repo, "readonly")


@pytest.mark.parametrize("tier", TIERS)
def test_a_login_that_does_not_show_the_application_gets_none_of_it(
    roster_repo: Path, tier: str
) -> None:
    login = f"quiet-{tier}"
    render = _render(roster_repo, login)

    assert tier_writes(render["control_system"]) == tier
    assert not panel_spec_enabled(_knob_panel(render) or {"enabled": False})
    assert "knob" not in render["claude_code"].get("servers", {})
    assert WRITE_TOOL not in _gated(_render_dir(roster_repo, login))


def test_a_shared_login_that_shows_the_application_is_never_armed(roster_repo: Path) -> None:
    panel = _knob_panel(_render(roster_repo, "logbook"))

    assert panel_spec_enabled(panel)
    assert panel["agent_routes"] == [GET_ROUTE]
    assert "launch_token_env" not in panel


def test_the_root_of_a_roster_deployment_is_projected_too(roster_repo: Path) -> None:
    panel = _knob_panel(rendered_config(roster_repo))

    assert panel_spec_enabled(panel)
    assert "denied_app_tools" in rendered_config(roster_repo)["dispatch"]


@pytest.mark.parametrize(
    ("tier", "routes", "token"),
    [
        ("asked", [GET_ROUTE, POST_ROUTE], True),
        ("allowed", [GET_ROUTE, POST_ROUTE], True),
        ("never", [GET_ROUTE], False),
    ],
)
def test_the_root_of_a_roster_deployment_follows_its_tier(
    lifecycle_repo: Path, tier: str, routes: list[dict[str, str]], token: bool
) -> None:
    _with_knob(lifecycle_repo)
    _set_config(lifecycle_repo, "control_system.writes", tier)

    panel = _knob_panel(rendered_config(_built(lifecycle_repo)))

    assert panel["agent_routes"] == routes
    assert ("launch_token_env" in panel) is token


def test_every_render_addresses_the_panel_at_the_deployment_port(roster_repo: Path) -> None:
    root = rendered_config(roster_repo)
    port = root["services"]["knob"]["port"]

    assert root["services"]["knob"]["env"] == ["KNOB_EXTRA"]
    for login in ("readonly", "readwrite", "admin"):
        assert _knob_panel(_render(roster_repo, login))["url"] == f"http://127.0.0.1:{port}"


def test_each_projected_service_gets_its_rendered_config_file(roster_repo: Path) -> None:
    for key in ("knob", "cache"):
        written = roster_repo / "build" / "services" / key / "app-config.yml"
        assert yaml.safe_load(written.read_text(encoding="utf-8")) == {"gain": 3}
        assert rendered_config(roster_repo)["services"][key]["config_file"] == (
            f"./services/{key}/app-config.yml"
        )


def test_projected_services_are_deployed_from_their_templates(roster_repo: Path) -> None:
    root = rendered_config(roster_repo)

    assert {"knob", "cache"} <= set(root["deployed_services"])
    assert (roster_repo / "build" / "services" / "cache" / "docker-compose.yml.j2").is_file()
    assert (roster_repo / "build" / "services" / "knob" / "docker-compose.yml.j2").is_file()


def test_an_unlisted_directory_is_reported_and_not_built(roster_repo: Path) -> None:
    root = rendered_config(roster_repo)

    assert "unlisted" not in root["services"]
    assert "unlisted" not in root["deployed_services"]
    assert "apps/unlisted/" in (roster_repo / ".build-output").read_text(encoding="utf-8")


def test_a_package_directory_with_no_manifest_on_a_profile_listing_nothing_still_builds(
    lifecycle_repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    package = lifecycle_repo / "apps" / "helper"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "helper"\nversion = "0.1.0"\n', encoding="utf-8"
    )

    result = invoke_build(lifecycle_repo)

    assert result.exit_code == 0, result.output + caplog.text
    assert "apps/helper/" in result.output


# --- the root as the one login ---------------------------------------------------


@pytest.mark.parametrize("tier", ["allowed", "asked"])
def test_a_roster_less_root_that_writes_gets_the_write_tool_and_routes(
    lifecycle_repo: Path, tier: str
) -> None:
    _with_knob(_roster_less(lifecycle_repo, tier))

    render = rendered_config(_built(lifecycle_repo))

    assert _knob_panel(render)["agent_routes"] == [GET_ROUTE, POST_ROUTE]
    assert render["claude_code"]["servers"]["knob"]["writes"] == ["knob_zero"]
    gated = json_hook_config(lifecycle_repo)["facility_write_tools"]
    assert "mcp__knob__knob_zero" in gated


def test_a_roster_less_root_at_never_gets_read_routes_only(lifecycle_repo: Path) -> None:
    _with_knob(_roster_less(lifecycle_repo, "never"))

    render = rendered_config(_built(lifecycle_repo))

    assert _knob_panel(render)["agent_routes"] == [GET_ROUTE]
    assert "launch_token_env" not in _knob_panel(render)


def json_hook_config(repo: Path) -> dict[str, Any]:
    path = repo / "build" / ".claude" / "hooks" / "hook_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_and_regenerate_write_the_same_bytes(lifecycle_repo: Path) -> None:
    from osprey.cli.templates.manager import TemplateManager

    _with_knob(_roster_less(lifecycle_repo))
    build = _built(lifecycle_repo) / "build"
    hook_config = build / ".claude" / "hooks" / "hook_config.json"
    config = build / "config.yml"
    before = (hook_config.read_bytes(), config.read_bytes())
    block = yaml.safe_load(config.read_text(encoding="utf-8"))["claude_code"]["servers"]["knob"]

    TemplateManager().regenerate_claude_code(build)

    assert (hook_config.read_bytes(), config.read_bytes()) == before
    after = yaml.safe_load(config.read_text(encoding="utf-8"))["claude_code"]["servers"]["knob"]
    assert after == block


def test_a_deployment_with_no_applications_renders_no_application_key(
    lifecycle_repo: Path,
) -> None:
    render = rendered_config(_built(lifecycle_repo))

    assert "apps" not in render
    assert "denied_app_tools" not in (render.get("dispatch") or {})


def test_a_host_variant_listing_an_application_reaches_every_render(lifecycle_repo: Path) -> None:
    _write_app(lifecycle_repo, _knob())
    variants = lifecycle_repo / "profiles"
    variants.mkdir(exist_ok=True)
    (variants / "lab.yml").write_text("apps:\n  knob: {}\n", encoding="utf-8")
    (lifecycle_repo / ".env.variant").write_text("OSPREY_PROFILE_VARIANT=lab\n", encoding="utf-8")

    _built(lifecycle_repo)

    assert panel_spec_enabled(_knob_panel(rendered_config(lifecycle_repo)))
    assert panel_spec_enabled(_knob_panel(_render(lifecycle_repo, "admin")))


# --- host bindings -----------------------------------------------------------------


def test_a_host_network_runtime_container_declares_its_bind_variable(lifecycle_repo: Path) -> None:
    _with_knob(lifecycle_repo, {"services": {"knob": {"network": "host"}}})

    render = rendered_config(_built(lifecycle_repo))

    assert render["services"]["knob"]["bind_env"] == "OSPREY_APP_BIND"


def test_a_bridge_network_runtime_container_declares_nothing(lifecycle_repo: Path) -> None:
    _with_knob(lifecycle_repo)

    render = rendered_config(_built(lifecycle_repo))

    assert "bind_env" not in render["services"]["knob"]


# --- the two readings of the host ---------------------------------------------------


def test_the_scratch_reading_and_the_render_project_the_same_services(
    lifecycle_repo: Path, tmp_path: Path
) -> None:
    from dataclasses import replace

    from osprey.apps.loader import load
    from osprey.cli.build_cmd import _inject_services
    from osprey.cli.build_profile import resolve_build_document

    _with_knob(lifecycle_repo)
    render = rendered_config(_built(lifecycle_repo))
    profile = resolve_build_document(lifecycle_repo / "profile.yml", None, ()).profile

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "config.yml").write_text("services: {}\n", encoding="utf-8")
    _inject_services(
        replace(profile, deploy_services=True),
        lifecycle_repo,
        scratch,
        app_manifests=load(lifecycle_repo),
    )
    reading = yaml.safe_load((scratch / "config.yml").read_text(encoding="utf-8"))

    for key in ("port", "url", "app", "config_file", "launch_token_env"):
        assert reading["services"]["knob"][key] == render["services"]["knob"][key]


def test_an_attached_render_is_told_the_panel_port_from_the_scratch_reading(
    lifecycle_repo: Path,
) -> None:
    _with_knob(lifecycle_repo)
    text = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    (lifecycle_repo / "profile.yml").write_text(
        text.replace("deploy_services: true", "deploy_services: false"), encoding="utf-8"
    )

    render = rendered_config(_built(lifecycle_repo))

    assert "knob" not in render.get("deployed_services", [])
    assert _knob_panel(render)["url"].startswith("http://127.0.0.1:")


# --- the deploy summary ---------------------------------------------------------------


def _summary_config(*, launch_token: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {"app": "knob", "url": "http://127.0.0.1:18950"}
    if launch_token is not None:
        row["launch_token_env"] = launch_token
    return {"deployed_services": ["knob"], "services": {"knob": row}}


def test_the_summary_names_the_launch_token_variable_under_its_service() -> None:
    from osprey.deployment.deploy_summary import application_summary

    block = application_summary(_summary_config(launch_token="KNOB_LAUNCH_TOKEN"))

    assert list(block) == ["apps.knob"]
    rows = dict(block["apps.knob"])
    assert rows["knob"] == "http://127.0.0.1:18950"
    assert rows["knob launch token"] == "KNOB_LAUNCH_TOKEN (minted by osprey up into .env)"


def test_the_summary_carries_no_token_line_without_a_launch_token() -> None:
    from osprey.deployment.deploy_summary import application_summary

    block = application_summary(_summary_config(launch_token=None))

    assert [label for label, _ in block["apps.knob"]] == ["knob"]


def test_a_built_deployment_names_each_service_under_its_application(roster_repo: Path) -> None:
    from osprey.deployment.deploy_summary import application_summary

    rows = dict(application_summary(rendered_config(roster_repo))["apps.knob"])

    assert rows["knob launch token"].startswith("KNOB_LAUNCH_TOKEN ")
    assert "cache launch token" not in rows
    assert rows["cache"].startswith("http://127.0.0.1:")


def test_the_summary_is_empty_with_no_applications() -> None:
    from osprey.deployment.deploy_summary import application_summary

    assert application_summary({"deployed_services": [], "services": {"postgresql": {}}}) == {}


def test_the_text_summary_carries_the_application_block() -> None:
    from osprey.deployment.deploy_summary import format_endpoint_summary

    text = format_endpoint_summary(_summary_config(launch_token="KNOB_LAUNCH_TOKEN"), [])

    assert "apps.knob" in text
    assert "KNOB_LAUNCH_TOKEN" in text
