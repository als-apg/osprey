"""The trajectory service's registration rows, each in the table that owns it.

A service is reachable only where every table that knows about deployed
services knows about it: the Reach Contract (what an attached persona render
is told and which credential its terminal is handed), the host-port preflight
(which key moves a contested port), the deploy summary (how its address is
shown), and the shipped preset that deploys it. Each table has its own drift
test elsewhere; this module pins the trajectory rows themselves, and proves on
a real build that a persona selecting the trajectory tab gets the tab and its
terminal gets the service's launch token, while one that does not gets neither.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.deployment import deploy_summary, host_ports
from osprey.deployment.reach import REACH_CONTRACTS, dotted_get, live_consumers
from osprey.deployment.web_terminals.personas import (
    BLUESKY_PANEL_ID,
    TRAJECTORY_PANEL_ID,
    personas_declaring_panel,
    personas_declaring_trajectory_panel,
    resolve_personas,
)
from osprey.port_layout import PORT_BASE_CONFIG_KEY, SLOTS_BY_NAME, default_port

_LAUNCH_TOKEN = "TRAJECTORY_LAUNCH_TOKEN"
_PANEL_LEAVES = ("url", "path", "label", "health_endpoint")
_URL = "${TRAJECTORY_URL:-http://localhost:10082}"


def resolve_preset(name: str) -> BuildProfile:
    """A bundled preset, its ``extends`` chain merged."""
    profile, _profile_dir = resolve_build_profile(None, preset=name)
    return profile


def _panel_config(**panel) -> dict:
    return {"web": {"panels": {TRAJECTORY_PANEL_ID: panel}}}


# ---------------------------------------------------------------------------
# The Reach Contract
# ---------------------------------------------------------------------------


def test_the_contract_projects_the_panels_four_leaves():
    contract = REACH_CONTRACTS["trajectory"]
    assert contract.service == "trajectory"
    assert [p.key for p in contract.projected] == [
        f"web.panels.{TRAJECTORY_PANEL_ID}.{leaf}" for leaf in _PANEL_LEAVES
    ]
    assert {p.panel for p in contract.projected} == {TRAJECTORY_PANEL_ID}


def test_the_projection_has_the_same_shape_as_the_bluesky_sidecars():
    """Both are panel-proxied services told to a persona by their panel's leaves."""
    ours = [p.key.rsplit(".", 1)[1] for p in REACH_CONTRACTS["trajectory"].projected]
    theirs = [p.key.rsplit(".", 1)[1] for p in REACH_CONTRACTS["bluesky_web"].projected]
    assert ours == theirs


def test_the_launch_token_is_granted_on_the_panel_declaration():
    (grant,) = REACH_CONTRACTS["trajectory"].credentials
    assert grant.env == _LAUNCH_TOKEN
    assert grant.gate is not None
    assert grant.gate(_panel_config(url=_URL, enabled=True)) is True
    assert grant.gate(_panel_config(url=_URL, enabled=False)) is False
    assert grant.gate({"web": {"panels": {BLUESKY_PANEL_ID: {"url": "x"}}}}) is False
    assert grant.gate({}) is False


def test_a_declared_panel_is_a_live_consumer_that_resolves_its_url():
    config = _panel_config(url=_URL, path="/panel/", enabled=True)
    live = [(c.service, consumer) for c, consumer in live_consumers(config)]
    ours = [consumer for service, consumer in live if service == "trajectory"]
    assert len(ours) == 1
    assert ours[0].resolves(config)
    assert not ours[0].resolves(_panel_config(enabled=True))


def test_a_pinned_url_names_an_external_service():
    contract = REACH_CONTRACTS["trajectory"]
    assert contract.names_external is not None
    assert contract.names_external(_panel_config(url=_URL))
    assert not contract.names_external({})


# ---------------------------------------------------------------------------
# Host ports and the deploy summary
# ---------------------------------------------------------------------------


def test_the_host_port_remedy_and_layout_slot():
    assert host_ports._SERVICE_REMEDY_KEYS["trajectory"] == "services.trajectory.port"
    assert host_ports._SERVICE_LAYOUT_SLOTS["trajectory"] == ("trajectory",)
    assert SLOTS_BY_NAME["trajectory"].config_key == "services.trajectory.port"


@pytest.mark.parametrize("base", [10000, 23000])
def test_a_host_network_binding_on_its_slot_moves_with_the_block(base):
    """The preset runs the service on the host network, where it publishes no
    ``ports:`` and the binding is derived from the rendered config."""
    port = default_port("trajectory", base=base)
    config = {
        "deployment": {"port_base": base},
        "services": {"trajectory": {"network": "host", "port": port}},
    }
    (binding,) = [
        b for b in host_ports.derive_host_network_bindings(config) if b.service == "trajectory"
    ]
    assert binding.host_network is True
    assert binding.host_ip == "127.0.0.1"
    assert binding.host_port == port
    assert host_ports._remedy_for_service("trajectory", None, port, base) == PORT_BASE_CONFIG_KEY


def test_a_hand_placed_port_names_the_services_own_key():
    base = 10000
    assert (
        host_ports._remedy_for_service("trajectory", None, base + 999, base)
        == "services.trajectory.port"
    )


def test_the_deploy_summary_shows_it_as_an_http_service_in_its_slot():
    assert "trajectory" in deploy_summary._HTTP_SERVICES
    assert deploy_summary._SERVICE_SLOTS["trajectory"] == "trajectory"


# ---------------------------------------------------------------------------
# The shipped preset
# ---------------------------------------------------------------------------


def test_the_preset_deploys_the_service_on_the_host_network():
    resolved = resolve_preset("control-assistant")
    assert resolved.trajectory is not None
    assert resolved.trajectory.network == "host"


def test_the_base_never_selects_the_write_panel():
    """The read-only login is built from the base, so a write tab stays off it."""
    resolved = resolve_preset("control-assistant")
    assert TRAJECTORY_PANEL_ID not in (resolved.web_panels or [])
    assert BLUESKY_PANEL_ID not in (resolved.web_panels or [])


# ---------------------------------------------------------------------------
# A real build whose write personas select the tab beside BLUESKY
# ---------------------------------------------------------------------------


def _select_trajectory_beside_bluesky(persona_file: Path) -> bool:
    """Add ``trajectory`` under ``bluesky`` in a persona's ``web_panels:``."""
    text = persona_file.read_text(encoding="utf-8")
    if not re.search(r"^  - bluesky\b", text, re.MULTILINE):
        return False
    if re.search(r"^  - trajectory\b", text, re.MULTILINE):
        return True
    text = re.sub(r"^(  - bluesky\b[^\n]*)$", r"\1\n  - trajectory", text, count=1, flags=re.M)
    persona_file.write_text(text, encoding="utf-8")
    return True


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory) -> Path:
    """``osprey init`` the hosting preset, select the tab where BLUESKY is, build."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    repo = tmp_path_factory.mktemp("trajectory-registration") / "my-facility"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    selected = [
        _select_trajectory_beside_bluesky(p) for p in sorted((repo / "personas").glob("*.yml"))
    ]
    assert any(selected)
    built = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return repo


@pytest.fixture(scope="module")
def host_config(built_repo: Path) -> dict:
    return yaml.safe_load((built_repo / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(host_config: dict) -> list[dict]:
    return resolve_personas(
        host_config["modules"]["web_terminals"], host_config.get("registry") or {}, strict=True
    )


def _persona_config(repo: Path, entry: dict) -> dict:
    return yaml.safe_load(
        (repo / "build" / entry["project"] / "config.yml").read_text(encoding="utf-8")
    )


def test_the_host_deploys_the_service_and_does_not_show_the_tab(host_config):
    assert "trajectory" in host_config["deployed_services"]
    assert host_config["services"]["trajectory"]["network"] == "host"
    assert host_config["web"]["panels"][TRAJECTORY_PANEL_ID].get("enabled") is False


def test_exactly_the_bluesky_personas_are_named_by_the_predicate(built_repo, host_config):
    named = personas_declaring_trajectory_panel(host_config, built_repo)
    bluesky = personas_declaring_panel(host_config, built_repo, BLUESKY_PANEL_ID)
    assert named, "no persona selects the tab — the fixture lost its precondition"
    assert named == bluesky


def test_a_selecting_render_carries_the_enabled_tab_told_by_its_host(
    built_repo, host_config, entries
):
    named = personas_declaring_trajectory_panel(host_config, built_repo)
    checked = 0
    for entry in entries:
        config = _persona_config(built_repo, entry)
        panel = dotted_get(config, f"web.panels.{TRAJECTORY_PANEL_ID}")
        if entry["persona"] in named:
            checked += 1
            assert panel["enabled"] is True, entry["persona"]
            for leaf in _PANEL_LEAVES:
                assert panel[leaf] == host_config["web"]["panels"][TRAJECTORY_PANEL_ID][leaf]
        else:
            assert not (panel or {}).get("enabled"), entry["persona"]
    assert checked


def test_the_web_terminal_render_hands_the_token_to_every_named_persona(
    built_repo, host_config, entries
):
    """Multi-user reach: through the artifacts seam the deploy renders with."""
    from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
    from osprey.deployment.web_terminals.render import render_web_terminals

    inputs = resolve_render_inputs(host_config, built_repo)
    named = personas_declaring_trajectory_panel(host_config, built_repo)
    assert inputs["trajectory_personas"] == named

    compose = yaml.safe_load(render_web_terminals(host_config, **inputs)["docker-compose.web.yml"])
    granted, withheld = 0, 0
    for entry in entries:
        environment = [
            str(line) for line in compose["services"][f"web-{entry['name']}"]["environment"]
        ]
        lines = [line for line in environment if line.partition("=")[0] == _LAUNCH_TOKEN]
        if entry["persona"] in named:
            granted += 1
            assert lines == [f"{_LAUNCH_TOKEN}=${{{_LAUNCH_TOKEN}:-}}"], entry["name"]
        else:
            withheld += 1
            assert lines == [], entry["name"]
    assert granted and withheld


def test_the_service_admits_every_named_users_own_secret(built_repo, host_config, entries):
    """The reverse grant, on the file ``osprey build`` wrote: a user whose
    terminal holds the token is also one the service's gate names."""
    from osprey.deployment.web_terminals.render import terminal_secret_env_var

    named = personas_declaring_trajectory_panel(host_config, built_repo)
    compose = yaml.safe_load(
        (built_repo / "build" / "services" / "trajectory" / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    environment = compose["services"]["trajectory"]["environment"]
    for entry in entries:
        var = terminal_secret_env_var(entry["name"])
        assert (var in environment) == (entry["persona"] in named), entry["name"]
