"""The Reach Contract, enforced from its one registry.

:mod:`osprey.deployment.reach` declares, per shared service, which in-container
consumers dial it, what the build projects into an attached render, which
credential each entitled container receives and which host directories it is
handed. Two kinds of test here read that declaration and nothing else:

* **Completeness.** Every service a shipped template can deploy — the app
  templates' ``services:`` blocks and every ``templates/services/<name>``
  directory the injectors copy — has a contract, or a contract that says why
  nothing in a container dials it. A service added without one fails here,
  which is the point: the registry is only a single source of truth while it
  is complete.
* **The seams, on a real built stack.** For every persona render the
  control-assistant preset builds: each consumer the render switches on
  resolves an endpoint; each credential its gate grants is a line in that
  persona's compose ``environment:``; each shared path its gate entitles is a
  mount in that persona's compose ``volumes:``. One walk over the registry,
  so a grant added to the registry without a compose line — or the reverse —
  is visible here rather than inside a container at first use.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from osprey.bluesky_bridge_connection import LANE_KEYS, SECOND_LANE_KEYS
from osprey.deployment.reach import (
    REACH_CONTRACTS,
    SHARED_PATHS,
    dotted_get,
    live_consumers,
    reach_errors,
)
from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.personas import resolve_personas
from osprey.deployment.web_terminals.render import render_web_terminals
from tests.cli.test_persona_presets import _build_persona_stack

pytestmark = pytest.mark.slow

_SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"
_TEMPLATES = _SRC / "templates"


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def _app_template_services() -> set[str]:
    """Every ``services.<name>`` a shipped app template renders."""
    names: set[str] = set()
    for template in (_TEMPLATES / "apps").glob("*/config.yml.j2"):
        text = template.read_text(encoding="utf-8")
        match = re.search(r"^services:\n((?:  .*\n|\n)*)", text, re.MULTILINE)
        if match is None:
            continue
        names.update(re.findall(r"^  ([a-z_]+):", match.group(1), re.MULTILINE))
    return names


def _service_template_dirs() -> set[str]:
    """Every service the injectors can copy into a project."""
    return {
        path.name
        for path in (_TEMPLATES / "services").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }


def _injected_services() -> set[str]:
    """Services the build injects that have no template of their own.

    A second plan lane (``bluesky.second_lane``) is written by the bluesky
    injector beside lane 1's block, reusing lane 1's service template — so
    neither scan above sees it, and without this it could never be missing.
    """
    return set(SECOND_LANE_KEYS.values())


def _deployable_services() -> set[str]:
    return _app_template_services() | _service_template_dirs() | _injected_services()


def test_every_deployable_service_has_a_contract():
    deployable = _deployable_services()
    assert deployable, "found no deployable services — the template scan is broken"
    missing = sorted(deployable - set(REACH_CONTRACTS))
    assert missing == [], (
        f"services with no Reach Contract: {missing}. Add a ReachContract to "
        f"osprey.deployment.reach — with its consumers and projected keys, or with "
        f"no_client_reach=True and a note saying why nothing in a container dials it."
    )


def test_every_contract_names_a_deployable_service():
    deployable = _deployable_services()
    stale = sorted(set(REACH_CONTRACTS) - deployable)
    assert stale == [], f"contracts for services no template deploys: {stale}"


def test_every_contract_says_how_it_is_reached():
    """A contract either has consumers and projected keys, is derived on another
    build path, or says nothing dials it — never silent on all three."""
    for name, contract in REACH_CONTRACTS.items():
        assert contract.service == name
        assert contract.note, f"{name}: a contract needs a one-line note"
        if contract.no_client_reach:
            assert not contract.consumers and not contract.projected, name
        elif contract.derived_by:
            assert not contract.projected, f"{name}: derived elsewhere, projects nothing here"
        else:
            assert contract.consumers, f"{name}: no consumer and no no_client_reach marker"
            assert contract.projected, f"{name}: consumers but nothing projected for them"


def test_every_projected_key_lives_under_a_known_prefix():
    for contract in REACH_CONTRACTS.values():
        for projected in contract.projected:
            if projected.panel is not None:
                assert projected.key.startswith(f"web.panels.{projected.panel}."), projected
            else:
                assert projected.key.startswith(f"services.{contract.service}."), projected


# ---------------------------------------------------------------------------
# The seams, on a real built stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_stack(tmp_path_factory) -> Path:
    return _build_persona_stack(tmp_path_factory.mktemp("reach-contract") / "my-facility")


@pytest.fixture(scope="module")
def host_config(built_stack: Path) -> dict:
    return yaml.safe_load((built_stack / "build" / "config.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(host_config: dict) -> list[dict]:
    return resolve_personas(
        host_config["modules"]["web_terminals"],
        host_config.get("registry") or {},
        (host_config.get("facility") or {}).get("prefix") or "",
        strict=True,
    )


@pytest.fixture(scope="module")
def web_compose(built_stack: Path, host_config: dict) -> dict:
    artifacts = render_web_terminals(host_config, **resolve_render_inputs(host_config, built_stack))
    return yaml.safe_load(artifacts["docker-compose.web.yml"])


def _persona_config(repo: Path, entry: dict) -> dict:
    return yaml.safe_load(
        (repo / "build" / entry["project"] / "config.yml").read_text(encoding="utf-8")
    )


def _env_names(service: dict) -> set[str]:
    names: set[str] = set()
    for line in service.get("environment") or []:
        names.add(str(line).partition("=")[0].strip())
    return names


def _mount_targets(service: dict) -> list[str]:
    targets = []
    for entry in service.get("volumes") or []:
        parts = str(entry).split(":")
        if len(parts) >= 2:
            targets.append(parts[1])
    return targets


def test_the_built_stack_refuses_nothing(built_stack, host_config, entries):
    """Every render the preset builds — the host's and each persona's — has an
    endpoint for every consumer it switches on."""
    assert reach_errors(host_config) == []
    for entry in entries:
        assert reach_errors(_persona_config(built_stack, entry)) == [], entry["persona"]


def test_every_live_consumer_in_every_persona_resolves(built_stack, entries):
    checked = 0
    for entry in entries:
        config = _persona_config(built_stack, entry)
        for contract, consumer in live_consumers(config):
            checked += 1
            assert consumer.resolves(config), (
                f"persona {entry['persona']!r}: {consumer.name} is on "
                f"({consumer.switch_key}) but resolves nothing for services.{contract.service}"
            )
    assert checked, "no persona switched any consumer on — the fixture lost its preconditions"


def test_every_projected_fact_matches_the_hosts(built_stack, host_config, entries):
    """What a persona was told is what the host render says — value for value."""
    checked = 0
    for entry in entries:
        config = _persona_config(built_stack, entry)
        for contract in REACH_CONTRACTS.values():
            for projected in contract.projected:
                told = dotted_get(config, projected.key)
                if told is None:
                    continue
                checked += 1
                assert told == dotted_get(host_config, projected.key), (
                    f"persona {entry['persona']!r}: {projected.key} is {told!r}, "
                    f"host says {dotted_get(host_config, projected.key)!r}"
                )
    assert checked


def test_every_granted_credential_is_in_the_personas_compose_block(
    built_stack, web_compose, entries
):
    """switch on ⇒ the credential's env line is on that persona's service, and
    a credential the gate withholds is NOT — the tier boundary, registry-side."""
    checked = 0
    for entry in entries:
        config = _persona_config(built_stack, entry)
        env = _env_names(web_compose["services"][f"web-{entry['name']}"])
        for contract in REACH_CONTRACTS.values():
            for grant in contract.credentials:
                entitled = grant.gate is None or grant.gate(config)
                checked += 1
                assert (grant.env in env) == entitled, (
                    f"persona {entry['persona']!r}: {grant.env} "
                    f"{'missing from' if entitled else 'leaked into'} web-{entry['name']} "
                    f"(services.{contract.service})"
                )
    assert checked


def test_every_entitled_shared_path_is_mounted(built_stack, web_compose, entries):
    """gate ⇒ a mount whose target ends with the configured relative path."""
    checked = 0
    for entry in entries:
        config = _persona_config(built_stack, entry)
        targets = _mount_targets(web_compose["services"][f"web-{entry['name']}"])
        for shared in SHARED_PATHS:
            raw = dotted_get(config, shared.config_key) or dotted_get(
                config, shared.config_key.replace(".mirror_path", ".settings.mirror_path")
            )
            if not shared.gate(config):
                continue
            checked += 1
            assert raw, f"{shared.config_key} entitles but names nothing"
            assert any(t.endswith(f"/{str(raw).strip()}") for t in targets), (
                f"persona {entry['persona']!r} is entitled to {shared.describe} "
                f"({shared.config_key}={raw!r}) but web-{entry['name']} mounts it nowhere: {targets}"
            )
    assert checked


def test_the_bluesky_web_sidecar_accepts_every_entitled_users_secret(
    built_stack, host_config, web_compose, entries, monkeypatch
):
    """A persona proxies its BLUESKY tab into the sidecar with the secret ITS
    container holds — its own, under the fixed ``OSPREY_TERMINAL_SECRET``.
    The sidecar's OWN compose file (the services stack — the web overlay is a
    separate single-file compose project that could never merge into it)
    lists every entitled user's variable beside the accept flag the web gate
    requires; a user whose persona shows no BLUESKY tab gets no key to it.
    """
    from osprey.deployment.compose_generator import prepare_compose_files
    from osprey.deployment.web_terminals.personas import config_declares_bluesky_panel
    from osprey.deployment.web_terminals.render import terminal_secret_env_var
    from osprey.interfaces.web_auth import ROSTER_ACCEPT_ENV

    assert "bluesky_web" in host_config["deployed_services"]
    # The deploy-faithful render: `osprey up` re-renders every services compose
    # file from the repo root with the persona projects in place. The build's
    # own render precedes the persona renders, so it is the deploy's that
    # carries the roster grant (see compose_generator._bluesky_panel_secret_env_vars).
    monkeypatch.chdir(built_stack / "build")
    _, compose_files = prepare_compose_files(str(built_stack / "build" / "config.yml"))
    (sidecar_path,) = [f for f in compose_files if f.endswith("/bluesky_web/docker-compose.yml")]
    sidecar_compose = yaml.safe_load(Path(sidecar_path).read_text(encoding="utf-8"))
    environment = sidecar_compose["services"]["bluesky-web"]["environment"]
    assert isinstance(environment, dict)
    assert environment.get(ROSTER_ACCEPT_ENV) == "1"
    handed = set(environment)
    # The overlay carries no such fragment any more: as a separate compose
    # project it would fail `osprey up` with an image-less service.
    assert "bluesky-web" not in web_compose["services"]

    checked = 0
    for entry in entries:
        var = terminal_secret_env_var(entry["name"])
        presented = f"OSPREY_TERMINAL_SECRET=${{{var}:-}}"
        assert presented in (web_compose["services"][f"web-{entry['name']}"]["environment"]), (
            f"web-{entry['name']} presents {var}"
        )
        entitled = config_declares_bluesky_panel(_persona_config(built_stack, entry))
        checked += entitled
        assert (var in handed) == entitled, (
            f"{var} {'missing from' if entitled else 'leaked into'} the bluesky-web sidecar"
        )
        if entitled:
            assert environment[var] == f"${{{var}:-}}"
    assert checked


# ---------------------------------------------------------------------------
# The refusal, on a synthetic render
# ---------------------------------------------------------------------------


def test_a_consumer_switched_on_with_nothing_to_dial_is_refused():
    """The state the whole contract exists to catch: hybrid logbook search
    switched on over ``services: {}``. The error names the switch that turned
    the consumer on and the projected key that would give it an endpoint, so
    the operator can act on either end."""
    config = {"ariel": {"search_modules": {"hybrid": {"enabled": True}}}}

    (error,) = reach_errors(config)

    assert "ARIEL hybrid search" in error
    assert "ariel.search_modules.hybrid.enabled" in error
    assert "services.qmd.port" in error


def test_a_degrading_consumer_is_not_refused():
    """The OKF panel's ranked search falls back to substring matching without
    a sidecar, by design — its contract says ``refuse=False``, so the same
    unresolved state that refuses hybrid search builds cleanly here (the
    ``reach`` health category still reports it)."""
    config = {
        "web": {"panels": {"okf": {"enabled": True}}},
        "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
    }

    live = [consumer.name for _, consumer in live_consumers(config)]
    assert "OKF panel ranked search" in live
    assert reach_errors(config) == []


def test_a_render_with_no_live_consumer_is_refused_nothing():
    assert reach_errors({}) == []


def test_bluesky_panel_secret_vars_follow_each_users_own_project(tmp_path):
    """The roster grant is per USER: a persona user by their persona's rendered
    config, a persona-less user by the deploy config they run (the same rule
    the web-terminal render grants every other credential by), in roster
    order."""
    from osprey.deployment.web_terminals.personas import bluesky_panel_secret_env_vars
    from osprey.deployment.web_terminals.render import terminal_secret_env_var

    for persona, declares in (("viewer", False), ("operator", True)):
        project = tmp_path / "build" / f"demo-{persona}"
        project.mkdir(parents=True)
        panels = {"bluesky": {"url": "http://localhost:8095"}} if declares else {}
        (project / "config.yml").write_text(
            yaml.safe_dump({"web": {"panels": panels}}), encoding="utf-8"
        )
    config = {
        "web": {"panels": {"bluesky": {"url": "http://localhost:8095"}}},
        "modules": {
            "web_terminals": {
                "personas": {
                    "viewer": {"project_path": "build/demo-viewer"},
                    "operator": {"project_path": "build/demo-operator"},
                },
                # Object entries carry the frozen `index` every materialized
                # roster has; a bare string is the legacy spelling.
                "users": [
                    {"name": "alice", "index": 0, "persona": "operator"},
                    {"name": "bob", "index": 1, "persona": "viewer"},
                    "carol",  # no persona: runs the deploy config, which shows the tab
                ],
            }
        },
    }

    assert bluesky_panel_secret_env_vars(config, tmp_path) == [
        terminal_secret_env_var("alice"),
        terminal_secret_env_var("carol"),
    ]

    config["web"]["panels"] = {}
    assert bluesky_panel_secret_env_vars(config, tmp_path) == [terminal_secret_env_var("alice")]


def test_the_graph_channel_finder_is_a_store_consumer_of_its_own():
    """The graph paradigm's channel finder dials the store through its own
    server, whether or not the `graph` MCP server is switched on — so a
    persona that switches that server off but keeps the channel finder is
    still projected the store's address, and refused without one."""
    from osprey.deployment.reach import project_attached_overrides

    attached = {
        "channel_finder": {"pipeline_mode": "graph"},
        "claude_code": {"servers": {"graph": {"enabled": False}}},
    }
    host = {"services": {"graphdb": {"port_host": 17687, "uri": "bolt://localhost:17687"}}}

    projected = project_attached_overrides(host, attached)
    assert projected["services.graphdb.port_host"] == 17687

    (error,) = reach_errors(attached)
    assert "graph channel finder" in error
    assert "channel_finder.pipeline_mode" in error

    # Channel finder switched off too: no consumer, nothing projected.
    off = {
        **attached,
        "claude_code": {
            "servers": {"graph": {"enabled": False}, "channel-finder": {"enabled": False}}
        },
    }
    assert project_attached_overrides(host, off) == {}
    assert reach_errors(off) == []


# ---------------------------------------------------------------------------
# Each plan lane's launch token, on a synthetic two-lane render
# ---------------------------------------------------------------------------


def _launch_token_grants(config: dict) -> dict[str, bool]:
    """Each lane's launch-token grant, as the registry's own gate answers it.

    Read off the contracts rather than restated, so a lane whose grant is
    rewired to some other posture shows up here as the wrong answer.
    """
    grants: dict[str, bool] = {}
    for lane in LANE_KEYS:
        for grant in REACH_CONTRACTS[lane].credentials:
            grants[grant.env] = grant.gate(config)
    return grants


def test_a_lanes_launch_token_follows_that_lanes_own_target():
    """The per-target boundary, lane by lane: a deployment built for a live
    machine, arming writes on its virtual-accelerator lane alone, hands out the
    VA lane's token and withholds the live lane's — even though the two lanes
    run in one container for one persona."""
    config = {
        "claude_code": {"servers": {"bluesky": {"enabled": True}}},
        "control_system": {
            "type": "epics",
            "writes_enabled": False,
            "connector": {"virtual_accelerator": {"writes_enabled": True}},
        },
        "services": {
            "bluesky": {"port": 8090, "target": "live"},
            "bluesky_va": {"port": 8091, "target": "va"},
        },
    }

    assert _launch_token_grants(config) == {
        "BLUESKY_LAUNCH_TOKEN": False,
        "BLUESKY_VA_LAUNCH_TOKEN": True,
        # Lane 1 serves this deployment's live target; there is no second live
        # lane to arm.
        "BLUESKY_LIVE_LAUNCH_TOKEN": False,
    }


def test_a_global_true_does_not_arm_a_lane_whose_own_block_says_false():
    """The mirror deployment — built for the simulator, told about its real
    machine by a connector block that disarms it. The deployment-wide key is
    the VA lane's posture by inheritance, and says nothing about the live lane,
    whose own block has already answered."""
    config = {
        "claude_code": {"servers": {"bluesky": {"enabled": True}}},
        "control_system": {
            "type": "virtual_accelerator",
            "writes_enabled": True,
            "connector": {"epics": {"writes_enabled": False}},
        },
        "services": {
            "bluesky": {"port": 8090, "target": "va"},
            "bluesky_live": {"port": 8091, "target": "live"},
        },
    }

    assert _launch_token_grants(config) == {
        "BLUESKY_LAUNCH_TOKEN": True,
        "BLUESKY_LIVE_LAUNCH_TOKEN": False,
        "BLUESKY_VA_LAUNCH_TOKEN": False,
    }


def test_no_lane_is_armed_without_the_bluesky_server():
    """A token for a server that never starts arms nothing while still handing
    the agent a live credential — so the server's own switch, whose registry
    default is off, gates every lane's grant."""
    config = {
        "control_system": {"type": "virtual_accelerator", "writes_enabled": True},
        "services": {
            "bluesky": {"port": 8090, "target": "va"},
            "bluesky_live": {"port": 8091, "target": "live"},
        },
    }

    assert set(_launch_token_grants(config).values()) == {False}

    config["claude_code"] = {"servers": {"bluesky": {"enabled": True}}}
    assert _launch_token_grants(config)["BLUESKY_LAUNCH_TOKEN"] is True
