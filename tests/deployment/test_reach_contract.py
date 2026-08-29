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
    project_attached_overrides,
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

    The live stand-in (``virtual_accelerator.live_standin``) is the same case:
    the VA injector writes a second ``services.live_standin`` block beside
    ``services.virtual_accelerator``, and the second container renders from the
    virtual_accelerator template, so no scan above names it either.
    """
    return set(SECOND_LANE_KEYS.values()) | {"live_standin"}


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
# The live stand-in, whose port every render is told
# ---------------------------------------------------------------------------

_STANDIN_PORT = 5074


def _standin_config(port: int = _STANDIN_PORT) -> dict:
    """A render shaped the way the build leaves one that stood a stand-in up.

    The gateways are the overwrite the VA injector performs: the ``epics``
    connector — the live target — dials the second virtual accelerator on
    loopback instead of the facility's gateway.
    """
    config: dict = {
        "control_system": {
            "type": "epics",
            "connector": {
                "epics": {
                    "gateways": {
                        "read_only": {
                            "address": "localhost",
                            "port": port,
                            "use_name_server": True,
                        },
                        "write_access": {
                            "address": "localhost",
                            "port": port,
                            "use_name_server": True,
                        },
                    }
                }
            },
        },
        "services": {
            "live_standin": {"path": "./services/virtual_accelerator", "port": port},
        },
    }
    return config


def _standin_contract():
    return REACH_CONTRACTS["live_standin"]


def test_the_live_standin_port_is_projected_ungated():
    """Every render is told the port, because the LABEL reads it.

    ``osprey_connectors.standin.live_standin_active`` decides from this one key
    whether an operator is shown ``LIVE MACHINE`` or ``LIVE MACHINE (stand-in)``,
    and a persona render carries no ``services:`` block of its own. A gate here
    would label the same machine two different ways depending on whether it was
    seen through a persona.
    """
    projected = _standin_contract().projected
    assert [key.key for key in projected] == ["services.live_standin.port"]
    assert projected[0].gate is None
    assert projected[0].panel is None


def test_the_standin_port_reaches_a_render_with_no_epics_connector():
    """The ungated projection, exercised through the build's own function.

    An attached project built from a template with no ``epics`` connector
    (hello_world, ariel_standalone) has no client for the stand-in and is still
    told where it is: its roster labels the same machine as everyone else's.
    """
    attached = {"services": {}}
    overrides = project_attached_overrides(_standin_config(), attached)

    assert overrides["services.live_standin.port"] == _STANDIN_PORT
    # ... and that render is not refused for a consumer it does not have.
    assert (
        reach_errors({**attached, **{"services": {"live_standin": {"port": _STANDIN_PORT}}}}) == []
    )


def test_a_host_without_a_standin_projects_nothing():
    host = _standin_config()
    del host["services"]["live_standin"]

    assert project_attached_overrides(host, {"services": {}}) == {}


def test_the_standin_consumer_dials_the_epics_read_only_gateway():
    contract = _standin_contract()
    (consumer,) = contract.consumers
    config = _standin_config()

    assert consumer.is_on(config)
    assert consumer.resolves(config)
    assert consumer.dial is not None
    assert consumer.dial(config) == ("localhost", _STANDIN_PORT)
    assert reach_errors(config) == []


def test_the_standin_consumer_has_nothing_to_dial_without_gateways():
    config = _standin_config()
    del config["control_system"]["connector"]["epics"]["gateways"]
    (consumer,) = _standin_contract().consumers

    assert consumer.dial is not None
    assert consumer.dial(config) is None
    # On with nothing to dial is the state the build refuses: the stand-in was
    # stood up and the connector that reaches it was left pointing nowhere.
    assert consumer.is_on(config) and not consumer.resolves(config)
    assert any("live stand-in" in error for error in reach_errors(config))


def test_a_deployment_with_no_standin_switches_the_consumer_off():
    config = _standin_config()
    del config["services"]["live_standin"]
    (consumer,) = _standin_contract().consumers

    assert not consumer.is_on(config)
    assert reach_errors(config) == []


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


def test_every_entitled_shared_path_in_every_persona_resolves(built_stack, entries):
    """The preset ships its bundle and the build anchors every render on the
    repo, so what each persona is entitled to is on the host it was built on."""
    checked = 0
    for entry in entries:
        config = _persona_config(built_stack, entry)
        for shared in SHARED_PATHS:
            if not shared.gate(config):
                continue
            checked += 1
            assert shared.unresolved(config, built_stack) is None, (
                f"persona {entry['persona']!r} is entitled to {shared.describe} "
                f"({shared.config_key}) but {shared.unresolved(config, built_stack)}"
            )
    assert checked


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


def test_a_degrading_consumer_is_not_refused(tmp_path):
    """The OKF panel's ranked search falls back to substring matching without
    a sidecar, by design — its contract says ``refuse=False``, so the same
    unresolved state that refuses hybrid search builds cleanly here (the
    ``reach`` health category still reports it)."""
    (tmp_path / "data" / "facility_knowledge").mkdir(parents=True)
    config = {
        "web": {"panels": {"okf": {"enabled": True}}},
        "facility_knowledge": {"bundle_path": "data/facility_knowledge"},
    }

    live = [consumer.name for _, consumer in live_consumers(config)]
    assert "OKF panel ranked search" in live
    assert reach_errors(config, repo_root=tmp_path) == []


def test_a_render_with_no_live_consumer_is_refused_nothing():
    assert reach_errors({}) == []


# A shared path is the directory-shaped half of the same contract: a render
# entitled to a host directory whose bind source is not there would have the
# container runtime create it (root-owned under a rootful daemon) or read an
# empty one the deploy provisioned on the spot. Anchored on the repo root the
# build passes, exactly as the bind source is.


def _entitled_to_bundle(bundle_path: str) -> dict:
    return {"facility_knowledge": {"bundle_path": bundle_path}}


def _entitled_to_mirror(mirror_path: str) -> dict:
    return {
        "ariel": {
            "enhancement_modules": {
                "qmd_export": {"enabled": True, "settings": {"mirror_path": mirror_path}}
            }
        }
    }


def test_every_shared_path_says_where_it_binds():
    """The registry is complete only while every entry can be resolved to a
    host directory and says whether the deploy provisions it."""
    for shared in SHARED_PATHS:
        assert callable(shared.host_dir), shared.config_key
        assert isinstance(shared.provisioned, bool), shared.config_key
        assert shared.describe, shared.config_key


def test_a_bundle_that_is_not_on_the_host_is_refused(tmp_path):
    """Authored content: nothing in the deploy fills it, so a key naming a
    directory that is not there is a typo or a bundle that was never put in
    place — refused at build time, naming the key, rather than bound empty."""
    (error,) = reach_errors(_entitled_to_bundle("data/facility_knowledge"), repo_root=tmp_path)

    assert "facility_knowledge.bundle_path" in error
    assert str(tmp_path / "data" / "facility_knowledge") in error


def test_a_bundle_on_the_host_is_not_refused(tmp_path):
    (tmp_path / "data" / "facility_knowledge").mkdir(parents=True)

    assert reach_errors(_entitled_to_bundle("data/facility_knowledge"), repo_root=tmp_path) == []


def test_a_bundle_path_naming_a_file_is_refused(tmp_path):
    (tmp_path / "bundle.tar").write_text("")

    (error,) = reach_errors(_entitled_to_bundle("bundle.tar"), repo_root=tmp_path)

    assert "not a directory" in error
    assert "facility_knowledge.bundle_path" in error


def test_an_absolute_bundle_path_is_read_as_given(tmp_path):
    """Operator-owned, outside the repo, not re-anchored — the same rule the
    renderers apply to the mount source."""
    elsewhere = tmp_path / "srv" / "okf"
    elsewhere.mkdir(parents=True)

    assert reach_errors(_entitled_to_bundle(str(elsewhere)), repo_root=tmp_path / "repo") == []
    (error,) = reach_errors(
        _entitled_to_bundle(str(tmp_path / "srv" / "okf-typo")), repo_root=tmp_path
    )
    assert str(tmp_path / "srv" / "okf-typo") in error


def test_a_render_naming_no_bundle_is_entitled_to_none(tmp_path):
    """No key, no entitlement, nothing to refuse — a dispatch worker mounts no
    bundle and must not be refused over a directory it never binds."""
    assert reach_errors({"facility_knowledge": {}}, repo_root=tmp_path) == []


def test_a_mirror_the_deploy_will_provision_is_not_refused(tmp_path):
    """A writer's output: the deploy creates it before the first bind
    (``ensure_shared_corpus_dir``), so a mirror that is not there yet is the
    ordinary first-deploy state."""
    assert reach_errors(_entitled_to_mirror("var/ariel_mirror"), repo_root=tmp_path) == []


def test_a_mirror_the_deploy_cannot_create_is_refused(tmp_path):
    """The one mirror state that ends root-owned: a path the deploy's mkdir
    would fail on, because what stands where its parent should be is not a
    writable directory."""
    (tmp_path / "blocker").write_text("")

    (error,) = reach_errors(_entitled_to_mirror("blocker/ariel_mirror"), repo_root=tmp_path)

    assert "ariel.enhancement_modules.qmd_export.mirror_path" in error
    assert "cannot create" in error
    assert str(tmp_path / "blocker") in error


def test_a_disabled_export_is_entitled_to_no_mirror(tmp_path):
    config = _entitled_to_mirror("blocker/ariel_mirror")
    config["ariel"]["enhancement_modules"]["qmd_export"]["enabled"] = False
    (tmp_path / "blocker").write_text("")

    assert reach_errors(config, repo_root=tmp_path) == []


# A deploying render — `deployed_services` non-empty — is the other side of the
# same contract. Its consumers' clients dial the port the deployment's OWN
# service publishes on loopback, and a consumer whose resolver always answers
# (a compiled-in default) cannot be refused for lack of an endpoint. It has to
# be refused for the deployment not running the service that endpoint implies.


def test_a_deploying_render_refuses_a_consumer_of_a_service_it_does_not_deploy():
    """The `bluesky: null` hole: removing the Bluesky stack from a profile is a
    two-key edit (the ``bluesky:`` block that injects the service, and
    ``claude_code.servers.bluesky.enabled`` that switches its consumer on), and
    a profile that removed only the block rendered clean — the bluesky MCP
    server then dialed 127.0.0.1:8090 at first use and found nothing."""
    config = {
        "claude_code": {"servers": {"bluesky": {"enabled": True}}},
        "deployed_services": ["postgresql"],
    }

    (error,) = reach_errors(config)

    assert "bluesky MCP server" in error
    assert "claude_code.servers.bluesky.enabled" in error
    assert "`bluesky`" in error
    assert "deployed_services" in error


def test_every_always_resolving_consumer_is_refused_the_same_way():
    """Each consumer whose client resolver always answers from a shipped
    default is refused on a deploying render that does not run its service —
    not only the bluesky one the hole was found in."""
    cases = {
        "postgresql": {"ariel": {"search_modules": {}}},
        "openobserve": {"claude_code": {"telemetry": {"enabled": True, "backend": "openobserve"}}},
        "virtual_accelerator": {"control_system": {"type": "virtual_accelerator"}},
    }
    for service, switched_on in cases.items():
        errors = reach_errors({**switched_on, "deployed_services": ["qmd"]})
        assert len(errors) == 1, (service, errors)
        assert f"`{service}`" in errors[0], (service, errors)


def test_an_attached_render_is_told_its_hosts_service_and_is_not_refused():
    """An attached render (``deploy_services: false``) renders
    ``deployed_services: []``: its clients dial the HOSTING deployment's
    published port on the shared network namespace, which is exactly what the
    projection is for. The deploying rule must not touch it — with the key
    empty or absent alike."""
    switched_on = {"claude_code": {"servers": {"bluesky": {"enabled": True}}}}

    assert reach_errors({**switched_on, "deployed_services": []}) == []
    assert reach_errors(switched_on) == []


def test_a_deploying_render_that_runs_the_service_is_not_refused():
    config = {
        "claude_code": {"servers": {"bluesky": {"enabled": True}}},
        "services": {"bluesky": {"port": 8090}},
        "deployed_services": ["postgresql", "bluesky"],
    }

    assert reach_errors(config) == []


def test_a_deploying_render_may_name_a_service_it_does_not_run():
    """The documented external shapes: a graph store the facility runs
    (``services.graphdb.uri`` with ``graphdb`` left OUT of
    ``deployed_services``), an ARIEL database named by DSN, a bridge named by
    URL. Each names the endpoint outright rather than deriving it from a
    service this deployment would publish, so there is nothing to refuse."""
    external = [
        {
            "channel_finder": {"pipeline_mode": "graph"},
            "services": {"graphdb": {"uri": "bolt://graph.facility.org:7687"}},
        },
        {"ariel": {"database": {"uri": "postgresql://ariel@db.facility.org/ariel"}}},
        {
            "claude_code": {"servers": {"bluesky": {"enabled": True}}},
            "bluesky": {"bridge_url": "http://bridge.facility.org:8090"},
        },
    ]
    for config in external:
        assert reach_errors({**config, "deployed_services": ["openobserve"]}) == [], config


def test_the_deploying_rule_has_no_consumer_to_refuse_where_no_client_dials():
    """Two kinds of contract carry no consumer at all, so no rule that walks
    live consumers can refuse them — and that is the truth of each:

    * ``derived_by`` (mongodb ← va_archiver): the archiver connector's client
      facts are derived from the ``va_archiver:`` block on their own build
      path, which refuses an attached profile that names no host itself.
    * ``no_client_reach`` (the recorder, the dispatch worker, the chat
      bridges): nothing inside a persona container dials them; each is a
      host-side writer or a worker that dials the persona, never the reverse.
    """
    silent = [
        contract
        for contract in REACH_CONTRACTS.values()
        if contract.derived_by or contract.no_client_reach
    ]
    assert {c.service for c in silent} >= {"mongodb", "archiver_recorder", "dispatch_worker"}
    for contract in silent:
        assert contract.consumers == (), contract.service

    # A deploying render that reads an archive and dispatches events, with none
    # of those services in its list, is refused nothing on their account.
    config = {
        "archiver": {"type": "mongodb_archiver"},
        "va_archiver": {"host": "localhost"},
        "deployed_services": ["postgresql"],
    }
    assert reach_errors(config) == []


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
