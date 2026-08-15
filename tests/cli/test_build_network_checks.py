"""The network-axis checks ``osprey build`` runs against its own render.

The axis is declared in the profile and realized in Jinja, and the gap between
those two is where a host-mode deployment fails silently: a template that never
adopted the shared macro renders a service onto the compose bridge while every
address the build derived says host, and a service that DID move takes its
compose DNS name with it, stranding whoever was still calling it by that name.
Neither shows up at boot — both surface as a request that resolves to nothing,
long after the build reported success.

So the checks read what was rendered, not what was declared, and refuse the
build:

* an inert ``network: host`` — declared, and no ``network_mode: host`` in the
  compose file it produced;
* a co-deployed dispatch consumer that does not share the pair's network, in
  either direction — the case the single ``dispatch.network`` knob cannot cover,
  because the consumer is a separate service with its own axis and the addresses
  it is rendered with are written for one side of the boundary or the other;
* an address that crosses the boundary — a hand-authored one on a host-mode
  service still naming a bridge-resident service, which osprey deliberately does
  not rewrite;
* a host-mode telemetry exporter aimed at the pinned OpenObserve port while the
  project publishes the store somewhere else.

The bridge default is covered too, and is the load-bearing case: today's
deployments must render through all of this without a word.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from ruamel.yaml import YAML

from osprey.cli.build_cmd import (
    _HOST_NETWORK,
    _OPENOBSERVE_ENDPOINT_PORT,
    _index_rendered_services,
    _names_service,
    _network_check_errors,
    _render_compose_files,
    _render_zones,
)
from osprey.cli.build_injectors import _copy_service_templates
from osprey.cli.build_profile_schema import DEFAULT_NETWORK_MODE, VALID_NETWORK_MODES
from osprey.deployment.compose_generator import prepare_compose_files

yaml_rt = YAML()


# ---------------------------------------------------------------------------
# Hand-built renders — a compose file with exactly the shape under test
# ---------------------------------------------------------------------------


def _write_render(project: Path, service_key: str, stanzas: dict[str, dict[str, Any]]) -> str:
    """Write one rendered compose file where ``services.<service_key>`` puts it."""
    out = project / "build" / "services" / service_key
    out.mkdir(parents=True, exist_ok=True)
    path = out / "docker-compose.yml"
    path.write_text(yaml.safe_dump({"services": stanzas}, sort_keys=False), encoding="utf-8")
    return str(path)


def _bridge_stanza(**extra: Any) -> dict[str, Any]:
    """A container stanza attached to the project network."""
    return {"image": "x:local", "networks": ["osprey-network"], **extra}


def _host_stanza(**extra: Any) -> dict[str, Any]:
    """A container stanza on the host's own network namespace."""
    return {"image": "x:local", "network_mode": _HOST_NETWORK, **extra}


def _config(project: Path, services: dict[str, dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A rendered config.yml, as the checks receive it from the renderer."""
    return {
        "project_name": "netcheck",
        "project_root": str(project),
        "build_dir": "./build",
        "system": {"timezone": "UTC"},
        "services": {
            name: {"path": f"./services/{name}", **block} for name, block in services.items()
        },
        "deployed_services": list(services),
        **extra,
    }


# ---------------------------------------------------------------------------
# Real renders — the bundled templates, through the path the CLI runs
# ---------------------------------------------------------------------------


def _dispatch_config(
    project: Path,
    *,
    dispatch_network: str | None,
    bridges: dict[str, str | None] | None = None,
    extra_services: dict[str, dict[str, Any]] | None = None,
    claude_code: dict[str, Any] | None = None,
) -> Path:
    """Write a config.yml deploying the dispatch pair, and optionally chat bridges.

    Mirrors what ``_inject_dispatch`` writes: the pair's ``network`` key appears
    on BOTH halves and only under host mode, which is the only shape a rendered
    config.yml ever carries.
    """
    dispatcher: dict[str, Any] = {
        "path": "./services/event_dispatcher",
        "port": 8020,
        "facility_name": "TESTFAC",
        "pv_strip_prefix": "TEST:",
    }
    worker: dict[str, Any] = {
        "path": "./services/dispatch_worker",
        "worker_count": 1,
        "worker_port_base": 9190,
        "worker_port_stride": 1,
        "workspace_mode": "shared",
    }
    if dispatch_network is not None:
        dispatcher["network"] = dispatch_network
        worker["network"] = dispatch_network

    services: dict[str, Any] = {"event_dispatcher": dispatcher, "dispatch_worker": worker}
    for bridge, mode in (bridges or {}).items():
        services[bridge] = {"path": f"./services/{bridge}"}
        if mode is not None:
            services[bridge]["network"] = mode
    services.update(extra_services or {})

    config: dict[str, Any] = {
        "project_name": "netcheck",
        "project_root": str(project),
        "build_dir": "./build",
        "system": {"timezone": "UTC"},
        "services": services,
        "deployed_services": list(services),
    }
    if claude_code is not None:
        config["claude_code"] = claude_code

    config_path = project / "config.yml"
    with open(config_path, "w") as handle:
        yaml_rt.dump(config, handle)
    return config_path


def _render(config_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch):
    """Render the project's compose files the way the build does, from inside it."""
    _copy_service_templates(project)
    monkeypatch.chdir(project)
    return prepare_compose_files(str(config_path))


@pytest.fixture
def facility_service(tmp_path: Path):
    """A facility-authored service directory, template body supplied per test."""

    def _make(project: Path, name: str, environment: dict[str, str], *, network: str | None):
        directory = project / "services" / name
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            "services:",
            f"  {name.replace('_', '-')}:",
            "    image: facility:local",
        ]
        if environment:
            lines.append("    environment:")
            lines.extend(f"      {key}: {value}" for key, value in environment.items())
        if network == _HOST_NETWORK:
            lines.append(f"    network_mode: {_HOST_NETWORK}")
        else:
            lines.append("    networks:")
            lines.append("      - osprey-network")
        (directory / "docker-compose.yml.j2").write_text("\n".join(lines) + "\n")
        block: dict[str, Any] = {"path": f"./services/{name}"}
        if network is not None:
            block["network"] = network
        return block

    return _make


# ---------------------------------------------------------------------------
# The constants these checks are written against
# ---------------------------------------------------------------------------


def test_host_network_constant_names_a_real_mode() -> None:
    """``_HOST_NETWORK`` is the non-default member of the axis vocabulary.

    The checks compare rendered and declared values against this one spelling,
    so a vocabulary that moved without it would leave every check reading a mode
    no template renders — and passing silently.
    """
    assert _HOST_NETWORK in VALID_NETWORK_MODES
    assert _HOST_NETWORK != DEFAULT_NETWORK_MODE


def test_openobserve_endpoint_port_matches_the_resolver_pin() -> None:
    """The port the telemetry check reasons about is the one the emitter derives.

    Bound to the resolver rather than restated: if the derived endpoint ever
    stops pinning a port, the check that exists BECAUSE it pins one has to be
    revisited rather than quietly keep asserting the old number.
    """
    from osprey.build.claude_code_telemetry import _resolve_telemetry_endpoint

    endpoint = _resolve_telemetry_endpoint(
        {"backend": "openobserve"}, in_container=True, openobserve_host="localhost"
    )
    assert endpoint == f"http://localhost:{_OPENOBSERVE_ENDPOINT_PORT}/api/default"


# ---------------------------------------------------------------------------
# Address matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "http://event-dispatcher:8020",
        "https://event-dispatcher/health",
        "event-dispatcher:8020",
        "event-dispatcher",
        "tcp://event-dispatcher:9190",
    ],
)
def test_names_service_matches_addresses(value: str) -> None:
    """Every spelling of "reach this service by its compose name" is an address."""
    assert _names_service(value, "event-dispatcher")


@pytest.mark.parametrize(
    "value",
    [
        "/app/event-dispatcher",
        "event-dispatcher-external:8020",
        "my-event-dispatcher:8020",
        "http://localhost:8020",
        "",
    ],
)
def test_names_service_ignores_non_addresses(value: str) -> None:
    """A name that appears as a path segment or inside a longer name is not a reference.

    The check refuses builds, so a false positive costs an operator a deployment
    they cannot ship — the match has to be an address, not a substring.
    """
    assert not _names_service(value, "event-dispatcher")


# ---------------------------------------------------------------------------
# (a) An inert `network: host`
# ---------------------------------------------------------------------------


def test_declared_host_service_rendered_on_the_bridge_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template that never adopted the macro fails the build, named with its remedy."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"legacy_service": {"network": _HOST_NETWORK}})
    rendered = _write_render(tmp_path, "legacy_service", {"legacy": _bridge_stanza()})

    errors = _network_check_errors(config, [rendered])

    assert len(errors) == 1
    message = errors[0]
    assert "services.legacy_service.network is 'host'" in message
    assert "network_mode: host" in message
    assert "./services/legacy_service/docker-compose.yml.j2" in message
    assert 'services/_network_axis.j2" as net' in message
    assert "net.network(services.legacy_service)" in message


def test_declared_host_service_rendered_on_the_host_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The axis realized in the render is what the check is looking for."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"legacy_service": {"network": _HOST_NETWORK}})
    rendered = _write_render(tmp_path, "legacy_service", {"legacy": _host_stanza()})

    assert _network_check_errors(config, [rendered]) == []


def test_inert_check_covers_every_stanza_of_a_multi_container_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One file renders many workers; a single one left on the bridge is the failure.

    The worker template loops over ``worker_count``, so a conditional that
    fires for the first instance and not the rest would otherwise pass.
    """
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"dispatch_worker": {"network": _HOST_NETWORK}})
    rendered = _write_render(
        tmp_path,
        "dispatch_worker",
        {"dispatch-worker-1": _host_stanza(), "dispatch-worker-2": _bridge_stanza()},
    )

    errors = _network_check_errors(config, [rendered])

    assert len(errors) == 1
    assert "services.dispatch_worker.network is 'host'" in errors[0]


def test_a_service_declaring_no_network_is_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The axis is opt-in: an undeclared service rendering on the bridge is correct."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"legacy_service": {}})
    rendered = _write_render(tmp_path, "legacy_service", {"legacy": _bridge_stanza()})

    assert _network_check_errors(config, [rendered]) == []


def test_a_declared_but_undeployed_service_is_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing renders for a service outside ``deployed_services``, so nothing is inert."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"legacy_service": {"network": _HOST_NETWORK}})
    config["deployed_services"] = []

    assert _network_check_errors(config, []) == []


# ---------------------------------------------------------------------------
# (b) Dispatch parity — a consumer that does not share the pair's network
# ---------------------------------------------------------------------------


def test_the_pair_address_contract_is_what_the_bridge_templates_render() -> None:
    """The variables the parity check keys off are the ones the templates emit.

    This check reads variable NAMES, because the host-mode form of these
    addresses carries nothing else to recognize the pair by. A template that
    renamed its half of that contract would leave the check looking for a
    variable nobody renders — silently, and with the templates then free to
    assume a parity nothing enforces.
    """
    from osprey.cli.build_cmd import _DISPATCH_PAIR_ADDRESS_VARS
    from osprey.cli.build_injectors import _locate_pkg_services

    for bridge in ("gchat_bridge", "nextcloud_bridge"):
        template = (_locate_pkg_services() / bridge / "docker-compose.yml.j2").read_text(
            encoding="utf-8"
        )
        for variable in _DISPATCH_PAIR_ADDRESS_VARS:
            assert f"{variable}:" in template, f"{bridge} no longer renders {variable}"


@pytest.mark.parametrize("bridge", ["gchat_bridge", "nextcloud_bridge"])
def test_host_dispatch_pair_with_a_bridge_resident_chat_bridge_fails(
    bridge: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real render of the shipped templates fails, naming both sides and the remedy.

    This is the pair-parity case the single ``dispatch.network`` knob cannot
    cover by construction: the bridge is a separate service with its own axis,
    and the addresses it is rendered with are the pair's compose DNS names.
    """
    config_path = _dispatch_config(tmp_path, dispatch_network=_HOST_NETWORK, bridges={bridge: None})
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    errors = _network_check_errors(config, compose_files)

    assert len(errors) == 1, "one topology mistake, reported once"
    message = errors[0]
    assert f"services.{bridge} is a co-deployed consumer of the dispatch pair" in message
    assert "services.event_dispatcher" in message and "services.dispatch_worker" in message
    assert f"`services.{bridge}.network: host`" in message
    assert "DISPATCHER_URL -> http://event-dispatcher:8020" in message
    assert "WORKER_URL -> http://dispatch-worker-1:9190" in message


@pytest.mark.parametrize("bridge", ["gchat_bridge", "nextcloud_bridge"])
def test_host_mode_chat_bridge_against_a_network_joined_pair_fails(
    bridge: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction, which nothing in the RENDER can give away.

    A host-mode bridge is rendered with the pair's addresses already rewritten
    onto the host's loopback — a form that assumes the pair is host-side too.
    Against a network-joined pair those addresses reach only what the pair
    publishes on the host, which does not include the worker at all. Nothing in
    the rendered value names the pair any more, so this is exactly the case the
    DNS-name scan cannot see and the parity rule exists for.
    """
    config_path = _dispatch_config(tmp_path, dispatch_network=None, bridges={bridge: _HOST_NETWORK})
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    errors = _network_check_errors(config, compose_files)

    assert len(errors) == 1
    message = errors[0]
    assert f"services.{bridge} runs on the host network" in message
    assert "services.event_dispatcher" in message and "services.dispatch_worker" in message
    assert "DISPATCHER_URL -> http://localhost:8020" in message
    assert "WORKER_URL -> http://localhost:9190" in message
    assert "`dispatch.network: host`" in message
    assert f"`services.{bridge}.network`" in message


def test_an_externally_hosted_pair_is_not_a_parity_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no pair deployed the bridge's addresses come from the environment.

    They are ``${DISPATCHER_URL:?...}`` references pointing wherever the site
    runs its dispatcher, so the bridge's own network mode is its own business.
    """
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"gchat_bridge": {"network": _HOST_NETWORK}})
    rendered = _write_render(
        tmp_path,
        "gchat_bridge",
        {
            "gchat-bridge": _host_stanza(
                environment={
                    "DISPATCHER_URL": "${DISPATCHER_URL:?set it}",
                    "WORKER_URL": "${WORKER_URL:?set it}",
                }
            )
        },
    )

    assert _network_check_errors(config, [rendered]) == []


def test_host_dispatch_pair_with_a_host_mode_chat_bridge_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With both sides on one network there is no boundary to cross.

    The bridge's URLs still name the pair here; they name it from the SAME
    namespace, which is a reachability question this check does not own (the
    addresses osprey emits are rewritten where they have to be).
    """
    config_path = _dispatch_config(
        tmp_path, dispatch_network=_HOST_NETWORK, bridges={"gchat_bridge": _HOST_NETWORK}
    )
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    assert _network_check_errors(config, compose_files) == []


def test_bridge_mode_dispatch_pair_with_a_chat_bridge_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's deployment — axis unset everywhere — renders without a word.

    The load-bearing case: these checks are new, and a false alarm here would
    fail every build that has nothing to do with the axis.
    """
    config_path = _dispatch_config(tmp_path, dispatch_network=None, bridges={"gchat_bridge": None})
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    assert _network_check_errors(config, compose_files) == []


def test_bridge_resident_consumer_is_named_by_its_compose_key_when_unowned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stanza no declared service accounts for is still named, by its compose key."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"event_dispatcher": {"network": _HOST_NETWORK}})
    pair = _write_render(tmp_path, "event_dispatcher", {"event-dispatcher": _host_stanza()})
    extra = tmp_path / "build" / "services" / "docker-compose.yml"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(
        yaml.safe_dump(
            {
                "services": {
                    "sidecar": _bridge_stanza(
                        environment={"UPSTREAM": "http://event-dispatcher:8020"}
                    )
                }
            }
        ),
        encoding="utf-8",
    )

    errors = _network_check_errors(config, [pair, str(extra)])

    assert len(errors) == 1
    assert "the compose service 'sidecar'" in errors[0]
    assert "`services.sidecar.network: host`" in errors[0]


# ---------------------------------------------------------------------------
# (c) A hand-authored address pointing off the host namespace
# ---------------------------------------------------------------------------


def test_facility_authored_address_from_a_host_service_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facility_service
) -> None:
    """A host-mode facility service still calling a bridge service by name fails.

    Osprey rewrites only the addresses it emits itself; a hand-authored one is
    refused with both services and the remedy named rather than silently
    deployed pointing at a name that resolves to nothing.
    """
    facility = facility_service(
        tmp_path,
        "facility_probe",
        {"ARCHIVE_URL": "http://site-archive:9000/ingest"},
        network=_HOST_NETWORK,
    )
    store = facility_service(tmp_path, "site_archive", {}, network=None)
    config_path = _dispatch_config(
        tmp_path,
        dispatch_network=None,
        extra_services={"facility_probe": facility, "site_archive": store},
    )
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    errors = _network_check_errors(config, compose_files)

    assert len(errors) == 1
    message = errors[0]
    assert "services.facility_probe runs on the host network" in message
    assert "services.site_archive" in message
    assert "ARCHIVE_URL -> http://site-archive:9000/ingest" in message
    assert "`services.site_archive.network: host`" in message


def test_facility_authored_address_naming_a_dispatch_half_names_the_pair_knob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facility_service
) -> None:
    """The remedy for a pair member is ``dispatch.network`` — the only knob it has.

    ``services.event_dispatcher.network`` is rejected outright by profile
    validation, so naming it here would hand the operator a second error.
    """
    facility = facility_service(
        tmp_path,
        "facility_probe",
        {"DISPATCH_URL": "http://event-dispatcher:8020"},
        network=_HOST_NETWORK,
    )
    config_path = _dispatch_config(
        tmp_path, dispatch_network=None, extra_services={"facility_probe": facility}
    )
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    errors = _network_check_errors(config, compose_files)

    assert len(errors) == 1
    assert "`dispatch.network: host`" in errors[0]
    assert "services.event_dispatcher.network" not in errors[0]


def test_a_host_service_reaching_localhost_is_not_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, facility_service
) -> None:
    """The rewritten form osprey emits under host mode carries no DNS name.

    This is why the check matches on the name rather than on the variable: an
    address that was rewritten onto the host's loopback stops matching, so the
    addresses osprey owns need no exemption list to stay out of the way.
    """
    facility = facility_service(
        tmp_path,
        "facility_probe",
        {"ARCHIVE_URL": "http://localhost:9000/ingest"},
        network=_HOST_NETWORK,
    )
    store = facility_service(tmp_path, "site_archive", {}, network=None)
    config_path = _dispatch_config(
        tmp_path,
        dispatch_network=None,
        extra_services={"facility_probe": facility, "site_archive": store},
    )
    config, compose_files = _render(config_path, tmp_path, monkeypatch)

    assert _network_check_errors(config, compose_files) == []


def test_two_bridge_services_addressing_each_other_are_not_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compose DNS is exactly what a bridge-resident consumer should be using."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"consumer": {}, "store": {}})
    consumer = _write_render(
        tmp_path,
        "consumer",
        {"consumer": _bridge_stanza(environment={"STORE_URL": "http://store:5080"})},
    )
    store = _write_render(tmp_path, "store", {"store": _bridge_stanza()})

    assert _network_check_errors(config, [consumer, store]) == []


def test_environment_declared_as_a_list_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compose accepts ``- KEY=value`` too, and a facility template may use it."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"probe": {"network": _HOST_NETWORK}, "store": {}})
    probe = _write_render(
        tmp_path,
        "probe",
        {"probe": _host_stanza(environment=["STORE_URL=http://store:5080", "QUIET"])},
    )
    store = _write_render(tmp_path, "store", {"store": _bridge_stanza()})

    errors = _network_check_errors(config, [probe, store])

    assert len(errors) == 1
    assert "STORE_URL -> http://store:5080" in errors[0]


# ---------------------------------------------------------------------------
# (d) The pinned OpenObserve endpoint port
# ---------------------------------------------------------------------------


def _telemetry(**overrides: Any) -> dict[str, Any]:
    """A live ``claude_code.telemetry`` block — enabled, openobserve, derived."""
    return {"telemetry": {"enabled": True, "backend": "openobserve", **overrides}}


def _otel_worker_render(project: Path, *, on_host: bool = True) -> str:
    """A rendered worker naming the telemetry store the way its mode does."""
    stanza = (
        _host_stanza(environment={"OSPREY_OTEL_OPENOBSERVE_HOST": "localhost"})
        if on_host
        else _bridge_stanza(environment={"OSPREY_OTEL_OPENOBSERVE_HOST": "openobserve"})
    )
    return _write_render(project, "dispatch_worker", {"dispatch-worker-1": stanza})


def test_host_mode_with_a_republished_store_and_live_telemetry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A derived endpoint aimed at 5080 while the store publishes elsewhere fails.

    The failure it prevents is silent: the exporter posts into a closed port
    and every metric and log is dropped without an error anywhere.
    """
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path,
        {"dispatch_worker": {"network": _HOST_NETWORK}, "openobserve": {"port": 5081}},
        claude_code=_telemetry(),
    )
    worker = _otel_worker_render(tmp_path)
    store = _write_render(tmp_path, "openobserve", {"openobserve": _bridge_stanza()})

    errors = _network_check_errors(config, [worker, store])

    assert len(errors) == 1
    message = errors[0]
    assert "services.dispatch_worker" in message
    assert "5080" in message and "5081" in message
    assert "`claude_code.telemetry.endpoint: http://localhost:5081/api/default`" in message


def test_the_endpoint_remedy_carries_the_configured_org(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The remedy is copy-pasteable, so it names the org this deployment uses."""
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path,
        {"dispatch_worker": {"network": _HOST_NETWORK}, "openobserve": {"port": 5081}},
        claude_code=_telemetry(openobserve={"org": "facility"}),
    )
    worker = _otel_worker_render(tmp_path)

    errors = _network_check_errors(config, [worker])

    assert "http://localhost:5081/api/facility" in errors[0]


def test_a_republished_store_with_inert_telemetry_warns_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With telemetry off the variable is inert, so the build proceeds — loudly.

    Refusing here would be an error about nothing; saying nothing would leave
    the trap armed for whoever enables telemetry later.
    """
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path,
        {"dispatch_worker": {"network": _HOST_NETWORK}, "openobserve": {"port": 5081}},
    )
    worker = _otel_worker_render(tmp_path)

    with caplog.at_level("WARNING"):
        errors = _network_check_errors(config, [worker])

    assert errors == []
    assert "claude_code.telemetry.endpoint" in caplog.text
    assert "5081" in caplog.text


def test_an_explicit_endpoint_settles_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is derived when the endpoint is written out, so nothing can be wrong."""
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path,
        {"dispatch_worker": {"network": _HOST_NETWORK}, "openobserve": {"port": 5081}},
        claude_code=_telemetry(endpoint="http://localhost:5081/api/default"),
    )
    worker = _otel_worker_render(tmp_path)

    assert _network_check_errors(config, [worker]) == []


def test_the_default_store_port_is_never_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing the store where the endpoint is pinned is the case that works."""
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path,
        {
            "dispatch_worker": {"network": _HOST_NETWORK},
            "openobserve": {"port": _OPENOBSERVE_ENDPOINT_PORT},
        },
        claude_code=_telemetry(),
    )
    worker = _otel_worker_render(tmp_path)

    assert _network_check_errors(config, [worker]) == []


def test_a_bridge_mode_worker_is_never_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the bridge the pin is the port the store LISTENS on, which never moves."""
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path,
        {"dispatch_worker": {}, "openobserve": {"port": 5081}},
        claude_code=_telemetry(),
    )
    worker = _otel_worker_render(tmp_path, on_host=False)
    store = _write_render(tmp_path, "openobserve", {"openobserve": _bridge_stanza()})

    assert _network_check_errors(config, [worker, store]) == []


# ---------------------------------------------------------------------------
# Reading the render back
# ---------------------------------------------------------------------------


def test_rendered_services_are_attributed_to_the_declaration_that_made_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stanza is tied to its ``services.<key>``, however the path is spelled."""
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"dispatch_worker": {"network": _HOST_NETWORK}})
    config["services"]["dispatch_worker"]["path"] = "services/dispatch_worker"
    rendered = _write_render(
        tmp_path,
        "dispatch_worker",
        {"dispatch-worker-1": _host_stanza(), "dispatch-worker-2": _host_stanza()},
    )

    indexed = _index_rendered_services(config, [rendered])

    assert {service.compose_name for service in indexed} == {
        "dispatch-worker-1",
        "dispatch-worker-2",
    }
    assert {service.config_key for service in indexed} == {"dispatch_worker"}
    assert all(service.on_host for service in indexed)


def test_blocks_that_are_not_mappings_do_not_crash_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config key written as the wrong shape is somebody else's error to report.

    These checks run on YAML a person edits. Whatever the consumer of a
    malformed key does about it, the network checks must walk past rather than
    end the build with a traceback that names none of it.
    """
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"legacy_service": {"network": _HOST_NETWORK}})
    config["claude_code"] = "telemetry"
    config["services"]["broken"] = "not-a-mapping"
    rendered = _write_render(tmp_path, "legacy_service", {"legacy": _host_stanza()})

    assert _network_check_errors(config, [rendered]) == []


def test_an_unparseable_compose_file_does_not_become_a_network_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken render is the renderer's failure to report, not this check's.

    Compose names the file and the line; a network-axis heading on top of that
    would only send the operator looking for a network problem.
    """
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, {"legacy_service": {}})
    broken = tmp_path / "build" / "services" / "legacy_service" / "docker-compose.yml"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("services:\n  - this: [is\n", encoding="utf-8")

    assert _network_check_errors(config, [str(broken)]) == []


# ---------------------------------------------------------------------------
# The build seam
# ---------------------------------------------------------------------------


def test_the_build_refuses_the_render_and_restores_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_render_compose_files`` turns a failed check into the build's own refusal.

    A ``UsageError`` rather than a crash: this is an operator's config to fix,
    and the build's handler prints it as such. The working directory it renders
    from is restored either way — the next build stages from wherever this one
    left the process.
    """
    zones = _render_zones(tmp_path)
    zones.stage.mkdir(parents=True)
    (zones.stage / "config.yml").write_text("{}\n", encoding="utf-8")

    config = _config(zones.stage, {"legacy_service": {"network": _HOST_NETWORK}})
    rendered = _write_render(zones.stage, "legacy_service", {"legacy": _bridge_stanza()})

    monkeypatch.setattr(
        "osprey.deployment.compose_generator.prepare_compose_files",
        lambda *args, **kwargs: (config, [rendered]),
    )
    before = os.getcwd()

    with pytest.raises(click.UsageError) as raised:
        _render_compose_files(zones)

    assert "Network axis check failed" in str(raised.value)
    assert "services.legacy_service.network is 'host'" in str(raised.value)
    assert os.getcwd() == before


def test_a_clean_render_returns_the_config_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checks are a gate, not a rewrite: a passing render is handed on as it was."""
    zones = _render_zones(tmp_path)
    zones.stage.mkdir(parents=True)
    (zones.stage / "config.yml").write_text("{}\n", encoding="utf-8")

    config = _config(zones.stage, {"legacy_service": {"network": _HOST_NETWORK}})
    rendered = _write_render(zones.stage, "legacy_service", {"legacy": _host_stanza()})

    monkeypatch.setattr(
        "osprey.deployment.compose_generator.prepare_compose_files",
        lambda *args, **kwargs: (config, [rendered]),
    )

    assert _render_compose_files(zones) == config


def test_a_runtime_root_build_renders_nothing_to_check(tmp_path: Path) -> None:
    """With compose generation skipped there is no render for the checks to read."""
    assert _render_compose_files(_render_zones(tmp_path), runtime_root="/app") is None
