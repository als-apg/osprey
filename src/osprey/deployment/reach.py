"""The Reach Contract: how a container reaches the services its deployment runs.

A deployment's services are reached by clients that do not deploy them — every
web-terminal persona is an *attached* render (``deploy_services: false``) whose
container shares the host's network namespace and dials the hosting
deployment's Postgres, its qmd sidecar, its graph store, its bluesky bridge,
its telemetry store — and each of those clients resolves its endpoint through
a different mechanism: a ``services.<name>`` block the build writes for a
deploying project, a compiled-in default, an env var a compose author sets.
Nothing tied the three together, so a client could be switched ON in a render
that carried nothing for it to resolve, and the failure surfaced at first use,
inside a per-user container: "no qmd sidecar is configured", telemetry posting
to a DNS name that resolves to nothing, a bridge dialed on a port the host
stopped publishing.

This module is the one place those facts are declared, so that every
enforcement reads the same declaration:

* **The build projects.** For an attached render in the same repo as its host,
  :func:`project_attached_overrides` copies each contract's client-facing
  keys from the HOST's rendered config into the attached render, gated on the
  consumer being switched on there. The build already knows every
  client-facing fact about its services; no persona restates one, and a
  service moved on the hosting profile moves every client with it.
  An attached profile built with no deployment in its repo is projected from
  what its app template deploys at the shipped defaults instead — the
  deployment it extends is one of that template — with its own ``config:``
  laid over them, which is where a host that differs is named.
* **The build refuses.** :func:`reach_errors` reads a rendered config and
  refuses a consumer that is on with nothing to resolve — the generic backstop
  for a render whose host, or whose app template, deploys no such service.
* **Tests check the seams.** The credential grants and shared paths declared
  here are what ``tests/deployment/test_reach_contract.py`` walks over a real
  built stack: switch on ⇒ endpoint resolves ∧ credential in the container's
  compose block ∧ shared directory mounted. And every ``services.<name>`` a
  shipped template can deploy must have a contract or say why it needs none.
* **``osprey health`` probes.** The ``reach`` category resolves each live
  consumer's endpoint from inside any container and knocks on it.

The registry is a mapping keyed by the ``services.<name>`` key, collected
here like :data:`osprey.registry.mcp.FRAMEWORK_SERVERS`. Predicates are the
ones the web-terminal render already grants credentials by
(:mod:`osprey.deployment.web_terminals.personas`), imported rather than
restated, so the registry and the render cannot disagree about who is
entitled to what.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from osprey.bluesky_bridge_connection import (
    BRIDGE_URL_ENV_VAR,
    DEFAULT_BRIDGE_PORT,
    SECOND_LANE_KEYS,
    bridge_url_from_config,
    lane_env_prefix,
)
from osprey.deployment.graphdb_service import DEFAULT_PORT as GRAPHDB_DEFAULT_PORT
from osprey.deployment.graphdb_service import (
    GRAPHDB_PASSWORD_ENV,
    GRAPHDB_PORT_CONFIG_KEY,
    GRAPHDB_SERVICE_NAME,
    resolve_graphdb_connection,
    resolve_graphdb_service_config,
)
from osprey.deployment.qmd_service import DEFAULT_PORT as QMD_DEFAULT_PORT
from osprey.deployment.qmd_service import PORT_CONFIG_KEY as QMD_PORT_CONFIG_KEY
from osprey.deployment.qmd_service import QMD_SERVICE_NAME, resolve_qmd_service_config
from osprey.deployment.web_terminals.personas import (
    BLUESKY_PANEL_ID,
    EVENTS_PANEL_ID,
    as_dict,
    config_declares_panel,
    config_needs_ariel_mirror,
    config_needs_ariel_password,
    config_needs_dispatcher_token,
    config_needs_facility_bundle,
    config_needs_graphdb_password,
    config_needs_launch_token,
)

__all__ = [
    "Consumer",
    "CredentialGrant",
    "ProjectedKey",
    "ReachContract",
    "SharedPath",
    "REACH_CONTRACTS",
    "SHARED_PATHS",
    "BLUESKY_PANEL_ID",
    "Dial",
    "dotted_get",
    "project_attached_overrides",
    "reach_errors",
    "live_consumers",
    "reach_dials",
]

Predicate = Callable[[Mapping[str, Any]], bool]

#: ``(host, port)`` — what a consumer's client actually connects to.
Dial = tuple[str, int]
Dialer = Callable[[Mapping[str, Any]], "Dial | None"]


def dotted_get(config: Mapping[str, Any] | None, dotted_key: str) -> Any:
    """The value at *dotted_key* in a nested config, or ``None`` when any
    segment is missing or not a mapping."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _servers_enabled(config: Mapping[str, Any], server: str, *, default: bool) -> bool:
    """Whether ``claude_code.servers.<server>.enabled`` leaves the server on.

    Read the way :func:`osprey.registry.mcp.resolve_servers` reads it: an
    explicit ``false`` switches a server off, an explicit ``true`` switches one
    on, and absence leaves the server's own default.
    """
    value = as_dict(
        as_dict(as_dict(as_dict(config).get("claude_code")).get("servers")).get(server)
    ).get("enabled")
    if value is False:
        return False
    if value is True:
        return True
    return default


@dataclass(frozen=True)
class Consumer:
    """One in-container client of a shared service.

    Attributes:
        name: What it is, for messages (``"ARIEL hybrid search"``).
        switch_key: The dotted config key that switches it on, for messages.
        is_on: Whether this rendered config switches the consumer on.
        resolves: Whether this rendered config gives the consumer an endpoint.
            A resolver that always answers (a compiled-in default it would
            dial) reports ``True`` unconditionally; such a consumer is served
            by projection, never by refusal.
        refuse: Whether ``on ∧ not resolves`` refuses the build. ``False`` for
            a consumer that degrades on purpose (the OKF panel's ranked
            search falls back to substring matching); the health probe still
            reports it.
        dial: The ``(host, port)`` this consumer's client connects to, resolved
            THROUGH the client's own resolver (the bridge URL the MCP server
            builds, the DSN the panel server derives, the OTLP endpoint the
            exporter posts to) — so the ``reach`` health category knocks on
            what the client dials, not on what the service block says.
            ``None`` from the dialer means the client has nothing to dial.
    """

    name: str
    switch_key: str
    is_on: Predicate
    resolves: Predicate
    refuse: bool = True
    dial: Dialer | None = None


@dataclass(frozen=True)
class ProjectedKey:
    """A client-facing fact the build copies from the host render into an
    attached render.

    Attributes:
        key: Dotted config key, in both renders.
        gate: Whether the attached render has a consumer for it. ``None``
            projects whenever the host render carries the key. A gate keeps a
            fact out of a render that would grow a surface from it — a
            ``services.graphdb`` block makes the graph MCP server render, so
            it is projected only where that server is not switched off.
        panel: When set, the key belongs to this panel's ``web.panels`` entry
            and is projected only for a render whose profile selects the
            panel (``web_panels:``), which the rendered config alone cannot
            tell: custom panels reach it through ``config:`` only.
    """

    key: str
    gate: Predicate | None = None
    panel: str | None = None


@dataclass(frozen=True)
class CredentialGrant:
    """An env var a container needs to authenticate to the service.

    Attributes:
        env: The variable's name in the container.
        gate: Which renders get it — the same predicate the web-terminal
            render grants by. ``None`` is ungated: every container gets it.
    """

    env: str
    gate: Predicate | None


@dataclass(frozen=True)
class ReachContract:
    """Everything a client needs to reach one shared service.

    Attributes:
        service: The ``services.<name>`` key.
        consumers: The in-container clients, each with its switch and resolver.
        projected: The keys copied into an attached render from its host.
        credentials: The env vars entitled containers receive.
        no_client_reach: The service is dialed by nothing inside a persona
            container (a host-side writer, a worker the host dials); the
            contract exists to say so, and ``note`` says why.
        derived_by: Name of the build block that already derives this
            service's client facts for attached renders on its own path
            (the archive: :func:`osprey.cli.build_profile_archiver.va_archiver_config_overrides`).
        note: One line for the completeness report.
    """

    service: str
    consumers: tuple[Consumer, ...] = ()
    projected: tuple[ProjectedKey, ...] = ()
    credentials: tuple[CredentialGrant, ...] = ()
    no_client_reach: bool = False
    derived_by: str | None = None
    note: str = ""


@dataclass(frozen=True)
class SharedPath:
    """A host directory bound into every entitled persona container.

    Attributes:
        config_key: The dotted key naming the directory.
        gate: Which renders are entitled — the predicate the render mounts by.
        describe: What lives there, for messages.
    """

    config_key: str
    gate: Predicate
    describe: str


# ---------------------------------------------------------------------------
# Switches and resolvers, one per consumer
# ---------------------------------------------------------------------------


def _hybrid_search_on(config: Mapping[str, Any]) -> bool:
    hybrid = dotted_get(config, "ariel.search_modules.hybrid")
    return isinstance(hybrid, Mapping) and hybrid.get("enabled") is not False


def _okf_ranked_search_on(config: Mapping[str, Any]) -> bool:
    return config_declares_panel(config, "okf") and config_needs_facility_bundle(config)


def _qmd_resolves(config: Mapping[str, Any]) -> bool:
    try:
        return resolve_qmd_service_config(config) is not None
    except ValueError:
        # A malformed block still names a sidecar this render means to dial;
        # the resolver reports the fault where it can be acted on.
        return True


def _graph_server_on(config: Mapping[str, Any]) -> bool:
    # The graph server's own default is "on when a store is configured"
    # (``condition="graphdb_configured"``), so absence reads as on here: the
    # question this switch answers is whether the render WANTS the server —
    # the projection's gate, asked of an attached render that has no store
    # yet — and the resolver below answers whether it has one.
    return _servers_enabled(config, "graph", default=True)


def _graphdb_resolves(config: Mapping[str, Any]) -> bool:
    try:
        return resolve_graphdb_service_config(config) is not None
    except ValueError:
        return True


def _graph_server_renders(config: Mapping[str, Any]) -> bool:
    # The consumer's switch: whether the server is IN this render — read the
    # way resolve_servers reads it, with its own default. Explicitly on with
    # no store is the one misconfiguration left to report; a render that
    # never asked for the server and has no store has no consumer.
    return _servers_enabled(config, "graph", default=_graphdb_resolves(config))


def _channel_finder_graph_on(config: Mapping[str, Any]) -> bool:
    # The graph paradigm's channel finder is its own MCP server, shipped on
    # the rendered pipeline (registry: condition="channel_finder_pipeline")
    # and dialing the store through its own server context — independent of
    # the `graph` server's switch. A persona that switches the graph server
    # off but keeps the channel finder still needs the store.
    return dotted_get(config, "channel_finder.pipeline_mode") == "graph" and _servers_enabled(
        config, "channel-finder", default=True
    )


def _graph_store_wanted(config: Mapping[str, Any]) -> bool:
    # The projection's gate: either client of the store makes the render
    # want its address.
    return _graph_server_on(config) or _channel_finder_graph_on(config)


def _ariel_on(config: Mapping[str, Any]) -> bool:
    return config_needs_ariel_password(config)


def _telemetry_on(config: Mapping[str, Any]) -> bool:
    telemetry = as_dict(as_dict(as_dict(config).get("claude_code")).get("telemetry"))
    return (
        bool(telemetry.get("enabled"))
        and telemetry.get("backend") == "openobserve"
        and not telemetry.get("endpoint")
    )


def _bluesky_server_on(config: Mapping[str, Any]) -> bool:
    return _servers_enabled(config, "bluesky", default=False)


def _second_lane_on(lane: str) -> Predicate:
    """The bluesky MCP server's consumer of a SECOND plan lane.

    A second lane exists only where the build rendered its block
    (``bluesky.second_lane`` on the deploying profile writes
    ``services.<lane>``), so its consumer is on where the bluesky server is
    on AND this render carries the block — a single-lane render has no such
    consumer at all, and a render told the lane by projection has one from
    that moment on. That is what lets one registry entry serve both shapes:
    absent on the host means absent from the persona means nothing to refuse.
    """

    def _on(config: Mapping[str, Any]) -> bool:
        return _bluesky_server_on(config) and isinstance(
            as_dict(config.get("services")).get(lane), Mapping
        )

    return _on


def _second_lane_launch_token_needed(lane: str) -> Predicate:
    # Lane 1's tier boundary (config_needs_launch_token), applied to a lane
    # this render carries: a persona entitled to arm lane 1 is entitled to arm
    # every lane it is told, since its session may be switched to either
    # target — and a render that carries no such lane has nothing to arm.
    def _needed(config: Mapping[str, Any]) -> bool:
        return config_needs_launch_token(config) and isinstance(
            as_dict(config.get("services")).get(lane), Mapping
        )

    return _needed


def _second_lane_resolves(lane: str) -> Predicate:
    # resolve_bridge_url(lane) has no default to fall back to for a second
    # lane: no port is a refusal there, so it is a refusal here.
    def _resolves(config: Mapping[str, Any]) -> bool:
        return bool(dotted_get(config, f"services.{lane}.port"))

    return _resolves


def _va_connector_on(config: Mapping[str, Any]) -> bool:
    from osprey_connectors.types import VIRTUAL_ACCELERATOR

    return bool(dotted_get(config, "control_system.type") == VIRTUAL_ACCELERATOR)


def _events_panel_on(config: Mapping[str, Any]) -> bool:
    return config_declares_panel(config, EVENTS_PANEL_ID)


def _bluesky_panel_on(config: Mapping[str, Any]) -> bool:
    return config_declares_panel(config, BLUESKY_PANEL_ID)


def _panel_url_resolves(panel_id: str) -> Predicate:
    def _resolves(config: Mapping[str, Any]) -> bool:
        return bool(dotted_get(config, f"web.panels.{panel_id}.url"))

    return _resolves


def _always(_config: Mapping[str, Any]) -> bool:
    return True


# ---------------------------------------------------------------------------
# What each consumer's client dials — through the client's own resolver
# ---------------------------------------------------------------------------


def _host_port(url: str, default_port: int) -> Dial | None:
    """``(host, port)`` of *url*, or ``None`` when it names no host."""
    try:
        parts = urlsplit(url if "//" in url else f"//{url}")
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not host:
        return None
    return host, port or default_port


def _qmd_dial(config: Mapping[str, Any]) -> Dial | None:
    try:
        resolved = resolve_qmd_service_config(config)
    except ValueError:
        return None
    return None if resolved is None else _host_port(resolved.base_url, QMD_DEFAULT_PORT)


def _graphdb_dial(config: Mapping[str, Any]) -> Dial | None:
    try:
        if resolve_graphdb_service_config(config) is None:
            return None
        connection = resolve_graphdb_connection(
            dotted_get(config, f"services.{GRAPHDB_SERVICE_NAME}")
        )
    except ValueError:
        return None
    return _host_port(connection.uri, GRAPHDB_DEFAULT_PORT)


def _postgres_dial(config: Mapping[str, Any]) -> Dial | None:
    from osprey.services.ariel_search.config import resolve_ariel_dsn

    try:
        dsn = resolve_ariel_dsn(
            as_dict(as_dict(config).get("ariel")),
            as_dict(dotted_get(config, "services.postgresql")) or None,
        )
    except (TypeError, ValueError):
        return None
    return _host_port(dsn, 5432)


def _telemetry_dial(config: Mapping[str, Any]) -> Dial | None:
    from osprey.build.claude_code_telemetry import resolve_runtime_telemetry_endpoint

    endpoint = resolve_runtime_telemetry_endpoint(config)
    return None if endpoint is None else _host_port(endpoint, 80)


def _bridge_dial(config: Mapping[str, Any]) -> Dial | None:
    # The env half of resolve_bridge_url, then the config half — the same
    # order, on the config the caller holds rather than the loaded one.
    url = os.environ.get(BRIDGE_URL_ENV_VAR) or bridge_url_from_config(config)
    return _host_port(url, DEFAULT_BRIDGE_PORT)


def _second_lane_dial(lane: str) -> Dialer:
    # resolve_bridge_url(lane) for a second lane: its own `<PREFIX>_BRIDGE_URL`
    # wins outright, else the loopback URL of its published port — there is
    # no `bluesky.bridge_url` for a second lane and no default, so no port is
    # nothing to dial.
    def _dial(config: Mapping[str, Any]) -> Dial | None:
        url = os.environ.get(f"{lane_env_prefix(lane)}_BRIDGE_URL")
        if url:
            return _host_port(url, DEFAULT_BRIDGE_PORT)
        port = dotted_get(config, f"services.{lane}.port")
        return ("127.0.0.1", int(port)) if port else None

    return _dial


def _panel_dial(panel_id: str) -> Dialer:
    def _dial(config: Mapping[str, Any]) -> Dial | None:
        url = dotted_get(config, f"web.panels.{panel_id}.url")
        return _host_port(str(url), 80) if url else None

    return _dial


def _va_dial(config: Mapping[str, Any]) -> Dial | None:
    # fill_gateway_ports' rule: an explicit gateway port wins, else the port
    # the deployment publishes, else the connector's default — on the
    # gateway's own address.
    from osprey_connectors.control_system.va_connector import DEFAULT_VA_PORT

    published = dotted_get(config, "services.virtual_accelerator.port") or DEFAULT_VA_PORT
    gateways = as_dict(dotted_get(config, "control_system.connector.virtual_accelerator.gateways"))
    for gateway in gateways.values():
        if isinstance(gateway, Mapping):
            address = str(gateway.get("address") or "localhost")
            port = gateway.get("port") or published
            break
    else:
        address, port = "localhost", published
    try:
        return address, int(port)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def _second_lane_contract(lane: str) -> ReachContract:
    """The contract of one second plan lane (``bluesky_va`` or ``bluesky_live``)."""
    prefix = lane_env_prefix(lane)
    return ReachContract(
        service=lane,
        consumers=(
            Consumer(
                name=f"bluesky MCP server, {lane} lane (and the approval hook's lane lookup)",
                switch_key="claude_code.servers.bluesky.enabled",
                is_on=_second_lane_on(lane),
                resolves=_second_lane_resolves(lane),
                dial=_second_lane_dial(lane),
            ),
        ),
        projected=(
            ProjectedKey(f"services.{lane}.port", gate=_bluesky_server_on),
            ProjectedKey(f"services.{lane}.target", gate=_bluesky_server_on),
        ),
        credentials=(
            CredentialGrant(f"{prefix}_LAUNCH_TOKEN", _second_lane_launch_token_needed(lane)),
        ),
        note=f"resolve_bridge_url({lane!r}) dials the lane's published port on loopback",
    )


REACH_CONTRACTS: dict[str, ReachContract] = {
    QMD_SERVICE_NAME: ReachContract(
        service=QMD_SERVICE_NAME,
        consumers=(
            Consumer(
                name="ARIEL hybrid search",
                switch_key="ariel.search_modules.hybrid.enabled",
                is_on=_hybrid_search_on,
                resolves=_qmd_resolves,
                dial=_qmd_dial,
            ),
            Consumer(
                name="OKF panel ranked search",
                switch_key="web.panels.okf",
                is_on=_okf_ranked_search_on,
                resolves=_qmd_resolves,
                # Falls back to substring matching without a sidecar, by
                # design (interfaces/okf_panel/app.py); reported, not refused.
                refuse=False,
                dial=_qmd_dial,
            ),
        ),
        projected=(
            ProjectedKey(
                QMD_PORT_CONFIG_KEY,
                gate=lambda config: _hybrid_search_on(config) or _okf_ranked_search_on(config),
            ),
        ),
        note="hybrid logbook search and the OKF panel dial the sidecar on loopback",
    ),
    GRAPHDB_SERVICE_NAME: ReachContract(
        service=GRAPHDB_SERVICE_NAME,
        consumers=(
            Consumer(
                name="graph MCP server",
                switch_key="claude_code.servers.graph.enabled",
                is_on=_graph_server_renders,
                resolves=_graphdb_resolves,
                # The server is gated on ``graphdb_configured`` and left out of
                # a render with no store — a coherent surface, never a
                # half-configured one. Refusing here would make every attached
                # render without a store fail; the profile rule refuses the
                # case that matters (channel_finder_mode: graph).
                refuse=False,
                dial=_graphdb_dial,
            ),
            Consumer(
                name="graph channel finder",
                switch_key="channel_finder.pipeline_mode",
                is_on=_channel_finder_graph_on,
                resolves=_graphdb_resolves,
                # Its own server, its own dial: a graph-paradigm channel finder
                # with no store fails every query at first use, so a render
                # that keeps it on with nothing to dial is refused — the
                # profile rule refuses the same shape before the render, and
                # this catches an attached render that was told nothing.
                dial=_graphdb_dial,
            ),
        ),
        projected=(
            # Both spellings of "where is the store": the published bolt port
            # for a store this deployment runs, the uri (and account) for one
            # it merely names. Gated on either client wanting it — the graph
            # server not switched off, or the channel finder on its graph
            # paradigm — since the block's presence is what makes the graph
            # server render, and a store the channel finder dials must be there
            # whether or not that server is.
            ProjectedKey(GRAPHDB_PORT_CONFIG_KEY, gate=_graph_store_wanted),
            ProjectedKey(f"services.{GRAPHDB_SERVICE_NAME}.uri", gate=_graph_store_wanted),
            ProjectedKey(f"services.{GRAPHDB_SERVICE_NAME}.username", gate=_graph_store_wanted),
        ),
        credentials=(CredentialGrant(GRAPHDB_PASSWORD_ENV, config_needs_graphdb_password),),
        note="the graph MCP server and the graph channel finder dial bolt on loopback",
    ),
    "postgresql": ReachContract(
        service="postgresql",
        consumers=(
            Consumer(
                name="ARIEL database (panel server and ariel MCP server)",
                switch_key="ariel",
                is_on=_ariel_on,
                # resolve_ariel_dsn always derives a DSN (shipped defaults);
                # the projection is what keeps it the host's.
                resolves=_always,
                dial=_postgres_dial,
            ),
        ),
        projected=(
            ProjectedKey("services.postgresql.port_host", gate=_ariel_on),
            ProjectedKey("services.postgresql.username", gate=_ariel_on),
            ProjectedKey("services.postgresql.database_name", gate=_ariel_on),
        ),
        credentials=(CredentialGrant("ARIEL_DB_PASSWORD", config_needs_ariel_password),),
        note="resolve_ariel_dsn derives the DSN from the block on loopback",
    ),
    "openobserve": ReachContract(
        service="openobserve",
        consumers=(
            Consumer(
                name="agent telemetry (OTLP exporter)",
                switch_key="claude_code.telemetry.enabled",
                is_on=_telemetry_on,
                # The endpoint always derives (host from context, port from the
                # block or the listen port); the projection keeps the port the
                # host's.
                resolves=_always,
                dial=_telemetry_dial,
            ),
        ),
        projected=(ProjectedKey("services.openobserve.port", gate=_telemetry_on),),
        # Every container emits as the one ingest account, by design (the
        # compose template's ZO_INGEST_SA_TOKEN note): ungated.
        credentials=(CredentialGrant("ZO_INGEST_SA_TOKEN", None),),
        note="the exporter posts to the published port on loopback",
    ),
    "bluesky": ReachContract(
        service="bluesky",
        consumers=(
            Consumer(
                name="bluesky MCP server (and the approval hook's bridge lookup)",
                switch_key="claude_code.servers.bluesky.enabled",
                is_on=_bluesky_server_on,
                # bridge_url_from_config always answers (the shipped default);
                # the projection keeps it the host's port.
                resolves=_always,
                dial=_bridge_dial,
            ),
        ),
        projected=(
            ProjectedKey("services.bluesky.port", gate=_bluesky_server_on),
            # Written by the build only on a two-lane deployment, where it says
            # which control target lane 1 serves; the lane resolver reads it
            # to pick the lane the session is on. Absent on a single-lane host
            # means absent here, which is that resolver's single-lane answer.
            ProjectedKey("services.bluesky.target", gate=_bluesky_server_on),
        ),
        credentials=(CredentialGrant("BLUESKY_LAUNCH_TOKEN", config_needs_launch_token),),
        note="resolve_bridge_url dials the published port on loopback",
    ),
    # The SECOND plan lane a deploying profile opts into (`bluesky.second_lane`),
    # one entry per name it can have — a lane is named for the target it
    # serves, and which one is rendered depends on the deployment baseline.
    # Not an app-template service: the injector writes the block beside lane
    # 1's, reusing the bluesky service template. Projected exactly like lane
    # 1, port and target, so a persona's session switched to the other
    # machine is routed to that machine's lane rather than refused as a
    # single-lane render — and armed by that lane's own token, granted on the
    # same tier boundary as lane 1's.
    **{lane: _second_lane_contract(lane) for lane in SECOND_LANE_KEYS.values()},
    "bluesky_web": ReachContract(
        service="bluesky_web",
        consumers=(
            Consumer(
                name="BLUESKY panel",
                switch_key=f"web.panels.{BLUESKY_PANEL_ID}",
                is_on=_bluesky_panel_on,
                resolves=_panel_url_resolves(BLUESKY_PANEL_ID),
                dial=_panel_dial(BLUESKY_PANEL_ID),
            ),
        ),
        projected=tuple(
            ProjectedKey(f"web.panels.{BLUESKY_PANEL_ID}.{leaf}", panel=BLUESKY_PANEL_ID)
            for leaf in ("url", "path", "label", "health_endpoint")
        ),
        # The reverse grant — the sidecar is handed each entitled user's own
        # secret — is in the sidecar's OWN compose file (services/bluesky_web,
        # rendered in the services stack), keyed on the same panel declaration
        # (personas.bluesky_panel_secret_env_vars).
        note="the panel proxy dials the sidecar at web.panels.bluesky.url with the user's own secret",
    ),
    "event_dispatcher": ReachContract(
        service="event_dispatcher",
        consumers=(
            Consumer(
                name="EVENTS panel",
                switch_key=f"web.panels.{EVENTS_PANEL_ID}",
                is_on=_events_panel_on,
                resolves=_panel_url_resolves(EVENTS_PANEL_ID),
                dial=_panel_dial(EVENTS_PANEL_ID),
            ),
        ),
        projected=tuple(
            ProjectedKey(f"web.panels.{EVENTS_PANEL_ID}.{leaf}", panel=EVENTS_PANEL_ID)
            for leaf in ("url", "path", "label", "health_endpoint")
        ),
        credentials=(CredentialGrant("EVENT_DISPATCHER_TOKEN", config_needs_dispatcher_token),),
        note="the panel proxy dials the dispatcher at web.panels.events.url",
    ),
    "virtual_accelerator": ReachContract(
        service="virtual_accelerator",
        consumers=(
            Consumer(
                name="virtual-accelerator connector (controls MCP server, osprey.runtime)",
                switch_key="control_system.type",
                is_on=_va_connector_on,
                # fill_gateway_ports always answers (DEFAULT_VA_PORT); the
                # projection keeps it the host's port.
                resolves=_always,
                dial=_va_dial,
            ),
        ),
        projected=(ProjectedKey("services.virtual_accelerator.port", gate=_va_connector_on),),
        note="the connector fills every gateway port from the block",
    ),
    "mongodb": ReachContract(
        service="mongodb",
        derived_by="va_archiver",
        # The connector's connection keys and MONGO_ROOT_PASSWORD's name come
        # from the ``va_archiver:`` block on their own path
        # (``archiver.mongodb_archiver.*``, ``config_archiver_password_env``),
        # which an attached project also gets and which refuses an attached
        # profile that names no host.
        note="derived from the va_archiver block; the archiver connector dials it on loopback",
    ),
    "archiver_recorder": ReachContract(
        service="archiver_recorder",
        no_client_reach=True,
        note="a host-side writer; a persona reads the store, not the recorder",
    ),
    "dispatch_worker": ReachContract(
        service="dispatch_worker",
        no_client_reach=True,
        note="the worker dials the persona's project, never the reverse",
    ),
    "nextcloud_bridge": ReachContract(
        service="nextcloud_bridge",
        no_client_reach=True,
        note="a chat poller that dials the dispatcher; nothing in a container dials it",
    ),
    "gchat_bridge": ReachContract(
        service="gchat_bridge",
        no_client_reach=True,
        note="a chat bridge that dials the dispatcher; nothing in a container dials it",
    ),
}

#: Host directories bound into entitled persona containers, each at the path
#: the container's own resolver derives for the key (see
#: :mod:`osprey.deployment.web_terminals.render`).
SHARED_PATHS: tuple[SharedPath, ...] = (
    SharedPath(
        "facility_knowledge.bundle_path",
        config_needs_facility_bundle,
        "the facility-knowledge bundle (OKF panel, facility_knowledge MCP server)",
    ),
    SharedPath(
        "ariel.enhancement_modules.qmd_export.mirror_path",
        config_needs_ariel_mirror,
        "the ARIEL qmd mirror (qmd_export writes it; the sidecar indexes it)",
    ),
)


# ---------------------------------------------------------------------------
# What the build does with the registry
# ---------------------------------------------------------------------------


def project_attached_overrides(
    host_config: Mapping[str, Any] | None,
    attached_config: Mapping[str, Any],
    *,
    selected_panels: Iterable[str] = (),
) -> dict[str, Any]:
    """The dotted keys an attached render is told from its host's render.

    For every contract's projected key: the host render carries a value AND
    the attached render has a consumer for it (the key's gate, or — for a
    panel key — the panel among *selected_panels*). Absent from the host
    render means absent here: a host that dropped its qmd block projects no
    ``services.qmd.port``, and the attached render's own refusal
    (:func:`reach_errors`) then names the consumer left without it.

    Args:
        host_config: The hosting deployment's rendered ``config.yml``, or
            ``None`` for an attached profile built with no host in its repo.
        attached_config: The attached render's config after its profile's own
            ``config:`` overlay — what the gates read.
        selected_panels: The attached profile's ``web_panels`` selection.

    Returns:
        ``{dotted_key: value}``, in registry order; empty without a host.
    """
    if not host_config:
        return {}
    panels = set(selected_panels)
    overrides: dict[str, Any] = {}
    for contract in REACH_CONTRACTS.values():
        for projected in contract.projected:
            value = dotted_get(host_config, projected.key)
            if value is None:
                continue
            if projected.panel is not None:
                if projected.panel not in panels:
                    continue
            elif projected.gate is not None and not projected.gate(attached_config):
                continue
            overrides[projected.key] = value
    return overrides


def live_consumers(config: Mapping[str, Any]) -> list[tuple[ReachContract, Consumer]]:
    """Every consumer *config* switches on, with its contract."""
    return [
        (contract, consumer)
        for contract in REACH_CONTRACTS.values()
        for consumer in contract.consumers
        if consumer.is_on(config)
    ]


def reach_dials(config: Mapping[str, Any]) -> list[tuple[ReachContract, Consumer, Dial | None]]:
    """Every live consumer with what its client dials from *config*.

    Resolved through each client's own resolver, in the process that asks —
    so from inside a per-user container this is the endpoint the agent's own
    tools connect to, env overrides included. ``None`` for a consumer whose
    client has nothing to dial (the same state :func:`reach_errors` refuses
    at build time, seen at run time).
    """
    return [
        (contract, consumer, consumer.dial(config) if consumer.dial else None)
        for contract, consumer in live_consumers(config)
    ]


def reach_errors(config: Mapping[str, Any]) -> list[str]:
    """Refuse a rendered config whose consumer is on with nothing to resolve.

    Read on the RENDERED config — the ground truth every client loads — so it
    holds for a deploying project that dropped a block its modules still use,
    for an attached render whose host projected nothing, and for a standalone
    attached profile that pinned nothing, alike.

    Returns:
        One error per unresolvable consumer whose contract refuses, naming
        the switch and the key that fixes it.
    """
    errors: list[str] = []
    for contract, consumer in live_consumers(config):
        if not consumer.refuse or consumer.resolves(config):
            continue
        keys = ", ".join(projected.key for projected in contract.projected) or (
            f"services.{contract.service}"
        )
        errors.append(
            f"{consumer.name} is switched on ({consumer.switch_key}) but this render carries "
            f"nothing for it to dial: no {keys}. An attached render (deploy_services: false) "
            f"is told these by the build — from its hosting deployment's render, or, built "
            f"on its own, from what its app template deploys — so the deployment it shares "
            f"a host with runs no such service. Name one under `config:` ({keys}), or "
            f"switch the consumer off."
        )
    return errors
