"""Where a connector-host child for a target will connect, derived from config alone.

**Derivation** (:func:`derive_endpoints`) restates
``EPICSConnector.connect()``'s gateway selection *positively*, from config
alone: for a target, the connector type it resolves to, the endpoint each
configured gateway role would produce, and the role this run will actually
select given the target's write posture and the run mode. The controls MCP
server's target switch, the library's multi-target
:class:`~osprey_connectors.ipc.ConnectorHostPool`, and every reader that only
needs to know where a target points (the compose generator, the endpoint
prober, the archiver recorder) take it from here; verifying what a child
reported against it is :mod:`osprey_connectors.ipc`'s.

**Collision with the live machine** (:func:`live_collision`) is the one
eligibility question every supervisor asks from derivation alone: an endpoint a
target that is never the real machine selects must not be one ``live`` derives.

The gateway role rule itself lives in the control-system layer, where
``EPICSConnector.connect()`` takes it; its types are re-exported here.

Nothing here imports a control-system client library, and importing this
module loads neither :mod:`osprey_connectors.ipc` nor its wire codec.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, SupportsIndex, SupportsInt

from osprey_connectors.config import config_flag, config_section, connector_block, subsection
from osprey_connectors.connector_types import (
    STANDIN_TYPES,
    TARGET_LIVE,
    TRANSPORT_IN_PROCESS,
    VIRTUAL_ACCELERATOR,
    ControlTarget,
    Transport,
    connector_transport,
)
from osprey_connectors.control_system._gateway_role import (
    GatewayRole,
    SelectedRole,
    _selected_role,
    readonly_gateway_refusal,
)
from osprey_connectors.execution_mode import is_readonly_run
from osprey_connectors.model import FrozenModel, ReadOnlyMapping
from osprey_connectors.targets import resolve_target
from osprey_connectors.write_tier import tier_allows

__all__ = [
    "DEFAULT_CA_PORT",
    "DEFAULT_PVA_PORT",
    "NEVER_LIVE_TYPES",
    "Endpoint",
    "GatewayMode",
    "GatewayRole",
    "GatewaysDeclared",
    "LiveCollision",
    "SelectedRole",
    "TargetDerivation",
    "declared_gateways",
    "derive_endpoints",
    "live_collision",
    "readonly_gateway_refusal",
    "same_endpoint",
]

# -- The environment modes gateway roles produce ----------------------------

#: How a gateway reaches EPICS: ``name_server`` (``use_name_server: true``:
#: EPICS_CA_NAME_SERVERS / EPICS_PVA_NAME_SERVERS) or ``addr_list``
#: (``use_name_server: false``: EPICS_CA_ADDR_LIST / EPICS_PVA_ADDR_LIST).
GatewayMode = Literal["name_server", "addr_list"]


#: The ports ``EPICSConnector.connect()`` falls back to when a gateway names
#: none. The virtual accelerator never reaches the CA default: its unset ports
#: are filled from ``services.virtual_accelerator.port`` first. The PVA default
#: applies to name servers only (TCP); an address-list entry that names no port
#: is passed to p4p without one, so its row carries ``None``.
DEFAULT_CA_PORT = 5064
DEFAULT_PVA_PORT = 5075

#: How a connector block spells its ``gateways`` table, as
#: :func:`declared_gateways` reads it: an explicit ``gateways: []`` (nothing to
#: probe), no table anybody filled in, or rows.
GatewaysDeclared = Literal["none_declared", "missing", "rows"]


class Endpoint(FrozenModel):
    """Where one gateway role points, and how the child will say so to EPICS."""

    #: The ``address`` the gateway block carries, as written: a string in any
    #: config that names a host, but whatever the config holds, so a block
    #: whose address is no host still derives a row the verifiers refuse.
    host: object
    #: The ``port`` the gateway block carries, as written (or the default the
    #: derivation filled in): usually an ``int``, but whatever the config holds.
    port: object
    mode: GatewayMode


class TargetDerivation(FrozenModel):
    """What a connector-host child *will* do for a target, derived from config.

    ``endpoints`` carries one row per configured role: ``read_only`` and
    ``write_access`` when the gateways table has them, plus ``pva`` when the
    block configures PVA routing *and* a PVA gateway — the same conjunction
    ``connect()`` requires before it touches a PVA environment variable.

    ``selected_role`` is the row the child will actually configure the process
    with: EPICS keeps one process-wide context, so exactly one gateway is used.

    ``transport`` is the wire the child's connector will speak
    (:func:`~osprey_connectors.connector_types.connector_transport`), so a child that
    built the other venue of the simulator is refused by what it echoes.

    ``gateways_declared`` is :func:`declared_gateways` of the block: only an
    explicit empty ``gateways: []`` is ``none_declared``, the facility saying
    this target has nothing to probe. A missing table is not that statement — a
    forgotten table must never read as a direct connection — so it is
    ``missing``. The simulator served in process reads no block,
    and keeps the default.
    """

    target: ControlTarget
    connector_type: str
    endpoints: ReadOnlyMapping[GatewayRole, Endpoint]
    selected_role: SelectedRole
    transport: Transport | None
    gateways_declared: GatewaysDeclared = "missing"

    @property
    def declares_no_endpoints(self) -> bool:
        """Whether the block spells ``gateways: []``, and so has nothing to probe."""
        return self.gateways_declared == "none_declared"

    def selected_endpoint(self) -> Endpoint | None:
        """The row the child will configure, or ``None`` when config has none."""
        return self.endpoints.get(self.selected_role)


def _mode(gateway: Mapping[str, object], key: str) -> GatewayMode:
    name_server = config_flag(gateway.get("use_name_server"), key=f"{key}.use_name_server")
    return "name_server" if name_server else "addr_list"


def _row(gateway: object, default_port: int | None, key: str) -> Endpoint | None:
    """One endpoint row, or ``None`` for a gateway ``connect()`` would ignore.

    ``connect()`` guards its environment derivation with ``if gateway_config:``,
    so a missing, non-mapping or empty gateway configures nothing at all — and a
    role that configures nothing is not a role this deployment can select.
    """
    if not isinstance(gateway, dict) or not gateway:
        return None
    return Endpoint(
        host=gateway.get("address", ""),
        port=gateway.get("port", default_port),
        mode=_mode(gateway, key),
    )


def _pva_globs(block: Mapping[str, object]) -> list[str]:
    """The PVA routing globs, normalized exactly as ``connect()`` normalizes."""
    pva_channels = block.get("pva_channels") or []
    if isinstance(pva_channels, str):
        pva_channels = [pva_channels]
    if not isinstance(pva_channels, list | tuple):
        return []
    return [str(pattern).strip() for pattern in pva_channels if str(pattern).strip()]


def derive_endpoints(
    config: object,
    target: ControlTarget,
    *,
    writes_enabled: bool | None = None,
    readonly_run: bool | None = None,
    config_path: str | None = None,
) -> TargetDerivation:
    """Derive the per-role endpoints and selected role for *target*.

    Args:
        config: The full rendered config mapping (``config.yml`` as loaded).
        target: The control target.
        writes_enabled: Whether writes are armed for *target*. Defaults to
            the deployment's write tier —
            :func:`~osprey_connectors.write_tier.tier_allows` over the
            ``control_system`` section, which is what the connector's write
            verdict reads — never the per-type ``writes_enabled`` key on its
            own; injectable so a caller (or a test) can derive the endpoints of
            a posture other than the configured one.
        readonly_run: Whether this is a readonly executor run. Defaults to
            :func:`~osprey_connectors.execution_mode.is_readonly_run`.
        config_path: The project config an unset virtual-accelerator gateway
            port is filled from — the file the child reads, so the port derived
            here is the port it connects to. ``None`` reads ``CONFIG_FILE``,
            else ``./config.yml``.

    Returns:
        The derivation, whose ``endpoints`` may be empty when the deployment has
        no gateways for this target — an eligibility question, not an error, so
        the derivation still answers "which role would be selected".

    Raises:
        ValueError: Propagated from
            :func:`~osprey_connectors.targets.resolve_target` when the target is
            unknown, or is ``live`` on a deployment that has never named its real
            machine, and from :func:`~osprey_connectors.config.config_flag` when a
            ``use_name_server`` spells neither true nor false.
            :func:`~osprey.mcp_server.control_system.target_eligibility.evaluate_eligibility`
            is where that becomes a reason rather than an exception.
    """
    control_system = config_section(config, "control_system")
    connector_type = resolve_target(control_system, target)
    transport = connector_transport(control_system, connector_type)

    if transport == TRANSPORT_IN_PROCESS:
        # The simulator served inside the child dials nothing, so there is no
        # block to read endpoints from and no gateway role to select.
        return TargetDerivation(
            target=target,
            connector_type=connector_type,
            endpoints={},
            selected_role="read_only",
            transport=transport,
        )

    if writes_enabled is None:
        writes_enabled = tier_allows(control_system)
    if readonly_run is None:
        readonly_run = is_readonly_run()

    raw_block = connector_block(config, connector_type)
    block: Mapping[str, object] = raw_block if isinstance(raw_block, dict) else {}
    # The virtual accelerator served from its container is a service this
    # project deploys, so an unset gateway port follows
    # services.virtual_accelerator.port. Filled through the
    # connector's own helper rather than restated, so the roster cannot name a
    # port the child will not use.
    if connector_type == VIRTUAL_ACCELERATOR:
        # Imported here: the connector module is only needed for this one
        # type, and the derivation should not load a connector to answer for
        # every other one.
        from osprey_connectors.control_system._va_connector import fill_gateway_ports

        block = fill_gateway_ports(block, config_path)

    block_key = f"control_system.connector.{connector_type}"
    gateways = subsection(block, "gateways")
    endpoints: dict[GatewayRole, Endpoint] = {}
    role: SelectedRole
    for role in ("read_only", "write_access"):
        row = _row(gateways.get(role), DEFAULT_CA_PORT, f"{block_key}.gateways.{role}")
        if row is not None:
            endpoints[role] = row

    # PVA is derived only under the same conjunction connect() requires: routing
    # globs AND a gateway. Globs without a gateway import p4p but touch no PVA
    # environment variable, so there is no endpoint to report.
    if _pva_globs(block):
        pva_gateway = block.get("pva_gateway")
        # connect() appends no port to an address list unless one is set; the
        # TCP default applies to name servers only.
        pva_key = f"{block_key}.pva_gateway"
        name_server = isinstance(pva_gateway, dict) and _mode(pva_gateway, pva_key) == "name_server"
        pva_row = _row(pva_gateway, DEFAULT_PVA_PORT if name_server else None, pva_key)
        if pva_row is not None:
            endpoints["pva"] = pva_row

    return TargetDerivation(
        target=target,
        connector_type=connector_type,
        endpoints=endpoints,
        selected_role=_selected_role(
            gateways, writes_enabled=bool(writes_enabled), readonly_run=bool(readonly_run)
        ),
        transport=transport,
        gateways_declared=declared_gateways(block),
    )


def declared_gateways(block: object) -> GatewaysDeclared:
    """How a connector *block* spells its ``gateways`` table: the one reader of that fact.

    Only the explicit empty list is ``none_declared``. A block that is not a
    mapping, or whose table is missing, ``null``, ``{}`` or anything but a
    non-empty mapping, is ``missing`` — a table nobody filled in, never "nothing
    to probe". A non-empty mapping is ``rows``.
    """
    if not isinstance(block, dict):
        return "missing"
    if block.get("gateways") == []:
        return "none_declared"
    return "rows" if subsection(block, "gateways") else "missing"


# -- Comparing two rows -----------------------------------------------------


def _as_int(value: object) -> int:
    """``int(value)`` for whatever ``int`` converts; ``TypeError`` for anything else."""
    if isinstance(value, str | bytes | SupportsInt | SupportsIndex):
        return int(value)
    raise TypeError(f"{type(value).__name__!r} is not a number")


def _ports_equal(expected: object, got: object) -> bool:
    """Whether two ports name the same port.

    ``connect()`` interpolates the port into a string environment value, so a
    child may report ``'5064'`` where the config carries ``5064``. Compared
    numerically when both are integral and textually otherwise, so a genuinely
    different port never compares equal.
    """
    try:
        return _as_int(expected) == _as_int(got)
    except (TypeError, ValueError):
        return str(expected) == str(got)


def _host_key(host: object) -> str:
    """*host* as two rows must both spell it to name the same machine.

    Stripped and case-folded, and nothing more: no name is resolved, and no two
    spellings of one address — ``localhost`` and ``127.0.0.1``, say — are
    folded together. A compose service name that resolves only inside its own
    network is a host this process may be unable to resolve at all, and a
    comparison that needed to would answer differently on every machine.
    """
    return str(host).strip().lower() if host is not None else ""


def same_endpoint(first: Endpoint, second: Endpoint) -> bool:
    """Whether two derived rows name the same address and port, as written.

    The CA/PVA *mode* is deliberately not compared: a name server and an
    address-list entry on one host and port reach the same server. Ports are
    compared as verification compares them (``'5064'`` is ``5064``); hosts as
    :func:`_host_key` spells them, so two different spellings of one machine
    are not caught here.
    """
    return _host_key(first.host) == _host_key(second.host) and _ports_equal(first.port, second.port)


#: Types that are never the facility's own machine, so an endpoint one of them
#: selects must never be an endpoint ``live`` derives.
NEVER_LIVE_TYPES = frozenset({VIRTUAL_ACCELERATOR, *STANDIN_TYPES})


class LiveCollision(FrozenModel):
    """A live gateway that a never-live target's selected endpoint lands on.

    ``selected`` and ``live`` name the same endpoint, each as its own block
    spells it; ``role`` and ``connector_type`` locate ``live`` in the config.
    """

    selected: Endpoint
    live: Endpoint
    role: SelectedRole
    connector_type: str


def live_collision(
    config: object, derivation: TargetDerivation, *, config_path: str | None = None
) -> LiveCollision | None:
    """The live gateway *derivation*'s selected endpoint lands on, or ``None``.

    Only a target whose type is in :data:`NEVER_LIVE_TYPES` is asked: its reads
    and writes must not reach an endpoint ``live`` derives from the same config,
    read-only or write-access alike, or they reach the real machine under a soft
    label. Addresses are compared as written (:func:`same_endpoint`), never
    resolved, so this is the negative half only — whatever the target is, it is
    not the machine. A deployment with no live machine has nothing to collide
    with.

    Args:
        config: The rendered config mapping *derivation* was derived from.
        derivation: The target being judged, from :func:`derive_endpoints`.
        config_path: See :func:`derive_endpoints`; pass the one *derivation* used.
    """
    selected = derivation.selected_endpoint()
    if selected is None or derivation.connector_type not in NEVER_LIVE_TYPES:
        return None
    try:
        live = derive_endpoints(
            config, TARGET_LIVE, writes_enabled=False, readonly_run=True, config_path=config_path
        )
    except ValueError:
        return None
    for role in ("read_only", "write_access"):
        row = live.endpoints.get(role)
        if row is not None and same_endpoint(selected, row):
            return LiveCollision(
                selected=selected, live=row, role=role, connector_type=live.connector_type
            )
    return None
