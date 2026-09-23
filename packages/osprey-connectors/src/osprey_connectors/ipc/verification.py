"""What a connector-host child should connect to, and whether it did.

Two pure questions every supervisor of :mod:`osprey_connectors.ipc.host`
children asks, answered once for all of them — the controls MCP server's target
switch and the library's multi-target
:class:`~osprey_connectors.ipc.pool.ConnectorHostPool` alike:

**Derivation** (:func:`derive_endpoints`) restates
``EPICSConnector.connect()``'s gateway selection *positively*, from config
alone: for a target, the connector type it resolves to, the endpoint each
configured gateway role would produce, and the role this run will actually
select given the target's write posture and the run mode.

**Verification** (:func:`verify_host_report`, and :func:`verify_child_report`
for the gateway case) compares a child's post-connect report against that
derivation: not "nothing looked wrong" but "the child configured exactly this
host, this port, this mode, for exactly the role the derivation selected". A
mismatch names the field, the expected value and the value received.

Nothing here imports a control-system client library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osprey_connectors.control_system.base import is_readonly_run
from osprey_connectors.types import VIRTUAL_ACCELERATOR, resolve_target, target_writes_enabled

__all__ = [
    "DEFAULT_CA_PORT",
    "DEFAULT_PVA_PORT",
    "MODE_ADDR_LIST",
    "MODE_NAME_SERVER",
    "REPORT_FIELDS",
    "ROLE_PVA",
    "ROLE_READ_ONLY",
    "ROLE_WRITE_ACCESS",
    "Endpoint",
    "TargetDerivation",
    "Verification",
    "connector_block",
    "derive_endpoints",
    "verify_child_report",
    "verify_host_report",
]

# -- Gateway roles and the environment modes they produce -------------------

ROLE_READ_ONLY = "read_only"
ROLE_WRITE_ACCESS = "write_access"
ROLE_PVA = "pva"

#: ``use_name_server: true`` — EPICS_CA_NAME_SERVERS / EPICS_PVA_NAME_SERVERS.
MODE_NAME_SERVER = "name_server"
#: ``use_name_server: false`` — EPICS_CA_ADDR_LIST / EPICS_PVA_ADDR_LIST.
MODE_ADDR_LIST = "addr_list"

#: The ports ``EPICSConnector.connect()`` falls back to when a gateway names
#: none. The virtual accelerator never reaches the CA default: its unset ports
#: are filled from ``services.virtual_accelerator.port`` first. The PVA default
#: applies to name servers only (TCP); an address-list entry that names no port
#: is passed to p4p without one, so its row carries ``None``.
DEFAULT_CA_PORT = 5064
DEFAULT_PVA_PORT = 5075


@dataclass(frozen=True)
class Endpoint:
    """Where one gateway role points, and how the child will say so to EPICS."""

    host: str
    port: Any
    mode: str

    def as_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "mode": self.mode}


@dataclass(frozen=True)
class TargetDerivation:
    """What a connector-host child *will* do for a target, derived from config.

    ``endpoints`` carries one row per configured role: ``read_only`` and
    ``write_access`` when the gateways table has them, plus ``pva`` when the
    block configures PVA routing *and* a PVA gateway — the same conjunction
    ``connect()`` requires before it touches a PVA environment variable.

    ``selected_role`` is the row the child will actually configure the process
    with: EPICS keeps one process-wide context, so exactly one gateway is used.
    """

    target: str
    connector_type: str
    endpoints: dict[str, Endpoint]
    selected_role: str

    def selected_endpoint(self) -> Endpoint | None:
        """The row the child will configure, or ``None`` when config has none."""
        return self.endpoints.get(self.selected_role)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "connector_type": self.connector_type,
            "endpoints": {role: row.as_dict() for role, row in self.endpoints.items()},
            "selected_role": self.selected_role,
        }


@dataclass(frozen=True)
class Verification:
    """Whether the child came up where the derivation said it would."""

    ok: bool
    field: str | None = None
    expected: Any = None
    got: Any = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "field": self.field,
            "expected": self.expected,
            "got": self.got,
            "detail": self.detail,
        }


def _section(config: Any, name: str) -> dict[str, Any]:
    """One top-level section of a rendered config, or an empty mapping.

    Nested sections only, the way ``MCPServerConfig`` and ``ConfigBuilder`` read
    this file: a top-level dotted line in ``config.yml`` configures nothing.
    """
    if not isinstance(config, dict):
        return {}
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _sub(section: dict[str, Any], name: str) -> dict[str, Any]:
    value = section.get(name)
    return value if isinstance(value, dict) else {}


def connector_block(config: Any, connector_type: str) -> Any:
    """The ``control_system.connector.<type>`` block, as written."""
    return _sub(_section(config, "control_system"), "connector").get(connector_type)


def _config_writes_enabled(config: Any, target: str) -> bool:
    """Whether this config arms writes for *target*, as the connector reads it.

    Per connector type, through the resolver rather than off the section: the
    type *target* selects may carry its own ``writes_enabled``, and the
    deployment-wide key answers only for a type that carries none.
    """
    return target_writes_enabled(_section(config, "control_system"), target)


def _mode(gateway: dict[str, Any]) -> str:
    return MODE_NAME_SERVER if gateway.get("use_name_server", False) else MODE_ADDR_LIST


def _row(gateway: Any, default_port: int | None) -> Endpoint | None:
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
        mode=_mode(gateway),
    )


def _pva_globs(block: dict[str, Any]) -> list[str]:
    """The PVA routing globs, normalized exactly as ``connect()`` normalizes."""
    pva_channels = block.get("pva_channels") or []
    if isinstance(pva_channels, str):
        pva_channels = [pva_channels]
    if not isinstance(pva_channels, list | tuple):
        return []
    return [str(pattern).strip() for pattern in pva_channels if str(pattern).strip()]


def _selected_role(gateways: dict[str, Any], *, writes_enabled: bool, readonly_run: bool) -> str:
    """The gateway role the child will configure the process with.

    A positive restatement of ``EPICSConnector.connect()``'s selection: the
    write-capable gateway is used only when writes are armed for the target being
    derived, this run is not a readonly sandbox run, and a ``write_access``
    gateway is actually configured. Every other combination — writes unarmed,
    readonly run, or no write-capable gateway to route through — lands on
    ``read_only``, which is also what ``connect()`` falls back to (with a
    warning) when writes are enabled and no write gateway exists.

    Takes the posture as an argument rather than resolving it: the caller holds
    the target, and one gateways table is all this selection is entitled to know
    about.
    """
    write_gateway = gateways.get(ROLE_WRITE_ACCESS) or {}
    if writes_enabled and write_gateway and not readonly_run:
        return ROLE_WRITE_ACCESS
    return ROLE_READ_ONLY


def derive_endpoints(
    config: Any,
    target: str,
    *,
    writes_enabled: bool | None = None,
    readonly_run: bool | None = None,
) -> TargetDerivation:
    """Derive the per-role endpoints and selected role for *target*.

    Args:
        config: The full rendered config mapping (``config.yml`` as loaded).
        target: The control target, ``'live'`` or ``'va'``.
        writes_enabled: Whether writes are armed for *target*. Defaults to
            this target's own posture —
            ``control_system.connector.<type>.writes_enabled`` where the
            resolved type states one, ``control_system.writes_enabled`` where it
            does not — which is the same value the connector reads; injectable
            so a caller (or a test) can derive the endpoints of a posture other
            than the configured one.
        readonly_run: Whether this is a readonly executor run. Defaults to
            :func:`~osprey_connectors.control_system.base.is_readonly_run`.

    Returns:
        The derivation, whose ``endpoints`` may be empty when the deployment has
        no gateways for this target — an eligibility question, not an error, so
        the derivation still answers "which role would be selected".

    Raises:
        ValueError: Propagated from
            :func:`~osprey_connectors.types.resolve_target` when the target is
            unknown, or is ``live`` on a deployment that has never named its real
            machine. :func:`evaluate_eligibility` is where that becomes a reason
            rather than an exception.
    """
    control_system = _section(config, "control_system")
    connector_type = resolve_target(control_system, target)

    if writes_enabled is None:
        writes_enabled = _config_writes_enabled(config, target)
    if readonly_run is None:
        readonly_run = is_readonly_run()

    raw_block = connector_block(config, connector_type)
    block = raw_block if isinstance(raw_block, dict) else {}
    # The virtual accelerator is a service this project deploys, so an unset
    # gateway port follows services.virtual_accelerator.port. Filled through the
    # connector's own helper rather than restated, so the roster cannot name a
    # port the child will not use.
    if connector_type == VIRTUAL_ACCELERATOR:
        # Imported here: the connector module is only needed for this one
        # type, and the derivation should not load a connector to answer for
        # every other one.
        from osprey_connectors.control_system.va_connector import fill_gateway_ports

        block = fill_gateway_ports(block)

    gateways = _sub(block, "gateways")
    endpoints: dict[str, Endpoint] = {}
    for role in (ROLE_READ_ONLY, ROLE_WRITE_ACCESS):
        row = _row(gateways.get(role), DEFAULT_CA_PORT)
        if row is not None:
            endpoints[role] = row

    # PVA is derived only under the same conjunction connect() requires: routing
    # globs AND a gateway. Globs without a gateway import p4p but touch no PVA
    # environment variable, so there is no endpoint to report.
    if _pva_globs(block):
        pva_gateway = block.get("pva_gateway")
        # connect() appends no port to an address list unless one is set; the
        # TCP default applies to name servers only.
        name_server = isinstance(pva_gateway, dict) and _mode(pva_gateway) == MODE_NAME_SERVER
        pva_row = _row(pva_gateway, DEFAULT_PVA_PORT if name_server else None)
        if pva_row is not None:
            endpoints[ROLE_PVA] = pva_row

    return TargetDerivation(
        target=target,
        connector_type=connector_type,
        endpoints=endpoints,
        selected_role=_selected_role(
            gateways, writes_enabled=bool(writes_enabled), readonly_run=bool(readonly_run)
        ),
    )


# ---------------------------------------------------------------------------
# (c) VERIFICATION
# ---------------------------------------------------------------------------

#: The fields a connector-host child reports after connecting, in the order it
#: sends them when it reports a tuple.
REPORT_FIELDS = ("selected_role", "mode", "host", "port", "_epics_configured")


def _report_mapping(report: Any) -> dict[str, Any]:
    """The child's report as a mapping, from either shape it may arrive in."""
    if isinstance(report, dict):
        return report
    values = list(report)
    if len(values) != len(REPORT_FIELDS):
        raise ValueError(
            f"A child report carries {len(REPORT_FIELDS)} fields "
            f"{REPORT_FIELDS}; got {len(values)}."
        )
    return dict(zip(REPORT_FIELDS, values, strict=True))


def _ports_equal(expected: Any, got: Any) -> bool:
    """Whether two ports name the same port.

    ``connect()`` interpolates the port into a string environment value, so a
    child may report ``'5064'`` where the config carries ``5064``. Compared
    numerically when both are integral and textually otherwise, so a genuinely
    different port never compares equal.
    """
    try:
        return int(expected) == int(got)
    except (TypeError, ValueError):
        return str(expected) == str(got)


def verify_child_report(derivation: TargetDerivation, report: Any) -> Verification:
    """Assert a child came up exactly where the derivation said it would.

    Positive and role-aware: every field of the selected role's endpoint is
    compared, and the child must also say it configured EPICS at all — a child
    that connected without a gateway leaves ``_epics_configured`` false and has
    silently inherited whatever CA environment the process already carried.

    Args:
        derivation: The derivation for the target being switched to, from
            :func:`derive_endpoints`.
        report: The child's post-connect report, either the mapping or the
            5-tuple ``(selected_role, mode, host, port, _epics_configured)``.

    Returns:
        A passing :class:`Verification`, or a failing one naming the field, the
        expected value and the value the child reported. The switch aborts on a
        failure and leaves the previous target active.

    Raises:
        ValueError: If *report* is a sequence of the wrong length — a malformed
            report is a protocol error, not a verification failure.
    """
    values = _report_mapping(report)

    if not values.get("_epics_configured"):
        return Verification(
            False,
            "_epics_configured",
            True,
            values.get("_epics_configured"),
            "The child reports it never configured an EPICS gateway, so its "
            "environment is whatever the process already carried rather than this "
            "target's.",
        )

    expected_role = derivation.selected_role
    got_role = values.get("selected_role")
    if got_role != expected_role:
        return Verification(
            False,
            "selected_role",
            expected_role,
            got_role,
            f"The child selected the {got_role!r} gateway where target "
            f"{derivation.target!r} derives {expected_role!r}.",
        )

    endpoint = derivation.selected_endpoint()
    if endpoint is None:
        return Verification(
            False,
            "endpoints",
            f"a derived {expected_role!r} endpoint",
            None,
            f"Target {derivation.target!r} has no derived {expected_role!r} endpoint "
            "to compare the child against; it should never have been switched to.",
        )

    for field_name, expected in (("mode", endpoint.mode), ("host", endpoint.host)):
        got = values.get(field_name)
        if got != expected:
            return Verification(
                False,
                field_name,
                expected,
                got,
                f"The child's {field_name} is {got!r} where target "
                f"{derivation.target!r} derives {expected!r} for its "
                f"{expected_role!r} gateway.",
            )

    if not _ports_equal(endpoint.port, values.get("port")):
        return Verification(
            False,
            "port",
            endpoint.port,
            values.get("port"),
            f"The child's port is {values.get('port')!r} where target "
            f"{derivation.target!r} derives {endpoint.port!r} for its "
            f"{expected_role!r} gateway.",
        )

    return Verification(
        True,
        detail=(
            f"The child is on target {derivation.target!r} via its {expected_role!r} "
            f"gateway at {endpoint.host}:{endpoint.port} ({endpoint.mode})."
        ),
    )


def verify_host_report(derivation: TargetDerivation, report: dict[str, Any]) -> Verification:
    """Assert the child came up where the derivation said it would.

    :func:`verify_child_report` answers this for every target whose config
    names a gateway. A deployment can also select a connector that talks to no
    gateway at all — the mock is one, and it is the generic template's default
    — and for that one the derivation has no endpoint and the child reports
    none. Nothing is verified there because there is no endpoint to get wrong,
    but the *symmetry* is: a child that configured Channel Access where the
    config derived nothing has inherited an environment from somewhere, and
    that is a mismatch as serious as any other.

    The unverified branch is entered only when the derivation has **no endpoint
    rows at all**. A target with rows whose *selected* role is missing — a
    write-only gateway table on a deployment that selects ``read_only``, say —
    is a gateway deployment with a hole in it, not a gatewayless one: its child
    connects to a real control system over whatever default broadcast address
    it finds, and that is precisely the unpinned-CA case verification exists to
    catch. It goes to :func:`verify_child_report`, which refuses it either for
    reporting no gateway or for having no derived endpoint to compare against.
    """
    if derivation.endpoints:
        return verify_child_report(derivation, report)

    if report.get("_epics_configured") or report.get("mode") or report.get("host"):
        return Verification(
            False,
            "_epics_configured",
            False,
            report.get("_epics_configured"),
            f"Target {derivation.target!r} derives no gateway on this deployment, but the "
            f"child reports it configured {report.get('mode')!r} routing to "
            f"{report.get('host')!r} — an endpoint this config never described.",
        )
    return Verification(
        True,
        detail=(
            f"Target {derivation.target!r} derives no gateway (connector type "
            f"{derivation.connector_type!r}) and the child configured none."
        ),
    )
