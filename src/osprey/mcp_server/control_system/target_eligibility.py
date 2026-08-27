"""The two named checks a control-system target must pass (FR-3).

A session may be pointed at the real machine or at the simulated one, and the
question "may it be pointed there" is answered twice, at two different moments,
by two checks that are deliberately not the same check:

**ELIGIBILITY** is answered from config alone, with no child process, no socket
and no side effect. It is what the roster reports for a target nobody has
switched to yet, and it is the gate the switch consults before it spawns
anything: is there a connector block for this target, does it have a gateways
table containing the role this deployment will actually select, does it name a
``probe_channel`` to prove itself with, would the switch create the one archiver
pairing that invents history, and — for the live machine — is the deployment in
the posture FR-8 requires. An ineligible target reports a machine-readable
reason, so the roster can say *why* rather than merely *no*.

**VERIFICATION** is answered after the connector-host child has connected, by
comparing what the child says it did against what this module derived it should
do. It is positive and role-aware: not "nothing looked wrong" but "the child
configured exactly this host, this port, this mode, for exactly the role the
derivation selected". A mismatch names the field, the expected value and the
value received, and the switch aborts on it, leaving the previous target active.

The two checks are separate because they fail for different reasons and at
different costs. Eligibility catches the config that could never work, before a
process is spawned. Verification catches the child that came up pointed
somewhere the config never described — the failure mode where every layer
reports success and the tool calls land on the wrong machine.

Both are computed from the same inputs the connector itself uses.
:func:`derive_endpoints` restates ``EPICSConnector.connect()``'s gateway
selection and environment derivation *positively* — the role it would pick given
the target's own write posture and
:func:`~osprey_connectors.control_system.base.is_readonly_run`, and the
CA/PVA mode each gateway would produce — rather than re-deciding it. Where a
pure helper already exists it is imported, never restated: the target resolves
through :func:`~osprey_connectors.types.resolve_target`, its write posture
through :func:`~osprey_connectors.types.target_writes_enabled`, and the virtual
accelerator's unset gateway ports fill through
:func:`~osprey_connectors.control_system.va_connector.fill_gateway_ports`, so
this module and the process it describes cannot disagree about where a target
lands, which port it lands on, or whether it may be written to once there.

Write posture is per connector type, so the two targets of one deployment are
not answered together: a facility whose baseline is the real machine can arm
writes on its simulator alone, and both checks then report a ``write_access``
gateway selected for ``va`` while ``live`` stays on ``read_only``. The
deployment-wide key is not a second posture beside that one — it is what a
target inherits when its own block says nothing about itself.

Session-relativity
------------------
Availability is not a property of a target alone; it is a property of a target
*and the session asking*. FR-8 gates a switch **to** the live machine on strict
limits posture plus an operator acknowledgment, **except** when live is the
deployment's own baseline and the session is returning to it — stranding a
session on the simulator is the less safe outcome, so coming home is never
gated. :func:`target_availability` therefore reports two answers:

* ``available_now`` — the predicate for the switch this session would make,
  including the return exemption when it applies;
* ``eligible_from_baseline`` — the same predicate evaluated as if the session
  sat on the deployment baseline, which is the static, session-independent view
  the roster shows alongside it. From baseline there is no return in progress,
  so a live target is judged there as a switch *toward* the live machine and
  does require the posture and the acknowledgment. The two answers differing for
  a live-baseline deployment with no acknowledgment set is that exemption made
  visible, not a contradiction.

The acknowledgment key is only ever tested for presence. The template ships a
real-hostname-shaped example value, so no string comparison against any known
default could distinguish an operator's answer from the shipped one; testing the
value would invent a distinction the config cannot carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osprey_connectors.control_system.base import is_readonly_run
from osprey_connectors.control_system.va_connector import fill_gateway_ports
from osprey_connectors.honesty import VA_MOCK_ARCHIVER_WHY, pairing_for_target
from osprey_connectors.types import (
    TARGET_LIVE,
    VIRTUAL_ACCELERATOR,
    resolve_target,
    target_writes_enabled,
)

# -- Config keys, spelled once ---------------------------------------------

#: Operator acknowledgment that the configured live gateways are this
#: facility's. Presence/non-emptiness only — never compared to a value.
ACK_KEY = "control_system.target_switch.live_gateway_acknowledged"
#: The same key's leaf, derived from it so the two cannot drift apart.
ACK_LEAF = ACK_KEY.rsplit(".", 1)[1]
LIMITS_ENABLED_KEY = "control_system.limits_checking.enabled"
ALLOW_UNLISTED_KEY = "control_system.limits_checking.allow_unlisted_channels"
#: Leaf key on a connector block; the full key names the resolved type.
PROBE_CHANNEL_KEY = "probe_channel"

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
#: are filled from ``services.virtual_accelerator.port`` first.
DEFAULT_CA_PORT = 5064
DEFAULT_PVA_PORT = 5075

# -- Switch directions ------------------------------------------------------

#: A switch toward a target that is not this deployment's baseline.
DIRECTION_AWAY = "away"
#: A switch back to the deployment baseline from somewhere else.
DIRECTION_BACK = "back"

# -- Machine-readable ineligibility reasons ---------------------------------

REASON_ALREADY_ACTIVE = "already_active"
REASON_TARGET_UNRESOLVABLE = "target_unresolvable"
REASON_CONNECTOR_BLOCK_MISSING = "connector_block_missing"
REASON_GATEWAYS_MISSING = "gateways_missing"
REASON_SELECTED_ROLE_MISSING = "selected_role_missing"
REASON_PROBE_CHANNEL_MISSING = "probe_channel_missing"
REASON_INVENTED_HISTORY = "invented_history"
REASON_LIMITS_POSTURE = "limits_posture"
REASON_OPERATOR_ACK_MISSING = "operator_ack_missing"


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
class Eligibility:
    """The config-only verdict on a target, with the reason for a refusal."""

    eligible: bool
    reason: str | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class TargetAvailability:
    """The roster's answer for one target, from where the session is standing."""

    target: str
    eligible: bool
    available_now: bool
    reason: str | None
    detail: str
    eligible_from_baseline: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "eligible": self.eligible,
            "available_now": self.available_now,
            "reason": self.reason,
            "detail": self.detail,
            "eligible_from_baseline": self.eligible_from_baseline,
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


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# (a) Endpoint derivation
# ---------------------------------------------------------------------------


def _mode(gateway: dict[str, Any]) -> str:
    return MODE_NAME_SERVER if gateway.get("use_name_server", False) else MODE_ADDR_LIST


def _row(gateway: Any, default_port: int) -> Endpoint | None:
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
        target: The session target, ``'live'`` or ``'va'``.
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
        pva_row = _row(block.get("pva_gateway"), DEFAULT_PVA_PORT)
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
# (b) ELIGIBILITY
# ---------------------------------------------------------------------------


def _is_set(value: Any) -> bool:
    """Whether a config value says something — presence and non-emptiness."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _strict_limits(config: Any) -> bool:
    """The FR-8 limits posture: checking on, and unlisted channels refused."""
    limits = _sub(_section(config, "control_system"), "limits_checking")
    return bool(limits.get("enabled", False)) and not bool(
        limits.get("allow_unlisted_channels", True)
    )


def _acknowledged(config: Any) -> bool:
    """Whether the operator has acknowledged the live gateways.

    Presence only. The shipped template's example value is a real-hostname-shaped
    string, so any comparison against a known default would be a test the config
    cannot actually answer.
    """
    switch = _sub(_section(config, "control_system"), "target_switch")
    return _is_set(switch.get(ACK_LEAF))


def evaluate_eligibility(
    config: Any,
    target: str,
    *,
    direction: str = DIRECTION_AWAY,
    writes_enabled: bool | None = None,
    readonly_run: bool | None = None,
) -> Eligibility:
    """Whether *target* could be switched to, from config alone.

    Checks run in order and the first failure is the reported reason, so the
    answer names the nearest thing to fix rather than the whole list:

    1. the target resolves to a connector type at all;
    2. ``control_system.connector.<type>`` exists;
    3. its ``gateways`` table is non-empty and carries the role this deployment
       would select *for this target* (a target armed for writes with only a read
       gateway selects ``read_only`` and is eligible; a target with only a write
       gateway whose own posture leaves writes unarmed selects ``read_only`` and
       is not);
    4. that block names a ``probe_channel`` — a target that cannot prove itself
       reachable is never switched to;
    5. honesty: pointing a session at the virtual accelerator while the archiver
       resolves to the mock would pair a modelled present with an invented past;
    6. FR-8 posture, for the live machine only, and only for a switch toward it.

    Args:
        config: The full rendered config mapping.
        target: The session target being judged.
        direction: :data:`DIRECTION_AWAY` for a switch toward a target that is
            not the deployment baseline, :data:`DIRECTION_BACK` for a return to
            the baseline. Only the live machine's FR-8 gate reads it.
        writes_enabled: See :func:`derive_endpoints`.
        readonly_run: See :func:`derive_endpoints`. It moves the *selected role*,
            which is what check 3 is asked about; it never makes a target
            eligible or ineligible on its own.

    Returns:
        The verdict, its machine-readable reason, and a sentence naming the fix.
    """
    try:
        derivation = derive_endpoints(
            config, target, writes_enabled=writes_enabled, readonly_run=readonly_run
        )
    except ValueError as exc:
        # Fail-closed at the resolver becomes a reason here: eligibility is the
        # one caller whose job is to answer "no, and here is why" rather than to
        # propagate. Every other caller establishes a target exists first.
        return Eligibility(False, REASON_TARGET_UNRESOLVABLE, str(exc))

    connector_type = derivation.connector_type
    block_key = f"control_system.connector.{connector_type}"
    raw_block = connector_block(config, connector_type)

    if not isinstance(raw_block, dict) or not raw_block:
        return Eligibility(
            False,
            REASON_CONNECTOR_BLOCK_MISSING,
            f"Target {target!r} resolves to connector type {connector_type!r}, but "
            f"this config has no '{block_key}' block. Configure that block (its "
            "gateways and probe_channel) to make the target switchable.",
        )

    gateways = _sub(raw_block, "gateways")
    if not gateways:
        return Eligibility(
            False,
            REASON_GATEWAYS_MISSING,
            f"'{block_key}.gateways' is empty or missing, so there is no endpoint "
            f"to point a connector at for target {target!r}.",
        )

    selected_role = derivation.selected_role
    if selected_role not in derivation.endpoints:
        return Eligibility(
            False,
            REASON_SELECTED_ROLE_MISSING,
            f"This deployment would select the {selected_role!r} gateway for target "
            f"{target!r}, but '{block_key}.gateways.{selected_role}' is missing or "
            f"empty. Configured roles: {sorted(gateways) or 'none'}.",
        )

    if not _is_set(raw_block.get(PROBE_CHANNEL_KEY)):
        return Eligibility(
            False,
            REASON_PROBE_CHANNEL_MISSING,
            f"'{block_key}.{PROBE_CHANNEL_KEY}' is not set. The switch reads that "
            f"channel to prove target {target!r} is reachable before making it "
            "active, so a target without one is never switched to.",
        )

    if connector_type == VIRTUAL_ACCELERATOR:
        pairing = pairing_for_target(config, target)
        if pairing.is_invented_history:
            return Eligibility(
                False,
                REASON_INVENTED_HISTORY,
                f"Refusing target {target!r}: {VA_MOCK_ARCHIVER_WHY} This deployment's "
                f"archiver.type is {pairing.archiver_phrase}. Configure a real archiver "
                "to point a session at the virtual accelerator.",
            )

    if target == TARGET_LIVE and direction != DIRECTION_BACK:
        # Returning to a live deployment baseline is exempt from both: a session
        # stranded on the simulator, unable to come home, is the less safe
        # outcome of the two this gate can produce.
        if not _strict_limits(config):
            return Eligibility(
                False,
                REASON_LIMITS_POSTURE,
                "Switching to the live machine requires the strict limits posture: "
                f"'{LIMITS_ENABLED_KEY}' true and '{ALLOW_UNLISTED_KEY}' false.",
            )
        if not _acknowledged(config):
            return Eligibility(
                False,
                REASON_OPERATOR_ACK_MISSING,
                f"Switching to the live machine requires '{ACK_KEY}' to be set — the "
                "operator confirming the configured gateways really are this "
                "facility's. Set it to your live gateway's hostname.",
            )

    return Eligibility(
        True,
        None,
        f"Target {target!r} is configured: connector type {connector_type!r}, "
        f"{selected_role!r} gateway selected.",
    )


def switch_direction(target: str, session_target: str, baseline_target: str) -> str:
    """Which way a switch to *target* runs, from where the session is standing.

    A switch is a *return* only when the target is this deployment's own baseline
    and the session is somewhere else. Everything else — including a target that
    happens to be the baseline while the session already sits on it — is a switch
    away, because there is no return in progress to exempt.
    """
    if target == baseline_target and session_target != baseline_target:
        return DIRECTION_BACK
    return DIRECTION_AWAY


def target_availability(
    config: Any,
    target: str,
    session_target: str,
    baseline_target: str,
    *,
    writes_enabled: bool | None = None,
    readonly_run: bool | None = None,
) -> TargetAvailability:
    """The roster's session-relative answer for one target.

    ``available_now`` answers the switch this session would make right now;
    ``eligible_from_baseline`` answers the same question as if the session sat on
    the deployment baseline, which is the static view (see the module docstring
    for why the two legitimately differ on a live-baseline deployment).

    A target the session is already on reports ``available_now`` false with
    :data:`REASON_ALREADY_ACTIVE`: switching to the active target is a no-op, and
    that is the truthful reason it is unavailable regardless of what the config
    says about it. The config verdict is still reported, unshadowed, in
    ``eligible`` and ``eligible_from_baseline``, so the roster never loses the
    configuration picture for the target it is standing on.

    Args:
        config: The full rendered config mapping.
        target: The prospective target being judged.
        session_target: The target this session is on right now.
        baseline_target: The target the deployment's own config selects.
        writes_enabled: See :func:`derive_endpoints`.
        readonly_run: See :func:`derive_endpoints`.
    """
    direction = switch_direction(target, session_target, baseline_target)
    verdict = evaluate_eligibility(
        config,
        target,
        direction=direction,
        writes_enabled=writes_enabled,
        readonly_run=readonly_run,
    )
    from_baseline = evaluate_eligibility(
        config,
        target,
        direction=switch_direction(target, baseline_target, baseline_target),
        writes_enabled=writes_enabled,
        readonly_run=readonly_run,
    )

    if target == session_target:
        return TargetAvailability(
            target=target,
            eligible=verdict.eligible,
            available_now=False,
            reason=REASON_ALREADY_ACTIVE,
            detail=f"Target {target!r} is already the session's active target.",
            eligible_from_baseline=from_baseline.eligible,
        )

    return TargetAvailability(
        target=target,
        eligible=verdict.eligible,
        available_now=verdict.eligible,
        reason=verdict.reason,
        detail=verdict.detail,
        eligible_from_baseline=from_baseline.eligible,
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
