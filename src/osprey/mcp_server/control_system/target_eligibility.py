"""The two named checks a control-system target must pass (FR-3).

A session may be pointed at the real machine or at the simulated one, and the
question "may it be pointed there" is answered twice, at two different moments,
by two checks that are deliberately not the same check:

**ELIGIBILITY** is answered from config alone, with no child process, no socket
and no side effect. It is what the roster reports for a target nobody has
switched to yet, and it is the gate the switch consults before it spawns
anything: is there a connector block for this target, does it have a gateways
table containing the role this deployment will actually select, does it name a
``probe_channel`` to prove itself with, does the ``standin`` target's block
still select the stand-in this deployment co-deploys, would the switch create
the one archiver pairing that invents history, and — for the two machines an
operator meets hardware behaviour on — is the deployment in the posture FR-8
requires. An ineligible target reports a machine-readable reason, so the roster
can say *why* rather than merely *no*.

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
limits posture plus an operator acknowledgment, and a switch **to** the stand-in
on the limits posture alone — the stand-in really is dialled and really behaves
like hardware, but the acknowledgment it needs was already given at build time
by the ``virtual_accelerator.live_standin`` line that stood it up. Both are
waived **except** when the target is the deployment's own baseline and the
session is returning to it — stranding a session on the simulator is the less
safe outcome, so coming home is never gated, and the baseline a deployment comes
home to may be ``standin`` as readily as ``live``.
:func:`target_availability` therefore reports two answers:

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

The write posture a live session actually has
---------------------------------------------
Config is not the last word on whether writes are armed for a target. An
operator narrows one target for one session from the header chip, and that
narrowing lives in the per-(session, target) posture store
(:mod:`osprey_connectors.session_store`), not in ``config.yml``. The connector
child reads it on every write and on its own gateway selection, so a parent
that derived the *configured* posture would derive ``write_access`` for a
target the child has just connected to on ``read_only`` — and
:func:`verify_child_report`, doing its job, would abort the switch on a
disagreement that is nobody's misconfiguration.

:func:`effective_writes_for_target` is therefore what every live-session caller
in this stack passes as ``writes_enabled``: the deployment ceiling, this run's
mode and the operator's narrowing, combined once by
:func:`~osprey_connectors.session_store.effective_writes` and never restated
here. :func:`derive_endpoints` itself keeps its config-only default, because it
is also asked hypothetical questions — what would this target select under
*that* posture — and a pure derivation is what makes those answerable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osprey.audit.posture import posture_session
from osprey_connectors import session_store
from osprey_connectors.control_system.base import is_readonly_run
from osprey_connectors.control_system.va_connector import fill_gateway_ports
from osprey_connectors.honesty import VA_MOCK_ARCHIVER_WHY, pairing_for_target
from osprey_connectors.standin import (
    ARCHIVER_RECORDER_SERVICE,
    LIVE_STANDIN_PORT_KEY,
    archive_belongs_to_standin,
    live_standin_active,
)
from osprey_connectors.types import (
    INVENTED_HISTORY_TYPES,
    TARGET_LIVE,
    TARGET_STANDIN,
    VIRTUAL_ACCELERATOR,
    LimitsPosture,
    resolve_target,
    target_writes_enabled,
    type_limits_posture,
)

# -- Config keys, spelled once ---------------------------------------------

#: Operator acknowledgment that the configured live gateways are this
#: facility's. Presence/non-emptiness only — never compared to a value.
ACK_KEY = "control_system.target_switch.live_gateway_acknowledged"
#: The same key's leaf, derived from it so the two cannot drift apart.
ACK_LEAF = ACK_KEY.rsplit(".", 1)[1]
#: The build-profile key that stands the stand-in up. Named in a refusal so the
#: operator is told the one line to delete, not merely which fact is true.
STANDIN_PROFILE_KEY = "virtual_accelerator.live_standin"
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
REASON_STANDIN_NOT_DEPLOYED = "standin_not_deployed"
REASON_INVENTED_HISTORY = "invented_history"
REASON_LIMITS_POSTURE = "limits_posture"
REASON_OPERATOR_ACK_MISSING = "operator_ack_missing"
REASON_ARCHIVE_BELONGS_TO_STANDIN = "archive_belongs_to_standin"


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


def effective_writes_for_target(section: Any, target: str) -> bool:
    """Whether writes are armed for *target* on THIS session, right now.

    The deployment ceiling for the target, ANDed with this run's mode and with
    the operator's own narrowing from the header chip — the whole of rule 3 of
    the posture-store contract, delegated to
    :func:`~osprey_connectors.session_store.effective_writes` rather than
    restated. What is added here is the session key: this process's
    ``OSPREY_POSTURE_SESSION`` stamp is the store's index, and reading it in one
    place is what stops the roster, the switch and the connector child from
    each deciding for themselves whose narrowing they are answering.

    Every live-session caller of :func:`derive_endpoints` in this stack passes
    the result as ``writes_enabled``. That is not a convenience: the child
    selects its gateway from exactly this value (through the same store), so a
    parent deriving anything else would hand :func:`verify_child_report` a
    ``selected_role`` mismatch and abort a switch that was configured
    correctly.

    Args:
        section: The ``control_system:`` config section — the same unit
            :func:`~osprey_connectors.types.target_writes_enabled` takes, so a
            caller that already resolved it does not re-resolve it here.
        target: The session control target the writes would land on.

    Returns:
        ``True`` only when the deployment arms this target, this is not a
        read-only run, and the operator has not narrowed the target for this
        session. The store can only narrow: nothing it holds widens *section*.
    """
    return session_store.effective_writes(section, posture_session(), target)


def _resolved_writes(config: Any, target: str, writes_enabled: bool | None) -> bool:
    """An explicit ``writes_enabled`` override, or this session's real posture.

    ``None`` means "the caller did not say", and for a question asked *about a
    live session* the truthful default is the session's own effective posture,
    not the configured one. A caller that wants the hypothetical — the roster
    asking what a target would select if it were narrowed, a test pinning a
    posture — says so by passing the value, and is answered verbatim.
    """
    if writes_enabled is not None:
        return bool(writes_enabled)
    return effective_writes_for_target(_section(config, "control_system"), target)


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


def endpoint_is_live_standin(config: Any, endpoint: Endpoint | None) -> bool:
    """Whether a derived *endpoint* is this deployment's own live stand-in.

    The step between :func:`derive_endpoints` — which says where a session would
    land — and :func:`~osprey_connectors.standin.live_standin_active`, which says
    whether landing there is landing on the stand-in. It lives beside the
    derivation because both of its callers reach it from one: the roster, which
    labels the target an operator is told they are on, and the archiver
    recorder, which decides from the same fact whether the readings it samples
    belong in this deployment's archive. Two copies of this step could coerce
    the port differently and answer differently, and a recorder that believed it
    was sampling a stand-in while an operator was being told ``LIVE MACHINE``
    would file model readings into a real machine's archive.

    Args:
        config: The full rendered config mapping.
        endpoint: The row the connector would configure — typically
            :meth:`TargetDerivation.selected_endpoint`. ``None`` when the
            deployment resolved none.

    Returns:
        ``True`` only for an endpoint the config proves is the stand-in.
        Everything else — no endpoint, a config that is not a mapping, a port
        value that names no port — answers ``False``, the direction every
        honesty predicate in this stack fails: an endpoint is a real machine
        until the config proves otherwise.
    """
    if endpoint is None or not isinstance(config, dict):
        return False
    try:
        port = int(endpoint.port)
    except (TypeError, ValueError):
        # The endpoint's port is whatever config carried, and a value that names
        # no port names no stand-in either.
        return False
    return bool(live_standin_active(config, endpoint_host=str(endpoint.host), endpoint_port=port))


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


def _limits_posture(config: Any, connector_type: str) -> LimitsPosture:
    """The limits posture *connector_type* runs under, with its answering key.

    Per connector type rather than per deployment, for the reason the write
    posture is: a deployment with a live machine beside a virtual accelerator
    holds one posture per machine, and a relaxation written for the simulator
    must not decide what the live gate sees. The resolved value travels with the
    key that answered so a refusal sends the operator to the line they can
    actually edit — the per-type one when a per-type block spoke, the
    deployment-wide one when none did.
    """
    return type_limits_posture(_section(config, "control_system"), connector_type)


def _selected_role_missing(
    config: Any, derivation: TargetDerivation, target: str
) -> Eligibility | None:
    """Check 3's verdict: the role selected here has no endpoint to select.

    Factored out of :func:`evaluate_eligibility` because a second caller asks
    the same question about a posture the session does not have yet:
    :func:`narrowing_refusal`, which is what a surface offering the narrowing
    consults before offering it. A refusal, or ``None`` when the selected role
    is configured — the same "a verdict is a refusal" shape the switch gate
    uses, so no caller can mistake a pass for a check it forgot to read.
    """
    selected_role = derivation.selected_role
    if selected_role in derivation.endpoints:
        return None
    connector_type = derivation.connector_type
    block_key = f"control_system.connector.{connector_type}"
    raw_block = connector_block(config, connector_type)
    gateways = _sub(raw_block if isinstance(raw_block, dict) else {}, "gateways")
    return Eligibility(
        False,
        REASON_SELECTED_ROLE_MISSING,
        f"This deployment would select the {selected_role!r} gateway for target "
        f"{target!r}, but '{block_key}.gateways.{selected_role}' is missing or "
        f"empty. Configured roles: {sorted(gateways) or 'none'}.",
    )


def narrowing_refusal(config: Any, target: str) -> Eligibility | None:
    """What narrowing *target* to read-only would cost this session, if anything.

    A narrowing moves the selected gateway role, and a deployment whose block
    configures ``write_access`` alone has nothing to move *to*: the session
    would select ``read_only``, find no such gateway, and the target would stop
    being switchable at all. That is
    :data:`REASON_SELECTED_ROLE_MISSING` arriving as a consequence of an
    operator action rather than of a config edit, and a surface that offers the
    narrowing owes the operator that sentence *before* they take it.

    Read-only and hypothetical: it asks what the target would derive under the
    narrowed posture, whatever this session's posture actually is. It reads no
    store and changes nothing.

    Args:
        config: The full rendered config mapping.
        target: The target the operator is considering narrowing.

    Returns:
        The refusal narrowing would earn, or ``None`` when the target stays
        usable read-only — which is the ordinary case, since a deployment that
        can be read from at all configures a ``read_only`` gateway.
    """
    try:
        # ``readonly_run`` is pinned false rather than read: the question is
        # what the *narrowing* costs, and a run that is already read-only would
        # otherwise answer it for every target at once.
        derivation = derive_endpoints(config, target, writes_enabled=False, readonly_run=False)
    except ValueError as exc:
        return Eligibility(False, REASON_TARGET_UNRESOLVABLE, str(exc))
    return _selected_role_missing(config, derivation, target)


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
    5. for ``standin`` only: the endpoint that block selects really is the
       co-deployed stand-in — the deployment built one, on this port, over
       loopback — so the block cannot be repointed at hardware under a soft
       label (:data:`REASON_STANDIN_NOT_DEPLOYED`);
    6. honesty: pointing a session at a machine this deployment stands up for
       itself — the virtual accelerator or the stand-in — while the archiver
       resolves to the mock would pair an invented present with an invented
       past;
    7. FR-8 posture, only for a switch *toward* a target, and split by which
       machine it is: the strict limits posture applies to ``live`` **and**
       ``standin``, both of which behave like hardware; the operator
       acknowledgment applies to ``live`` alone, since the stand-in's
       equivalent was said at build time by ``virtual_accelerator.live_standin``
       (:data:`REASON_LIMITS_POSTURE`, :data:`REASON_OPERATOR_ACK_MISSING`);
    8. for ``live`` only, and only toward it: the archive is not the stand-in's.
       A deployment that records its own store beside a stand-in records the
       stand-in, and a real machine's readings must not land in that store
       (:data:`REASON_ARCHIVE_BELONGS_TO_STANDIN`).

    Args:
        config: The full rendered config mapping.
        target: The session target being judged.
        direction: :data:`DIRECTION_AWAY` for a switch toward a target that is
            not the deployment baseline, :data:`DIRECTION_BACK` for a return to
            the baseline. Only the FR-8 gates (checks 7 and 8) read it, and the
            baseline a return exempts may be any of the three targets — a
            deployment whose own ``control_system.type`` is ``live_standin``
            comes home to ``standin``.
        writes_enabled: Whether writes are armed for *target*. Unlike
            :func:`derive_endpoints`, whose default is config alone, this
            defaults to :func:`effective_writes_for_target` — the posture the
            session asking actually has, operator narrowing included. This is
            the answer a live session gets, so it has to be the answer the
            child will select its gateway on. Pass the value to ask about a
            posture other than this session's.
        readonly_run: See :func:`derive_endpoints`. It moves the *selected role*,
            which is what check 3 is asked about; it never makes a target
            eligible or ineligible on its own. The effective *writes_enabled*
            above already folds it in, so overriding it alone changes nothing.

    Returns:
        The verdict, its machine-readable reason, and a sentence naming the fix.
    """
    try:
        derivation = derive_endpoints(
            config,
            target,
            writes_enabled=_resolved_writes(config, target, writes_enabled),
            readonly_run=readonly_run,
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
    role_missing = _selected_role_missing(config, derivation, target)
    if role_missing is not None:
        return role_missing

    if not _is_set(raw_block.get(PROBE_CHANNEL_KEY)):
        return Eligibility(
            False,
            REASON_PROBE_CHANNEL_MISSING,
            f"'{block_key}.{PROBE_CHANNEL_KEY}' is not set. The switch reads that "
            f"channel to prove target {target!r} is reachable before making it "
            "active, so a target without one is never switched to.",
        )

    if target == TARGET_STANDIN and not endpoint_is_live_standin(
        config, derivation.selected_endpoint()
    ):
        endpoint = derivation.selected_endpoint()
        where = f"{endpoint.host}:{endpoint.port}" if endpoint is not None else "nowhere"
        return Eligibility(
            False,
            REASON_STANDIN_NOT_DEPLOYED,
            f"Refusing target {target!r}: '{block_key}.gateways.{selected_role}' "
            f"selects {where}, which is not the stand-in this deployment "
            f"co-deploys on '{LIVE_STANDIN_PORT_KEY}' over the loopback "
            "interface. The stand-in target names the soft IOC the deployment "
            "runs for itself, so a block pointed anywhere else would put a real "
            f"machine behind a soft label. Point '{block_key}.gateways' at the "
            f"stand-in's own port on loopback, or use {TARGET_LIVE!r} for the "
            "machine this facility authored.",
        )

    if connector_type in INVENTED_HISTORY_TYPES:
        pairing = pairing_for_target(config, target)
        if pairing.is_invented_history:
            return Eligibility(
                False,
                REASON_INVENTED_HISTORY,
                f"Refusing target {target!r}: the archive belongs to the machine. A "
                f"session pointed at {connector_type!r} needs an archiver that "
                f"recorded that machine: {VA_MOCK_ARCHIVER_WHY} This deployment's "
                f"archiver.type is {pairing.archiver_phrase}. Set `archiver.type` to "
                "a real archiver — this deployment's own store — before switching a "
                "session onto this target.",
            )

    # Returning to a deployment's own baseline is exempt from FR-8 — a session
    # stranded on the simulator, unable to come home, is the less safe outcome
    # of the two this gate can produce — and the baseline may be either machine
    # a session can be careful around, so the exemption follows the direction
    # rather than naming a target.
    switching_away = direction != DIRECTION_BACK

    if switching_away and target in (TARGET_LIVE, TARGET_STANDIN):
        # The strict limits posture guards both machines an operator meets
        # hardware behaviour on. The stand-in really is dialled, really refuses
        # out-of-limit writes and really carries `real_machine` — a rehearsal on
        # a permissive posture would rehearse the wrong facility.
        posture = _limits_posture(config, connector_type)
        if not posture.strict:
            # Both keys off the posture, so a per-type block that answered is
            # the line the operator is sent to and the deployment-wide one it
            # overrides is not mentioned at all.
            enabled_key = posture.key("enabled")
            allow_unlisted_key = posture.key("allow_unlisted_channels")
            return Eligibility(
                False,
                REASON_LIMITS_POSTURE,
                f"Switching to target {target!r} requires the strict limits posture: "
                f"'{enabled_key}' true and '{allow_unlisted_key}' false.",
            )

    if target == TARGET_LIVE and switching_away:
        # The acknowledgment is the live machine's alone. It is the operator
        # saying the configured gateways really are this facility's, and the
        # stand-in's equivalent was already said at build time by the profile
        # line that stood it up: `virtual_accelerator.live_standin` names the
        # port, and nothing about that endpoint is a gateway anyone could
        # mistake for the facility's.
        if not _acknowledged(config):
            return Eligibility(
                False,
                REASON_OPERATOR_ACK_MISSING,
                f"Switching to the live machine requires '{ACK_KEY}' to be set — the "
                "operator confirming the configured gateways really are this "
                "facility's. Set it to your live gateway's hostname.",
            )
        if archive_belongs_to_standin(config):
            return Eligibility(
                False,
                REASON_ARCHIVE_BELONGS_TO_STANDIN,
                f"Refusing target {target!r}: the deployment behind this archive "
                f"store records it from the stand-in ('{STANDIN_PROFILE_KEY}' stood "
                f"one up and its '{ARCHIVER_RECORDER_SERVICE}' samples it), so the "
                "history in that store is the stand-in's. Selecting the live machine "
                "would splice a real machine's readings onto a stand-in's past, in "
                "one store nothing afterwards can tell apart. Switch that "
                f"deployment's '{ARCHIVER_RECORDER_SERVICE}' service off, or drop "
                f"'{STANDIN_PROFILE_KEY}' so it stops standing a stand-in up — in "
                "the hosting profile, for an attached session — and rebuild before "
                "switching a session onto the live machine.",
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
        writes_enabled: See :func:`evaluate_eligibility`. Resolved once here and
            handed to both verdicts below: the two differ only in direction, and
            a posture read twice could differ between them if the operator
            narrowed the target in between.
        readonly_run: See :func:`derive_endpoints`.
    """
    resolved_writes = _resolved_writes(config, target, writes_enabled)
    direction = switch_direction(target, session_target, baseline_target)
    verdict = evaluate_eligibility(
        config,
        target,
        direction=direction,
        writes_enabled=resolved_writes,
        readonly_run=readonly_run,
    )
    from_baseline = evaluate_eligibility(
        config,
        target,
        direction=switch_direction(target, baseline_target, baseline_target),
        writes_enabled=resolved_writes,
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
