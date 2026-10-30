"""Connector type constants, and the fallbacks a factory applies to them.

Single source of truth for built-in connector type name strings.
Custom connectors use dotted module paths (e.g., 'mypackage.MoatConnector')
and don't need constants here.

The resolvers at the bottom are here rather than in
:mod:`osprey_connectors.factory` so that a guard deciding what a deployment
*will build* can share them with the factory that builds it. A guard which
re-implements "what does this config select" is a guard that can disagree with
the answer, and the disagreement is a bypass rather than a discrepancy — see
:mod:`osprey_connectors.honesty`.

:func:`resolve_archiver_settings` is the same arrangement for the other half of
an archiver selection: which block its ``connect()`` is handed. The build
refuses what it would refuse, from the same helpers.

:func:`resolve_target` extends that to the run-time question "which machine is
this session pointed at". It is the same shape of answer for the same reason:
several holders follow a control target — the connector-host process, its child,
an executor sandbox — and a holder that translates ``live`` into a connector type
privately can route a tool call somewhere the roster never claimed. A target
names a *machine*, and there are three of them: the facility's own control
system (``live``), the virtual accelerator (``va``), and the live stand-in
(``standin``) — a soft IOC a deployment runs for itself, which is a third
machine with a connector block of its own rather than a mode of the first. That
is what keeps ``live`` meaning the machine the facility authored on a deployment
that runs the stand-in beside it. The target is an *argument* here, never
something this module reads from the environment: an environment-reading
override would make the honesty predicate describe a deployment the process is
no longer running.

:func:`tier_writes` answers "how far may this deployment write" from the one tier key,
``control_system.writes``, for the same reason. The lint that reads the config,
the profile card that tells an operator what they hold, and the connector that
would perform the write all have to agree about the answer. Two readings of one
posture is one of them being wrong about a machine somebody can move.

:class:`LimitsPosture` is the same story for limits checking, which is likewise
per connector type: a deployment may run its simulator ``optional`` while its
live machine runs ``exclusive``. It carries the resolved posture together
with the config key that answered it, so a refusal names the line an operator
would have to edit rather than one some per-type block overrides.

Where the write tier reads an unusable value as ``never`` and is done, the
limits family cannot: "no limits checking configured" is itself a permissive
answer, so a leaf written as something no reader can turn into a boolean is
reported as a block that failed to state it rather than as a block that stated
nothing. Such a posture answers neither leaf, and a caller that must act blocks
every write instead of waving them through unchecked.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "CLI_ARCHIVER_TYPES",
    "CLI_CONTROL_SYSTEM_TYPES",
    "CONTROL_TARGETS",
    "DEFAULT_VA_PORT",
    "DOOCS",
    "DOOCS_ARCHIVER",
    "EPICS",
    "EPICS_ARCHIVER",
    "INVENTED_HISTORY_TYPES",
    "IN_PROCESS",
    "LIMITS_CHECKING_LEAF",
    "LIMITS_LEAVES",
    "LIMITS_MODES",
    "LIMITS_MODE_EXCLUSIVE",
    "LIMITS_MODE_OPTIONAL",
    "LIVE_STANDIN",
    "LimitsPosture",
    "MOCK_ARCHIVER",
    "MONGODB_ARCHIVER",
    "MYA_ARCHIVER",
    "RETIRED_CONTROL_SYSTEM_TYPES",
    "SERVED",
    "SERVING_KEY",
    "SERVING_LEAF",
    "SERVING_MODES",
    "SET_CONTROL_SYSTEM_TYPES",
    "SIMULATED_TYPES",
    "STANDIN_TYPES",
    "TANGO",
    "TARGET_LIVE",
    "TARGET_STANDIN",
    "TARGET_VA",
    "TIER_WORDS",
    "TRANSPORT_CA",
    "TRANSPORT_DOOCS",
    "TRANSPORT_IN_PROCESS",
    "TRANSPORT_TANGO",
    "UNSET_TYPE_WARNING",
    "VIRTUAL_ACCELERATOR",
    "WRITES_KEY",
    "archiver_settings_errors",
    "archiver_settings_key",
    "baseline_target",
    "configured_targets",
    "connector_transport",
    "incomplete_limits_blocks",
    "is_simulated",
    "limits_database_required",
    "machine_kind",
    "most_restrictive_limits_posture",
    "published_machine_kind",
    "resolve_archiver_settings",
    "resolve_archiver_type",
    "resolve_control_system_type",
    "resolve_serving",
    "resolve_target",
    "retired_type_message",
    "speaks_channel_access",
    "switch_capable",
    "talks_to_network",
    "target_configured",
    "target_limits_posture",
    "tier_allows",
    "tier_limits_database_targets",
    "tier_writes",
    "type_limits_posture",
]

logger = logging.getLogger("osprey_connectors.types")

#: Config shapes the second-real-block warning has already been said about, so
#: a resolver asked on every roster render says it once. Keyed by the shape
#: itself — a deployment has one, and a test that edits a config gets a fresh
#: line for the shape it edited to.
_SECOND_REAL_BLOCK_WARNED: set[tuple[str, tuple[str, ...]]] = set()

# -- Control system connector types (have implementations) --
EPICS = "epics"
VIRTUAL_ACCELERATOR = "virtual_accelerator"
DOOCS = "doocs"
TANGO = "tango"

#: The live stand-in: a facility-shaped soft IOC a deployment runs for itself.
#: A connector type of its own — served by the EPICS connector, but keyed apart
#: from ``epics`` so that the facility's authored ``epics`` block stays the one
#: thing ``live`` can mean, and the stand-in gets its own
#: ``control_system.connector.live_standin`` block to be configured from.
LIVE_STANDIN = "live_standin"

#: Channel Access port the shipped Virtual Accelerator soft-IOC serves on. It is
#: here rather than only in the VA connector because the deployment layer fills
#: the same default when it dials the VA, and it must not import a concrete
#: connector (and the EPICS client stack behind it) to learn one integer.
DEFAULT_VA_PORT = 5064

# -- Where the simulator runs --
#: The leaf under ``control_system.connector.virtual_accelerator`` that says
#: where the simulator runs: ``served`` from its container over Channel Access,
#: or ``in_process`` inside the process that asks, with no network at all.
SERVING_LEAF = "serving"
SERVED = "served"
IN_PROCESS = "in_process"
SERVING_MODES = (SERVED, IN_PROCESS)
SERVING_KEY = f"control_system.connector.{VIRTUAL_ACCELERATOR}.{SERVING_LEAF}"

# -- Transports: the wire a connector speaks --
TRANSPORT_IN_PROCESS = "in_process"
TRANSPORT_CA = "ca"
TRANSPORT_DOOCS = "doocs"
TRANSPORT_TANGO = "tango"

#: The wire each built-in type speaks when it is served over a network. There
#: is no PVAccess row: the EPICS-family connectors carry PVAccess channels beside
#: Channel Access, behind the same gateway gate, so ``ca`` covers both. A
#: connector that speaks only PVAccess adds its own row.
_TRANSPORTS = {
    EPICS: TRANSPORT_CA,
    VIRTUAL_ACCELERATOR: TRANSPORT_CA,
    LIVE_STANDIN: TRANSPORT_CA,
    DOOCS: TRANSPORT_DOOCS,
    TANGO: TRANSPORT_TANGO,
}

# -- Retired type names --
#: Type names no deployment may state any more, each with the type that now
#: means it. A table rather than a missing registration, so that every site
#: refusing one says what to write instead.
RETIRED_CONTROL_SYSTEM_TYPES = {"mock": VIRTUAL_ACCELERATOR}

#: What a factory says when it builds from a section that states no type.
UNSET_TYPE_WARNING = (
    f"control_system.type is not set; defaulting to '{VIRTUAL_ACCELERATOR}' "
    "served in process (no network). Set control_system.type explicitly to "
    "select a connector."
)

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"
MONGODB_ARCHIVER = "mongodb_archiver"
DOOCS_ARCHIVER = "doocs_archiver"
MYA_ARCHIVER = "mya_archiver"

# -- CLI choice lists (only types with implementations) --
CLI_CONTROL_SYSTEM_TYPES = [EPICS, VIRTUAL_ACCELERATOR, DOOCS, TANGO]
CLI_ARCHIVER_TYPES = [
    MOCK_ARCHIVER,
    EPICS_ARCHIVER,
    MONGODB_ARCHIVER,
    DOOCS_ARCHIVER,
    MYA_ARCHIVER,
]

#: Settable, not initable. A deployment may be pointed at its stand-in once it
#: has one — ``osprey set connector=live_standin`` is how a session starts on
#: the soft IOC — but ``osprey init`` never offers it: a fresh project has no
#: stand-in to point at, so a project that began on one would name a machine
#: that does not exist. Hence one list for what the ``connector:`` shorthand
#: accepts and another, narrower one for what ``init`` will materialize.
SET_CONTROL_SYSTEM_TYPES = [*CLI_CONTROL_SYSTEM_TYPES, LIVE_STANDIN]

# -- Control targets --
# What a *deployment* can be pointed at, as opposed to what a *config* selects.
# Deliberately three names and not one per connector type: a target names a
# machine (the facility's own, the virtual accelerator, the stand-in soft IOC),
# and which connector type reaches that machine is this module's job to answer,
# not the caller's.
TARGET_LIVE = "live"
TARGET_VA = "va"
TARGET_STANDIN = "standin"
CONTROL_TARGETS = [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]

#: The topology those three names bound, spelled once and quoted by every
#: answer that runs into it. One deployment describes one real machine: the
#: target vocabulary has a single name for it, and the connector table is keyed
#: by connector type, so a second machine on the same protocol cannot be written
#: down at all. That is a product shape rather than an oversight — a complex
#: whose sessions must reach two real machines runs one deployment per machine,
#: which is also what keeps a session's posture, limits and archive about one
#: machine each. Said here so a deployer meets it in the error and in the how-to
#: rather than in the silence of a configured block nothing ever offers.
_ONE_REAL_MACHINE = (
    "A deployment names one real machine — exactly one connector block that is "
    "neither simulated nor the stand-in — and 'live' is that machine. A complex "
    "with two real machines runs one deployment per machine."
)

# -- Write posture --
#: The login tier's write ceiling, dotted as a caller spells it for
#: ``get_config_value``: ``"never"``, ``"asked"`` or ``"allowed"``, the furthest
#: position a login may pick. Target-blind — a tier that allows writes allows
#: them for every machine, and which machine is actually written is the
#: session's switch to decide, not the tier's. Read by :func:`tier_writes`; when
#: the key is absent the tier is ``never``.
WRITES_KEY = "control_system.writes"

#: :data:`WRITES_KEY`'s leaf inside the ``control_system:`` section.
_WRITES_LEAF = "writes"

#: The two values of :data:`WRITES_KEY` that arm anything, and the refusing word.
_TIER_ALLOWED: Literal["allowed"] = "allowed"
_TIER_ASKED: Literal["asked"] = "asked"
_TIER_NEVER: Literal["never"] = "never"

#: The three tier words, narrowest first: every answer :func:`tier_writes` gives.
TIER_WORDS: tuple[Literal["never"], Literal["asked"], Literal["allowed"]] = (
    _TIER_NEVER,
    _TIER_ASKED,
    _TIER_ALLOWED,
)

#: Types that serve a machine nobody has to be careful around. They are the
#: reason ``live`` cannot simply be "whatever the config selects": a deployment
#: whose baseline is one of these has not yet said what its real machine is.
SIMULATED_TYPES = (VIRTUAL_ACCELERATOR,)


def is_simulated(connector_type: str) -> bool:
    """Whether a connector type serves the simulator rather than a machine.

    A type predicate: the simulator answers True whether it is served or runs
    in process. It never says whether a connector dials a network; that is
    :func:`speaks_channel_access` and :func:`talks_to_network`.
    """
    return connector_type in SIMULATED_TYPES


#: Types that serve the live stand-in. Reachable only through the ``standin``
#: target: a stand-in is a machine in its own right, so it is never a candidate
#: for a deployment's ``live`` one — not as the baseline the derivation starts
#: from, and not as a block it falls back to. A deployment running the stand-in
#: beside its facility's own block still has exactly one real machine.
STANDIN_TYPES = (LIVE_STANDIN,)

#: Types whose history is invented rather than recorded, so pairing one with a
#: synthesizing archiver leaves nothing that ever tells a reader the data is
#: made up. Both are machines a deployment stands up for itself, and neither has
#: an archive of its own — see :mod:`osprey_connectors.honesty`.
INVENTED_HISTORY_TYPES = (VIRTUAL_ACCELERATOR, LIVE_STANDIN)

#: The target each self-standing machine's type is the baseline of. A type
#: absent from this table describes the facility's own machine, hence ``live``.
_BASELINE_TARGETS = {VIRTUAL_ACCELERATOR: TARGET_VA, LIVE_STANDIN: TARGET_STANDIN}


def _resolve_type(section: Any, fallback: str) -> str:
    """The connector type a factory builds from a config section.

    A section that is missing, is not a mapping, or carries no usable ``type``
    resolves to *fallback* — the factory's documented fail-closed default, which
    it announces with a ``… is not set; defaulting to …`` warning. For a control
    system that default is the simulator in process, which dials nothing; the
    transport (:func:`connector_transport`), never the type, says whether a
    connector dials a network. Empty and
    ``None`` count as absent (YAML gives ``None`` for a bare ``type:``); any
    other value is returned as written, so a typo reaches the factory's
    "Unknown … type" error rather than being quietly rounded to something.
    """
    value = section.get("type") if isinstance(section, dict) else None
    return str(value) if value else fallback


def resolve_archiver_type(section: Any) -> str:
    """The archiver an ``archiver:`` config section actually selects.

    Absent means the mock: a config that says nothing about its archiver is a
    config that gets the synthesizing one, which is the fact the honesty rule is
    really about.
    """
    return _resolve_type(section, MOCK_ARCHIVER)


def resolve_control_system_type(section: Any) -> str:
    """The control system a ``control_system:`` config section actually selects.

    A section that states no type selects the simulator, and
    :func:`resolve_serving` runs it in process, so it dials nothing. The
    transport (:func:`connector_transport`), never the type, says whether a
    connector dials a network.
    """
    return _resolve_type(section, VIRTUAL_ACCELERATOR)


def _states_type(section: Any) -> bool:
    return isinstance(section, dict) and bool(section.get("type"))


def resolve_serving(section: Any) -> str:
    """Where the simulator a ``control_system:`` section selects runs.

    A section that states no type is the fail-safe default, the simulator in
    process, and the leaf is not read: a section that names nothing must dial
    nothing. A stated ``virtual_accelerator`` answers
    :data:`SERVING_KEY`, else :data:`SERVED`. Any other stated type is
    :data:`SERVED`: only the simulator has an in-process venue.

    Raises:
        ValueError: when the leaf holds a value outside :data:`SERVING_MODES`.
    """
    if not _states_type(section):
        return IN_PROCESS
    if resolve_control_system_type(section) != VIRTUAL_ACCELERATOR:
        return SERVED
    connector = section.get("connector")
    block = connector.get(VIRTUAL_ACCELERATOR) if isinstance(connector, dict) else None
    value = block.get(SERVING_LEAF) if isinstance(block, dict) else None
    if value is None:
        return SERVED
    if value not in SERVING_MODES:
        raise ValueError(f"{SERVING_KEY} is {value!r}; it is {SERVED!r} or {IN_PROCESS!r}.")
    return str(value)


def connector_transport(section: Any, connector_type: str | None = None) -> str | None:
    """The wire a connector built from *section* speaks.

    Computed once, here, from the type and the serving leaf, so every check that
    asks "does this speak Channel Access" or "does this dial a network" reads
    one derived fact and never the type word. *connector_type* defaults to
    :func:`resolve_control_system_type` of *section*; a caller building for a
    target passes the type the target resolves to.

    Returns:
        :data:`TRANSPORT_IN_PROCESS` for the simulator served in process, the
        type's :data:`_TRANSPORTS` row otherwise, and ``None`` for a type with
        no row (a dotted custom connector).

    Raises:
        ValueError: when the serving leaf holds an unknown value.
    """
    if connector_type is None:
        connector_type = resolve_control_system_type(section)
    if connector_type == VIRTUAL_ACCELERATOR and resolve_serving(section) == IN_PROCESS:
        return TRANSPORT_IN_PROCESS
    return _TRANSPORTS.get(connector_type)


def speaks_channel_access(section: Any, connector_type: str | None = None) -> bool:
    """Whether a connector built from *section* speaks real Channel Access.

    The facility's own EPICS machine, the simulator served from its container,
    or the live stand-in soft IOC. The queue worker builds its devices over
    Channel Access, so these are the connectors plans can execute against today
    and every other one browses. That is a property of the worker's device
    layer, not of the plan stack: a facility whose machine speaks another
    protocol executes plans once a device layer for it exists, and its
    transport joins this one — this is not a statement that no other protocol
    can ever run plans.
    """
    return connector_transport(section, connector_type) == TRANSPORT_CA


def talks_to_network(section: Any, connector_type: str | None = None) -> bool:
    """Whether a connector built from *section* dials anything at all.

    False only for the simulator served in process. A transport this module has
    no row for (a dotted custom connector) answers True: what it dials is not
    known, so it is not assumed to dial nothing.
    """
    return connector_transport(section, connector_type) != TRANSPORT_IN_PROCESS


def retired_type_message(value: str) -> str:
    """The one sentence every site refusing a retired type name quotes."""
    replacement = RETIRED_CONTROL_SYSTEM_TYPES.get(value, VIRTUAL_ACCELERATOR)
    return (
        f"`{value}` is retired: the simulator in process is `control_system.type: "
        f"{replacement}` with `{SERVING_KEY}: {IN_PROCESS}` (`osprey set "
        f"connector={replacement} config.{SERVING_KEY}={IN_PROCESS}`), then "
        "rebuild with `osprey build`."
    )


#: The one block under ``archiver:`` a connector's settings are read from,
#: whichever route selected the connector. It has no dot in it, so a profile's
#: ``archiver.settings.<leaf>`` renders to exactly the block the factory reads.
_ARCHIVER_SETTINGS_LEAF = "settings"

#: Keys of the ``archiver:`` section that are not a connector's settings block:
#: the selector, the settings block itself, and archiver_read's point budget.
_ARCHIVER_SECTION_LEAVES = ("type", _ARCHIVER_SETTINGS_LEAF, "auto_bin_points")


def _own_name_block(connector_type: str) -> str | None:
    """The older spelling of a short-named archiver's settings, or ``None``.

    A built-in, or a connector registered under a short name, may still keep
    its settings in a block carrying its own name (``archiver.epics_archiver``).
    A dotted module path has no such block: a profile splits keys on every dot,
    so a block keyed by one could never have been written from ``config:``.
    """
    return None if "." in connector_type else connector_type


def _archiver_settings_homes(section: Any) -> list[tuple[str, Any]]:
    """Every block in *section* that configures the selected archiver, in order."""
    if not isinstance(section, dict):
        return []
    homes: list[tuple[str, Any]] = []
    if _ARCHIVER_SETTINGS_LEAF in section:
        homes.append((_ARCHIVER_SETTINGS_LEAF, section[_ARCHIVER_SETTINGS_LEAF]))
    own = _own_name_block(resolve_archiver_type(section))
    if own is not None and own != _ARCHIVER_SETTINGS_LEAF and own in section:
        homes.append((own, section[own]))
    return homes


def _two_homes(own: str) -> str:
    return (
        f"`archiver.settings` and `archiver.{own}` both configure {own}; keep one "
        f"— `archiver.settings` is the spelling for every archiver"
    )


def _not_a_block(key: str, value: Any) -> str:
    return (
        f"`archiver.{key}` is {value!r}, not a block; it holds the selected "
        f"archiver's settings as a mapping"
    )


def archiver_settings_key(section: Any) -> str:
    """The dotted key of the block that configures *section*'s archiver.

    ``archiver.<type>`` when a short-named archiver is configured by its own-name
    block alone, else ``archiver.settings`` — so a message about a setting names
    the line an operator actually wrote.
    """
    homes = _archiver_settings_homes(section)
    home = homes[0][0] if len(homes) == 1 else _ARCHIVER_SETTINGS_LEAF
    return f"archiver.{home}"


def resolve_archiver_settings(section: Any) -> dict[str, Any]:
    """The settings block the selected archiver's ``connect()`` is handed.

    ``archiver.settings`` for every archiver; for a short-named one, its own-name
    block when ``settings`` is absent. Absent, or a bare ``settings:``, is ``{}``.

    Raises:
        ValueError: when both spellings are present, or the block is not a
            mapping. :func:`archiver_settings_errors` refuses the same configs
            at build time, with the same words.
    """
    homes = _archiver_settings_homes(section)
    if len(homes) > 1:
        raise ValueError(_two_homes(homes[1][0]))
    if not homes or homes[0][1] is None:
        return {}
    key, value = homes[0]
    if not isinstance(value, dict):
        raise ValueError(_not_a_block(key, value))
    return dict(value)


def archiver_settings_errors(section: Any) -> list[str]:
    """Every block under a rendered ``archiver:`` section that no archiver reads, named.

    A lint, like :func:`incomplete_limits_blocks`, and read by the build over the
    config a deployment runs. It reports, in section order: each mapping under a
    key no reader takes (anything but :data:`_ARCHIVER_SECTION_LEAVES`, a built-in
    archiver's name, or the selected type's own name when it has no dot); then
    the two refusals :func:`resolve_archiver_settings` raises, in its words. A
    scalar is not a block and is not reported. Never raises.
    """
    if not isinstance(section, dict):
        return []
    selected = resolve_archiver_type(section)
    own = _own_name_block(selected)
    known = {*_ARCHIVER_SECTION_LEAVES, *CLI_ARCHIVER_TYPES, *([own] if own else [])}
    errors = [
        f"`{key}:` under `archiver:` is a block no archiver reads; the selected "
        f"archiver, {selected}, takes its settings from `archiver.settings`"
        for key, value in section.items()
        if isinstance(value, dict) and key not in known
    ]
    homes = _archiver_settings_homes(section)
    if len(homes) > 1:
        errors.append(_two_homes(homes[1][0]))
    errors.extend(
        _not_a_block(key, value)
        for key, value in homes
        if value is not None and not isinstance(value, dict)
    )
    return errors


def resolve_target(section: Any, target: Any) -> str:
    """The connector type a session *target* selects, in a given deployment.

    The returned type is also the key of the block the factory reads its
    settings from — ``control_system.connector.<type>`` — so one string answers
    both "what do I build" and "where are its gateways".

    ``va`` and ``standin`` are the same answer everywhere: ``va`` is the
    deployment's simulator, and the stand-in is the soft IOC the deployment runs
    for itself, configured from the block that carries its name. The factory
    serves ``va`` in the venue ``connector.virtual_accelerator.serving`` names,
    and only a deployment whose own type is ``virtual_accelerator`` may serve it
    in process; the type this function answers is the same in either venue.
    ``live`` is deployment-specific and is the half that can refuse:

    - When the section's own type is a real control system, that is the live
      machine, and it is returned as written — including a value this module
      does not recognize, which reaches the factory's "Unknown … type" error
      exactly as :func:`resolve_control_system_type` already lets it. A second
      non-simulated block beside it is not a second target: no name reaches it
      (:data:`_ONE_REAL_MACHINE`), so it is warned about rather than skipped in
      silence.
    - When the section's own type is simulated or is a stand-in (or absent,
      which resolves to the simulator in process), the deployment has not named
      its real machine there, so the live type is taken from the connector
      table: exactly one block that is neither simulated nor a stand-in means
      the deployment has said which machine it means. None, or more than one, raises. The stand-in
      is skipped on both sides of that derivation, so standing it up beside a
      facility's ``epics`` block never makes ``live`` ambiguous and never
      answers it with the stand-in.

    Nothing is inferred from the absence of evidence. A deployment that has
    never been told about its real machine gets an error naming what it would
    have to configure, because the alternative — resolving ``live`` to the one
    connector type that talks to hardware because it is the usual one — is a
    tool call arriving at a real machine on the strength of a guess.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        target: The control target, one of :data:`CONTROL_TARGETS`. An argument,
            never read from the environment, and never defaulted.

    Returns:
        The connector type, which doubles as the connector sub-block key.

    Raises:
        ValueError: If *target* is not a known target (including ``None`` and
            blank), or if it is ``live`` on a deployment with no single
            derivable live control system.
    """
    if target == TARGET_VA:
        return VIRTUAL_ACCELERATOR
    if target == TARGET_STANDIN:
        return LIVE_STANDIN
    if target == TARGET_LIVE:
        return _live_type(section)
    raise ValueError(
        f"Unknown control target {target!r}. Valid targets are "
        f"{TARGET_LIVE!r}, {TARGET_VA!r} and {TARGET_STANDIN!r}, spelled "
        f"exactly. {_ONE_REAL_MACHINE} A control target is always stated, never "
        "defaulted — there is no target a caller gets by saying nothing, "
        "because the one it would get could be the real machine."
    )


def baseline_target(section: Any) -> str:
    """The target a deployment's own ``control_system:`` section selects.

    ``va`` for the simulator, in whichever venue its leaf names (a section
    that states no type is the simulator in process, so it is on ``va``),
    ``standin`` for the live stand-in, and ``live`` for every real control
    system.

    Beside :func:`resolve_control_system_type` rather than restated by each
    holder, because "which target am I on when nobody has switched" is asked by
    the supervisor, the switch predicate below, and every baseline-pinned reader
    — and three answers to it is three ways for one deployment to be described.
    """
    return _BASELINE_TARGETS.get(resolve_control_system_type(section), TARGET_LIVE)


def switch_capable(section: Any) -> bool:
    """Whether a deployment gives a session more than one target to be pointed at.

    The predicate itself, in the module that already owns "which connector type
    does a target mean here". Two layers ask the question and must never answer
    it differently: the controls server decides from it whether its tools are
    served by a connector-host child, and the build decides from it whether the
    agent's frozen safety rule describes a switchable machine at all. A build
    that promises a switch the runtime will not perform is worse than one that
    never mentions it.

    One rule: a deployment is switch-capable when its section resolves and
    **at least two targets are configured** (:func:`configured_targets`, the
    same resolve-and-read-the-block enumeration every roster walks; the
    baseline is always one of them).

    Which two is not this predicate's business: a facility rehearsing on a
    stand-in beside its simulator, with no live machine authored yet, has
    exactly the two-machine world the switch exists for — demanding the
    ``live``/``va`` pair specifically would lock that deployment's posture
    toggles for want of a machine it never claimed to have.

    Deliberately *not* checked: ``probe_channel``, the gateways table, the
    operator acknowledgment. Those decide whether a target may be switched *to*
    — a per-switch question answered with a reason the operator is told. This is
    the coarser question of whether the deployment is in the multi-target world.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes. Callers holding a whole
            rendered config pass ``config.get("control_system")``.

    Returns:
        ``True`` when the section resolves and configures at least two
        targets. Never raises: a section that is not a mapping or does not
        resolve is not capable.
    """
    if not isinstance(section, dict):
        return False
    try:
        return len(configured_targets(section)) >= 2
    except ValueError:
        return False


def target_configured(section: Any, target: Any) -> bool:
    """Whether *section* carries the connector block *target* is served by.

    Both halves of "does this deployment have that machine", asked the way the
    runtime asks them: :func:`resolve_target` names the connector type the
    target selects HERE — refusing rather than guessing a live machine the
    config never described — and that type's own
    ``control_system.connector.<type>`` block is what such a connector is
    configured from. A target that does not resolve is not one this deployment
    has, and an empty block is not a configured one.

    The one spelling of a question every enumerator asks:
    :func:`configured_targets` of each machine that is not the baseline —
    which is also what :func:`switch_capable` counts — and the deployment
    layer's Reach Contracts of the target a projected port would serve. Two
    predicates over one pair of keys would already be two chances to disagree
    about which machines a deployment has.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        target: The control target being asked about.

    Returns:
        Whether the target resolves and its block is a non-empty mapping. Never
        raises: an unresolvable target is simply not configured.
    """
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return False
    connector = section.get("connector") if isinstance(section, dict) else None
    block = connector.get(connector_type) if isinstance(connector, dict) else None
    return isinstance(block, dict) and bool(block)


def configured_targets(section: Any) -> list[str]:
    """The targets a session on this deployment can actually be pointed at.

    In :data:`CONTROL_TARGETS` order, which is the order every render, roster
    and state file reads them in. The deployment's :func:`baseline_target` is
    always one of them — a session sits on it whether or not the config wrote a
    block for the connector ``control_system.type`` builds — and each of the
    others is here when :func:`target_configured` says so, which is the same
    resolve-and-read-the-block pair :func:`switch_capable` and the deployment
    layer's Reach Contracts decide it by.

    The order is the constant's and not the baseline's, so a deployment that
    gained no target gains no reordering: the two-target deployments that have
    no stand-in enumerate exactly as they did before ``standin`` existed, down
    to the order their rendered ``settings.json`` lists them in.

    This list, and never the constant, is what an enumerator walks — a roster,
    a prober, a posture render, a state file. :data:`CONTROL_TARGETS` is the
    *vocabulary*: it grows when a new kind of machine exists, and a holder that
    loops it hands every deployment a slot for every machine anybody could run.
    Widening it to three would otherwise grow a ``standin`` row, probe and
    posture on deployments with no ``control_system.connector.live_standin``
    block at all — a machine that is not there, described as if it were.
    Which targets are configured is a property of the rendered config, so it is
    read from the config.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.

    Returns:
        The configured targets in :data:`CONTROL_TARGETS` order, the baseline
        among them. Never raises and never empty: a section that is missing or
        malformed still has a baseline, and that one target is what such a
        deployment is on.
    """
    baseline = baseline_target(section)
    return [
        target
        for target in CONTROL_TARGETS
        if target == baseline or target_configured(section, target)
    ]


def tier_writes(section: Any) -> Literal["never", "asked", "allowed"]:
    """The write ceiling this deployment's login tier sets: ``never``, ``asked`` or ``allowed``.

    :data:`WRITES_KEY` decides when the section carries it, and then only the
    literal strings ``"asked"`` and ``"allowed"`` arm, matched exactly; ``"never"``,
    any other string (the switch's retired ``"on"`` among them), ``True``, ``None``
    and every other value answer ``never``. When the key is absent the tier is
    ``never``: the retired arming leaves are not read at run time, so a render
    built before the tier key existed writes nowhere until it is rebuilt (the
    build's normalizer writes the key into every render).

    Target-blind by design: a tier that allows writes allows them for every
    machine. The tier is a ceiling, not a selection — the session's switch picks
    the machine.

    Never raises. A section that is not a mapping answers ``never``.
    """
    value = section.get(_WRITES_LEAF) if isinstance(section, dict) else None
    if isinstance(value, str):
        if value == _TIER_ASKED:
            return _TIER_ASKED
        if value == _TIER_ALLOWED:
            return _TIER_ALLOWED
    return _TIER_NEVER


def tier_allows(section: Any) -> bool:
    """Whether the tier arms anything: :func:`tier_writes` is not ``never``. Never raises.

    This reads the ceiling only: ``asked`` and ``allowed`` both answer true, so
    whether a write prompts is not decided here.
    """
    return tier_writes(section) != _TIER_NEVER


#: What kind of machine each control target names by its name alone, for the
#: words a refusal uses about it. ``standin`` is ``real`` here: a name cannot
#: tell this deployment's stand-in container from a ``standin`` target pointed at
#: anything else, so only the published report (:func:`published_machine_kind`)
#: may call it the rehearsal machine.
_MACHINE_KINDS: dict[str, Literal["real", "rehearsal", "simulator"]] = {
    TARGET_LIVE: "real",
    TARGET_STANDIN: "real",
    TARGET_VA: "simulator",
}

#: The kind a published report may claim for ``standin``. Anything else it says
#: about that target, or nothing at all, reads as the real machine.
_REHEARSAL: Literal["rehearsal"] = "rehearsal"

#: The report key the controls server publishes each target's kind under.
_MACHINE_KIND_KEY = "machine_kind"


def machine_kind(section: Any, target: Any) -> Literal["real", "rehearsal", "simulator"] | None:
    """The kind of machine *target* names by its name: ``real`` or ``simulator``.

    ``live`` and ``standin`` are the real machine, ``va`` the simulator. The
    name never answers ``rehearsal``: whether a ``standin`` target really is
    this deployment's stand-in is an endpoint fact only the controls server's
    report states, read by :func:`published_machine_kind`. ``None`` when
    :func:`resolve_target` cannot resolve the target in this deployment — an
    unknown target, or ``live`` on a deployment that never named its real
    machine. Never raises.
    """
    try:
        resolve_target(section, target)
    except ValueError:
        return None
    return _MACHINE_KINDS.get(target)


def _published_targets(directory: Any) -> dict[str, Any]:
    """Per-target metadata from the first live controls server's report, or ``{}``.

    The hook twin's ``_published_targets`` restated: report files in name order,
    the PID taken from the file name, a dead server's report skipped (never
    deleted), and the first live report carrying a non-empty ``targets``
    mapping is the answer. Never raises.
    """
    # Imported here: control_context imports this module.
    from osprey_connectors import control_context

    prefix = control_context.REPORT_FILE_PREFIX
    suffix = control_context.REPORT_FILE_SUFFIX
    try:
        names = sorted(
            n for n in os.listdir(directory) if n.startswith(prefix) and n.endswith(suffix)
        )
    except (OSError, TypeError, ValueError):
        return {}
    for name in names:
        try:
            pid = int(name[len(prefix) : -len(suffix)])
        except ValueError:
            continue
        if not control_context.is_process_alive(pid):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                report = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        targets = report.get("targets")
        if isinstance(targets, dict) and targets:
            return targets
    return {}


def published_machine_kind(
    target: Any, *, state_dir: Any = None
) -> Literal["real", "rehearsal", "simulator"]:
    """The kind of machine *target* is, as this deployment's controls server publishes it.

    The one source every surface reads (C-KIND). ``standin`` is ``rehearsal``
    only when the first live server's report — in the hook twin's order, dead
    servers' reports ignored — says so for it; no live report, no key, or any
    other value reads ``real``, the cautious words. So a process that sees no
    report (an executor sandbox, a bluesky or dispatch container, a deployment
    with no server running) names a ``standin`` the real machine. Every other
    target answers from the name table: ``va`` is the simulator, ``live`` and
    anything unknown the real machine — a report can never make them a
    rehearsal.

    Args:
        target: The control target.
        state_dir: The directory holding the servers' reports; ``None`` reads
            :func:`osprey_connectors.posture_store.state_dir`.

    Returns:
        ``real``, ``rehearsal`` or ``simulator``. Never raises.
    """
    if target != TARGET_STANDIN:
        return _MACHINE_KINDS.get(target, "real") if isinstance(target, str) else "real"
    try:
        if state_dir is None:
            # Imported here: posture_store imports this module.
            from osprey_connectors import posture_store

            state_dir = posture_store.state_dir()
        if state_dir is None:
            return "real"
        meta = _published_targets(os.fspath(state_dir)).get(TARGET_STANDIN)
    except Exception:  # the kind is a word an operator reads; it must not raise
        logger.debug("Could not read the published machine kind", exc_info=True)
        return "real"
    if isinstance(meta, dict) and meta.get(_MACHINE_KIND_KEY) == _REHEARSAL:
        return _REHEARSAL
    return "real"


# -- Limits posture --
# Limits checking is per connector type for the same reason write posture is: a
# deployment with a live machine beside a virtual accelerator has one posture
# per machine, not one per deployment. The block overrides *whole* — a per-type
# block states both leaves and then answers alone — so the value below carries
# the two leaves together with the key that answered them.

#: The limits block's own name, both deployment-wide
#: (``control_system.limits_checking``) and inside one connector block
#: (``control_system.connector.<type>.limits_checking``). One name because it is
#: one block at two scopes, the per-type one overriding the other whole.
LIMITS_CHECKING_LEAF = "limits_checking"

#: The leaves a limits block has to state to answer at all. ``database_path`` is
#: deliberately not among them: the deployment mounts one limits database, so a
#: per-type block that omits the path is complete rather than incomplete.
LIMITS_LEAVES = ("enabled", "mode")

#: The two limits modes. ``exclusive``: the limits database is the complete
#: list of writable channels, and a channel with no record is refused.
#: ``optional``: a channel with a record is checked against it, and a channel
#: with no record is written with no limits.
LIMITS_MODE_EXCLUSIVE = "exclusive"
LIMITS_MODE_OPTIONAL = "optional"
LIMITS_MODES = (LIMITS_MODE_EXCLUSIVE, LIMITS_MODE_OPTIONAL)

#: Every leaf the deployment-wide ``limits_checking`` block defines: the posture
#: pair plus the database path, which is deployment-wide only. A per-type block
#: defines :data:`LIMITS_LEAVES` alone. A block carrying any other leaf is
#: refused by :func:`incomplete_limits_blocks`, so a misspelt or retired leaf
#: cannot sit in a config looking like a posture nobody reads.
_LIMITS_BLOCK_LEAVES = (*LIMITS_LEAVES, "database_path")


@dataclass(frozen=True)
class LimitsPosture:
    """A resolved limits posture, together with the config key that answered it.

    The value and its key travel as one because a refusal has to send an
    operator to the line they can actually edit. A deployment that set
    ``mode: optional`` for its simulator alone has two keys in play,
    and a refusal naming the deployment-wide one would have the operator flip a
    line the per-type block overrides — a change that does nothing, on a
    posture they did not mean to touch. Resolving the value without carrying
    its key back is what makes that mistake available, so this type does not
    offer the value on its own.

    Both leaves are tri-state. ``None`` is "no block said", which is not the
    same as a block stating a value: silence is what a deployment that never
    configured limits checking has, and every write path treats it as no
    permission rather than as a decision. Only :data:`LIMITS_LEAVES` are
    carried; the limits database path stays deployment-wide.

    Attributes:
        enabled: Whether limits checking is on, or ``None`` when unstated.
        mode: One of :data:`LIMITS_MODES`, or ``None`` when unstated. Write
            paths write a channel with no record only on ``optional``.
        connector_type: The connector type whose block answered, or ``None``
            when the deployment-wide block did. It selects the key spelling and
            nothing else.
        incomplete: The :data:`LIMITS_LEAVES` the answering block failed to
            state, in :data:`LIMITS_LEAVES` order — a leaf a per-type block
            omitted, or a leaf either block wrote as something no reader can
            use. Empty for a well-formed block. A non-empty value
            means the posture answered nothing — both leaves are ``None`` —
            and a reader that must act builds a failsafe rather than guessing
            which half of the block was meant. The one posture not read from a
            single block, :func:`most_restrictive_limits_posture`'s fold, may
            carry definite leaves beside a non-empty tuple; readers check this
            field first for exactly that reason.
    """

    enabled: bool | None
    mode: str | None
    connector_type: str | None
    incomplete: tuple[str, ...] = ()

    def key(self, leaf: str) -> str:
        """The config key this posture's *leaf* was read from.

        ``control_system.connector.<type>.limits_checking.<leaf>`` when a
        connector block answered, and ``control_system.limits_checking.<leaf>``
        when the deployment-wide block did. A custom connector's dotted module
        path (``'mypackage.MoatConnector'``) is one key and is interpolated
        whole, never split on its dots.

        A posture holding no type at all — the deployment-wide answer, and the
        fallback a target that resolves to no type gets — spells the
        deployment-wide key, which is where such a posture was read from
        anyway.

        Args:
            leaf: One of :data:`LIMITS_LEAVES`.

        Returns:
            The dotted key, as a caller spells it for ``get_config_value`` and
            as a refusal names it to an operator.
        """
        if not self.connector_type:
            return f"control_system.{LIMITS_CHECKING_LEAF}.{leaf}"
        return f"control_system.connector.{self.connector_type}.{LIMITS_CHECKING_LEAF}.{leaf}"

    @property
    def block_key(self) -> str:
        """The dotted key of the whole ``limits_checking`` block, without a leaf.

        What a refusal names when the block itself is the problem — it is not a
        mapping, or it failed to state leaves an operator now has to go and
        write. Spelled off :meth:`key` rather than rebuilt, so a refusal about a
        block and a refusal about one of its leaves cannot disagree about where
        the block is.
        """
        return self.key(LIMITS_LEAVES[0]).rsplit(".", 1)[0]

    @property
    def strict(self) -> bool:
        """Whether this posture refuses everything the limits database does not list.

        Limits checking on *and* the mode explicitly ``exclusive`` — the only
        definition of "strict" in the codebase, so the target roster and the
        docs cannot drift into two readings of one word.

        Explicit on both leaves: ``None`` is a deployment that never stated a
        posture, and a deployment that stated nothing has refused nothing.
        Counting silence as strict would advertise a guarantee no config line
        backs, on exactly the deployments least likely to have checked.
        """
        return self.enabled is True and self.mode == LIMITS_MODE_EXCLUSIVE


def type_limits_posture(section: Any, connector_type: str | None) -> LimitsPosture:
    """The limits posture one connector *type* runs under.

    ``control_system.connector.<type>.limits_checking`` when that block is
    there, and ``control_system.limits_checking`` when it is not, where
    *connector_type* is one key. A custom connector's dotted module path
    (``'mypackage.MoatConnector'``) names a single block, never a path through
    several, so the type is looked up whole and never split on its dots.

    The per-type block overrides **whole**, and that is the rule the rest of the
    family is built on:

    - **Absent** — no connector table, no block for this type, a connector
      block that is not a mapping, or one carrying no ``limits_checking`` key
      at all — inherits the deployment-wide block, both leaves tri-state. That
      is the compatibility story: a deployment that says nothing per type keeps
      the posture it had when the deployment-wide block was the only posture
      there was.
    - **Present but not a mapping** — ``limits_checking: 'true'``, a list, a
      bare ``limits_checking:`` — is a block written and unreadable, not an
      absent one, so it answers nothing with both leaves incomplete. Inheriting
      there would hand the deployment-wide posture to a type whose own block
      says something nobody can read, which is the one case where inheritance
      would be a guess rather than compatibility.
    - **Present and complete** answers alone. Not one leaf is borrowed from the
      deployment-wide block, because a facility that wrote a block under its
      live machine has described *that* machine, and a global relaxation meant
      for a simulator must not complete it into permission on hardware.
    - **Present and failing to state a leaf** answers *nothing*: both leaves
      ``None``, and :attr:`LimitsPosture.incomplete` names what it failed to
      state. Half a block is a config the build and ``osprey validate`` refuse
      outright; a hand-edited or older render can still carry one, and then a
      reader that must act builds a failsafe — blocking every write — rather
      than guessing which half was meant.

    A block fails to state a leaf in two ways, and only the first is particular
    to per-type blocks. A per-type block may **omit** one, which the
    deployment-wide block may do freely because it inherits nothing. Either
    block may **write one unreadably** — the ``None`` YAML gives a bare
    ``enabled:``, the quoted ``'true'``, a ``1``, an unexpanded
    ``'${LIMITS_ON}'`` — and that is a failure in both scopes. It has to be:
    an unreadable leaf read as merely unset would leave ``enabled`` meaning
    "this deployment configured no limits checking", so a deployment that wrote
    ``enabled: 'true'`` to switch checking *on* would have its writes go
    unchecked. Reported as unstated, it costs a config edit; read as unset, it
    would cost the guarantee the line was written to provide. Both readings are
    the build refusal's too (:func:`incomplete_limits_blocks`), so a config the
    build accepts and one the runtime enforces cannot come apart.

    Keyed by type rather than by control target because a connector, the
    factory that builds it, an IPC child all know which connector type they are and never which target
    selected it. :func:`target_limits_posture` is this same answer for the
    holders that carry a target instead.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        connector_type: A connector type name, as
            :func:`resolve_control_system_type` and :func:`resolve_target`
            return it — which is also its connector sub-block key. ``None``
            (or an empty name) asks the deployment-wide question, which is what
            a caller holding no type at all has always asked.

    Returns:
        The resolved :class:`LimitsPosture`, carrying the type whose block
        answered — or ``None`` for the deployment-wide block, which is the key
        such a posture was read from anyway. Never raises: a section or a
        connector table that is not a mapping is a deployment that has said
        nothing, and a deployment that has said nothing states no posture — a
        ``limits_checking`` value that is present but not a mapping is instead
        an unreadable block, incomplete on both leaves.
    """
    if connector_type:
        connector = section.get("connector") if isinstance(section, dict) else None
        block = connector.get(connector_type) if isinstance(connector, dict) else None
        if isinstance(block, dict) and LIMITS_CHECKING_LEAF in block:
            limits = block[LIMITS_CHECKING_LEAF]
            if not isinstance(limits, dict):
                return LimitsPosture(None, None, connector_type, LIMITS_LEAVES)
            return _block_limits_posture(limits, connector_type, whole=True)
    return _global_limits_posture(section)


def target_limits_posture(section: Any, target: Any) -> LimitsPosture:
    """The limits posture one session *target* runs under.

    :func:`type_limits_posture` for the type that target resolves to, so a
    holder following the control target and the connector that knows only its
    own type read one posture and not two. The roster row, the stdlib hook, the
    executor and the tool layer all arrive here; the connector, the factory and
    the IPC child arrive at :func:`type_limits_posture` — and a deployment that
    described one machine must not answer them differently.

    A target that does not resolve answers the deployment-wide block instead.
    Two shapes reach that branch and both mean the same thing: there is no
    per-type block to consult because there is no type. An unknown target names
    nothing. ``live`` on a deployment baselined on the simulator in process with
    no real machine authored names a machine the config never described, which
    :func:`resolve_target` refuses to guess.
    Such a deployment has only ever had the one deployment-wide block, and
    keeping it is parity rather than a fallback: a target this deployment never
    configured still has the one block every such deployment had. Refusing here
    would instead take the posture away from every deployment that never had a
    second target.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        target: The control target, one of :data:`CONTROL_TARGETS`.

    Returns:
        The resolved :class:`LimitsPosture` for that target, carrying the type
        whose block answered — or ``None`` for the deployment-wide block, which
        is both what an unresolvable target gets and what a resolvable one gets
        when its type wrote no block. Never raises.
    """
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return _global_limits_posture(section)
    return type_limits_posture(section, connector_type)


def most_restrictive_limits_posture(section: Any) -> LimitsPosture:
    """The limits posture that holds across every target a session here can select.

    What a caller with no target of its own has to assume. The stdlib hook is
    the one that needs it: when the deployment's control target cannot be read —
    no state written yet, an unreadable state directory, a target that resolves
    to nothing — it still has to decide about a write, and the machine it is
    deciding about is any of the reachable ones.

    The reachable set is every
    :func:`configured_targets` when the deployment renders the switch
    (:func:`switch_capable`), and otherwise the single connector
    ``control_system.type`` builds, read by *type* through
    :func:`type_limits_posture`. Reading the baseline by type rather than by
    target is what keeps a deployment without the switch from folding a
    second block's relaxation into an answer no session here could ever reach;
    and walking the configured targets rather than
    :data:`CONTROL_TARGETS` keeps :func:`target_limits_posture`'s
    unresolvable-target fallback from voting for a machine nobody stood up.

    The two leaves fold in opposite directions, each toward the answer that
    costs a config edit rather than a machine:

    - ``enabled`` is ``True`` when *any* reachable posture has it ``True``.
      Limits checking being on somewhere is a constraint that may apply to the
      write in hand.
    - ``mode`` is ``optional`` only when *every* reachable posture is
      ``optional``. Permission to write a channel the limits database does not
      list holds only where every machine grants it, so one ``exclusive``
      target — or one that states nothing, ``None`` and an incomplete block
      alike counting as not-``optional`` — makes the answer ``exclusive``.

    Incompleteness travels with them, as the union of every reachable
    posture's :attr:`LimitsPosture.incomplete` in :data:`LIMITS_LEAVES` order.
    One reachable machine whose block cannot be read makes the fold incomplete,
    and a reader that must act therefore builds the blocking failsafe. Dropping
    it would be the worst failure available here: the ``enabled`` fold sends an
    incomplete posture's ``None`` to ``False``, which reads as "checking off" — and a validator built from *that* is no validator at
    all, so the caller with the least information about which machine it is
    touching would be the one waved through. This is the fail-closed direction
    :func:`LimitsValidator.from_config_most_restrictive`'s callers are promised.

    The result is derived rather than read, which is why both leaves come back
    definite where a single block's would be tri-state, and why the posture
    carries no connector type: no per-type line answers for a union, and naming
    one would send an operator to edit a machine rather than the answer. The
    deployment-wide keys :meth:`LimitsPosture.key` then spells are the honest
    ones — they are the lines that decide every target that wrote no block.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.

    Returns:
        The folded :class:`LimitsPosture`, always carrying ``None`` for its
        connector type, and incomplete when any reachable posture is. Never
        raises and never folds an empty set:
        :func:`configured_targets` always holds the baseline, and a deployment
        that says nothing still has one target.
    """
    if switch_capable(section):
        postures = [
            target_limits_posture(section, target) for target in configured_targets(section)
        ]
    else:
        postures = [type_limits_posture(section, resolve_control_system_type(section))]
    return LimitsPosture(
        enabled=any(posture.enabled is True for posture in postures),
        mode=(
            LIMITS_MODE_OPTIONAL
            if all(posture.mode == LIMITS_MODE_OPTIONAL for posture in postures)
            else LIMITS_MODE_EXCLUSIVE
        ),
        connector_type=None,
        incomplete=tuple(
            leaf
            for leaf in LIMITS_LEAVES
            if any(leaf in posture.incomplete for posture in postures)
        ),
    )


def limits_database_required(posture: LimitsPosture, *, path_configured: bool) -> bool:
    """Whether a target with this limits posture needs a limits database.

    The one answer every caller gives before it demands the file — the
    connector's validator, the compose mount, the build check, a service start
    and the bridge's startup gate — so none of them can refuse a deployment
    another one runs.

    A database is required when limits checking is on and either the mode is
    not ``optional`` or a path is configured. ``optional`` with no path is a
    deployment that legitimately keeps no limits file: every channel is
    unlisted, and an unlisted channel is written with no limits, so there is
    no file whose absence could matter. An unstated mode counts as not
    optional, because silence is no permission to skip the database. A path
    that is configured is always required, so a configured file that fails to
    load stays a refusal everywhere rather than quietly checking nothing.

    Args:
        posture: The resolved posture of the target being asked about.
        path_configured: Whether the caller's own config names a limits
            database path (the live lookup at runtime, the rendered config at
            render and build time).

    Returns:
        ``True`` when a limits database must exist and load.
    """
    return posture.enabled is True and (posture.mode != LIMITS_MODE_OPTIONAL or path_configured)


def tier_limits_database_targets(section: Any, *, path_configured: bool) -> list[str]:
    """The targets the login tier lets write whose limits posture needs a database.

    The tier is a ceiling, not a selection: a render whose :func:`tier_writes`
    answers ``asked`` or ``allowed`` lets a session write to every :func:`configured_targets`,
    so every one of them whose resolved :func:`target_limits_posture` makes
    :func:`limits_database_required` true — limits checking on, and a mode
    other than ``optional`` or a path configured — needs the channel-limits
    database to exist and load. A ``never`` render writes nowhere and so arms
    none. This is the one list a deployment-level caller reads before it
    demands the file (the compose mount, the build check), so none of them can
    refuse a deployment another one runs; a caller that serves one lane or one
    type asks :func:`limits_database_required` for that posture directly.

    The write half is deployment-wide and the limits half is per target, so
    both are read off the same machine by construction. ``incomplete`` is not
    read here: a half-written block builds a blocking failsafe at every write
    surface, and a caller that refuses it names the leaves itself
    (:attr:`LimitsPosture.incomplete`).

    Args:
        section: The ``control_system:`` config section.
        path_configured: Whether the config names a limits database path; the
            path is deployment-wide, so one answer serves every target.

    Returns:
        The qualifying targets in :func:`configured_targets` order; empty for a
        ``never`` render. Never raises.
    """
    if not tier_allows(section):
        return []
    return [
        target
        for target in configured_targets(section)
        if limits_database_required(
            target_limits_posture(section, target), path_configured=path_configured
        )
    ]


def incomplete_limits_blocks(section: Any) -> list[str]:
    """Every half-written per-type limits block in a rendered section, named.

    A per-type ``limits_checking`` block overrides whole, so a block that states
    one leaf states no posture at all: :func:`type_limits_posture` answers
    ``None`` twice for it and a write path that must act builds a failsafe. The
    build and ``osprey validate`` refuse such a config outright rather than ship
    a deployment whose limits posture silently stopped working, and this is what
    they read to do it — the rendered config itself, so a spelling the profile
    layer could not classify is still caught before it reaches a machine.

    A lint and not a posture: it answers about a section rather than about a
    target, and each line it emits names the key an operator has to fix. It
    reports exactly what :func:`type_limits_posture` reads as a block that
    failed to state a leaf — the two share :func:`_unstated_limits_leaves`, so
    a config the build accepts and one the runtime enforces cannot come apart —
    which is the two failures a block can have:

    - A **per-type** block that omits a leaf. The deployment-wide block is not
      reported for an omission: it inherits nothing, so a leaf it never carried
      is the tri-state's ``None`` and the shape every deployment predating
      per-type blocks has.
    - **Either** block writing a leaf as something no reader can use — for
      ``enabled`` a quoted ``'true'``, a ``1``, a bare ``enabled:``, an
      unexpanded ``'${LIMITS_ON}'``; for ``mode`` anything but one of
      :data:`LIMITS_MODES`. Environment expansion yields strings, so
      this is what a deployment gets when it wires a limits leaf to a variable
      nothing set. It blocks every write at runtime, which is the safe
      direction but a poor way to find out; refusing the build is where that
      config is cheap to fix.
    - **Either** ``limits_checking`` being present but not a mapping at all,
      which gets one line naming the block and quoting what was found there
      instead of a line per leaf: there are no leaves to name.
    - The deployment-wide block carrying a leaf outside
      :data:`_LIMITS_BLOCK_LEAVES`, or a per-type block one outside
      :data:`LIMITS_LEAVES`. No reader consults such a leaf, so the posture
      its author meant by it is not the one the deployment runs.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes — the *rendered* one, as
            the build produced it, since that is the config a deployment runs.

    Returns:
        One line per unstated leaf and per undefined one: the deployment-wide
        block first, then the connector blocks in the order the section carries
        them, leaves in :data:`LIMITS_LEAVES` order and undefined leaves sorted
        after them, so one refusal lists everything an operator has to fix. Empty for a section whose blocks are complete, absent, or
        not blocks at all. Never raises: a lint that crashed on a malformed
        config would fail a build without saying what is wrong with it.
    """
    if not isinstance(section, dict):
        return []
    connector = section.get("connector")
    blocks: list[tuple[Any, Any, bool]] = []
    if LIMITS_CHECKING_LEAF in section:
        blocks.append((None, section[LIMITS_CHECKING_LEAF], False))
    if isinstance(connector, dict):
        blocks.extend(
            (connector_type, block[LIMITS_CHECKING_LEAF], True)
            for connector_type, block in connector.items()
            if isinstance(block, dict) and LIMITS_CHECKING_LEAF in block
        )
    errors: list[str] = []
    for connector_type, block, whole in blocks:
        # The posture this block would answer with, built for its keys alone —
        # so the line naming the block and the lines naming its leaves are
        # spelled by one thing.
        posture = LimitsPosture(None, None, connector_type)
        if not isinstance(block, dict):
            errors.append(
                f"{posture.block_key} is {block!r}, not a block; a limits "
                f"block is a mapping stating {LIMITS_LEAVES[0]} and {LIMITS_LEAVES[1]}"
            )
            continue
        for leaf, written in _unstated_limits_leaves(block, whole=whole):
            if written:
                errors.append(
                    f"{posture.key(leaf)} is {block[leaf]!r}, not {_LEAF_VALUES[leaf]}; "
                    "a limits leaf that cannot be read states no posture and blocks "
                    "every write as a failsafe"
                )
            else:
                errors.append(
                    f"{posture.key(leaf)} is missing; a per-type limits block must state "
                    f"both {LIMITS_LEAVES[0]} and {LIMITS_LEAVES[1]}"
                )
        defined = LIMITS_LEAVES if whole else _LIMITS_BLOCK_LEAVES
        errors.extend(
            f"{posture.key(leaf)} is not a limits leaf; a limits block states "
            f"{LIMITS_LEAVES[0]} and {LIMITS_CHECKING_LEAF}.{LIMITS_LEAVES[1]}: "
            f"{' | '.join(LIMITS_MODES)}"
            for leaf in sorted(str(key) for key in block if key not in defined)
        )
    return errors


def _global_limits_posture(section: Any) -> LimitsPosture:
    """The deployment-wide posture, ``control_system.limits_checking``.

    A leaf this block does not carry is the tri-state and not a malformed
    block: silence here is the shape every deployment predating per-type blocks
    has, and only a per-type block, which overrides whole, has to state both
    leaves to answer at all. A leaf it *does* carry but spells as something
    no reader can use is unreadable in either block alike — see
    :func:`_unstated_limits_leaves` — and so is the whole block when
    ``limits_checking`` is present but is not a mapping, which is incomplete on
    both leaves rather than the silence of a deployment that wrote none.
    """
    if not isinstance(section, dict) or LIMITS_CHECKING_LEAF not in section:
        return _block_limits_posture({}, None, whole=False)
    block = section[LIMITS_CHECKING_LEAF]
    if not isinstance(block, dict):
        return LimitsPosture(None, None, None, LIMITS_LEAVES)
    return _block_limits_posture(block, None, whole=False)


def _unstated_limits_leaves(block: dict[str, Any], *, whole: bool) -> list[tuple[str, bool]]:
    """The leaves a limits block fails to state, and whether each was written.

    One classifier for both scopes and both readers — the posture resolvers and
    the build's render check — because "which leaves did this block fail to
    state" decided twice is a config the build accepts and the runtime refuses,
    or worse the other way round.

    Two ways a block fails to state a leaf, and they are told apart by the
    boolean each entry carries:

    - **Absent** (``False``), which is a failure only under *whole*: a per-type
      block overrides the deployment-wide pair entire, so it has to state both
      leaves to answer at all. The deployment-wide block is read with
      ``whole=False``, where an absent leaf is simply the tri-state's ``None``.
    - **Written but unreadable** (``True``) — for ``enabled`` a quoted
      ``'true'``, a ``1``, a bare ``enabled:``, an unexpanded
      ``'${LIMITS_ON}'`` that no environment resolved; for ``mode`` anything
      but one of :data:`LIMITS_MODES` — which is a failure in *either* scope. This is the leaf that
      cannot be allowed to read as a plain unset: unset ``enabled`` means "no
      limits checking configured", so a deployment that wrote
      ``enabled: 'true'`` meaning to switch checking *on* would have every
      write waved through unchecked. Reporting it unstated instead makes the
      posture incomplete, and an incomplete posture blocks. The config edit is
      one line; the alternative is unchecked writes on a machine somebody
      believed was guarded.

    Args:
        block: A ``limits_checking`` mapping, deployment-wide or per type.
        whole: Whether an absent leaf is a failure, which it is exactly when
            the block overrides another one entire.

    Returns:
        ``(leaf, was_written)`` in :data:`LIMITS_LEAVES` order, empty for a
        block that states everything it has to.
    """
    unstated: list[tuple[str, bool]] = []
    for leaf in LIMITS_LEAVES:
        if leaf not in block:
            if whole:
                unstated.append((leaf, False))
        elif _LEAF_READERS[leaf](block[leaf]) is None:
            unstated.append((leaf, True))
    return unstated


def _block_limits_posture(
    block: dict[str, Any], connector_type: str | None, *, whole: bool
) -> LimitsPosture:
    """One ``limits_checking`` mapping read into a posture.

    A block that fails to state any leaf answers *nothing* — both leaves
    ``None``, with :attr:`LimitsPosture.incomplete` naming what it failed to
    state. Half an answer is not available: neither leaf of a block written
    with the other one unreadable can be trusted to mean what a reader would do
    with it, and a caller that must act builds a failsafe from ``incomplete``
    rather than guessing.
    """
    unstated = _unstated_limits_leaves(block, whole=whole)
    if unstated:
        return LimitsPosture(None, None, connector_type, tuple(leaf for leaf, _ in unstated))
    enabled_leaf, mode_leaf = LIMITS_LEAVES
    return LimitsPosture(
        enabled=_limits_leaf(block.get(enabled_leaf)),
        mode=_limits_mode(block.get(mode_leaf)),
        connector_type=connector_type,
    )


def _limits_leaf(value: Any) -> bool | None:
    """One limits leaf as a literal boolean, or ``None`` for anything else.

    The literal booleans and nothing else are readable. A quoted ``'true'``, a
    ``1``, a bare ``enabled:`` — values no write tier reads as an arming
    either — are not.

    Unlike write posture, ``None`` here is not itself the safe answer, which is
    why no caller reads this directly: an unset ``enabled`` means "this
    deployment configured no limits checking", so a leaf written unreadably and
    reported as merely unset would turn an attempt to switch checking *on* into
    writes that are never checked at all. :func:`_unstated_limits_leaves` is
    what the resolvers use, and it separates a leaf nobody wrote from a leaf
    nobody can read.
    """
    if value is True:
        return True
    if value is False:
        return False
    return None


def _limits_mode(value: Any) -> str | None:
    """The ``mode`` leaf as one of :data:`LIMITS_MODES`, or ``None`` for anything else.

    The two spellings and nothing else are readable: a boolean, a different
    case, a bare ``mode:`` each state no mode, and a block that states no mode
    is one :func:`_unstated_limits_leaves` reports rather than one a reader
    guesses at.
    """
    if isinstance(value, str) and value in LIMITS_MODES:
        return value
    return None


#: How each of :data:`LIMITS_LEAVES` is read, and what a refusal says a
#: readable value looks like.
_LEAF_READERS: dict[str, Any] = {LIMITS_LEAVES[0]: _limits_leaf, LIMITS_LEAVES[1]: _limits_mode}
_LEAF_VALUES = {
    LIMITS_LEAVES[0]: "a literal true or false",
    LIMITS_LEAVES[1]: " or ".join(LIMITS_MODES),
}


def _real_blocks(section: Any, never_live: tuple[str, ...]) -> list[str]:
    """The connector blocks that could describe a real machine, sorted."""
    connector = section.get("connector") if isinstance(section, dict) else None
    if not isinstance(connector, dict):
        return []
    return sorted(key for key in connector if isinstance(key, str) and key not in never_live)


def _report_second_real_block(section: Any, baseline: str, never_live: tuple[str, ...]) -> None:
    """Say so when a config describes a real machine no target can reach.

    A deployment whose own type is its real machine never consults the connector
    table for ``live``, so a second non-simulated block is not ambiguous here —
    it is unreachable, and quietly so: no control target resolves to it, and the
    roster it never appears in looks exactly like a roster for a deployment that
    configured one machine. Nothing else in the stack will ever mention it, so
    the resolver that skipped it is where a deployer has to hear about it.

    Once per config shape, not once per call: this resolver answers every roster
    render, and a line repeated at that rate is a line nobody reads.
    """
    others = [name for name in _real_blocks(section, never_live) if name != baseline]
    if not others:
        return
    seen = (baseline, tuple(others))
    if seen in _SECOND_REAL_BLOCK_WARNED:
        return
    _SECOND_REAL_BLOCK_WARNED.add(seen)
    logger.warning(
        "'control_system.connector' carries %s beside this deployment's own %r block, "
        "and no control target reaches %s. %s",
        ", ".join(repr(name) for name in others),
        baseline,
        "them" if len(others) > 1 else "it",
        _ONE_REAL_MACHINE,
    )


def _live_type(section: Any) -> str:
    """The control system type that reaches this deployment's real machine.

    Neither a simulated type nor a stand-in one can be that machine, and the
    exclusion holds on both sides of the derivation — the baseline it starts
    from, and the candidate blocks it falls back to. A stand-in is a machine the
    deployment stands up itself; counting it would either answer ``live`` with
    the stand-in, or make ``live`` ambiguous on exactly the deployments that run
    the stand-in beside the block naming their facility's own machine.

    Either way the answer is one machine — :data:`_ONE_REAL_MACHINE` — and both
    ways of missing it are said out loud: a baseline that cannot derive one
    raises, and a baseline that *is* one, beside a second real block no target
    can reach, is warned about.
    """
    never_live = SIMULATED_TYPES + STANDIN_TYPES
    baseline = resolve_control_system_type(section)
    if baseline not in never_live:
        _report_second_real_block(section, baseline, never_live)
        return baseline

    candidates = _real_blocks(section, never_live)
    if len(candidates) == 1:
        return candidates[0]

    found = ", ".join(repr(name) for name in candidates) if candidates else "none"
    raise ValueError(
        f"Target {TARGET_LIVE!r} has no control system on this deployment: "
        f"'control_system.type' resolves to {baseline!r}, which is simulated, "
        f"and the live blocks under 'control_system.connector' are: {found}. "
        f"{_ONE_REAL_MACHINE} Configure exactly one such block (an {EPICS!r} or "
        f"{DOOCS!r} block, say). It is not inferred, so that no session reaches "
        "a real machine this config never described."
    )
