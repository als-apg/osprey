"""Which machine a control target means, in one deployment.

:func:`resolve_target` answers the run-time question "which machine is this
session pointed at". It is the same shape of answer as
:func:`~osprey_connectors.connector_types.resolve_control_system_type` for the
same reason: several holders follow a control target — the connector-host
process, its child, an executor sandbox — and a holder that translates ``live``
into a connector type privately can route a tool call somewhere the roster
never claimed. A target names a *machine*, and there are three of them: the
facility's own control system (``live``), the virtual accelerator (``va``), and
the live stand-in (``standin``) — a soft IOC a deployment runs for itself, which
is a third machine with a connector block of its own rather than a mode of the
first. That is what keeps ``live`` meaning the machine the facility authored on
a deployment that runs the stand-in beside it. The target is an *argument*
here, never something this module reads from the environment: an
environment-reading override would make the framework's archiver-pairing rule
describe a deployment the process is no longer running.

The target vocabulary itself (:data:`~osprey_connectors.connector_types.ControlTarget`,
:data:`~osprey_connectors.connector_types.MachineKind`) is
:mod:`osprey_connectors.connector_types`', because each built-in row is written
in it.
"""

import logging

from osprey_connectors.config_sections import ConnectorBlock, ControlSystemSection, Unreadable
from osprey_connectors.connector_types import (
    BUILTIN_CONNECTORS,
    CONTROL_TARGETS,
    DOOCS,
    EPICS,
    LIVE_STANDIN,
    SIMULATED_TYPES,
    STANDIN_TYPES,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    VIRTUAL_ACCELERATOR,
    ControlTarget,
    MachineKind,
    is_control_target,
    resolve_control_system_type,
)
from osprey_connectors.errors import ConfigurationError

__all__ = [
    "baseline_target",
    "configured_targets",
    "machine_kind",
    "resolve_target",
    "switch_capable",
    "target_configured",
]

logger = logging.getLogger("osprey_connectors.targets")

#: Config shapes the second-real-block warning has already been said about, so
#: a resolver asked on every roster render says it once. Keyed by the shape
#: itself — a deployment has one, and a test that edits a config gets a fresh
#: line for the shape it edited to.
_SECOND_REAL_BLOCK_WARNED: set[tuple[str, tuple[str, ...]]] = set()

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

#: The target each self-standing machine's type is the baseline of, read off
#: each row's
#: :attr:`~osprey_connectors.connector_types.BuiltinConnector.baseline_target`.
#: A type absent from this table describes the facility's own machine, hence
#: ``live``.
_BASELINE_TARGETS: dict[str, ControlTarget] = {
    row.name: row.baseline_target
    for row in BUILTIN_CONNECTORS
    if row.baseline_target is not None and row.baseline_target != TARGET_LIVE
}


def resolve_target(section: ControlSystemSection, target: object) -> str:
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
      exactly as
      :func:`~osprey_connectors.connector_types.resolve_control_system_type`
      already lets it. A second non-simulated block beside it is not a second
      target: no name reaches it (:data:`_ONE_REAL_MACHINE`), so it is warned
      about rather than skipped in silence.
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
        section: The ``control_system:`` config section, read
            (:class:`~osprey_connectors.config_sections.ControlSystemSection`).
        target: The control target, one of
            :data:`~osprey_connectors.connector_types.CONTROL_TARGETS`. An
            argument, never read from the environment, and never defaulted.

    Returns:
        The connector type, which doubles as the connector sub-block key.

    Raises:
        ValueError: If *target* is not a known target (including ``None`` and
            blank).
        ConfigurationError: If it is ``live`` on a deployment with no single
            derivable live control system (``key`` ``control_system.connector``).
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


def baseline_target(section: ControlSystemSection) -> ControlTarget:
    """The target a deployment's own ``control_system:`` section selects.

    ``va`` for the simulator, in whichever venue its leaf names (a section
    that states no type is the simulator in process, so it is on ``va``),
    ``standin`` for the live stand-in, and ``live`` for every real control
    system.

    Beside :func:`~osprey_connectors.connector_types.resolve_control_system_type`
    rather than restated by each holder, because "which target am I on when
    nobody has switched" is asked by the supervisor, the switch predicate below,
    and every baseline-pinned reader — and three answers to it is three ways for
    one deployment to be described.
    """
    return _BASELINE_TARGETS.get(resolve_control_system_type(section), TARGET_LIVE)


def switch_capable(section: ControlSystemSection) -> bool:
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
        section: The ``control_system:`` config section, read
            (:class:`~osprey_connectors.config_sections.ControlSystemSection`). Callers holding a whole rendered config pass
            ``config.get("control_system")``.

    Returns:
        ``True`` when the section resolves and configures at least two
        targets. Never raises: a section that does not resolve is not capable.
    """
    try:
        return len(configured_targets(section)) >= 2
    except ValueError:
        return False


def target_configured(section: ControlSystemSection, target: object) -> bool:
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
        section: The ``control_system:`` config section, read
            (:class:`~osprey_connectors.config_sections.ControlSystemSection`).
        target: The control target being asked about.

    Returns:
        Whether the target resolves and its block is a non-empty mapping. Never
        raises: an unresolvable target is simply not configured.
    """
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return False
    block = section.block(connector_type)
    return isinstance(block, ConnectorBlock) and block.written


def configured_targets(section: ControlSystemSection) -> list[ControlTarget]:
    """The targets a session on this deployment can actually be pointed at.

    In :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` order, which
    is the order every render, roster and state file reads them in. The
    deployment's :func:`baseline_target` is always one of them — a session sits
    on it whether or not the config wrote a block for the connector
    ``control_system.type`` builds — and each of the others is here when
    :func:`target_configured` says so, which is the same resolve-and-read-the-block
    pair :func:`switch_capable` and the deployment layer's Reach Contracts
    decide it by.

    The order is the constant's and not the baseline's, so a deployment that
    gained no target gains no reordering: the two-target deployments that have
    no stand-in enumerate exactly as they did before ``standin`` existed, down
    to the order their rendered ``settings.json`` lists them in.

    This list, and never the constant, is what an enumerator walks — a roster,
    a prober, a posture render, a state file.
    :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` is the
    *vocabulary*: it grows when a new kind of machine exists, and a holder that
    loops it hands every deployment a slot for every machine anybody could run.
    Widening it to three would otherwise grow a ``standin`` row, probe and
    posture on deployments with no ``control_system.connector.live_standin``
    block at all — a machine that is not there, described as if it were.
    Which targets are configured is a property of the rendered config, so it is
    read from the config.

    Args:
        section: The ``control_system:`` config section, read
            (:class:`~osprey_connectors.config_sections.ControlSystemSection`).

    Returns:
        The configured targets in
        :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` order, the
        baseline among them. Never raises and never empty: a section that is
        missing or malformed still has a baseline, and that one target is what
        such a deployment is on.
    """
    baseline = baseline_target(section)
    return [
        target
        for target in CONTROL_TARGETS
        if target == baseline or target_configured(section, target)
    ]


#: What kind of machine each control target names by its name alone, for the
#: words a refusal uses about it. ``standin`` is ``real`` here: a name cannot
#: tell this deployment's stand-in container from a ``standin`` target pointed at
#: anything else, so only the published report
#: (:func:`~osprey_connectors.control_context.published_machine_kind`) may call
#: it the rehearsal machine.
_MACHINE_KINDS: dict[ControlTarget, MachineKind] = {
    TARGET_LIVE: "real",
    TARGET_STANDIN: "real",
    TARGET_VA: "simulator",
}


def machine_kind(section: ControlSystemSection, target: object) -> MachineKind | None:
    """The kind of machine *target* names by its name: ``real`` or ``simulator``.

    ``live`` and ``standin`` are the real machine, ``va`` the simulator. The
    name never answers ``rehearsal``: whether a ``standin`` target really is
    this deployment's stand-in is an endpoint fact only the controls server's
    report states, read by
    :func:`~osprey_connectors.control_context.published_machine_kind`. ``None``
    when :func:`resolve_target` cannot resolve the target in this deployment —
    an unknown target, or ``live`` on a deployment that never named its real
    machine. Never raises.
    """
    try:
        resolve_target(section, target)
    except ValueError:
        return None
    return _MACHINE_KINDS.get(target) if is_control_target(target) else None


def _real_blocks(section: ControlSystemSection, never_live: tuple[str, ...]) -> list[str]:
    """The connector blocks that could describe a real machine, sorted."""
    if isinstance(section.connectors, Unreadable):
        return []
    return sorted(key for key in section.connectors if key not in never_live)


def _report_second_real_block(
    section: ControlSystemSection, baseline: str, never_live: tuple[str, ...]
) -> None:
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


def _live_type(section: ControlSystemSection) -> str:
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
    raise ConfigurationError(
        f"Target {TARGET_LIVE!r} has no control system on this deployment: "
        f"'control_system.type' resolves to {baseline!r}, which is simulated, "
        f"and the live blocks under 'control_system.connector' are: {found}. "
        f"{_ONE_REAL_MACHINE} Configure exactly one such block (an {EPICS!r} or "
        f"{DOOCS!r} block, say). It is not inferred, so that no session reaches "
        "a real machine this config never described.",
        key="control_system.connector",
    )
