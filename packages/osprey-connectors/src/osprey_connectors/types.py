"""Connector type constants, and the fallbacks a factory applies to them.

Single source of truth for built-in connector type name strings.
Custom connectors use dotted module paths (e.g., 'mypackage.TangoConnector')
and don't need constants here.

The resolvers at the bottom are here rather than in
:mod:`osprey_connectors.factory` so that a guard deciding what a deployment
*will build* can share them with the factory that builds it. A guard which
re-implements "what does this config select" is a guard that can disagree with
the answer, and the disagreement is a bypass rather than a discrepancy — see
:mod:`osprey_connectors.honesty`.

:func:`resolve_target` extends that to the run-time question "which machine is
this session pointed at". It is the same shape of answer for the same reason:
several holders follow a session target — the connector-host process, its child,
an executor sandbox — and a holder that translates ``live`` into a connector type
privately can route a tool call somewhere the roster never claimed. The target is
an *argument* here, never something this module reads from the environment: an
environment-reading override would make the honesty predicate describe a
deployment the process is no longer running.

:func:`type_writes_enabled` and :func:`target_writes_enabled` answer "may this
deployment write" the same way and for the same reason. Write posture is per
connector type, so a deployment whose real machine is a live one can arm its
simulator alone; and the lint that reads the config, the persona that tells an
operator what they hold, and the connector that would perform the write all have
to agree about which types are armed. Two readings of one posture is one of them
being wrong about a machine somebody can move.
"""

from typing import Any

# -- Control system connector types (have implementations) --
MOCK = "mock"
EPICS = "epics"
VIRTUAL_ACCELERATOR = "virtual_accelerator"
DOOCS = "doocs"

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"
MONGODB_ARCHIVER = "mongodb_archiver"
DOOCS_ARCHIVER = "doocs_archiver"

# -- CLI choice lists (only types with implementations) --
CLI_CONTROL_SYSTEM_TYPES = [MOCK, EPICS, VIRTUAL_ACCELERATOR, DOOCS]
CLI_ARCHIVER_TYPES = [MOCK_ARCHIVER, EPICS_ARCHIVER, MONGODB_ARCHIVER, DOOCS_ARCHIVER]

# -- Session targets --
# What a *session* can be pointed at, as opposed to what a *config* selects.
# Deliberately two names and not one per connector type: a target names a
# machine ("the real one" / "the simulated one"), and which connector type
# reaches that machine is this module's job to answer, not the caller's.
TARGET_LIVE = "live"
TARGET_VA = "va"
CONTROL_TARGETS = [TARGET_LIVE, TARGET_VA]

# -- Write posture --
#: The deployment-wide write posture, dotted as a caller spells it for
#: ``get_config_value`` and as every refusal names it to an operator.
WRITES_ENABLED_KEY = "control_system.writes_enabled"

#: The same posture inside one connector block —
#: ``control_system.connector.<type>.writes_enabled``. A leaf name, not a path:
#: it is looked up in an already-resolved block, whose own key is the connector
#: type in full.
TYPE_WRITES_ENABLED_LEAF = "writes_enabled"

#: Types that serve a machine nobody has to be careful around. They are the
#: reason ``live`` cannot simply be "whatever the config selects": a deployment
#: whose baseline is one of these has not yet said what its real machine is.
_SIMULATED_TYPES = (MOCK, VIRTUAL_ACCELERATOR)


def _resolve_type(section: Any, fallback: str) -> str:
    """The connector type a factory builds from a config section.

    A section that is missing, is not a mapping, or carries no usable ``type``
    resolves to *fallback* — the factory's documented fail-closed default, which
    it announces with a ``… is not set; defaulting to …`` warning. Empty and
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
    """The control system a ``control_system:`` config section actually selects."""
    return _resolve_type(section, MOCK)


def resolve_target(section: Any, target: Any) -> str:
    """The connector type a session *target* selects, in a given deployment.

    The returned type is also the key of the block the factory reads its
    settings from — ``control_system.connector.<type>`` — so one string answers
    both "what do I build" and "where are its gateways".

    ``va`` is the same answer everywhere: a virtual accelerator is a virtual
    accelerator regardless of what the deployment was built for. ``live`` is
    deployment-specific and is the half that can refuse:

    - When the section's own type is a real control system, that is the live
      machine, and it is returned as written — including a value this module
      does not recognize, which reaches the factory's "Unknown … type" error
      exactly as :func:`resolve_control_system_type` already lets it.
    - When the section's own type is simulated (or absent, which resolves to the
      mock), the deployment has not named its real machine there, so the live
      type is taken from the connector table: exactly one non-simulated block
      configured means the deployment has said which machine it means. None, or
      more than one, raises.

    Nothing is inferred from the absence of evidence. A deployment that has
    never been told about its real machine gets an error naming what it would
    have to configure, because the alternative — resolving ``live`` to the one
    connector type that talks to hardware because it is the usual one — is a
    tool call arriving at a real machine on the strength of a guess.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        target: The session target, one of :data:`CONTROL_TARGETS`. An argument,
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
    if target == TARGET_LIVE:
        return _live_type(section)
    raise ValueError(
        f"Unknown control target {target!r}. Valid targets are "
        f"{TARGET_LIVE!r} and {TARGET_VA!r}, spelled exactly. A session target "
        "is always stated, never defaulted — there is no target a caller gets "
        "by saying nothing, because the one it would get could be the real "
        "machine."
    )


def baseline_target(section: Any) -> str:
    """The target a deployment's own ``control_system:`` section selects.

    ``va`` for a virtual accelerator, ``live`` for everything else — including a
    mock deployment, whose ``live`` may well be underivable, because ``live`` is
    still the target its section describes.

    Beside :func:`resolve_control_system_type` rather than restated by each
    holder, because "which target am I on when nobody has switched" is asked by
    the supervisor, the switch predicate below, and every baseline-pinned reader
    — and three answers to it is three ways for one deployment to be described.
    """
    return TARGET_VA if resolve_control_system_type(section) == VIRTUAL_ACCELERATOR else TARGET_LIVE


def switch_capable(section: Any) -> bool:
    """Whether a deployment can be pointed at either target.

    The predicate itself, in the module that already owns "which connector type
    does a target mean here". Two layers ask the question and must never answer
    it differently: the controls server decides from it whether its tools are
    served by a connector-host child, and the build decides from it whether the
    agent's frozen safety rule describes a switchable machine at all. A build
    that promises a switch the runtime will not perform is worse than one that
    never mentions it.

    Three conditions, none of which is sufficient alone:

    1. **Both targets resolve** to a connector type at all
       (:func:`resolve_target`, which raises rather than guess a live machine).
    2. **The deployment's own control system is one of the two targets.**
       ``resolve_target`` answers ``live`` for configs with no business
       switching: a ``mock`` deployment that happens to carry an ``epics`` block
       resolves ``live`` to ``epics``, and treating that as switchable would
       point a session at a real machine the config never selected. Requiring
       the baseline target to resolve back to ``control_system.type`` rules it
       out.
    3. **Both types have a non-empty connector block**, since that block is what
       a connector is configured from.

    Deliberately *not* checked: ``probe_channel``, the gateways table, the
    operator acknowledgment. Those decide whether a target may be switched *to*
    — a per-switch question answered with a reason the operator is told. This is
    the coarser question of whether the deployment is in the two-target world.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes. Callers holding a whole
            rendered config pass ``config.get("control_system")``.

    Returns:
        ``True`` when both targets are configured and consistent. Never raises:
        every malformed, partial or contradictory config is simply not capable.
    """
    if not isinstance(section, dict):
        return False
    try:
        types_by_target = {
            target: resolve_target(section, target) for target in (TARGET_LIVE, TARGET_VA)
        }
    except ValueError:
        return False

    declared = resolve_control_system_type(section)
    if types_by_target[baseline_target(section)] != declared:
        return False

    connector = section.get("connector")
    if not isinstance(connector, dict):
        return False
    return all(
        isinstance(connector.get(name), dict) and bool(connector.get(name))
        for name in types_by_target.values()
    )


def type_writes_enabled(section: Any, connector_type: str) -> bool:
    """Whether a deployment arms writes for one connector *type*.

    A tri-state on ``control_system.connector.<type>.writes_enabled``, where
    *connector_type* is one key. A custom connector's dotted module path
    (``'mypackage.TangoConnector'``) names a single block, never a path through
    several, so the type is looked up whole and never split on its dots:

    - **Absent** — no connector table, no block for this type, a block that is
      not a mapping, or a mapping without the leaf — inherits
      ``control_system.writes_enabled``. That is the whole compatibility story:
      a deployment that says nothing per type keeps the posture it had when the
      deployment-wide key was the only posture there was.
    - **Literally ``True``** arms writes for this type.
    - **Any other value** leaves them unarmed, and does *not* fall back to the
      deployment-wide key. A facility that wrote ``false`` under its live block
      has said something about that machine, and a global ``true`` armed for the
      simulator must not talk it into arming hardware. Values nobody can be sure
      of land on the same side: the ``None`` YAML gives a bare
      ``writes_enabled:``, the quoted string ``'true'``, ``1``. Unarmed is the
      reading that costs an operator a config edit rather than a machine.

    Keyed by type rather than by session target because that is the identity
    most holders have: a connector, the factory that builds it, an IPC child, a
    queueserver stamp all know which connector type they are and never which
    target selected it. :func:`target_writes_enabled` is this same answer for
    the holders that carry a target instead — the roster, the stdlib hooks, the
    executor.

    Never reads the environment, and in particular never consults
    ``is_readonly_run()``. This is the deployment posture on its own; a caller
    that must also honor a read-only run ANDs the two, so that what a config
    describes stays what this function reports.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        connector_type: A connector type name, as
            :func:`resolve_control_system_type` and :func:`resolve_target`
            return it — which is also its connector sub-block key.

    Returns:
        ``True`` only when writes are armed for that type. Never raises: a
        section that is not a mapping is a deployment that has said nothing,
        and a deployment that has said nothing is not armed.
    """
    connector = section.get("connector") if isinstance(section, dict) else None
    block = connector.get(connector_type) if isinstance(connector, dict) else None
    if not isinstance(block, dict) or TYPE_WRITES_ENABLED_LEAF not in block:
        return _global_writes_enabled(section)
    return block[TYPE_WRITES_ENABLED_LEAF] is True


def target_writes_enabled(section: Any, target: Any) -> bool:
    """Whether a deployment arms writes for one session *target*.

    :func:`type_writes_enabled` for the type that target resolves to, so a
    holder following the session target and a connector that knows only its own
    type read one posture and not two.

    A target that does not resolve answers :data:`WRITES_ENABLED_KEY` instead.
    Two shapes reach that branch and both mean the same thing — there is no
    per-type block to consult because there is no type. An unknown target names
    nothing. ``live`` on a mock or hello_world-style deployment names a machine
    the config never described, which :func:`resolve_target` refuses to guess;
    such a deployment has only ever had the one deployment-wide posture, and
    keeping it is parity rather than a fallback. Refusing here would instead
    take the posture away from every deployment that never had a second target.

    Args:
        section: The ``control_system:`` config section.
        target: The session target, one of :data:`CONTROL_TARGETS`.

    Returns:
        ``True`` only when writes are armed for that target. Never raises.
    """
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return _global_writes_enabled(section)
    return type_writes_enabled(section, connector_type)


def writes_enabled_key(connector_type: str | None) -> str:
    """The config key that decides write posture for one connector *type*.

    ``control_system.connector.<type>.writes_enabled`` — the one line a facility
    edits to arm a machine and leave the others alone — or
    :data:`WRITES_ENABLED_KEY` for a caller holding no type at all, which is
    where :func:`type_writes_enabled` read that posture from anyway.

    A refusal names the key that answered it, and that is the whole point of
    spelling the key in one place: an operator sent to the deployment-wide key
    would arm every target at once, the machine they deliberately left unarmed
    included, and a refusal naming a key some other block overrides would have
    them flip a line that changes nothing.
    """
    if not connector_type:
        return WRITES_ENABLED_KEY
    return f"control_system.connector.{connector_type}.{TYPE_WRITES_ENABLED_LEAF}"


def target_writes_enabled_key(section: Any, target: Any) -> str:
    """The config key that decides write posture for one session *target*.

    :func:`writes_enabled_key` for the type that target resolves to, and the
    deployment-wide key for a target that resolves to none — the same two
    branches :func:`target_writes_enabled` reads the posture from, so the key a
    refusal names is always the key that answered it. Never raises.
    """
    try:
        connector_type = resolve_target(section, target)
    except ValueError:
        return WRITES_ENABLED_KEY
    return writes_enabled_key(connector_type)


def session_posture(section: Any) -> dict[str, bool]:
    """Write posture per target a *session* on this deployment can be pointed at.

    Both of :data:`CONTROL_TARGETS` when the deployment renders the switch
    (:func:`switch_capable`), each through :func:`target_writes_enabled`.
    Without the switch a session sits on the one connector
    ``control_system.type`` builds, so the answer is that type's own posture
    under its :func:`baseline_target` — read by type on purpose. ``live`` is the
    switch's derivation, and on a mock deployment that happens to carry an
    armed ``epics`` block it would name a machine no session here ever
    reaches; the built connector's reference monitor reads the mock's posture,
    and a render that read the other would promise a guarantee the runtime
    does not share.

    Something that must speak about "the deployment's targets" without holding
    one — a posture button, a permissions render, a lint — iterates this rather
    than :data:`CONTROL_TARGETS`, because :func:`target_writes_enabled` answers
    an unresolvable target from the deployment-wide key for a holder that *has*
    that target, and a speculative loop over both would let that fallback arm
    a machine the config never described.
    """
    if switch_capable(section):
        return {target: target_writes_enabled(section, target) for target in CONTROL_TARGETS}
    built = resolve_control_system_type(section)
    return {baseline_target(section): type_writes_enabled(section, built)}


def any_target_writes_enabled(section: Any) -> bool:
    """Whether writes are armed for at least one target a session here can select.

    The union a caller with no target of its own may take, over
    :func:`session_posture`. A deployment that says nothing per type answers
    its deployment-wide key here exactly as it did when that key was the only
    posture; one whose reachable targets are all explicitly unarmed answers
    ``False`` no matter what the deployment-wide key says.
    """
    return any(session_posture(section).values())


def _global_writes_enabled(section: Any) -> bool:
    """The deployment-wide posture, ``control_system.writes_enabled``.

    Explicitly ``True`` and nothing else — the same reading the per-type leaf
    gets, so a quoted ``'true'`` or a ``1`` arms nothing at either level.
    """
    return isinstance(section, dict) and section.get(TYPE_WRITES_ENABLED_LEAF) is True


def _live_type(section: Any) -> str:
    """The control system type that reaches this deployment's real machine."""
    baseline = resolve_control_system_type(section)
    if baseline not in _SIMULATED_TYPES:
        return baseline

    connector = section.get("connector") if isinstance(section, dict) else None
    candidates: list[str] = []
    if isinstance(connector, dict):
        candidates = sorted(
            key for key in connector if isinstance(key, str) and key not in _SIMULATED_TYPES
        )
    if len(candidates) == 1:
        return candidates[0]

    found = ", ".join(repr(name) for name in candidates) if candidates else "none"
    raise ValueError(
        f"Target {TARGET_LIVE!r} has no control system on this deployment: "
        f"'control_system.type' resolves to {baseline!r}, which is simulated, "
        f"and the live blocks under 'control_system.connector' are: {found}. "
        "Exactly one non-simulated connector block is what names the real "
        f"machine (an {EPICS!r} or {DOOCS!r} block, say); configure it there. "
        "It is not inferred, so that no session reaches a real machine this "
        "config never described."
    )
