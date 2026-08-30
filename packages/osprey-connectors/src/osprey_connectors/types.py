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

:func:`type_writes_enabled` and :func:`target_writes_enabled` answer "may this
deployment write" the same way and for the same reason. Write posture is per
connector type, so a deployment whose real machine is a live one can arm its
simulator alone; and the lint that reads the config, the persona that tells an
operator what they hold, and the connector that would perform the write all have
to agree about which types are armed. Two readings of one posture is one of them
being wrong about a machine somebody can move.

:class:`LimitsPosture` is the same story for limits checking, which is likewise
per connector type: a deployment may relax unlisted channels on its simulator
while its live machine refuses them. It carries the resolved posture together
with the config key that answered it, so a refusal names the line an operator
would have to edit rather than one some per-type block overrides.

Where the write family reads an unusable value as "not armed" and is done, the
limits family cannot: "no limits checking configured" is itself a permissive
answer, so a leaf written as something no reader can turn into a boolean is
reported as a block that failed to state it rather than as a block that stated
nothing. Such a posture answers neither leaf, and a caller that must act blocks
every write instead of waving them through unchecked.
"""

from dataclasses import dataclass
from typing import Any

# -- Control system connector types (have implementations) --
MOCK = "mock"
EPICS = "epics"
VIRTUAL_ACCELERATOR = "virtual_accelerator"
DOOCS = "doocs"

#: The live stand-in: a facility-shaped soft IOC a deployment runs for itself.
#: A connector type of its own — served by the EPICS connector, but keyed apart
#: from ``epics`` so that the facility's authored ``epics`` block stays the one
#: thing ``live`` can mean, and the stand-in gets its own
#: ``control_system.connector.live_standin`` block to be configured from.
LIVE_STANDIN = "live_standin"

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"
MONGODB_ARCHIVER = "mongodb_archiver"
DOOCS_ARCHIVER = "doocs_archiver"

# -- CLI choice lists (only types with implementations) --
CLI_CONTROL_SYSTEM_TYPES = [MOCK, EPICS, VIRTUAL_ACCELERATOR, DOOCS]
CLI_ARCHIVER_TYPES = [MOCK_ARCHIVER, EPICS_ARCHIVER, MONGODB_ARCHIVER, DOOCS_ARCHIVER]

#: Settable, not initable. A deployment may be pointed at its stand-in once it
#: has one — ``osprey set connector=live_standin`` is how a session starts on
#: the soft IOC — but ``osprey init`` never offers it: a fresh project has no
#: stand-in to point at, so a project that began on one would name a machine
#: that does not exist. Hence one list for what the ``connector:`` shorthand
#: accepts and another, narrower one for what ``init`` will materialize.
SET_CONTROL_SYSTEM_TYPES = [*CLI_CONTROL_SYSTEM_TYPES, LIVE_STANDIN]

# -- Session targets --
# What a *session* can be pointed at, as opposed to what a *config* selects.
# Deliberately three names and not one per connector type: a target names a
# machine (the facility's own, the virtual accelerator, the stand-in soft IOC),
# and which connector type reaches that machine is this module's job to answer,
# not the caller's.
TARGET_LIVE = "live"
TARGET_VA = "va"
TARGET_STANDIN = "standin"
CONTROL_TARGETS = [TARGET_LIVE, TARGET_VA, TARGET_STANDIN]

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

#: Types that speak real Channel Access — the facility's own EPICS machine, a
#: virtual-accelerator soft-IOC, or the live stand-in soft-IOC. The queue worker
#: builds its devices over Channel Access and nothing else, so these are the
#: only types plans can execute against; every other type browses.
CHANNEL_ACCESS_TYPES = (EPICS, VIRTUAL_ACCELERATOR, LIVE_STANDIN)

#: The target each self-standing machine's type is the baseline of. A type
#: absent from this table describes the facility's own machine, hence ``live``.
_BASELINE_TARGETS = {VIRTUAL_ACCELERATOR: TARGET_VA, LIVE_STANDIN: TARGET_STANDIN}


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

    ``va`` and ``standin`` are the same answer everywhere: a virtual accelerator
    is a virtual accelerator regardless of what the deployment was built for,
    and the stand-in is the soft IOC the deployment runs for itself, configured
    from the block that carries its name. ``live`` is deployment-specific and is
    the half that can refuse:

    - When the section's own type is a real control system, that is the live
      machine, and it is returned as written — including a value this module
      does not recognize, which reaches the factory's "Unknown … type" error
      exactly as :func:`resolve_control_system_type` already lets it.
    - When the section's own type is simulated or is a stand-in (or absent,
      which resolves to the mock), the deployment has not named its real machine
      there, so the live type is taken from the connector table: exactly one
      block that is neither simulated nor a stand-in means the deployment has
      said which machine it means. None, or more than one, raises. The stand-in
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
    if target == TARGET_STANDIN:
        return LIVE_STANDIN
    if target == TARGET_LIVE:
        return _live_type(section)
    raise ValueError(
        f"Unknown control target {target!r}. Valid targets are "
        f"{TARGET_LIVE!r}, {TARGET_VA!r} and {TARGET_STANDIN!r}, spelled "
        "exactly. A session target is always stated, never defaulted — there is "
        "no target a caller gets by saying nothing, because the one it would "
        "get could be the real machine."
    )


def baseline_target(section: Any) -> str:
    """The target a deployment's own ``control_system:`` section selects.

    ``va`` for a virtual accelerator, ``standin`` for the live stand-in, and
    ``live`` for everything else — including a mock deployment, whose ``live``
    may well be underivable, because ``live`` is still the target its section
    describes.

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

    Two conditions, neither sufficient alone:

    1. **The deployment's baseline target resolves back to its own control
       system.** ``resolve_target`` answers ``live`` for configs with no
       business switching: a ``mock`` deployment that happens to carry an
       ``epics`` block resolves ``live`` to ``epics``, and treating that as
       switchable would point a session at a real machine the config never
       selected. Requiring :func:`baseline_target` to resolve back to
       ``control_system.type`` rules it out. The baseline is *resolved*, never
       assumed to be ``live`` or ``va``: a ``live_standin`` deployment is
       baselined on ``standin`` and is no less switchable for it.
    2. **At least two targets are configured** (:func:`configured_targets`,
       the same resolve-and-read-the-block enumeration every roster walks).
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
        ``True`` when the section is consistent and configures at least two
        targets. Never raises: every malformed, partial or contradictory config
        is simply not capable.
    """
    if not isinstance(section, dict):
        return False
    try:
        if resolve_target(section, baseline_target(section)) != resolve_control_system_type(
            section
        ):
            return False
    except ValueError:
        return False
    return len(configured_targets(section)) >= 2


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
        target: The session target being asked about.

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

    Every one of :func:`configured_targets` when the deployment renders the
    switch (:func:`switch_capable`), each through :func:`target_writes_enabled`
    — the configured targets and never :data:`CONTROL_TARGETS`, which is the
    vocabulary of machines that can exist rather than the ones this deployment
    has. Looping the constant would give a deployment with no
    ``control_system.connector.live_standin`` block a ``standin`` key anyway,
    armed or not from the deployment-wide flag through
    :func:`target_writes_enabled`'s unresolvable-target fallback: a posture
    published for a stand-in nobody stood up. Widening the vocabulary must not
    grow a slot on a deployment the widening did not change.

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
        targets = configured_targets(section)
        return {target: target_writes_enabled(section, target) for target in targets}
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
LIMITS_LEAVES = ("enabled", "allow_unlisted_channels")


@dataclass(frozen=True)
class LimitsPosture:
    """A resolved limits posture, together with the config key that answered it.

    The value and its key travel as one because a refusal has to send an
    operator to the line they can actually edit. A deployment that relaxed
    ``allow_unlisted_channels`` for its simulator alone has two keys in play,
    and a refusal naming the deployment-wide one would have the operator flip a
    line the per-type block overrides — a change that does nothing, on a
    posture they did not mean to touch. Resolving the value without carrying
    its key back is what makes that mistake available, so this type does not
    offer the value on its own.

    Both leaves are tri-state. ``None`` is "no block said", which is not the
    same as a block saying ``false``: silence is what a deployment that never
    configured limits checking has, and every write path treats it as no
    permission rather than as a decision. Only :data:`LIMITS_LEAVES` are
    carried; the limits database path stays deployment-wide.

    Attributes:
        enabled: Whether limits checking is on, or ``None`` when unstated.
        allow_unlisted: Whether channels absent from the limits database may be
            written, or ``None`` when unstated. Write paths allow only on
            ``True``.
        connector_type: The connector type whose block answered, or ``None``
            when the deployment-wide block did. It selects the key spelling and
            nothing else.
        incomplete: The :data:`LIMITS_LEAVES` the answering block failed to
            state, in :data:`LIMITS_LEAVES` order — a leaf a per-type block
            omitted, or a leaf either block wrote as something other than a
            literal boolean. Empty for a well-formed block. A non-empty value
            means the posture answered nothing — both leaves are ``None`` —
            and a reader that must act builds a failsafe rather than guessing
            which half of the block was meant. The one posture not read from a
            single block, :func:`most_restrictive_limits_posture`'s fold, may
            carry definite leaves beside a non-empty tuple; readers check this
            field first for exactly that reason.
    """

    enabled: bool | None
    allow_unlisted: bool | None
    connector_type: str | None
    incomplete: tuple[str, ...] = ()

    def key(self, leaf: str) -> str:
        """The config key this posture's *leaf* was read from.

        ``control_system.connector.<type>.limits_checking.<leaf>`` when a
        connector block answered, and ``control_system.limits_checking.<leaf>``
        when the deployment-wide block did. A custom connector's dotted module
        path (``'mypackage.TangoConnector'``) is one key and is interpolated
        whole, never split on its dots.

        A posture holding no type at all — the deployment-wide answer, and the
        fallback a target that resolves to no type gets — spells the
        deployment-wide key, which is where such a posture was read from
        anyway. Same reading :func:`writes_enabled_key` gives the same case.

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

        Limits checking on *and* unlisted channels explicitly refused — the only
        definition of "strict" in the codebase, so the switch gate, the target
        roster and the docs cannot drift into three readings of one word.

        Explicit on both leaves: ``None`` is a deployment that never stated a
        posture, and a deployment that stated nothing has refused nothing.
        Counting silence as strict would advertise a guarantee no config line
        backs, on exactly the deployments least likely to have checked.
        """
        return self.enabled is True and self.allow_unlisted is False


def type_limits_posture(section: Any, connector_type: str | None) -> LimitsPosture:
    """The limits posture one connector *type* runs under.

    ``control_system.connector.<type>.limits_checking`` when that block is
    there, and ``control_system.limits_checking`` when it is not, where
    *connector_type* is one key. A custom connector's dotted module path
    (``'mypackage.TangoConnector'``) names a single block, never a path through
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

    Keyed by type rather than by session target for the reason
    :func:`type_writes_enabled` is: a connector, the factory that builds it, an
    IPC child all know which connector type they are and never which target
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
    holder following the session target and the connector that knows only its
    own type read one posture and not two. The roster row, the stdlib hook, the
    executor and the tool layer all arrive here; the connector, the factory and
    the IPC child arrive at :func:`type_limits_posture` — and a deployment that
    described one machine must not answer them differently.

    A target that does not resolve answers the deployment-wide block instead.
    Two shapes reach that branch and both mean the same thing: there is no
    per-type block to consult because there is no type. An unknown target names
    nothing. ``live`` on a mock or hello_world-style deployment names a machine
    the config never described, which :func:`resolve_target` refuses to guess.
    Such a deployment has only ever had the one deployment-wide block, and
    keeping it is parity rather than a fallback — the same reading
    :func:`target_writes_enabled` takes, so the two postures a refusal may quote
    cannot disagree about which deployments have a second machine. Refusing here
    would instead take the posture away from every deployment that never had a
    second target.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes.
        target: The session target, one of :data:`CONTROL_TARGETS`.

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
    the one that needs it: when the session's control target cannot be read —
    no state written yet, an unreadable state directory, a target that resolves
    to nothing — it still has to decide about a write, and the machine it is
    deciding about is any of the reachable ones.

    The reachable set is exactly :func:`session_posture`'s: every
    :func:`configured_targets` when the deployment renders the switch
    (:func:`switch_capable`), and otherwise the single connector
    ``control_system.type`` builds, read by *type* through
    :func:`type_limits_posture`. Reading the baseline by type rather than by
    target is what keeps a mock deployment that happens to carry an ``epics``
    block from folding that block's relaxation into an answer no session here
    could ever reach; and walking the configured targets rather than
    :data:`CONTROL_TARGETS` keeps :func:`target_limits_posture`'s
    unresolvable-target fallback from voting for a machine nobody stood up.

    The two leaves fold in opposite directions, each toward the answer that
    costs a config edit rather than a machine:

    - ``enabled`` is ``True`` when *any* reachable posture has it ``True``.
      Limits checking being on somewhere is a constraint that may apply to the
      write in hand.
    - ``allow_unlisted`` is ``True`` only when *every* reachable posture has it
      ``True``. Permission to write a channel the limits database does not list
      holds only where every machine grants it, so one strict target — or one
      that states nothing, ``None`` and an incomplete block alike counting as
      not-``True`` — makes the answer strict.

    Incompleteness travels with them, as the union of every reachable
    posture's :attr:`LimitsPosture.incomplete` in :data:`LIMITS_LEAVES` order.
    One reachable machine whose block cannot be read makes the fold incomplete,
    and a reader that must act therefore builds the blocking failsafe. Dropping
    it would be the worst failure available here: both leaf folds send an
    incomplete posture's ``None`` to ``False``, which reads as "checking off,
    nothing permitted" — and a validator built from *that* is no validator at
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
        allow_unlisted=all(posture.allow_unlisted is True for posture in postures),
        connector_type=None,
        incomplete=tuple(
            leaf
            for leaf in LIMITS_LEAVES
            if any(leaf in posture.incomplete for posture in postures)
        ),
    )


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
    - **Either** block writing a leaf as something no reader can turn into a
      boolean — a quoted ``'true'``, a ``1``, a bare ``enabled:``, an
      unexpanded ``'${LIMITS_ON}'``. Environment expansion yields strings, so
      this is what a deployment gets when it wires a limits leaf to a variable
      nothing set. It blocks every write at runtime, which is the safe
      direction but a poor way to find out; refusing the build is where that
      config is cheap to fix.
    - **Either** ``limits_checking`` being present but not a mapping at all,
      which gets one line naming the block and quoting what was found there
      instead of a line per leaf: there are no leaves to name.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`resolve_control_system_type` takes — the *rendered* one, as
            the build produced it, since that is the config a deployment runs.

    Returns:
        One line per unstated leaf: the deployment-wide block first, then the
        connector blocks in the order the section carries them, leaves in
        :data:`LIMITS_LEAVES` order, so one refusal lists everything an operator
        has to fix. Empty for a section whose blocks are complete, absent, or
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
                    f"{posture.key(leaf)} is {block[leaf]!r}, not a literal true or false; "
                    "a limits leaf that cannot be read states no posture and blocks "
                    "every write as a failsafe"
                )
            else:
                errors.append(
                    f"{posture.key(leaf)} is missing; a per-type limits block must state "
                    f"both {LIMITS_LEAVES[0]} and {LIMITS_LEAVES[1]}"
                )
    return errors


def _global_writes_enabled(section: Any) -> bool:
    """The deployment-wide posture, ``control_system.writes_enabled``.

    Explicitly ``True`` and nothing else — the same reading the per-type leaf
    gets, so a quoted ``'true'`` or a ``1`` arms nothing at either level.
    """
    return isinstance(section, dict) and section.get(TYPE_WRITES_ENABLED_LEAF) is True


def _global_limits_posture(section: Any) -> LimitsPosture:
    """The deployment-wide posture, ``control_system.limits_checking``.

    A leaf this block does not carry is the tri-state and not a malformed
    block: silence here is the shape every deployment predating per-type blocks
    has, and only a per-type block, which overrides whole, has to state both
    leaves to answer at all. A leaf it *does* carry but spells as something
    other than a literal boolean is unreadable in either block alike — see
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
    - **Written but unreadable** (``True``) — a quoted ``'true'``, a ``1``, a
      bare ``enabled:``, an unexpanded ``'${LIMITS_ON}'`` that no environment
      resolved — which is a failure in *either* scope. This is the leaf that
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
        elif _limits_leaf(block[leaf]) is None:
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
    enabled_leaf, allow_unlisted_leaf = LIMITS_LEAVES
    return LimitsPosture(
        enabled=_limits_leaf(block.get(enabled_leaf)),
        allow_unlisted=_limits_leaf(block.get(allow_unlisted_leaf)),
        connector_type=connector_type,
    )


def _limits_leaf(value: Any) -> bool | None:
    """One limits leaf as a literal boolean, or ``None`` for anything else.

    The literal booleans and nothing else are readable. A quoted ``'true'``, a
    ``1``, a bare ``enabled:`` — the same values :func:`type_writes_enabled`
    refuses to read as an arming — are not.

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


def _live_type(section: Any) -> str:
    """The control system type that reaches this deployment's real machine.

    Neither a simulated type nor a stand-in one can be that machine, and the
    exclusion holds on both sides of the derivation — the baseline it starts
    from, and the candidate blocks it falls back to. A stand-in is a machine the
    deployment stands up itself; counting it would either answer ``live`` with
    the stand-in, or make ``live`` ambiguous on exactly the deployments that run
    the stand-in beside the block naming their facility's own machine.
    """
    never_live = _SIMULATED_TYPES + STANDIN_TYPES
    baseline = resolve_control_system_type(section)
    if baseline not in never_live:
        return baseline

    connector = section.get("connector") if isinstance(section, dict) else None
    candidates: list[str] = []
    if isinstance(connector, dict):
        candidates = sorted(
            key for key in connector if isinstance(key, str) and key not in never_live
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
