"""The limits posture a connector type, or a session target, runs under.

:class:`LimitsPosture` is the limits half of a deployment's posture, which is
per connector type like the write tier's machine: a deployment may run its
simulator ``optional`` while its live machine runs ``exclusive``. It carries
the resolved posture together with the config key that answered it, so a
refusal names the line an operator would have to edit rather than one some
per-type block overrides.

Where the write tier reads an unusable value as ``never`` and is done, the
limits family cannot: "no limits checking configured" is itself a permissive
answer, so a leaf written as something no reader can turn into a boolean is
reported as a block that failed to state it rather than as a block that stated
nothing. Such a posture answers neither leaf, and a caller that must act blocks
every write instead of waving them through unchecked.
"""

from typing import Final, get_args

from osprey_connectors.config_sections import (
    ConnectorBlock,
    ControlSystemSection,
    LimitsBlock,
    LimitsMode,
    Unreadable,
)
from osprey_connectors.connector_types import resolve_control_system_type
from osprey_connectors.model import FrozenModel
from osprey_connectors.targets import configured_targets, resolve_target, switch_capable
from osprey_connectors.write_tier import tier_allows

__all__ = [
    "LIMITS_CHECKING_LEAF",
    "LIMITS_LEAVES",
    "LIMITS_MODES",
    "LIMITS_MODE_EXCLUSIVE",
    "LIMITS_MODE_OPTIONAL",
    "LimitsPosture",
    "incomplete_limits_blocks",
    "limits_database_required",
    "most_restrictive_limits_posture",
    "target_limits_posture",
    "tier_limits_database_targets",
    "type_limits_posture",
]

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

#: The two leaves of a limits posture, by name.
_ENABLED_LEAF = "enabled"
_MODE_LEAF = "mode"

#: The leaves a limits block has to state to answer at all. ``database_path`` is
#: deliberately not among them: the deployment mounts one limits database, so a
#: per-type block that omits the path is complete rather than incomplete.
LIMITS_LEAVES = (_ENABLED_LEAF, _MODE_LEAF)

#: The two limits modes
#: (:data:`~osprey_connectors.config_sections.LimitsMode`), by name.
LIMITS_MODE_EXCLUSIVE: Final = "exclusive"
LIMITS_MODE_OPTIONAL: Final = "optional"
#: Every :data:`LimitsMode` word, ``exclusive`` first.
LIMITS_MODES: tuple[LimitsMode, ...] = get_args(LimitsMode)

#: Every leaf the deployment-wide ``limits_checking`` block defines: the posture
#: pair plus the database path, which is deployment-wide only. A per-type block
#: defines :data:`LIMITS_LEAVES` alone. A block carrying any other leaf is
#: refused by :func:`incomplete_limits_blocks`, so a misspelt or retired leaf
#: cannot sit in a config looking like a posture nobody reads.
_LIMITS_BLOCK_LEAVES = (*LIMITS_LEAVES, "database_path")


class LimitsPosture(FrozenModel):
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
        mode: A :data:`LimitsMode`, or ``None`` when unstated. Write
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
    mode: LimitsMode | None
    connector_type: str | None
    incomplete: tuple[str, ...] = ()

    def key(self, leaf: str) -> str:
        """The config key this posture's *leaf* was read from.

        ``control_system.connector.<type>.limits_checking.<leaf>`` when a
        connector block answered, and ``control_system.limits_checking.<leaf>``
        when the deployment-wide block did. A custom connector's dotted module
        path (``'mypackage.SiteConnector'``) is one key and is interpolated
        whole, never split on its dots.

        A posture holding no type at all — the deployment-wide answer, and the
        fallback a target that resolves to no type gets — spells the
        deployment-wide key, which is where such a posture was read from
        anyway.

        Args:
            leaf: One of :data:`LIMITS_LEAVES`.

        Returns:
            The dotted key, as a caller spells it for ``LoadedConfig.get`` and
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
        return self.key(_ENABLED_LEAF).rsplit(".", 1)[0]

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


def type_limits_posture(section: ControlSystemSection, connector_type: str | None) -> LimitsPosture:
    """The limits posture one connector *type* runs under.

    ``control_system.connector.<type>.limits_checking`` when that block is
    there, and ``control_system.limits_checking`` when it is not, where
    *connector_type* is one key. A custom connector's dotted module path
    (``'mypackage.SiteConnector'``) names a single block, never a path through
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
            :func:`~osprey_connectors.connector_types.resolve_control_system_type` takes.
        connector_type: A connector type name, as
            :func:`~osprey_connectors.connector_types.resolve_control_system_type` and
            :func:`~osprey_connectors.targets.resolve_target` return it — which is also
            its connector sub-block key. ``None`` (or an empty name) asks the
            deployment-wide question, which is what a caller holding no type at all has
            always asked.

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
        block = section.block(connector_type)
        if isinstance(block, ConnectorBlock) and block.limits_checking is not None:
            limits = block.limits_checking
            if isinstance(limits, Unreadable):
                return LimitsPosture(
                    enabled=None,
                    mode=None,
                    connector_type=connector_type,
                    incomplete=LIMITS_LEAVES,
                )
            return _block_limits_posture(limits, connector_type, whole=True)
    return _global_limits_posture(section)


def target_limits_posture(section: ControlSystemSection, target: object) -> LimitsPosture:
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
    :func:`~osprey_connectors.targets.resolve_target` refuses to guess.
    Such a deployment has only ever had the one deployment-wide block, and
    keeping it is parity rather than a fallback: a target this deployment never
    configured still has the one block every such deployment had. Refusing here
    would instead take the posture away from every deployment that never had a
    second target.

    Args:
        section: The ``control_system:`` config section, in the same shape
            :func:`~osprey_connectors.connector_types.resolve_control_system_type`
            takes.
        target: The control target, one of
            :data:`~osprey_connectors.connector_types.CONTROL_TARGETS`.

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


def most_restrictive_limits_posture(section: ControlSystemSection) -> LimitsPosture:
    """The limits posture that holds across every target a session here can select.

    What a caller with no target of its own has to assume. The stdlib hook is
    the one that needs it: when the deployment's control target cannot be read —
    no state written yet, an unreadable state directory, a target that resolves
    to nothing — it still has to decide about a write, and the machine it is
    deciding about is any of the reachable ones.

    The reachable set is every
    :func:`~osprey_connectors.targets.configured_targets` when the deployment renders the switch
    (:func:`~osprey_connectors.targets.switch_capable`), and otherwise the single connector
    ``control_system.type`` builds, read by *type* through
    :func:`type_limits_posture`. Reading the baseline by type rather than by
    target is what keeps a deployment without the switch from folding a
    second block's relaxation into an answer no session here could ever reach;
    and walking the configured targets rather than
    :data:`~osprey_connectors.connector_types.CONTROL_TARGETS` keeps :func:`target_limits_posture`'s
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
            :func:`~osprey_connectors.connector_types.resolve_control_system_type` takes.

    Returns:
        The folded :class:`LimitsPosture`, always carrying ``None`` for its connector
        type, and incomplete when any reachable posture is. Never raises and never folds
        an empty set: :func:`~osprey_connectors.targets.configured_targets` always holds
        the baseline, and a deployment that says nothing still has one target.
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


def tier_limits_database_targets(
    section: ControlSystemSection, *, path_configured: bool
) -> list[str]:
    """The targets the login tier lets write whose limits posture needs a database.

    The tier is a ceiling, not a selection: a render whose
    :func:`~osprey_connectors.write_tier.tier_writes` answers ``asked`` or ``allowed``
    lets a session write to every :func:`~osprey_connectors.targets.configured_targets`,
    so every one of them whose resolved :func:`target_limits_posture` makes
    :func:`limits_database_required` true — limits checking on, and a mode other than
    ``optional`` or a path configured — needs the channel-limits database to exist and
    load. A ``never`` render writes nowhere and so arms none. This is the one list a
    deployment-level caller reads before it demands the file (the compose mount, the
    build check), so none of them can refuse a deployment another one runs; a caller
    that serves one lane or one type asks :func:`limits_database_required` for that
    posture directly.

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
        The qualifying targets in
        :func:`~osprey_connectors.targets.configured_targets` order; empty for a
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


def incomplete_limits_blocks(section: ControlSystemSection) -> list[str]:
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
            :func:`~osprey_connectors.connector_types.resolve_control_system_type` takes
            — the *rendered* one, as the build produced it, since that is the config a
            deployment runs.

    Returns:
        One line per unstated leaf and per undefined one: the deployment-wide
        block first, then the connector blocks in the order the section carries
        them, leaves in :data:`LIMITS_LEAVES` order and undefined leaves sorted
        after them, so one refusal lists everything an operator has to fix. Empty for a section whose blocks are complete, absent, or
        not blocks at all. Never raises: a lint that crashed on a malformed
        config would fail a build without saying what is wrong with it.
    """
    # A lint quotes what was written, so it walks the section as written: a
    # connector key that is not a string is still a block an operator wrote.
    raw = section.raw
    connector = raw.get("connector")
    blocks: list[tuple[str | None, object, bool]] = []
    if LIMITS_CHECKING_LEAF in raw:
        blocks.append((None, raw[LIMITS_CHECKING_LEAF], False))
    if isinstance(connector, dict):
        # A YAML key need not be a string. One that is not is spelled as text,
        # as the key names it; an empty one names the deployment-wide block.
        blocks.extend(
            (str(connector_type) if connector_type else None, block[LIMITS_CHECKING_LEAF], True)
            for connector_type, block in connector.items()
            if isinstance(block, dict) and LIMITS_CHECKING_LEAF in block
        )
    errors: list[str] = []
    for connector_type, block, whole in blocks:
        # The posture this block would answer with, built for its keys alone —
        # so the line naming the block and the lines naming its leaves are
        # spelled by one thing.
        posture = LimitsPosture(enabled=None, mode=None, connector_type=connector_type)
        if not isinstance(block, dict):
            errors.append(
                f"{posture.block_key} is {block!r}, not a block; a limits "
                f"block is a mapping stating {_ENABLED_LEAF} and {_MODE_LEAF}"
            )
            continue
        parsed = LimitsBlock.read(block, posture.block_key)
        for leaf, written in _unstated_limits_leaves(parsed, whole=whole):
            if written:
                errors.append(
                    f"{posture.key(leaf)} is {block[leaf]!r}, not {_LEAF_VALUES[leaf]}; "
                    "a limits leaf that cannot be read states no posture and blocks "
                    "every write as a failsafe"
                )
            else:
                errors.append(
                    f"{posture.key(leaf)} is missing; a per-type limits block must state "
                    f"both {_ENABLED_LEAF} and {_MODE_LEAF}"
                )
        defined = LIMITS_LEAVES if whole else _LIMITS_BLOCK_LEAVES
        errors.extend(
            f"{posture.key(leaf)} is not a limits leaf; a limits block states "
            f"{_ENABLED_LEAF} and {LIMITS_CHECKING_LEAF}.{_MODE_LEAF}: "
            f"{' | '.join(LIMITS_MODES)}"
            for leaf in sorted(str(key) for key in block if key not in defined)
        )
    return errors


def _global_limits_posture(section: ControlSystemSection) -> LimitsPosture:
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
    block = section.limits_checking
    if block is None:
        return _block_limits_posture(LimitsBlock(), None, whole=False)
    if isinstance(block, Unreadable):
        return LimitsPosture(enabled=None, mode=None, connector_type=None, incomplete=LIMITS_LEAVES)
    return _block_limits_posture(block, None, whole=False)


def _unstated_limits_leaves(block: LimitsBlock, *, whole: bool) -> list[tuple[str, bool]]:
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
        block: A ``limits_checking`` block, deployment-wide or per type, as
            :class:`~osprey_connectors.config_sections.LimitsBlock` read it.
        whole: Whether an absent leaf is a failure, which it is exactly when
            the block overrides another one entire.

    Returns:
        ``(leaf, was_written)`` in :data:`LIMITS_LEAVES` order, empty for a
        block that states everything it has to.
    """
    unstated: list[tuple[str, bool]] = []
    for leaf, value in ((_ENABLED_LEAF, block.enabled), (_MODE_LEAF, block.mode)):
        if value is None:
            if whole:
                unstated.append((leaf, False))
        elif isinstance(value, Unreadable):
            unstated.append((leaf, True))
    return unstated


def _block_limits_posture(
    block: LimitsBlock, connector_type: str | None, *, whole: bool
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
        return LimitsPosture(
            enabled=None,
            mode=None,
            connector_type=connector_type,
            incomplete=tuple(leaf for leaf, _ in unstated),
        )
    # Neither leaf is unreadable here: each is its value or unstated.
    return LimitsPosture(
        enabled=None if isinstance(block.enabled, Unreadable) else block.enabled,
        mode=None if isinstance(block.mode, Unreadable) else block.mode,
        connector_type=connector_type,
    )


#: What a refusal says a readable value of each of :data:`LIMITS_LEAVES` looks
#: like. Which values are readable is
#: :class:`~osprey_connectors.config_sections.LimitsBlock`'s rule.
_LEAF_VALUES = {
    _ENABLED_LEAF: "a literal true or false",
    _MODE_LEAF: " or ".join(LIMITS_MODES),
}
