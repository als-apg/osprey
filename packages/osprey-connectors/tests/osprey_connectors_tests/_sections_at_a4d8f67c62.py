"""The 23 section functions as they were at a4d8f67c62, copied verbatim, for the equivalence test.

Each function and the private helpers it calls is copied from its module at
a4d8f67c62 with its docstring dropped and every copied name prefixed ``_old``;
the bodies are unchanged. They read the raw ``control_system`` mapping, as
they did, so :mod:`test_section_equivalence` can hand both generations one
section and compare their answers. Constants and the record readers come from
the live modules: only the section reading is under test.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard, cast

from osprey_connectors import _control_state, control_context
from osprey_connectors.config_sections import LimitsMode, TierWord
from osprey_connectors.connector_types import (
    _TRANSPORTS,
    CONTROL_TARGETS,
    DOOCS,
    EPICS,
    IN_PROCESS,
    LIVE_STANDIN,
    SERVED,
    SERVING_KEY,
    SERVING_LEAF,
    SIMULATED_TYPES,
    STANDIN_TYPES,
    TARGET_LIVE,
    TARGET_STANDIN,
    TARGET_VA,
    TRANSPORT_CA,
    TRANSPORT_IN_PROCESS,
    VIRTUAL_ACCELERATOR,
    ControlTarget,
    MachineKind,
    Serving,
    Transport,
    is_control_target,
)
from osprey_connectors.errors import ConfigurationError
from osprey_connectors.execution_mode import is_readonly_run
from osprey_connectors.identity import acting_identity
from osprey_connectors.limits_posture import (
    _ENABLED_LEAF,
    _LEAF_VALUES,
    _LIMITS_BLOCK_LEAVES,
    _MODE_LEAF,
    LIMITS_CHECKING_LEAF,
    LIMITS_LEAVES,
    LIMITS_MODE_EXCLUSIVE,
    LIMITS_MODE_OPTIONAL,
    LIMITS_MODES,
    LimitsPosture,
    limits_database_required,
)
from osprey_connectors.model import evolve
from osprey_connectors.refusals import VerdictReason
from osprey_connectors.targets import _BASELINE_TARGETS, _MACHINE_KINDS, _ONE_REAL_MACHINE
from osprey_connectors.write_switch import (
    _LOGIN_TIERS,
    _LOGIN_TIERS_LEAF,
    NO_OWNER,
    Allowed,
    ArmedPosition,
    NoOwner,
    OwnerTierTerm,
    Refused,
    WriteCeiling,
    WriteVerdict,
    _Armed,
    _RecordSeen,
    current_owner,
    session_hold_refusal,
)

logger = logging.getLogger("osprey_connectors.targets")

#: ``write_tier._WRITES_LEAF`` at a4d8f67c62.
_WRITES_LEAF = "writes"

#: The second-real-block warning's latch, apart from the live one.
_old__SECOND_REAL_BLOCK_WARNED: set[tuple[str, tuple[str, ...]]] = set()

# -- connector_types.py at a4d8f67c62 ---------------------------------------------


def _old__resolve_type(section: object, fallback: str) -> str:
    value = section.get("type") if isinstance(section, dict) else None
    return str(value) if value else fallback


def _old_resolve_control_system_type(section: object) -> str:
    return _old__resolve_type(section, VIRTUAL_ACCELERATOR)


def _old__states_type(section: object) -> TypeGuard[dict[object, object]]:
    return isinstance(section, dict) and bool(section.get("type"))


def _old_resolve_serving(section: object) -> Serving:
    if not _old__states_type(section):
        return IN_PROCESS
    if _old_resolve_control_system_type(section) != VIRTUAL_ACCELERATOR:
        return SERVED
    connector = section.get("connector")
    block: object = connector.get(VIRTUAL_ACCELERATOR) if isinstance(connector, dict) else None
    value: object = block.get(SERVING_LEAF) if isinstance(block, dict) else None
    if value is None or value == SERVED:
        return SERVED
    if value == IN_PROCESS:
        return IN_PROCESS
    raise ConfigurationError(
        f"{SERVING_KEY} is {value!r}; it is {SERVED!r} or {IN_PROCESS!r}.", key=SERVING_KEY
    )


def _old_connector_transport(
    section: object, connector_type: str | None = None
) -> Transport | None:
    if connector_type is None:
        connector_type = _old_resolve_control_system_type(section)
    if connector_type == VIRTUAL_ACCELERATOR and _old_resolve_serving(section) == IN_PROCESS:
        return TRANSPORT_IN_PROCESS
    return _TRANSPORTS.get(connector_type)


def _old_speaks_channel_access(section: object, connector_type: str | None = None) -> bool:
    return _old_connector_transport(section, connector_type) == TRANSPORT_CA


def _old_talks_to_network(section: object, connector_type: str | None = None) -> bool:
    return _old_connector_transport(section, connector_type) != TRANSPORT_IN_PROCESS


# -- targets.py at a4d8f67c62 -----------------------------------------------------


def _old_resolve_target(section: object, target: object) -> str:
    if target == TARGET_VA:
        return VIRTUAL_ACCELERATOR
    if target == TARGET_STANDIN:
        return LIVE_STANDIN
    if target == TARGET_LIVE:
        return _old__live_type(section)
    raise ValueError(
        f"Unknown control target {target!r}. Valid targets are "
        f"{TARGET_LIVE!r}, {TARGET_VA!r} and {TARGET_STANDIN!r}, spelled "
        f"exactly. {_ONE_REAL_MACHINE} A control target is always stated, never "
        "defaulted — there is no target a caller gets by saying nothing, "
        "because the one it would get could be the real machine."
    )


def _old_baseline_target(section: object) -> ControlTarget:
    return _BASELINE_TARGETS.get(_old_resolve_control_system_type(section), TARGET_LIVE)


def _old_switch_capable(section: object) -> bool:
    if not isinstance(section, dict):
        return False
    try:
        return len(_old_configured_targets(section)) >= 2
    except ValueError:
        return False


def _old_target_configured(section: object, target: object) -> bool:
    try:
        connector_type = _old_resolve_target(section, target)
    except ValueError:
        return False
    connector = section.get("connector") if isinstance(section, dict) else None
    block = connector.get(connector_type) if isinstance(connector, dict) else None
    return isinstance(block, dict) and bool(block)


def _old_configured_targets(section: object) -> list[ControlTarget]:
    baseline = _old_baseline_target(section)
    return [
        target
        for target in CONTROL_TARGETS
        if target == baseline or _old_target_configured(section, target)
    ]


def _old_machine_kind(section: object, target: object) -> MachineKind | None:
    try:
        _old_resolve_target(section, target)
    except ValueError:
        return None
    return _MACHINE_KINDS.get(target) if is_control_target(target) else None


def _old__real_blocks(section: object, never_live: tuple[str, ...]) -> list[str]:
    connector = section.get("connector") if isinstance(section, dict) else None
    if not isinstance(connector, dict):
        return []
    return sorted(key for key in connector if isinstance(key, str) and key not in never_live)


def _old__report_second_real_block(
    section: object, baseline: str, never_live: tuple[str, ...]
) -> None:
    others = [name for name in _old__real_blocks(section, never_live) if name != baseline]
    if not others:
        return
    seen = (baseline, tuple(others))
    if seen in _old__SECOND_REAL_BLOCK_WARNED:
        return
    _old__SECOND_REAL_BLOCK_WARNED.add(seen)
    logger.warning(
        "'control_system.connector' carries %s beside this deployment's own %r block, "
        "and no control target reaches %s. %s",
        ", ".join(repr(name) for name in others),
        baseline,
        "them" if len(others) > 1 else "it",
        _ONE_REAL_MACHINE,
    )


def _old__live_type(section: object) -> str:
    never_live = SIMULATED_TYPES + STANDIN_TYPES
    baseline = _old_resolve_control_system_type(section)
    if baseline not in never_live:
        _old__report_second_real_block(section, baseline, never_live)
        return baseline

    candidates = _old__real_blocks(section, never_live)
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


# -- write_tier.py at a4d8f67c62 --------------------------------------------------


def _old_tier_writes(section: object) -> TierWord:
    value = section.get(_WRITES_LEAF) if isinstance(section, dict) else None
    if value == "asked":
        return "asked"
    if value == "allowed":
        return "allowed"
    return "never"


def _old_tier_allows(section: object) -> bool:
    return _old_tier_writes(section) != "never"


# -- limits_posture.py at a4d8f67c62 ----------------------------------------------


def _old_type_limits_posture(section: object, connector_type: str | None) -> LimitsPosture:
    if connector_type:
        connector = section.get("connector") if isinstance(section, dict) else None
        block = connector.get(connector_type) if isinstance(connector, dict) else None
        if isinstance(block, dict) and LIMITS_CHECKING_LEAF in block:
            limits = block[LIMITS_CHECKING_LEAF]
            if not isinstance(limits, dict):
                return LimitsPosture(
                    enabled=None,
                    mode=None,
                    connector_type=connector_type,
                    incomplete=LIMITS_LEAVES,
                )
            return _old__block_limits_posture(limits, connector_type, whole=True)
    return _old__global_limits_posture(section)


def _old_target_limits_posture(section: object, target: object) -> LimitsPosture:
    try:
        connector_type = _old_resolve_target(section, target)
    except ValueError:
        return _old__global_limits_posture(section)
    return _old_type_limits_posture(section, connector_type)


def _old_most_restrictive_limits_posture(section: object) -> LimitsPosture:
    if _old_switch_capable(section):
        postures = [
            _old_target_limits_posture(section, target)
            for target in _old_configured_targets(section)
        ]
    else:
        postures = [_old_type_limits_posture(section, _old_resolve_control_system_type(section))]
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


def _old_tier_limits_database_targets(section: object, *, path_configured: bool) -> list[str]:
    if not _old_tier_allows(section):
        return []
    return [
        target
        for target in _old_configured_targets(section)
        if limits_database_required(
            _old_target_limits_posture(section, target), path_configured=path_configured
        )
    ]


def _old_incomplete_limits_blocks(section: object) -> list[str]:
    if not isinstance(section, dict):
        return []
    connector = section.get("connector")
    blocks: list[tuple[str | None, object, bool]] = []
    if LIMITS_CHECKING_LEAF in section:
        blocks.append((None, section[LIMITS_CHECKING_LEAF], False))
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
        for leaf, written in _old__unstated_limits_leaves(block, whole=whole):
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


def _old__global_limits_posture(section: object) -> LimitsPosture:
    if not isinstance(section, dict) or LIMITS_CHECKING_LEAF not in section:
        return _old__block_limits_posture({}, None, whole=False)
    block = section[LIMITS_CHECKING_LEAF]
    if not isinstance(block, dict):
        return LimitsPosture(enabled=None, mode=None, connector_type=None, incomplete=LIMITS_LEAVES)
    return _old__block_limits_posture(block, None, whole=False)


def _old__unstated_limits_leaves(
    block: Mapping[str, object], *, whole: bool
) -> list[tuple[str, bool]]:
    unstated: list[tuple[str, bool]] = []
    for leaf in LIMITS_LEAVES:
        if leaf not in block:
            if whole:
                unstated.append((leaf, False))
        elif _old__LEAF_READERS[leaf](block[leaf]) is None:
            unstated.append((leaf, True))
    return unstated


def _old__block_limits_posture(
    block: Mapping[str, object], connector_type: str | None, *, whole: bool
) -> LimitsPosture:
    unstated = _old__unstated_limits_leaves(block, whole=whole)
    if unstated:
        return LimitsPosture(
            enabled=None,
            mode=None,
            connector_type=connector_type,
            incomplete=tuple(leaf for leaf, _ in unstated),
        )
    return LimitsPosture(
        enabled=_old__limits_leaf(block.get(_ENABLED_LEAF)),
        mode=_old__limits_mode(block.get(_MODE_LEAF)),
        connector_type=connector_type,
    )


def _old__limits_leaf(value: object) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    return None


def _old__limits_mode(value: object) -> LimitsMode | None:
    if value == LIMITS_MODE_EXCLUSIVE:
        return LIMITS_MODE_EXCLUSIVE
    if value == LIMITS_MODE_OPTIONAL:
        return LIMITS_MODE_OPTIONAL
    return None


_old__LEAF_READERS: dict[str, Callable[[object], bool | str | None]] = {
    _ENABLED_LEAF: _old__limits_leaf,
    _MODE_LEAF: _old__limits_mode,
}


# -- write_switch.py at a4d8f67c62 ------------------------------------------------


def _old__tier_word(value: object) -> TierWord | None:
    if value == "never":
        return "never"
    if value == "asked":
        return "asked"
    if value == "allowed":
        return "allowed"
    return None


def _old_owner_tier_term(section: object, owner: str | NoOwner | None) -> OwnerTierTerm:
    try:
        if not (isinstance(section, dict) and _LOGIN_TIERS_LEAF in section):
            return "unknown" if _control_state.tree_is_bound() else "skip"
        login_tiers = section[_LOGIN_TIERS_LEAF]
        if not isinstance(login_tiers, dict) or not all(
            isinstance(login, str) and login and isinstance(tier, str) and tier in _LOGIN_TIERS
            for login, tier in login_tiers.items()
        ):
            return "unknown"
        if not isinstance(owner, str) or not owner:
            return "unknown"
        tier = login_tiers.get(owner)
    except Exception:  # a rung that cannot read the map refuses; it never raises
        return "unknown"
    return _old__tier_word(tier) or "unknown"


def _old_write_ceiling(
    section: object, owner: str | NoOwner | None, *, tree_bound: bool | None = None
) -> WriteCeiling:
    if not (_control_state.tree_is_bound() if tree_bound is None else tree_bound):
        return _old_tier_writes(section)
    rung = _old_owner_tier_term(section, owner)
    return "unknown" if rung == "skip" else rung


def _old_write_verdict(
    section: object,
    target: str | None,
    *,
    owner: str | NoOwner | None = None,
    ignore_launch_pin: bool = False,
) -> WriteVerdict:
    return _old__write_verdict(section, target, owner, ignore_launch_pin, None)


def _old__write_verdict(
    section: object,
    target: str | None,
    owner: str | NoOwner | None,
    ignore_launch_pin: bool,
    seen: _RecordSeen | None,
) -> WriteVerdict:
    named = target if isinstance(target, str) else None
    kind = _old__verdict_kind(section, target)
    judged: VerdictReason | _Armed
    try:
        judged = _old__write_refusal(section, target, owner, ignore_launch_pin, seen)
    except Exception:  # a verdict that cannot be reached refuses; it never raises
        logger.warning("Write verdict for %r could not be reached; refusing", target, exc_info=True)
        judged = VerdictReason.RECORD_UNAVAILABLE
        if seen is not None:
            seen.read = False
    if isinstance(judged, _Armed):
        return Allowed(target=named, kind=kind, position=judged.position)
    return Refused(reason=judged, target=named, kind=kind)


def _old__verdict_kind(section: object, target: object) -> MachineKind | None:
    if not is_control_target(target):
        return None
    try:
        if _old_machine_kind(section, target) is None:
            return None
        return control_context.published_machine_kind(target)
    except Exception:  # the kind is a word in a refusal; it must not decide one
        logger.debug("Could not name the machine kind of %r", target, exc_info=True)
        return None


def _old_backstop_verdict(section: object) -> WriteVerdict:
    if is_readonly_run():
        return Refused(reason=VerdictReason.READONLY_RUN, target=None, kind=None)
    try:
        allows = _old_tier_allows(section)
    except Exception:  # _old_tier_allows never raises by contract; refuse if it does
        logger.warning("Backstop tier could not be read; refusing", exc_info=True)
        allows = False
    if not allows:
        return Refused(reason=VerdictReason.TIER_NEVER, target=None, kind=None)
    return Allowed(target=None, kind=None, position=None)


def _old_execution_verdict(
    section: object,
    target: str | None,
    *,
    owner: str | NoOwner | None = None,
    ignore_launch_pin: bool = False,
    session_keys: Sequence[str] | None = None,
) -> WriteVerdict:
    verdict = _old_write_verdict(section, target, owner=owner, ignore_launch_pin=ignore_launch_pin)
    return _old__with_hold(section, verdict, session_keys)


def _old__with_hold(
    section: object, verdict: WriteVerdict, session_keys: Sequence[str] | None
) -> WriteVerdict:
    if not _old_backstop_verdict(section).allowed:
        return verdict
    held = session_hold_refusal(session_keys=session_keys)
    if held is None:
        return verdict
    return evolve(held, target=verdict.target, kind=verdict.kind)


def _old__write_refusal(
    section: object,
    target: str | None,
    owner: str | NoOwner | None,
    ignore_launch_pin: bool,
    seen: _RecordSeen | None = None,
) -> VerdictReason | _Armed:
    if is_readonly_run():
        return VerdictReason.READONLY_RUN
    if not _old_tier_allows(section):
        return VerdictReason.TIER_NEVER
    if not is_control_target(target):  # _old_resolve_target refuses it too; said here for the type
        return VerdictReason.TARGET_UNKNOWN
    try:
        _old_resolve_target(section, target)
    except Exception:  # ValueError by contract; anything else is no resolution either
        return VerdictReason.TARGET_UNKNOWN

    resolved = current_owner(owner)
    if resolved is NO_OWNER and _control_state.is_dispatch_run():
        return VerdictReason.OWNER_UNSCHEDULED
    if resolved is NO_OWNER or not isinstance(resolved, str) or not resolved.strip():
        return VerdictReason.OWNER_UNKNOWN
    # A host holds one login's switch: an owner other than that login is refused
    # before its tier or any record is read (a tree bind reads each owner's own).
    if not _control_state.tree_is_bound() and resolved.strip() != acting_identity():
        return VerdictReason.OWNER_FOREIGN

    rung = _old_owner_tier_term(section, resolved)
    if rung == "unknown":
        return VerdictReason.OWNER_UNKNOWN
    if rung == "never":
        return VerdictReason.TIER_NEVER

    if _control_state.tree_is_bound():
        tree = _control_state.control_context_tree()
        if tree is None:
            logger.warning(
                "Write record unreadable for %r: %s", resolved, _control_state.tree_bind_unusable()
            )
            return VerdictReason.RECORD_UNAVAILABLE
        read = control_context.read_tree_context(tree, resolved)
        if read.unavailable is not None:
            logger.warning("Write record unreadable for %r: %s", resolved, read.unavailable)
            return VerdictReason.RECORD_UNAVAILABLE
        record = read.record
    else:
        own = control_context.read_record_status()
        record, status = own.value, own.status
        if status == "unreadable":
            logger.warning("Write record unreadable on this host; refusing")
            return VerdictReason.RECORD_UNAVAILABLE
        if status != "ok":
            record = None
    if seen is not None:
        seen.read, seen.record = True, record

    stored = control_context.switch_position(record, target)
    if stored == "off":
        return VerdictReason.SWITCH_OFF

    if not ignore_launch_pin and _control_state.launch_pin_term(target) != "none":
        return VerdictReason.LAUNCH_PINNED
    ceiling = _old_write_ceiling(section, resolved)
    return _Armed(cast("ArmedPosition", min(stored, ceiling, key=_LOGIN_TIERS.index)))
