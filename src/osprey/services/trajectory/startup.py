"""What the trajectory service checks and builds before it serves anything.

Startup either hands back one :class:`TrajectoryStartup` or refuses with a
:class:`StartupRefused`, a :class:`SystemExit` whose message names what an
operator has to fix: the config key, the environment variable, the file, the
model or the slot. Nothing it checks is retried later; everything it refuses is
a deployment that would serve the wrong machine, write unchecked, or read an
owner's record nobody else reads.

In order:

1. **The target.** ``services.trajectory.target`` names one of
   :data:`~osprey_connectors.types.CONTROL_TARGETS`; absent, it is the
   deployment's baseline (:func:`~osprey_connectors.types.baseline_target`).
   ``services.trajectory.model`` optionally names the model whose slots the
   service commands. The service's tunable keys
   (:mod:`~osprey.services.trajectory.constants`) are read beside them, each
   refused naming its key when it is not a value of its kind.
2. **The shared directories.** ``OSPREY_CONTROL_CONTEXT_TREE`` must be set, so
   an owner's record is always read from the tree every holder reads, and
   ``OSPREY_GUARDED_RUN_DIR`` must be set and be the parent of
   :func:`~osprey.runtime.guarded_run.guarded_run_dir` for the target, so the
   lock this service takes is the one every other writer of the target takes.
3. **The limits file.** When the target arms writes and requires a limits
   database (:func:`~osprey_connectors.types.limits_database_required`: limits
   checking on, and a mode other than ``optional`` or a path configured) — the
   per-target pairing
   :func:`~osprey_connectors.types.any_armed_target_requires_limits_database`
   folds and :func:`~osprey.deployment.compose_generator.resolve_limits_mount`
   mounts the file on — the configured limits database must exist. ``optional``
   with no path is legal: every channel is unlisted and written with no limits.
   :meth:`LimitsValidator.from_config` never raises, so this is the one place
   its absence turns into a refusal rather than a fail-safe validator.
4. **The command set.** ``trajectory_slots`` of the build's
   ``facility_facts.json``: no model → the service starts with its writing
   actions unavailable; one model, or the one ``services.trajectory.model``
   names → its ``arm``, ``armed`` and ``injection_request`` addresses, and
   its ``tags``, each facility signal to its address on every BPM; several
   and none named → refused, naming them.
5. **The raw-put block**, armed once per process with the executor's armed
   contract, so a client put no connector let through is refused here in the
   words it is refused everywhere else.
6. **The sequence module directory**:
   ``OSPREY_TRAJECTORY_SEQUENCE_DIR`` when set (a container, where the compose
   render binds the directory read-only), else
   ``services.trajectory.sequence_dir`` against the project root (a host
   run), else none; its sequence module is loaded and validated
   (:func:`~osprey.services.trajectory.sequence.load_sequence`). A directory
   holding no module leaves the service without one; a module that fails to
   load or validate is refused. With a trajectory model, a module slot
   tagging a signal absent from the model's ``tags`` is refused, naming the
   slot and the signal; with none, the module is not checked against tags.
7. **One connector** for the target, built with ``fresh_reads`` on — every
   read asks the machine, so a shot never reads a buffer a cache kept — and
   connected. ``connect()`` latches nothing about reachability; that is asked
   live, per call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.services.trajectory.constants import (
    ACTION_TIMEOUT_KEY,
    ARMED_TIMEOUT_KEY,
    CONTINUOUS_MAX_SHOTS_KEY,
    DEFAULT_ACTION_TIMEOUT_S,
    DEFAULT_ARMED_TIMEOUT_S,
    DEFAULT_CONTINUOUS_MAX_SHOTS,
    DEFAULT_LOSS_SUM_CUTOFF,
    DEFAULT_RAW_WINDOW_TURNS,
    DEFAULT_SHOT_SUM_THRESHOLD,
    LOSS_SUM_CUTOFF_KEY,
    RAW_WINDOW_TURNS_KEY,
    SEQUENCE_DIR_ENV_VAR,
    SEQUENCE_DIR_KEY,
    SERVICE_KEY,
    SHOT_SUM_THRESHOLD_KEY,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GUARDED_RUN_DIR_ENV_VAR",
    "MODEL_KEY",
    "SERVICE_KEY",
    "TARGET_KEY",
    "StartupRefused",
    "TrajectoryCommands",
    "TrajectorySettings",
    "TrajectoryStartup",
    "arm_raw_put_block",
    "check_limits_file",
    "check_shared_dirs",
    "resolve_sequence_dir",
    "select_commands",
    "select_tags",
    "service_settings",
    "start",
]

#: The control target the service drives.
TARGET_KEY = f"{SERVICE_KEY}.target"
#: The model whose trajectory slots the service commands, when the build has several.
MODEL_KEY = f"{SERVICE_KEY}.model"

#: Where the deployment mounts every target's guarded-run directory.
GUARDED_RUN_DIR_ENV_VAR = "OSPREY_GUARDED_RUN_DIR"


class StartupRefused(SystemExit):
    """The service refuses to start; the message names what to fix.

    A :class:`SystemExit` carrying a string, so an uncaught refusal ends the
    process with a non-zero status and prints the message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"trajectory service refused to start: {message}")

    def __str__(self) -> str:
        return str(self.code)


@dataclass(frozen=True)
class TrajectoryCommands:
    """The addresses one model's trajectory measurement commands and watches.

    Attributes:
        model: The model's name.
        arm: The arm command of each monitor, in the facts' order.
        injection_request: The injection request command.
        injection_busy: The flag that reads while an injection is under way.
        armed: The armed flag of each monitor, in the facts' order: 1 once
            its arm took, 0 again once its buffer filled. Watched, never
            written.
    """

    model: str
    arm: tuple[str, ...]
    injection_request: str
    injection_busy: str
    armed: tuple[str, ...] = ()

    @property
    def commands(self) -> frozenset[str]:
        """Every address the service may write: the arm commands and the request."""
        return frozenset((*self.arm, self.injection_request))


@dataclass(frozen=True)
class TrajectorySettings:
    """The service's tunable keys, as read; units in :mod:`.constants`.

    Attributes:
        loss_sum_cutoff: ``services.trajectory.loss_sum_cutoff``.
        shot_sum_threshold: ``services.trajectory.shot_sum_threshold``.
        armed_timeout_s: ``services.trajectory.armed_timeout_s``.
        raw_window_turns: ``services.trajectory.raw_window_turns``.
        continuous_max_shots: ``services.trajectory.continuous_max_shots``.
        sequence_dir: ``services.trajectory.sequence_dir`` as written, or
            ``None``; :func:`resolve_sequence_dir` turns it into a directory.
        action_timeout_s: ``services.trajectory.action_timeout_s``.
    """

    loss_sum_cutoff: float = DEFAULT_LOSS_SUM_CUTOFF
    shot_sum_threshold: float = DEFAULT_SHOT_SUM_THRESHOLD
    armed_timeout_s: float = DEFAULT_ARMED_TIMEOUT_S
    raw_window_turns: int = DEFAULT_RAW_WINDOW_TURNS
    continuous_max_shots: int = DEFAULT_CONTINUOUS_MAX_SHOTS
    sequence_dir: str | None = None
    action_timeout_s: float = DEFAULT_ACTION_TIMEOUT_S


@dataclass
class TrajectoryStartup:
    """What a started service holds.

    Attributes:
        target: The control target the service drives.
        connector: The one connector, built for ``target`` and connected.
        commands: The selected model's slots, or ``None`` when the build has no
            trajectory model and the writing actions are unavailable.
        guarded_run_dir: The target's guarded-run directory.
        settings: The service's tunable keys.
        tag_addresses: Facility signal to its addresses, one per BPM in BPM
            order, from the selected model's ``tags``; empty when the build
            has no trajectory model.
        sequence_dir: The directory the sequence module is loaded from, or
            ``None`` when the deployment names none.
        sequence: The loaded sequence module, once its builder filled the
            slot; ``None`` while there is none.
        tool: The correction tool
            (:class:`~osprey.services.trajectory.step.TrajectoryTool`), or why
            the service has none
            (:class:`~osprey.services.trajectory.step.ToolUnavailable`); ``None``
            until its builder filled the slot.
    """

    target: str
    connector: Any
    commands: TrajectoryCommands | None
    guarded_run_dir: Path
    settings: TrajectorySettings = field(default_factory=TrajectorySettings)
    tag_addresses: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    sequence_dir: Path | None = None
    sequence: Any = None
    tool: Any = None

    @property
    def writing_available(self) -> bool:
        """Whether the build gave the service anything to command."""
        return self.commands is not None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(block: Mapping[str, Any], key: str, default: float, *, positive: bool) -> float:
    """``block[leaf of key]`` as a finite number; refused naming ``key`` otherwise."""
    from osprey.utils.seconds import non_negative_seconds, positive_seconds

    value = block.get(key.rsplit(".", 1)[1])
    if value is None:
        return default
    number = positive_seconds(value) if positive else non_negative_seconds(value)
    if number is None:
        kind = "greater than zero" if positive else "zero or more"
        raise StartupRefused(f"{key} must be a finite number {kind}, not {value!r}")
    return number


def _count(block: Mapping[str, Any], key: str, default: int) -> int:
    """``block[leaf of key]`` as a whole number above zero; refused naming ``key`` otherwise."""
    value = block.get(key.rsplit(".", 1)[1])
    if value is None:
        return default
    count: int | None = None
    if isinstance(value, int) and not isinstance(value, bool):
        count = value
    elif isinstance(value, str) and value.strip().isdigit():
        count = int(value.strip())
    if count is None or count < 1:
        raise StartupRefused(f"{key} must be a whole number greater than zero, not {value!r}")
    return count


def _settings(block: Mapping[str, Any]) -> TrajectorySettings:
    sequence_dir = block.get(SEQUENCE_DIR_KEY.rsplit(".", 1)[1])
    if sequence_dir is not None and (not isinstance(sequence_dir, str) or not sequence_dir.strip()):
        raise StartupRefused(f"{SEQUENCE_DIR_KEY} must name a directory, not {sequence_dir!r}")
    return TrajectorySettings(
        loss_sum_cutoff=_number(
            block, LOSS_SUM_CUTOFF_KEY, DEFAULT_LOSS_SUM_CUTOFF, positive=False
        ),
        shot_sum_threshold=_number(
            block, SHOT_SUM_THRESHOLD_KEY, DEFAULT_SHOT_SUM_THRESHOLD, positive=False
        ),
        armed_timeout_s=_number(block, ARMED_TIMEOUT_KEY, DEFAULT_ARMED_TIMEOUT_S, positive=True),
        raw_window_turns=_count(block, RAW_WINDOW_TURNS_KEY, DEFAULT_RAW_WINDOW_TURNS),
        continuous_max_shots=_count(block, CONTINUOUS_MAX_SHOTS_KEY, DEFAULT_CONTINUOUS_MAX_SHOTS),
        sequence_dir=sequence_dir,
        action_timeout_s=_number(
            block, ACTION_TIMEOUT_KEY, DEFAULT_ACTION_TIMEOUT_S, positive=True
        ),
    )


def service_settings(
    config: Mapping[str, Any],
) -> tuple[str, str | None, TrajectorySettings]:
    """The target, the optional model and the tunable keys ``services.trajectory`` names.

    Args:
        config: The loaded project config.

    Returns:
        ``(target, model, settings)``; the target defaults to the deployment's
        baseline, every tunable key to its default in :mod:`.constants`.

    Raises:
        StartupRefused: The block, the target, the model or a tunable key is
            not usable.
    """
    from osprey_connectors.types import CONTROL_TARGETS, baseline_target

    block = _mapping(config.get("services")).get("trajectory")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise StartupRefused(f"{SERVICE_KEY} must be a mapping, not {block!r}")
    target = block.get("target")
    if target is None:
        target = baseline_target(_mapping(config.get("control_system")))
    if target not in CONTROL_TARGETS:
        raise StartupRefused(
            f"{TARGET_KEY} is {target!r}; it must be one of {', '.join(CONTROL_TARGETS)}"
        )
    model = block.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise StartupRefused(f"{MODEL_KEY} must name a model, not {model!r}")
    return str(target), model, _settings(block)


def resolve_sequence_dir(settings: TrajectorySettings, project_root: Path) -> Path | None:
    """The directory the sequence module is loaded from, or ``None``.

    The environment's :data:`~.constants.SEQUENCE_DIR_ENV_VAR` wins: the
    compose render sets it beside the read-only bind of
    ``services.trajectory.sequence_dir``, whose host path means nothing inside
    the container. A host run reads the key itself, a relative path against
    the project root. Whether the directory holds a module is the loader's
    question, not this one.

    Args:
        settings: The service's tunable keys.
        project_root: The deployment's project root.

    Returns:
        The resolved directory, or ``None`` when neither names one.
    """
    mounted = os.environ.get(SEQUENCE_DIR_ENV_VAR, "").strip()
    if mounted:
        return Path(mounted)
    if settings.sequence_dir is None:
        return None
    path = Path(settings.sequence_dir).expanduser()
    return path if path.is_absolute() else project_root / path


def check_shared_dirs(target: str) -> Path:
    """Refuse unless the control-context tree and the guarded-run directory are shared.

    Args:
        target: The service's control target.

    Returns:
        The target's guarded-run directory.

    Raises:
        StartupRefused: A variable is unset, the target's guarded-run directory
            cannot be resolved, or ``OSPREY_GUARDED_RUN_DIR`` is not its parent.
    """
    from osprey.runtime.guarded_run import GuardedRunDirError, guarded_run_dir
    from osprey_connectors.posture_store import CONTROL_CONTEXT_TREE_ENV_VAR

    if not os.environ.get(CONTROL_CONTEXT_TREE_ENV_VAR, "").strip():
        raise StartupRefused(
            f"{CONTROL_CONTEXT_TREE_ENV_VAR} is unset; the service reads every owner's "
            "control-context record from that tree"
        )
    mounted = os.environ.get(GUARDED_RUN_DIR_ENV_VAR, "").strip()
    if not mounted:
        raise StartupRefused(
            f"{GUARDED_RUN_DIR_ENV_VAR} is unset; the service locks the target in that "
            "directory, beside every other writer"
        )
    try:
        directory = guarded_run_dir(target)
    except GuardedRunDirError as exc:
        raise StartupRefused(
            f"{GUARDED_RUN_DIR_ENV_VAR} is {mounted}, but the guarded-run directory of "
            f"target {target!r} cannot be used: {exc}"
        ) from exc
    if Path(mounted).resolve() != directory.parent.resolve():
        raise StartupRefused(
            f"{GUARDED_RUN_DIR_ENV_VAR} is {mounted}, which is not the parent of "
            f"{directory}, the guarded-run directory of target {target!r}"
        )
    return directory


def check_limits_file(section: Mapping[str, Any], target: str) -> None:
    """Refuse when the target arms writes, requires a limits database, and has none.

    The requirement is :func:`~osprey_connectors.types.limits_database_required`,
    asked of the live config's path: ``optional`` with no path starts, and a
    configured path must name a file whatever the mode.

    Args:
        section: The ``control_system:`` config section.
        target: The service's control target.

    Raises:
        StartupRefused: The pairing holds and the configured file is not
            configured or not there.
    """
    from osprey_connectors.control_system.limits_validator import (
        LIMITS_DATABASE_CONFIG_KEY,
        LimitsValidator,
    )
    from osprey_connectors.types import (
        limits_database_required,
        target_limits_posture,
        target_writes_enabled,
    )

    if not target_writes_enabled(section, target):
        return
    configured = LimitsValidator.configured_database_path()
    if not limits_database_required(
        target_limits_posture(section, target), path_configured=configured is not None
    ):
        return
    if configured is None:
        raise StartupRefused(
            f"target {target!r} arms writes and checks limits, but "
            f"{LIMITS_DATABASE_CONFIG_KEY} names no limits file"
        )
    if not Path(configured).is_file():
        raise StartupRefused(
            f"target {target!r} arms writes and checks limits, but the limits file "
            f"{configured} ({LIMITS_DATABASE_CONFIG_KEY}) is missing"
        )


def _commands_of(model: str, slots: Any) -> TrajectoryCommands:
    """The selected model's slots, refused when one the service needs is empty.

    ``arm`` and ``armed`` are lists of addresses, one per monitor and of equal
    length; ``injection_request`` and the flag reachability reads,
    ``injection_busy``, are one address each.
    """
    slots = _mapping(slots)
    lists: dict[str, list[str]] = {}
    for slot in ("arm", "armed"):
        value = slots.get(slot)
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(a, str) and a for a in value)
        ):
            raise StartupRefused(
                f"model {model!r} states no address for the trajectory slot {slot!r}"
            )
        lists[slot] = value
    if len(lists["arm"]) != len(lists["armed"]):
        raise StartupRefused(
            f"model {model!r} states {len(lists['arm'])} 'arm' addresses but "
            f"{len(lists['armed'])} 'armed' addresses; each monitor has one of each"
        )
    for slot in ("injection_request", "injection_busy"):
        value = slots.get(slot)
        if not isinstance(value, str) or not value:
            raise StartupRefused(
                f"model {model!r} states no address for the trajectory slot {slot!r}"
            )
    return TrajectoryCommands(
        model=model,
        arm=tuple(lists["arm"]),
        injection_request=slots["injection_request"],
        injection_busy=slots["injection_busy"],
        armed=tuple(lists["armed"]),
    )


def select_commands(facts: Mapping[str, Any], model: str | None) -> TrajectoryCommands | None:
    """The command set ``trajectory_slots`` gives the service.

    Args:
        facts: The build's ``facility_facts.json``.
        model: The model ``services.trajectory.model`` names, or ``None``.

    Returns:
        The selected model's slots, or ``None`` when the build has no
        trajectory model and none is named.

    Raises:
        StartupRefused: The named model has no slots, several models have
            slots and none is named, or the selected model leaves a slot empty.
    """
    by_model = _mapping(facts.get("trajectory_slots"))
    names = sorted(str(name) for name in by_model)
    if model is not None:
        if model not in by_model:
            have = ", ".join(names) or "none"
            raise StartupRefused(
                f"{MODEL_KEY} is {model!r}, which has no trajectory slots (models with "
                f"slots: {have})"
            )
        return _commands_of(model, by_model[model])
    if not names:
        return None
    if len(names) > 1:
        raise StartupRefused(
            f"models {', '.join(names)} all have trajectory slots; name one in {MODEL_KEY}"
        )
    return _commands_of(names[0], by_model[names[0]])


def select_tags(
    facts: Mapping[str, Any], commands: TrajectoryCommands | None
) -> dict[str, tuple[str, ...]]:
    """The selected model's ``tags``: each facility signal to its addresses, one per BPM.

    Args:
        facts: The build's ``facility_facts.json``.
        commands: The command set :func:`select_commands` selected, or ``None``.

    Returns:
        Signal to addresses in BPM order; empty when there is no command set or
        the model states no ``tags``.

    Raises:
        StartupRefused: The model's ``tags`` is not a mapping of signal to a
            list of addresses.
    """
    if commands is None:
        return {}
    model = commands.model
    tags = _mapping(_mapping(facts.get("trajectory_slots")).get(model)).get("tags")
    if tags is None:
        return {}
    if not isinstance(tags, Mapping):
        raise StartupRefused(
            f"model {model!r} states 'tags' as {tags!r}, not a mapping of signal to addresses"
        )
    selected: dict[str, tuple[str, ...]] = {}
    for signal, addresses in tags.items():
        if not isinstance(addresses, list) or not all(isinstance(a, str) and a for a in addresses):
            raise StartupRefused(
                f"model {model!r} states 'tags' signal {signal!r} as {addresses!r}, "
                "not a list of addresses"
            )
        selected[str(signal)] = tuple(addresses)
    return selected


_RAW_PUT_ARMED = False


def _install_raw_put_block() -> None:
    from osprey.runtime import raw_put_block
    from osprey.services.python_executor.execution.wrapper import armed_contract

    raw_put_block.install("armed", **armed_contract(refuse_rpc=False))


def arm_raw_put_block() -> None:
    """Refuse every raw client put in this process that no connector let through.

    The executor's armed contract, as the notebook kernel arms it: rpc stays
    allowed, since nothing here bounds it. Armed once per process; a second
    call does nothing.
    """
    global _RAW_PUT_ARMED
    if _RAW_PUT_ARMED:
        return
    _install_raw_put_block()
    _RAW_PUT_ARMED = True


def _facts(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The build's agent facts, read beside the rendered config of the repo."""
    from osprey.facility.views.facts import read_facts
    from osprey_connectors.workspace import rendered_config_path, resolve_project_root

    render = rendered_config_path(resolve_project_root(config)).parent
    return read_facts(render, str(config.get("project_name") or "osprey"))


async def _connect(section: Mapping[str, Any], target: str) -> Any:
    """One connector for ``target``: the section with its type re-pointed, connected.

    The type's own block gets ``fresh_reads`` on, for this connector only: a
    shot reads waveforms the machine just filled, and a monitor cache would
    answer the previous shot's.
    """
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey_connectors.types import resolve_target

    config = dict(section)
    try:
        connector_type = resolve_target(section, target)
    except ValueError as exc:
        raise StartupRefused(f"{TARGET_KEY} is {target!r}: {exc}") from exc
    config["type"] = connector_type
    blocks = dict(_mapping(section.get("connector")))
    blocks[connector_type] = {**_mapping(blocks.get(connector_type)), "fresh_reads": True}
    config["connector"] = blocks
    register_builtin_connectors()
    return await ConnectorFactory.create_control_system_connector(config, control_target=target)


async def start() -> TrajectoryStartup:
    """Check the deployment and build what the service serves from.

    Returns:
        The started service's target, connector, command set, lock directory,
        tunable keys, tag map, sequence module directory, sequence module and
        correction tool (or why there is none).

    Raises:
        StartupRefused: Any check in the module's list fails.
    """
    from osprey.services.trajectory.sequence import check_tags, load_sequence
    from osprey.services.trajectory.step import build_service_tool
    from osprey_connectors.workspace import load_osprey_config, resolve_project_root

    config = load_osprey_config()
    section = _mapping(config.get("control_system"))
    target, model, settings = service_settings(config)
    directory = check_shared_dirs(target)
    check_limits_file(section, target)
    facts = _facts(config)
    commands = select_commands(facts, model)
    tag_addresses = select_tags(facts, commands)
    if commands is None:
        logger.warning(
            "trajectory service: the build has no trajectory model; writing actions are unavailable"
        )
    sequence_dir = resolve_sequence_dir(settings, resolve_project_root(config))
    sequence = load_sequence(sequence_dir, commands)
    if commands is not None and sequence is not None:
        check_tags(sequence, tag_addresses)
    arm_raw_put_block()
    connector = await _connect(section, target)
    logger.info(
        "trajectory service: target %s, model %s",
        target,
        commands.model if commands else "none",
    )
    return TrajectoryStartup(
        target=target,
        connector=connector,
        commands=commands,
        guarded_run_dir=directory,
        settings=settings,
        tag_addresses=tag_addresses,
        sequence_dir=sequence_dir,
        sequence=sequence,
        tool=build_service_tool(config, commands.model if commands else None),
    )
