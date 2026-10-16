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
   service commands.
2. **The shared directories.** ``OSPREY_CONTROL_CONTEXT_TREE`` must be set, so
   an owner's record is always read from the tree every holder reads, and
   ``OSPREY_GUARDED_RUN_DIR`` must be set and be the parent of
   :func:`~osprey.runtime.guarded_run.guarded_run_dir` for the target, so the
   lock this service takes is the one every other writer of the target takes.
3. **The limits file.** When the target arms writes and checks limits — the
   per-target pairing
   :func:`~osprey_connectors.types.any_armed_target_checks_limits` folds and
   :func:`~osprey.deployment.compose_generator.resolve_limits_mount` mounts the
   file on — the configured limits database must exist.
   :meth:`LimitsValidator.from_config` never raises, so this is the one place
   its absence turns into a refusal rather than a fail-safe validator.
4. **The command set.** ``trajectory_slots`` of the build's
   ``facility_facts.json``: no model → the service starts with its writing
   actions unavailable; one model, or the one ``services.trajectory.model``
   names → its ``arm`` and ``injection_request`` addresses; several and none
   named → refused, naming them.
5. **The raw-put block**, armed once per process with the executor's armed
   contract, so a client put no connector let through is refused here in the
   words it is refused everywhere else.
6. **One connector** for the target, built and connected. ``connect()`` latches
   nothing about reachability; that is asked live, per call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GUARDED_RUN_DIR_ENV_VAR",
    "MODEL_KEY",
    "SERVICE_KEY",
    "TARGET_KEY",
    "StartupRefused",
    "TrajectoryCommands",
    "TrajectoryStartup",
    "arm_raw_put_block",
    "check_limits_file",
    "check_shared_dirs",
    "select_commands",
    "service_settings",
    "start",
]

#: The service's block in the rendered ``config.yml``.
SERVICE_KEY = "services.trajectory"
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
    """

    model: str
    arm: tuple[str, ...]
    injection_request: str
    injection_busy: str

    @property
    def commands(self) -> frozenset[str]:
        """Every address the service may write: the arm commands and the request."""
        return frozenset((*self.arm, self.injection_request))


@dataclass
class TrajectoryStartup:
    """What a started service holds.

    Attributes:
        target: The control target the service drives.
        connector: The one connector, built for ``target`` and connected.
        commands: The selected model's slots, or ``None`` when the build has no
            trajectory model and the writing actions are unavailable.
        guarded_run_dir: The target's guarded-run directory.
    """

    target: str
    connector: Any
    commands: TrajectoryCommands | None
    guarded_run_dir: Path

    @property
    def writing_available(self) -> bool:
        """Whether the build gave the service anything to command."""
        return self.commands is not None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def service_settings(config: Mapping[str, Any]) -> tuple[str, str | None]:
    """The target and the optional model ``services.trajectory`` names.

    Args:
        config: The loaded project config.

    Returns:
        ``(target, model)``; the target defaults to the deployment's baseline.

    Raises:
        StartupRefused: The block, the target or the model is not usable.
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
    return str(target), model


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
    """Refuse when the target arms writes, checks limits, and has no limits file.

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
    from osprey_connectors.types import target_limits_posture, target_writes_enabled

    if not target_writes_enabled(section, target):
        return
    if target_limits_posture(section, target).enabled is not True:
        return
    configured = LimitsValidator.configured_database_path()
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

    ``arm`` is a list of addresses; ``injection_request`` and the flag
    reachability reads, ``injection_busy``, are one address each.
    """
    slots = _mapping(slots)
    arm = slots.get("arm")
    if not isinstance(arm, list) or not arm or not all(isinstance(a, str) and a for a in arm):
        raise StartupRefused(f"model {model!r} states no address for the trajectory slot 'arm'")
    for slot in ("injection_request", "injection_busy"):
        value = slots.get(slot)
        if not isinstance(value, str) or not value:
            raise StartupRefused(
                f"model {model!r} states no address for the trajectory slot {slot!r}"
            )
    return TrajectoryCommands(
        model=model,
        arm=tuple(arm),
        injection_request=slots["injection_request"],
        injection_busy=slots["injection_busy"],
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
    """One connector for ``target``: the section with its type re-pointed, connected."""
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors
    from osprey_connectors.types import resolve_target

    config = dict(section)
    try:
        config["type"] = resolve_target(section, target)
    except ValueError as exc:
        raise StartupRefused(f"{TARGET_KEY} is {target!r}: {exc}") from exc
    register_builtin_connectors()
    return await ConnectorFactory.create_control_system_connector(config, control_target=target)


async def start() -> TrajectoryStartup:
    """Check the deployment and build what the service serves from.

    Returns:
        The started service's target, connector, command set and lock directory.

    Raises:
        StartupRefused: Any check in the module's list fails.
    """
    from osprey_connectors.workspace import load_osprey_config

    config = load_osprey_config()
    section = _mapping(config.get("control_system"))
    target, model = service_settings(config)
    directory = check_shared_dirs(target)
    check_limits_file(section, target)
    commands = select_commands(_facts(config), model)
    if commands is None:
        logger.warning(
            "trajectory service: the build has no trajectory model; writing actions are unavailable"
        )
    arm_raw_put_block()
    connector = await _connect(section, target)
    logger.info(
        "trajectory service: target %s, model %s",
        target,
        commands.model if commands else "none",
    )
    return TrajectoryStartup(
        target=target, connector=connector, commands=commands, guarded_run_dir=directory
    )
