"""Read the ``pv_directory:`` config section once, for every layer that needs it.

:func:`resolve_section` checks the section and :func:`predicates` turns its
answer into the context keys the render, the build validator, the MCP server,
the panel and health all use; none of them computes these keys itself.

A malformed section raises :class:`SectionError` naming the key, which the build
turns into a refusal. A section that is valid but cannot serve everything it
names resolves with :attr:`ResolvedSection.warnings`: no ``type`` on a build
without a simulator, a Kafka feed without its client, connectors without PV
directory support. ``None`` means no section: absent, or set to ``null``.

The connectors are reached through the shim inside functions only, so this
module imports on every connectors release.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.services.pv_directory.demo import DEMO_PREFIX, resolve_replay_file

__all__ = [
    "ALARM_REPLAY",
    "INTERNAL_TYPES",
    "KAFKA_ALARM_STATE",
    "KAFKA_CLIENT_MISSING",
    "NO_SOURCE_WARNING",
    "PREDICATES",
    "SIMULATOR_VIEW",
    "ResolvedSection",
    "SectionError",
    "predicates",
    "resolve_section",
]

SECTION_KEY = "pv_directory"

#: Internal builtin types: the connectors pick them on ``va``; a config never names them.
SIMULATOR_VIEW = "simulator_view"
ALARM_REPLAY = "alarm_replay"
INTERNAL_TYPES = (SIMULATOR_VIEW, ALARM_REPLAY)

#: The alarm-state type that reads the alarm server's Kafka topic.
KAFKA_ALARM_STATE = "phoebus_alarm_server"

KAFKA_CLIENT_MISSING = (
    "Kafka client not installed "
    "(install osprey-framework[alarm-state]; for an image, through OSPREY_PIP_SPEC)"
)

NO_SOURCE_WARNING = (
    "pv_directory has no type and the build has no simulator: "
    "every section answers 'not available for this target'"
)

#: The context keys, in the order every consumer lists them.
PREDICATES = (
    "pv_directory_configured",
    "pv_directory_alarm_log",
    "pv_directory_alarm_state",
    "pv_directory_alarm_config",
    "pv_directory_put_log",
)

_SUB_BLOCKS = ("alarm_log", "alarm_state", "put_log")


class SectionError(ValueError):
    """The ``pv_directory`` section is malformed at :attr:`key`."""

    def __init__(self, key: str, problem: str) -> None:
        super().__init__(f"{key}: {problem}")
        self.key = key
        self.problem = problem


@dataclass(frozen=True)
class ResolvedSection:
    """A checked ``pv_directory`` section.

    Attributes:
        section: The section as configured.
        type: The facility directory's type, or ``None`` for a VA-only section.
        alarm_log: The alarm logger's type, or ``None``.
        alarm_state: The alarm feed's type, or ``None``.
        replay_file: The ``va`` alarm replay as configured, or ``None``.
        put_log: The put log's type, or ``None``.
        warnings: What the section names and this build cannot serve, one line each.
    """

    section: Mapping[str, Any]
    type: str | None = None
    alarm_log: str | None = None
    alarm_state: str | None = None
    replay_file: str | None = None
    put_log: str | None = None
    warnings: tuple[str, ...] = ()


def predicates(resolved: ResolvedSection | None) -> dict[str, bool]:
    """The context keys for *resolved*; all false without a section."""
    if resolved is None:
        return dict.fromkeys(PREDICATES, False)
    alarm_state = resolved.alarm_state is not None or resolved.replay_file is not None
    return {
        "pv_directory_configured": True,
        "pv_directory_alarm_log": resolved.alarm_log is not None,
        "pv_directory_alarm_state": alarm_state,
        "pv_directory_alarm_config": resolved.alarm_log is not None or alarm_state,
        "pv_directory_put_log": resolved.put_log is not None,
    }


def resolve_section(
    config: Mapping[str, Any], *, project_dir: Path | None = None
) -> ResolvedSection | None:
    """Check the ``pv_directory`` section of *config*.

    Args:
        config: The full config mapping: whether the build has a simulator is
            read from outside the section.
        project_dir: The rendered project's directory. A project-relative
            ``replay_file`` is checked only when it is given.

    Returns:
        ``None`` when the section is absent or ``null``, else the section with
        its warnings.

    Raises:
        SectionError: The section is malformed: a block that is not a mapping, a
            ``type`` without ``settings.url``, a Kafka alarm feed without
            ``bootstrap_servers`` or ``topic``, a ``replay_file`` that does not
            resolve, or a ``type`` naming an internal builtin.
    """
    section = config.get(SECTION_KEY)
    if section is None:
        return None
    _require_mapping(section, SECTION_KEY)

    directory_type = _http_type(section, SECTION_KEY)
    blocks: dict[str, Mapping[str, Any]] = {}
    for name in _SUB_BLOCKS:
        block = section.get(name)
        if block is None:
            blocks[name] = {}
            continue
        _require_mapping(block, f"{SECTION_KEY}.{name}")
        blocks[name] = block
    alarm_log = _http_type(blocks["alarm_log"], f"{SECTION_KEY}.alarm_log")
    put_log = _http_type(blocks["put_log"], f"{SECTION_KEY}.put_log")
    alarm_state = _alarm_state_type(blocks["alarm_state"])
    replay_file = _replay_file(blocks["alarm_state"], project_dir)

    return ResolvedSection(
        section=section,
        type=directory_type,
        alarm_log=alarm_log,
        alarm_state=alarm_state,
        replay_file=replay_file,
        put_log=put_log,
        warnings=_warnings(config, directory_type, alarm_state),
    )


def _warnings(
    config: Mapping[str, Any], directory_type: str | None, alarm_state: str | None
) -> tuple[str, ...]:
    from osprey.services.pv_directory._connectors import (
        DIRECTORY_CONNECTORS_AVAILABLE,
        FLOOR_REASON,
        has_simulator,
    )

    if not DIRECTORY_CONNECTORS_AVAILABLE:
        return (f"{SECTION_KEY}: {FLOOR_REASON}",)
    warnings: list[str] = []
    if directory_type is None and not has_simulator(config):
        warnings.append(NO_SOURCE_WARNING)
    if alarm_state == KAFKA_ALARM_STATE and not _kafka_client_installed():
        warnings.append(f"{SECTION_KEY}.alarm_state: {KAFKA_CLIENT_MISSING}")
    return tuple(warnings)


def _kafka_client_installed() -> bool:
    return importlib.util.find_spec("aiokafka") is not None


def _require_mapping(value: Any, key: str) -> None:
    if not isinstance(value, Mapping):
        raise SectionError(key, f"must be a mapping, not {type(value).__name__}")


def _settings(block: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    settings = block.get("settings")
    if settings is None:
        return {}
    if not isinstance(settings, Mapping):
        raise SectionError(f"{key}.settings", f"must be a mapping, not {type(settings).__name__}")
    return settings


def _user_type(block: Mapping[str, Any], key: str) -> str | None:
    """The block's ``type``, refused when it is not text or names an internal builtin."""
    value = block.get("type")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SectionError(f"{key}.type", "must be a non-empty string")
    if value in INTERNAL_TYPES:
        raise SectionError(
            f"{key}.type", f"{value!r} is chosen by OSPREY on the virtual accelerator"
        )
    return value


def _http_type(block: Mapping[str, Any], key: str) -> str | None:
    service = _user_type(block, key)
    settings = _settings(block, key)
    if service is not None and not _given(settings.get("url")):
        raise SectionError(f"{key}.settings.url", f"type {service!r} needs the service URL")
    return service


def _alarm_state_type(block: Mapping[str, Any]) -> str | None:
    key = f"{SECTION_KEY}.alarm_state"
    service = _user_type(block, key)
    settings = _settings(block, key)
    if service == KAFKA_ALARM_STATE:
        for name in ("bootstrap_servers", "topic"):
            if not _given(settings.get(name)):
                raise SectionError(
                    f"{key}.settings.{name}", f"type {service!r} needs the Kafka {name}"
                )
    return service


def _replay_file(block: Mapping[str, Any], project_dir: Path | None) -> str | None:
    key = f"{SECTION_KEY}.alarm_state.replay_file"
    value = block.get("replay_file")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SectionError(key, "must be a non-empty string")
    if not value.startswith(DEMO_PREFIX) and project_dir is None:
        return value
    try:
        path = resolve_replay_file(value, project_dir)
    except ValueError as exc:
        raise SectionError(key, str(exc)) from None
    if not path.is_file():
        where = "packaged with OSPREY" if value.startswith(DEMO_PREFIX) else "in the project"
        raise SectionError(key, f"{value!r} is not a replay file {where}")
    return value


def _given(value: Any) -> bool:
    """Whether a setting holds text: a non-blank string or a list of them."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(isinstance(item, str) and item.strip() for item in value)
    return False
