"""The ``channel_directory.fleet`` sub-block, checked once into :class:`FleetSettings`.

The IOCs view, its refreshers and its tools read the fleet's configuration from
this one object; none of them reads the sub-block itself. :func:`resolve_fleet`
fills every default, so a sub-block spelled as ``fleet: {}`` is the whole
feature on the deployment's own target with the directory's heartbeat as every
IOC's liveness PV.

A malformed value raises :class:`~osprey.services.channel_directory.resolve.SectionError`
naming the key, which the build turns into a refusal.

The connectors are imported inside functions only, so this module imports on
every connectors release.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from osprey.services.channel_directory.resolve import SectionError
from osprey.utils.seconds import positive_seconds

__all__ = [
    "DEFAULT_CLASS",
    "DEFAULT_LISTING_DEADLINE_S",
    "DEFAULT_PERIOD_S",
    "DEFAULT_RETENTION_DAYS",
    "FLEET_KEY",
    "GROUP_BY",
    "HEARTBEAT_ROLE",
    "SERVER_PLACEHOLDER",
    "KINDS",
    "MIN_PERIOD_S",
    "NOT_REFRESHED_IN_PROCESS",
    "FleetClass",
    "FleetSettings",
    "FleetTarget",
    "HealthChannel",
    "LivenessChannel",
    "ChannelSelector",
    "resolve_fleet",
]

#: The sub-block's dotted key, the prefix of every key a refusal names.
FLEET_KEY = "channel_directory.fleet"

#: Seconds between refreshes when ``period_s`` is not set.
DEFAULT_PERIOD_S = 30.0

#: The shortest period D6 allows, whatever the read bound.
MIN_PERIOD_S = 5.0

#: Days a no-longer-listed IOC is remembered after it goes quiet.
DEFAULT_RETENTION_DAYS = 7.0

#: Seconds one listing of the directory and its liveness lookups may take.
DEFAULT_LISTING_DEADLINE_S = 60.0

#: The fields a deployment may group the view by.
GROUP_BY = ("host", "class_name", "group")

#: A liveness PV's kinds: a counter must move to be alive; a status must answer.
KINDS = ("counter", "status")

#: The class of every IOC no roster source assigns one, and of every IOC off ``live``.
DEFAULT_CLASS = "default"

#: The IOC health role whose suffix is the default class's liveness PV.
HEARTBEAT_ROLE = "heartbeat"

#: The record-name suffix of the heartbeat when ``display.server_health_suffixes``
#: names none; the simulator's heartbeat signal, spelled as ``Facts`` defaults it.
_DEFAULT_HEARTBEAT_SUFFIX = "HEARTBEAT"

#: The placeholder a name pattern substitutes the IOC's name for.
SERVER_PLACEHOLDER = "{server}"

#: Why a target is not refreshed when its simulator runs inside each login.
NOT_REFRESHED_IN_PROCESS = (
    "the virtual accelerator is served in process: each login reads its own "
    "instance, so there is no shared machine to refresh"
)

#: Every key the sub-block takes.
_FLEET_LEAVES = (
    "targets",
    "period_s",
    "retention_days",
    "listing_deadline_s",
    "group_by",
    "classes",
    "roster_sources",
)

_ROSTER_REFERENCE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")

Kind = Literal["counter", "status"]
GroupBy = Literal["host", "class_name", "group"]


@dataclass(frozen=True, kw_only=True)
class ChannelSelector:
    """How one IOC's liveness or health PV is found.

    Exactly one of :attr:`suffix` and :attr:`pattern` is set.

    Attributes:
        suffix: A record-name suffix looked up in the directory, among the IOC's PVs.
        pattern: The PV's name with :data:`SERVER_PLACEHOLDER` standing for the IOC's
            name, for an IOC the directory does not list.
        role: The IOC health role the selector stands for, when it is one of
            ``display.server_health_suffixes``: a simulator directory selects that
            role by its served signal rather than by suffix.
    """

    suffix: str | None = None
    pattern: str | None = None
    role: str | None = None

    def address_for(self, ioc: str) -> str | None:
        """The PV's name for *ioc* when the selector is a pattern, else ``None``."""
        if self.pattern is None:
            return None
        return self.pattern.replace(SERVER_PLACEHOLDER, ioc)


@dataclass(frozen=True, kw_only=True)
class LivenessChannel:
    """A class's liveness PV: where it is and how it is judged."""

    selector: ChannelSelector
    kind: Kind


@dataclass(frozen=True, kw_only=True)
class HealthChannel:
    """A class's health PV and its good range; an open side is ``None``."""

    selector: ChannelSelector
    low: float | None = None
    high: float | None = None

    def in_range(self, value: float) -> bool:
        """Whether *value* lies inside the good range, bounds included."""
        if self.low is not None and value < self.low:
            return False
        return not (self.high is not None and value > self.high)


@dataclass(frozen=True, kw_only=True)
class FleetClass:
    """One IOC class: its liveness PV (``None``: it has none) and its health PVs."""

    name: str
    liveness: LivenessChannel | None
    health: tuple[HealthChannel, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FleetTarget:
    """One named target.

    Attributes:
        name: ``live``, ``va`` or ``standin``.
        refreshable: Whether a refresher runs for it.
        reason: Why it is not refreshed, when it is not.
        read_bound_s: The connector's call bound, the longest one read may take;
            ``None`` for a target that is not refreshed.
    """

    name: str
    refreshable: bool = True
    reason: str | None = None
    read_bound_s: float | None = None


@dataclass(frozen=True, kw_only=True)
class FleetSettings:
    """The resolved ``channel_directory.fleet`` sub-block.

    Attributes:
        targets: The named targets in configured order.
        period_s: Seconds between refreshes of each target.
        read_bound_s: The longest call bound of a refreshed target; ``period_s``
            is at least twice it.
        retention_days: Days a no-longer-listed IOC is kept after it goes quiet.
        listing_deadline_s: Seconds one listing of the directory may take.
        group_by: The field the view groups by.
        default_class: The class of an IOC no roster source assigns one.
        classes: The named classes, by name; applied on ``live`` only.
        roster_sources: ``module:attribute`` references; read on ``live`` only.
    """

    targets: tuple[FleetTarget, ...]
    period_s: float = DEFAULT_PERIOD_S
    read_bound_s: float = 0.0
    retention_days: float = DEFAULT_RETENTION_DAYS
    listing_deadline_s: float = DEFAULT_LISTING_DEADLINE_S
    group_by: GroupBy = "host"
    default_class: FleetClass = field(
        default_factory=lambda: FleetClass(
            name=DEFAULT_CLASS,
            liveness=LivenessChannel(
                selector=ChannelSelector(suffix=_DEFAULT_HEARTBEAT_SUFFIX, role=HEARTBEAT_ROLE),
                kind="counter",
            ),
        )
    )
    classes: Mapping[str, FleetClass] = field(default_factory=dict)
    roster_sources: tuple[str, ...] = ()

    def target(self, name: str) -> FleetTarget | None:
        """The named target *name*, or ``None`` when the fleet does not name it."""
        return next((target for target in self.targets if target.name == name), None)

    def refreshed_targets(self) -> tuple[FleetTarget, ...]:
        """The targets a refresher runs for."""
        return tuple(target for target in self.targets if target.refreshable)

    def class_for(self, name: str | None, *, target: str) -> FleetClass:
        """The class an IOC of class *name* has on *target*.

        Named classes apply to ``live`` only; elsewhere, and for a name no class
        carries, the IOC has :attr:`default_class`.
        """
        if target != "live" or name is None:
            return self.default_class
        return self.classes.get(name, self.default_class)


def resolve_fleet(config: Mapping[str, Any], block: Any) -> FleetSettings:
    """Check the ``channel_directory.fleet`` sub-block *block* of *config*.

    Args:
        config: The full config mapping: the targets and their call bounds are
            read from ``control_system``, the heartbeat suffix from
            ``channel_directory.display``.
        block: The sub-block as configured; a mapping.

    Raises:
        SectionError: A value is malformed, naming its key; or ``period_s`` is
            shorter than twice the read bound or than :data:`MIN_PERIOD_S`.
    """
    if not isinstance(block, Mapping):
        raise SectionError(FLEET_KEY, f"must be a mapping, not {type(block).__name__}")
    fleet: Mapping[str, Any] = block
    _refuse_unknown(fleet, FLEET_KEY, set(_FLEET_LEAVES))
    control_system = config.get("control_system")
    section = config.get("channel_directory")
    display = section.get("display") if isinstance(section, Mapping) else None

    targets = _targets(fleet.get("targets"), control_system)
    read_bound = max((t.read_bound_s or 0.0 for t in targets), default=0.0)
    period = _seconds(fleet.get("period_s"), "period_s", DEFAULT_PERIOD_S)
    floor = max(MIN_PERIOD_S, 2 * read_bound)
    if period < floor:
        raise SectionError(
            f"{FLEET_KEY}.period_s",
            f"{period:g}s is shorter than {floor:g}s: at least {MIN_PERIOD_S:g}s and at "
            f"least twice the longest read bound of a refreshed target ({read_bound:g}s)",
        )
    return FleetSettings(
        targets=targets,
        period_s=period,
        read_bound_s=read_bound,
        retention_days=_seconds(
            fleet.get("retention_days"), "retention_days", DEFAULT_RETENTION_DAYS
        ),
        listing_deadline_s=_seconds(
            fleet.get("listing_deadline_s"), "listing_deadline_s", DEFAULT_LISTING_DEADLINE_S
        ),
        group_by=_group_by(fleet.get("group_by")),
        default_class=_default_class(display),
        classes=_classes(fleet.get("classes")),
        roster_sources=_roster_sources(fleet.get("roster_sources")),
    )


# -- targets ---------------------------------------------------------------


def _targets(value: Any, control_system: Any) -> tuple[FleetTarget, ...]:
    from osprey_connectors.types import (
        CONTROL_TARGETS,
        baseline_target,
        configured_targets,
    )

    key = f"{FLEET_KEY}.targets"
    if value is None:
        names = [baseline_target(control_system)]
    else:
        if not isinstance(value, list) or not value:
            raise SectionError(key, "must be a non-empty list of target names")
        names = []
        configured = configured_targets(control_system)
        for name in value:
            if name not in CONTROL_TARGETS:
                raise SectionError(
                    key, f"{name!r} is not a target; targets are {', '.join(CONTROL_TARGETS)}"
                )
            if name not in configured:
                raise SectionError(
                    key,
                    f"{name!r} is not configured in control_system; this deployment has "
                    f"{', '.join(configured)}",
                )
            if name in names:
                raise SectionError(key, f"{name!r} is named twice")
            names.append(name)
    return tuple(_target(name, control_system) for name in names)


def _target(name: str, control_system: Any) -> FleetTarget:
    from osprey_connectors.call_timeout import call_timeout_s
    from osprey_connectors.types import (
        TRANSPORT_IN_PROCESS,
        connector_transport,
        resolve_target,
    )

    connector_type = resolve_target(control_system, name)
    if connector_transport(control_system, connector_type) == TRANSPORT_IN_PROCESS:
        return FleetTarget(name=name, refreshable=False, reason=NOT_REFRESHED_IN_PROCESS)
    connector = control_system.get("connector") if isinstance(control_system, Mapping) else None
    own = connector.get(connector_type) if isinstance(connector, Mapping) else None
    try:
        bound = call_timeout_s(own if isinstance(own, Mapping) else {}, connector_type)
    except ValueError as exc:
        raise SectionError(
            f"control_system.connector.{connector_type}.timeout_s", str(exc)
        ) from None
    return FleetTarget(name=name, read_bound_s=bound)


# -- scalars ---------------------------------------------------------------


def _seconds(value: Any, leaf: str, default: float) -> float:
    if value is None:
        return default
    seconds = positive_seconds(value)
    if seconds is None:
        raise SectionError(f"{FLEET_KEY}.{leaf}", f"must be a positive number, got {value!r}")
    return seconds


def _group_by(value: Any) -> GroupBy:
    if value is None:
        return "host"
    if value not in GROUP_BY:
        raise SectionError(
            f"{FLEET_KEY}.group_by", f"{value!r} is not one of {', '.join(GROUP_BY)}"
        )
    return cast(GroupBy, value)


def _roster_sources(value: Any) -> tuple[str, ...]:
    key = f"{FLEET_KEY}.roster_sources"
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SectionError(key, f"must be a list of module:attribute references, not {value!r}")
    for reference in value:
        if not isinstance(reference, str) or not _ROSTER_REFERENCE.match(reference):
            raise SectionError(key, f"{reference!r} is not a module:attribute reference")
    if len(set(value)) != len(value):
        raise SectionError(key, "names one source twice")
    return tuple(value)


# -- classes ---------------------------------------------------------------


def _default_class(display: Any) -> FleetClass:
    """The default class: the heartbeat of ``display.server_health_suffixes``, a counter.

    Read the way ``Facts`` reads the suffixes: a mapping naming a known role
    replaces the defaults, so one that names no heartbeat leaves the default
    class without a liveness PV.
    """
    value = display.get("server_health_suffixes") if isinstance(display, Mapping) else None
    suffix: str | None = _DEFAULT_HEARTBEAT_SUFFIX
    if isinstance(value, Mapping):
        listed = {
            role: text
            for role, text in value.items()
            if role in ("heartbeat", "uptime", "engineer") and isinstance(text, str) and text
        }
        if listed:
            suffix = listed.get(HEARTBEAT_ROLE)
    liveness = (
        None
        if suffix is None
        else LivenessChannel(
            selector=ChannelSelector(suffix=suffix, role=HEARTBEAT_ROLE), kind="counter"
        )
    )
    return FleetClass(name=DEFAULT_CLASS, liveness=liveness)


def _classes(value: Any) -> dict[str, FleetClass]:
    key = f"{FLEET_KEY}.classes"
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SectionError(key, f"must be a mapping of class name to class, not {value!r}")
    classes: dict[str, FleetClass] = {}
    for name, spec in value.items():
        where = f"{key}.{name}"
        if not isinstance(name, str) or not name.strip():
            raise SectionError(key, f"class name {name!r} must be a non-empty string")
        if name == DEFAULT_CLASS:
            raise SectionError(
                where,
                "the default class is the directory's heartbeat "
                "(channel_directory.display.server_health_suffixes); name a class of your own",
            )
        if not isinstance(spec, Mapping):
            raise SectionError(where, f"must be a mapping, not {type(spec).__name__}")
        unknown = sorted(set(spec) - {"liveness", "health"})
        if unknown:
            raise SectionError(f"{where}.{unknown[0]}", "is not a class key (liveness, health)")
        classes[name] = FleetClass(
            name=name,
            liveness=_liveness(spec.get("liveness"), f"{where}.liveness"),
            health=_health(spec.get("health"), f"{where}.health"),
        )
    return classes


def _liveness(value: Any, where: str) -> LivenessChannel | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SectionError(where, f"must be a mapping, not {type(value).__name__}")
    _refuse_unknown(value, where, {"suffix", "pattern", "kind"})
    kind = value.get("kind")
    if kind not in KINDS:
        raise SectionError(f"{where}.kind", f"{kind!r} is not one of {', '.join(KINDS)}")
    return LivenessChannel(selector=_selector(value, where), kind=kind)


def _health(value: Any, where: str) -> tuple[HealthChannel, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SectionError(where, f"must be a list, not {type(value).__name__}")
    health: list[HealthChannel] = []
    for index, entry in enumerate(value):
        at = f"{where}[{index}]"
        if not isinstance(entry, Mapping):
            raise SectionError(at, f"must be a mapping, not {type(entry).__name__}")
        _refuse_unknown(entry, at, {"suffix", "pattern", "good"})
        low, high = _good_range(entry.get("good"), f"{at}.good")
        health.append(HealthChannel(selector=_selector(entry, at), low=low, high=high))
    return tuple(health)


def _selector(spec: Mapping[str, Any], where: str) -> ChannelSelector:
    suffix = spec.get("suffix")
    pattern = spec.get("pattern")
    if (suffix is None) == (pattern is None):
        raise SectionError(where, "names its PV by exactly one of suffix and pattern")
    if suffix is not None:
        if not isinstance(suffix, str) or not suffix.strip():
            raise SectionError(f"{where}.suffix", "must be a non-empty string")
        return ChannelSelector(suffix=suffix)
    if not isinstance(pattern, str) or SERVER_PLACEHOLDER not in pattern:
        raise SectionError(
            f"{where}.pattern", f"must be a string holding {SERVER_PLACEHOLDER}, got {pattern!r}"
        )
    if "{" in pattern.replace(SERVER_PLACEHOLDER, "") or "}" in pattern.replace(
        SERVER_PLACEHOLDER, ""
    ):
        raise SectionError(f"{where}.pattern", f"{SERVER_PLACEHOLDER} is its only placeholder")
    return ChannelSelector(pattern=pattern)


def _good_range(value: Any, where: str) -> tuple[float | None, float | None]:
    if not isinstance(value, list) or len(value) != 2:
        raise SectionError(where, f"must be [low, high], got {value!r}")
    bounds = [_bound(item, where) for item in value]
    low, high = bounds
    if low is None and high is None:
        raise SectionError(where, "bounds nothing: give low, high or both")
    if low is not None and high is not None and low > high:
        raise SectionError(where, f"low {low:g} is above high {high:g}")
    return low, high


def _bound(item: Any, where: str) -> float | None:
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
        raise SectionError(where, f"{item!r} is not a number")
    return float(item)


def _refuse_unknown(spec: Mapping[str, Any], where: str, allowed: set[str]) -> None:
    unknown = sorted(str(name) for name in set(spec) - allowed)
    if unknown:
        raise SectionError(
            f"{where}.{unknown[0]}", f"is not a key here ({', '.join(sorted(allowed))})"
        )
