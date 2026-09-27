"""Clock schedule of a clock-time cron trigger.

A clock schedule is a set of wall-clock times of day, optionally limited to a
set of weekdays, and it is read in one IANA zone. It carries no state: the
slot a trigger waits for is a function of the schedule, the zone and "now".
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import time
from typing import Any

__all__ = ["DAY_NAMES", "ClockSchedule", "parse_clock_schedule"]

# Index equals ``date.weekday()``.
DAY_NAMES: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

_TIME_PATTERN = re.compile(r"^([01][0-9]|2[0-3]):[0-5][0-9]$")
_CLOCK_KEYS = frozenset({"at", "days"})


@dataclass(frozen=True)
class ClockSchedule:
    """Wall-clock times of day, optionally limited to weekdays.

    Attributes:
        times: Times of day, sorted and unique.
        days: Weekday indices (``date.weekday()``); ``None`` means every day.
    """

    times: tuple[time, ...]
    days: frozenset[int] | None = None


def _refuse(trigger_name: str, message: str) -> ValueError:
    return ValueError(f"Trigger '{trigger_name}' {message}")


def _parse_time(trigger_name: str, entry: Any) -> time:
    if isinstance(entry, int) and not isinstance(entry, bool):
        # YAML 1.1 reads an unquoted ``17:00`` as the base-60 integer 1020.
        hours, minutes = divmod(entry, 60)
        quoted = f'"{hours:02d}:{minutes:02d}"'
        raise _refuse(
            trigger_name,
            f"field 'source_config.at' has the number {entry}: YAML read an unquoted "
            f"time as a number; quote it, for example {quoted}",
        )
    if not isinstance(entry, str) or not _TIME_PATTERN.match(entry):
        raise _refuse(
            trigger_name,
            f"field 'source_config.at' entry {entry!r} is not a quoted 24-hour \"HH:MM\" time",
        )
    hour, minute = entry.split(":")
    return time(int(hour), int(minute))


def parse_clock_schedule(
    trigger_name: str, source_config: Mapping[str, Any]
) -> ClockSchedule | None:
    """Read the ``at``/``days`` keys of a cron trigger's ``source_config``.

    Args:
        trigger_name: Name of the trigger, used in every refusal message.
        source_config: The trigger's ``source_config`` mapping.

    Returns:
        The parsed schedule, or ``None`` when neither ``at`` nor ``days`` is
        present (an interval trigger).

    Raises:
        ValueError: The schedule cannot be read; the message names the trigger
            and the key.
    """
    if "at" not in source_config and "days" not in source_config:
        return None

    if "at" not in source_config:
        raise _refuse(trigger_name, "field 'source_config.days' needs 'source_config.at'")
    if "interval_sec" in source_config:
        raise _refuse(
            trigger_name,
            "sets both 'source_config.at' and 'source_config.interval_sec'; "
            "a cron trigger uses one or the other",
        )
    extra = sorted(str(key) for key in source_config if key not in _CLOCK_KEYS)
    if extra:
        raise _refuse(
            trigger_name,
            f"has unknown 'source_config' key(s) {', '.join(extra)} beside 'at'; "
            "a clock trigger takes only 'at' and 'days'",
        )

    raw_times = source_config["at"]
    if not isinstance(raw_times, list) or not raw_times:
        raise _refuse(
            trigger_name, "field 'source_config.at' must be a non-empty list of \"HH:MM\" times"
        )
    times: list[time] = []
    for entry in raw_times:
        parsed = _parse_time(trigger_name, entry)
        if parsed in times:
            raise _refuse(trigger_name, f"field 'source_config.at' repeats {entry!r}")
        times.append(parsed)

    days: frozenset[int] | None = None
    if "days" in source_config:
        raw_days = source_config["days"]
        if not isinstance(raw_days, list) or not raw_days:
            raise _refuse(
                trigger_name,
                f"field 'source_config.days' must be a non-empty list of {', '.join(DAY_NAMES)}",
            )
        indices: list[int] = []
        for entry in raw_days:
            if entry not in DAY_NAMES:
                raise _refuse(
                    trigger_name,
                    f"field 'source_config.days' entry {entry!r} is not one of "
                    f"{', '.join(DAY_NAMES)}",
                )
            index = DAY_NAMES.index(entry)
            if index in indices:
                raise _refuse(trigger_name, f"field 'source_config.days' repeats {entry!r}")
            indices.append(index)
        days = frozenset(indices)

    return ClockSchedule(times=tuple(sorted(times)), days=days)
