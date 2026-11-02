"""Runtime safety switches: the safety features an admin may turn off without a restart.

A feature registers here at startup, and only when the rendered profile turns it
on: the profile decides what runs, this registry only lets an admin take a
running feature out for a while. Three rules keep the safe state the default.

* **Off is the only widening.** A switch moves between the profile's ``on`` and
  ``off``; it can never turn on what the profile left off, because there is
  nothing registered to turn on.
* **A restart returns to the profile.** The state lives in this process and
  nowhere else, so an off a person forgot about ends with the process.
* **No registry, no off.** :func:`feature_is_on` answers ``True`` for an app
  that never built one and for a name nothing registered, so a missed wiring
  leaves a feature running rather than silently stopped.

The registry decides nothing about who may move a switch; the route does
(:mod:`osprey.interfaces.web_terminal.routes.safety_switches`). Each consumer asks
at the moment it would act, so a move takes effect on the next prompt without
respawning anything: an in-process one through :func:`feature_is_on`, code
review's consumers, which run in the agent's processes, through a panel-tier read
(:mod:`osprey.interfaces.web_terminal.routes.code_review`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

#: The watchdog runner's switch: every enabled watchdog of this build at once.
WATCHDOG_SWITCH = "watchdog"

#: Code review's switch: the review of the agent's Python runs and notebook cells.
CODE_REVIEW_SWITCH = "code_review"


@dataclass
class _Switch:
    name: str
    label: str
    note: str = ""
    off_by: str | None = None
    off_at: float | None = None

    @property
    def on(self) -> bool:
        return self.off_at is None


class SafetySwitches:
    """The switchable safety features of one web terminal process."""

    def __init__(self) -> None:
        self._switches: dict[str, _Switch] = {}

    def register(self, name: str, label: str, note: str = "") -> None:
        """Make *name* switchable; it starts on. Registering again changes nothing.

        *note* says what an off leaves running or stopped, for the off dialog.
        """
        self._switches.setdefault(name, _Switch(name, label, note))

    def __contains__(self, name: object) -> bool:
        return name in self._switches

    def is_on(self, name: str) -> bool:
        """Whether *name* runs now; an unregistered name answers ``True``."""
        switch = self._switches.get(name)
        return switch is None or switch.on

    def set(self, name: str, *, on: bool, by: str) -> bool:
        """Move *name*; whether it changed. ``KeyError`` for an unregistered name."""
        switch = self._switches[name]
        if switch.on == on:
            return False
        switch.off_by, switch.off_at = (None, None) if on else (by, time.time())
        return True

    def snapshot(self) -> list[dict[str, Any]]:
        """Every switch, in registration order, as the GET route serves it."""
        return [_row(s) for s in self._switches.values()]

    def row(self, name: str) -> dict[str, Any] | None:
        """One switch as :meth:`snapshot` serves it, or ``None`` when *name* is unregistered."""
        switch = self._switches.get(name)
        return None if switch is None else _row(switch)


def _row(switch: _Switch) -> dict[str, Any]:
    return {
        "name": switch.name,
        "label": switch.label,
        "note": switch.note,
        "on": switch.on,
        "off_by": switch.off_by,
        "off_at": switch.off_at,
    }


def switches_of(app: Any) -> SafetySwitches | None:
    """The registry the lifespan built, or ``None``."""
    switches = getattr(getattr(app, "state", None), "safety_switches", None)
    return switches if isinstance(switches, SafetySwitches) else None


def feature_is_on(app: Any, name: str) -> bool:
    """Whether the feature *name* runs now in *app*; ``True`` with no registry."""
    switches = switches_of(app)
    return switches is None or switches.is_on(name)
