"""The plan device classes the bridge ships, found by walking its ``devices`` package.

A shipped plan device class is one with a verb that may write: ``set``,
``trigger`` or ``stop``. Each of those states the writes of that verb, and
the tests that read the statements walk this list, so a new class is held to
them the moment it ships. :data:`ONE_OF_EACH` builds one device of every such
class; a class without a builder fails the test that pins the two equal.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable
from typing import Any

from osprey.services.bluesky_bridge import devices as devices_package
from osprey.services.bluesky_bridge.devices.connector import (
    ConnectorPositioner,
    ConnectorSettable,
    ConnectorTriggerable,
)
from osprey.services.bluesky_bridge.devices.mock import (
    MockReadable,
    MockSettable,
    MockTriggerable,
)
from osprey_connectors.command import CommandAct

#: The verbs a plan device may write from, each with its statement.
VERB_STATEMENTS = (("set", "set_writes"), ("trigger", "trigger_writes"), ("stop", "stop_writes"))


def shipped_plan_device_classes() -> list[type]:
    """Every class of the bridge's ``devices`` package that has a verb that may write."""
    classes: list[type] = []
    for module_info in pkgutil.iter_modules(devices_package.__path__):
        module = importlib.import_module(f"{devices_package.__name__}.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if any(callable(getattr(cls, verb, None)) for verb, _ in VERB_STATEMENTS):
                classes.append(cls)
    return classes


def _connector_settable(connector: Any) -> Any:
    return ConnectorSettable(connector, "PLAN:DEVICES:SP", name="settable")


def _connector_positioner(connector: Any) -> Any:
    return ConnectorPositioner(
        connector,
        "PLAN:DEVICES:POSITION:SP",
        "PLAN:DEVICES:POSITION:RB",
        started_by=CommandAct("PLAN:DEVICES:MOVE", "PLAN:DEVICES:MOVE:STATUS", 0, 1.0),
        name="positioner",
    )


def _connector_triggerable(connector: Any) -> Any:
    return ConnectorTriggerable(
        connector, "PLAN:DEVICES:CMD", "PLAN:DEVICES:STATUS", idle=0, timeout_s=1.0, name="command"
    )


#: One builder per shipped plan device class, each taking the connector it drives.
ONE_OF_EACH: dict[type, Callable[[Any], Any]] = {
    ConnectorSettable: _connector_settable,
    ConnectorPositioner: _connector_positioner,
    ConnectorTriggerable: _connector_triggerable,
    MockSettable: lambda _connector: MockSettable(name="PLAN:DEVICES:MOCK:SP"),
    MockReadable: lambda _connector: MockReadable(name="PLAN:DEVICES:MOCK:RB"),
    MockTriggerable: lambda _connector: MockTriggerable(name="PLAN:DEVICES:MOCK:CMD"),
}
