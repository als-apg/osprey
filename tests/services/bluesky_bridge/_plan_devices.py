"""The plan device classes the bridge ships, found by walking its ``devices`` package.

A shipped plan device class is one with a verb that may write: ``set``,
``trigger`` or ``stop``. Each of those states the writes of that verb, and
the tests that read the statements walk this list, so a new class is held to
them the moment it ships.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from osprey.services.bluesky_bridge import devices as devices_package

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
