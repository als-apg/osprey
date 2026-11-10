"""The framework baseline registers only services that import.

A service registration naming a class that does not exist fails on every
registry start and registers nothing, so it is dead weight that only logs a
warning.
"""

from __future__ import annotations

import importlib

from osprey.registry.builtins import FrameworkRegistryProvider


def test_every_framework_service_registration_resolves():
    cfg = FrameworkRegistryProvider().get_registry_config()

    unresolved = [
        f"{reg.module_path}.{reg.class_name}"
        for reg in cfg.services
        if not hasattr(importlib.import_module(reg.module_path), reg.class_name)
    ]

    assert unresolved == []


def test_baseline_ships_no_services():
    cfg = FrameworkRegistryProvider().get_registry_config()

    assert cfg.services == []
