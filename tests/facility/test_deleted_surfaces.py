"""Deleted surfaces stay deleted.

Each row names a module that survives and a name it no longer defines. The
module is imported and asked for the name, so a symbol brought back under its
old spelling fails here even where no text sweep reaches it.
"""

from __future__ import annotations

import importlib

import pytest

#: Surviving module -> names it no longer defines.
DELETED_NAMES: dict[str, tuple[str, ...]] = {
    "osprey.cli.build_profile_va_faults": (
        "STANDIN_BPM_ERRORS_ENV",
        "effective_standin_bpm_errors",
    ),
    "osprey.deployment.compose_generator": ("_standin_perturbation",),
    "osprey.deployment.container_lifecycle": (
        "_SOLVED_BASELINES_KIND",
        "_STANDIN_OFFSET_AXES",
        "_STANDIN_TRANSFORM_KIND",
        "_baselines_fingerprint",
        "_preflight_build_derived_env",
        "_preflight_served_lattice",
        "_served_monitor_readings",
        "_solved_monitor_baselines",
        "_standin_bpm_error_spec",
        "_standin_seed_transform",
    ),
    "osprey.deployment.reset": ("DERIVED_ENV_BANNER",),
    "osprey_connectors.dotenv": (
        "BUILD_DERIVED_BANNER",
        "BUILD_DERIVED_KEYS",
        "VA_LATTICE_DEFAULT",
        "VA_LATTICE_KEY",
        "resolved_va_lattice",
    ),
}


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module, names in sorted(DELETED_NAMES.items()) for name in names],
)
def test_the_deleted_name_is_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


def test_the_rows_are_sorted() -> None:
    assert list(DELETED_NAMES) == sorted(DELETED_NAMES)
    for module, names in DELETED_NAMES.items():
        assert list(names) == sorted(names), module
