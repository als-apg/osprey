"""Every app OSPREY bundles passes the source checks of ``osprey app check``.

A deployment that turns a bundled app on runs these checks on its copy, so a
bundled app that fails one is refused in every deployment. The JS checks run in
their own lane; here are the checks that read the source alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import osprey
from osprey.apps.origin import AppOrigin
from osprey.cli.app_cmd import (
    CheckedApp,
    _boundary_violations,
    _ready_violations,
    _style_violations,
)

BUNDLED = sorted((Path(osprey.__file__).parent / "profiles" / "presets" / "apps").iterdir())


def test_the_bundled_apps_are_found() -> None:
    assert {app.name for app in BUNDLED} >= {"lattice", "orbit-bump"}


@pytest.mark.parametrize("app_dir", BUNDLED, ids=lambda app_dir: app_dir.name)
def test_a_bundled_app_passes_the_source_checks(app_dir: Path) -> None:
    apps = [CheckedApp(app_dir.name, AppOrigin("deployment", app_dir))]

    violations = [
        *_boundary_violations(apps),
        *_style_violations(apps),
        *_ready_violations(apps),
    ]

    assert violations == []
