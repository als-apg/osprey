"""The limits view of a packaged tree, and a validator loading it under a mode.

``render_limits`` builds a preset's packaged ``data/facility`` tree in memory
and writes its facility outputs into a render directory, returning the limits
database the render carries. ``validator_under`` loads that file the way a
deployment does, with ``control_system.limits_checking.mode`` stated.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors.control_system._limits_validator import LimitsValidator
from tests._builds import FRAMEWORK_CHECKS

__all__ = [
    "packaged_facility_dir",
    "render_limits",
    "validator_under",
    "without_limits_findings",
]

#: How a ``facility validate`` limits finding's count line begins; its address
#: lines follow it, indented two spaces.
LIMITS_FINDING = "limits: "


def without_limits_findings(stderr: str) -> str:
    """*stderr* of ``facility validate`` without its limits findings: each
    ``limits:`` line and the indented address lines under it. A test about
    other lines of the verb reads them through this."""
    kept: list[str] = []
    inside = False
    for line in stderr.splitlines(keepends=True):
        if line.startswith(LIMITS_FINDING):
            inside = True
            continue
        if inside and line.startswith("  "):
            continue
        inside = False
        kept.append(line)
    return "".join(kept)


def packaged_facility_dir(bundle: str) -> Path:
    """The packaged facility tree an app bundle's preset lands at ``data/facility``."""
    from tests._preset_data import packaged_facility_dir as composed_facility_dir

    return composed_facility_dir(bundle)


def render_limits(
    tmp_path: Path, bundle: str, *, added: Iterable[Mapping[str, Any]] = (), name: str = "render"
) -> Path:
    """Render a packaged tree's facility outputs and return its limits database.

    Args:
        tmp_path: Where the tree's copy and the render go.
        bundle: The app bundle whose tree is built.
        added: Limits records appended to the copy's ``limits.yaml``.
        name: The render's directory name under ``tmp_path``.

    Returns:
        The render's ``data/channel_limits.json``.
    """
    from osprey.facility.views.outputs import render_facility_outputs
    from osprey_facility.build import build_facility

    facility_dir = tmp_path / f"{name}-facility"
    shutil.copytree(packaged_facility_dir(bundle), facility_dir)
    added = list(added)
    if added:
        limits_file = facility_dir / "limits.yaml"
        limits = yaml.safe_load(limits_file.read_text(encoding="utf-8"))
        limits["records"].extend(dict(record) for record in added)
        limits_file.write_text(yaml.safe_dump(limits, sort_keys=False), encoding="utf-8")
    render_dir = tmp_path / name
    render_dir.mkdir()
    doc = build_facility(facility_dir, project_name=bundle, extra_checks=FRAMEWORK_CHECKS)
    render_facility_outputs(render_dir, doc, {}, facility_dir)
    return render_dir / "data" / "channel_limits.json"


def validator_under(monkeypatch: pytest.MonkeyPatch, db_file: Path, mode: str) -> LimitsValidator:
    """Load a limits database under ``mode``, stated deployment-wide."""
    values = {
        "control_system": {"limits_checking": {"enabled": True, "mode": mode}},
        "control_system.limits_checking.database_path": str(db_file),
        "project_root": None,
    }
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr("osprey_connectors.config.default_config_path", lambda: None)
    validator = LimitsValidator.from_config()
    assert isinstance(validator, LimitsValidator)
    return validator
