"""Render the shared build under its deployment's ``config.yml`` with another channel-finder mode.

The shared build is ``built_control_assistant`` (``tests/_builds.py``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tests._builds import BuiltProject


def render_built(built: BuiltProject, render_dir: Path, mode: str) -> list[Path]:
    """Render the build's facility file into ``render_dir``, selecting channel-finder ``mode``."""
    from osprey.facility.render import render_facility_outputs

    config = yaml.safe_load((built.build_dir / "config.yml").read_text(encoding="utf-8"))
    config["channel_finder"]["pipeline_mode"] = mode
    render_dir.mkdir(parents=True)
    return render_facility_outputs(render_dir, built.facility, config, built.facility_dir)
