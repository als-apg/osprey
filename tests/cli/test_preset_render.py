"""Tests for the ``control-assistant`` preset's Virtual Accelerator default.

The preset's ``control_system.type`` defaults to ``virtual_accelerator`` (the
VA soft-IOC ships and is deployed unconditionally as part of the turn-key
Bluesky stack, so a fresh tutorial project's plans drive it end to end
out of the box). ``mock`` is the documented fallback for environments with no
containers to depend on — its non-tracking readbacks make plans browse-only —
and is reachable via ``osprey set connector=mock``.

Covers three angles:

1. In-process profile resolution: the base preset and its two extends
   children (``control-assistant-readonly``, ``control-assistant-readwrite``)
   all carry the ``virtual_accelerator`` override (fast, no build).
2. A full ``osprey build`` of the bare preset: the rendered ``config.yml``
   carries ``control_system.type: virtual_accelerator``.
3. The same rendered build's ``deployed_services`` list is unchanged by the
   flip — it is driven by the ``bluesky:``/``virtual_accelerator:``/
   ``bluesky_web:`` injector blocks, not by ``control_system.type``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.cli.init_cmd import init

CONTROL_SYSTEM_TYPE_KEY = "control_system.type"


def resolve_preset(name: str) -> BuildProfile:
    """Resolve a bundled preset by name, fully merging its ``extends`` chain."""
    profile, _profile_dir = resolve_build_profile(None, preset=name)
    return profile


class TestControlAssistantPresetDefault:
    """The base preset and its extends children all default to the VA."""

    def test_base_preset_defaults_to_virtual_accelerator(self) -> None:
        base = resolve_preset("control-assistant")
        assert base.config.get(CONTROL_SYSTEM_TYPE_KEY) == "virtual_accelerator"

    @pytest.mark.parametrize(
        "preset_name", ["control-assistant-readonly", "control-assistant-readwrite"]
    )
    def test_extends_children_inherit_virtual_accelerator_default(self, preset_name: str) -> None:
        """Neither persona overrides ``control_system.type`` — both inherit the
        base preset's new default through their ``extends:`` chain."""
        profile = resolve_preset(preset_name)
        assert profile.config.get(CONTROL_SYSTEM_TYPE_KEY) == "virtual_accelerator"


@pytest.fixture(scope="module")
def rendered_preset_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the bare ``control-assistant`` preset (no overrides) into a tmp
    repo. Module-scoped: the build is the slow part and every test in this
    section only reads the resulting config.yml.
    """
    tmp_path = tmp_path_factory.mktemp("preset-va-default")
    runner = CliRunner()
    repo = tmp_path / "preset-va-default"
    created = runner.invoke(
        init, [str(repo), "--preset", "control-assistant", "--no-git"], catch_exceptions=False
    )
    assert created.exit_code == 0, created.output
    result = runner.invoke(
        build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    project_dir = repo / "build"
    assert (project_dir / "config.yml").exists()
    return project_dir


@pytest.fixture(scope="module")
def rendered_preset_config(rendered_preset_project: Path) -> dict:
    return yaml.safe_load((rendered_preset_project / "config.yml").read_text(encoding="utf-8"))


class TestControlAssistantRenderedConfig:
    """A real ``osprey build`` of the bare preset carries the flip through to
    the rendered ``config.yml`` — not just the in-process profile object."""

    def test_rendered_control_system_type_is_virtual_accelerator(
        self, rendered_preset_config: dict
    ) -> None:
        assert rendered_preset_config["control_system"]["type"] == "virtual_accelerator"

    def test_rendered_deployed_services_unchanged_by_the_flip(
        self, rendered_preset_config: dict
    ) -> None:
        """``deployed_services`` is driven by the preset's injector blocks
        (``bluesky:``, ``virtual_accelerator:``, ``bluesky_web:``), all of
        which are unconditional on ``control_system.type`` — the connector-type
        flip must not drop or gate any of them. Mirrors
        ``TestControlAssistantTurnkeyPlanServices`` in test_build_profile.py,
        which asserted the same membership before this preset defaulted to the
        VA."""
        deployed = rendered_preset_config["deployed_services"]
        assert "bluesky" in deployed
        assert "virtual_accelerator" in deployed
        assert "bluesky_web" in deployed
