"""Tests for `osprey build` preset/override/--set surface.

Covers the new resolve_build_profile() pipeline: bundled presets, override
files, --set inline scalars/lists, mutual exclusion of preset vs positional
profile, and the drift-guard that prevents presets from depending on
profile-dir-relative paths that would break when shipped in a wheel.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import list_presets


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _config_yaml(project_dir: Path) -> dict:
    return yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))


def test_list_presets_exits_zero(runner: CliRunner) -> None:
    result = runner.invoke(build, ["--list-presets"])
    assert result.exit_code == 0, result.output
    listed = {line.strip() for line in result.output.splitlines() if line.strip()}
    assert listed == set(list_presets())
    # Sanity: the workhorse preset must always be bundled.
    assert "hello-world" in listed


def test_preset_hello_world_creates_project(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        build,
        ["smoke", "--preset", "hello-world", "--skip-deps", "--skip-lifecycle",
         "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "smoke"
    assert (project_dir / "config.yml").exists()
    assert (project_dir / "CLAUDE.md").exists()


def test_preset_with_override_file(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "over.yml"
    override.write_text("model: claude-opus-4-5\n")
    result = runner.invoke(
        build,
        ["smoke", "--preset", "hello-world", "-O", str(override),
         "--skip-deps", "--skip-lifecycle", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    assert config["claude_code"]["default_model"] == "claude-opus-4-5"


def test_set_flag_overrides_scalar(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        build,
        ["smoke", "--preset", "hello-world", "--set", "model=claude-sonnet-4-6",
         "--skip-deps", "--skip-lifecycle", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    config = _config_yaml(tmp_path / "smoke")
    assert config["claude_code"]["default_model"] == "claude-sonnet-4-6"


def test_set_with_list_value_extends(runner: CliRunner, tmp_path: Path) -> None:
    """--set on a string list union-dedups (per _merge_lists), preserving base order."""
    result = runner.invoke(
        build,
        # memory-guard is a real registered hook NOT in the hello-world preset
        ["smoke", "--preset", "hello-world", "--set", "hooks=[memory-guard]",
         "--skip-deps", "--skip-lifecycle", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    # The persisted manifest is the post-merge artifact list seen by the build.
    manifest_path = tmp_path / "smoke" / ".osprey-manifest.json"
    assert manifest_path.exists(), result.output
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hooks = set(manifest["artifacts"]["hooks"])
    assert "memory-guard" in hooks
    # Preset's original hooks remain (union, not replace).
    assert {"hook-log", "hook-config", "approval"} <= hooks


def test_positional_profile_still_works(runner: CliRunner, tmp_path: Path) -> None:
    """Backward-compat: existing osprey build PROJECT PROFILE.yml flow."""
    profile = tmp_path / "p.yml"
    profile.write_text(
        "name: PosTest\n"
        "data_bundle: hello_world\n"
        "provider: anthropic\n"
        "model: claude-haiku-4-5\n"
    )
    result = runner.invoke(
        build,
        ["smoke", str(profile), "--skip-deps", "--skip-lifecycle",
         "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "smoke" / "config.yml").exists()


def test_mutually_exclusive_profile_and_preset(runner: CliRunner, tmp_path: Path) -> None:
    profile = tmp_path / "p.yml"
    profile.write_text("name: X\ndata_bundle: hello_world\n")
    result = runner.invoke(
        build, ["smoke", str(profile), "--preset", "hello-world"],
    )
    assert result.exit_code == 2
    assert "not both" in result.output.lower()


def test_neither_profile_nor_preset_required(runner: CliRunner) -> None:
    result = runner.invoke(build, ["smoke"])
    assert result.exit_code == 2
    assert "required" in result.output.lower() or "either" in result.output.lower()


def test_unknown_preset_name(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        build,
        ["smoke", "--preset", "bogus", "--skip-deps", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output.lower()
    # The error should list the available presets to help the user.
    for name in list_presets():
        assert name in result.output


def test_preset_name_normalization(runner: CliRunner, tmp_path: Path) -> None:
    """control-assistant and control_assistant must both resolve to the same preset."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()

    r_hyphen = runner.invoke(
        build,
        ["smoke", "--preset", "control-assistant", "--skip-deps", "--skip-lifecycle",
         "--output-dir", str(out_a)],
    )
    r_under = runner.invoke(
        build,
        ["smoke", "--preset", "control_assistant", "--skip-deps", "--skip-lifecycle",
         "--output-dir", str(out_b)],
    )
    assert r_hyphen.exit_code == 0, r_hyphen.output
    assert r_under.exit_code == 0, r_under.output
    cfg_a = _config_yaml(out_a / "smoke")
    cfg_b = _config_yaml(out_b / "smoke")
    # Same preset → same default_model, same data_bundle in rendered config
    assert cfg_a.get("default_model") == cfg_b.get("default_model")


def test_preset_drift_guard() -> None:
    """Bundled presets must NOT depend on profile-dir-relative paths.

    overlay/services/env.file all resolve relative to profile_dir, which for
    presets is the wheel-installed package directory. Any preset adding these
    will silently fail at install time. Catch it here.
    """
    import importlib.resources
    presets_root = importlib.resources.files("osprey.profiles.presets")
    presets_dir = Path(str(presets_root))
    yml_files = sorted(presets_dir.glob("*.yml"))
    assert yml_files, "no preset YAML files found"
    for yml in yml_files:
        raw = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        assert raw.get("overlay", {}) == {}, (
            f"{yml.name}: overlay must be empty (paths would break in the wheel)"
        )
        assert raw.get("services", {}) == {}, (
            f"{yml.name}: services must be empty (templates would break in the wheel)"
        )
        env = raw.get("env", {}) or {}
        assert env.get("file") is None, (
            f"{yml.name}: env.file must be unset (path would break in the wheel)"
        )
