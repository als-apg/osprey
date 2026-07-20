"""Tests for ``osprey build --emit-profile`` scaffolding mode.

This mode writes an editable profile directory and exits — it does not render
a project. The generated ``profile.yml`` must be a valid input to the normal
build path so the user can immediately rebuild from it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import resolve_build_profile
from osprey.errors import BuildProfileError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_emit_profile_writes_expected_tree(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "my-profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "hello-world"])
    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / "overlays" / "rules" / ".gitkeep").is_file()
    assert (target / "overlays" / "skills" / ".gitkeep").is_file()
    assert (target / "overlays" / "agents" / ".gitkeep").is_file()


def test_emit_profile_seed_extends_preset(runner: CliRunner, tmp_path: Path) -> None:
    """The seed ``profile.yml`` must include the preset name in its ``extends:`` line."""
    target = tmp_path / "my-profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "hello-world"])
    assert result.exit_code == 0, result.output
    text = (target / "profile.yml").read_text()
    assert "extends: hello-world" in text


def test_emit_profile_normalizes_preset_name(runner: CliRunner, tmp_path: Path) -> None:
    """``--preset control_assistant`` (underscored) writes the hyphenated form in extends."""
    target = tmp_path / "my-profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "control_assistant"])
    assert result.exit_code == 0, result.output
    text = (target / "profile.yml").read_text()
    assert "extends: control-assistant" in text


def test_emit_profile_yaml_loads_through_resolver(runner: CliRunner, tmp_path: Path) -> None:
    """The generated ``profile.yml`` must round-trip through ``resolve_build_profile``."""
    target = tmp_path / "my-profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "hello-world"])
    assert result.exit_code == 0, result.output
    resolved, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
    # Preset's data_bundle/provider flow through the extends chain.
    assert resolved.data_bundle == "hello_world"
    assert resolved.provider == "anthropic"


def test_emit_profile_then_build_succeeds(runner: CliRunner, tmp_path: Path) -> None:
    """End-to-end round-trip: scaffold profile, build a project from it."""
    profile_dir = tmp_path / "my-profile"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    scaffold = runner.invoke(build, ["--emit-profile", str(profile_dir), "--preset", "hello-world"])
    assert scaffold.exit_code == 0, scaffold.output

    rebuild = runner.invoke(
        build,
        [
            "from-profile",
            str(profile_dir / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert rebuild.exit_code == 0, rebuild.output
    project = out_dir / "from-profile"
    assert (project / "config.yml").exists()
    # Manifest records profile-source (not preset-source) for round-trip builds.
    import json

    manifest = json.loads((project / ".osprey-manifest.json").read_text())
    assert manifest["build_args"]["source"] == "profile"


def test_emit_profile_round_trip_matches_preset_artifacts(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A scaffold-then-build project must produce the same artifact selection as
    ``--preset`` directly (since the seed adds no overrides).
    """
    # Direct preset build
    preset_out = tmp_path / "preset-out"
    preset_out.mkdir()
    r1 = runner.invoke(
        build,
        [
            "viaP",
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(preset_out),
        ],
    )
    assert r1.exit_code == 0, r1.output

    # Scaffold + build via profile
    prof = tmp_path / "p"
    r2 = runner.invoke(build, ["--emit-profile", str(prof), "--preset", "hello-world"])
    assert r2.exit_code == 0, r2.output
    profile_out = tmp_path / "profile-out"
    profile_out.mkdir()
    r3 = runner.invoke(
        build,
        [
            "viaF",
            str(prof / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(profile_out),
        ],
    )
    assert r3.exit_code == 0, r3.output

    import json

    m1 = json.loads((preset_out / "viaP" / ".osprey-manifest.json").read_text())
    m2 = json.loads((profile_out / "viaF" / ".osprey-manifest.json").read_text())
    # Same artifact selection; only the source-of-truth differs.
    assert m1["artifacts"] == m2["artifacts"]
    assert m1["build_args"]["source"] == "preset"
    assert m2["build_args"]["source"] == "profile"


def test_emit_profile_requires_preset(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(build, ["--emit-profile", str(tmp_path / "p")])
    assert result.exit_code == 2, result.output
    assert "--emit-profile requires --preset" in result.output


def test_emit_profile_rejects_unknown_preset(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(build, ["--emit-profile", str(tmp_path / "p"), "--preset", "bogus"])
    assert result.exit_code == 2, result.output
    assert "bogus" in result.output.lower()


def test_emit_profile_rejects_existing_target(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "already-there"
    target.mkdir()
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "hello-world"])
    assert result.exit_code == 2, result.output
    assert "already exists" in result.output.lower()


@pytest.mark.parametrize(
    "extra_args,token",
    [
        (["--output-dir", "/tmp/x"], "--output-dir"),
        (["--force"], "--force"),
        (["--set", "model=x"], "--set"),
        (["--override", "/tmp/o.yml"], "--override"),
        (["--stream"], "--stream"),
        (["--skip-lifecycle"], "--skip-lifecycle"),
        (["--skip-deps"], "--skip-deps"),
        (["--tier", "3"], "--tier"),
    ],
)
def test_emit_profile_rejects_project_render_flags(
    runner: CliRunner, tmp_path: Path, extra_args: list[str], token: str
) -> None:
    """Any flag that only makes sense for project rendering must be rejected."""
    target = tmp_path / "p"
    result = runner.invoke(
        build,
        ["--emit-profile", str(target), "--preset", "hello-world", *extra_args],
    )
    assert result.exit_code == 2, result.output
    assert "cannot be combined" in result.output.lower()
    assert token in result.output


def test_emit_profile_rejects_positional_args(runner: CliRunner, tmp_path: Path) -> None:
    """Positional PROJECT_NAME / PROFILE must not be passed alongside --emit-profile."""
    target = tmp_path / "p"
    # Project name only
    r1 = runner.invoke(
        build,
        ["pname", "--emit-profile", str(target), "--preset", "hello-world"],
    )
    assert r1.exit_code == 2, r1.output
    assert "PROJECT_NAME" in r1.output

    # Project name + profile path
    other = tmp_path / "other.yml"
    other.write_text("name: x\n")
    target2 = tmp_path / "p2"
    r2 = runner.invoke(
        build,
        [
            "pname",
            str(other),
            "--emit-profile",
            str(target2),
            "--preset",
            "hello-world",
        ],
    )
    assert r2.exit_code == 2, r2.output


class TestTierSelectionRules:
    """Tier selection is restricted to {1, 3} on every configuration path, and
    tier 1 is in_context-only. These rejected-combo cases pin the rule so a
    tier-2 (retired) or tier1+non-in_context request fails with a rule-naming
    error rather than an opaque downstream scaffolding FileNotFoundError.
    """

    def test_cli_tier_2_rejected_by_choice(self, runner: CliRunner, tmp_path: Path) -> None:
        """``--tier 2`` is no longer a valid choice — click rejects it at parse time."""
        out = tmp_path / "out"
        out.mkdir()
        result = runner.invoke(
            build,
            [
                "proj",
                "--preset",
                "hello-world",
                "--tier",
                "2",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(out),
            ],
        )
        assert result.exit_code == 2, result.output
        assert "--tier" in result.output
        # click's invalid-choice message names the rejected value against {1,3}.
        assert "'2' is not one of" in result.output

    def test_profile_tier_2_rejected(self, tmp_path: Path) -> None:
        """A profile YAML with ``tier: 2`` fails validation naming the {1,3} rule."""
        prof = tmp_path / "profile.yml"
        prof.write_text("name: t\nchannel_finder_mode: in_context\ntier: 2\n")
        with pytest.raises(BuildProfileError, match="tier must be 1 or 3"):
            resolve_build_profile(prof.resolve(), preset=None)

    def test_profile_tier1_hierarchical_rejected(self, tmp_path: Path) -> None:
        """tier 1 paired with a non-in_context paradigm fails at validation with
        the tier rule — not later as a scaffolding FileNotFoundError."""
        prof = tmp_path / "profile.yml"
        prof.write_text("name: t\nchannel_finder_mode: hierarchical\ntier: 1\n")
        with pytest.raises(
            BuildProfileError, match="tier 1 requires channel_finder_mode: in_context"
        ):
            resolve_build_profile(prof.resolve(), preset=None)

    def test_profile_tier1_in_context_accepted(self, tmp_path: Path) -> None:
        """The valid tier-1 combo (in_context) resolves cleanly."""
        prof = tmp_path / "profile.yml"
        prof.write_text("name: t\nchannel_finder_mode: in_context\ntier: 1\n")
        resolved, _ = resolve_build_profile(prof.resolve(), preset=None)
        assert resolved.tier == 1
        assert resolved.resolved_tier() == 1

    def test_cli_tier1_override_on_hierarchical_rejected(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A CLI ``--tier 1`` override applied over a hierarchical profile is
        caught by the post-override re-validation, so the build aborts on the
        tier rule instead of reaching (and FileNotFound-ing in) scaffolding."""
        prof = tmp_path / "profile.yml"
        prof.write_text(
            "name: t\n"
            "data_bundle: hello_world\n"
            "provider: anthropic\n"
            "channel_finder_mode: hierarchical\n"
        )
        out = tmp_path / "out"
        out.mkdir()
        result = runner.invoke(
            build,
            [
                "proj",
                str(prof),
                "--tier",
                "1",
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(out),
            ],
        )
        # Aborts (exit 1) at validation; must not surface as an uncaught
        # FileNotFoundError from materialize_tier_artifacts.
        assert result.exit_code == 1, result.output
        assert not isinstance(result.exception, FileNotFoundError)
