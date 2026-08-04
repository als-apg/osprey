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
from osprey.cli.build_profile import list_presets, resolve_build_profile
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


def test_emit_profile_is_standalone(runner: CliRunner, tmp_path: Path) -> None:
    """The emitted ``profile.yml`` is fully explicit — no ``extends:``, the
    preset's content materialized as real keys."""
    import yaml

    target = tmp_path / "my-profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "hello-world"])
    assert result.exit_code == 0, result.output
    text = (target / "profile.yml").read_text()
    assert not any(line.startswith("extends:") for line in text.splitlines()), text
    parsed = yaml.safe_load(text)
    # The preset's own keys are explicit in the emitted file.
    assert parsed["data_bundle"] == "hello_world"
    assert parsed["provider"] == "anthropic"


def test_emit_profile_normalizes_preset_name(runner: CliRunner, tmp_path: Path) -> None:
    """``--preset control_assistant`` (underscored) resolves to the hyphenated preset."""
    import yaml

    target = tmp_path / "my-profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", "control_assistant"])
    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["data_bundle"] == "control_assistant"
    assert "control-assistant" in (target / "README.md").read_text()


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


@pytest.mark.parametrize("preset", list_presets())
def test_emit_profile_then_build_succeeds_for_every_preset(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """Every bundled preset must survive the scaffold-then-build round-trip.

    The emitted profile is a standalone materialization of the preset produced
    by :mod:`osprey.cli.build_profile_emit`. Emission could silently drift out
    of sync with what ``osprey build`` accepts — for a single preset or a whole
    class of them. This test is the standing guard against that drift.

    The parametrization is derived from ``list_presets()`` rather than hardcoded,
    so a newly added preset is covered automatically with no edit here.
    """
    profile_dir = tmp_path / "profile"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    scaffold = runner.invoke(build, ["--emit-profile", str(profile_dir), "--preset", preset])
    assert scaffold.exit_code == 0, f"emit-profile failed for preset {preset!r}: {scaffold.output}"

    rebuild = runner.invoke(
        build,
        [
            "proj",
            str(profile_dir / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert rebuild.exit_code == 0, f"build failed for preset {preset!r}: {rebuild.output}"
    assert (out_dir / "proj" / "config.yml").is_file(), (
        f"preset {preset!r} built without a config.yml"
    )


@pytest.mark.parametrize("preset", list_presets())
def test_emit_profile_resolves_identical_to_preset(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """Full-field parity: the emitted standalone profile resolves to the same
    ``BuildProfile`` as the preset itself (display name aside). This is the
    self-sufficiency guarantee — services, MCP servers, dispatch, env wiring,
    artifact lists, everything the preset configures survives materialization.
    """
    import dataclasses

    target = tmp_path / "profile"
    result = runner.invoke(build, ["--emit-profile", str(target), "--preset", preset])
    assert result.exit_code == 0, result.output

    from_preset, _ = resolve_build_profile(None, preset=preset)
    from_emitted, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
    d_preset = dataclasses.asdict(from_preset)
    d_emitted = dataclasses.asdict(from_emitted)
    d_preset.pop("name")
    d_emitted.pop("name")
    assert d_emitted == d_preset


def test_emit_profile_preserves_preset_comments(runner: CliRunner, tmp_path: Path) -> None:
    """The materialized profile keeps the preset's inline documentation, and an
    overridden value is replaced in place under its original comment."""
    target = tmp_path / "my-profile"
    result = runner.invoke(
        build,
        [
            "--emit-profile",
            str(target),
            "--preset",
            "control-assistant",
            "--set",
            "provider=als-apg",
        ],
    )
    assert result.exit_code == 0, result.output
    text = (target / "profile.yml").read_text()
    # Section and key comments from the preset survive.
    assert "Default LLM provider and model" in text
    assert "Gate hardware-write tool calls on human approval prompt" in text
    # The override replaced the value in place — no duplicate key, old value gone.
    assert "provider: als-apg" in text
    assert "provider: anthropic" not in text


def test_emit_profile_extending_preset_materializes_chain(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A preset that itself uses ``extends`` (readonly persona) emits flat:
    base content + child overrides, each with their own file's comments."""
    target = tmp_path / "ro-profile"
    result = runner.invoke(
        build, ["--emit-profile", str(target), "--preset", "control-assistant-readonly"]
    )
    assert result.exit_code == 0, result.output
    text = (target / "profile.yml").read_text()
    assert not any(line.startswith("extends:") for line in text.splitlines()), text
    # Child-layer keys and their comments are carried.
    assert "deploy_services: false" in text
    assert "control_system.writes_enabled: false" in text
    assert "read-only" in text
    # Base-layer content and comments are materialized too.
    assert "data_bundle: control_assistant" in text
    assert "Default LLM provider and model" in text


def test_emit_profile_build_matches_preset_build_surfaces(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Self-sufficiency, proven at the project level: building from the emitted
    profile produces the same deployed services, service scaffolding, MCP
    servers, and claude_code config as building from the preset directly.

    Uses the richest bundled preset (services + panels + dispatch + VA) so a
    materialization gap in any injector-driving section would surface here.
    """
    import json

    import yaml

    preset = "control-assistant"
    preset_out = tmp_path / "via-preset"
    profile_out = tmp_path / "via-profile"
    preset_out.mkdir()
    profile_out.mkdir()

    r1 = runner.invoke(
        build,
        [
            "proj",
            "--preset",
            preset,
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(preset_out),
        ],
    )
    assert r1.exit_code == 0, r1.output

    prof = tmp_path / "profile"
    r2 = runner.invoke(build, ["--emit-profile", str(prof), "--preset", preset])
    assert r2.exit_code == 0, r2.output
    r3 = runner.invoke(
        build,
        [
            "proj",
            str(prof / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(profile_out),
        ],
    )
    assert r3.exit_code == 0, r3.output

    c_preset = yaml.safe_load((preset_out / "proj" / "config.yml").read_text())
    c_profile = yaml.safe_load((profile_out / "proj" / "config.yml").read_text())
    assert c_profile["deployed_services"] == c_preset["deployed_services"]
    assert c_profile["claude_code"] == c_preset["claude_code"]

    def service_tree(root: Path) -> list[str]:
        return sorted(p.relative_to(root).as_posix() for p in (root / "services").rglob("*"))

    assert service_tree(profile_out / "proj") == service_tree(preset_out / "proj")

    def mcp_servers(root: Path) -> list[str]:
        return sorted(json.loads((root / ".mcp.json").read_text())["mcpServers"].keys())

    assert mcp_servers(profile_out / "proj") == mcp_servers(preset_out / "proj")


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


class TestEmitProfileBakedOverrides:
    """``--set`` / ``-O`` passed alongside ``--emit-profile`` are baked into the
    emitted ``profile.yml`` as real keys, so a validated build one-liner carries
    straight into a facility profile without hand-editing.
    """

    def test_set_pairs_baked_and_resolvable(self, runner: CliRunner, tmp_path: Path) -> None:
        target = tmp_path / "my-profile"
        result = runner.invoke(
            build,
            [
                "--emit-profile",
                str(target),
                "--preset",
                "hello-world",
                "--set",
                "provider=als-apg",
                "--set",
                "model=opus",
            ],
        )
        assert result.exit_code == 0, result.output
        text = (target / "profile.yml").read_text()
        assert "provider: als-apg" in text
        assert "model: opus" in text
        # Replaced in place — the preset's original values must be gone.
        assert "provider: anthropic" not in text
        assert "model: haiku" not in text
        resolved, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
        assert resolved.provider == "als-apg"
        assert resolved.model == "opus"

    def test_override_file_baked(self, runner: CliRunner, tmp_path: Path) -> None:
        override = tmp_path / "o.yml"
        override.write_text("model: sonnet\n")
        target = tmp_path / "my-profile"
        result = runner.invoke(
            build,
            [
                "--emit-profile",
                str(target),
                "--preset",
                "hello-world",
                "-O",
                str(override),
            ],
        )
        assert result.exit_code == 0, result.output
        resolved, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
        assert resolved.model == "sonnet"

    def test_set_wins_over_override_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Layer order matches the render path: -O file(s), then --set on top."""
        override = tmp_path / "o.yml"
        override.write_text("model: sonnet\n")
        target = tmp_path / "my-profile"
        result = runner.invoke(
            build,
            [
                "--emit-profile",
                str(target),
                "--preset",
                "hello-world",
                "-O",
                str(override),
                "--set",
                "model=opus",
            ],
        )
        assert result.exit_code == 0, result.output
        resolved, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
        assert resolved.model == "opus"

    def test_name_override_replaces_seed_name(self, runner: CliRunner, tmp_path: Path) -> None:
        """A --set name=... must land in the seed's own name: slot — the emitted
        YAML must not carry two name: keys."""
        target = tmp_path / "my-profile"
        result = runner.invoke(
            build,
            [
                "--emit-profile",
                str(target),
                "--preset",
                "hello-world",
                "--set",
                "name=My Facility",
            ],
        )
        assert result.exit_code == 0, result.output
        text = (target / "profile.yml").read_text()
        name_lines = [ln for ln in text.splitlines() if ln.startswith("name:")]
        assert len(name_lines) == 1, text
        resolved, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
        assert resolved.name == "My Facility"

    def test_extends_override_rejected(self, runner: CliRunner, tmp_path: Path) -> None:
        """The emitted profile always extends the --preset; overriding extends
        would silently detach it."""
        target = tmp_path / "my-profile"
        result = runner.invoke(
            build,
            [
                "--emit-profile",
                str(target),
                "--preset",
                "hello-world",
                "--set",
                "extends=control-assistant",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "extends" in result.output
        assert not target.exists()

    def test_invalid_override_cleans_up_scaffold(self, runner: CliRunner, tmp_path: Path) -> None:
        """Overrides producing an invalid profile fail fast — and leave no
        half-written scaffold behind."""
        target = tmp_path / "my-profile"
        result = runner.invoke(
            build,
            [
                "--emit-profile",
                str(target),
                "--preset",
                "hello-world",
                "--set",
                "tier=2",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "tier" in result.output
        assert not target.exists()

    def test_emit_then_build_carries_overrides(self, runner: CliRunner, tmp_path: Path) -> None:
        """End-to-end: baked model override survives into the rendered config.yml."""
        import yaml as _yaml

        profile_dir = tmp_path / "my-profile"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        scaffold = runner.invoke(
            build,
            [
                "--emit-profile",
                str(profile_dir),
                "--preset",
                "hello-world",
                "--set",
                "model=opus",
            ],
        )
        assert scaffold.exit_code == 0, scaffold.output

        rebuild = runner.invoke(
            build,
            [
                "proj",
                str(profile_dir / "profile.yml"),
                "--skip-deps",
                "--skip-lifecycle",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert rebuild.exit_code == 0, rebuild.output
        config = _yaml.safe_load((out_dir / "proj" / "config.yml").read_text())
        assert config["claude_code"]["default_model"] == "opus"


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
