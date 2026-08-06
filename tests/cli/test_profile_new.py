"""Contract tests for ``osprey profile new``.

The verb that turns a bundled preset into an editable profile directory: an
explicit standalone ``profile.yml``, the bundle's data tree copied verbatim,
the overlay seed, and a tutorial README. Carries the whole scaffolding contract
the removed ``osprey build`` scaffold flag was held to, extended with the
data-tree materialization it never did and the preset-parity checks that prove
nothing is lost on the way from preset to profile.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.profile_cmd import profile
from osprey.errors import BuildProfileError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _new(runner: CliRunner, target: Path, preset: str, *extra: str):
    return runner.invoke(profile, ["new", str(target), "--preset", preset, *extra])


def _build_from(runner: CliRunner, profile_dir: Path, out_dir: Path, name: str = "proj"):
    return runner.invoke(
        build,
        [
            name,
            str(profile_dir / "profile.yml"),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(out_dir),
        ],
    )


# ---------------------------------------------------------------------------
# Tree shape and standalone-ness
# ---------------------------------------------------------------------------


def test_writes_expected_tree(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "my-profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / "data").is_dir()
    for seed in ("rules", "skills", "agents", "web-terminal-context"):
        assert (target / "overlays" / seed / ".gitkeep").is_file(), seed


def test_profile_is_standalone(runner: CliRunner, tmp_path: Path) -> None:
    """No ``extends:`` — the preset's content is materialized as real keys."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    text = (target / "profile.yml").read_text()
    assert not any(line.startswith("extends:") for line in text.splitlines()), text
    parsed = yaml.safe_load(text)
    assert parsed["app_template"] == "hello_world"
    assert parsed["provider"] == "anthropic"


def test_data_key_is_active_and_points_at_the_materialized_tree(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The whole point of the verb: the profile reads its own data tree."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["data"] == "data"
    resolved, profile_dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.resolved_data_root(profile_dir) == (target / "data").resolve()


def test_preset_name_is_normalized(runner: CliRunner, tmp_path: Path) -> None:
    """``--preset control_assistant`` (underscored) resolves to the hyphenated preset."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, "control_assistant").exit_code == 0

    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["app_template"] == "control_assistant"
    assert "control-assistant" in (target / "README.md").read_text()


def test_extends_chain_preset_materializes_flat(runner: CliRunner, tmp_path: Path) -> None:
    """A preset that itself uses ``extends`` emits flat: base content plus child
    overrides, each with their own file's comments."""
    target = tmp_path / "ro-profile"

    assert _new(runner, target, "control-assistant-readonly").exit_code == 0

    text = (target / "profile.yml").read_text()
    assert not any(line.startswith("extends:") for line in text.splitlines()), text
    assert "deploy_services: false" in text
    assert "control_system.writes_enabled: false" in text
    assert "app_template: control_assistant" in text


def test_preset_comments_survive(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "my-profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    text = (target / "profile.yml").read_text()
    assert "Default LLM provider and model" in text
    assert "Gate hardware-write tool calls on human approval prompt" in text


# ---------------------------------------------------------------------------
# Data materialization (D1/FR2: literal copy, no render steps)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
def test_data_tree_is_byte_identical_to_the_bundle(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """Verbatim copy — every bundled file arrives unchanged, staging included.

    The sole exception is build exhaust the wheel does not ship either
    (``_EXCLUDED_DATA_SUBTREES``), so that a source checkout which has run the
    benchmarks materializes the same tree a wheel install does.
    """
    from osprey.cli.profile_cmd import _EXCLUDED_DATA_SUBTREES
    from osprey.cli.templates.manager import TemplateManager

    target = tmp_path / "p"
    assert _new(runner, target, preset).exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    source = TemplateManager().template_root / "apps" / resolved.data_bundle / "data"

    copied = sorted(p.relative_to(target / "data") for p in (target / "data").rglob("*"))
    original = sorted(
        rel
        for rel in (p.relative_to(source) for p in source.rglob("*"))
        if not any(rel.parts[: len(excluded)] == excluded for excluded in _EXCLUDED_DATA_SUBTREES)
    )
    assert copied == original
    for rel in original:
        src, dst = source / rel, target / "data" / rel
        if src.is_file():
            assert src.read_bytes() == dst.read_bytes(), rel


def test_readme_describes_only_staging_dirs_the_bundle_ships(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The tutorial has to match what copytree actually landed — both in which
    directories it names and where they are."""
    rich = tmp_path / "rich"
    lean = tmp_path / "lean"
    assert _new(runner, rich, "control-assistant").exit_code == 0
    assert _new(runner, lean, "hello-world").exit_code == 0

    rich_readme = (rich / "README.md").read_text()
    for described in ("channel_databases/tiers/", "benchmarks/cross_paradigm/", "raw/"):
        assert described in rich_readme, described
        # Named at the location copytree really put it.
        assert (rich / "data" / described.rstrip("/")).is_dir(), described

    lean_readme = (lean / "README.md").read_text()
    assert "build-time input" not in lean_readme
    assert "tiers/" not in lean_readme


def test_stray_j2_in_bundle_data_is_not_rendered(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profile data is content, never templates (D5): a `.j2` file keeps its
    name and its unrendered body."""
    from osprey.cli.templates import manager as manager_mod

    fake_root = tmp_path / "templates"
    real_root = manager_mod.TemplateManager().template_root
    shutil.copytree(real_root, fake_root)
    stray = fake_root / "apps" / "hello_world" / "data" / "stray.txt.j2"
    stray.write_text("{{ never_rendered }}\n", encoding="utf-8")
    monkeypatch.setattr(manager_mod.TemplateManager, "_get_template_root", lambda self: fake_root)

    target = tmp_path / "p"
    assert _new(runner, target, "hello-world").exit_code == 0

    landed = target / "data" / "stray.txt.j2"
    assert landed.is_file()
    assert landed.read_text(encoding="utf-8") == "{{ never_rendered }}\n"


# ---------------------------------------------------------------------------
# SC1: every preset materializes and then builds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
def test_materialize_then_build_succeeds_for_every_preset(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    profile_dir = tmp_path / "profile"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    created = _new(runner, profile_dir, preset)
    assert created.exit_code == 0, f"profile new failed for {preset!r}: {created.output}"

    built = _build_from(runner, profile_dir, out_dir)
    assert built.exit_code == 0, f"build failed for {preset!r}: {built.output}"
    assert (out_dir / "proj" / "config.yml").is_file()


def test_built_project_data_comes_from_the_profile(runner: CliRunner, tmp_path: Path) -> None:
    """An edit to the profile's data tree reaches the built project — proof the
    build sources data from the profile rather than the package."""
    profile_dir = tmp_path / "profile"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert _new(runner, profile_dir, "hello-world").exit_code == 0

    marker = profile_dir / "data" / "facility_marker.txt"
    marker.write_text("edited by the facility\n", encoding="utf-8")

    assert _build_from(runner, profile_dir, out_dir).exit_code == 0

    landed = out_dir / "proj" / "data" / "facility_marker.txt"
    assert landed.is_file()
    assert landed.read_text(encoding="utf-8") == "edited by the facility\n"


# ---------------------------------------------------------------------------
# Baked overrides
# ---------------------------------------------------------------------------


def test_set_pairs_are_baked_and_resolvable(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "my-profile"

    assert _new(runner, target, "hello-world", "--set", "model=opus").exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.model == "opus"


def test_override_file_is_baked(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "o.yml"
    override.write_text("model: sonnet\nprovider: als-apg\n", encoding="utf-8")
    target = tmp_path / "my-profile"

    assert _new(runner, target, "hello-world", "-O", str(override)).exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.model == "sonnet"
    assert resolved.provider == "als-apg"


def test_set_wins_over_override_file(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "o.yml"
    override.write_text("model: sonnet\n", encoding="utf-8")
    target = tmp_path / "my-profile"

    result = _new(runner, target, "hello-world", "-O", str(override), "--set", "model=opus")

    assert result.exit_code == 0, result.output
    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.model == "opus"


def test_name_override_replaces_the_directory_derived_name(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-profile"

    assert _new(runner, target, "hello-world", "--set", "name=ALS Control").exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.name == "ALS Control"


def test_baked_override_survives_into_the_built_project(runner: CliRunner, tmp_path: Path) -> None:
    profile_dir = tmp_path / "my-profile"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert _new(runner, profile_dir, "hello-world", "--set", "model=opus").exit_code == 0

    assert _build_from(runner, profile_dir, out_dir).exit_code == 0

    config = yaml.safe_load((out_dir / "proj" / "config.yml").read_text())
    assert config["claude_code"]["default_model"] == "opus"


# ---------------------------------------------------------------------------
# Negative / atomicity matrix (SC5)
# ---------------------------------------------------------------------------


def test_preset_is_required(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(profile, ["new", str(tmp_path / "p")])

    assert result.exit_code == 2
    assert "--preset" in result.output


def test_unknown_preset_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"

    result = _new(runner, target, "not-a-real-preset")

    assert result.exit_code == 2
    assert "Unknown preset" in result.output
    assert not target.exists()


def test_existing_target_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    target.mkdir()
    (target / "keepme.txt").write_text("mine\n", encoding="utf-8")

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 2
    assert "already exists" in result.output
    # Untouched — no partial materialization over a user's directory.
    assert (target / "keepme.txt").read_text(encoding="utf-8") == "mine\n"
    assert not (target / "profile.yml").exists()


def test_header_carries_the_flow_diagram(runner: CliRunner, tmp_path: Path) -> None:
    """The top comment block shows profile -> build -> project -> deploy."""
    target = tmp_path / "p"
    assert _new(runner, target, "hello-world").exit_code == 0

    text = (target / "profile.yml").read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:45])

    assert "PROFILE" in head
    assert "PROJECT" in head
    assert "DEPLOYMENT" in head
    assert "edit profile -> rebuild -> redeploy" in head
    # Every diagram line is a YAML comment and fits a standard terminal.
    for line in head.splitlines():
        if "PROFILE" in line or "-->" in line:
            assert line.startswith("#")
            assert len(line) <= 80


def test_persona_profiles_do_not_repeat_the_flow_diagram(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    assert _new(runner, target, "control-assistant").exit_code == 0

    persona_files = sorted((target / "personas").glob("*.yml"))
    assert persona_files, "control-assistant should emit persona siblings"
    for persona_file in persona_files:
        assert "edit profile -> rebuild -> redeploy" not in persona_file.read_text(encoding="utf-8")


def test_existing_target_error_suggests_force(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    target.mkdir()
    (target / "profile.yml").write_text("name: Old\n", encoding="utf-8")

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 2
    assert "--force" in result.output


# ---------------------------------------------------------------------------
# --force: replace an existing materialized profile
# ---------------------------------------------------------------------------


def test_force_replaces_existing_profile_directory(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    assert _new(runner, target, "hello-world").exit_code == 0
    # User edits + stray files that a re-materialization must not keep.
    (target / "profile.yml").write_text("name: Edited Away\n", encoding="utf-8")
    (target / "data" / "stray.txt").write_text("stale\n", encoding="utf-8")

    result = _new(runner, target, "hello-world", "--force")

    assert result.exit_code == 0, result.output
    profile_text = (target / "profile.yml").read_text(encoding="utf-8")
    assert "Edited Away" not in profile_text
    assert not (target / "data" / "stray.txt").exists(), "stale file survived --force"


def test_force_bakes_new_set_pairs(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    assert _new(runner, target, "hello-world").exit_code == 0

    result = _new(runner, target, "hello-world", "--force", "--set", "model=sonnet")

    assert result.exit_code == 0, result.output
    resolved, _ = resolve_build_profile(target / "profile.yml", None, (), ())
    assert resolved.model == "sonnet"


def test_force_refuses_directory_that_is_not_a_profile(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    target.mkdir()
    (target / "keepme.txt").write_text("mine\n", encoding="utf-8")

    result = _new(runner, target, "hello-world", "--force")

    assert result.exit_code == 2
    assert "profile.yml" in result.output
    # Refused means untouched.
    assert (target / "keepme.txt").read_text(encoding="utf-8") == "mine\n"


def test_force_allows_replacing_an_empty_directory(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"
    target.mkdir()

    result = _new(runner, target, "hello-world", "--force")

    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()


def test_force_with_bad_preset_leaves_existing_profile_untouched(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--force must not delete anything before the new profile is fully rendered."""
    target = tmp_path / "p"
    assert _new(runner, target, "hello-world").exit_code == 0
    original = (target / "profile.yml").read_text(encoding="utf-8")

    result = _new(runner, target, "no-such-preset", "--force")

    assert result.exit_code == 2
    assert (target / "profile.yml").read_text(encoding="utf-8") == original


def test_extends_override_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "o.yml"
    override.write_text("extends: control-assistant\n", encoding="utf-8")
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world", "-O", str(override))

    assert result.exit_code == 2
    assert "extends" in result.output
    assert not target.exists()


def test_invalid_override_leaves_no_partial_directory(runner: CliRunner, tmp_path: Path) -> None:
    """The atomicity guarantee: a bad layer fails and materializes nothing."""
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world", "--set", "tier=2")

    assert result.exit_code == 2
    assert "tier" in result.output
    assert not target.exists()


def test_data_override_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    """`profile new` materializes the tree, so pointing `data:` elsewhere is a
    mistake — and the preset-mode guard catches it before anything is written."""
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world", "--set", "data=/somewhere/else")

    assert result.exit_code == 2
    assert "data" in result.output
    assert not target.exists()


def test_app_template_override_selects_the_copied_bundle(runner: CliRunner, tmp_path: Path) -> None:
    """The copied tree follows the RESOLVED bundle, not the preset's default —
    `--set app_template=...` has to move the data with it."""
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world", "--set", "app_template=channel_finder_standalone")

    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["app_template"] == "channel_finder_standalone"
    # The channel-finder bundle's tree, not hello-world's lone limits file.
    assert (target / "data" / "channel_databases" / "hierarchical.json").is_file()
    assert not (target / "data" / "channel_limits.json").exists()


def test_data_override_via_file_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    """The `-O` route into `data:` is closed too, not just `--set`."""
    override = tmp_path / "o.yml"
    override.write_text("data: /somewhere/else\n", encoding="utf-8")
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world", "-O", str(override))

    assert result.exit_code == 2
    assert "data" in result.output
    assert not target.exists()


def test_failure_after_mkdir_removes_the_target(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-mkdir cleanup path: everything the seven cases above miss,
    because they all fail before the directory exists."""
    import shutil as shutil_mod

    from osprey.cli import profile_cmd

    boom = RuntimeError("disk went away mid-copy")

    def explode(src, dst, *args, **kwargs):
        raise boom

    monkeypatch.setattr(shutil_mod, "copytree", explode)
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world")

    assert result.exit_code != 0
    assert not target.exists(), "a partial profile directory survived the failure"
    # The original cause is not swallowed by the cleanup.
    assert result.exception is boom
    assert profile_cmd is not None  # import kept meaningful for the reader


def test_failed_round_trip_after_mkdir_removes_the_target(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same cleanup, reached through the round-trip rather than the copy."""
    from osprey.cli import build_profile

    # profile_cmd imports the name from the facade, so that is the binding a
    # patch has to replace — patching build_profile_resolve would be invisible.
    real = build_profile.resolve_build_profile

    def fail_on_round_trip(profile_path, preset, *args, **kwargs):
        # The up-front call resolves the preset (profile_path is None); the
        # round-trip is the one that reads the written profile file.
        if profile_path is not None:
            raise BuildProfileError("simulated round-trip failure")
        return real(profile_path, preset, *args, **kwargs)

    monkeypatch.setattr(build_profile, "resolve_build_profile", fail_on_round_trip)
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 2
    assert "simulated round-trip failure" in result.output
    assert "Nothing was materialized" in result.output
    assert not target.exists()


def test_round_trip_failure_without_layers_does_not_blame_overrides(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `-O` and no `--set` means the user supplied nothing to blame."""
    from osprey.cli import build_profile

    real = build_profile.resolve_build_profile

    def fail_on_round_trip(profile_path, preset, *args, **kwargs):
        if profile_path is not None:
            raise BuildProfileError("simulated round-trip failure")
        return real(profile_path, preset, *args, **kwargs)

    monkeypatch.setattr(build_profile, "resolve_build_profile", fail_on_round_trip)

    result = _new(runner, tmp_path / "p", "hello-world")

    assert "Overrides produce" not in result.output
    assert "does not validate" in result.output


def test_missing_override_file_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p"

    result = _new(runner, target, "hello-world", "-O", str(tmp_path / "nope.yml"))

    assert result.exit_code == 2
    assert not target.exists()


# ---------------------------------------------------------------------------
# Parity with the preset: nothing is lost in materialization
# ---------------------------------------------------------------------------


def _applied_config(scratch: Path, config: dict) -> dict:
    """Return the config.yml a profile's ``config`` block writes.

    Runs the real writer the build uses, so two differently-spelled override
    blocks that land the same values compare equal.
    """
    from osprey.utils.config_writer import config_update_fields

    scratch.write_text("{}\n", encoding="utf-8")
    config_update_fields(scratch, config)
    return yaml.safe_load(scratch.read_text(encoding="utf-8"))


def _take_persona_build_profiles(applied: dict) -> dict[str, str]:
    """Remove and return the persona catalog's ``build_profile`` values.

    The one field materialization deliberately rewrites (D7a): a profile that
    stands up the web-terminal persona stack emits sibling persona profiles and
    repoints the catalog at them, so this key cannot match the preset's. It is
    lifted out here so the rest of the catalog — projects, paths, ports — is
    still compared strictly, and so the rewrite itself can be asserted rather
    than merely tolerated.
    """
    modules = applied.get("modules")
    web_terminals = modules.get("web_terminals") if isinstance(modules, dict) else None
    catalog = web_terminals.get("personas") if isinstance(web_terminals, dict) else None
    if not isinstance(catalog, dict):
        return {}
    return {
        name: entry.pop("build_profile")
        for name, entry in catalog.items()
        if isinstance(entry, dict) and "build_profile" in entry
    }


@pytest.mark.parametrize("preset", list_presets())
def test_resolves_identical_to_the_preset(runner: CliRunner, tmp_path: Path, preset: str) -> None:
    """Full-field parity: the materialized profile resolves to the same
    ``BuildProfile`` as the preset itself (display name, schema stamp, and the
    now-local data root aside). This is the self-sufficiency guarantee —
    services, MCP servers, dispatch, env wiring, artifact lists, everything the
    preset configures survives materialization.

    ``requires_osprey_version`` is excluded by contract, not by convenience: a
    materialized profile outlives the release that wrote it, so it stamps the
    schema floor a future reader needs. Presets carry no stamp because they ship
    with the release that understands them. ``data`` differs by design — the
    profile reads its own copied tree, which is the point of the verb.

    ``config`` is compared by what it writes rather than key-for-key: emission
    collapses a key pair like ``modules.web_terminals`` +
    ``modules.web_terminals.enabled`` into one key, which is a different dict
    spelling of the same config.yml. Its persona ``build_profile`` values are
    compared separately, against the rewrite the verb is specified to perform.
    """
    import dataclasses

    from osprey.cli.build_profile_emit import emits_persona_profiles

    target = tmp_path / "profile"
    assert _new(runner, target, preset).exit_code == 0

    from_preset, _ = resolve_build_profile(None, preset=preset)
    from_materialized, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
    d_preset = dataclasses.asdict(from_preset)
    d_new = dataclasses.asdict(from_materialized)
    for stamped in ("name", "requires_osprey_version", "data"):
        d_preset.pop(stamped)
        d_new.pop(stamped)

    new_config = _applied_config(tmp_path / "new.yml", d_new.pop("config"))
    preset_config = _applied_config(tmp_path / "preset.yml", d_preset.pop("config"))
    new_personas = _take_persona_build_profiles(new_config)
    preset_personas = _take_persona_build_profiles(preset_config)
    assert new_config == preset_config
    # Rewritten to the emitted siblings for a profile that deploys the stack;
    # untouched for one that only inherits the catalog with the module off.
    rewrites = emits_persona_profiles(from_preset.config)
    assert new_personas == {
        name: (f"personas/{name}.yml" if rewrites else value)
        for name, value in preset_personas.items()
    }
    assert d_new == d_preset


def test_facility_extension_guidance_is_appended(runner: CliRunner, tmp_path: Path) -> None:
    """Sections no bundled preset carries — facility MCP servers and custom
    artifact categories — are appended as commented guidance, and the guidance
    is suppressed for a section the written profile actually defines."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    text = (target / "profile.yml").read_text()
    assert "# mcp_servers:" in text
    assert "#   lattice:" in text
    assert "# artifact_server:" in text
    assert '#       color: "#4C9AFF"' in text

    # A profile that defines mcp_servers itself gets the real key, not the hint.
    override = tmp_path / "o.yml"
    override.write_text(
        "mcp_servers:\n"
        "  facility_tools:\n"
        "    command: /usr/bin/facility-mcp\n"
        "    permissions:\n"
        "      allow: [ping]\n"
    )
    with_servers = tmp_path / "with-servers"

    assert _new(runner, with_servers, "control-assistant", "-O", str(override)).exit_code == 0

    text = (with_servers / "profile.yml").read_text()
    assert "# mcp_servers:" not in text
    assert "Facility MCP servers" not in text
    assert "facility_tools:" in text
    # The category guidance is independent — still appended.
    assert "# artifact_server:" in text


def test_build_surfaces_match_a_direct_preset_build(runner: CliRunner, tmp_path: Path) -> None:
    """Self-sufficiency, proven at the project level: building from the
    materialized profile produces the same deployed services, service
    scaffolding, MCP servers, and claude_code config as building from the preset
    directly.

    Uses the richest bundled preset (services + panels + dispatch + VA) so a
    materialization gap in any injector-driving section would surface here.
    """
    import json

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
    assert _new(runner, prof, preset).exit_code == 0
    assert _build_from(runner, prof, profile_out).exit_code == 0

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


def test_round_trip_matches_preset_artifacts(runner: CliRunner, tmp_path: Path) -> None:
    """A materialize-then-build project selects the same artifacts as
    ``--preset`` directly (the profile adds no overrides), and each build
    records the source it was actually invoked from.
    """
    import json

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

    prof = tmp_path / "p"
    profile_out = tmp_path / "profile-out"
    profile_out.mkdir()
    assert _new(runner, prof, "hello-world").exit_code == 0
    assert _build_from(runner, prof, profile_out, name="viaF").exit_code == 0

    m1 = json.loads((preset_out / "viaP" / ".osprey-manifest.json").read_text())
    m2 = json.loads((profile_out / "viaF" / ".osprey-manifest.json").read_text())
    # Same artifact selection; only the source-of-truth differs.
    assert m1["artifacts"] == m2["artifacts"]
    assert m1["build_args"]["source"] == "preset"
    assert m2["build_args"]["source"] == "profile"
