"""Contract tests for the profile ``osprey profile new`` materializes.

The verb that turns a bundled preset into an editable profile directory: an
explicit standalone ``profile.yml``, the bundle's data tree copied verbatim,
the secret channel, and a tutorial README. Carries the whole scaffolding contract
the removed ``osprey build`` scaffold flag was held to, extended with the
data-tree materialization it never did and the preset-parity checks that prove
nothing is lost on the way from preset to profile.

The command creates the whole facility repo around that profile; the repo's own
shape is pinned in ``test_profile_new_facility_repo.py``. Here the repo is only
the container — every test names the ``profile/`` directory inside it and asserts
on that.
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
    """Create a facility repo whose profile directory is *target*.

    The command is given the repo root, which is *target*'s parent — it writes
    the profile into ``<repo>/profile/``. Tests name the profile directory
    because that is what they assert on.
    """
    return runner.invoke(profile, ["new", str(target.parent), "--preset", preset, *extra])


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
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / "data").is_dir()


def test_no_overlays_tree_is_seeded(runner: CliRunner, tmp_path: Path) -> None:
    """`overlay:` was removed with FR-5 — it now hard-errors at load, so a
    seeded `overlays/` would be a directory nothing reads and the README's
    instructions would point operators at a mechanism that no longer exists.
    Artifacts go in convention directories at the profile root instead."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    assert not (target / "overlays").exists()


def test_profile_is_standalone(runner: CliRunner, tmp_path: Path) -> None:
    """No ``extends:`` — the preset's content is materialized as real keys."""
    target = tmp_path / "my-facility" / "profile"

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
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["data"] == "data"
    resolved, profile_dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.resolved_data_root(profile_dir) == (target / "data").resolve()


def test_preset_name_is_normalized(runner: CliRunner, tmp_path: Path) -> None:
    """``--preset control_assistant`` (underscored) resolves to the hyphenated preset."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control_assistant").exit_code == 0

    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["app_template"] == "control_assistant"
    assert "control-assistant" in (target / "README.md").read_text()


def test_extends_chain_preset_materializes_flat(runner: CliRunner, tmp_path: Path) -> None:
    """A preset that itself uses ``extends`` emits flat: base content plus child
    overrides, each with their own file's comments."""
    target = tmp_path / "ro-facility" / "profile"

    assert _new(runner, target, "control-assistant-readonly").exit_code == 0

    text = (target / "profile.yml").read_text()
    assert not any(line.startswith("extends:") for line in text.splitlines()), text
    assert "deploy_services: false" in text
    assert "control_system.writes_enabled: false" in text
    assert "app_template: control_assistant" in text


def test_preset_comments_survive(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    text = (target / "profile.yml").read_text()
    assert "Default LLM provider and model" in text
    assert "Gate hardware-write tool calls on human approval prompt" in text


# ---------------------------------------------------------------------------
# Trigger config (FR-3: the profile owns the file its dispatch block runs on)
# ---------------------------------------------------------------------------


def _bundled_triggers(name: str) -> Path:
    from osprey.cli.build_profile import _triggers_dir

    return _triggers_dir() / name


def test_bundled_triggers_are_materialized_and_repointed(runner: CliRunner, tmp_path: Path) -> None:
    """A preset naming a bundled trigger set gets its own copy, and the emitted
    ``dispatch.triggers`` names that copy rather than the bundled name."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    materialized = target / "triggers.yml"
    assert materialized.is_file()
    assert materialized.read_bytes() == _bundled_triggers("tutorial_triggers.yml").read_bytes()
    parsed = yaml.safe_load((target / "profile.yml").read_text())
    assert parsed["dispatch"]["triggers"] == "triggers.yml"


def test_profile_without_a_dispatch_block_materializes_no_triggers(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Nothing to own, so nothing is written — and no dispatch block appears."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    assert not (target / "triggers.yml").exists()
    assert yaml.safe_load((target / "profile.yml").read_text()).get("dispatch") is None


def test_emitted_triggers_reference_resolves_inside_the_profile(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The whole point of FR-3: after materialization the profile names only
    profile-local files, so the directory is movable and buildable on its own."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    resolved, profile_dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.dispatch is not None
    named = (profile_dir / resolved.dispatch.triggers).resolve()
    assert named == (target / "triggers.yml").resolve()
    assert named.is_file()


def test_persona_deltas_do_not_restate_the_triggers_reference(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The host owns ``triggers.yml`` for the whole stack. A persona delta
    carries only its own keys, so it neither repeats the dispatch block nor
    re-anchors the path — the implicit merge resolves it against the host."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    for persona_file in sorted((target / "personas").glob("*.yml")):
        parsed = yaml.safe_load(persona_file.read_text())
        assert "dispatch" not in parsed, persona_file.name
        assert not (target / "personas" / "triggers.yml").exists()


# ---------------------------------------------------------------------------
# Per-user context slots (FR-5: seeded from the roster, never frozen)
# ---------------------------------------------------------------------------


def _roster_names(preset: str) -> list[str]:
    from osprey.cli.build_profile_emit import effective_web_terminals
    from osprey.deployment.web_terminals.personas import normalize_users

    resolved, _dir = resolve_build_profile(None, preset)
    web_terminals = effective_web_terminals(resolved.config)
    return [entry["name"] for entry in normalize_users(web_terminals.get("users"))]


@pytest.mark.parametrize("preset", ["control-assistant"])
def test_per_user_context_directories_are_seeded_from_the_roster(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """One empty slot per roster user, so a facility writing per-user context
    has an obvious home for it from the first minute."""
    target = tmp_path / "my-facility" / "profile"
    expected = _roster_names(preset)
    assert expected  # the assertion below must not pass vacuously

    assert _new(runner, target, preset).exit_code == 0

    context_dir = target / "web-terminal-context"
    assert sorted(p.name for p in context_dir.iterdir()) == sorted(expected)
    for user in expected:
        assert (context_dir / user).is_dir()


def test_seeded_context_directories_carry_no_content(runner: CliRunner, tmp_path: Path) -> None:
    """Slots, not literals. Anything written here would become context the
    agent reads, and would freeze at materialization time — the build derives
    what to copy from the roster it resolves then."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control-assistant").exit_code == 0

    for user_dir in (target / "web-terminal-context").iterdir():
        assert [p.name for p in user_dir.iterdir()] == [".gitkeep"]


def test_profile_without_a_web_terminal_module_seeds_no_context(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    assert not (target / "web-terminal-context").exists()


# ---------------------------------------------------------------------------
# Secrets: the profile owns them (FR-1)
# ---------------------------------------------------------------------------


def _provider_key_vars() -> list[str]:
    """Every provider API-key variable, from the registry the seeding reads."""
    from osprey.cli.templates.scaffolding import provider_api_key_entries

    return [entry["var"] for entry in provider_api_key_entries()]


@pytest.fixture
def no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every provider key, so a developer's own exports cannot leak in.

    The seeding reads ``os.environ`` directly, so a real ``ANTHROPIC_API_KEY``
    in the session running the suite would otherwise decide the outcome.
    """
    for var in _provider_key_vars():
        monkeypatch.delenv(var, raising=False)


def test_only_keys_of_referenced_providers_are_seeded(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    """The keys of the providers this profile names, and nothing else — the
    profile is now where a facility's secrets live, so what lands there must be
    predictable, and importing a whole shell keyring is more than it needs."""
    from osprey.utils.dotenv import parse_dotenv_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    target = tmp_path / "my-facility" / "profile"

    # hello-world runs on `provider: anthropic`; it names openai nowhere.
    assert _new(runner, target, "hello-world").exit_code == 0

    env_path = target / ".env"
    assert env_path.is_file()
    assert parse_dotenv_file(env_path) == {"ANTHROPIC_API_KEY": "sk-ant-test"}


def test_a_switched_provider_takes_its_own_key(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    """The rule reads the RESOLVED profile, so a `--set provider=` that the
    emitted profile.yml records moves which key is seeded with it."""
    from osprey.utils.dotenv import parse_dotenv_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world", "--set", "provider=openai").exit_code == 0

    assert parse_dotenv_file(target / ".env") == {"OPENAI_API_KEY": "sk-openai-test"}


def test_a_provider_configured_under_api_providers_is_referenced(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    """A profile that configures a provider's endpoint intends to reach it, so
    its key is seeded even when the agent runs on a different one."""
    from osprey.utils.dotenv import parse_dotenv_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-test")
    target = tmp_path / "my-facility" / "profile"

    result = _new(
        runner,
        target,
        "hello-world",
        "--set",
        "config={api.providers.cborg.base_url: https://example.invalid}",
    )

    assert result.exit_code == 0, result.output
    assert parse_dotenv_file(target / ".env") == {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "CBORG_API_KEY": "sk-cborg-test",
    }


def test_persona_deltas_contribute_their_own_providers() -> None:
    """A persona delta anchors its secrets at the profile root, so it reads the
    SAME `.env` — a persona that switches provider needs its key in there too."""
    from osprey.cli.build_profile_model import BuildProfile
    from osprey.cli.profile_cmd import _referenced_providers

    host = BuildProfile(name="Host", provider="anthropic")

    assert _referenced_providers(host, {"ops": {"provider": "cborg"}}) == {"anthropic", "cborg"}
    # A delta that overrides neither key inherits the host's selection.
    assert _referenced_providers(host, {"ops": {"model": "sonnet"}}) == {"anthropic"}


def test_a_malformed_persona_delta_is_reported_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One parse validates the emitted deltas and feeds every later reader, so a
    bad one gets the diagnostic naming its file rather than a raw YAML
    traceback — and gets it before the target directory exists."""
    import osprey.cli.build_profile_emit as emit_mod
    from osprey.cli.profile_cmd import _materialize_profile_directory

    monkeypatch.setattr(
        emit_mod, "emit_persona_delta_yaml", lambda **kwargs: "provider: [unclosed\n"
    )
    target = tmp_path / "my-facility" / "profile"

    with pytest.raises(BuildProfileError, match="is not valid YAML"):
        _materialize_profile_directory(target, "control-assistant")

    assert not target.exists()


def test_unreferenced_exported_keys_are_named_not_dropped_silently(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    """Seen and skipped has to read differently from lost: the operator exported
    these, and is told which ones the profile had no use for."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert "Not seeded: OPENAI_API_KEY" in result.output


def test_only_unreferenced_keys_exported_writes_no_env_and_says_why(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    """Nothing exported at all and nothing this profile can use are different
    situations with different remedies, so they are not reported the same way."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert not (target / ".env").exists()
    assert "no key for the providers it references" in result.output
    assert "Not seeded: OPENAI_API_KEY" in result.output


def test_seeded_env_file_is_owner_only(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    assert (target / ".env").stat().st_mode & 0o777 == 0o600


def test_no_exported_keys_writes_the_example_but_no_env(
    runner: CliRunner, tmp_path: Path, no_provider_keys: None
) -> None:
    """An empty ``.env`` reads as a configured one. With nothing to seed, the
    documented variable list is the whole deliverable."""
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert not (target / ".env").exists()
    assert (target / ".env.example").is_file()


def test_env_example_documents_the_whole_variable_set(
    runner: CliRunner, tmp_path: Path, no_provider_keys: None
) -> None:
    """One template renders this file into a profile and into a project, so the
    two cannot document different variables."""
    from osprey.cli.templates.scaffolding import service_token_var_entries

    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    content = (target / ".env.example").read_text(encoding="utf-8")
    for var in _provider_key_vars():
        assert var in content, f"{var} missing from the profile .env.example"
    for entry in service_token_var_entries():
        assert entry["var"] in content, f"{entry['var']} missing from the profile .env.example"


def test_env_example_documents_the_profiles_own_env_block(
    runner: CliRunner, tmp_path: Path, no_provider_keys: None
) -> None:
    """The `env:` block is documentation, not values — required vars arrive
    bare, declared defaults arrive with theirs."""
    target = tmp_path / "my-facility" / "profile"

    result = _new(
        runner,
        target,
        "hello-world",
        "--set",
        "env.required=[FACILITY_ENDPOINT]",
        "--set",
        "env.defaults={LOG_LEVEL: info}",
    )

    assert result.exit_code == 0, result.output
    lines = (target / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "FACILITY_ENDPOINT=" in lines
    assert "LOG_LEVEL=info" in lines


def test_gitignore_covers_every_env_variant_except_the_example(
    runner: CliRunner, tmp_path: Path, no_provider_keys: None
) -> None:
    """`.env*` rather than `.env`: the directory also accumulates the
    `.env.lock` the write-back path creates, and neither belongs in git."""
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world").exit_code == 0

    entries = (target / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env*" in entries
    assert "!.env.example" in entries


def test_summary_names_the_secret_files_it_wrote(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_provider_keys: None
) -> None:
    """The caller is told where their secrets now live, and which keys were
    taken from their shell."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert ".env.example" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_summary_says_no_env_was_written_when_nothing_was_exported(
    runner: CliRunner, tmp_path: Path, no_provider_keys: None
) -> None:
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 0, result.output
    assert "not written" in result.output


# ---------------------------------------------------------------------------
# Provenance (FR-6)
# ---------------------------------------------------------------------------


def test_materialized_profile_records_the_preset_it_came_from(
    runner: CliRunner, tmp_path: Path
) -> None:
    from osprey.cli.build_profile_merge import compute_preset_hash

    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "control_assistant").exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.provenance is not None
    # Normalized, not the underscored spelling the caller typed: what is
    # recorded has to be a name a later build can look the preset up by.
    assert resolved.provenance.preset == "control-assistant"
    assert resolved.provenance.preset_hash == compute_preset_hash("control-assistant")


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

    target = tmp_path / "p-facility" / "profile"
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
    rich = tmp_path / "rich-facility" / "profile"
    lean = tmp_path / "lean-facility" / "profile"
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


def test_readme_teaches_convention_dirs_not_the_removed_overlay_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    """This README is the first thing an operator reads. `overlay:` now hard-
    errors at load, so instructions to write one would break every new profile
    on its first build."""
    from osprey.cli.profile_conventions import CONVENTION_SOURCES

    target = tmp_path / "my-facility" / "profile"
    assert _new(runner, target, "hello-world").exit_code == 0

    readme = (target / "README.md").read_text()
    for source in CONVENTION_SOURCES:
        assert f"`{source}/`" in readme, f"{source}/ is not documented"
    assert "overlay:" not in readme
    assert "overlays/" not in readme


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

    target = tmp_path / "p-facility" / "profile"
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
    profile_dir = tmp_path / "facility" / "profile"
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
    profile_dir = tmp_path / "facility" / "profile"
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
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world", "--set", "model=opus").exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.model == "opus"


def test_override_file_is_baked(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "o.yml"
    override.write_text("model: sonnet\nprovider: als-apg\n", encoding="utf-8")
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world", "-O", str(override)).exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.model == "sonnet"
    assert resolved.provider == "als-apg"


def test_set_wins_over_override_file(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "o.yml"
    override.write_text("model: sonnet\n", encoding="utf-8")
    target = tmp_path / "my-facility" / "profile"

    result = _new(runner, target, "hello-world", "-O", str(override), "--set", "model=opus")

    assert result.exit_code == 0, result.output
    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.model == "opus"


def test_name_override_replaces_the_directory_derived_name(
    runner: CliRunner, tmp_path: Path
) -> None:
    target = tmp_path / "my-facility" / "profile"

    assert _new(runner, target, "hello-world", "--set", "name=ALS Control").exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert resolved.name == "ALS Control"


def test_baked_override_survives_into_the_built_project(runner: CliRunner, tmp_path: Path) -> None:
    profile_dir = tmp_path / "my-facility" / "profile"
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
    target = tmp_path / "p-facility" / "profile"

    result = _new(runner, target, "not-a-real-preset")

    assert result.exit_code == 2
    assert "Unknown preset" in result.output
    assert not target.exists()


def test_existing_target_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p-facility" / "profile"
    target.mkdir(parents=True)
    (target / "keepme.txt").write_text("mine\n", encoding="utf-8")

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 2
    assert "already exists" in result.output
    # Untouched — no partial materialization over a user's directory.
    assert (target / "keepme.txt").read_text(encoding="utf-8") == "mine\n"
    assert not (target / "profile.yml").exists()


def test_header_carries_the_flow_diagram(runner: CliRunner, tmp_path: Path) -> None:
    """The top comment block shows profile -> build -> project -> deploy."""
    target = tmp_path / "p-facility" / "profile"
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
    target = tmp_path / "p-facility" / "profile"
    assert _new(runner, target, "control-assistant").exit_code == 0

    persona_files = sorted((target / "personas").glob("*.yml"))
    assert persona_files, "control-assistant should emit persona siblings"
    for persona_file in persona_files:
        assert "edit profile -> rebuild -> redeploy" not in persona_file.read_text(encoding="utf-8")


def test_existing_target_error_suggests_force(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p-facility" / "profile"
    target.mkdir(parents=True)
    (target / "profile.yml").write_text("name: Old\n", encoding="utf-8")

    result = _new(runner, target, "hello-world")

    assert result.exit_code == 2
    assert "--force" in result.output


# ---------------------------------------------------------------------------
# --force: replace an existing materialized profile
# ---------------------------------------------------------------------------


def test_force_replaces_existing_profile_directory(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p-facility" / "profile"
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
    target = tmp_path / "p-facility" / "profile"
    assert _new(runner, target, "hello-world").exit_code == 0

    result = _new(runner, target, "hello-world", "--force", "--set", "model=sonnet")

    assert result.exit_code == 0, result.output
    resolved, _ = resolve_build_profile(target / "profile.yml", None, (), ())
    assert resolved.model == "sonnet"


def test_force_refuses_directory_that_is_not_a_profile(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p-facility" / "profile"
    target.mkdir(parents=True)
    (target / "keepme.txt").write_text("mine\n", encoding="utf-8")

    result = _new(runner, target, "hello-world", "--force")

    assert result.exit_code == 2
    assert "profile.yml" in result.output
    # Refused means untouched.
    assert (target / "keepme.txt").read_text(encoding="utf-8") == "mine\n"


def test_force_allows_replacing_an_empty_directory(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p-facility" / "profile"
    target.mkdir(parents=True)

    result = _new(runner, target, "hello-world", "--force")

    assert result.exit_code == 0, result.output
    assert (target / "profile.yml").is_file()


def test_force_with_bad_preset_leaves_existing_profile_untouched(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--force must not delete anything before the new profile is fully rendered."""
    target = tmp_path / "p-facility" / "profile"
    assert _new(runner, target, "hello-world").exit_code == 0
    original = (target / "profile.yml").read_text(encoding="utf-8")

    result = _new(runner, target, "no-such-preset", "--force")

    assert result.exit_code == 2
    assert (target / "profile.yml").read_text(encoding="utf-8") == original


def test_extends_override_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    override = tmp_path / "o.yml"
    override.write_text("extends: control-assistant\n", encoding="utf-8")
    target = tmp_path / "p-facility" / "profile"

    result = _new(runner, target, "hello-world", "-O", str(override))

    assert result.exit_code == 2
    assert "extends" in result.output
    assert not target.exists()


def test_invalid_override_leaves_no_partial_directory(runner: CliRunner, tmp_path: Path) -> None:
    """The atomicity guarantee: a bad layer fails and materializes nothing."""
    target = tmp_path / "p-facility" / "profile"

    result = _new(runner, target, "hello-world", "--set", "tier=2")

    assert result.exit_code == 2
    assert "tier" in result.output
    assert not target.exists()


def test_data_override_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    """`profile new` materializes the tree, so pointing `data:` elsewhere is a
    mistake — and the preset-mode guard catches it before anything is written."""
    target = tmp_path / "p-facility" / "profile"

    result = _new(runner, target, "hello-world", "--set", "data=/somewhere/else")

    assert result.exit_code == 2
    assert "data" in result.output
    assert not target.exists()


def test_app_template_override_selects_the_copied_bundle(runner: CliRunner, tmp_path: Path) -> None:
    """The copied tree follows the RESOLVED bundle, not the preset's default —
    `--set app_template=...` has to move the data with it."""
    target = tmp_path / "p-facility" / "profile"

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
    target = tmp_path / "p-facility" / "profile"

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
    target = tmp_path / "p-facility" / "profile"

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
    target = tmp_path / "p-facility" / "profile"

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

    result = _new(runner, tmp_path / "p-facility" / "profile", "hello-world")

    assert "Overrides produce" not in result.output
    assert "does not validate" in result.output


def test_missing_override_file_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "p-facility" / "profile"

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


def _take_dispatch_triggers(resolved: dict) -> str | None:
    """Remove and return the dispatch block's ``triggers`` value, if any.

    The other field materialization deliberately rewrites (FR-3): the profile
    owns a copy of the trigger config, so the emitted key names that copy and
    cannot match the preset's bundled name. Lifted out so the rest of the
    dispatch block — worker count, ports, timeouts — is still compared strictly.
    """
    dispatch = resolved.get("dispatch")
    if not isinstance(dispatch, dict):
        return None
    return dispatch.pop("triggers", None)


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
    with the release that understands them. ``provenance`` is the same shape of
    exclusion — a preset was not materialized from anything — and is asserted
    rather than merely dropped. ``data`` differs by design: the profile reads
    its own copied tree, which is the point of the verb.

    ``config`` is compared by what it writes rather than key-for-key: emission
    collapses a key pair like ``modules.web_terminals`` +
    ``modules.web_terminals.enabled`` into one key, which is a different dict
    spelling of the same config.yml. Its persona ``build_profile`` values and
    ``dispatch.triggers`` are compared separately, against the rewrites the verb
    is specified to perform.
    """
    import dataclasses

    from osprey.cli.build_profile_emit import emits_persona_profiles
    from osprey.cli.build_profile_merge import compute_preset_hash

    target = tmp_path / "facility" / "profile"
    assert _new(runner, target, preset).exit_code == 0

    from_preset, _ = resolve_build_profile(None, preset=preset)
    from_materialized, _ = resolve_build_profile((target / "profile.yml").resolve(), preset=None)
    d_preset = dataclasses.asdict(from_preset)
    d_new = dataclasses.asdict(from_materialized)
    # A preset was not materialized from anything; the profile records exactly
    # the preset it came from and that preset's hash at materialization time.
    assert d_preset.pop("provenance") is None
    assert d_new.pop("provenance") == {
        "preset": preset,
        "preset_hash": compute_preset_hash(preset),
    }
    for stamped in ("name", "requires_osprey_version", "data"):
        d_preset.pop(stamped)
        d_new.pop(stamped)

    preset_triggers = _take_dispatch_triggers(d_preset)
    # Rewritten to the profile's own copy wherever the preset declares dispatch.
    assert _take_dispatch_triggers(d_new) == ("triggers.yml" if preset_triggers else None)

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
    target = tmp_path / "my-facility" / "profile"

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
    with_servers = tmp_path / "with-servers" / "profile"

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

    prof = tmp_path / "facility" / "profile"
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

    prof = tmp_path / "p-facility" / "profile"
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
