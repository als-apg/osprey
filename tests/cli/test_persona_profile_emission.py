"""Persona-profile emission: the trigger predicate and what ``profile new`` writes.

A profile that stands up the multi-user web-terminal stack owns its personas
too (D7a): ``osprey profile new`` emits a sibling ``personas/<name>.yml`` per
catalog entry, each pointing at the ONE facility data tree with ``data:
../data``, and rewrites the catalog's ``build_profile`` values to name those
files instead of bundled preset names.

Whether a profile triggers that emission is decided by
:func:`~osprey.cli.build_profile_emit.emits_persona_profiles`, whose ORDER is
load-bearing: the four child presets inherit the base's whole
``modules.web_terminals`` subtree (``enabled: true``, personas and all) and
switch it off with a separate dotted ``modules.web_terminals.enabled: false``.
Reading either key on its own says "enabled" — only collapsing first and then
folding the subtree gives the right answer, which is why the 2-true/4-false
matrix below is pinned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.build_profile_emit import (
    effective_web_terminals,
    emits_persona_profiles,
    persona_catalog,
)
from osprey.cli.profile_cmd import profile

# The two bundled presets that stand up the multi-user stack themselves.
TRIGGER_PRESETS = ("control-assistant", "multi-user-demo")

# Their four persona children: each inherits the catalog AND turns the module
# off. Emitting personas-of-a-persona from these would be self-referential.
CHILD_PRESETS = (
    "control-assistant-readonly",
    "control-assistant-readwrite",
    "multi-user-demo-readonly",
    "multi-user-demo-readwrite",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _new(runner: CliRunner, target: Path, preset: str, *extra: str):
    return runner.invoke(profile, ["new", str(target), "--preset", preset, *extra])


# ---------------------------------------------------------------------------
# The trigger predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
def test_trigger_matrix_over_every_bundled_preset(preset: str) -> None:
    """True for exactly the two stack-hosting presets, False for all seven others."""
    resolved, _dir = resolve_build_profile(None, preset)

    assert emits_persona_profiles(resolved.config) is (preset in TRIGGER_PRESETS)


@pytest.mark.parametrize("preset", CHILD_PRESETS)
def test_child_presets_carry_the_pair_that_makes_order_load_bearing(preset: str) -> None:
    """The evidence behind the four False verdicts: each child really does hold
    an inherited ``modules.web_terminals`` subtree that is enabled and full of
    personas, PLUS a dotted ``enabled: false`` that overrides it. An
    unordered read would answer True."""
    resolved, _dir = resolve_build_profile(None, preset)
    config = resolved.config

    inherited = config["modules.web_terminals"]
    assert inherited["enabled"] is True
    assert inherited["personas"]  # the catalog is inherited whole
    assert config["modules.web_terminals.enabled"] is False

    # Folded in the specified order, the deeper key wins and the answer flips.
    assert effective_web_terminals(config)["enabled"] is False
    assert emits_persona_profiles(config) is False


def test_separate_dotted_keys_fold_into_one_subtree() -> None:
    """Neither key prefixes the other, so collapse alone leaves them apart — the
    fold step is what makes this profile trigger."""
    config = {
        "modules.web_terminals.enabled": True,
        "modules.web_terminals.personas": {"ops": {"build_profile": "hello-world"}},
    }

    assert emits_persona_profiles(config) is True
    assert set(persona_catalog(config)) == {"ops"}


def test_nested_ancestor_spelling_is_folded_too() -> None:
    """A ``modules:`` key carrying the subtree nested inside it addresses the
    same thing; the predicate must not miss it just because no key is spelled
    ``modules.web_terminals``."""
    config = {
        "modules": {"web_terminals": {"enabled": True, "personas": {"ops": {}}}},
    }

    assert emits_persona_profiles(config) is True


def test_deeper_key_wins_over_a_nested_ancestor() -> None:
    config = {
        "modules": {"web_terminals": {"enabled": True, "personas": {"ops": {}}}},
        "modules.web_terminals.enabled": False,
    }

    assert emits_persona_profiles(config) is False


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="no-web-terminals-at-all"),
        pytest.param(
            {"modules.web_terminals": {"enabled": True, "personas": {}}}, id="empty-catalog"
        ),
        pytest.param({"modules.web_terminals": {"enabled": True}}, id="no-catalog-key"),
        pytest.param(
            {"modules.web_terminals": {"personas": {"ops": {}}}}, id="catalog-but-not-enabled"
        ),
        pytest.param(
            {"modules.web_terminals": {"enabled": False, "personas": {"ops": {}}}},
            id="explicitly-disabled",
        ),
        pytest.param(
            {"modules.web_terminals": {"enabled": True, "personas": "readonly"}},
            id="catalog-is-not-a-mapping",
        ),
        pytest.param({"modules.web": {"enabled": True, "personas": {"ops": {}}}}, id="near-miss"),
    ],
)
def test_non_triggering_shapes(config: dict) -> None:
    assert emits_persona_profiles(config) is False
    assert persona_catalog(config) == {}


# ---------------------------------------------------------------------------
# What `profile new` writes for a triggering preset
# ---------------------------------------------------------------------------


def _catalog_of(profile_path: Path) -> dict:
    parsed = yaml.safe_load(profile_path.read_text())
    return parsed["config"]["modules.web_terminals"]["personas"]


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_sibling_persona_profiles_are_emitted(runner: CliRunner, tmp_path: Path, preset: str):
    """One ``personas/<name>.yml`` per catalog entry, and nothing else in there."""
    target = tmp_path / "my-profile"
    resolved, _dir = resolve_build_profile(None, preset)
    expected = set(persona_catalog(resolved.config))

    assert _new(runner, target, preset).exit_code == 0

    persona_dir = target / "personas"
    assert persona_dir.is_dir()
    assert {p.name for p in persona_dir.iterdir()} == {f"{name}.yml" for name in expected}


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_each_persona_profile_reads_the_shared_data_tree(
    runner: CliRunner, tmp_path: Path, preset: str
):
    """``data: ../data`` — one facility tree for the whole stack, so a single
    edit under ``<profile>/data/`` reaches every persona render."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, preset).exit_code == 0

    for persona_file in sorted((target / "personas").iterdir()):
        parsed = yaml.safe_load(persona_file.read_text())
        assert parsed["data"] == "../data", persona_file.name
        persona_resolved, persona_dir = resolve_build_profile(persona_file.resolve(), None)
        assert persona_resolved.resolved_data_root(persona_dir) == (target / "data").resolve()


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_catalog_is_rewritten_to_point_at_the_sibling_profiles(
    runner: CliRunner, tmp_path: Path, preset: str
):
    """The whole point: the emitted stack names FILES the facility owns, not
    bundled preset names that would ignore its data tree."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, preset).exit_code == 0

    catalog = _catalog_of(target / "profile.yml")
    assert catalog  # the rewrite must not have emptied it
    for name, entry in catalog.items():
        assert entry["build_profile"] == f"personas/{name}.yml"
        assert (target / entry["build_profile"]).is_file()
        # Everything else the catalog entry carries survives untouched.
        assert entry["project_path"].endswith(entry["project"])


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_persona_profiles_are_standalone_and_keep_their_posture(
    runner: CliRunner, tmp_path: Path, preset: str
):
    """Emitted flat (no ``extends:``) and still carrying the one axis the
    persona presets exist to differ on."""
    target = tmp_path / "my-profile"

    assert _new(runner, target, preset).exit_code == 0

    postures = {}
    for persona_file in sorted((target / "personas").iterdir()):
        text = persona_file.read_text()
        assert not any(line.startswith("extends:") for line in text.splitlines()), persona_file
        parsed = yaml.safe_load(text)
        postures[persona_file.stem] = parsed["config"]["control_system.writes_enabled"]
    assert postures == {"readonly": False, "readwrite": True}


@pytest.mark.parametrize("preset", ("hello-world", "ariel-standalone", *CHILD_PRESETS))
def test_non_trigger_presets_emit_no_personas_directory(
    runner: CliRunner, tmp_path: Path, preset: str
):
    target = tmp_path / "my-profile"

    assert _new(runner, target, preset).exit_code == 0

    assert not (target / "personas").exists()


def test_emitted_stack_builds_end_to_end(runner: CliRunner, tmp_path: Path) -> None:
    """The host profile and every persona profile it emitted each render a
    project, and the persona project's data tree byte-matches the host's — the
    shared-tree promise, verified after an edit to the profile's data."""
    from osprey.cli.build_cmd import build

    target = tmp_path / "my-profile"
    assert _new(runner, target, "multi-user-demo").exit_code == 0

    # Edit the ONE facility data tree the whole stack reads.
    edited = target / "data" / "facility-marker.txt"
    edited.write_text("mark\n", encoding="utf-8")

    out = tmp_path / "out"
    host = runner.invoke(
        build,
        ["host", str(target / "profile.yml"), "--skip-deps", "--skip-lifecycle", "-o", str(out)],
    )
    assert host.exit_code == 0, host.output

    persona_file = target / "personas" / "readonly.yml"
    persona = runner.invoke(
        build,
        ["persona", str(persona_file), "--skip-deps", "--skip-lifecycle", "-o", str(out)],
    )
    assert persona.exit_code == 0, persona.output

    host_data = out / "host" / "data"
    persona_data = out / "persona" / "data"
    assert (host_data / "facility-marker.txt").read_bytes() == b"mark\n"
    assert (persona_data / "facility-marker.txt").read_bytes() == b"mark\n"
    host_tree = sorted(p.relative_to(host_data) for p in host_data.rglob("*"))
    persona_tree = sorted(p.relative_to(persona_data) for p in persona_data.rglob("*"))
    assert host_tree == persona_tree
    for rel in host_tree:
        if (host_data / rel).is_file():
            assert (host_data / rel).read_bytes() == (persona_data / rel).read_bytes(), rel


# ---------------------------------------------------------------------------
# Baked model selection reaches the personas too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_baked_model_selection_is_replayed_into_every_persona(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """A provider/model chosen at ``profile new`` time must retint the WHOLE
    stack, not just the host.

    The canonical flow bakes the choice here and then builds with no flags, so
    nothing is recorded as an explicit override in the project manifest and the
    deploy-time forwarding path has nothing to forward. Without replay, persona
    terminals would run against a provider the facility may hold no credentials
    for while the host works — a failure that only shows up in a deployed
    container.
    """
    target = tmp_path / "my-profile"

    result = _new(
        runner,
        target,
        preset,
        "--set",
        "provider=cborg",
        "--set",
        "model=opus",
        "--set",
        "channel_finder_mode=in_context",
        "--set",
        "tier=1",
    )

    assert result.exit_code == 0, result.output
    host = yaml.safe_load((target / "profile.yml").read_text())
    assert (host["provider"], host["model"]) == ("cborg", "opus")
    persona_files = sorted((target / "personas").iterdir())
    assert persona_files  # the assertions below must not pass vacuously
    for persona_file in persona_files:
        parsed = yaml.safe_load(persona_file.read_text())
        assert parsed["provider"] == "cborg", persona_file.name
        assert parsed["model"] == "opus", persona_file.name
        assert parsed["channel_finder_mode"] == "in_context", persona_file.name
        # Exactly the model-selection keys travel. `tier` is baked into the
        # host but is not whole-stack intent, so it stays behind.
        assert host["tier"] == 1
        assert "tier" not in parsed, persona_file.name


def test_baked_override_file_model_selection_is_replayed_too(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The replay reads the merged ``-O`` + ``--set`` result, so a choice made
    in an override file carries exactly as far as one made inline."""
    override = tmp_path / "o.yml"
    override.write_text("provider: cborg\nmodel: opus\n", encoding="utf-8")
    target = tmp_path / "my-profile"

    assert _new(runner, target, "multi-user-demo", "-O", str(override)).exit_code == 0

    for persona_file in sorted((target / "personas").iterdir()):
        parsed = yaml.safe_load(persona_file.read_text())
        assert (parsed["provider"], parsed["model"]) == ("cborg", "opus"), persona_file.name


# ---------------------------------------------------------------------------
# Rejections: a catalog whose personas cannot be materialized
# ---------------------------------------------------------------------------


def _persona_override(tmp_path: Path, personas: dict) -> Path:
    """An ``-O`` layer adding persona entries to the web-terminal catalog.

    Spelled with the literal dotted ``modules.web_terminals`` key the presets
    use, so the deep-merge lands inside the inherited subtree instead of
    replacing it.
    """
    path = tmp_path / "personas-override.yml"
    path.write_text(
        yaml.safe_dump({"config": {"modules.web_terminals": {"personas": personas}}}),
        encoding="utf-8",
    )
    return path


def test_persona_rendering_a_different_app_template_is_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    """One shared ``../data`` tree cannot serve two app templates. Caught before
    anything is written, and every affected persona is named at once."""
    target = tmp_path / "my-profile"

    result = _new(runner, target, "multi-user-demo", "--set", "app_template=hello_world")

    assert result.exit_code == 2
    assert "cannot serve both" in result.output
    assert "readonly" in result.output and "readwrite" in result.output
    assert not target.exists()  # fail-before-mutating


@pytest.mark.parametrize("bad_name", ["a/b", ".."])
def test_persona_name_that_is_not_a_plain_file_name_is_rejected(
    runner: CliRunner, tmp_path: Path, bad_name: str
) -> None:
    """The catalog key becomes a file name under ``personas/``, so a separator
    (or a traversal) would write outside the directory."""
    override = _persona_override(
        tmp_path,
        {bad_name: {"project": "x", "project_path": "../x", "build_profile": "multi-user-demo"}},
    )
    target = tmp_path / "my-profile"

    result = _new(runner, target, "multi-user-demo", "-O", str(override))

    assert result.exit_code == 2
    assert "plain name" in result.output
    assert not target.exists()


def test_persona_build_profile_that_does_not_resolve_is_rejected(
    runner: CliRunner, tmp_path: Path
) -> None:
    override = _persona_override(
        tmp_path,
        {
            "ghost": {
                "project": "g",
                "project_path": "../g",
                "build_profile": "no-such-preset",
            }
        },
    )
    target = tmp_path / "my-profile"

    result = _new(runner, target, "multi-user-demo", "-O", str(override))

    assert result.exit_code == 2
    assert "ghost" in result.output
    assert "does not resolve" in result.output
    assert not target.exists()


def test_persona_with_no_build_profile_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    override = _persona_override(tmp_path, {"bare": {"project": "b", "project_path": "../b"}})
    target = tmp_path / "my-profile"

    result = _new(runner, target, "multi-user-demo", "-O", str(override))

    assert result.exit_code == 2
    assert "bare" in result.output
    assert "no build_profile" in result.output
    assert not target.exists()


def test_every_unusable_persona_is_reported_in_one_error(runner: CliRunner, tmp_path: Path) -> None:
    """Accumulated errors: a user fixing a catalog sees the whole list, not the
    first problem followed by another run and another problem."""
    override = _persona_override(
        tmp_path,
        {
            "a/b": {"project": "x", "project_path": "../x", "build_profile": "multi-user-demo"},
            "ghost": {"project": "g", "project_path": "../g", "build_profile": "no-such-preset"},
        },
    )
    target = tmp_path / "my-profile"

    result = _new(runner, target, "multi-user-demo", "-O", str(override))

    assert result.exit_code == 2
    assert "a/b" in result.output
    assert "ghost" in result.output
    # One error, not two runs' worth: both problems under a single header.
    assert result.output.count("Cannot materialize the persona profiles") == 1
    assert not target.exists()


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_deploy_auto_render_reaches_the_emitted_sibling_profiles(
    runner: CliRunner, tmp_path: Path, preset: str, monkeypatch
) -> None:
    """The seam between the two halves of this feature: what the emitter writes
    into the catalog is what ``osprey deploy up``'s auto-render resolves and
    hands to ``osprey build``.

    Driven from the REAL rendered project — its config.yml carries the rewritten
    catalog and its manifest carries the ``profile_path_abs`` the relative
    ``personas/<name>.yml`` values are anchored on — so nothing here is
    hand-assembled around the code under test.
    """
    from osprey.cli.build_cmd import build
    from osprey.deployment.web_terminals import persona_images
    from osprey.deployment.web_terminals.personas import resolve_personas
    from osprey.utils.config import ConfigBuilder

    target = tmp_path / "my-profile"
    assert _new(runner, target, preset).exit_code == 0
    out = tmp_path / "out"
    result = runner.invoke(
        build,
        ["host", str(target / "profile.yml"), "--skip-deps", "--skip-lifecycle", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output

    project_root = out / "host"
    config = ConfigBuilder(str(project_root / "config.yml")).raw_config
    web_terminals = config["modules"]["web_terminals"]
    users = resolve_personas(web_terminals, config.get("registry", {}), "test", strict=False)
    # Persona project_paths are `../<project>` siblings of the deployed project;
    # from this tmp cwd none exist, so every persona needs an auto-render.
    monkeypatch.chdir(project_root)
    calls: list[list[str]] = []
    monkeypatch.setattr(persona_images.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    persona_images.auto_render_missing_personas(config, users, {}, project_root=project_root)

    rendered_from = {call[call.index("build") + 2] for call in calls}
    expected = {
        str((target / "personas" / f"{name}.yml").resolve())
        for name in persona_catalog(resolve_build_profile(None, preset)[0].config)
    }
    assert rendered_from == expected
    assert all(Path(path).is_file() for path in rendered_from)
    # Positional, never --preset: these are files, not bundled preset names.
    assert not any("--preset" in call for call in calls)


# ---------------------------------------------------------------------------
# Build exhaust never travels with the data tree (wheel/source parity)
# ---------------------------------------------------------------------------


def test_data_copy_ignore_drops_only_the_named_subtree(tmp_path: Path) -> None:
    """``benchmarks/results`` is dropped; a same-named directory anywhere else,
    and its sibling staging dirs, are not."""
    import shutil

    from osprey.cli.profile_cmd import _data_copy_ignore

    source = tmp_path / "src"
    for rel in ("benchmarks/results", "benchmarks/cross_paradigm", "results", "raw/results"):
        (source / rel).mkdir(parents=True)
        (source / rel / "f.txt").write_text("x", encoding="utf-8")

    shutil.copytree(source, tmp_path / "dst", ignore=_data_copy_ignore(source))

    dst = tmp_path / "dst"
    assert not (dst / "benchmarks" / "results").exists()
    assert (dst / "benchmarks" / "cross_paradigm" / "f.txt").is_file()
    assert (dst / "results" / "f.txt").is_file()
    assert (dst / "raw" / "results" / "f.txt").is_file()


def test_benchmark_results_in_a_source_checkout_are_not_materialized(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A source checkout that has run the channel-finder benchmark holds
    ``data/benchmarks/results/``; a wheel install never does (hatch excludes
    it). Emission must be the same tree either way, so the copy drops it."""
    import shutil

    from osprey.cli.templates.manager import TemplateManager

    results = (
        TemplateManager().template_root / "apps" / "control_assistant" / "data" / "benchmarks"
    ) / "results"
    created_dir = not results.exists()
    results.mkdir(parents=True, exist_ok=True)
    exhaust = results / "run-from-a-test.json"
    exhaust.write_text("{}", encoding="utf-8")
    try:
        target = tmp_path / "my-profile"
        assert _new(runner, target, "control-assistant").exit_code == 0

        assert not (target / "data" / "benchmarks" / "results").exists()
        # The sibling staging dir the bundle really ships still comes across.
        assert (target / "data" / "benchmarks" / "cross_paradigm").is_dir()
    finally:
        exhaust.unlink(missing_ok=True)
        if created_dir:
            shutil.rmtree(results, ignore_errors=True)
