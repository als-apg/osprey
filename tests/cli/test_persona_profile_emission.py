"""Persona-profile emission: the trigger predicate and what ``osprey init`` writes.

A profile that stands up the multi-user web-terminal stack owns its personas
too (D7a): ``osprey init`` emits a ``personas/<name>.yml`` per catalog entry
beside the repo's ``profile.yml``, and rewrites the catalog's ``build_profile``,
``project`` and ``project_path`` values to name those files and this repo's own
build zone instead of bundled preset names.

Each emitted file is a pure DELTA (FR-10) — the persona preset's own layer and
nothing else, with no ``extends:``. Sitting in ``personas/`` beside the repo's
``profile.yml`` IS the inheritance: a persona render merges the delta over that
profile and anchors every profile-relative path at the repo root. So the whole
stack shares ONE facility data tree, one trigger config, and one set of
convention dirs, and none of them is restated in a persona file.

Whether a profile triggers that emission is decided by
:func:`~osprey.cli.build_profile_emit.emits_persona_profiles`, whose ORDER is
load-bearing: the child presets inherit the base's whole
``modules.web_terminals`` subtree (``enabled: true``, personas and all) and
switch it off with a separate dotted ``modules.web_terminals.enabled: false``.
Reading either key on its own says "enabled" — only collapsing first and then
folding the subtree gives the right answer, which is why the trigger matrix
below is pinned.
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
from osprey.cli.init_cmd import init

# The bundled preset(s) that stand up the multi-user stack themselves.
TRIGGER_PRESETS = ("control-assistant",)

# Its persona children: each inherits the catalog AND turns the module off.
# Emitting personas-of-a-persona from these would be self-referential.
CHILD_PRESETS = (
    "control-assistant-readonly",
    "control-assistant-readwrite",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _new(runner: CliRunner, target: Path, preset: str, *extra: str):
    """Materialize a deployment repo at *target*.

    The repo root IS the source zone, so ``personas/`` sits directly beside the
    ``profile.yml`` the deltas merge over. ``--no-git`` throughout: nothing here
    reads the history.
    """
    return runner.invoke(init, [str(target), "--preset", preset, "--no-git", *extra])


# ---------------------------------------------------------------------------
# The trigger predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
def test_trigger_matrix_over_every_bundled_preset(preset: str) -> None:
    """True for exactly the stack-hosting preset(s), False for every other."""
    resolved, _dir = resolve_build_profile(None, preset)

    assert emits_persona_profiles(resolved.config) is (preset in TRIGGER_PRESETS)


@pytest.mark.parametrize("preset", CHILD_PRESETS)
def test_child_presets_carry_the_pair_that_makes_order_load_bearing(preset: str) -> None:
    """The evidence behind the child False verdicts: each child really does hold
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
# What `osprey init` writes for a triggering preset
# ---------------------------------------------------------------------------


def _catalog_of(profile_path: Path) -> dict:
    parsed = yaml.safe_load(profile_path.read_text())
    return parsed["config"]["modules.web_terminals"]["personas"]


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_sibling_persona_profiles_are_emitted(runner: CliRunner, tmp_path: Path, preset: str):
    """One ``personas/<name>.yml`` per catalog entry, and nothing else in there."""
    target = tmp_path / "my-facility"
    resolved, _dir = resolve_build_profile(None, preset)
    expected = set(persona_catalog(resolved.config))

    assert _new(runner, target, preset).exit_code == 0

    persona_dir = target / "personas"
    assert persona_dir.is_dir()
    assert {p.name for p in persona_dir.iterdir()} == {f"{name}.yml" for name in expected}


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_no_persona_restates_the_shared_data_tree(runner: CliRunner, tmp_path: Path, preset: str):
    """One facility tree for the whole stack, named once. The host profile
    carries ``data: data``; a persona delta names no tree at all, because the
    merge anchors it at the host's directory. A ``data:`` here would be a second
    copy of the same decision, free to drift from the host's."""
    target = tmp_path / "my-facility"

    assert _new(runner, target, preset).exit_code == 0

    assert yaml.safe_load((target / "profile.yml").read_text())["data"] == "data"
    persona_files = sorted((target / "personas").iterdir())
    assert persona_files  # the loop below must not pass vacuously
    for persona_file in persona_files:
        assert "data" not in yaml.safe_load(persona_file.read_text()), persona_file.name


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_catalog_is_rewritten_to_point_at_the_sibling_profiles(
    runner: CliRunner, tmp_path: Path, preset: str
):
    """The whole point: the emitted stack names FILES the facility owns, not
    bundled preset names that would ignore its data tree."""
    target = tmp_path / "my-facility"

    assert _new(runner, target, preset).exit_code == 0

    catalog = _catalog_of(target / "profile.yml")
    assert catalog  # the rewrite must not have emptied it
    for name, entry in catalog.items():
        assert entry["build_profile"] == f"personas/{name}.yml"
        assert (target / entry["build_profile"]).is_file()
        # Everything else the catalog entry carries survives untouched.
        assert entry["project_path"].endswith(entry["project"])


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_persona_profiles_are_deltas_and_keep_their_posture(
    runner: CliRunner, tmp_path: Path, preset: str
):
    """Emitted as a pure DELTA over the host profile: the host stays the single
    source of truth, and each persona file carries only what makes it that
    persona — with the write posture pinned explicitly in the delta, where no
    host edit can silently override it.

    There is no ``extends:``: living in ``personas/`` beside the host profile is
    what makes the file a delta, and a written ``extends:`` there is rejected at
    build time."""
    target = tmp_path / "my-facility"

    assert _new(runner, target, preset).exit_code == 0

    postures = {}
    for persona_file in sorted((target / "personas").iterdir()):
        parsed = yaml.safe_load(persona_file.read_text())
        assert "extends" not in parsed, persona_file.name
        # The big sections are inherited, not restated.
        for inherited_key in ("app_template", "provider", "model", "requires_osprey_version"):
            assert inherited_key not in parsed, (persona_file.name, inherited_key)
        postures[persona_file.stem] = parsed["config"]["control_system.writes_enabled"]
    assert postures == {"readonly": False, "readwrite": True}


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_host_profile_edits_are_not_shadowed_by_any_persona(
    runner: CliRunner, tmp_path: Path, preset: str
):
    """The point of the delta shape: edit the host profile once and every
    persona follows — no re-materialization, no hand-mirroring into the persona
    files. That holds because no persona file carries a competing value, which
    is what this pins; that the merge then applies the host's value is pinned by
    the implicit-delta resolver's own tests."""
    target = tmp_path / "my-facility"
    assert _new(runner, target, preset).exit_code == 0

    host_file = target / "profile.yml"
    text = host_file.read_text(encoding="utf-8")
    assert "provider: anthropic" in text
    host_file.write_text(text.replace("provider: anthropic", "provider: cborg"), encoding="utf-8")

    resolved, _dir = resolve_build_profile(host_file.resolve(), None)
    assert resolved.provider == "cborg"
    persona_files = sorted((target / "personas").iterdir())
    assert persona_files
    for persona_file in persona_files:
        assert "provider" not in yaml.safe_load(persona_file.read_text()), persona_file.name


@pytest.mark.parametrize("preset", ("hello-world", "ariel-standalone", *CHILD_PRESETS))
def test_non_trigger_presets_emit_no_personas_directory(
    runner: CliRunner, tmp_path: Path, preset: str
):
    target = tmp_path / "my-facility"

    assert _new(runner, target, preset).exit_code == 0

    assert not (target / "personas").exists()


def test_emitted_host_profile_builds_end_to_end(runner: CliRunner, tmp_path: Path) -> None:
    """The emitted host profile renders a project reading the facility data tree,
    verified after an edit to that tree. The persona projects read the same tree
    because the delta names none of its own — building one is the implicit
    resolver's job, and its own tests cover that half."""
    from osprey.cli.build_cmd import build

    target = tmp_path / "my-facility"
    assert _new(runner, target, "control-assistant").exit_code == 0

    # Edit the ONE facility data tree the whole stack reads.
    edited = target / "data" / "facility-marker.txt"
    edited.write_text("mark\n", encoding="utf-8")

    host = runner.invoke(build, ["--repo", str(target), "--skip-deps", "--skip-lifecycle"])
    assert host.exit_code == 0, host.output

    assert (target / "build" / "data" / "facility-marker.txt").read_bytes() == b"mark\n"
    for persona_file in sorted((target / "personas").iterdir()):
        assert "data" not in yaml.safe_load(persona_file.read_text()), persona_file.name


# ---------------------------------------------------------------------------
# Baked model selection reaches the personas too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_baked_model_selection_reaches_every_persona_by_inheritance(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """A provider/model chosen at ``osprey init`` time retints the WHOLE stack:
    baked once into the host profile and inherited by every persona through the
    implicit merge. Nothing is copied into the persona files — the host is the
    single place the choice lives, which is what makes one edit enough.

    ``tier`` travels too, unlike under the old replay allowlist: the stack
    shares one data tree, so a persona materializing a different
    channel-database tier than its host was an inconsistency, not a feature.
    """
    target = tmp_path / "my-facility"

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
    assert host["tier"] == 1
    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert (resolved.provider, resolved.model) == ("cborg", "opus")
    assert resolved.channel_finder_mode == "in_context"
    assert resolved.tier == 1
    persona_files = sorted((target / "personas").iterdir())
    assert persona_files  # the assertions below must not pass vacuously
    for persona_file in persona_files:
        parsed = yaml.safe_load(persona_file.read_text())
        for key in ("provider", "model", "channel_finder_mode", "tier"):
            assert key not in parsed, (persona_file.name, key)


def test_baked_override_file_model_selection_is_inherited_too(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A choice made in an ``-O`` override file bakes into the host profile and
    reaches every persona exactly as far as an inline one — it lands in the host
    and nowhere else."""
    override = tmp_path / "o.yml"
    override.write_text("provider: cborg\nmodel: opus\n", encoding="utf-8")
    target = tmp_path / "my-facility"

    assert _new(runner, target, "control-assistant", "-O", str(override)).exit_code == 0

    resolved, _dir = resolve_build_profile((target / "profile.yml").resolve(), None)
    assert (resolved.provider, resolved.model) == ("cborg", "opus")
    for persona_file in sorted((target / "personas").iterdir()):
        parsed = yaml.safe_load(persona_file.read_text())
        assert "provider" not in parsed and "model" not in parsed, persona_file.name


def test_persona_preset_outside_the_host_chain_is_rejected(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog entry whose preset does not extend the host's preset is not a
    delta over this profile: its own base would be silently dropped. There is no
    approximation to fall back to, so ``osprey init`` refuses and says what to
    do instead.

    No bundled preset shares control-assistant's app template from outside its
    extends chain, so the out-of-chain verdict is simulated for the readonly
    persona — its raw ``extends`` is repointed away from the host and the chain
    predicate pinned false. The branch stays defensive against exactly such a
    preset appearing.
    """
    from osprey.cli import build_profile_presets

    real_load = build_profile_presets._load_preset_raw
    real_reaches = build_profile_presets._preset_extends_chain_reaches

    def out_of_chain_raw(name: str):
        raw, path = real_load(name)
        if name == "control-assistant-readonly":
            raw = {**raw, "extends": "hello-world"}
        return raw, path

    def out_of_chain_for_readonly(child: str, ancestor: str) -> bool:
        if child == "control-assistant-readonly":
            return False
        return real_reaches(child, ancestor)

    monkeypatch.setattr(build_profile_presets, "_load_preset_raw", out_of_chain_raw)
    monkeypatch.setattr(
        build_profile_presets, "_preset_extends_chain_reaches", out_of_chain_for_readonly
    )
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant")

    assert result.exit_code == 2
    assert "readonly" in result.output
    assert "does not extend 'control-assistant'" in result.output
    assert not target.exists()  # fail-before-mutating


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
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", "--set", "app_template=hello_world")

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
        {bad_name: {"project": "x", "project_path": "../x", "build_profile": "control-assistant"}},
    )
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", "-O", str(override))

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
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", "-O", str(override))

    assert result.exit_code == 2
    assert "ghost" in result.output
    assert "does not resolve" in result.output
    assert not target.exists()


def test_persona_with_no_build_profile_is_rejected(runner: CliRunner, tmp_path: Path) -> None:
    override = _persona_override(tmp_path, {"bare": {"project": "b", "project_path": "../b"}})
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", "-O", str(override))

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
            "a/b": {"project": "x", "project_path": "../x", "build_profile": "control-assistant"},
            "ghost": {"project": "g", "project_path": "../g", "build_profile": "no-such-preset"},
        },
    )
    target = tmp_path / "my-facility"

    result = _new(runner, target, "control-assistant", "-O", str(override))

    assert result.exit_code == 2
    assert "a/b" in result.output
    assert "ghost" in result.output
    # One error, not two runs' worth: both problems under a single header.
    assert result.output.count("Cannot materialize the persona profiles") == 1
    assert not target.exists()


@pytest.mark.parametrize("preset", TRIGGER_PRESETS)
def test_the_build_renders_every_catalog_entry_the_emitter_wrote(
    runner: CliRunner, tmp_path: Path, preset: str, monkeypatch
) -> None:
    """The seam between the two halves of this feature: every persona the
    emitter puts in the catalog is one the build renders at the path that entry
    names, and the start path agrees.

    Both ends are real — a real `osprey init` writes the catalog and the deltas,
    a real `osprey build` renders them — because the defect this guards against
    lives between them: a catalog value and a render location derived
    separately would agree in a unit test of either one and disagree on disk.
    """
    from osprey.cli.build_cmd import build
    from osprey.deployment.web_terminals import persona_images
    from osprey.deployment.web_terminals.personas import resolve_personas
    from osprey.utils.config import ConfigBuilder

    target = tmp_path / "my-facility"
    assert _new(runner, target, preset).exit_code == 0
    result = runner.invoke(build, ["--repo", str(target), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output

    config = ConfigBuilder(str(target / "build" / "config.yml")).raw_config
    web_terminals = config["modules"]["web_terminals"]
    catalog = web_terminals["personas"]
    expected = set(persona_catalog(resolve_build_profile(None, preset)[0].config))
    assert set(catalog) == expected

    for name in expected:
        # The delta the emitter wrote, and the render the build made from it.
        assert (target / "personas" / f"{name}.yml").is_file()
        rendered = target / "build" / f"{target.name}-{name}"
        assert rendered.is_dir(), f"{name} was never rendered"
        # The catalog value is repo-relative and resolves to exactly that.
        assert (target / catalog[name]["project_path"]).resolve() == rendered.resolve()
        assert catalog[name]["build_profile"] == f"personas/{name}.yml"

    # And the start path, which reads the catalog rather than the deltas, finds
    # every one of them. `up` chdirs into the repo before provisioning.
    monkeypatch.chdir(target)
    users = resolve_personas(web_terminals, config.get("registry", {}), "test", strict=False)
    persona_images.verify_persona_renders(config, users, repo_root=target)


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
        target = tmp_path / "my-facility"
        assert _new(runner, target, "control-assistant").exit_code == 0

        assert not (target / "data" / "benchmarks" / "results").exists()
        # The sibling staging dir the bundle really ships still comes across.
        assert (target / "data" / "benchmarks" / "cross_paradigm").is_dir()
    finally:
        exhaust.unlink(missing_ok=True)
        if created_dir:
            shutil.rmtree(results, ignore_errors=True)
