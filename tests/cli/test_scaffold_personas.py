"""Emitting persona delta files into a repo that was built from another preset.

The case under test is the one ``osprey init`` cannot cover: a ``hello-world``
deployment that adopts ``control-assistant``'s web-terminal stack later. Init's
own emission path refuses that by design — it checks each persona preset against
the *host's* preset — so these tests pin that
:func:`osprey.cli.scaffold_personas.emit_persona_files` gets the five files
written anyway, and that every one of them parses as a delta.

The second half of the same adoption is
:func:`osprey.cli.scaffold_personas.repoint_persona_catalog`, which points the
pasted-in catalog at those files. Its tests pasted the preset's web-terminal
block into the repo first, because a catalog is exactly what a ``hello-world``
repo does not have.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.scaffold_personas import (
    PersonaReport,
    emit_persona_files,
    repoint_persona_catalog,
)
from osprey.errors import BuildProfileError

#: The preset whose catalog the personas come from. Deliberately NOT the preset
#: the repo under test was created with — that mismatch is the whole point.
PERSONA_PRESET = "control-assistant"

#: Every persona ``control-assistant`` catalogs, in the order the preset writes
#: them. Pinned rather than derived so a silent catalog change is a test
#: failure and not an agreement between the code and itself.
PERSONA_NAMES = ("readonly", "readwrite", "admin", "logbook", "knowledge")

#: The name the emitted headers title each persona with.
REPO_NAME = "Test Facility"


def init_hello_world_repo(target: Path) -> Path:
    """Create a bare ``hello-world`` deployment repo at *target*.

    A real ``osprey init`` rather than a hand-made directory: the emission
    writes into the repo's source zone beside a real ``profile.yml``, and the
    later catalog repoint reads that file.

    Returns:
        The repo root, for chaining.
    """
    from osprey.cli.init_cmd import init

    result = CliRunner().invoke(init, [str(target), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    return target


def persona_files(repo_root: Path) -> list[Path]:
    """Every file the emission could have written, in catalog order."""
    return [repo_root / "personas" / f"{name}.yml" for name in PERSONA_NAMES]


def parse_written_deltas(repo_root: Path) -> dict[str, object]:
    """Parse the emitted files the way the framework's one persona parse does.

    Reusing ``_parsed_persona_deltas`` rather than a bare ``yaml.safe_load`` is
    the point of the assertion: it is the check the emission itself runs, so a
    file that satisfies it is a file the rest of the toolchain will accept.
    """
    from osprey.cli.profile_cmd import _parsed_persona_deltas

    texts = {path.stem: path.read_text(encoding="utf-8") for path in persona_files(repo_root)}
    return dict(_parsed_persona_deltas(texts))


@pytest.fixture
def hello_world_repo(tmp_path: Path) -> Path:
    """A freshly created ``hello-world`` repo with no ``personas/`` directory."""
    return init_hello_world_repo(tmp_path / "hello-repo")


# ===================================================================
# First emission
# ===================================================================


def test_emit_writes_one_file_per_catalogued_persona(hello_world_repo: Path) -> None:
    report = emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    assert isinstance(report, PersonaReport)
    assert report.names == list(PERSONA_NAMES)
    assert report.written == [f"personas/{name}.yml" for name in PERSONA_NAMES]
    assert report.skipped == []
    assert [path.name for path in persona_files(hello_world_repo) if path.exists()] == [
        f"{name}.yml" for name in PERSONA_NAMES
    ]


def test_emit_produces_files_that_parse_as_persona_deltas(hello_world_repo: Path) -> None:
    """The parse is the only check the ``extends:`` line surgery left YAML."""
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    deltas = parse_written_deltas(hello_world_repo)

    assert sorted(deltas) == sorted(PERSONA_NAMES)
    assert all(delta for delta in deltas.values()), "a delta that parsed to nothing is not a delta"


def test_emit_writes_deltas_that_carry_no_extends_key(hello_world_repo: Path) -> None:
    """A file under ``personas/`` is merged over the host by position.

    An ``extends:`` left in would re-anchor every profile-relative path at the
    named preset instead of at the repo, which is the failure this shape exists
    to avoid.
    """
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    deltas = parse_written_deltas(hello_world_repo)

    assert all("extends" not in delta for delta in deltas.values())  # type: ignore[operator]


def test_emit_titles_every_persona_with_the_repo_name(hello_world_repo: Path) -> None:
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    deltas = parse_written_deltas(hello_world_repo)

    assert {name: delta["name"] for name, delta in deltas.items()} == {  # type: ignore[index]
        name: f"{REPO_NAME} ({name})" for name in PERSONA_NAMES
    }


def test_emit_from_a_preset_without_personas_writes_nothing(hello_world_repo: Path) -> None:
    """``hello-world`` catalogs no personas, so there is nothing to emit — and
    no empty ``personas/`` directory left behind to suggest otherwise."""
    report = emit_persona_files(hello_world_repo, REPO_NAME, "hello-world")

    assert report == PersonaReport(written=[], skipped=[], names=[])
    assert not (hello_world_repo / "personas").exists()


# ===================================================================
# Re-emission
# ===================================================================


def test_emit_skips_files_that_already_exist(hello_world_repo: Path) -> None:
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    report = emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    assert report.written == []
    assert report.skipped == [f"personas/{name}.yml" for name in PERSONA_NAMES]
    assert report.names == list(PERSONA_NAMES)


def test_emit_leaves_an_edited_persona_file_untouched(hello_world_repo: Path) -> None:
    """A file in the repo is the operator's; re-running must not revert it."""
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)
    edited = hello_world_repo / "personas" / "readonly.yml"
    edited.write_text("name: Edited By Hand\n", encoding="utf-8")

    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    assert edited.read_text(encoding="utf-8") == "name: Edited By Hand\n"


def test_emit_with_force_rewrites_every_persona_file(hello_world_repo: Path) -> None:
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)
    originals = {path: path.read_text(encoding="utf-8") for path in persona_files(hello_world_repo)}
    for path in originals:
        path.write_text("name: Tampered\n", encoding="utf-8")

    report = emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET, force=True)

    assert report.written == [f"personas/{name}.yml" for name in PERSONA_NAMES]
    assert report.skipped == []
    assert {path: path.read_text(encoding="utf-8") for path in originals} == originals


# ===================================================================
# Refusals — nothing reaches the repo
# ===================================================================


def test_emit_writes_nothing_when_one_delta_cannot_be_emitted(
    hello_world_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emission and parsing both finish before the first write, so a catalog
    with one bad entry leaves the repo exactly as it was."""
    from osprey.cli import build_profile_emit

    real_emit = build_profile_emit.emit_persona_delta_yaml

    def fail_on_the_third(preset_name: str, *args: object, **kwargs: object) -> str:
        if preset_name.endswith("-admin"):
            raise BuildProfileError("declares no extends chain")
        return real_emit(preset_name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(build_profile_emit, "emit_persona_delta_yaml", fail_on_the_third)

    with pytest.raises(BuildProfileError) as excinfo:
        emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    assert "'admin'" in str(excinfo.value)
    assert not (hello_world_repo / "personas").exists()


def test_emit_refuses_a_persona_name_that_is_not_a_file_name(
    hello_world_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name becomes a path under ``personas/``, so a separator in it would
    write outside the directory the operator asked about."""
    from osprey.cli import build_profile_emit

    monkeypatch.setattr(
        build_profile_emit,
        "persona_catalog",
        lambda config: {"../escape": {"build_profile": "control-assistant-readonly"}},
    )

    with pytest.raises(BuildProfileError, match="plain name"):
        emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    assert not (hello_world_repo / "personas").exists()


def test_emit_refuses_a_catalog_entry_with_no_build_profile(
    hello_world_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.cli import build_profile_emit

    monkeypatch.setattr(
        build_profile_emit, "persona_catalog", lambda config: {"readonly": {"project": "x"}}
    )

    with pytest.raises(BuildProfileError, match="build_profile"):
        emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)

    assert not (hello_world_repo / "personas").exists()


def test_emit_refuses_an_unknown_preset(hello_world_repo: Path) -> None:
    with pytest.raises(BuildProfileError):
        emit_persona_files(hello_world_repo, REPO_NAME, "no-such-preset")

    assert not (hello_world_repo / "personas").exists()


# ===================================================================
# Repointing the catalog at the emitted files
# ===================================================================

#: A comment nobody but this test writes, dropped into the repo's profile
#: before the rewrite. Its survival is the proof the write-back went through
#: the round-trip writer and not a plain dump.
SENTINEL_COMMENT = "# hand-written note that must survive the repoint"


def adopt_persona_catalog(repo_root: Path) -> None:
    """Paste ``control-assistant``'s web-terminal block into *repo_root*'s profile.

    The manual step this module exists to follow: an operator copies the block —
    roster, personas catalog and all — out of the preset into a repo that was
    built from something else. Taken from the preset's own authored ``config:``
    so the key keeps the dotted spelling the preset ships, which is the spelling
    the repoint has to survive beside.
    """
    from osprey.cli.build_profile_resolve import preset_authored_config
    from osprey.utils.config_writer import load_config_document, save_config_document

    block_key = "modules.web_terminals"
    block = preset_authored_config(PERSONA_PRESET)[block_key]

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    document["config"][block_key] = block
    document.yaml_set_start_comment(SENTINEL_COMMENT)
    save_config_document(profile_path, document)


def loaded_catalog(repo_root: Path) -> dict[str, dict[str, object]]:
    """The repo profile's persona catalog, read the way the CLI reads it."""
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.build_profile_emit import persona_catalog

    profile, _profile_dir = resolve_build_profile(repo_root / "profile.yml", None)
    return persona_catalog(profile.config)


@pytest.fixture
def adopting_repo(hello_world_repo: Path) -> Path:
    """A ``hello-world`` repo that has pasted the persona catalog in and emitted
    the five delta files, but whose catalog still names the bundled presets."""
    adopt_persona_catalog(hello_world_repo)
    emit_persona_files(hello_world_repo, REPO_NAME, PERSONA_PRESET)
    return hello_world_repo


def test_repoint_rewrites_every_catalogued_persona(adopting_repo: Path) -> None:
    assert repoint_persona_catalog(adopting_repo, list(PERSONA_NAMES)) == len(PERSONA_NAMES)

    catalog = loaded_catalog(adopting_repo)
    assert {name: catalog[name]["build_profile"] for name in PERSONA_NAMES} == {
        name: f"personas/{name}.yml" for name in PERSONA_NAMES
    }


def test_repoint_takes_the_names_the_emission_reported(adopting_repo: Path) -> None:
    """The command layer hands the report's names straight through."""
    report = emit_persona_files(adopting_repo, REPO_NAME, PERSONA_PRESET)

    assert repoint_persona_catalog(adopting_repo, report.names) == len(PERSONA_NAMES)


def test_repoint_keys_the_render_off_the_repo_directory(adopting_repo: Path) -> None:
    """``project`` and ``project_path`` name the deployment, not the preset —
    neither is knowable until the repo has a directory name."""
    repoint_persona_catalog(adopting_repo, list(PERSONA_NAMES))

    catalog = loaded_catalog(adopting_repo)
    assert catalog["readonly"]["project"] == f"{adopting_repo.name}-readonly"
    assert catalog["readonly"]["project_path"] == f"build/{adopting_repo.name}-readonly"


def test_repoint_is_idempotent(adopting_repo: Path) -> None:
    repoint_persona_catalog(adopting_repo, list(PERSONA_NAMES))
    after_first = (adopting_repo / "profile.yml").read_bytes()

    assert repoint_persona_catalog(adopting_repo, list(PERSONA_NAMES)) == 0
    assert (adopting_repo / "profile.yml").read_bytes() == after_first


def test_repoint_keeps_the_profile_comments(adopting_repo: Path) -> None:
    """The profile is a hand-edited document, so the write-back is an edit of
    it rather than a regeneration."""
    before = (adopting_repo / "profile.yml").read_text(encoding="utf-8")
    assert SENTINEL_COMMENT in before

    repoint_persona_catalog(adopting_repo, list(PERSONA_NAMES))

    after = (adopting_repo / "profile.yml").read_text(encoding="utf-8")
    assert SENTINEL_COMMENT in after
    assert "Which control system to talk to" in after


def test_repoint_skips_a_name_the_catalog_does_not_carry(adopting_repo: Path) -> None:
    """A name with no entry is nobody's error here — the command layer is the
    one that knows whether the operator should hear about it."""
    counted = repoint_persona_catalog(adopting_repo, [*PERSONA_NAMES, "not-in-the-catalog"])

    assert counted == len(PERSONA_NAMES)
    assert "not-in-the-catalog" not in loaded_catalog(adopting_repo)


def test_repoint_rewrites_only_the_names_it_was_given(adopting_repo: Path) -> None:
    assert repoint_persona_catalog(adopting_repo, ["readonly"]) == 1

    catalog = loaded_catalog(adopting_repo)
    assert catalog["readonly"]["build_profile"] == "personas/readonly.yml"
    assert catalog["admin"]["build_profile"] == "control-assistant-admin"


def test_repoint_on_a_repo_with_no_catalog_changes_nothing(hello_world_repo: Path) -> None:
    """``hello-world`` never pasted a catalog in, so there is nothing to point
    anywhere — and the profile is left byte-identical."""
    before = (hello_world_repo / "profile.yml").read_bytes()

    assert repoint_persona_catalog(hello_world_repo, list(PERSONA_NAMES)) == 0
    assert (hello_world_repo / "profile.yml").read_bytes() == before


# ===================================================================
# The verb — osprey scaffold personas
# ===================================================================


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def run_personas(runner: CliRunner, repo_root: Path, *args: str):
    """Invoke the verb against *repo_root* through the real CLI.

    ``--repo`` rather than a chdir: what the flag selects is part of what is
    under test, and it keeps the tests free of process-wide state.
    """
    from osprey.cli.main import cli

    return runner.invoke(cli, ["scaffold", "personas", "--repo", str(repo_root), *args])


def set_recorded_preset(repo_root: Path, preset: str | None) -> None:
    """Rewrite (or remove) the ``provenance:`` block's preset.

    ``osprey init`` always records one, so a profile without it has to be made:
    it is the hand-written case, where only the operator knows the preset.
    """
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    if preset is None:
        del document["provenance"]
    else:
        document["provenance"]["preset"] = preset
    save_config_document(profile_path, document)


def drop_catalog_entry(repo_root: Path, name: str) -> None:
    """Remove one persona from the repo's pasted-in catalog."""
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = repo_root / "profile.yml"
    document = load_config_document(profile_path)
    del document["config"]["modules.web_terminals"]["personas"][name]
    save_config_document(profile_path, document)


@pytest.fixture
def catalogued_repo(hello_world_repo: Path) -> Path:
    """A ``hello-world`` repo that has pasted the catalog in and emitted nothing.

    The state an operator is in the moment they finish the copy: the profile
    names five personas, every entry still points at a bundled preset, and
    ``personas/`` does not exist.
    """
    adopt_persona_catalog(hello_world_repo)
    return hello_world_repo


def test_the_verb_writes_every_persona_file_and_repoints_the_catalog(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    result = run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET)

    assert result.exit_code == 0, result.output
    assert [path.name for path in persona_files(catalogued_repo) if path.exists()] == [
        f"{name}.yml" for name in PERSONA_NAMES
    ]
    catalog = loaded_catalog(catalogued_repo)
    assert {name: catalog[name]["build_profile"] for name in PERSONA_NAMES} == {
        name: f"personas/{name}.yml" for name in PERSONA_NAMES
    }


def test_the_verb_reports_every_file_and_the_catalog_count(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    result = run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET)

    assert result.exit_code == 0, result.output
    for name in PERSONA_NAMES:
        assert f"personas/{name}.yml" in result.output
    assert f"catalog: {len(PERSONA_NAMES)} entries repointed" in result.output


def test_a_second_run_writes_nothing_and_leaves_the_catalog_alone(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    assert run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET).exit_code == 0
    unchanged = (catalogued_repo / "profile.yml").read_bytes()

    result = run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET)

    assert result.exit_code == 0, result.output
    assert result.output.count("exists, use --force") == len(PERSONA_NAMES)
    assert "catalog: already current" in result.output
    assert (catalogued_repo / "profile.yml").read_bytes() == unchanged


def test_force_rewrites_a_persona_file_the_operator_edited(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    assert run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET).exit_code == 0
    edited = catalogued_repo / "personas" / "readonly.yml"
    edited.write_text("name: Edited By Hand\n", encoding="utf-8")

    result = run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET, "--force")

    assert result.exit_code == 0, result.output
    assert "exists, use --force" not in result.output
    assert edited.read_text(encoding="utf-8") != "name: Edited By Hand\n"


def test_without_from_the_recorded_preset_is_used(runner: CliRunner, catalogued_repo: Path) -> None:
    """A repo materialized from the persona preset never has to name it twice."""
    set_recorded_preset(catalogued_repo, PERSONA_PRESET)

    result = run_personas(runner, catalogued_repo)

    assert result.exit_code == 0, result.output
    assert all(path.exists() for path in persona_files(catalogued_repo))


def test_without_from_and_without_provenance_the_flag_is_named(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    """Nothing records the preset, so the message has to say what to pass."""
    set_recorded_preset(catalogued_repo, None)

    result = run_personas(runner, catalogued_repo)

    assert result.exit_code != 0
    assert "--from" in result.output
    assert not (catalogued_repo / "personas").exists()


def test_a_profile_with_no_catalog_is_refused_by_name(
    runner: CliRunner, hello_world_repo: Path
) -> None:
    """``hello-world`` never pasted a catalog in, so the refusal names the block."""
    result = run_personas(runner, hello_world_repo, "--from", PERSONA_PRESET)

    assert result.exit_code != 0
    assert "modules.web_terminals.personas" in result.output
    assert not (hello_world_repo / "personas").exists()


def test_an_unknown_preset_is_refused_with_the_list_of_presets(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    result = run_personas(runner, catalogued_repo, "--from", "no-such-preset")

    assert result.exit_code != 0
    assert "no-such-preset" in result.output
    assert PERSONA_PRESET in result.output
    assert not (catalogued_repo / "personas").exists()


def test_a_persona_this_profile_does_not_catalog_is_named(
    runner: CliRunner, catalogued_repo: Path
) -> None:
    """The emission follows the preset's catalog and the repoint follows the
    repo's, so a persona in one and not the other gets a file nothing reads."""
    drop_catalog_entry(catalogued_repo, "admin")

    result = run_personas(runner, catalogued_repo, "--from", PERSONA_PRESET)

    assert result.exit_code == 0, result.output
    assert "admin" in result.output
    assert (catalogued_repo / "personas" / "admin.yml").exists()
    assert f"catalog: {len(PERSONA_NAMES) - 1} entries repointed" in result.output
