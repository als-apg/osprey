"""``osprey scaffold tiers`` and ``osprey scaffold apps``.

Each verb materializes the tier files or the app files a deployment preset's
web-terminal roster names into an existing repo, through the same pipeline
``osprey init`` runs. The contract pinned here:

* a fresh emission writes exactly the files init would, byte for byte;
* the default preset is the one the repo records in its provenance, and
  ``--from`` names another (the adoption case: a ``hello-world`` repo taking
  ``control-assistant``'s tiers);
* a file that already matches is left untouched, a file that differs is refused
  without ``--force`` — and one refusal writes nothing at all — and ``--force``
  replaces it;
* each verb touches only its own kind.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.scaffold_cmd import scaffold

#: The preset that carries a web-terminal roster.
ROSTER_PRESET = "control-assistant"

#: What ``control-assistant``'s roster names, pinned rather than derived so a
#: silent roster change is a test failure and not the code agreeing with itself.
TIERS = ("admin", "readonly", "readwrite")
APPS = ("channel-finder", "knowledge", "logbook")


def _init(target: Path, preset: str) -> Path:
    from osprey.cli.init_cmd import init

    result = CliRunner().invoke(init, [str(target), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output
    return target


def _run(*args: str):
    return CliRunner().invoke(scaffold, list(args))


@pytest.fixture(scope="module")
def roster_repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One ``control-assistant`` init, copied per test (init is slow)."""
    return _init(tmp_path_factory.mktemp("ca") / "repo", ROSTER_PRESET)


@pytest.fixture
def roster_repo(roster_repo_template: Path, tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / "repo"
    shutil.copytree(roster_repo_template, target, symlinks=True)
    return target


@pytest.fixture
def hello_repo(tmp_path: Path) -> Path:
    return _init(tmp_path / "hello", "hello-world")


def _files(repo: Path, kind: str, names: tuple[str, ...]) -> list[Path]:
    return [repo / kind / f"{name}.yml" for name in names]


# ===================================================================
# Byte equality with init
# ===================================================================


@pytest.mark.parametrize(("kind", "names"), [("tiers", TIERS), ("apps", APPS)])
def test_rerun_on_an_init_repo_is_unchanged(roster_repo: Path, kind: str, names) -> None:
    """Init's files and this verb's are the same pipeline: nothing to rewrite."""
    before = {p: p.read_bytes() for p in _files(roster_repo, kind, names)}

    result = _run(kind, "--repo", str(roster_repo))

    assert result.exit_code == 0, result.output
    for name in names:
        assert f"{kind}/{name}.yml (unchanged)" in result.output
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize(("kind", "names"), [("tiers", TIERS), ("apps", APPS)])
def test_deleted_files_come_back_byte_identical(roster_repo: Path, kind: str, names) -> None:
    before = {p: p.read_bytes() for p in _files(roster_repo, kind, names)}
    for path in before:
        path.unlink()

    result = _run(kind, "--repo", str(roster_repo))

    assert result.exit_code == 0, result.output
    for name in names:
        assert f"{kind}/{name}.yml (created)" in result.output
    assert {p: p.read_bytes() for p in before} == before


def test_tiers_verb_leaves_apps_alone_and_vice_versa(roster_repo: Path) -> None:
    for path in _files(roster_repo, "tiers", TIERS) + _files(roster_repo, "apps", APPS):
        path.unlink()

    assert _run("tiers", "--repo", str(roster_repo)).exit_code == 0
    assert all(p.exists() for p in _files(roster_repo, "tiers", TIERS))
    assert not any(p.exists() for p in _files(roster_repo, "apps", APPS))

    assert _run("apps", "--repo", str(roster_repo)).exit_code == 0
    assert all(p.exists() for p in _files(roster_repo, "apps", APPS))


# ===================================================================
# Overwrite refusal and --force
# ===================================================================


@pytest.mark.parametrize("kind", ["tiers", "apps"])
def test_an_edited_file_is_refused_and_nothing_is_written(roster_repo: Path, kind: str) -> None:
    names = TIERS if kind == "tiers" else APPS
    edited, missing = _files(roster_repo, kind, names)[:2]
    edited.write_text(edited.read_text(encoding="utf-8") + "# my edit\n", encoding="utf-8")
    missing.unlink()

    result = _run(kind, "--repo", str(roster_repo))

    assert result.exit_code == 1, result.output
    assert f"{kind}/{edited.name} not written" in result.output
    assert "--force" in result.output
    assert edited.read_text(encoding="utf-8").endswith("# my edit\n")
    assert not missing.exists(), "one refusal must stop the whole emission"


@pytest.mark.parametrize("kind", ["tiers", "apps"])
def test_force_replaces_an_edited_file(roster_repo: Path, kind: str) -> None:
    names = TIERS if kind == "tiers" else APPS
    edited = _files(roster_repo, kind, names)[0]
    original = edited.read_bytes()
    edited.write_text("name: mine\n", encoding="utf-8")

    result = _run(kind, "--repo", str(roster_repo), "--force")

    assert result.exit_code == 0, result.output
    assert f"{kind}/{edited.name} (replaced)" in result.output
    assert edited.read_bytes() == original


# ===================================================================
# --from: another preset's tiers/apps into a repo that has none
# ===================================================================


def test_from_preset_adopts_tiers_into_a_hello_world_repo(hello_repo: Path) -> None:
    result = _run("tiers", "--repo", str(hello_repo), "--from", ROSTER_PRESET)

    assert result.exit_code == 0, result.output
    written = _files(hello_repo, "tiers", TIERS)
    assert all(p.exists() for p in written)
    assert not (hello_repo / "apps").exists(), "the tiers verb must not create apps/"
    for path in written:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict)
        assert "extends" not in doc
        assert doc["provenance"]["preset"] == path.stem


def test_from_preset_adopts_apps_into_a_hello_world_repo(hello_repo: Path) -> None:
    result = _run("apps", "--repo", str(hello_repo), "--from", ROSTER_PRESET)

    assert result.exit_code == 0, result.output
    for path in _files(hello_repo, "apps", APPS):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "extends" not in doc
        assert doc["provenance"]["preset"] == path.stem
    assert not (hello_repo / "tiers").exists()


@pytest.mark.parametrize(("kind", "names"), [("tiers", TIERS), ("apps", APPS)])
def test_adopted_files_this_roster_does_not_name_are_flagged(
    hello_repo: Path, kind: str, names
) -> None:
    """Written, but nothing deploys them until this repo's roster names them.

    Without the hint an operator who adopted another preset's files could take
    them for live.
    """
    result = _run(kind, "--repo", str(hello_repo), "--from", ROSTER_PRESET)

    assert result.exit_code == 0, result.output
    assert "not named by this repo's roster" in result.output
    for name in names:
        assert f"{kind}/{name}.yml" in result.output.split("not named by this repo's roster")[1]


@pytest.mark.parametrize(("kind", "names"), [("tiers", TIERS), ("apps", APPS)])
def test_files_this_roster_names_are_not_flagged(roster_repo: Path, kind: str, names) -> None:
    for path in _files(roster_repo, kind, names):
        path.unlink()

    result = _run(kind, "--repo", str(roster_repo))

    assert result.exit_code == 0, result.output
    assert "not named by this repo's roster" not in result.output


def test_only_the_unnamed_files_are_flagged(hello_repo: Path) -> None:
    """A roster that names some of the adopted apps flags only the rest."""
    from osprey.utils.config_writer import load_config_document, save_config_document

    profile_path = hello_repo / "profile.yml"
    document = load_config_document(profile_path)
    document["config"]["modules.web_terminals"] = {
        "enabled": True,
        "default_tier": "readonly",
        "users": [],
        "shared": ["knowledge"],
    }
    save_config_document(profile_path, document)

    result = _run("apps", "--repo", str(hello_repo), "--from", ROSTER_PRESET)

    assert result.exit_code == 0, result.output
    flagged = result.output.split("not named by this repo's roster")[1]
    assert "apps/logbook.yml" in flagged
    assert "apps/channel-finder.yml" in flagged
    assert "apps/knowledge.yml" not in flagged


@pytest.mark.parametrize("kind", ["tiers", "apps"])
def test_a_preset_without_a_roster_has_nothing_to_emit(hello_repo: Path, kind: str) -> None:
    """The default preset of a hello-world repo names no tier and no app."""
    result = _run(kind, "--repo", str(hello_repo))

    assert result.exit_code != 0
    assert "nothing to emit" in result.output
    assert not (hello_repo / kind).exists()


def test_an_unknown_preset_is_a_usage_error(hello_repo: Path) -> None:
    result = _run("tiers", "--repo", str(hello_repo), "--from", "no-such-preset")

    assert result.exit_code == 2, result.output
    assert not (hello_repo / "tiers").exists()


# ===================================================================
# The retired verb
# ===================================================================


def test_the_scaffold_group_offers_tiers_and_apps() -> None:
    from osprey.cli.scaffold_cmd import scaffold as group

    assert {"tiers", "apps"} <= set(group.commands)
