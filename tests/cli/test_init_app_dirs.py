"""``osprey init`` copies the applications a preset lists into ``apps/<name>/``.

A preset's ``apps:`` block names applications whose directories sit beside the
preset as ``apps/<name>/``. ``init`` copies each named one into the new repo as
deployment source, drops the byte-code a source checkout leaves in it, and
writes a ``.provenance`` sidecar inside it: the preset it came from, that
preset's hash, and the digest of the tree as copied. ``osprey validate`` and
``osprey init --revalidate`` compare that digest with the tree on disk, skipping
the files the build itself writes into the directory, so a build never reads as
an edit. A directory the preset does not list is not copied.
"""

from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile_presets
from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_drift import (
    APP_PROVENANCE_FILENAME,
    app_tree_digest,
    preset_drift_report,
)
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.init_cmd import init
from osprey.cli.profile_conventions import unselected_convention_artifacts

pytestmark = pytest.mark.usefixtures("container_runtime_is_up")

PRESET = "knob-demo"
APP = "knob"


def _write_yaml(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


@pytest.fixture
def app_preset(fake_presets: Path) -> Path:
    """The bundled presets plus ``knob-demo``, which lists the application ``knob``.

    The bundled tree is copied first so ``extends: control-assistant`` resolves;
    ``other`` sits beside ``knob`` and is listed by no preset. ``knob`` carries the
    byte-code a source checkout leaves once tests have run.
    """
    bundled = Path(str(importlib.resources.files(build_profile_presets._PRESETS_PACKAGE)))
    shutil.copytree(
        bundled, fake_presets, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True
    )

    _write_yaml(
        fake_presets / f"{PRESET}.yml",
        {"name": "Knob Demo", "extends": "control-assistant", "apps": {APP: {}}},
    )
    knob = fake_presets / "apps" / APP
    _write_yaml(knob / "app.yml", {"name": APP, "containers": {APP: {"runtime": "osprey"}}})
    (knob / "panel").mkdir()
    (knob / "panel" / "index.html").write_text("<p>knob</p>\n", encoding="utf-8")
    (knob / "__pycache__").mkdir()
    (knob / "__pycache__" / "tool.cpython-311.pyc").write_bytes(b"\0bytecode")
    (knob / "stray.pyc").write_bytes(b"\0bytecode")
    _write_yaml(
        fake_presets / "apps" / "other" / "app.yml",
        {"name": "other", "containers": {"other": {"runtime": "osprey"}}},
    )
    return fake_presets


def _init(runner: CliRunner, target: Path, *extra: str):
    return runner.invoke(
        init, [str(target), "--preset", PRESET, "--no-git", *extra], catch_exceptions=False
    )


@pytest.fixture
def repo(app_preset: Path, runner: CliRunner, tmp_path: Path) -> Path:  # noqa: ARG001
    target = tmp_path / "site"
    result = _init(runner, target)
    assert result.exit_code == 0, result.output
    return target


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_extends_carries_no_application(repo: Path, tmp_path: Path) -> None:
    """An application is init material: ``extends:`` of its preset runs none of them.

    The profile init writes lists and holds what it runs; a profile that extends
    the same preset and holds nothing runs nothing, with no refusal.
    """
    assert set(yaml.safe_load((repo / "profile.yml").read_text())["apps"]) == {APP}

    extending = tmp_path / "extending" / "profile.yml"
    _write_yaml(extending, {"name": "Extending", "extends": PRESET, "data": "data"})
    (extending.parent / "data").mkdir()
    assert resolve_build_profile(extending, None)[0].apps == {}
    assert resolve_build_profile(None, PRESET)[0].apps == {}


def test_a_login_file_keeps_its_selection(app_preset: Path) -> None:
    """A login's ``apps:`` says what its card shows; it is not the preset-side key."""
    selection = {"apps": {APP: {"shown": False}}}
    _write_yaml(app_preset / "tiers" / "narrow.yml", selection)
    assert build_profile_presets._load_preset_raw("tiers/narrow")[0] == selection


def test_init_copies_the_listed_application_without_byte_code(app_preset: Path, repo: Path) -> None:
    copied = _files(repo / "apps" / APP)
    copied.pop(APP_PROVENANCE_FILENAME)

    assert copied == {
        "app.yml": (app_preset / "apps" / APP / "app.yml").read_bytes(),
        "panel/index.html": b"<p>knob</p>\n",
    }


def test_an_application_the_preset_does_not_list_is_not_copied(repo: Path) -> None:
    assert sorted(path.name for path in (repo / "apps").iterdir()) == [APP]


def test_the_sidecar_names_the_preset_its_hash_and_the_tree(repo: Path) -> None:
    sidecar = yaml.safe_load((repo / "apps" / APP / APP_PROVENANCE_FILENAME).read_text())

    assert sidecar == {
        "preset": PRESET,
        "preset_hash": compute_preset_hash(PRESET),
        "tree_digest": app_tree_digest(repo / "apps" / APP),
    }


def test_the_tree_digest_ignores_byte_code(app_preset: Path, repo: Path) -> None:
    source = app_preset / "apps" / APP
    shutil.rmtree(source / "__pycache__")
    (source / "stray.pyc").unlink()

    assert app_tree_digest(source) == app_tree_digest(repo / "apps" / APP)


def _provenance(repo: Path):
    from osprey.cli.build_profile_schema import ProfileProvenance

    block = yaml.safe_load((repo / "profile.yml").read_text())["provenance"]
    return ProfileProvenance(preset=block["preset"], preset_hash=block["preset_hash"])


def _app_findings(repo: Path):
    report = preset_drift_report(repo / "profile.yml", _provenance(repo))
    return [f for f in report.findings if f.subject.startswith("apps/")]


def test_an_unedited_application_reports_no_drift(repo: Path) -> None:
    assert _app_findings(repo) == []


def test_an_edited_application_is_a_drift_note(repo: Path) -> None:
    (repo / "apps" / APP / "panel" / "index.html").write_text("<p>mine</p>\n", encoding="utf-8")

    (finding,) = _app_findings(repo)

    assert finding.subject == f"apps/{APP}/"
    assert finding.kind == "value"
    assert not finding.marked


def test_a_marker_in_the_sidecar_claims_the_edit(repo: Path) -> None:
    (repo / "apps" / APP / "panel" / "index.html").write_text("<p>mine</p>\n", encoding="utf-8")
    sidecar = repo / "apps" / APP / APP_PROVENANCE_FILENAME
    sidecar.write_text(
        f"# DEVIATION: apps/{APP} carries our own panel\n" + sidecar.read_text(),
        encoding="utf-8",
    )

    (finding,) = _app_findings(repo)

    assert finding.marked


def test_files_the_build_writes_into_the_application_are_not_drift(repo: Path) -> None:
    app_dir = repo / "apps" / APP
    manifest = yaml.safe_load((app_dir / "app.yml").read_text())
    manifest["vendor"] = [{"url": "https://example.org/lib.js", "into": "static/lib.js"}]
    sidecar = yaml.safe_load((app_dir / APP_PROVENANCE_FILENAME).read_text())
    _write_yaml(app_dir / "app.yml", manifest)
    # The sidecar is re-stated for the edited manifest, so only the build's
    # writes below can move the digest.
    sidecar["tree_digest"] = app_tree_digest(app_dir)
    _write_yaml(app_dir / APP_PROVENANCE_FILENAME, sidecar)

    (app_dir / "panel" / "vendor").mkdir()
    (app_dir / "panel" / "vendor" / "d3.js").write_text("// d3\n", encoding="utf-8")
    (app_dir / "static").mkdir()
    (app_dir / "static" / "lib.js").write_text("// lib\n", encoding="utf-8")

    assert app_tree_digest(app_dir) == sidecar["tree_digest"]
    assert _app_findings(repo) == []


def test_revalidate_reports_an_edited_application_and_leaves_it(
    repo: Path, runner: CliRunner
) -> None:
    edited = repo / "apps" / APP / "panel" / "index.html"
    edited.write_text("<p>mine</p>\n", encoding="utf-8")

    result = runner.invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert f"apps/{APP}/" in result.output
    assert edited.read_text(encoding="utf-8") == "<p>mine</p>\n"


def test_revalidate_is_quiet_about_an_unedited_application(repo: Path, runner: CliRunner) -> None:
    result = runner.invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert f"apps/{APP}/" not in result.output


def test_a_re_init_without_force_refuses_and_keeps_the_application(
    repo: Path, runner: CliRunner
) -> None:
    edited = repo / "apps" / APP / "panel" / "index.html"
    edited.write_text("<p>mine</p>\n", encoding="utf-8")

    result = _init(runner, repo)

    assert result.exit_code == 2
    assert "--force" in result.output
    assert edited.read_text(encoding="utf-8") == "<p>mine</p>\n"


def test_force_replaces_the_application_with_the_presets_copy(
    repo: Path, runner: CliRunner
) -> None:
    (repo / "apps" / APP / "panel" / "index.html").write_text("<p>mine</p>\n", encoding="utf-8")
    (repo / "apps" / APP / "mine.txt").write_text("mine\n", encoding="utf-8")

    result = _init(runner, repo, "--force")

    assert result.exit_code == 0, result.output
    assert (repo / "apps" / APP / "panel" / "index.html").read_text() == "<p>knob</p>\n"
    assert not (repo / "apps" / APP / "mine.txt").exists()


def test_the_copied_application_is_not_a_convention_artifact(repo: Path) -> None:
    assert not [
        name for name in unselected_convention_artifacts(repo, ()) if name.startswith("apps")
    ]


def test_vendor_files_are_ignored_by_git(repo: Path) -> None:
    assert "\n/apps/*/panel/vendor/\n" in (repo / ".gitignore").read_text(encoding="utf-8")
