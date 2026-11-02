"""``osprey init --revalidate`` moves the login files a repo keeps in ``apps/`` to ``tiers/``.

``apps/`` holds applications only. A deployment initialised before logins moved
carries its shared logins as ``apps/<name>.yml``; the rewrite moves each one to
``tiers/<name>.yml`` after its write-posture pass, so a moved file always says
``never``. One that was never edited — it hashes to what ``osprey init`` wrote
into ``apps/`` (frozen beside the move) or to what it writes into ``tiers/``
today — is re-emitted from the bundled login it names. An edited one keeps its
lines, gains ``lists: own`` and ``control_system.writes: never``, and the step
left to do by hand says how ``osprey validate`` reports its differences.

The released layout's path is retargeted the same way: a 2026.9 deployment's
logbook card in the retired layer directory becomes ``tiers/logbook.yml`` with
``lists: own``.

``tests/fixtures/apps_login_files/logbook.yml`` is ``apps/logbook.yml`` exactly as
``osprey init`` wrote it before the move.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from tests.fixtures.lifecycle_repo import build_exemplar_repo, exemplar_source_files

_FROZEN_APPS_LOGBOOK = (
    Path(__file__).resolve().parents[1] / "fixtures" / "apps_login_files" / "logbook.yml"
)


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)


def _revalidate(root: Path):
    return CliRunner().invoke(init, [str(root), "--revalidate"], catch_exceptions=False)


def _repo_with_logbook_in_apps(tmp_path: Path, text: str) -> Path:
    """The exemplar with its shared ``logbook`` login kept in ``apps/`` instead."""
    root = build_exemplar_repo(tmp_path / "als-exemplar")
    (root / "tiers" / "logbook.yml").unlink()
    (root / "apps").mkdir()
    (root / "apps" / "logbook.yml").write_text(text, encoding="utf-8")
    return root


def test_an_unedited_login_init_wrote_into_apps_is_re_emitted_in_tiers(tmp_path: Path) -> None:
    root = _repo_with_logbook_in_apps(tmp_path, _FROZEN_APPS_LOGBOOK.read_text(encoding="utf-8"))

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert not (root / "apps" / "logbook.yml").exists()
    assert (root / "tiers" / "logbook.yml").read_text(encoding="utf-8") == (
        exemplar_source_files()["tiers/logbook.yml"]
    )
    assert "Left for you to do by hand" not in result.output


def test_an_unedited_login_in_todays_text_is_re_emitted_in_tiers(tmp_path: Path) -> None:
    today = exemplar_source_files()["tiers/logbook.yml"]
    root = _repo_with_logbook_in_apps(tmp_path, today)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert not (root / "apps" / "logbook.yml").exists()
    assert (root / "tiers" / "logbook.yml").read_text(encoding="utf-8") == today


def test_an_edited_login_is_moved_as_written_with_lists_own_and_never(tmp_path: Path) -> None:
    edited = _FROZEN_APPS_LOGBOOK.read_text(encoding="utf-8").replace(
        "default_panel: ariel", "default_panel: workspace  # ours"
    )
    root = _repo_with_logbook_in_apps(tmp_path, edited)

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert not (root / "apps" / "logbook.yml").exists()
    moved = (root / "tiers" / "logbook.yml").read_text(encoding="utf-8")
    assert "default_panel: workspace  # ours" in moved
    data = yaml.safe_load(moved)
    assert data["lists"] == "own"
    assert data["config"]["control_system.writes"] == "never"
    steps = " ".join(result.output.split("Left for you to do by hand", 1)[1].split())
    assert "apps/logbook.yml has been edited, so it moved to tiers/logbook.yml" in steps
    assert "`# DEVIATION:`" in steps


def test_a_moved_login_that_spelled_no_tier_key_is_given_never(tmp_path: Path) -> None:
    root = _repo_with_logbook_in_apps(
        tmp_path, "name: Logbook Research\nconfig:\n  web.ui_mode: expert\n"
    )

    assert _revalidate(root).exit_code == 0

    data = yaml.safe_load((root / "tiers" / "logbook.yml").read_text(encoding="utf-8"))
    assert data["lists"] == "own"
    assert data["config"] == {"web.ui_mode": "expert", "control_system.writes": "never"}


def test_a_moved_login_whose_retired_keys_armed_says_never(tmp_path: Path) -> None:
    root = _repo_with_logbook_in_apps(
        tmp_path,
        "name: Logbook Research\nconfig:\n  control_system.writes_enabled: true\n",
    )

    assert _revalidate(root).exit_code == 0

    data = yaml.safe_load((root / "tiers" / "logbook.yml").read_text(encoding="utf-8"))
    assert data["config"]["control_system.writes"] == "never"
    assert "writes_enabled" not in (root / "tiers" / "logbook.yml").read_text(encoding="utf-8")


def test_a_login_in_apps_whose_tiers_name_is_taken_is_left_for_the_operator(
    tmp_path: Path,
) -> None:
    root = build_exemplar_repo(tmp_path / "als-exemplar")
    (root / "apps").mkdir()
    (root / "apps" / "logbook.yml").write_text("name: Mine\n", encoding="utf-8")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    assert (root / "apps" / "logbook.yml").read_text(encoding="utf-8") == "name: Mine\n"
    assert "tiers/logbook.yml already exists, so it was not moved" in " ".join(
        result.output.split()
    )


def test_the_move_leaves_application_directories_alone(tmp_path: Path) -> None:
    root = _repo_with_logbook_in_apps(tmp_path, _FROZEN_APPS_LOGBOOK.read_text(encoding="utf-8"))
    (root / "apps" / "knob").mkdir()
    (root / "apps" / "knob" / "app.yml").write_text("name: knob\n", encoding="utf-8")

    assert _revalidate(root).exit_code == 0

    assert (root / "apps" / "knob" / "app.yml").read_text(encoding="utf-8") == "name: knob\n"
    assert sorted(p.name for p in (root / "apps").iterdir()) == ["knob"]


def test_a_second_run_after_the_move_rewrites_nothing(tmp_path: Path) -> None:
    root = _repo_with_logbook_in_apps(tmp_path, _FROZEN_APPS_LOGBOOK.read_text(encoding="utf-8"))
    assert _revalidate(root).exit_code == 0
    before = (root / "tiers" / "logbook.yml").read_text(encoding="utf-8")

    second = _revalidate(root)

    assert second.exit_code == 0
    assert "nothing to rewrite" in second.output
    assert (root / "tiers" / "logbook.yml").read_text(encoding="utf-8") == before


def test_a_retired_logbook_card_becomes_the_shared_login_in_tiers(tmp_path: Path) -> None:
    """The released layout: a 2026.9 card the logbook preset made becomes ``tiers/``."""
    from tests.cli.test_init_revalidate import _previous_release_repo

    root = _previous_release_repo(tmp_path / "als-exemplar")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    data = yaml.safe_load((root / "tiers" / "logbook.yml").read_text(encoding="utf-8"))
    assert data["lists"] == "own"
    assert data["provenance"]["preset"] == "logbook"
    assert not (root / "apps").exists()


def test_a_landing_group_typed_apps_is_respelled_shared(tmp_path: Path) -> None:
    """``type: apps`` reads as ``shared`` for one release; the rewrite respells it."""
    root = build_exemplar_repo(tmp_path / "als-exemplar")
    profile = root / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count("- type: shared\n") == 1
    profile.write_text(text.replace("- type: shared\n", "- type: apps\n"), encoding="utf-8")

    result = _revalidate(root)

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(profile.read_text(encoding="utf-8"))["config"]
    groups = config["modules.web_terminals"]["landing"]["groups"]
    assert [g["type"] for g in groups if g.get("type") in ("apps", "shared")] == ["shared"]
    assert profile.read_text(encoding="utf-8") == text
