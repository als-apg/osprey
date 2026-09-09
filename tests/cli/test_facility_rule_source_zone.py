"""The facility description lives in the source zone, not in ``build/``.

``build/`` is documented as 100% disposable — ``rm -rf build`` is a supported
thing to do to a deployment repo. So the one rendered artifact an operator is
expected to rewrite in their own words, ``rules/facility.md``, has to have a
copy somewhere else: the profile's own ``rules/`` convention directory, which
the build copies into ``build/.claude/rules/`` like any other profile rule.

These tests drive the real commands (``osprey init``, ``osprey build``) rather
than the render helpers, because the round trip is the claim: what init seeds,
what a build carries in, and what survives a wiped ``build/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init

BUILD_FLAGS = ["--skip-deps", "--skip-lifecycle"]

#: A preset whose `rules:` selection includes the facility rule. hello-world
#: deliberately leaves it out, and a deployment that does not select the rule
#: has no facility description to keep anywhere.
PRESET = "control-assistant"


def _init(repo: Path) -> None:
    result = CliRunner().invoke(init, [str(repo), "--preset", PRESET, "--no-git"])
    assert result.exit_code == 0, result.output


def _build(repo: Path) -> None:
    result = CliRunner().invoke(build, ["--repo", str(repo), *BUILD_FLAGS])
    assert result.exit_code == 0, result.output


def _source_rule(repo: Path) -> Path:
    return repo / "rules" / "facility.md"


def _built_rule(repo: Path) -> Path:
    return repo / "build" / ".claude" / "rules" / "facility.md"


@pytest.fixture(scope="module")
def initialized_repo(tmp_path_factory) -> Path:
    repo = tmp_path_factory.mktemp("facility-rule") / "deployment"
    _init(repo)
    return repo


def test_init_seeds_the_rule_in_the_source_zone(initialized_repo):
    """``osprey init`` writes the facility rule where an operator edits it."""
    rule = _source_rule(initialized_repo)
    assert rule.is_file(), (
        "osprey init must seed rules/facility.md in the repo; "
        f"repo holds {sorted(p.name for p in initialized_repo.iterdir())}"
    )
    assert "Facility Identity" in rule.read_text(encoding="utf-8")


def test_the_edited_rule_survives_a_wiped_build(tmp_path):
    """The operator's text comes back from the source zone after ``rm -rf build``."""
    repo = tmp_path / "deployment"
    _init(repo)
    _build(repo)

    edited = "# Ring Facility\n\nThe operator wrote this.\n"
    _source_rule(repo).write_text(edited, encoding="utf-8")
    shutil.rmtree(repo / "build")
    _build(repo)

    assert _built_rule(repo).read_text(encoding="utf-8") == edited


def test_a_build_only_rule_is_migrated_into_the_source_zone(tmp_path):
    """A repo from before the move keeps the text it only had in ``build/``.

    The deployment repos that exist today have no ``rules/`` directory and an
    edited ``build/.claude/rules/facility.md``. The next build has to rescue
    that file rather than render over it.
    """
    repo = tmp_path / "deployment"
    _init(repo)
    _build(repo)

    # The shape of a repo built before the rule moved: no source-zone copy, and
    # the operator's own text in the disposable zone.
    _source_rule(repo).unlink()
    custom = "# Legacy Facility\n\nEdited in build/ years ago.\n"
    _built_rule(repo).write_text(custom, encoding="utf-8")

    _build(repo)

    assert _source_rule(repo).read_text(encoding="utf-8") == custom
    assert _built_rule(repo).read_text(encoding="utf-8") == custom
