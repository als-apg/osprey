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
import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from tests._builds import strip_machine
from tests._preset_data import packaged_facility_dir
from tests.cli._lifecycle_build import invoke_build

#: A preset whose `rules:` selection includes the facility rule. hello-world
#: deliberately leaves it out, and a deployment that does not select the rule
#: has no facility description to keep anywhere.
PRESET = "control-assistant"


def _init(repo: Path) -> None:
    result = CliRunner().invoke(init, [str(repo), "--preset", PRESET, "--no-git"])
    assert result.exit_code == 0, result.output


def _build(repo: Path) -> None:
    result = invoke_build(repo)
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
    text = rule.read_text(encoding="utf-8")
    assert "data/facility_facts.md" in text
    assert _bundled_identity_name() not in text


def _bundled_identity_name() -> str:
    """The display name of the facility the preset names, read from the package."""
    identity = packaged_facility_dir("control_assistant") / "identity.yaml"
    return yaml.safe_load(identity.read_text(encoding="utf-8"))["name"]


def test_a_replaced_facility_leaves_no_stale_identity(tmp_path):
    """The rule is untouched by a facility swap and the facts page states the new identity."""
    repo = tmp_path / "deployment"
    _init(repo)
    _build(repo)
    seeded = _source_rule(repo).read_bytes()
    old_name = _bundled_identity_name()

    new_name = "Second Test Facility"
    strip_machine(repo)
    (repo / "data" / "facility" / "identity.yaml").write_text(
        yaml.safe_dump({"code": "stf", "name": new_name}), encoding="utf-8"
    )
    _build(repo)

    assert _source_rule(repo).read_bytes() == seeded
    for rule_path in (_source_rule(repo), _built_rule(repo)):
        rule = rule_path.read_text(encoding="utf-8")
        assert old_name not in rule
        assert new_name not in rule
    facts = (repo / "build" / "data" / "facility_facts.md").read_text(encoding="utf-8")
    assert f"# {new_name}\n" in facts
    assert "`stf`" in facts
    assert old_name not in facts


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
    edited ``build/.claude/rules/facility.md``. ``osprey init --revalidate`` has
    to rescue that file rather than let the next build render over it.
    """
    repo = tmp_path / "deployment"
    _init(repo)
    _build(repo)

    # The shape of a repo built before the rule moved: no source-zone copy, the
    # library rule selected by its bare name, and the operator's own text in the
    # disposable zone.
    _source_rule(repo).unlink()
    profile = repo / "profile.yml"
    profile.write_text(
        profile.read_text(encoding="utf-8").replace("  - rules/facility ", "  - facility "),
        encoding="utf-8",
    )
    custom = "# Legacy Facility\n\nEdited in build/ years ago.\n"
    _built_rule(repo).write_text(custom, encoding="utf-8")

    # `osprey init --revalidate` moves it into the profile and names it by its
    # path; the build carries it from there.
    revalidated = CliRunner().invoke(init, [str(repo), "--revalidate"])
    assert revalidated.exit_code == 0, revalidated.output
    assert "  - rules/facility" in profile.read_text(encoding="utf-8")
    _build(repo)

    assert _source_rule(repo).read_text(encoding="utf-8") == custom
    assert _built_rule(repo).read_text(encoding="utf-8") == custom


def test_the_seeder_reads_nothing_from_a_render():
    """Only ``osprey init`` seeds the description, and there is no render yet; a
    copy left in ``build/`` by an older repo is ``osprey init --revalidate``'s to
    rescue, so the seeder takes no render to read."""
    import inspect

    from osprey.cli.build_persistence import ensure_profile_facility_rule

    assert "build_dir" not in inspect.signature(ensure_profile_facility_rule).parameters
