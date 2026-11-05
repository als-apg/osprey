"""A repo with no machine in it builds and validates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.main import cli
from osprey.facility.settings import NO_READBACK_WARNING, SERVED_PROBE_KEYS
from osprey.facility.views.facts import zero_source_facts
from tests._builds import init_machineless

PROJECT = "machineless"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_machineless(tmp_path, PROJECT)


def test_a_repo_with_no_facility_tree_builds(repo: Path) -> None:
    result = CliRunner().invoke(
        cli, ["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
    )

    assert result.exit_code == 0, result.output
    assert result.stderr.count(NO_READBACK_WARNING) == 1
    facts = json.loads((repo / "build" / "data" / "facility_facts.json").read_text("utf-8"))
    assert facts == zero_source_facts(facts["identity"])
    assert facts["identity"]["name"] == PROJECT

    config = yaml.safe_load((repo / "build" / "config.yml").read_text("utf-8"))
    for key in SERVED_PROBE_KEYS:
        node = config
        for part in key.split("."):
            node = (node or {}).get(part)
        assert node is None, key
    assert "archiver" not in ((config.get("health") or {}).get("categories") or {})

    facility = repo / "data" / "facility"
    assert [
        path.relative_to(facility).as_posix() for path in facility.rglob("*") if path.is_file()
    ] == ["knowledge/index.md"]


def test_a_repo_with_no_facility_tree_validates(repo: Path) -> None:
    result = CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])
    assert result.exit_code == 0, result.output
