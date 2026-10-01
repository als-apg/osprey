"""The virtual-accelerator channel set of a tree that stages no channel database.

Such a tree's channels are the records of the facility file the build writes
at the root of the render, and the manifest step asks the channel roster for
them once that file is on disk -- whatever channel-finder mode the profile
selects. A build therefore answers from the render it is writing, never from
the one a previous build left behind: a first build from a clean ``build/``
gets a manifest, and a second one from a clean ``build/`` gets the same one.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.manifest.build import (
    MANIFEST_FILENAME,
    prepare_project_manifest,
)
from osprey.services.virtual_accelerator.manifest.paths import DEFAULT_TIER
from tests._facility_file import write_facility_file
from tests.cli.test_build_va_manifest_honesty import _graph_config, _graph_repo

_ADDRESSES = {
    "SR:MAG:HCM:01:CURRENT:SP",
    "SR:MAG:HCM:01:CURRENT:RB",
    "SR:DIAG:BPM:01:POSITION:X",
}


@pytest.fixture(autouse=True)
def _cold_roster_cache():
    """Every test reads its own facility file cold; none inherits another's read."""
    import osprey.channel_roster as channel_roster

    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _build(repo: Path) -> bytes:
    """Run ``osprey build`` in *repo* and return the manifest it wrote."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command

    previous = Path.cwd()
    os.chdir(repo)
    try:
        result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)
    assert result.exit_code == 0, result.output
    return (repo / "build" / "data" / "simulation" / MANIFEST_FILENAME).read_bytes()


def test_a_first_build_gets_a_manifest_and_a_second_clean_build_the_same_one(tmp_path):
    """No previous render is consulted, so a clean ``build/`` changes nothing."""
    repo = _graph_repo(tmp_path / "repo")
    assert not (repo / "build").exists()

    first = _build(repo)

    assert {c["address"] for c in json.loads(first)["channels"]} == _ADDRESSES

    shutil.rmtree(repo / "build")
    second = _build(repo)

    assert second == first


def test_a_stale_render_does_not_feed_the_manifest(tmp_path):
    """The roster is asked about the render being built, not the outgoing one."""
    repo = _graph_repo(tmp_path / "repo")
    _build(repo)
    # The outgoing render now states a channel the project's tree never did.
    write_facility_file(
        repo / "build",
        {"records/channels.yaml": [{"id": "SR:STALE:CHANNEL"}]},
    )

    rebuilt = json.loads(_build(repo))

    assert {c["address"] for c in rebuilt["channels"]} == _ADDRESSES


def test_the_facility_file_names_no_paradigm_as_its_source(tmp_path):
    """No paradigm database loaded, so none is named; the file is, by its own key."""
    root = _graph_repo(tmp_path / "repo", facility_file=True)

    prepared = prepare_project_manifest(root / "data", DEFAULT_TIER, config=_graph_config(root))

    metadata = prepared.manifest["_metadata"]
    assert metadata["source_paradigms"] == []
    assert metadata["absent_paradigms"] == []
    assert metadata["source_corpus"] == "facility.json"


def test_the_channel_finder_mode_does_not_decide_whether_the_roster_is_asked(tmp_path):
    """A config naming no channel-finder mode gets the facility file's channels too."""
    root = _graph_repo(tmp_path / "repo", facility_file=True)

    prepared = prepare_project_manifest(
        root / "data", DEFAULT_TIER, config={"config_dir": str(root)}
    )

    assert {c["address"] for c in prepared.manifest["channels"]} == _ADDRESSES


def test_a_build_in_another_channel_finder_mode_serves_the_facility_file_too(tmp_path):
    """Staging no database is what sends the build to the roster, not the mode."""
    repo = _graph_repo(tmp_path / "repo")
    profile = repo / "profile.yml"
    profile.write_text(
        profile.read_text().replace("channel_finder_mode: graph", "channel_finder_mode: in_context")
    )

    manifest = json.loads(_build(repo))

    assert {c["address"] for c in manifest["channels"]} == _ADDRESSES
    assert manifest["_metadata"]["source_corpus"] == "facility.json"
