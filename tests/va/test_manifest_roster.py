"""The virtual-accelerator channel set of a tree that stages no channel database.

Such a tree's channels are the records of the facility file the build writes
at the root of the render, and the manifest step asks the channel roster for
them once that file is on disk -- whatever channel-finder mode the profile
selects.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator.manifest.build import (
    prepare_project_manifest,
)
from osprey.services.virtual_accelerator.manifest.paths import DEFAULT_TIER
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
