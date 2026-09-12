"""Tests for the watchdog primitives the interface file watchers share."""

from __future__ import annotations

import os

import pytest

from osprey.interfaces.fs_watch import ChangeStamp, evict_subtree

_STAMP: ChangeStamp = (False, 0, 0)


def _listing() -> dict[str, ChangeStamp]:
    return {"note.txt": _STAMP}


@pytest.mark.unit
class TestEvictingADirectorysSubtree:
    """A path that is no longer in the tree keeps no listing, and neither does
    anything that was under it."""

    def test_the_directorys_own_listing_goes(self, tmp_path):
        listings = {str(tmp_path / "sub"): _listing()}

        evict_subtree(listings, tmp_path / "sub")

        assert listings == {}

    def test_a_child_and_a_grandchild_go_with_it(self, tmp_path):
        gone = tmp_path / "sub"
        listings = {
            str(gone): _listing(),
            str(gone / "child"): _listing(),
            str(gone / "child" / "grandchild"): _listing(),
        }

        evict_subtree(listings, gone)

        assert listings == {}

    def test_a_sibling_that_merely_shares_a_prefix_stays(self, tmp_path):
        """``/a/bc`` is not under ``/a/b`` — the prefix carries the separator."""
        neighbour = str(tmp_path / "subtle")
        listings = {str(tmp_path / "sub"): _listing(), neighbour: _listing()}

        evict_subtree(listings, tmp_path / "sub")

        assert list(listings) == [neighbour]

    def test_evicting_a_path_the_map_does_not_hold_is_a_no_op(self, tmp_path):
        kept = str(tmp_path / "sub")
        listings = {kept: _listing()}

        evict_subtree(listings, tmp_path / "never_listed")

        assert list(listings) == [kept]

    def test_the_separator_is_the_platforms_own(self, tmp_path):
        """The keys are ``str(Path)``, so the boundary is ``os.sep``."""
        gone = tmp_path / "sub"
        listings = {f"{gone}{os.sep}child": _listing()}

        evict_subtree(listings, gone)

        assert listings == {}
