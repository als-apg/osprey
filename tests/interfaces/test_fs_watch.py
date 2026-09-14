"""Tests for the watchdog primitives the interface file watchers share."""

from __future__ import annotations

import os
import threading
import time

import pytest

from osprey.interfaces.fs_watch import (
    DEFAULT_RECONCILE_SECONDS,
    ChangeStamp,
    Reconciler,
    entry_stamp,
    evict_subtree,
    one_level_listing,
    reconcile_interval_seconds,
)

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


@pytest.mark.unit
class TestStampingOneEntry:
    """A stamp taken of a path and a stamp taken by listing its parent describe
    the same file identically — otherwise a change recorded through one could
    not be diffed against the other."""

    def test_a_files_stamp_is_the_one_its_parents_listing_holds(self, tmp_path):
        note = tmp_path / "note.txt"
        note.write_text("hello")

        assert entry_stamp(note) == one_level_listing(tmp_path)["note.txt"]

    def test_a_directory_stamps_as_one(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()

        stamp = entry_stamp(sub)

        assert stamp is not None
        assert stamp[0] is True
        assert stamp == one_level_listing(tmp_path)["sub"]

    def test_a_path_that_is_not_there_has_no_stamp(self, tmp_path):
        assert entry_stamp(tmp_path / "never_existed") is None

    def test_a_rewrite_moves_the_stamp(self, tmp_path):
        note = tmp_path / "note.txt"
        note.write_text("hello")
        before = entry_stamp(note)

        time.sleep(0.01)
        note.write_text("a different length entirely")

        assert entry_stamp(note) != before


@pytest.mark.unit
class TestTheReconciliationTimer:
    """The second trigger: it keeps running, it stops on demand, and one bad
    pass does not end it."""

    def test_the_pass_runs_repeatedly(self):
        ran = threading.Semaphore(0)
        reconciler = Reconciler(0.01, ran.release)
        reconciler.start()
        try:
            for _ in range(3):
                assert ran.acquire(timeout=5), "the pass stopped running"
        finally:
            reconciler.stop()

    def test_stop_ends_the_thread(self):
        reconciler = Reconciler(0.01, lambda: None)
        reconciler.start()
        thread = reconciler._thread

        reconciler.stop()

        assert thread is not None
        assert not thread.is_alive()

    def test_a_second_stop_is_a_no_op(self):
        reconciler = Reconciler(0.01, lambda: None)
        reconciler.start()
        reconciler.stop()

        reconciler.stop()  # must not raise

    def test_stopping_one_that_never_started_is_a_no_op(self):
        Reconciler(0.01, lambda: None).stop()  # must not raise

    def test_a_pass_that_raises_does_not_end_the_loop(self):
        calls = threading.Semaphore(0)

        def explode() -> None:
            calls.release()
            raise RuntimeError("one scandir went wrong")

        reconciler = Reconciler(0.01, explode)
        reconciler.start()
        try:
            for _ in range(3):
                assert calls.acquire(timeout=5), "the loop ended on the first failure"
        finally:
            reconciler.stop()

    def test_the_interval_is_readable(self):
        assert Reconciler(1.5, lambda: None).interval == 1.5


@pytest.mark.unit
class TestTheConfiguredInterval:
    """One setting for both watchers, and no value that switches the pass off."""

    def test_no_config_at_all_answers_with_the_default(self):
        """The state every unit test runs in: nothing primed, nothing raised."""
        assert reconcile_interval_seconds() == DEFAULT_RECONCILE_SECONDS

    def test_a_configured_positive_number_is_honoured(self, monkeypatch):
        monkeypatch.setattr("osprey.utils.config.get_config_value", lambda key, default=None: 7.5)

        assert reconcile_interval_seconds() == 7.5

    def test_a_configured_integer_is_honoured_as_seconds(self, monkeypatch):
        monkeypatch.setattr("osprey.utils.config.get_config_value", lambda key, default=None: 5)

        assert reconcile_interval_seconds() == 5.0

    @pytest.mark.parametrize("configured", [0, -1, True, "2.0"])
    def test_anything_else_is_logged_and_the_default_kept(self, monkeypatch, caplog, configured):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value", lambda key, default=None: configured
        )

        with caplog.at_level("WARNING", logger="osprey.interfaces.fs_watch"):
            assert reconcile_interval_seconds() == DEFAULT_RECONCILE_SECONDS

        assert "web.file_watch_reconcile_interval_s" in caplog.text
