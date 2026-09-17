"""The repo-cleanliness guard judges what THIS run did, not what it found.

``no_agent_data_in_the_repo`` in ``tests/conftest.py`` fails a session that left
a mark in ``<repo>/var/agent_data``. What it compares is a snapshot taken before
any test, so a directory that was already there no longer buys the whole session
silence — only the entries this run added are a failure. These tests pin that
comparison, on directories of their own under ``tmp_path``; none of them touches
``<repo>/var/agent_data``.
"""

import os

from tests._repo_cleanliness import DirectoryState, snapshot, what_this_run_did


def test_a_directory_this_run_created_is_named(tmp_path):
    """A directory that did not exist at the baseline and does now is the leak."""
    marker = tmp_path / "agent_data"
    baseline = snapshot(marker)

    marker.mkdir()

    assert what_this_run_did(baseline, marker) == f"created {marker}"


def test_a_directory_the_run_never_touched_is_not_reported(tmp_path):
    """A developer's own deployment directory is none of the suite's business.

    This is the exemption the guard has always had, and it still holds: the
    directory is there before the run and the run adds nothing to it.
    """
    marker = tmp_path / "agent_data"
    marker.mkdir()
    (marker / "a-real-deployment").mkdir()
    baseline = snapshot(marker)

    assert what_this_run_did(baseline, marker) is None


def test_an_entry_added_to_a_pre_existing_directory_is_named(tmp_path):
    """An entry this run put into a directory it found is still a leak.

    Under the existence rule this replaces, this case returned nothing at all —
    and a directory left behind by a previous interrupted run is exactly this
    shape, so the first leak disarmed the guard for every run after it.
    """
    marker = tmp_path / "agent_data"
    marker.mkdir()
    (marker / "a-real-deployment").mkdir()
    baseline = snapshot(marker)

    (marker / "artifacts").mkdir()

    assert what_this_run_did(baseline, marker) == f"added artifacts to {marker}"


def test_an_entry_that_came_and_went_is_reported_without_a_name(tmp_path):
    """A mark made and unmade before anyone looked is reported by its mtime.

    The mtime is moved by hand rather than by a real create-and-delete because
    a filesystem whose timestamps are coarse can leave the mtime where it was
    across both operations inside one tick, which would make a real
    create-and-delete a flaky way to reach this branch. The branch is what is
    pinned here, not the kernel's timestamp granularity.
    """
    marker = tmp_path / "agent_data"
    marker.mkdir()
    baseline = snapshot(marker)
    assert baseline.mtime_ns is not None

    os.utime(marker, ns=(baseline.mtime_ns + 1_000_000_000,) * 2)

    assert what_this_run_did(baseline, marker) == f"added and removed entries in {marker}"


def test_a_directory_that_is_still_missing_is_not_reported(tmp_path):
    """The guard is silent in every green run, where the directory never appears."""
    marker = tmp_path / "agent_data"
    baseline = snapshot(marker)

    assert what_this_run_did(baseline, marker) is None


def test_an_unreadable_directory_reads_as_absent(tmp_path):
    """A directory that cannot be read is absent, not a leak.

    Both ``except OSError`` branches return this sentinel, which is what keeps
    a permission error from being reported as something the run did.
    """
    assert snapshot(tmp_path / "nothing-here") == DirectoryState(exists=False)
