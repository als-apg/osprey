"""The repo-cleanliness guards judge what THIS run did, not what they found.

``no_agent_data_in_the_repo`` in the shared conftest guards fails a session that
left a mark in ``<repo>/var/agent_data``. What it compares is a snapshot taken
before any test, so a directory that was already there no longer buys the whole
session silence — only the entries this run added are a failure.
``no_writes_into_the_packaged_templates`` in ``tests/conftest.py`` does the same
for a whole tree: it snapshots every directory of the packaged templates and
names each directory where something appeared, went, or came and went. These
tests pin both comparisons, on directories of their own under ``tmp_path``; none
of them touches the checkout.
"""

import inspect
import os
import py_compile
import shutil
from pathlib import Path

import pytest

import osprey.templates
from osprey_connectors_tests._harness._repo_cleanliness import (
    DirectoryState,
    snapshot,
    tree_snapshot,
    what_this_run_did,
    what_this_run_did_to_tree,
)


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


def _a_tree(root):
    """A small nested tree: ``root/top.txt``, ``root/a/mid.txt``, ``root/a/b/deep.txt``."""
    (root / "a" / "b").mkdir(parents=True)
    (root / "top.txt").write_text("top")
    (root / "a" / "mid.txt").write_text("mid")
    (root / "a" / "b" / "deep.txt").write_text("deep")


def test_an_untouched_tree_reports_nothing(tmp_path):
    """A tree the run only read is silent, which is every green run."""
    root = tmp_path / "templates"
    _a_tree(root)
    baseline = tree_snapshot(root)

    assert what_this_run_did_to_tree(baseline, root) == []


def test_a_file_added_deep_in_the_tree_names_its_directory(tmp_path):
    """A leak below the top level is named by the directory that holds it."""
    root = tmp_path / "templates"
    _a_tree(root)
    baseline = tree_snapshot(root)

    (root / "a" / "b" / "new.txt").write_text("leak")

    assert what_this_run_did_to_tree(baseline, root) == [f"added new.txt to {root / 'a' / 'b'}"]


def test_a_file_removed_from_the_tree_names_its_directory(tmp_path):
    """A shipped file a test deleted is as much a change to the tree as an added one."""
    root = tmp_path / "templates"
    _a_tree(root)
    baseline = tree_snapshot(root)

    (root / "a" / "mid.txt").unlink()

    assert what_this_run_did_to_tree(baseline, root) == [f"removed mid.txt from {root / 'a'}"]


def test_an_entry_that_came_and_went_in_a_subdirectory_names_it(tmp_path):
    """A write a test made and deleted again is reported by its directory's mtime.

    The entry is really created and unlinked, and the mtime is then moved by
    hand as well: a filesystem whose timestamps are coarse can leave the mtime
    where it was across both operations inside one tick, and the branch is what
    is pinned here, not the kernel's timestamp granularity.
    """
    root = tmp_path / "templates"
    _a_tree(root)
    baseline = tree_snapshot(root)
    then = baseline["a"].mtime_ns
    assert then is not None

    (root / "a" / "transient.txt").write_text("gone soon")
    (root / "a" / "transient.txt").unlink()
    os.utime(root / "a", ns=(then + 1_000_000_000,) * 2)

    assert what_this_run_did_to_tree(baseline, root) == [
        f"added and removed entries in {root / 'a'}"
    ]


def test_bytecode_caches_are_not_a_leak(tmp_path):
    """Importing a packaged module writes ``__pycache__``; that is not a test's write."""
    root = tmp_path / "templates"
    _a_tree(root)
    baseline = tree_snapshot(root)
    then = baseline["a"].mtime_ns
    assert then is not None

    cache = root / "a" / "__pycache__"
    cache.mkdir()
    (cache / "mid.cpython-313.pyc").write_bytes(b"")
    os.utime(root / "a", ns=(then + 1_000_000_000,) * 2)

    assert what_this_run_did_to_tree(baseline, root) == []


@pytest.mark.parametrize("cached_at_baseline", [False, True], ids=["born", "reborn"])
def test_a_template_module_byte_compiled_mid_session_is_not_a_leak(tmp_path, cached_at_baseline):
    """Byte-code churn beside a packaged module is never a test's write.

    Another worker compiles the module and a prune removes its cache again, in
    either order, so the directory can end the session with the same names it
    started with and a moved mtime. The module is a real template's source,
    copied, compiled and cleared under ``tmp_path``; the mtime is moved by hand
    for the same reason as in the came-and-went test above.
    """
    packaged = Path(osprey.templates.__file__).parent / "services" / "teams_bridge" / "relay"
    root = tmp_path / "templates"
    relay = root / "relay"
    shutil.copytree(packaged, relay, ignore=shutil.ignore_patterns("__pycache__"))
    module = relay / "validation.py"
    if cached_at_baseline:
        py_compile.compile(str(module), doraise=True)
    baseline = tree_snapshot(root)
    then = baseline["relay"].mtime_ns
    assert then is not None

    py_compile.compile(str(module), doraise=True)
    shutil.rmtree(relay / "__pycache__", ignore_errors=True)
    if cached_at_baseline:
        py_compile.compile(str(module), doraise=True)
    os.utime(relay, ns=(then + 1_000_000_000,) * 2)

    assert what_this_run_did_to_tree(baseline, root) == []


def test_a_missing_tree_is_empty_and_silent(tmp_path):
    """A tree that is not there has no directories and nothing to report."""
    absent = tmp_path / "no-templates"

    baseline = tree_snapshot(absent)

    assert baseline == {}
    assert what_this_run_did_to_tree(baseline, absent) == []


def test_the_template_guard_reads_the_tree_through_this_module():
    """The packaged-template guard has no walk of its own.

    A second walk would compare paths only and miss a write that came and went,
    which is the case ``what_this_run_did_to_tree`` exists to catch.
    """
    import tests.conftest as root_conftest

    guard = root_conftest.no_writes_into_the_packaged_templates
    source = inspect.getsource(getattr(guard, "__wrapped__", guard))

    assert "os.walk" not in source
    assert "what_this_run_did_to_tree" in source
    assert not hasattr(root_conftest, "_template_tree_paths")
