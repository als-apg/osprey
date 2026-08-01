"""Tests for the xdist scheduler override in tests/conftest.py.

`FileOrGroupScheduling` overrides `_split_scope`, an underscore method of
xdist's scheduler extension point. These tests pin both its grouping behaviour
and the fact that upstream still routes through that method, so an xdist
upgrade that shifts the seam fails loudly instead of silently un-pinning the
suite.
"""

import pytest

from tests.conftest import FileOrGroupScheduling, pytest_xdist_make_scheduler

pytest.importorskip("xdist", reason="scheduler override only applies when xdist is installed")

from xdist.scheduler.loadscope import LoadScopeScheduling  # noqa: E402


class _FakeConfig:
    """Minimal pytest.Config stand-in for scheduler construction.

    The scheduler's __init__ only reads the `tx` value (to count nodes); the
    hook only reads the `dist` option.
    """

    def __init__(self, dist: str, tx: int = 2):
        self._dist = dist
        self._tx = ["popen//python=python"] * tx

    def getvalue(self, name):
        if name == "tx":
            return self._tx
        raise KeyError(name)

    def getoption(self, name, default=None):
        if name == "dist":
            return self._dist
        return default


def _scheduler() -> FileOrGroupScheduling:
    return FileOrGroupScheduling(_FakeConfig("loadgroup"))


# ===================================================================
# Scope splitting
# ===================================================================


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        # Unmarked nodeids collapse to their file path: loadfile semantics.
        ("tests/foo/test_alpha.py::test_one", "tests/foo/test_alpha.py"),
        ("tests/foo/test_alpha.py::TestCls::test_one", "tests/foo/test_alpha.py"),
        ("tests/foo/test_alpha.py::test_one[case-a]", "tests/foo/test_alpha.py"),
        # A parametrize value containing '@' is not a group suffix.
        ("tests/foo/test_alpha.py::test_one[user@example.com]", "tests/foo/test_alpha.py"),
        # Doctest nodeids are file-scoped too.
        ("tests/foo/__init__.py::foo.foo", "tests/foo/__init__.py"),
    ],
)
def test_unmarked_nodeids_group_by_file(nodeid, expected):
    assert _scheduler()._split_scope(nodeid) == expected


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        ("tests/foo/test_alpha.py::test_one@ariel-db", "ariel-db"),
        ("tests/foo/test_beta.py::TestCls::test_two@ariel-db", "ariel-db"),
        # Group suffix wins even when the parametrize value also holds an '@'.
        ("tests/foo/test_alpha.py::test_one[user@example.com]@ariel-db", "ariel-db"),
    ],
)
def test_marked_nodeids_group_by_group_name(nodeid, expected):
    assert _scheduler()._split_scope(nodeid) == expected


def test_marked_files_share_a_scope_and_unmarked_files_do_not():
    sched = _scheduler()
    marked = [
        "tests/a/test_one.py::test_x@ariel-db",
        "tests/b/test_two.py::test_y@ariel-db",
    ]
    unmarked = [
        "tests/a/test_one.py::test_x",
        "tests/a/test_one.py::test_z",
        "tests/b/test_two.py::test_y",
    ]

    assert len({sched._split_scope(n) for n in marked}) == 1
    assert len({sched._split_scope(n) for n in unmarked}) == 2


def test_upstream_still_dispatches_through_split_scope():
    """Guard: xdist's schedule() must still call `_split_scope`.

    If an upgrade renames or inlines this seam the override goes silently
    inert, so pin the call site by name rather than by behaviour.
    """
    assert "_split_scope" in LoadScopeScheduling.schedule.__code__.co_names


# ===================================================================
# Hook inertness
# ===================================================================


def test_hook_returns_scheduler_for_loadgroup():
    sched = pytest_xdist_make_scheduler(_FakeConfig("loadgroup"), None)
    assert isinstance(sched, FileOrGroupScheduling)


@pytest.mark.parametrize("dist", ["loadfile", "load", "loadscope", "worksteal", "each", "no"])
def test_hook_is_inert_for_other_dist_modes(dist):
    """The e2e lane runs --dist loadfile; ad-hoc runs use load/worksteal."""
    assert pytest_xdist_make_scheduler(_FakeConfig(dist), None) is None
