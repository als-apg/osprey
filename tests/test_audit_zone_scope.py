"""The record spool is redirected before a module-scoped fixture is built.

``_isolate_module_spool_zone`` in ``tests/conftest.py`` points
``OSPREY_SPOOL_DIR`` at a throwaway spool root for a whole module, and
``_isolate_spool_zone`` narrows it to each test's own root on top. What is
pinned here is that layering: a module-scoped fixture is built inside the
module root, a test body writes into its own, a test that re-points the spool
unwinds to its own root rather than to the checkout, the real-deployment lanes
are left alone, and an audit entry fired from a test lands in that test's
spool. The module sits at the root of the tree because the fixtures are
suite-wide: a module outside ``tests/interfaces`` is what proves it.

The module-scoped fixture below stands in for a module-scoped app. It records
where the spool resolved instead of starting a server, so the ordering is
asserted in milliseconds.
"""

import os
from pathlib import Path

import pytest

from osprey.agent_record.contract import RECORD_OFF_ENV, SPOOL_DIR_ENV
from osprey.audit import writer
from osprey.audit.envelope import DECISION_ALLOWED, POSTURE_SOURCE_PROCESS
from tests.conftest import SPOOL_TERMINAL_IDENTITY, _spool_zone

#: A subject no other test writes, so the entry is unmistakably this module's.
_SUBJECT = "mcp__python__zone_scope_probe"


class _Node:
    """The one attribute of a pytest item the spool zone reads."""

    def __init__(self, nodeid: str):
        self.nodeid = nodeid


@pytest.fixture(scope="module")
def spool_at_module_setup() -> str | None:
    """Where the spool resolved while a module-scoped fixture was built."""
    return os.environ.get(SPOOL_DIR_ENV)


def test_a_module_scoped_fixture_is_built_inside_the_module_zone(
    spool_at_module_setup, module_spool_zone
):
    """A module-scoped fixture writes into the module's root, not the checkout.

    The module zone is requested by name deliberately, because this is the
    assertion that discriminates. "The spool is not inside the checkout" would
    not: ``no_spool_in_the_repo`` already refuses a write there, so a weaker
    assertion would pass with the module fixture removed.
    """
    assert spool_at_module_setup == str(module_spool_zone.root / SPOOL_TERMINAL_IDENTITY)


def test_a_test_body_writes_into_its_own_zone(spool_at_module_setup, spool_zone, module_spool_zone):
    """The per-test redirection still wins inside a test body.

    This is the regression the module-scoped fixture could cause: a per-test
    root that stopped winning would put two tests' entries in one spool.
    """
    assert os.environ[SPOOL_DIR_ENV] == str(spool_zone.root / SPOOL_TERMINAL_IDENTITY)
    assert spool_zone.root != module_spool_zone.root
    assert os.environ[SPOOL_DIR_ENV] != spool_at_module_setup


def test_two_tests_never_share_a_root(spool_zone, tmp_path):
    """Each test's root sits under that test's own ``tmp_path``."""
    assert spool_zone.root.is_relative_to(tmp_path)


def test_a_test_that_repoints_the_spool_unwinds_to_its_own_zone(spool_zone, monkeypatch):
    """A test's own redirection unwinds to its zone, not to the resolver.

    The zone holds a ``MonkeyPatch`` of its own, set up ahead of the shared
    ``monkeypatch``, so a test's ``undo()`` lifts only what the test itself
    changed and the spool falls back to the test's own root.
    """
    own = str(spool_zone.root / SPOOL_TERMINAL_IDENTITY)
    monkeypatch.delenv(SPOOL_DIR_ENV)
    assert SPOOL_DIR_ENV not in os.environ

    monkeypatch.undo()

    assert os.environ[SPOOL_DIR_ENV] == own


@pytest.mark.parametrize(
    "nodeid",
    ["tests/e2e/test_stack.py::test_probe", "tests/va/e2e/test_stack.py::test_probe"],
)
def test_the_real_deployment_lanes_are_left_alone(nodeid, tmp_path):
    """A real-deployment item records against the checkout: the spool is not moved."""
    before = os.environ.get(SPOOL_DIR_ENV)
    with _spool_zone(_Node(nodeid), tmp_path / "spool", tmp_path / "record") as zone:
        assert os.environ.get(SPOOL_DIR_ENV) == before
        zone.spool_for("bob")
        assert os.environ.get(SPOOL_DIR_ENV) == before
    assert os.environ.get(SPOOL_DIR_ENV) == before


def test_any_other_item_is_moved_and_restored(tmp_path):
    """The contrast to the lanes above: an ordinary item's spool moves, then unwinds."""
    before = os.environ.get(SPOOL_DIR_ENV)
    root = tmp_path / "spool"
    with _spool_zone(_Node("tests/audit/test_probe.py::test_probe"), root, tmp_path / "record"):
        assert os.environ[SPOOL_DIR_ENV] == str(root / SPOOL_TERMINAL_IDENTITY)
    assert os.environ.get(SPOOL_DIR_ENV) == before


def test_an_audit_entry_lands_in_the_tests_own_spool(
    monkeypatch, recorded_entries, module_spool_zone
):
    """An audit entry fired from a test body is in this test's record, not the module's."""
    monkeypatch.delenv(RECORD_OFF_ENV, raising=False)

    entry_id = writer.record(
        surface="mcp_python",
        posture="sandbox",
        posture_source=POSTURE_SOURCE_PROCESS,
        session=None,
        subject=_SUBJECT,
        decision=DECISION_ALLOWED,
        reason="allowed",
    )

    assert entry_id is not None
    (entry,) = [e for e in recorded_entries("audit") if e.body["subject"] == _SUBJECT]
    assert entry.identity == SPOOL_TERMINAL_IDENTITY
    assert Path(os.environ[SPOOL_DIR_ENV]).name == entry.identity
    assert [
        e for e in module_spool_zone.recorded_entries("audit") if e.body["subject"] == _SUBJECT
    ] == []
