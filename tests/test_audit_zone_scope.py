"""The ledger seam is redirected before a module-scoped fixture is built.

``_isolate_module_audit_zone`` in ``tests/conftest.py`` holds
``writer.audit_dir`` pointed at a throwaway zone for a whole module, and
``_isolate_audit_zone`` narrows it to each test's own zone on top. What is
pinned here is that layering: a module-scoped fixture is built inside the module
zone, a test body files into its own, and a test that re-points the seam unwinds
to the module zone rather than to the live ledger. The module sits at the
root of the tree because the fixture is suite-wide: a module outside
``tests/interfaces`` is what proves it.

The module-scoped fixture below stands in for a module-scoped app. It records
where the seam resolved instead of starting a server, so the ordering is
asserted in milliseconds.
"""

from pathlib import Path

import pytest

from osprey.audit import writer


@pytest.fixture(scope="module")
def zone_at_module_setup() -> Path:
    """Where the ledger resolved while a module-scoped fixture was built."""
    return writer.audit_dir()


def test_a_module_scoped_fixture_is_built_inside_the_module_zone(
    zone_at_module_setup, _isolate_module_audit_zone
):
    """A module-scoped fixture sees the directory's zone, not the live ledger.

    The private fixture is requested by name deliberately, because this is the
    assertion that discriminates. "The zone is not inside the checkout" would
    not: ``agent_data_never_the_checkout`` (``tests/conftest.py``) already
    diverts ``resolve_project_root`` for the whole session, so an unredirected
    ``audit_dir()`` lands in a throwaway root too and that weaker assertion
    passes with the fixture removed.
    """
    assert zone_at_module_setup == _isolate_module_audit_zone


def test_a_test_body_files_into_its_own_zone(zone_at_module_setup, _isolate_audit_zone):
    """The per-test redirection still wins inside a test body.

    This is the regression the module-scoped fixture could cause: a per-test
    zone that stopped winning would put two tests' records in one file.
    """
    assert writer.audit_dir() == _isolate_audit_zone
    assert _isolate_audit_zone != zone_at_module_setup


def test_a_test_that_repoints_the_seam_unwinds_to_the_module_zone(
    zone_at_module_setup, monkeypatch, tmp_path
):
    """A test's own redirection unwinds to the module zone, not to the resolver.

    ``monkeypatch`` is function-scoped, so the instance this test holds is the
    same object ``_isolate_audit_zone`` used: the undos of one seam stack in one
    list and unwind LIFO, and what is left underneath is the module zone. That
    property is the one the comment above ``_requests_auth_seam`` in
    ``tests/interfaces/conftest.py`` says the interfaces tree turns on. Without
    the module-scoped fixture what is left is the live resolver.
    """
    monkeypatch.setattr(writer, "audit_dir", lambda: tmp_path)
    assert writer.audit_dir() == tmp_path

    monkeypatch.undo()

    assert writer.audit_dir() == zone_at_module_setup
