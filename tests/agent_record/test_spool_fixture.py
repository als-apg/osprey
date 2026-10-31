"""The suite-wide spool fixtures: every test writes into a spool root of its
own, ``spool_for(identity)`` switches the writing identity, a module-scoped
twin holds the root for apps built at module scope, ``recorded_entries()``
consumes the root into an embedded backend and fails on a refused line, and
the repo guard refuses a run that writes under the checkout's ``var/spool``.
"""

from __future__ import annotations

import os

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import identities, writer
from osprey.utils.workspace import SPOOL_DIR_RELPATH
from osprey_connectors_tests._harness import _guards, _repo_cleanliness
from osprey_connectors_tests._paths import REPO_ROOT

#: What the module-scoped twin pointed the writer at while this module's own
#: module-scoped fixtures were being set up.
_MODULE_SEEN: dict[str, str] = {}


def _ids(entries) -> list[str]:
    """The writer's ``entry_id`` of each spool entry (``sp:<identity>:<entry_id>``)."""
    return [e.key.rsplit(":", 1)[1] for e in entries]


@pytest.fixture(scope="module")
def module_app_spool(module_spool_zone):
    """Stands in for an app built at module scope: it writes while being built."""
    _MODULE_SEEN["env"] = os.environ.get(c.SPOOL_DIR_ENV, "")
    _MODULE_SEEN["entry"] = writer.write(c.KIND_AUDIT, {"what": "module setup"}) or ""
    return module_spool_zone


# --- the per-test root ---------------------------------------------------------------


def test_every_test_writes_under_its_own_tmp_root(spool_zone, tmp_path):
    assert spool_zone.root.is_relative_to(tmp_path)
    assert os.environ[c.SPOOL_DIR_ENV] == str(spool_zone.root / spool_zone.terminal)
    assert writer.spool_dir() == spool_zone.root / spool_zone.terminal


def test_the_default_identity_is_a_terminal(spool_zone, recorded_entries):
    entry_id = writer.write(c.KIND_AUDIT, {"what": "x"})
    assert entry_id
    (entry,) = recorded_entries()
    assert _ids([entry]) == [entry_id]
    assert entry.identity == spool_zone.terminal
    assert identities.identity_class(spool_zone.config(), entry.identity) == c.SOURCE_TERMINAL


def test_a_line_is_recorded_exactly_once_across_calls(recorded_entries):
    first = writer.write(c.KIND_AUDIT, {"what": "one"})
    assert _ids(recorded_entries()) == [first]
    second = writer.write(c.KIND_AUDIT, {"what": "two"})
    assert sorted(_ids(recorded_entries())) == sorted([first, second])


def test_recorded_entries_filters_by_kind(recorded_entries):
    writer.write(c.KIND_AUDIT, {"what": "a"})
    feedback = writer.write(c.KIND_FEEDBACK, {"what": "f"})
    assert _ids(recorded_entries(kind=c.KIND_FEEDBACK)) == [feedback]


def test_spool_for_returns_and_selects_the_identitys_own_dir(spool_zone, spool_for):
    directory = spool_for(spool_zone.bridge)
    assert directory == spool_zone.root / spool_zone.bridge
    assert os.environ[c.SPOOL_DIR_ENV] == str(directory)
    assert spool_for() == spool_zone.root / spool_zone.terminal


def test_a_dispatch_identity_may_record_a_run(spool_zone, spool_for, recorded_entries):
    spool_for(spool_zone.dispatch)
    entry_id = writer.write(c.KIND_RUN, {"phase": "accept"}, run_id="r1", session_id="s1")
    assert entry_id
    (entry,) = recorded_entries()
    assert (entry.key, entry.kind, entry.identity) == (
        c.run_key("r1", c.RUN_ACCEPT),
        c.KIND_RUN,
        spool_zone.dispatch,
    )


def test_a_bridge_identity_may_record_a_bridge_turn(spool_zone, spool_for, recorded_entries):
    spool_for(spool_zone.bridge)
    entry_id = writer.write(c.KIND_BRIDGE_TURN, {"text": "hi"}, session_id="s1")
    assert entry_id
    (entry,) = recorded_entries()
    assert (entry.kind, entry.identity) == (c.KIND_BRIDGE_TURN, spool_zone.bridge)


def test_lines_from_several_identities_are_all_consumed(spool_zone, spool_for, recorded_entries):
    writer.write(c.KIND_AUDIT, {"what": "terminal"})
    spool_for(spool_zone.dispatch)
    writer.write(c.KIND_AUDIT, {"what": "worker"})
    spool_for(spool_zone.bridge)
    writer.write(c.KIND_AUDIT, {"what": "bridge"})
    assert sorted(e.identity for e in recorded_entries()) == sorted(
        [spool_zone.terminal, spool_zone.dispatch, spool_zone.bridge]
    )


def test_another_identity_is_declared_in_the_consumers_config(
    spool_zone, spool_for, recorded_entries
):
    spool_for("carol")
    writer.write(c.KIND_AUDIT, {"what": "x"})
    (entry,) = recorded_entries()
    assert entry.identity == "carol"
    assert identities.identity_class(spool_zone.config(), "carol") == c.SOURCE_TERMINAL


def test_a_refused_line_fails_the_test(recorded_entries):
    # A terminal may not send a run: the consumer refuses the line.
    assert writer.write(c.KIND_RUN, {"phase": "accept"}, run_id="r1", session_id="s1")
    with pytest.raises(pytest.fail.Exception, match="refused"):
        recorded_entries()


def test_a_refused_line_fails_every_later_call_too(spool_for, spool_zone, recorded_entries):
    spool_for(spool_zone.bridge)
    writer.write(c.KIND_RUN, {"phase": "accept"}, run_id="r1", session_id="s1")
    with pytest.raises(pytest.fail.Exception, match="refused"):
        recorded_entries()
    with pytest.raises(pytest.fail.Exception, match="refused"):
        recorded_entries()


def test_a_lifted_monkeypatch_keeps_the_root(spool_zone, monkeypatch):
    monkeypatch.setenv(c.SPOOL_DIR_ENV, "/somewhere/else")
    monkeypatch.undo()
    assert os.environ[c.SPOOL_DIR_ENV] == str(spool_zone.root / spool_zone.terminal)


def test_a_test_that_unsets_the_variable_gets_the_root_back(monkeypatch):
    monkeypatch.delenv(c.SPOOL_DIR_ENV)
    assert c.SPOOL_DIR_ENV not in os.environ


def test_the_unset_variable_was_restored_for_the_next_test(spool_zone):
    assert os.environ[c.SPOOL_DIR_ENV] == str(spool_zone.root / spool_zone.terminal)


# --- the module-scoped twin ----------------------------------------------------------


def test_a_module_scoped_app_writes_under_the_module_root(module_app_spool, spool_zone):
    module_root = module_app_spool.root
    assert module_root != spool_zone.root
    assert _MODULE_SEEN["env"] == str(module_root / module_app_spool.terminal)
    assert _MODULE_SEEN["entry"]
    assert _ids(module_app_spool.recorded_entries()) == [_MODULE_SEEN["entry"]]


def test_the_test_body_writes_under_the_per_test_root(module_app_spool, spool_zone):
    assert os.environ[c.SPOOL_DIR_ENV] == str(spool_zone.root / spool_zone.terminal)
    assert not str(spool_zone.root).startswith(str(module_app_spool.root))


# --- the repo guard ------------------------------------------------------------------


def test_the_guard_watches_the_checkouts_spool():
    assert _guards._SPOOL_MARKER == REPO_ROOT / SPOOL_DIR_RELPATH


def test_the_guard_is_armed_for_this_suite(request):
    assert "no_spool_in_the_repo" in request.fixturenames


def test_the_guard_names_a_spool_dir_the_run_added(tmp_path):
    marker = tmp_path / "var" / "spool"
    marker.mkdir(parents=True)
    baseline = _repo_cleanliness.snapshot(marker)
    assert _guards._spool_leak(baseline, marker, None) is None
    (marker / "alice").mkdir()
    message = _guards._spool_leak(baseline, marker, "tests/x.py::test_y (worker gw0)")
    assert message is not None
    assert "alice" in message
    assert "tests/x.py::test_y" in message


def test_the_guard_names_a_spool_root_the_run_created(tmp_path):
    marker = tmp_path / "var" / "spool"
    baseline = _repo_cleanliness.snapshot(marker)
    marker.mkdir(parents=True)
    message = _guards._spool_leak(baseline, marker, None)
    assert message is not None
    assert f"created {marker}" in message


def test_the_guard_leaves_a_spool_that_was_already_there(tmp_path):
    marker = tmp_path / "var" / "spool"
    (marker / "operator").mkdir(parents=True)
    baseline = _repo_cleanliness.snapshot(marker)
    assert _guards._spool_leak(baseline, marker, None) is None


def test_no_test_here_wrote_into_the_checkouts_spool():
    assert (
        _repo_cleanliness.what_this_run_did(_guards._SPOOL_BASELINE, _guards._SPOOL_MARKER) is None
    )


def test_the_record_directory_is_outside_the_spool_root(spool_zone, recorded_entries):
    writer.write(c.KIND_AUDIT, {"what": "x"})
    recorded_entries()
    assert sorted(p.name for p in spool_zone.root.iterdir()) == [spool_zone.terminal]
