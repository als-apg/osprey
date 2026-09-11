"""The test session is hermetic against the developer's own shell and ``.env``.

CI runs with no ``.env``; a developer's checkout normally has one, and the config
loader reads it into ``os.environ`` with override semantics from a module-level
logger call -- so importing almost any ``osprey`` module is enough to publish
every key in that file. ``restore_environ`` in ``tests/conftest.py`` clears the
keys that would otherwise make a local run diverge from CI.
"""

import os
import time

import pytest

from tests import _env_scope_guard as guard
from tests._env_scope_guard import (
    LONG_LIVED_ENV_WRITES,
    SCOPES_OUTLIVING_A_MODULE,
    record_env_delta,
    restore_module_environment,
)

#: Keys ``restore_environ`` clears on the way in to every test.
PRISTINE_KEYS = ("OSPREY_CONFIG", "CONFIG_FILE", "TZ")


@pytest.mark.parametrize("key", PRISTINE_KEYS)
def test_developer_environment_does_not_reach_tests(key):
    """A key from the developer's shell or ``.env`` never reaches a test body."""
    assert key not in os.environ, (
        f"{key} reached a test from the developer's shell or .env. CI has "
        "neither, so this run does not reproduce what CI does."
    )


def test_process_timezone_agrees_with_environ():
    """``TZ`` and the C library's cached zone are in step.

    Clearing ``TZ`` without ``time.tzset()`` would leave the process running on
    the zone ``.env`` selected while ``os.environ`` claimed otherwise -- the next
    test to call ``tzset()`` would then shift zones mid-test and trip the
    host-timezone leak guard for a reason that has nothing to do with that test.
    """
    before = time.tzname
    time.tzset()
    assert time.tzname == before, (
        f"process timezone {before} does not match a tzset() of the current "
        f"environment ({time.tzname}); TZ was cleared without a tzset()."
    )


class TestModuleScopedRestore:
    """``restore_module_environment`` drops a module's writes, keeps the rest.

    The two halves pull in opposite directions: a module-scoped fixture that
    loads a seeded repo's ``.env`` must not outlive its module, while a
    session-scoped fixture created part-way through that same module must — it
    is still alive when the next module runs, and pytest will not set it up
    again.
    """

    def test_a_module_scoped_write_does_not_outlive_the_module(self):
        with restore_module_environment():
            os.environ["OSPREY_TEST_MODULE_WRITE"] = "from-the-module-fixture"
        assert "OSPREY_TEST_MODULE_WRITE" not in os.environ

    def test_a_longer_lived_fixtures_write_survives_the_module(self):
        """The regression: rolling this back strands a live fixture."""
        with restore_module_environment():
            # What the hook records around a session-scoped fixture's setup.
            before = dict(os.environ)
            os.environ["OSPREY_TEST_SESSION_WRITE"] = "from-the-session-fixture"
            record_env_delta(before)

            os.environ["OSPREY_TEST_MODULE_WRITE"] = "from-the-module-fixture"

        try:
            assert os.environ.get("OSPREY_TEST_SESSION_WRITE") == "from-the-session-fixture"
            assert "OSPREY_TEST_MODULE_WRITE" not in os.environ
        finally:
            os.environ.pop("OSPREY_TEST_SESSION_WRITE", None)

    def test_a_longer_lived_fixtures_deletion_survives_the_module(self):
        os.environ["OSPREY_TEST_PRE_EXISTING"] = "set-before-the-module"
        try:
            with restore_module_environment():
                before = dict(os.environ)
                del os.environ["OSPREY_TEST_PRE_EXISTING"]
                record_env_delta(before)
            assert "OSPREY_TEST_PRE_EXISTING" not in os.environ
        finally:
            os.environ.pop("OSPREY_TEST_PRE_EXISTING", None)

    def test_a_nested_guard_leaves_the_enclosing_record_alone(self):
        """Every test in this module already runs inside the autouse guard.

        A guard that reset the record globally would, entered a second time,
        throw away what the enclosing one had recorded -- and its module would
        then roll a live session fixture's writes back at teardown, which is
        the failure the record exists to prevent.
        """
        before = dict(os.environ)
        os.environ["OSPREY_TEST_ENCLOSING_WRITE"] = "from-the-session-fixture"
        record_env_delta(before)
        try:
            with restore_module_environment():
                os.environ["OSPREY_TEST_MODULE_WRITE"] = "from-the-module-fixture"

            assert LONG_LIVED_ENV_WRITES.get("OSPREY_TEST_ENCLOSING_WRITE") == (
                "from-the-session-fixture"
            )
            assert "OSPREY_TEST_MODULE_WRITE" not in os.environ
        finally:
            os.environ.pop("OSPREY_TEST_ENCLOSING_WRITE", None)
            LONG_LIVED_ENV_WRITES.pop("OSPREY_TEST_ENCLOSING_WRITE", None)

    def test_a_nested_guards_long_lived_write_is_handed_outwards(self):
        """A session fixture set up inside the inner region outlives both.

        The inner guard replays the write over its own snapshot; the enclosing
        one has to hear about it too, or it rolls the same write back later.
        """
        with restore_module_environment():
            before = dict(os.environ)
            os.environ["OSPREY_TEST_SESSION_WRITE"] = "x"
            record_env_delta(before)
        try:
            assert os.environ.get("OSPREY_TEST_SESSION_WRITE") == "x"
            assert LONG_LIVED_ENV_WRITES.get("OSPREY_TEST_SESSION_WRITE") == "x"
        finally:
            os.environ.pop("OSPREY_TEST_SESSION_WRITE", None)
            LONG_LIVED_ENV_WRITES.pop("OSPREY_TEST_SESSION_WRITE", None)

    def test_a_region_that_encloses_nothing_hands_its_record_to_nobody(self, monkeypatch):
        """One module's long-lived writes are not the next module's to replay.

        Handing them outwards is right only while something is out there. The
        outermost region has no one to tell, and a record that kept growing
        would have every later module replaying writes it never saw.
        """
        monkeypatch.setattr(guard, "_OPEN_REGIONS", [])
        monkeypatch.setattr(guard, "LONG_LIVED_ENV_WRITES", {})

        with restore_module_environment():
            before = dict(os.environ)
            os.environ["OSPREY_TEST_SESSION_WRITE"] = "x"
            guard.record_env_delta(before)
        try:
            assert os.environ.get("OSPREY_TEST_SESSION_WRITE") == "x"
            assert guard.LONG_LIVED_ENV_WRITES == {}
        finally:
            os.environ.pop("OSPREY_TEST_SESSION_WRITE", None)

    def test_only_scopes_that_outlive_a_module_are_measured(self):
        """The hook skips function and module scope; measuring them would make
        every module-scoped write look long-lived and defeat the guard."""
        assert SCOPES_OUTLIVING_A_MODULE == {"session", "package"}
        assert "module" not in SCOPES_OUTLIVING_A_MODULE
        assert "function" not in SCOPES_OUTLIVING_A_MODULE


@pytest.fixture(scope="session")
def _session_env_probe():
    """A session-scoped fixture that publishes an environment variable."""
    os.environ["OSPREY_TEST_HOOK_PROBE"] = "1"
    yield
    os.environ.pop("OSPREY_TEST_HOOK_PROBE", None)


def test_the_conftest_hook_attributes_a_session_fixtures_write(_session_env_probe):
    """End to end: the hook is installed and measures the setup that matters.

    The unit tests above drive ``record_env_delta`` by hand; this is what proves
    ``pytest_fixture_setup`` in ``tests/conftest.py`` actually calls it, so a
    pytest release that changes the hook's shape fails here rather than silently
    reverting the guard to the behaviour it was written to replace.
    """
    assert LONG_LIVED_ENV_WRITES.get("OSPREY_TEST_HOOK_PROBE") == "1"
