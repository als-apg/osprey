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
