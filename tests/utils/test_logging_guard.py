"""Tests for the ``restore_root_logging`` guard in tests/conftest.py.

``configure_logging()`` mutates process-global logging state. The guard undoes
that between tests so ``caplog`` behaves identically no matter what ran before.

Several tests here come in ordered pairs: the first deliberately leaks, the
second — which pytest runs immediately after it, in file order, on the same
xdist worker — asserts the leak is gone.

Expected values come from ``_PRISTINE_LOGGING``, the guard's own baseline,
rather than from a level captured in a sibling test. That is deliberate: an
`import litellm` anywhere in the process raises the httpx logger to WARNING as
an import side effect, and six ``osprey`` modules import litellm — three inside
functions, and three (the LiteLLM provider adapter, the in-context
channel-finder query tool, the benchmark harness) at module level, where
collection alone is enough to trigger it. So a value captured in one test is not
a reliable expectation for the next one — the number it produces (WARNING) is
also indistinguishable from what ``configure_logging()`` produces.
``_PRISTINE_LOGGING`` is captured at conftest import, before any of that, and is
exactly the contract the guard promises.
"""

from __future__ import annotations

import logging

import pytest
from rich.logging import RichHandler

from osprey.utils.logger import QUIET_THIRD_PARTY_LOGGERS, configure_logging

# Private by name, but this is the module that tests it.
from tests.conftest import _PRISTINE_LOGGING

_PROBE = "osprey.tests.logging_guard_probe"


def _rich_handler_count() -> int:
    return sum(isinstance(h, RichHandler) for h in logging.getLogger().handlers)


def _assert_third_party_pristine() -> None:
    for name in QUIET_THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == _PRISTINE_LOGGING["third_party"][name]


# ===================================================================
# configure_logging() leaves no trace for the next test
# ===================================================================


def test_configure_logging_mutates_global_state():
    """Leak on purpose. The pre-conditions also pin that nothing leaked into us."""
    root = logging.getLogger()
    assert _rich_handler_count() == 0, "a previous test leaked a RichHandler onto the root logger"
    _assert_third_party_pristine()

    # Move a third-party level somewhere configure_logging() will not land on,
    # so the follow-up test can prove the restore actually ran.
    logging.getLogger("httpx").setLevel(logging.DEBUG)

    configure_logging(logging.DEBUG)

    assert _rich_handler_count() == 1
    assert root.level == logging.DEBUG
    for name in QUIET_THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_configure_logging_left_no_trace():
    """Runs right after the leaking test: root logging is back to the baseline."""
    assert _rich_handler_count() == 0
    assert logging.getLogger().level == _PRISTINE_LOGGING["root_level"]
    _assert_third_party_pristine()

    # Non-vacuous: the previous test explicitly set this one to DEBUG.
    assert logging.getLogger("httpx").level != logging.DEBUG


# ===================================================================
# caplog keeps working
# ===================================================================


def test_caplog_captures_after_configure_logging(caplog):
    """The guard must not strip caplog's handler, and configure_logging must not either."""
    configure_logging()

    with caplog.at_level(logging.INFO, logger=_PROBE):
        logging.getLogger(_PROBE).info("during")

    assert "during" in caplog.text


def test_caplog_captures_in_the_test_after_a_configuring_one(caplog):
    """Runs right after a test that configured logging; capture is unaffected."""
    with caplog.at_level(logging.INFO, logger=_PROBE):
        logging.getLogger(_PROBE).info("after")

    assert "after" in caplog.text
    assert _rich_handler_count() == 0


# ===================================================================
# Configuration done by higher-scoped fixtures
# ===================================================================


@pytest.fixture(scope="module")
def configuring_module_fixture():
    """A module-scoped fixture that configures logging, as a CLI-invoking one would.

    Mirrors ``tests/cli/test_dockerfile_template.py``, whose module-scoped
    fixtures run ``osprey build`` through ``CliRunner``. pytest sets higher-scoped
    fixtures up first, so this lands on the root logger *before* the
    function-scoped guard runs.
    """
    configure_logging()


def test_higher_scoped_configuration_is_undone_for_the_test(configuring_module_fixture):
    """Both the handler and the third-party levels the fixture set are already undone."""
    assert _rich_handler_count() == 0
    _assert_third_party_pristine()


def test_higher_scoped_configuration_stays_undone_afterwards():
    """And stays undone for later tests that never request the fixture."""
    assert _rich_handler_count() == 0
    _assert_third_party_pristine()


def test_third_party_capture_is_not_silenced_by_a_previous_test(caplog):
    """A prior configure_logging() must not decide whether httpx records are captured.

    Two tests above this one raised httpx to WARNING through configure_logging().
    Without the guard's third-party level restore, whether this capture works at
    all would depend on which files ran first on this worker.
    """
    assert logging.getLogger("httpx").level == _PRISTINE_LOGGING["third_party"]["httpx"]

    with caplog.at_level(logging.INFO, logger="httpx"):
        logging.getLogger("httpx").info("third-party record")

    assert "third-party record" in caplog.text
