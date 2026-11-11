"""pytest plugin: a session that skipped a test fails, so a lane whose tests all skip goes red.

pytest exits 0 when every test it collected skipped, so a CI lane whose tests
all skip for want of a runtime, an extra or a key reports a green check over
nothing. Loaded, this plugin turns a session that passed with a skipped test
into a failed one (exit 1) and prints why as a GitHub ``::error::`` annotation.
A session that collected nothing already exits 5 and stays red; the plugin
names the cause.

What counts matches a junit report's ``skipped``: a test skipped in setup or
call, an xfail, and a module skipped at collection; a flake absorbed by a
rerun skips nothing, since a rerun is not a testcase. Under xdist the controller
receives every worker's reports, so the count is taken there; a worker never
decides.

``--no-skips-allow-partial`` is the gate of ``privilege-split-e2e``, whose audit
module carries documented per-test guards: the session fails only when every
test skipped, and a partial skip is a warning. ``--no-skips-message`` is printed
when the gate passes.

Loaded with ``-p no_skips`` and this directory on ``PYTHONPATH``; ci.yml's
skip-gated pytest steps do both. Standard library and pytest only.
"""

from __future__ import annotations

import pytest

DEFAULT_MESSAGE = "every test ran; none skipped"
#: Skipped tests named in the annotation; pytest's own ``-rs`` summary lists the rest.
NAMED = 10


class NoSkips:
    """Counts testcases and skips the way a junit report does, and fails a session that skipped."""

    def __init__(self, config: pytest.Config) -> None:
        self.message: str = config.getoption("no_skips_message")
        self.allow_partial: bool = config.getoption("no_skips_allow_partial")
        self.tests: set[str] = set()
        self.skipped: set[str] = set()
        self.lines: list[str] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.tests.add(report.nodeid)
        if report.skipped:
            self.skipped.add(report.nodeid)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        # A module skipped at collection is one skipped testcase in a junit report.
        if report.skipped:
            self.tests.add(report.nodeid)
            self.skipped.add(report.nodeid)

    def verdict(self) -> tuple[bool, list[str]]:
        """Return whether the gate passes, and the lines that say why."""
        total, skipped = len(self.tests), len(self.skipped)
        counts = f"no-skips gate: {total} test(s), {skipped} skipped"
        if total == 0:
            return False, [counts, "::error::the session collected no tests"]
        if not skipped:
            return True, [counts, self.message]
        if not self.allow_partial:
            return False, [counts, f"::error::{skipped} skipped test(s): {self._names()}"]
        if skipped >= total:
            return False, [counts, f"::error::every one of the {total} test(s) skipped"]
        return True, [counts, f"::warning::{skipped} of {total} test(s) skipped: {self._names()}"]

    def _names(self) -> str:
        names = sorted(self.skipped)
        more = f" and {len(names) - NAMED} more" if len(names) > NAMED else ""
        return ", ".join(names[:NAMED]) + more

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        passed, self.lines = self.verdict()
        if not passed and exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        terminalreporter.section("no-skips gate")
        for line in self.lines:
            terminalreporter.write_line(line)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("no_skips", "fail the session on a skipped test")
    group.addoption(
        "--no-skips-message", default=DEFAULT_MESSAGE, help="printed when no test skipped"
    )
    group.addoption(
        "--no-skips-allow-partial",
        action="store_true",
        help="fail only a session whose every test skipped; warn on a partial skip",
    )


def pytest_configure(config: pytest.Config) -> None:
    # An xdist worker's reports reach the controller, which judges the whole session.
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(NoSkips(config), "no_skips_gate")
