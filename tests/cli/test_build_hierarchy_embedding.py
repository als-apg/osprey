"""The hierarchy a build embeds in the channel-finder agent, and its failure log.

``resolve_hierarchy_context`` reads the deployment's channel database at render
time and embeds its levels and naming pattern in the agent's prompt, so the
agent needs no ``hierarchy_info()`` round trip to know the shape of the
facility. The shortcut is optional by design — a database that will not open
costs the agent a shortcut, not the build — and that is exactly what makes it
easy to lose silently: the render carries on, the prompt quietly falls back,
and nothing goes red.

Two properties keep it honest. The embedding has to actually happen for the
reference deployment, which is what a database malformed in a way no test read
would otherwise catch; and the swallowed failure has to be *reported* as one
line, not as a raw traceback. A traceback under an exit-0 build is worse than
no message at all: it is an alarm the operator is being trained to scroll past.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.templates.claude_code import resolve_hierarchy_context

#: This module renders without a venv and without lifecycle hooks; neither has
#: any bearing on what the render embeds. Same flags as the other build suites.
CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

RESOLVER_LOGGER = "osprey.cli.templates.claude_code"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _build(runner: CliRunner, repo: Path):
    """Run ``osprey build`` from *repo*, as an operator standing there would."""
    previous = Path.cwd()
    os.chdir(repo)
    try:
        return runner.invoke(build, CI_FLAGS)
    finally:
        os.chdir(previous)


def _channel_finder_config(db_path: str) -> dict:
    return {"pipelines": {"hierarchical": {"database": {"path": db_path}}}}


class TestUnreadableDatabaseIsReportedNotDumped:
    """The swallowed failure arm: one line naming the file, never a traceback."""

    def test_a_malformed_database_costs_the_shortcut_and_nothing_else(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        db = tmp_path / "channels.json"
        db.write_text(json.dumps({"tree": {}, "hierarchy": {"levels": []}}), encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger=RESOLVER_LOGGER):
            resolved = resolve_hierarchy_context(_channel_finder_config("channels.json"), tmp_path)

        assert resolved is None
        assert len(caplog.records) == 1

    def test_the_warning_carries_no_traceback(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``exc_info`` on the record is what a handler renders as a traceback.

        Asserting on the record rather than on rendered text is what makes this
        independent of terminal width — the Rich handler wraps a traceback
        across lines that no substring survives.
        """
        db = tmp_path / "channels.json"
        db.write_text(json.dumps({"tree": {}, "hierarchy": {"levels": []}}), encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger=RESOLVER_LOGGER):
            resolve_hierarchy_context(_channel_finder_config("channels.json"), tmp_path)

        record = caplog.records[0]
        assert record.exc_info is None
        assert "channels.json" in record.getMessage(), (
            f"the warning must name the database it could not read: {record.getMessage()!r}"
        )

    def test_a_database_that_is_not_there_is_reported_the_same_way(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=RESOLVER_LOGGER):
            resolved = resolve_hierarchy_context(_channel_finder_config("absent.json"), tmp_path)

        assert resolved is None
        assert [r for r in caplog.records if r.exc_info] == []


class TestTheReferenceDeploymentEmbedsItsHierarchy:
    """The positive property, against a real build of the exemplar repo."""

    def test_the_rendered_agent_carries_the_naming_pattern(
        self, runner: CliRunner, lifecycle_repo: Path
    ) -> None:
        """The exemplar's own channel database has to load, or the prompt is poorer.

        This fails the moment the exemplar ships a database the loader rejects,
        which is the failure the build itself only ever warned about.
        """
        assert _build(runner, lifecycle_repo).exit_code == 0

        agent = lifecycle_repo / "build" / ".claude" / "agents" / "channel-finder.md"
        prompt = agent.read_text(encoding="utf-8")
        assert "naming_pattern" in prompt
        assert "{ring}:{system}:{family}:{device}:{field}:{subfield}" in prompt

    def test_the_build_log_holds_no_traceback(
        self, runner: CliRunner, lifecycle_repo: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A build that exits 0 must not print a stack trace on the way there.

        Deliberately about tracebacks in general rather than about this one
        exception: the next swallowed failure will be a different class, and it
        should fail here rather than teach the same lesson again. ``exc_info``
        names the offenders; the text check is the general net.

        Both read the captured log rather than the command's output, because
        the command's output cannot see a traceback at all — the Rich logging
        handler writes to its own console, and an ``exc_info`` warning raised
        mid-build leaves ``result.output`` completely unchanged. Asserting on
        it would pass forever.
        """
        with caplog.at_level(logging.WARNING):
            result = _build(runner, lifecycle_repo)

        assert result.exit_code == 0, result.output
        dumped = [
            f"{record.name}: {record.getMessage()}" for record in caplog.records if record.exc_info
        ]
        assert dumped == [], f"a successful build logged {len(dumped)} traceback(s): {dumped}"
        assert "Traceback" not in caplog.text
