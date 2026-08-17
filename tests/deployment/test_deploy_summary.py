"""Tests for the post-deploy endpoint summary.

Every ``osprey up`` ends with a summary of what is reachable where,
derived from the rendered compose files' published host ports — plus an
unconditional web-terminal line, so a project *without* a web tier says so
explicitly instead of silently binding nothing.

The summary is the deploy's own output, so it is *printed* rather than logged:
an INFO record is not rendered on a normal run, and a fact only a
``--verbose`` run shows is a fact the deploy did not report. The needle tests
below hold both halves of that claim side by side — the block is in the default
view, its logged form no longer paints there, and the record is still emitted
for file and aggregation sinks.
"""

from __future__ import annotations

import io
import logging
import re

import pytest
from rich.console import Console
from rich.logging import RichHandler

from osprey.cli.altitude import install_gate, lift_gate
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.styles import osprey_theme
from osprey.deployment import deploy_summary

#: Anything Rich writes that is not text: styles, cursor moves, erases.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


@pytest.fixture
def compose_file(tmp_path):
    path = tmp_path / "docker-compose.yml"
    path.write_text(
        """
services:
  event-dispatcher:
    ports:
      - "127.0.0.1:8020:8020"
  openobserve:
    ports:
      - "127.0.0.1:5080:5080"
  postgresql:
    ports:
      - "127.0.0.1:5432:5432"
""",
        encoding="utf-8",
    )
    return str(path)


def test_summary_lists_published_ports(compose_file):
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [compose_file])
    assert "demo" in text
    assert "event-dispatcher" in text
    assert "http://127.0.0.1:8020" in text
    assert "http://127.0.0.1:5080" in text
    # postgres is not an HTTP service — plain address, no scheme
    assert "127.0.0.1:5432" in text
    assert "http://127.0.0.1:5432" not in text


def test_summary_says_web_terminal_not_configured(compose_file):
    """The load-bearing line: absence must be an explicit negative signal."""
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [compose_file])
    assert "web terminal" in text
    assert "not configured" in text


def test_summary_shows_landing_url_when_web_enabled(compose_file):
    config = {
        "project_name": "demo",
        "modules": {"web_terminals": {"enabled": True, "nginx_port": 9080}},
    }
    text = deploy_summary.format_endpoint_summary(config, [compose_file])
    assert "http://127.0.0.1:9080" in text
    assert "not configured" not in text


def test_summary_handles_no_published_ports(tmp_path):
    empty = tmp_path / "docker-compose.yml"
    empty.write_text("services: {}\n", encoding="utf-8")
    text = deploy_summary.format_endpoint_summary({"project_name": "demo"}, [str(empty)])
    assert "web terminal" in text  # the unconditional line survives an empty stack


def test_log_endpoint_summary_never_raises(tmp_path):
    """Advisory output must not be able to fail a deploy that succeeded."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [str(tmp_path / "missing.yml")])


# ---------------------------------------------------------------------------
# The promoted summary: printed, not logged
# ---------------------------------------------------------------------------


class _RecordingReporter(PhaseReporter):
    """A reporter whose console is a buffer, so printed output is readable.

    Subclasses the real reporter rather than standing in for it: what is being
    pinned is that the summary goes through the renderer's own ``out()`` seam,
    which a look-alike would not prove.
    """

    def __init__(self, console: Console) -> None:
        super().__init__(color=False)
        self._console = console

    def out(self) -> Console:
        return self._console


class _RecordSink(logging.Handler):
    """Every record the loggers emitted, captured before any handler filter."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _Probe:
    """What the terminal showed, and what the loggers emitted, side by side.

    ``printed`` is the echo path — where a promoted fact now lives.
    ``rendered`` is what the root ``RichHandler`` painted, after the altitude
    gate has had its say. ``messages`` is the raw record stream, which the gate
    never touches: it is a handler filter, so a record it drops is still
    emitted and still reaches every sink.
    """

    def __init__(
        self,
        printed: io.StringIO,
        painted: io.StringIO,
        sink: _RecordSink,
        handler: RichHandler,
    ) -> None:
        self._printed = printed
        self._painted = painted
        self._sink = sink
        self.handler = handler

    @property
    def printed(self) -> str:
        """The verb's own output, with escape sequences stripped."""
        return _ANSI.sub("", self._printed.getvalue())

    @property
    def rendered(self) -> str:
        """What the logging handler painted, with escape sequences stripped."""
        return _ANSI.sub("", self._painted.getvalue())

    @property
    def messages(self) -> list[str]:
        """The formatted message of every captured record."""
        return [record.getMessage() for record in self._sink.records]


@pytest.fixture
def probe(restore_root_logging):
    """Capture the printed stream and the painted log stream of one call.

    A local instrument rather than ``tests/cli``'s ``terminal_probe`` fixture,
    which that package's ``conftest`` does not share with this one. Both halves
    are built the same way: a themed terminal console writing into a buffer, the
    altitude gate a normal run installs, and a record sink underneath the gate.
    """
    printed = io.StringIO()
    painted = io.StringIO()
    console_kwargs = {
        "theme": osprey_theme,
        "force_terminal": True,
        "width": 100,
        "color_system": "truecolor",
    }

    previous = install_reporter(_RecordingReporter(Console(file=printed, **console_kwargs)))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = RichHandler(
        console=Console(file=painted, **console_kwargs),
        markup=True,
        show_path=False,
        show_time=False,
        show_level=True,
    )
    install_gate(handler)
    sink = _RecordSink()
    root.addHandler(handler)
    root.addHandler(sink)
    try:
        yield _Probe(printed, painted, sink, handler)
    finally:
        root.removeHandler(handler)
        root.removeHandler(sink)
        install_reporter(previous)


def test_the_endpoint_summary_is_in_the_default_view(probe, compose_file):
    """The needle: printed on a normal run, no longer painted by the logger."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])
    # ERROR, not WARNING: the gate keeps WARNING off the terminal while a
    # reporter is installed, and this fixture installs one.
    logging.getLogger("deployment.summary").error("armed witness")

    # Echoed: what the operator ran the verb to find out.
    assert "Service endpoints (demo):" in probe.printed
    assert "event-dispatcher" in probe.printed
    assert "http://127.0.0.1:8020" in probe.printed

    # The armed witness proves the painting console is live and the gate is the
    # only reason the summary is missing from it — an absence assertion against
    # a console nobody ever wrote to passes for the wrong reason.
    assert "armed witness" in probe.rendered
    assert "Service endpoints" not in probe.rendered
    assert "http://127.0.0.1:8020" not in probe.rendered

    # Nothing was lost: the record still carries the whole block for sinks.
    assert any("Service endpoints (demo):" in message for message in probe.messages)
    assert any("http://127.0.0.1:8020" in message for message in probe.messages)


def test_the_logged_block_still_reaches_a_verbose_transcript(probe, compose_file):
    """Lifting the gate is what ``-v`` does, and the record paints again there.

    The record half is kept at its old level deliberately: a file or
    aggregation sink gets the whole summary in one record, and a transcript run
    is the one place a reader asked for it twice.
    """
    lift_gate(probe.handler)

    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])

    assert "Service endpoints (demo):" in probe.printed  # echo class is unconditional
    assert "Service endpoints (demo):" in probe.rendered


def test_printed_summary_keeps_the_entries_and_their_order(probe, compose_file):
    """Same endpoints, same order — only the vocabulary changed."""
    config = {
        "project_name": "demo",
        "modules": {"web_terminals": {"enabled": True, "nginx_port": 9080}},
    }
    entries = deploy_summary.endpoint_entries(config, [compose_file])

    deploy_summary.log_endpoint_summary(config, [compose_file])

    rows = [line for line in probe.printed.splitlines() if line.startswith("  ")]
    assert len(rows) == len(entries)
    for row, (service, address) in zip(rows, entries, strict=True):
        assert row.startswith(f"  {service}")
        assert row.rstrip().endswith(address)


def test_printed_summary_uses_the_section_shape(probe, compose_file):
    """``output.section`` vocabulary: a title, then one indented aligned row."""
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [compose_file])

    lines = [line.rstrip() for line in probe.printed.splitlines() if line.strip()]
    assert lines[0] == "Service endpoints (demo):"
    rows = lines[1:]
    assert all(row.startswith("  ") for row in rows)
    # Labels are padded to one width, so the addresses line up as a column.
    starts = {row.index("http") for row in rows if "http" in row}
    assert len(starts) == 1


def test_a_failed_summary_prints_nothing_and_is_recorded(probe, monkeypatch):
    """A summary that cannot be derived stays silent on the operator's terminal."""

    def boom(config, compose_files):
        raise RuntimeError("no compose files")

    monkeypatch.setattr(deploy_summary, "endpoint_entries", boom)
    # The reason lives at debug grade, so the run has to be one that emits it.
    logging.getLogger().setLevel(logging.DEBUG)
    deploy_summary.log_endpoint_summary({"project_name": "demo"}, [])

    assert "Service endpoints" not in probe.printed
    assert any("Endpoint summary skipped" in message for message in probe.messages)


def test_every_deploy_exit_path_shares_this_one_seam():
    """The exit paths inherit the printed form because they all land here.

    ``deploy_up`` reports its endpoints from several exits — the empty-stack
    early return, the web-terminals-only return, the detached run and the
    foreground ``execvpe`` — and each calls this one function, so the promotion
    lives in one place rather than in four.
    """
    from osprey.deployment import container_lifecycle

    assert container_lifecycle.log_endpoint_summary is deploy_summary.log_endpoint_summary
