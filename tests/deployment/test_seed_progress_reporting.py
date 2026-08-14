"""The archiver seed's progress callback, reported as phase sub-steps.

Seeding an archive base is the one deploy step that runs for minutes, so its
progress callback is bridged to the open phase: each firing that survives the
rate limit becomes a ``· seeding archive: …`` line. These tests drive the
callback directly with synthetic :class:`SeedReport` values — no store, no
containers — and assert the three things the bridge owns: what a line says,
that the rate limit holds, and that a callback outside a verb is inert.
"""

from __future__ import annotations

import re

import pytest

from osprey.cli.phase_reporter import (
    NullReporter,
    PhaseReporter,
    current_reporter,
    install_reporter,
)
from osprey.deployment import container_lifecycle
from osprey.simulation.archiver_seed import SeedReport


class RecordingReporter(PhaseReporter):
    """A real reporter that keeps its lines instead of printing them."""

    def __init__(self) -> None:
        super().__init__(color=False)
        self.lines: list[str] = []

    def emit(self, text: str, style: str | None = None) -> None:
        self.lines.append(text)


@pytest.fixture
def restore_reporter():
    """Put the module handle back however a test left it."""
    previous = current_reporter()
    try:
        yield
    finally:
        install_reporter(previous)


@pytest.fixture
def phase(restore_reporter):
    """A recording reporter with a phase open, as a verb would leave it."""
    recording = RecordingReporter()
    install_reporter(recording)
    recording.phase("Seeding archive")
    return recording


def steps(reporter: RecordingReporter) -> list[str]:
    """The sub-step lines, in order, as names only.

    The indent, the bullet and the lap duration a step line carries when the lap
    is long enough all belong to the reporter's own rendering tests; what this
    module asserts is the name the bridge composed.
    """
    return [
        re.sub(r" \([^()]*\)$", "", line.strip()[2:])
        for line in reporter.lines
        if line.strip().startswith("·")
    ]


def a_report(documents: int, channels: int = 12) -> SeedReport:
    """A running report as the seeder hands one to the callback."""
    return SeedReport(documents=documents, channels=channels, chunks=1, elapsed_s=1.0)


def test_progress_reports_documents_and_channels_as_a_step(phase, monkeypatch):
    monkeypatch.setattr(container_lifecycle, "_ARCHIVER_PROGRESS_INTERVAL_S", 0.0)
    report = container_lifecycle._seed_progress_reporter()

    report(a_report(1_234_567, channels=1_024))

    assert steps(phase) == ["seeding archive: 1,234,567 documents written across 1,024 channels"]


def test_every_firing_past_the_interval_reports_the_running_count(phase, monkeypatch):
    monkeypatch.setattr(container_lifecycle, "_ARCHIVER_PROGRESS_INTERVAL_S", 0.0)
    report = container_lifecycle._seed_progress_reporter()

    for documents in (500, 1_000, 1_500):
        report(a_report(documents))

    assert steps(phase) == [
        "seeding archive: 500 documents written across 12 channels",
        "seeding archive: 1,000 documents written across 12 channels",
        "seeding archive: 1,500 documents written across 12 channels",
    ]


def test_the_rate_limit_swallows_the_burst_of_per_chunk_firings(phase):
    """The seeder fires once per chunk; at the shipped interval none get through.

    The default interval is the shipped one and the clock is the real one, so a
    tight burst is entirely inside the first window — which is exactly the flood
    the limit exists to stop.
    """
    report = container_lifecycle._seed_progress_reporter()

    for chunk in range(200):
        report(a_report(chunk * 500))

    assert steps(phase) == []


def test_a_callback_outside_a_verb_reports_nothing(restore_reporter, monkeypatch):
    """Seeding also runs from ``sim apply`` and from tests, with no phase open."""
    monkeypatch.setattr(container_lifecycle, "_ARCHIVER_PROGRESS_INTERVAL_S", 0.0)
    recording = RecordingReporter()
    install_reporter(recording)

    container_lifecycle._seed_progress_reporter()(a_report(500))

    assert recording.lines == []


def test_progress_prints_nothing_under_a_null_reporter(restore_reporter, monkeypatch, capsys):
    """Under ``--verbose`` the raw seeder output is what the operator watches."""
    monkeypatch.setattr(container_lifecycle, "_ARCHIVER_PROGRESS_INTERVAL_S", 0.0)
    null = NullReporter(verbose=True)
    install_reporter(null)
    null.phase("Seeding archive")
    capsys.readouterr()

    container_lifecycle._seed_progress_reporter()(a_report(500))

    assert capsys.readouterr().out == ""
