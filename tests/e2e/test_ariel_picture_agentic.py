"""Agentic e2e: the logbook agent finds the right entry, then reads the answer off its plot.

The logbook holds four entries about sector orbits and RF, so the agent must
first work out which entry the question means. Only that entry has an
attachment: a trend plot. The question asks how far the orbit drifted, which is
what a trend plot shows, and the entry's text says the trend is in the plot
without giving the number. The peak drift, ``0.83`` mm, is in the plot's pixels
and nowhere else: not in any entry's text, not in the stored filename, not in
the PNG's metadata. The curve's top sits between two axis ticks, so an estimate
from the curve alone does not give ``0.83``. An answer that names it therefore
read the picture, and the only route to the picture is the ``attachment_view``
tool inside the ``logbook-search`` subagent. The scenario and every assertion
are in :mod:`tests.e2e.ariel_picture`.

The plot fixture, ``fixtures/ariel_picture/orbit_drift_peak.png``, is a
matplotlib line plot of the horizontal offset over 24 h, titled
``Peak drift 0.83 mm`` with ``peak 0.83 mm`` boxed inside the axes and y ticks
every 0.2 mm, saved with no text metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.ariel_picture import FIXTURES, PictureCase, run_picture_case
from tests.e2e.sdk_helpers import HAS_SDK

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agentic_benchmark,
    pytest.mark.requires_e2e_provider,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
]

#: A neutral question: it names no plot, picture or attachment. It asks for the
#: size of the drift, which the target entry's text leaves to its trend plot.
#: The lane has no caption model, so the agent must decide on its own, from the
#: text and the attachment summary (``viewable: true``, no caption), that the
#: picture is worth a look.
CASE = PictureCase(
    plot_path=FIXTURES / "orbit_drift_peak.png",
    raw_text=(
        "Sector 9 horizontal orbit drift study\n\n"
        "Recorded the overnight drift of the horizontal orbit in sector 9 during the "
        "corrector study. The trend is in the attached plot."
    ),
    prompt="How far did the horizontal orbit drift during the sector 9 overnight study?",
    answer="0.83",
    distractors=(
        "Sector 7 horizontal orbit drift, overnight\n\nThe horizontal orbit in sector 7 "
        "drifted slowly overnight. Feedback was off; it was re-enabled at the start of "
        "the day shift.",
        "Sector 9 vertical orbit check after corrector swap\n\nChecked the vertical orbit "
        "in sector 9 after swapping the corrector power supply. No change seen.",
        "Sector 9 RF tuning notes\n\nRetuned the sector 9 RF cavity after the morning "
        "access. Beam lifetime back to normal.",
    ),
)


@pytest.mark.flaky(reruns=2)  # agentic; absorbs a rare miss of the delegation directive
@pytest.mark.asyncio
async def test_agent_reads_the_attached_plot(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Asked a neutral question, the logbook agent finds the entry, opens its plot and reports the drift.

    Asked next to open the entry, the agent shows it with the plot it discussed.
    """
    await run_picture_case(CASE, tmp_path, request, monkeypatch, caplog)
