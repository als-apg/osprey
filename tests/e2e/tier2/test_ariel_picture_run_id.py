"""Tier 2 model benchmark: the logbook agent looks at a plot nothing tells it to look at.

Tier 2 tests measure what a model can do in OSPREY rather than gate a change:
the task is fair, but a capable model may still miss it. They run on demand
(the ``tier2-model-benchmark`` CI job) and in the model matrix, never on a pull
request.

The logbook holds one entry whose only attachment is a plot. The question asks
for the study's run identifier, a value no reader would expect a trend plot to
carry, and the entry's text gives no hint that it is there: it says only that
the trend is in the attached plot. The plot carries the nonsense token
``QX-7713`` in its pixels and nowhere else: not in the entry's text, not in the
stored filename, not in the PNG's metadata. An answer that names the token
therefore read the picture, and the only route to the picture is the
``attachment_view`` tool inside the ``logbook-search`` subagent. A model that
answers with the entry id instead has taken the logbook's key for something the
entry records. The scenario and every assertion are in
:mod:`tests.e2e.ariel_picture`.

The test carries no rerun: a benchmark scores each attempt as it comes.

The plot fixture, ``fixtures/ariel_picture/orbit_drift_run_id.png``, is a
matplotlib line plot titled ``Run QX-7713`` with the same token boxed inside the
axes, saved with no text metadata.
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

#: A neutral question: it names no plot, picture or attachment, and the run
#: identifier it asks for is only in the plot's pixels. The lane has no caption
#: model, so the agent must decide on its own, from the attachment summary
#: (``viewable: true``, no caption), that the picture is worth a look.
CASE = PictureCase(
    plot_path=FIXTURES / "orbit_drift_run_id.png",
    raw_text=(
        "Sector 9 horizontal orbit drift study\n\n"
        "Recorded the overnight drift of the horizontal orbit in sector 9 during the "
        "corrector study. The trend is in the attached plot."
    ),
    prompt="What is the run identifier of the sector 9 horizontal orbit drift study in the logbook?",
    answer="QX-7713",
)


@pytest.mark.asyncio
async def test_agent_finds_the_run_id_in_an_unmentioned_plot(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Asked for a run identifier, the logbook agent finds it in the entry's plot.

    Asked next to open the entry, the agent shows it with the plot it discussed.
    """
    await run_picture_case(CASE, tmp_path, request, monkeypatch, caplog)
