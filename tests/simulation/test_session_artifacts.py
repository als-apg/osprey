"""The artifacts of a scenario's past sessions render in the gallery's forms.

The plot of the ``demo-history`` sector 7 session is read from the archive
composite at the anchor, so around 14:32:08 yesterday it shows the stored current
dropping and the sector 7 pressure spiking, from the scenario's own archiver
events. The same anchor renders the same bytes.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey.simulation.session_artifacts import (
    PLOT_DIV_ID,
    PLOT_STEP_S,
    ArtifactPayload,
    render_artifact,
    render_session_artifacts,
)
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.simulation.series import anchored_instant
from osprey_connectors.simulation.sessions import (
    AnswerStep,
    ArtifactSpec,
    ResolvedTurn,
    SessionTurn,
    ToolStep,
    parse_sessions,
    resolve,
)

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
SCENARIO_FILE = TEMPLATE_FACILITY / "scenarios" / "demo-history.yaml"
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}

DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
SR07 = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
ANCHOR = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
NAMES = ["demo-history"]


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project whose render holds the example facility's simulator view, as a build writes it."""
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    root = tmp_path_factory.mktemp("project")
    doc = build_facility(TEMPLATE_FACILITY, project_name="example")
    write_simulator_view(
        root / "data" / "simulator",
        ViewInputs(
            records=FacilityRecords(doc),
            rendered_config=VIEW_CONFIG,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(VIEW_CONFIG, FacilityRecords(doc)),
            reported=None,
        ),
    )
    (root / "config.yml").write_text(yaml.safe_dump({}), encoding="utf-8")
    return root


def _sessions(anchor: datetime = ANCHOR):
    raw = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))["sessions"]
    return resolve(parse_sessions(raw), anchor)


@pytest.fixture(scope="module")
def payloads(project: Path) -> dict[tuple[str, str], ArtifactPayload]:
    return render_session_artifacts(project, NAMES, _sessions(), ANCHOR)


def _figure_traces(html: str) -> list[dict[str, Any]]:
    """The trace list a ``Plotly.newPlot`` call in the page draws."""
    marker = f'Plotly.newPlot(                        "{PLOT_DIV_ID}",'
    start = html.index(marker) + len(marker)
    traces, _end = json.JSONDecoder().raw_decode(html[start:].lstrip())
    return traces


def _event_instant() -> float:
    event = {"at_when": {"days_ago": 1, "time": "14:32:08"}}
    return anchored_instant(event, ANCHOR.timestamp(), get_facility_timezone())


def _axis_seconds(axis: list[str]) -> list[float]:
    tz = get_facility_timezone()
    return [datetime.fromisoformat(stamp).replace(tzinfo=tz).timestamp() for stamp in axis]


# ---------------------------------------------------------------------------
# The plot
# ---------------------------------------------------------------------------


def test_every_artifact_of_the_scenario_renders(payloads) -> None:
    assert set(payloads) == {
        ("sector-7-beam-loss", "loss-plot"),
        ("sector-7-beam-loss", "loss-followup-draft"),
        ("orbit-table-and-stats", "orbit-response-stats"),
    }


def test_the_plot_is_the_gallery_plot_form_without_plotly_inlined(payloads) -> None:
    plot = payloads[("sector-7-beam-loss", "loss-plot")]

    assert plot.artifact_type == "plot_html"
    assert plot.mime_type == "text/html"
    assert plot.filename == "stored_current_and_sector_7_pressure_around_1432.html"
    assert plot.title == "Stored current and sector 7 pressure around 14:32"
    assert plot.tool_source == "create_interactive_plot"
    assert plot.kind == ""
    html = plot.content.decode("utf-8")
    assert html.startswith("<html>")
    assert "Plotly.newPlot" in html
    # The gallery serves the 4.8 MB bundle; a payload never carries it.
    assert "plotly.js v" not in html
    assert "<script src=" not in html
    assert len(plot.content) < 200_000


def test_the_plot_shows_the_current_drop_and_the_pressure_spike_at_1432_yesterday(
    payloads,
) -> None:
    traces = {
        trace["name"]: trace
        for trace in _figure_traces(
            payloads[("sector-7-beam-loss", "loss-plot")].content.decode("utf-8")
        )
    }
    assert set(traces) == {DCCT, SR07}
    event = _event_instant()
    assert datetime.fromtimestamp(event, get_facility_timezone()).strftime("%H:%M:%S") == "14:32:08"

    for name in (DCCT, SR07):
        seconds = _axis_seconds(traces[name]["x"])
        assert seconds[0] < event - 300 and seconds[-1] > event + 300
        assert event in seconds, "the event instant is a sample, so its peak is drawn"

    seconds = _axis_seconds(traces[DCCT]["x"])
    at = seconds.index(event)
    current = traces[DCCT]["y"]
    pressure = traces[SR07]["y"]
    # Well before the burst: stored beam near 500 mA, the gauge at its base.
    assert 490 < current[0] < 510
    assert pressure[0] < 1e-8
    # At 14:32:08: the current is down by about 18 mA, the gauge up near 1.5e-07 Torr.
    assert current[0] - current[at] > 15
    assert min(current) == pytest.approx(current[at], abs=2.0)
    assert pressure[at] == pytest.approx(1.5e-07, rel=0.2)
    assert max(pressure) == pressure[at]
    # Both recover by the end of the window.
    assert current[-1] - current[at] > 15
    assert pressure[-1] < 1e-8


def test_the_plot_samples_are_the_archive_samples(project: Path, payloads) -> None:
    from osprey_connectors.simulation.archive import build
    from osprey_connectors.simulation.view import SimulatorView

    traces = {
        trace["name"]: trace
        for trace in _figure_traces(
            payloads[("sector-7-beam-loss", "loss-plot")].content.decode("utf-8")
        )
    }
    seconds = _axis_seconds(traces[DCCT]["x"])
    assert seconds[1] - seconds[0] == PLOT_STEP_S
    with build(SimulatorView.of_project(project), NAMES, anchor_s=ANCHOR.timestamp()) as archive:
        archived = archive.samples([DCCT, SR07], seconds)

    assert traces[DCCT]["y"] == [float(v) for v in archived[DCCT]]
    assert traces[SR07]["y"] == [float(v) for v in archived[SR07]]


def test_the_same_anchor_renders_identical_bytes(project: Path, payloads) -> None:
    again = render_session_artifacts(project, NAMES, _sessions(), ANCHOR)

    assert again == payloads
    assert {key: p.fields() for key, p in again.items()} == {
        key: p.fields() for key, p in payloads.items()
    }


def test_another_anchor_moves_the_plot(project: Path, payloads) -> None:
    later = datetime(2026, 11, 6, 12, 0, 0, tzinfo=UTC)
    moved = render_session_artifacts(project, NAMES, _sessions(later), later)

    key = ("sector-7-beam-loss", "loss-plot")
    assert moved[key].content != payloads[key].content
    assert (
        moved[("sector-7-beam-loss", "loss-followup-draft")]
        == payloads[("sector-7-beam-loss", "loss-followup-draft")]
    )


# ---------------------------------------------------------------------------
# Note and data
# ---------------------------------------------------------------------------


def test_the_note_is_the_registered_markdown(payloads) -> None:
    note = payloads[("sector-7-beam-loss", "loss-followup-draft")]

    assert note.artifact_type == "markdown"
    assert note.mime_type == "text/markdown"
    assert note.filename == "draft_follow-up_to_demo-036_sector_7_vacuum_burst.md"
    assert note.tool_source == "artifact_register"
    text = note.content.decode("utf-8")
    assert text.startswith("**Follow-up to DEMO-036")
    assert "SR07 ion pump" in text


def test_the_data_is_the_executed_statistics_as_json(payloads) -> None:
    data = payloads[("orbit-table-and-stats", "orbit-response-stats")]

    assert data.artifact_type == "json"
    assert data.mime_type == "application/json"
    assert data.filename == "orbit_response_stats_bpm01_to_bpm06.json"
    assert data.tool_source == "execute"
    assert json.loads(data.content) == {
        "x": {
            "mean_um": 3.83,
            "rms_um": 12.72,
            "std_um": 12.13,
            "worst_bpm": "BPM03",
            "worst_um": 21.0,
        },
        "y": {
            "mean_um": 0.83,
            "rms_um": 3.14,
            "std_um": 3.02,
            "worst_bpm": "BPM03",
            "worst_um": 5.0,
        },
    }


def test_fields_carry_the_seed_file_payload_half(payloads) -> None:
    data = payloads[("orbit-table-and-stats", "orbit-response-stats")]
    fields = data.fields()

    assert set(fields) == {
        "filename",
        "title",
        "artifact_type",
        "mime_type",
        "kind",
        "tool_source",
        "content_b64",
    }
    assert base64.b64decode(fields["content_b64"]) == data.content


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _turn(*steps: ToolStep | AnswerStep, channels: tuple[str, ...] = ()) -> ResolvedTurn:
    return ResolvedTurn(
        turn=SessionTurn(
            prompt="p", at_offset_s=0, channels=channels, logbook_ids=(), steps=tuple(steps)
        ),
        at=ANCHOR,
    )


def test_a_naive_anchor_is_refused(project: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        render_session_artifacts(project, NAMES, _sessions(), ANCHOR.replace(tzinfo=None))


def test_a_note_without_registered_content_is_refused() -> None:
    spec = ArtifactSpec(slug="n", form="note", title="N", source="x.md")
    with pytest.raises(ValueError, match="artifact 'n'.*artifact_register"):
        render_artifact(spec, _turn(AnswerStep(answer="done")))


def test_data_without_a_json_result_is_refused() -> None:
    spec = ArtifactSpec(slug="d", form="data", title="D", source="x.json")
    step = ToolStep(server="python", name="execute", input={}, result="printed text")
    with pytest.raises(ValueError, match="artifact 'd'.*execute"):
        render_artifact(spec, _turn(step))


def test_registered_json_content_is_data() -> None:
    spec = ArtifactSpec(slug="d", form="data", title="D", source="x.json")
    step = ToolStep(
        server="osprey_workspace",
        name="artifact_register",
        input={"title": "D", "content": '{"a": 1}'},
        result="saved",
    )
    payload = render_artifact(spec, _turn(step))
    assert json.loads(payload.content) == {"a": 1}
    assert payload.tool_source == "artifact_register"


def test_a_plot_without_an_archive_is_refused() -> None:
    spec = ArtifactSpec(slug="p", form="plot", title="P", source="x.json")
    with pytest.raises(ValueError, match="archive"):
        render_artifact(spec, _turn(channels=(DCCT,)))
