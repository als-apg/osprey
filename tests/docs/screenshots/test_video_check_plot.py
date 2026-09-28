"""Tests for the demo-video plot artifact check.

CI-safe: pure functions over in-memory artifact lists and a fake HTML fetcher.
"""

from __future__ import annotations

import json

import pytest
from docs.screenshots import video_check_plot

SCATTER3D_HTML = (
    '<div id="p"></div><script>Plotly.newPlot("p", ['
    '{"type": "scatter3d", "name": "SR01C:BPM1"},'
    '{"type": "scatter3d", "name": "SR01C:BPM2"},'
    '{"type": "scatter3d", "name": "SR01C:BPM3"}]);</script>'
)
SCATTER2D_HTML = SCATTER3D_HTML.replace("scatter3d", "scatter")


def _plot(artifact_id: str, **overrides) -> dict:
    artifact = {
        "id": artifact_id,
        "artifact_type": "plot_html",
        "tool_source": "create_interactive_plot",
        "filename": f"{artifact_id}.html",
    }
    artifact.update(overrides)
    return artifact


def _fetcher(pages: dict[str, str]):
    calls: list[str] = []

    def fetch(path: str) -> str:
        calls.append(path)
        return pages[path]

    fetch.calls = calls
    return fetch


def test_scatter3d_with_three_bpms_is_found():
    fetch = _fetcher({"/files/a1/a1.html": SCATTER3D_HTML})
    assert video_check_plot.find_plot_artifact([_plot("a1")], set(), fetch) == "a1"
    assert fetch.calls == ["/files/a1/a1.html"]


def test_2d_scatter_is_rejected():
    fetch = _fetcher({"/files/a1/a1.html": SCATTER2D_HTML})
    assert video_check_plot.find_plot_artifact([_plot("a1")], set(), fetch) is None


def test_archiver_artifact_is_ignored_without_fetch():
    archiver = _plot("arch", artifact_type="archiver_data", tool_source="archiver_read")
    fetch = _fetcher({})
    assert video_check_plot.find_plot_artifact([archiver], set(), fetch) is None
    assert fetch.calls == []


def test_plot_html_from_other_tool_is_ignored():
    other = _plot("x", tool_source="archiver_read")
    fetch = _fetcher({"/files/x/x.html": SCATTER3D_HTML})
    assert video_check_plot.find_plot_artifact([other], set(), fetch) is None


def test_short_labels_are_accepted():
    html = '{"type": "scatter3d"} labels: BPM 01, BPM-2, BPM3'
    fetch = _fetcher({"/files/a1/a1.html": html})
    assert video_check_plot.find_plot_artifact([_plot("a1")], set(), fetch) == "a1"


def test_two_bpms_only_are_rejected():
    html = '{"type": "scatter3d"} BPM1 BPM2 BPM1 BPM02'
    fetch = _fetcher({"/files/a1/a1.html": html})
    assert video_check_plot.find_plot_artifact([_plot("a1")], set(), fetch) is None


def test_bpm_label_with_trailing_digit_does_not_count():
    html = '{"type": "scatter3d"} BPM1 BPM2 BPM31'
    fetch = _fetcher({"/files/a1/a1.html": html})
    assert video_check_plot.find_plot_artifact([_plot("a1")], set(), fetch) is None


def test_pre_existing_ids_are_excluded():
    fetch = _fetcher({"/files/old/old.html": SCATTER3D_HTML, "/files/new/new.html": SCATTER3D_HTML})
    artifacts = [_plot("old"), _plot("new")]
    assert video_check_plot.find_plot_artifact(artifacts, {"old"}, fetch) == "new"
    assert fetch.calls == ["/files/new/new.html"]


def test_only_pre_existing_plot_returns_none():
    fetch = _fetcher({"/files/old/old.html": SCATTER3D_HTML})
    assert video_check_plot.find_plot_artifact([_plot("old")], {"old"}, fetch) is None


def test_fetch_error_skips_artifact():
    def fetch(path: str) -> str:
        if "bad" in path:
            raise OSError("connection refused")
        return SCATTER3D_HTML

    artifacts = [_plot("bad"), _plot("good")]
    assert video_check_plot.find_plot_artifact(artifacts, set(), fetch) == "good"


def test_file_path_is_url_quoted():
    artifact = _plot("a 1", filename="plot #1.html")
    assert video_check_plot.artifact_file_path(artifact) == "/files/a%201/plot%20%231.html"


# --- the correlation plot ------------------------------------------------------


# An inlined Plotly bundle names every trace type and words like "gauge";
# only the figure passed to Plotly.newPlot may count.
_BUNDLE = "<script>/* plotly.js */ var t={scatter3d:1,indicator:{gauge:1}};</script>"
_TEMPLATE = {"data": {"scatter3d": [{"type": "scatter3d"}]}}
_BPMS = [f"SR:DIAG:BPM:0{i}:POSITION:X" for i in range(1, 7)]
_GAUGE = "SR:VAC:GAUGE:SR01:PRESSURE:RB"


def _figure(traces, title="Sector 1 correlation") -> str:
    layout = {"title": {"text": title}, "template": _TEMPLATE}
    return (
        f'<html><head>{_BUNDLE}</head><body><div id="g"></div><script>'
        f'Plotly.newPlot("g", {json.dumps(traces)}, {json.dumps(layout)}, {{}});'
        "</script></body></html>"
    )


_HEATMAP = [{"type": "heatmap", "x": [*_BPMS, _GAUGE], "y": [*_BPMS, _GAUGE], "z": [[1]]}]
_CORR = _figure(_HEATMAP)


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (_CORR, True),
        (
            _figure(
                [{"type": "heatmap", "x": [*_BPMS, "vacuum pressure"], "y": _BPMS, "z": [[1]]}]
            ),
            True,
        ),
        (
            _figure([{"type": "splom", "dimensions": [{"label": b} for b in [*_BPMS, _GAUGE]]}]),
            True,
        ),
        (_figure([{"type": "heatmap", "x": _BPMS, "y": _BPMS, "z": [[1]]}]), False),
        (
            _figure([{"type": "heatmap", "x": [_BPMS[0], _GAUGE], "y": [_BPMS[0]], "z": [[1]]}]),
            False,
        ),
        (_figure([{"type": "scatter3d", "x": [1], "text": [*_BPMS, _GAUGE]}]), False),
        ("heatmap " + " ".join([*_BPMS, _GAUGE]), False),
    ],
    ids=["heatmap", "vacuum-word", "splom", "no-vacuum", "one-bpm", "3d", "no-figure"],
)
def test_a_correlation_plot_names_several_bpms_and_the_vacuum(html, expected) -> None:
    assert video_check_plot.html_is_bpm_vacuum_correlation(html) is expected


def test_the_bundle_and_template_never_make_a_plot_look_3d() -> None:
    assert video_check_plot.figure_trace_types(_CORR) == ["heatmap"]


def test_the_correlation_finder_skips_old_artifacts_and_the_3d_plot() -> None:
    artifacts = [
        {
            "id": "old",
            "filename": "o.html",
            "artifact_type": "plot_html",
            "tool_source": "create_interactive_plot",
        },
        {
            "id": "p3d",
            "filename": "p.html",
            "artifact_type": "plot_html",
            "tool_source": "create_interactive_plot",
        },
        {
            "id": "corr",
            "filename": "c.html",
            "artifact_type": "plot_html",
            "tool_source": "create_interactive_plot",
        },
    ]
    html = {"/files/old/o.html": _CORR, "/files/p3d/p.html": _CORR, "/files/corr/c.html": _CORR}
    found = video_check_plot.find_correlation_artifact(
        artifacts, {"old", "p3d"}, lambda path: html[path]
    )
    assert found == "corr"


def test_the_newest_of_several_correlation_plots_is_the_one_on_screen() -> None:
    # The agent may draw a scatter matrix and then a heatmap; the last one
    # landed is the one on screen, and the one the post must attach.
    def art(artifact_id: str, stamp: str) -> dict:
        return {
            "id": artifact_id,
            "filename": f"{artifact_id}.html",
            "artifact_type": "plot_html",
            "tool_source": "create_interactive_plot",
            "timestamp": stamp,
        }

    artifacts = [art("matrix", "2026-09-27T05:18:00"), art("heat", "2026-09-27T05:19:00")]
    found = video_check_plot.find_correlation_artifact(artifacts, set(), lambda _path: _CORR)
    assert found == "heat"
    found = video_check_plot.find_correlation_artifact(
        list(reversed(artifacts)), set(), lambda _path: _CORR
    )
    assert found == "heat"


# --- empty plots ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("traces", "empty"),
    [
        ([], True),
        ([{"type": "scatter", "x": [], "y": []}], True),
        ([{"type": "scatter3d", "x": [], "y": [], "z": []}], True),
        ([{"type": "heatmap", "z": [[]]}], True),
        ([{"type": "scatter", "x": [1], "y": [2]}], False),
        ([{"type": "heatmap", "z": [[1, 0], [0, 1]]}], False),
        ([{"type": "splom", "dimensions": [{"values": [1, 2]}]}], False),
        ([{"type": "scatter", "x": []}, {"type": "histogram", "x": [3]}], False),
    ],
)
def test_an_empty_plot_draws_no_data(traces, empty) -> None:
    assert video_check_plot.plot_is_empty(_figure(traces)) is empty


def test_html_without_a_figure_is_not_called_empty() -> None:
    # Not a Plotly page (a table, an image): nothing to judge.
    assert video_check_plot.plot_is_empty("<table><tr><td>1</td></tr></table>") is False
