"""Recognize the demo take's 3D BPM plot among the artifact server's artifacts.

The demo prompt asks the agent for an interactive 3D scatter of three BPMs. A
take counts only when a *new* artifact (one not present before the prompt) is
an interactive plot whose HTML is a ``scatter3d`` trace naming three distinct
BPMs. The HTML fetch is injected so this module stays network-free and testable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from urllib.parse import quote

PLOT_ARTIFACT_TYPE = "plot_html"
PLOT_TOOL_SOURCE = "create_interactive_plot"
TRACE_MARKER = "scatter3d"
BPM_LABEL = re.compile(r"BPM\W*0?([1-3])\b")
REQUIRED_BPMS = 3


def artifact_file_path(artifact: dict) -> str:
    """Return the artifact server path serving ``artifact``'s file content."""
    artifact_id = quote(str(artifact.get("id") or ""), safe="")
    filename = quote(str(artifact.get("filename") or ""), safe="")
    return f"/files/{artifact_id}/{filename}"


def html_is_bpm_scatter3d(html: str) -> bool:
    """True when ``html`` holds a 3D scatter naming three distinct BPMs."""
    if TRACE_MARKER not in html:
        return False
    return len(set(BPM_LABEL.findall(html))) >= REQUIRED_BPMS


# The correlation prompt asks for every horizontal BPM of the first sector
# together with the vacuum: the plot must name several BPMs and a vacuum
# reading, and it is a 2D chart (a heatmap or a scatter matrix), never the
# 3D scatter of the first prompt. Only the figure handed to Plotly.newPlot is
# read: a saved plot may inline the whole Plotly library, and its layout
# template names every trace type, 3D included.
ANY_BPM_LABEL = re.compile(r"BPM\W*0?(\d{1,2})\b")
CORRELATION_MIN_BPMS = 3
VACUUM_LABEL = re.compile(r"\bVAC\b|vacuum|pressure|gauge", re.IGNORECASE)
_NEW_PLOT = re.compile(r"Plotly\.newPlot\(\s*(?:\"[^\"]*\"|'[^']*'|[\w.]+)\s*,\s*")


def _figure(html: str) -> tuple[list, dict] | None:
    """The data and layout of the first ``Plotly.newPlot`` call in ``html``."""
    decoder = json.JSONDecoder()
    for match in _NEW_PLOT.finditer(html):
        try:
            data, end = decoder.raw_decode(html, match.end())
            rest = html[end:].lstrip()
            layout = decoder.raw_decode(rest[1:].lstrip())[0] if rest.startswith(",") else {}
        except ValueError:
            continue
        if isinstance(data, list):
            return data, layout if isinstance(layout, dict) else {}
    return None


# Where a trace keeps its data, by the traces the agent draws.
_DATA_KEYS = ("x", "y", "z", "values", "r", "lat", "lon", "open", "cells")


def _has_data(value: object) -> bool:
    if isinstance(value, list):
        return (
            any(_has_data(v) for v in value)
            if value and isinstance(value[0], list)
            else bool(value)
        )
    if isinstance(value, dict):  # Plotly's typed-array encoding: {"dtype": ..., "bdata": ...}
        return bool(value.get("bdata") or value.get("_inputArray"))
    return False


def plot_is_empty(html: str) -> bool:
    """True when ``html`` is a Plotly figure whose traces draw no data at all.

    A page that holds no figure is not called empty: there is nothing to judge.
    """
    figure = _figure(html)
    if figure is None:
        return False
    for trace in figure[0]:
        if not isinstance(trace, dict):
            continue
        if any(_has_data(trace.get(key)) for key in _DATA_KEYS):
            return False
        dimensions = trace.get("dimensions")
        if isinstance(dimensions, list) and any(
            isinstance(d, dict) and _has_data(d.get("values")) for d in dimensions
        ):
            return False
    return True


def figure_trace_types(html: str) -> list[str] | None:
    """The trace types the figure draws, or None when ``html`` holds no figure."""
    figure = _figure(html)
    return None if figure is None else _trace_types(figure[0])


def _trace_types(data: list) -> list[str]:
    return [str(t.get("type") or "scatter") for t in data if isinstance(t, dict)]


def html_is_bpm_vacuum_correlation(html: str) -> bool:
    """True when ``html`` is a 2D plot naming several BPMs and a vacuum reading."""
    figure = _figure(html)
    if figure is None:
        return False
    data, layout = figure
    types = _trace_types(data)
    if not types or TRACE_MARKER in types:
        return False
    drawn = json.dumps(data) + json.dumps({k: v for k, v in layout.items() if k != "template"})
    if len(set(ANY_BPM_LABEL.findall(drawn))) < CORRELATION_MIN_BPMS:
        return False
    return VACUUM_LABEL.search(drawn) is not None


def find_correlation_artifact(
    artifacts: Iterable[dict],
    before_ids: set[str],
    fetch_html: Callable[[str], str],
) -> str | None:
    """Return the id of the newest new artifact that is the BPM–vacuum correlation plot.

    The agent may draw more than one; the newest is the one left on screen.
    """
    newest_first = sorted(artifacts, key=lambda a: str(a.get("timestamp") or ""), reverse=True)
    return find_plot_artifact(newest_first, before_ids, fetch_html, html_is_bpm_vacuum_correlation)


def find_plot_artifact(
    artifacts: Iterable[dict],
    before_ids: set[str],
    fetch_html: Callable[[str], str],
    matches: Callable[[str], bool] = html_is_bpm_scatter3d,
) -> str | None:
    """Return the id of the first new plot artifact whose HTML ``matches``, else None.

    ``matches`` defaults to the 3D BPM scatter. ``artifacts`` is the
    ``/api/artifacts`` list; ``before_ids`` are the ids that existed before the
    prompt and are never matched. ``fetch_html`` receives the
    ``/files/{id}/{filename}`` path and returns the file's HTML; an ``OSError``
    from it (including ``urllib.error.URLError``) skips that artifact.
    """
    for artifact in artifacts:
        artifact_id = artifact.get("id")
        if not artifact_id or artifact_id in before_ids:
            continue
        if artifact.get("artifact_type") != PLOT_ARTIFACT_TYPE:
            continue
        if artifact.get("tool_source") != PLOT_TOOL_SOURCE:
            continue
        try:
            html = fetch_html(artifact_file_path(artifact))
        except OSError:
            continue
        if matches(html or ""):
            return artifact_id
    return None
