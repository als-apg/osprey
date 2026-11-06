"""Render the artifacts a scenario's past agent sessions left behind.

A scenario's ``sessions`` block states, turn by turn, which artifacts the agent
produced: a ``plot``, a ``note`` or a ``data`` record (:class:`ArtifactSpec`).
This module turns each one into the payload the gallery stores, in the gallery's
own forms, so a seeded artifact is indistinguishable from one the agent saved:

* ``plot`` -- a self-contained Plotly HTML page (``plot_html``) of the turn's
  channels. Its samples are read from the archive composite
  (:func:`osprey_connectors.simulation.archive.build`) at the same anchor and
  active set :func:`~osprey.simulation.apply.seed_archiver` writes the stored
  archive from, so the plot and the archive agree by construction rather than by
  a second copy of the numbers. The page carries no plotting library: the
  gallery serves Plotly itself and re-themes the figure, which therefore names
  no colours or fonts.
* ``note`` -- markdown, the ``content`` the turn's ``artifact_register`` call
  saved.
* ``data`` -- JSON, the result of the turn's ``execute`` call (or the JSON
  ``content`` of an ``artifact_register`` call).

Every payload is a pure function of the scenario, the active set and the anchor:
the same anchor gives the same bytes, so a reseed leaves the gallery unchanged.
The payload carries the seed file's content half (``filename``, ``title``,
``artifact_type``, ``mime_type``, ``kind``, ``tool_source``, ``content_b64``);
the seed-file writer adds the ids and the instant.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from osprey.simulation.sessions import (
        ArtifactSpec,
        ResolvedSession,
        ResolvedTurn,
        ToolStep,
    )
    from osprey_connectors.simulation.archive import ArchiveComposite

__all__ = [
    "PLOT_DIV_ID",
    "PLOT_MARGIN_S",
    "PLOT_STEP_S",
    "ArtifactPayload",
    "PlotSeries",
    "plot_series",
    "render_artifact",
    "render_session_artifacts",
]

#: Seconds of history shown either side of the turn's archiver events.
PLOT_MARGIN_S = 600.0
#: Seconds between two plotted samples: fine enough to resolve a 40 s spike.
PLOT_STEP_S = 2.0
#: The window a plot shows, ending at its turn, when its channels carry no event.
PLOT_QUIET_WINDOW_S = 1800.0
#: Plotly names a figure's ``div`` with a random uuid unless told otherwise.
PLOT_DIV_ID = "osprey-session-plot"

# The tools whose call produces each form, in the order they are preferred.
_PLOT_TOOLS = ("create_interactive_plot", "create_static_plot")
_NOTE_TOOLS = ("artifact_register",)
_DATA_TOOLS = ("execute", "artifact_register")

# Each form's gallery shape: artifact type, MIME type, file extension.
_GALLERY_FORMS = {
    "plot": ("plot_html", "text/html", ".html"),
    "note": ("markdown", "text/markdown", ".md"),
    "data": ("json", "application/json", ".json"),
}


@dataclass(frozen=True)
class ArtifactPayload:
    """One artifact as the gallery stores it.

    Attributes:
        filename: The stored file's name: the slugged title and the form's extension.
        title: The title the artifact is listed under.
        artifact_type: The gallery form: ``plot_html``, ``markdown`` or ``json``.
        mime_type: The content's MIME type.
        kind: The registered extension type; ``""`` for a base form.
        tool_source: The tool whose call produced it.
        content: The file's bytes.
    """

    filename: str
    title: str
    artifact_type: str
    mime_type: str
    kind: str
    tool_source: str
    content: bytes

    def fields(self) -> dict[str, str]:
        """The payload's seed-file fields, the content base64-encoded as ``content_b64``."""
        return {
            "filename": self.filename,
            "title": self.title,
            "artifact_type": self.artifact_type,
            "mime_type": self.mime_type,
            "kind": self.kind,
            "tool_source": self.tool_source,
            "content_b64": base64.b64encode(self.content).decode("ascii"),
        }


@dataclass(frozen=True)
class PlotSeries:
    """The samples a plot shows, exactly as the archive composite gives them.

    Attributes:
        t_s: Epoch seconds of every sample, ascending.
        values: Each channel's samples, one per timestamp, in turn order.
    """

    t_s: tuple[float, ...]
    values: Mapping[str, tuple[float, ...]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_session_artifacts(
    project_dir: Path,
    names: Sequence[str],
    sessions: Sequence[ResolvedSession],
    anchor: datetime,
) -> dict[tuple[str, str], ArtifactPayload]:
    """Render every artifact of the resolved sessions.

    Args:
        project_dir: Root of the built project; supplies the simulator view.
        names: The active scenario names, as ``seed_archiver`` is given them.
        sessions: The sessions, resolved against ``anchor``.
        anchor: The apply-time anchor T0 the scenario state file records: the
            one the archive is written from. Must be timezone-aware.

    Returns:
        Each artifact's payload, keyed by ``(session slug, artifact slug)``.

    Raises:
        ValueError: ``anchor`` is naive, or an artifact's turn carries nothing
            its form can be rendered from.
    """
    if anchor.tzinfo is None:
        raise ValueError(f"anchor must be timezone-aware, got {anchor!r}")
    turns = [
        (session, turn)
        for session in sessions
        for part in session.parts
        for turn in part.turns
        if turn.artifacts
    ]
    if not turns:
        return {}

    from osprey.simulation.apply import active_archiver_events
    from osprey_connectors.config import get_facility_timezone
    from osprey_connectors.simulation.archive import build
    from osprey_connectors.simulation.view import SimulatorView

    tz = get_facility_timezone()
    events = active_archiver_events(project_dir, names)
    payloads: dict[tuple[str, str], ArtifactPayload] = {}
    archive: ArchiveComposite | None = None
    try:
        for session, turn in turns:
            for artifact in turn.artifacts:
                if artifact.spec.form == "plot" and archive is None:
                    archive = build(
                        SimulatorView.of_project(project_dir), names, anchor_s=anchor.timestamp()
                    )
                where = f"sessions[{session.session.slug}].artifacts[{artifact.spec.slug}]"
                payloads[(session.session.slug, artifact.spec.slug)] = render_artifact(
                    artifact.spec,
                    turn,
                    archive=archive,
                    events=events,
                    anchor_s=anchor.timestamp(),
                    tz=tz,
                    where=where,
                )
    finally:
        if archive is not None:
            archive.close()
    return payloads


def render_artifact(
    spec: ArtifactSpec,
    turn: ResolvedTurn,
    *,
    archive: ArchiveComposite | None = None,
    events: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    anchor_s: float | None = None,
    tz: ZoneInfo | None = None,
    where: str | None = None,
) -> ArtifactPayload:
    """Render one artifact of a turn in its gallery form.

    Args:
        spec: The artifact.
        turn: The resolved turn that produced it.
        archive: The archive composite of the active set at the anchor; a
            ``plot`` reads its samples from it.
        events: The active set's archiver events by channel; a ``plot``
            centres its window on those of its channels.
        anchor_s: The anchor the archive was built at, in epoch seconds.
        tz: The facility zone the plot's time axis is drawn in (UTC when omitted).
        where: Location prefix for error messages.

    Raises:
        ValueError: The form is unknown, a ``plot`` has no archive, or the turn
            carries nothing the form can be rendered from.
    """
    where = where or f"artifact {spec.slug!r}"
    if spec.form == "plot":
        if archive is None or anchor_s is None:
            raise ValueError(
                f"{where}: a plot is rendered from the archive at its anchor, and none was given"
            )
        return _render_plot(
            spec, turn, archive, events or {}, anchor_s, tz or ZoneInfo("UTC"), where
        )
    if spec.form == "note":
        return _render_note(spec, turn, where)
    if spec.form == "data":
        return _render_data(spec, turn, where)
    raise ValueError(f"{where}: unknown artifact form {spec.form!r}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_series(
    archive: ArchiveComposite,
    turn: ResolvedTurn,
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    anchor_s: float,
    tz: ZoneInfo,
) -> PlotSeries:
    """The samples a turn's plot shows.

    The window spans the archiver events the turn's channels carry, widened by
    :data:`PLOT_MARGIN_S` either side and sampled every :data:`PLOT_STEP_S`
    from the first event's instant, so every event lands on a sample. A turn
    whose channels carry no event shows the :data:`PLOT_QUIET_WINDOW_S` before
    its prompt.

    Args:
        archive: The archive composite of the active set at ``anchor_s``.
        turn: The resolved turn the plot belongs to.
        events: The active set's archiver events, by channel.
        anchor_s: The anchor the archive was built at, in epoch seconds.
        tz: The facility zone an ``at_when`` event is placed in.

    Raises:
        ValueError: The turn names no channel the archive holds.
    """
    channels = [c for c in turn.turn.channels if c in set(archive.scalar_addresses)]
    if not channels:
        raise ValueError(
            f"turn at {turn.at.isoformat()}: none of its channels {list(turn.turn.channels)} "
            "is archived, so there is nothing to plot"
        )
    instants = _event_instants(events, channels, anchor_s, tz)
    if instants:
        first = min(instants)
        start = first - PLOT_MARGIN_S
        stop = max(instants) + PLOT_MARGIN_S
    else:
        stop = math.floor(turn.at.timestamp())
        start = stop - PLOT_QUIET_WINDOW_S
    count = int((stop - start) // PLOT_STEP_S) + 1
    t_s = tuple(start + index * PLOT_STEP_S for index in range(count))
    samples = archive.samples(channels, t_s)
    return PlotSeries(
        t_s=t_s,
        values={address: tuple(float(v) for v in samples[address]) for address in channels},
    )


def _event_instants(
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    channels: Sequence[str],
    anchor_s: float,
    tz: ZoneInfo,
) -> list[float]:
    """The instants of the anchored (``at_offset`` / ``at_when``) events on ``channels``."""
    from osprey_connectors.simulation.series import anchored_instant

    return [
        anchored_instant(dict(event), anchor_s, tz)
        for address in channels
        for event in events.get(address, ())
        if "at_offset" in event or "at_when" in event
    ]


def _render_plot(
    spec: ArtifactSpec,
    turn: ResolvedTurn,
    archive: ArchiveComposite,
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    anchor_s: float,
    tz: ZoneInfo,
    where: str,
) -> ArtifactPayload:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    try:
        series = plot_series(archive, turn, events, anchor_s, tz)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    axis = [datetime.fromtimestamp(t, tz).strftime("%Y-%m-%d %H:%M:%S") for t in series.t_s]
    channels = list(series.values)
    figure = make_subplots(specs=[[{"secondary_y": len(channels) > 1}]])
    for index, address in enumerate(channels):
        figure.add_trace(
            go.Scatter(x=axis, y=list(series.values[address]), mode="lines", name=address),
            secondary_y=index > 0,
        )
    figure.update_layout(title={"text": spec.title}, xaxis_title=f"Time ({tz.key})")
    if len(channels) > 1:
        # A second channel of a different magnitude reads on its own axis, as the
        # turn's own plotting code draws it.
        figure.update_yaxes(title_text=channels[0], secondary_y=False)
        figure.update_yaxes(title_text=channels[1], secondary_y=True)
    html = figure.to_html(include_plotlyjs=False, full_html=True, div_id=PLOT_DIV_ID)
    step = _producing_step(turn, spec, _PLOT_TOOLS)
    return _payload(spec, step.name if step is not None else _PLOT_TOOLS[0], html)


# ---------------------------------------------------------------------------
# Note and data
# ---------------------------------------------------------------------------


def _render_note(spec: ArtifactSpec, turn: ResolvedTurn, where: str) -> ArtifactPayload:
    step = _producing_step(turn, spec, _NOTE_TOOLS)
    content = step.input.get("content") if step is not None else None
    if step is None or not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"{where}: a note is the markdown 'content' of the turn's artifact_register "
            "call, and the turn has none"
        )
    return _payload(spec, step.name, content)


def _render_data(spec: ArtifactSpec, turn: ResolvedTurn, where: str) -> ArtifactPayload:
    step = _producing_step(turn, spec, _DATA_TOOLS)
    value: Any = None
    if step is not None:
        if step.name == "artifact_register":
            content = step.input.get("content")
            if isinstance(content, str):
                try:
                    value = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{where}: the registered content is not JSON: {exc}") from exc
            else:
                value = content
        else:
            value = step.result
    if step is None or not isinstance(value, Mapping | list):
        raise ValueError(
            f"{where}: a data artifact is the JSON result of the turn's execute call "
            "(or the JSON content of its artifact_register call), and the turn has none"
        )
    return _payload(spec, step.name, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _producing_step(
    turn: ResolvedTurn, spec: ArtifactSpec, tools: Sequence[str]
) -> ToolStep | None:
    """The turn's call that produced ``spec``: by tool preference, the one titled as it first."""
    from osprey.simulation.sessions import ToolStep

    calls = [step for step in turn.turn.steps if isinstance(step, ToolStep)]
    for tool in tools:
        matching = [step for step in calls if step.name == tool]
        for step in matching:
            if step.input.get("title") == spec.title:
                return step
        if matching:
            return matching[-1]
    return None


def _payload(spec: ArtifactSpec, tool_source: str, text: str) -> ArtifactPayload:
    """*spec*'s payload in its form's gallery shape; the file name is the slugged title."""
    from osprey.stores.artifact_store import _slugify

    artifact_type, mime_type, extension = _GALLERY_FORMS[spec.form]
    return ArtifactPayload(
        filename=f"{_slugify(spec.title)}{extension}",
        title=spec.title,
        artifact_type=artifact_type,
        mime_type=mime_type,
        kind="",
        tool_source=tool_source,
        content=text.encode("utf-8"),
    )
