"""MCP tool: bring up a live Phoebus Data Browser for a PV list + time range.

Reincarnates the ``.plt``-generation capability of the retired
``phoebus_launch`` server (``als-profiles/mcp_servers/phoebus/``) as a
first-class action of the native Phoebus agent — see
``.claude/plans/phoebus-als-integration/ADDENDUM-databrowser-reincarnation.md``,
Task 0.5. Two things changed on migration:

* The embedded LLM styling call is gone. The calling agent is already an LLM
  — it supplies styling as structured tool arguments (``styling=``) instead
  of a natural-language ``query`` that used to be sent to a second model.
* Live-open instead of a ``myapp://`` hand-off. This tool writes a styled
  ``.plt`` into the workspace (reusing ``plt_generator``) and POSTs it to the
  agent bridge's ``POST /open`` — the same bridge-client path
  ``phoebus_open_panel`` uses (``bridge_tools._http_post_open``) — so the
  Data Browser opens directly in the shared control-room Phoebus.

Facility-neutral archiver binding: the archiver-appliance URL bound into the
generated ``.plt`` is never hardcoded here — it is resolved from
``PHOEBUS_ARCHIVER_URL`` (env) or ``phoebus.archiver_url`` (config.yml), and
omitted entirely (live-only PVs, no historical backfill) when neither is set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import anyio

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.phoebus import plt_generator
from osprey.mcp_server.phoebus.models import (
    AnnotationConfig,
    LineStyle,
    PlotConfig,
    PointType,
    PVConfig,
    TimeRange,
    TraceType,
)
from osprey.mcp_server.phoebus.server import mcp
from osprey.mcp_server.phoebus.tools.bridge_tools import (
    _UNREACHABLE_HINTS,
    _bridge_error_message,
    _http_post_open,
)
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.tools.phoebus_databrowser")

# Default color palette for multiple PVs — parity with the retired server's
# DEFAULT_COLORS rotation (phoebus_launch.py).
_DEFAULT_COLORS: list[tuple[int, int, int]] = [
    (0, 100, 200),  # Blue
    (200, 0, 0),  # Red
    (0, 150, 0),  # Green
    (200, 100, 0),  # Orange
    (150, 0, 150),  # Purple
    (0, 150, 150),  # Teal
    (150, 150, 0),  # Olive
    (100, 100, 100),  # Gray
]

_MAX_TITLE_CHANNELS = 5  # channels named in the auto-generated title before "+N more"


def _archiver_url() -> str | None:
    """Resolve the archiver-appliance URL bound into generated ``.plt`` files.

    Facility-neutral: no default archiver is baked in here (see module
    docstring). Resolution order:

    1. ``PHOEBUS_ARCHIVER_URL`` env var — wins outright when set.
    2. ``phoebus.archiver_url`` in config.yml.
    3. ``None`` — PVs are emitted without an ``<archive>`` binding.
    """
    env = os.environ.get("PHOEBUS_ARCHIVER_URL", "").strip()
    if env:
        return env
    config = load_osprey_config()
    value = config.get("phoebus", {}).get("archiver_url")
    return str(value) if value else None


def _plot_dir() -> Path:
    """Resolve the directory generated ``.plt`` files are written to."""
    config = load_osprey_config()
    out = Path(config.get("phoebus", {}).get("plot_dir", "./_agent_data/plots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _default_title(channels: list[str]) -> str:
    if len(channels) <= _MAX_TITLE_CHANNELS:
        names = ", ".join(channels)
    else:
        shown = ", ".join(channels[:_MAX_TITLE_CHANNELS])
        names = f"{shown} +{len(channels) - _MAX_TITLE_CHANNELS} more"
    return f"Data Browser: {names}"


def _build_plot_config(
    channels: list[str],
    title: str | None,
    start_time: str,
    end_time: str,
    styling: dict | None,
) -> PlotConfig:
    """Translate tool arguments into a ``PlotConfig``, applying the retired
    server's defaults (color rotation, appearance) wherever ``styling``
    leaves a value unspecified."""
    styling = styling or {}
    per_pv: dict = styling.get("pvs", {})

    pvs: list[PVConfig] = []
    for i, name in enumerate(channels):
        pv_style: dict = per_pv.get(name, {})
        color = pv_style.get("color")
        if color is None:
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        pvs.append(
            PVConfig(
                name=name,
                display_name=pv_style.get("display_name", name),
                color_red=color[0],
                color_green=color[1],
                color_blue=color[2],
                trace_type=TraceType(pv_style.get("trace_type", TraceType.LINE.value)),
                line_style=LineStyle(pv_style.get("line_style", LineStyle.SOLID.value)),
                line_width=pv_style.get("line_width", 2),
                point_type=PointType(pv_style.get("point_type", PointType.NONE.value)),
                point_size=pv_style.get("point_size", 2),
                axis=pv_style.get("axis", 0),
            )
        )

    bg = styling.get("background", (255, 255, 255))
    fg = styling.get("foreground", (0, 0, 0))
    annotations = [AnnotationConfig(**a) for a in styling.get("annotations", [])]

    return PlotConfig(
        title=title or _default_title(channels),
        pvs=pvs,
        time_range=TimeRange(start=start_time, end=end_time),
        annotations=annotations,
        background_red=bg[0],
        background_green=bg[1],
        background_blue=bg[2],
        foreground_red=fg[0],
        foreground_green=fg[1],
        foreground_blue=fg[2],
        show_grid=styling.get("show_grid", True),
        show_legend=styling.get("show_legend", True),
        show_toolbar=styling.get("show_toolbar", True),
        scroll=styling.get("scroll", True),
        update_period=styling.get("update_period", 3.0),
        axis_name=styling.get("axis_name", "Values"),
        auto_scale=styling.get("auto_scale", True),
        axis_min=styling.get("axis_min"),
        axis_max=styling.get("axis_max"),
        log_scale=styling.get("log_scale", False),
    )


@mcp.tool()
async def phoebus_open_databrowser(
    channels: list[str],
    start_time: str = "-24 hours",
    end_time: str = "now",
    title: str | None = None,
    styling: dict | None = None,
) -> str:
    """Bring up a live Phoebus Data Browser for a PV list and time range.

    Generates a styled Data Browser ``.plt`` (archiver-bound when an archiver
    URL is configured) and opens it in the shared Phoebus via the agent
    bridge's ``POST /open``, returning a handle so the result can be
    perceived visually with ``phoebus_snapshot`` (a Data Browser is a JavaFX
    chart, not a widget tree — ``phoebus_perceive``/``phoebus_drive`` do not
    apply to it).

    Args:
        channels: List of PV/channel names to plot. Required, non-empty.
        start_time: Start of the time range (e.g. ``"-24 hours"``,
            ``"2024-01-01 00:00:00"``).
        end_time: End of the time range (e.g. ``"now"``).
        title: Plot title. Defaults to a title listing the channels.
        styling: Optional structured styling, supplied by the calling agent
            (no embedded LLM styling call is made here):
            ``{"pvs": {"<channel>": {"color": [r,g,b], "display_name": str,
            "trace_type": "LINE"|"AREA"|"STEP"|"BARS",
            "line_style": "SOLID"|"DASH"|"DOT"|"DASHDOT", "line_width": int,
            "point_type": "NONE"|"CIRCLE"|"SQUARE"|"DIAMOND"|"TRIANGLE",
            "point_size": int, "axis": int}, ...},
            "background": [r,g,b], "foreground": [r,g,b],
            "show_grid": bool, "show_legend": bool, "show_toolbar": bool,
            "scroll": bool, "update_period": float, "axis_name": str,
            "auto_scale": bool, "axis_min": float, "axis_max": float,
            "log_scale": bool, "annotations": [...]}``. Any key omitted uses
            the retired server's original default.

    Returns:
        JSON ``{"status": "success", "handle": "handle:d-N", "plt_file":
        "<path>", "id": "d-N", "ready": <bool>, "channel_count": <int>}``.
    """
    if not channels:
        make_error(
            "validation_error",
            "No channels provided.",
            ["Pass at least one PV/channel name in 'channels'."],
        )

    plot_config = _build_plot_config(channels, title, start_time, end_time, styling)
    plt_path = plt_generator.create_plt_from_config(
        plot_config,
        workspace_dir=_plot_dir(),
        archiver_url=_archiver_url(),
    )

    try:
        status, body = await anyio.to_thread.run_sync(_http_post_open, {"resource": plt_path})
    except Exception as exc:
        make_error(
            "phoebus_unreachable",
            f"Could not reach the Phoebus bridge: {exc}",
            _UNREACHABLE_HINTS,
        )

    if status != 200:
        make_error(
            "phoebus_open_failed",
            _bridge_error_message(body, status),
            [
                "Confirm the Phoebus bridge routes .plt resources to the Data "
                "Browser application (app-aware POST /open)."
            ],
        )

    display_id: str = body.get("id", "")
    if not display_id:
        make_error(
            "phoebus_open_failed",
            "Phoebus bridge returned a success response with no display id.",
            ["This may indicate a bridge version mismatch; check the bridge logs."],
        )

    return json.dumps(
        {
            "status": "success",
            "handle": f"handle:{display_id}",
            "plt_file": plt_path,
            "id": display_id,
            "ready": bool(body.get("ready", False)),
            "channel_count": len(channels),
        }
    )
