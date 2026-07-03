"""Phoebus Data Browser data models.

Pydantic models for configuring Data Browser plots, traces, and annotations.

Migrated from the retired ``phoebus_launch`` server
(``als-profiles/mcp_servers/phoebus/models.py``) as part of reincarnating its
``.plt``-generation capability inside the native ``phoebus_open_databrowser``
tool (see ``.claude/plans/phoebus-als-integration/ADDENDUM-databrowser-reincarnation.md``,
Task 0.5). Facility-neutral: nothing here names a specific facility, archiver,
or host — that binding is supplied by the caller (see ``plt_generator``).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TraceType(Enum):
    """Available trace types for Data Browser plots."""

    LINE = "LINE"
    AREA = "AREA"
    STEP = "STEP"
    BARS = "BARS"


class LineStyle(Enum):
    """Available line styles for traces."""

    SOLID = "SOLID"
    DASH = "DASH"
    DOT = "DOT"
    DASHDOT = "DASHDOT"


class PointType(Enum):
    """Available point types for traces."""

    NONE = "NONE"
    CIRCLE = "CIRCLE"
    SQUARE = "SQUARE"
    DIAMOND = "DIAMOND"
    TRIANGLE = "TRIANGLE"


class PVConfig(BaseModel):
    """Configuration for a single PV trace in the Data Browser."""

    name: str = Field(description="PV name (e.g., SR:DCCT)")
    display_name: str | None = Field(
        default=None, description="Human-readable display name for the PV"
    )
    color_red: int = Field(default=0, description="Red component of RGB color (0-255)")
    color_green: int = Field(default=100, description="Green component of RGB color (0-255)")
    color_blue: int = Field(default=200, description="Blue component of RGB color (0-255)")
    trace_type: TraceType = Field(
        default=TraceType.LINE, description="Type of trace (LINE, AREA, STEP, BARS)"
    )
    line_style: LineStyle = Field(
        default=LineStyle.SOLID, description="Line style (SOLID, DASH, DOT, DASHDOT)"
    )
    line_width: int = Field(default=2, description="Line width in pixels")
    point_type: PointType = Field(default=PointType.NONE, description="Point marker type")
    point_size: int = Field(default=2, description="Point marker size")
    axis: int = Field(default=0, description="Which axis to use (0=left, 1=right, etc.)")

    @property
    def color(self) -> tuple[int, int, int]:
        """Get color as RGB tuple."""
        return (self.color_red, self.color_green, self.color_blue)


class TimeRange(BaseModel):
    """Time range specification for the Data Browser."""

    start: str | datetime = Field(
        description="Start time - can be datetime, 'now', '-2 hours', etc."
    )
    end: str | datetime = Field(description="End time - can be datetime, 'now', etc.")


class AnnotationConfig(BaseModel):
    """Configuration for a plot annotation."""

    text: str = Field(description="Annotation text content")
    time_position: str | datetime = Field(description="Time position for annotation")
    value_position: float = Field(description="Value position for annotation")
    offset_x: float = Field(default=20.0, description="X offset from position")
    offset_y: float = Field(default=20.0, description="Y offset from position")
    pv_index: int = Field(default=0, description="Index of PV this annotation refers to")


class PlotConfig(BaseModel):
    """High-level configuration for a Data Browser plot."""

    title: str = Field(description="Title for the plot")
    pvs: list[PVConfig] = Field(description="List of PV configurations to plot")
    time_range: TimeRange | None = Field(default=None, description="Time range for the plot")
    annotations: list[AnnotationConfig] = Field(
        default_factory=list, description="Plot annotations"
    )

    # Plot appearance
    background_red: int = Field(default=255, description="Background red component (0-255)")
    background_green: int = Field(default=255, description="Background green component (0-255)")
    background_blue: int = Field(default=255, description="Background blue component (0-255)")
    foreground_red: int = Field(default=0, description="Foreground red component (0-255)")
    foreground_green: int = Field(default=0, description="Foreground green component (0-255)")
    foreground_blue: int = Field(default=0, description="Foreground blue component (0-255)")
    show_grid: bool = Field(default=True, description="Whether to show grid lines")
    show_legend: bool = Field(default=True, description="Whether to show legend")
    show_toolbar: bool = Field(default=True, description="Whether to show toolbar")

    # Time axis
    scroll: bool = Field(default=True, description="Whether to enable scrolling")
    update_period: float = Field(default=3.0, description="Update period in seconds")

    # Axis configuration
    axis_name: str = Field(default="Values", description="Name for the axis")
    auto_scale: bool = Field(default=True, description="Whether to auto-scale the axis")
    axis_min: float | None = Field(
        default=None, description="Minimum axis value (if not auto-scale)"
    )
    axis_max: float | None = Field(
        default=None, description="Maximum axis value (if not auto-scale)"
    )
    log_scale: bool = Field(default=False, description="Whether to use logarithmic scale")

    @property
    def background_color(self) -> tuple[int, int, int]:
        """Get background color as RGB tuple."""
        return (self.background_red, self.background_green, self.background_blue)

    @property
    def foreground_color(self) -> tuple[int, int, int]:
        """Get foreground color as RGB tuple."""
        return (self.foreground_red, self.foreground_green, self.foreground_blue)
