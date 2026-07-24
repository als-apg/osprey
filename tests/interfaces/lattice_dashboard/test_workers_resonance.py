"""Tests for the resonance-diagram worker (tune diagram)."""

from __future__ import annotations

from osprey.interfaces.lattice_dashboard.workers.resonance import (
    ORDER_COLORS,
    _add_resonance_lines,
    build_figure,
)


class TestAddResonanceLines:
    """_add_resonance_lines draws colored order-N lines plus legend proxies."""

    def test_lines_and_legend_entries(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        _add_resonance_lines(fig, (0.2, 0.4), (0.0, 0.2))

        # Legend proxy traces: exactly one per order, showlegend=True
        legend_traces = [t for t in fig.data if t.showlegend]
        assert len(legend_traces) == len(ORDER_COLORS)
        # Every drawn resonance line uses a known order color
        line_colors = {t.line.color for t in fig.data if t.mode == "lines"}
        assert line_colors.issubset(set(ORDER_COLORS.values()))

    def test_degenerate_range_still_adds_legend(self):
        import plotly.graph_objects as go

        fig = go.Figure()
        _add_resonance_lines(fig, (0.3, 0.3), (0.3, 0.3))
        # No resonance lines fit, but the 5 legend proxies are always added
        assert len([t for t in fig.data if t.showlegend]) == len(ORDER_COLORS)


class TestResonanceBuildFigure:
    """build_figure marks the working point and (optionally) the baseline."""

    def test_working_point_marker_is_themed(self):
        fig = build_figure(0.34, 0.22)
        wp = [t for t in fig.data if t.name == "Working point"]
        assert len(wp) == 1
        # Marker color is left to the client themer via the meta tag
        assert wp[0].meta == "themed-fg-marker"
        assert wp[0].x[0] == 0.34
        assert wp[0].y[0] == 0.22

    def test_axis_range_centered_on_working_point(self):
        fig = build_figure(0.34, 0.22)
        assert list(fig.layout.xaxis.range) == [0.34 - 0.1, 0.34 + 0.1]
        assert list(fig.layout.yaxis.range) == [0.22 - 0.1, 0.22 + 0.1]

    def test_no_baseline_marker_by_default(self):
        fig = build_figure(0.34, 0.22)
        assert not any(t.name == "Baseline" for t in fig.data)

    def test_baseline_marker_added(self):
        fig = build_figure(0.34, 0.22, baseline_nux=0.35, baseline_nuy=0.23)
        baseline = [t for t in fig.data if t.name == "Baseline"]
        assert len(baseline) == 1
        assert baseline[0].x[0] == 0.35
        assert baseline[0].y[0] == 0.23
