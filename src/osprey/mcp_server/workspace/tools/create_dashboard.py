"""MCP tool: create_dashboard — generate standalone interactive HTML dashboards.

Executes Bokeh plotting code in a sandboxed subprocess. Output is produced
exclusively via ``save_artifact(layout, "title")`` calls in user code,
which triggers the Bokeh detection branch in ``save_artifact()`` to
serialize the layout to a self-contained HTML file. No server subprocess
needed, no stdout markers, no auto-detection of variable names.
"""

import importlib
import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.mcp_server.workspace.tools._viz_common import (
    build_viz_response,
    collect_and_register_artifacts,
)

logger = logging.getLogger("osprey.mcp_server.tools.create_dashboard")

# Bokeh + data science imports prepended to dashboard code
_DASHBOARD_PREAMBLE = """\
import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot, layout
from bokeh.models import (
    ColumnDataSource, HoverTool, Slider, Select, Div,
    ColorBar, LinearColorMapper, Range1d, Legend, LegendItem,
)
from bokeh.palettes import Spectral6, Category10, Viridis256
from bokeh.resources import CDN
"""


@mcp.tool()
async def create_dashboard(
    code: str,
    title: str,
    description: str = "",
) -> str:
    """Create an interactive Bokeh dashboard as a standalone HTML artifact.

    Executes Python code that creates Bokeh figures and layouts. You MUST call
    ``save_artifact(layout, "title")`` on your Bokeh layout object to produce
    output — layouts are not auto-detected or auto-saved.

    The code has access to Bokeh imports (figure, layouts, models, palettes)
    plus numpy and pandas.

    Args:
        code: Python code using Bokeh to create figures/layouts.
        title: Dashboard title.
        description: Description of the dashboard.

    Returns:
        JSON with artifact_ids and context_entry_id.
    """
    if not code or not code.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "No dashboard code provided.",
                ["Provide Python code that creates a Bokeh dashboard."],
            )
        )

    # Check Bokeh availability before attempting execution
    try:
        importlib.import_module("bokeh")
    except ImportError:
        return json.dumps(
            make_error(
                "dependency_missing",
                "Bokeh is not installed. Interactive dashboards require the bokeh package.",
                [
                    "Install with: uv sync",
                    "Or use create_interactive_plot with Plotly for interactive HTML charts.",
                    "Plotly figures are self-contained HTML with interactivity.",
                ],
            )
        )

    # Build full code: preamble + user code (no epilogue)
    full_code = _DASHBOARD_PREAMBLE + "\n" + code

    # Execute via the sandboxed executor
    from osprey.mcp_server.workspace.execution.sandbox_executor import (
        create_sandbox_execution_folder,
        execute_sandbox_code,
    )

    execution_folder = create_sandbox_execution_folder()
    exec_result = await execute_sandbox_code(
        code=full_code, execution_folder=execution_folder
    )

    if not exec_result.success:
        return json.dumps(
            make_error(
                "execution_error",
                f"Dashboard creation failed: {exec_result.error_message or exec_result.stderr}",
                [
                    "Check your Bokeh code for syntax or runtime errors.",
                    "Ensure you call save_artifact(layout, 'title') to produce output.",
                    "Ensure bokeh is installed: uv sync",
                ],
            )
        )

    # Collect artifacts with category and embedded metadata
    artifact_ids = collect_and_register_artifacts(
        exec_result, title, description,
        tool_source="create_dashboard",
        category="dashboard",
        code=code,
        stdout=exec_result.stdout,
    )

    if not artifact_ids:
        return json.dumps(
            make_error(
                "execution_error",
                "Dashboard code ran but produced no artifacts.",
                [
                    "Ensure your code calls save_artifact(layout, 'title') on a Bokeh layout.",
                    "Example: save_artifact(column(p1, p2), 'My Dashboard')",
                ],
            )
        )

    return json.dumps(
        build_viz_response(artifact_ids, title, exec_result.stdout),
        default=str,
    )
