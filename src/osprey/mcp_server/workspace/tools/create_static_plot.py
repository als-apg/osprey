"""MCP tool: create_static_plot — execute matplotlib/seaborn plotting code.

Runs user-provided static plotting code with auto-imported data science
libraries and default styling. Output is produced exclusively via
``save_artifact(fig, "title")`` calls in the user code — there is no
auto-capture of matplotlib figures.
"""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.mcp_server.workspace.tools._viz_common import (
    build_data_reader,
    build_viz_response,
    collect_and_register_artifacts,
)

logger = logging.getLogger("osprey.mcp_server.tools.create_static_plot")

# Matplotlib-specific preamble
_STATIC_PREAMBLE = """\
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats

# Default styling
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'image.cmap': 'viridis',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.autolayout': True,
})
"""


@mcp.tool()
async def create_static_plot(
    code: str,
    title: str,
    description: str = "",
    data_source: str | None = None,
) -> str:
    """Execute matplotlib/seaborn plotting code and save results as PNG artifacts.

    Runs Python plotting code with auto-imported libraries (numpy, pandas,
    matplotlib, scipy) and default styling. You MUST call
    ``save_artifact(fig, "title")`` in your code to produce output — figures
    are not auto-captured.

    Args:
        code: Python plotting code using matplotlib or seaborn.
        title: Human-readable title for the plot.
        description: Description of what the plot shows.
        data_source: Optional data reference to auto-load as the ``data``
            variable before your code runs. Accepts three forms:
            (1) context entry ID (numeric string, e.g. "2"),
            (2) artifact ID (12-char hex), or
            (3) workspace file path.
            When provided, ``data`` is a **pandas DataFrame** — do NOT
            re-parse or re-unwrap it; use it directly (e.g. ``data.columns``,
            ``data['col']``).

    Returns:
        JSON with artifact_ids, context_entry_id, and preview info.
    """
    if not code or not code.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "No plotting code provided.",
                ["Provide Python code that creates a static matplotlib/seaborn plot."],
            )
        )

    # Build the full code to execute
    parts = [_STATIC_PREAMBLE]

    if data_source:
        parts.append(build_data_reader(data_source))

    parts.append(code)

    full_code = "\n".join(parts)

    # Execute via the sandboxed executor
    from osprey.mcp_server.workspace.execution.sandbox_executor import (
        create_sandbox_execution_folder,
        execute_sandbox_code,
    )

    execution_folder = create_sandbox_execution_folder()
    exec_result = await execute_sandbox_code(code=full_code, execution_folder=execution_folder)

    if not exec_result.success:
        return json.dumps(
            make_error(
                "execution_error",
                f"Static plot creation failed: {exec_result.error_message or exec_result.stderr}",
                [
                    "Check your plotting code for syntax or runtime errors.",
                    "Ensure you call save_artifact(fig, 'title') to produce output.",
                    "Ensure data variables are defined or use data_source parameter.",
                ],
            )
        )

    # Collect artifacts with category and embedded metadata
    artifact_ids = collect_and_register_artifacts(
        exec_result,
        title,
        description,
        tool_source="create_static_plot",
        category="visualization",
        code=code,
        stdout=exec_result.stdout,
        data_source=data_source,
    )

    return json.dumps(
        build_viz_response(artifact_ids, title, exec_result.stdout),
        default=str,
    )
