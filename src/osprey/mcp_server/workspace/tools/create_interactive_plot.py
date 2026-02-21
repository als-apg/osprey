"""MCP tool: create_interactive_plot — execute Plotly code for interactive HTML charts.

Runs user-provided Plotly code with auto-imported data science libraries.
Output is produced exclusively via ``save_artifact(fig, "title")`` calls
in the user code — there is no auto-capture.
"""

import json
import logging

from osprey.mcp_server.common import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.mcp_server.workspace.tools._viz_common import (
    build_data_reader,
    collect_and_register_artifacts,
    save_to_data_context,
)

logger = logging.getLogger("osprey.mcp_server.tools.create_interactive_plot")

# Data-only preamble — no matplotlib imports
_INTERACTIVE_PREAMBLE = """\
import numpy as np
import pandas as pd
import scipy.stats as stats
"""


@mcp.tool()
async def create_interactive_plot(
    code: str,
    title: str,
    description: str = "",
    data_source: str | None = None,
) -> str:
    """Execute Plotly code and save results as interactive HTML artifacts.

    Runs Python code with auto-imported data libraries (numpy, pandas, scipy).
    You MUST call ``save_artifact(fig, "title")`` on your Plotly figure to
    produce output — figures are not auto-captured.

    Args:
        code: Python code using Plotly to create interactive charts.
        title: Human-readable title for the chart.
        description: Description of what the chart shows.
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
                ["Provide Python code that creates an interactive Plotly chart."],
            )
        )

    # Build the full code to execute
    parts = [_INTERACTIVE_PREAMBLE]

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
                f"Interactive plot creation failed: "
                f"{exec_result.error_message or exec_result.stderr}",
                [
                    "Check your Plotly code for syntax or runtime errors.",
                    "Ensure you call save_artifact(fig, 'title') to produce output.",
                    "Ensure data variables are defined or use data_source parameter.",
                ],
            )
        )

    # Collect artifacts from manifest (save_artifact() calls in user code)
    artifact_ids = collect_and_register_artifacts(
        exec_result, title, description, tool_source="create_interactive_plot"
    )

    # Save to DataContext
    response = save_to_data_context(
        tool="create_interactive_plot",
        title=title,
        description=description,
        code=code,
        artifact_ids=artifact_ids,
        stdout=exec_result.stdout,
        data_source=data_source,
        data_type="visualization",
    )
    return json.dumps(response, default=str)
