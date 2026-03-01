"""MCP tools: graph analysis — extract, compare, and save reference datasets.

Three tools for graph data extraction and comparison:
  - ``graph_extract``: Send a chart image to DePlot for data extraction.
  - ``graph_compare``: Compare two datasets using mathematical metrics.
  - ``graph_save_reference``: Save a dataset as a named reference for future comparisons.
"""

import json
import logging
from pathlib import Path

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.graph_tools")


# ---------------------------------------------------------------------------
# Tool 1: graph_extract
# ---------------------------------------------------------------------------
@mcp.tool()
async def graph_extract(
    image_path: str,
    preprocess: bool = True,
    title: str | None = None,
) -> str:
    """Extract numerical data from a chart/graph image using the DePlot service.

    Sends the image to the DePlot service for AI-based data extraction.
    The extracted data is saved to DataContext for comparison and analysis.

    Args:
        image_path: Path to the chart image file (PNG, JPEG).
        preprocess: Apply OpenCV preprocessing (chart detection, contrast
            enhancement) before extraction. Default True.
        title: Optional descriptive title for the extraction.

    Returns:
        JSON with extracted data summary and context_entry_id.
        Use the data_file path for full data access.
    """
    # Validate image path
    path = Path(image_path)
    if not path.exists():
        return json.dumps(
            make_error(
                "validation_error",
                f"Image file not found: {image_path}",
                ["Check the path and ensure the file exists."],
            )
        )

    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        return json.dumps(
            make_error(
                "validation_error",
                f"Unsupported image format: {path.suffix}",
                ["Supported formats: PNG, JPEG, BMP, TIFF."],
            )
        )

    # Check DePlot service availability
    from osprey.mcp_server.workspace.tools.graph_client import DePlotClient

    client = DePlotClient()
    if not await client.is_available():
        return json.dumps(
            make_error(
                "service_unavailable",
                "DePlot service is not running.",
                [
                    "Start it with: uv run python -m osprey.services.deplot",
                    "First run will download the model (~1GB).",
                    "Default port: 8095 (configurable in config.yml under deplot.port).",
                ],
            )
        )

    # Extract data
    try:
        result = await client.extract(str(path), preprocess=preprocess)
    except Exception as exc:
        logger.exception("graph_extract: DePlot extraction failed")
        return json.dumps(
            make_error(
                "extraction_error",
                f"DePlot extraction failed: {exc}",
                [
                    "Check that the image contains a readable chart.",
                    "Try with preprocess=False if the image is already clean.",
                ],
            )
        )

    # Save to ArtifactStore (unified)
    from osprey.mcp_server.artifact_store import get_artifact_store

    desc = title or f"Graph extraction from {path.name}"
    columns = result.get("columns", [])
    data = result.get("data", [])

    store = get_artifact_store()
    entry = store.save_data(
        tool="graph_extract",
        data={
            "columns": columns,
            "data": data,
            "raw_table": result.get("raw_table", ""),
            "title": result.get("title", title or ""),
            "source_image": str(path),
            "preprocessed": preprocess,
        },
        title=desc,
        description=desc,
        summary={
            "title": desc,
            "columns": columns,
            "num_points": len(data),
            "source_image": str(path),
        },
        access_details={
            "format": "tabular",
            "columns": columns,
            "num_points": len(data),
            "hint": "Read the data file for full extracted values.",
        },
        category="graph_extraction",
    )

    return json.dumps(entry.to_tool_response(), default=str)


# ---------------------------------------------------------------------------
# Tool 2: graph_compare
# ---------------------------------------------------------------------------
@mcp.tool()
async def graph_compare(
    current_entry_id: str,
    reference_entry_id: str | None = None,
    reference_query: str | None = None,
    metrics: list[str] | None = None,
) -> str:
    """Compare two graph datasets using mathematical similarity metrics.

    Computes RMSE, correlation, DTW distance, and peak shift between a
    current dataset and a reference. Works with data from graph_extract,
    archiver_read, or any artifact entry with tabular data.

    Args:
        current_entry_id: Artifact ID of the current/new data.
        reference_entry_id: Artifact ID of the reference data.
            Either this or reference_query is required.
        reference_query: Search query to find a reference dataset by name.
            Searches entries with category="graph_reference".
        metrics: Which metrics to compute. Options: rmse, correlation,
            dtw_distance, peak_shift. Default: all available.

    Returns:
        JSON with comparison metrics and interpretation hints.
    """
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()

    # Load current data
    current_entry = store.get_entry(current_entry_id)
    if current_entry is None:
        return json.dumps(
            make_error(
                "not_found",
                f"No artifact entry found with ID {current_entry_id}.",
                ["Use data_list to see available entries."],
            )
        )

    # Load reference data
    ref_entry = None
    if reference_entry_id is not None:
        ref_entry = store.get_entry(reference_entry_id)
        if ref_entry is None:
            return json.dumps(
                make_error(
                    "not_found",
                    f"No reference entry found with ID {reference_entry_id}.",
                    ["Use data_list to see available entries."],
                )
            )
    elif reference_query:
        matches = store.list_entries(
            search=reference_query,
            category_filter="graph_reference",
        )
        if not matches:
            return json.dumps(
                make_error(
                    "not_found",
                    f"No graph reference found matching '{reference_query}'.",
                    [
                        "Use data_list(category_filter='graph_reference') to see saved references.",
                        "Save a reference first with graph_save_reference.",
                    ],
                )
            )
        ref_entry = matches[-1]  # Most recent match
    else:
        return json.dumps(
            make_error(
                "validation_error",
                "Either reference_entry_id or reference_query is required.",
                [
                    "Provide reference_entry_id for a specific entry.",
                    "Provide reference_query to search by name.",
                ],
            )
        )

    # Load data from files
    try:
        current_data = _load_entry_data(store, current_entry.id)
        ref_data = _load_entry_data(store, ref_entry.id)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        return json.dumps(
            make_error(
                "data_error",
                f"Failed to load data from entries: {exc}",
                ["Ensure the data files exist and contain valid tabular data."],
            )
        )

    # Run comparison
    from osprey.mcp_server.workspace.tools.graph_comparison import compare_datasets

    try:
        results = compare_datasets(current_data, ref_data, metrics=metrics)
    except Exception as exc:
        logger.exception("graph_compare: comparison failed")
        return json.dumps(
            make_error(
                "comparison_error",
                f"Comparison computation failed: {exc}",
                ["Check that both datasets contain numeric data."],
            )
        )

    # Save comparison result to ArtifactStore (unified)
    comparison_data = {
        "current_entry_id": current_entry.id,
        "reference_entry_id": ref_entry.id,
        "current_description": current_entry.description,
        "reference_description": ref_entry.description,
        **results,
    }

    entry = store.save_data(
        tool="graph_compare",
        data=comparison_data,
        title=f"Comparison: {current_entry.description} vs {ref_entry.description}",
        description=f"Comparison: {current_entry.description} vs {ref_entry.description}",
        summary={
            "current_id": current_entry.id,
            "reference_id": ref_entry.id,
            **results["metrics"],
        },
        access_details={
            "format": "comparison_result",
            "metrics_computed": list(results["metrics"].keys()),
        },
        category="graph_comparison",
    )

    # Return inline results + context entry
    response = entry.to_tool_response()
    response["comparison"] = comparison_data
    return json.dumps(response, default=str)


# ---------------------------------------------------------------------------
# Tool 3: graph_save_reference
# ---------------------------------------------------------------------------
@mcp.tool()
async def graph_save_reference(
    source_entry_id: str,
    title: str = "",
    description: str = "",
) -> str:
    """Save a dataset as a named reference for future graph comparisons.

    Copies data from an existing artifact entry (e.g., from graph_extract
    or archiver_read) into a new reference entry that can be found by
    graph_compare using reference_query.

    Args:
        source_entry_id: Artifact ID to copy as a reference.
        title: Descriptive title for the reference (e.g., "Nominal beam current profile").
        description: Detailed description of what this reference represents.

    Returns:
        JSON with the new reference entry ID and details.
    """
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()

    # Load source entry
    source_entry = store.get_entry(source_entry_id)
    if source_entry is None:
        return json.dumps(
            make_error(
                "not_found",
                f"No artifact entry found with ID {source_entry_id}.",
                ["Use data_list to see available entries."],
            )
        )

    # Load source data
    try:
        source_data = _load_entry_data(store, source_entry_id)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        return json.dumps(
            make_error(
                "data_error",
                f"Failed to load data from entry {source_entry_id}: {exc}",
                ["Ensure the data file exists and contains valid tabular data."],
            )
        )

    # Load full source file for payload preservation
    source_path = store.get_file_path(source_entry_id)
    if source_path:
        with open(source_path) as f:
            source_payload = json.load(f)
    else:
        source_payload = {"data": source_data}

    ref_title = title or f"Reference: {source_entry.description}"
    ref_desc = description or f"Saved from entry {source_entry_id}"

    entry = store.save_data(
        tool="graph_save_reference",
        data={
            **source_payload,
            "reference_title": ref_title,
            "reference_description": ref_desc,
            "source_entry_id": source_entry_id,
            "source_tool": source_entry.tool_source,
        },
        title=ref_title,
        description=ref_desc,
        summary={
            "title": ref_title,
            "source_entry_id": source_entry_id,
            "source_tool": source_entry.tool_source,
            "num_points": len(source_data),
        },
        access_details={
            "format": "tabular",
            "num_points": len(source_data),
            "hint": "Use graph_compare with reference_query to find and compare against this reference.",
        },
        category="graph_reference",
    )

    return json.dumps(entry.to_tool_response(), default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_entry_data(store, entry_id: str) -> list[list]:
    """Load tabular data from an artifact entry's data file.

    Handles different data formats stored by graph_extract, archiver_read, etc.
    Data files are raw JSON (no envelope).

    Returns:
        List of [x, y] data points.

    Raises:
        FileNotFoundError: If the data file doesn't exist.
        KeyError: If the data file doesn't contain expected structure.
    """
    file_path = store.get_file_path(entry_id)
    if file_path is None:
        raise FileNotFoundError(f"Data file not found for entry {entry_id}")

    with open(file_path) as f:
        payload = json.load(f)

    # Handle legacy DataContext envelope format (migration compat)
    if isinstance(payload, dict) and "_osprey_metadata" in payload:
        payload = payload.get("data", {})

    # graph_extract format: {"columns": [...], "data": [[x,y], ...]}
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]

    # archiver_read format: {"values": [...]} or similar
    if isinstance(payload, dict) and "values" in payload:
        values = payload["values"]
        if isinstance(values, list) and values:
            if isinstance(values[0], (int, float)):
                return [[i, v] for i, v in enumerate(values)]
            return values

    # Already a list of data points
    if isinstance(payload, list):
        return payload

    raise KeyError(f"Could not extract tabular data from entry {entry_id}")
