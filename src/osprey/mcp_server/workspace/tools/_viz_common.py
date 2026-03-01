"""Shared utilities for visualization tools (static plot, interactive plot, dashboard).

Provides data-loading code generation, artifact collection, and save
patterns used by all three visualization tools.
"""

import logging
import re

logger = logging.getLogger("osprey.mcp_server.workspace.tools._viz_common")

# Hex ID pattern for artifact IDs (12-char hex)
ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def resolve_data_source(data_source: str) -> str:
    """Resolve a data_source value to an absolute file path.

    Handles two forms:
    - **Artifact ID** (12-char hex): resolved via ArtifactStore
    - **File path**: returned as-is

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    if ARTIFACT_ID_RE.match(data_source):
        from osprey.mcp_server.artifact_store import get_artifact_store

        store = get_artifact_store()
        path = store.get_file_path(data_source)
        if path is None:
            raise FileNotFoundError(f"Artifact {data_source!r} not found")
        return str(path)

    # Assume file path
    return data_source


def build_data_loading_code(data_source: str) -> str:
    """Generate preamble code to load data from an artifact ID, context entry ID, or file path.

    Resolution is done server-side (not in the sandbox) so that the generated
    code always contains an absolute file path.
    """
    resolved_path = resolve_data_source(data_source)
    return f"""\
import os
_data_path = {resolved_path!r}
if not os.path.exists(_data_path):
    raise FileNotFoundError(f"Data file not found: {{_data_path}}")
"""


def build_data_reader(data_source: str) -> str:
    """Generate code to read data and auto-convert to a pandas DataFrame.

    The generated code loads data from the resolved file path and performs
    automatic format detection and unwrapping:

    - CSV/Excel/Parquet → ``data`` is a pandas DataFrame
    - JSON with legacy OSPREY metadata envelope → unwrapped, then converted to DataFrame
    - JSON with archiver nested format → unwrapped, then converted to DataFrame

    After this code runs, **``data`` is always a pandas DataFrame** (for
    tabular sources) or a raw string (for unrecognized formats).
    """
    loading = build_data_loading_code(data_source)
    return (
        loading
        + """\
if _data_path.endswith('.csv'):
    data = pd.read_csv(_data_path)
elif _data_path.endswith('.json'):
    import json as _json
    with open(_data_path) as _f:
        data = _json.load(_f)
    # Unwrap legacy OSPREY metadata envelope (if present)
    if isinstance(data, dict) and '_osprey_metadata' in data and 'data' in data:
        data = data['data']
    # Handle archiver nested format: {query: ..., dataframe: {columns, index, data}}
    if isinstance(data, dict) and 'dataframe' in data:
        data = data['dataframe']
    # Handle split-orient format: {columns, index, data}
    if isinstance(data, dict) and 'columns' in data and 'index' in data and 'data' in data:
        data = pd.DataFrame(data['data'], columns=data['columns'], index=data['index'])
    elif isinstance(data, dict):
        data = pd.DataFrame(data)
    elif isinstance(data, list):
        data = pd.DataFrame(data)
elif _data_path.endswith(('.xls', '.xlsx')):
    data = pd.read_excel(_data_path)
elif _data_path.endswith('.parquet'):
    data = pd.read_parquet(_data_path)
else:
    # Try CSV as default
    try:
        data = pd.read_csv(_data_path)
    except Exception:
        with open(_data_path) as _f:
            data = _f.read()
if hasattr(data, 'shape'):
    print(f"data_source loaded: {type(data).__name__} with shape {data.shape}, columns: {list(data.columns)}")
"""
    )


def collect_and_register_artifacts(
    exec_result,
    title: str,
    description: str,
    tool_source: str,
    category: str = "",
    code: str = "",
    stdout: str = "",
    data_source: str | None = None,
) -> list[str]:
    """Save exec_result.artifacts to the artifact store, return artifact IDs.

    When ``category`` is provided, each artifact is tagged with it and
    visualization metadata (code, stdout, data_source) is embedded in the
    artifact's ``metadata`` dict — no separate JSON blob is created.
    """
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    artifact_ids: list[str] = []

    for art in exec_result.artifacts:
        try:
            viz_metadata: dict = {}
            if code:
                viz_metadata["code"] = code
            if stdout:
                viz_metadata["stdout"] = stdout
            if data_source:
                viz_metadata["data_source"] = data_source

            art_entry = store.save_file(
                file_content=art["path"].read_bytes(),
                filename=art["path"].name,
                artifact_type=art["artifact_type"],
                title=art["title"],
                description=art["description"],
                mime_type=art["mime_type"],
                tool_source=tool_source,
                metadata=viz_metadata or None,
            )

            if category:
                art_entry.category = category
                art_entry.source_agent = "data-visualizer"
                # TODO: Replace with a public API method once BaseStore
                # exposes one (currently only _save_index exists).
                store._save_index()

            artifact_ids.append(art_entry.id)
        except Exception:
            logger.debug("Artifact save failed", exc_info=True)

    return artifact_ids


def build_viz_response(artifact_ids: list[str], title: str, stdout: str = "") -> dict:
    """Build a tool response dict for visualization tools (no separate artifact)."""
    response: dict = {
        "status": "success",
        "title": title,
        "artifact_ids": artifact_ids,
        "artifact_id": artifact_ids[0] if artifact_ids else None,
        "artifact_count": len(artifact_ids),
    }
    if stdout:
        response["stdout"] = stdout
    if artifact_ids:
        try:
            from osprey.mcp_server.common import gallery_url

            response["gallery_url"] = gallery_url()
        except Exception:
            pass
    return response
