"""Shared utilities for visualization tools (static plot, interactive plot, dashboard).

Provides data-loading code generation, artifact collection, and DataContext
save patterns used by all three visualization tools.
"""

import logging
import re

logger = logging.getLogger("osprey.mcp_server.workspace.tools._viz_common")

# Hex ID pattern for artifact IDs (12-char hex)
ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# Numeric string pattern for DataContext entry IDs (e.g., "2", "15")
CONTEXT_ENTRY_ID_RE = re.compile(r"^\d+$")


def resolve_data_source(data_source: str) -> str:
    """Resolve a data_source value to an absolute file path.

    Handles three forms:
    - **Artifact ID** (12-char hex): resolved via ArtifactStore
    - **Context entry ID** (numeric string): resolved via DataContext
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

    if CONTEXT_ENTRY_ID_RE.match(data_source):
        from osprey.mcp_server.data_context import get_data_context

        ctx = get_data_context()
        path = ctx.get_file_path(int(data_source))
        if path is None:
            raise FileNotFoundError(
                f"Data context entry {data_source} not found or file missing"
            )
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
# Load data from file (resolved from data_source={data_source!r})
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
    - JSON with OSPREY metadata envelope → unwrapped, then converted to DataFrame
    - JSON with archiver nested format → unwrapped, then converted to DataFrame

    After this code runs, **``data`` is always a pandas DataFrame** (for
    tabular sources) or a raw string (for unrecognized formats).
    """
    loading = build_data_loading_code(data_source)
    return loading + """\
# Auto-detect format and load into `data` (pandas DataFrame for tabular sources)
if _data_path.endswith('.csv'):
    data = pd.read_csv(_data_path)
elif _data_path.endswith('.json'):
    import json as _json
    with open(_data_path) as _f:
        data = _json.load(_f)
    # Unwrap OSPREY metadata envelope
    if isinstance(data, dict) and 'data' in data:
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
# -- data_source loading complete --
# `data` is a pandas DataFrame (columns: {list(data.columns) if hasattr(data, 'columns') else 'N/A'})
if hasattr(data, 'shape'):
    print(f"data_source loaded: {type(data).__name__} with shape {data.shape}, columns: {list(data.columns)}")
"""


def collect_and_register_artifacts(
    exec_result,
    title: str,
    description: str,
    tool_source: str,
) -> list[str]:
    """Save exec_result.artifacts to the artifact store, return artifact IDs."""
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    artifact_ids: list[str] = []

    for art in exec_result.artifacts:
        try:
            art_entry = store.save_file(
                file_content=art["path"].read_bytes(),
                filename=art["path"].name,
                artifact_type=art["artifact_type"],
                title=art["title"],
                description=art["description"],
                mime_type=art["mime_type"],
                tool_source=tool_source,
            )
            artifact_ids.append(art_entry.id)
        except Exception:
            logger.debug("Artifact save failed", exc_info=True)

    return artifact_ids


def save_to_data_context(
    tool: str,
    title: str,
    description: str,
    code: str,
    artifact_ids: list[str],
    stdout: str,
    data_source: str | None = None,
    data_type: str = "visualization",
) -> dict:
    """Save visualization result to DataContext and return tool response dict."""
    from osprey.mcp_server.data_context import get_data_context

    ctx = get_data_context()
    entry = ctx.save(
        tool=tool,
        data={
            "title": title,
            "description": description,
            "code": code,
            "data_source": data_source,
            "artifact_ids": artifact_ids,
            "stdout": stdout,
        },
        description=f"{tool}: {title}",
        summary={
            "title": title,
            "artifact_count": len(artifact_ids),
            "artifact_ids": artifact_ids,
        },
        access_details={
            "format": data_type,
            "artifact_ids": artifact_ids,
            "data_source": data_source,
        },
        data_type=data_type,
    )

    response = entry.to_tool_response()
    response["artifact_ids"] = artifact_ids
    if artifact_ids:
        try:
            from osprey.mcp_server.common import gallery_url

            response["gallery_url"] = gallery_url()
        except Exception:
            pass
    return response
