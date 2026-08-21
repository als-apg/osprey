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
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        path = store.get_file_path(data_source)
        if path is None:
            raise FileNotFoundError(f"Artifact {data_source!r} not found")
        return str(path)

    # Assume file path
    return data_source


def build_data_loading_code(data_source: str) -> str:
    """Generate preamble code to load data from an artifact ID or file path.

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
    - ``.npy`` (a channel_read image/waveform artifact) → ``data`` is the raw
      ``numpy.ndarray``, not a DataFrame, so 2-D frames plot directly with
      ``imshow`` and higher-rank stacks keep their shape
    - JSON with legacy OSPREY metadata envelope → unwrapped, then converted to DataFrame
    - JSON with the archiver ``series`` envelope (``{query: ..., series:
      {channel: {timestamps, values}}}``) → pivoted to a wide DataFrame: one
      column per channel, indexed by the union of all channels' timestamps
      (``NaN`` where a channel has no sample; nothing is forward-filled).
    - JSON with the legacy split-orient archiver ``dataframe`` envelope
      (``{query: ..., dataframe: {columns, index, data}}``) → unwrapped, then
      converted to DataFrame.

    After this code runs, **``data`` is always a pandas DataFrame** (for
    tabular sources), a ``numpy.ndarray`` (for ``.npy`` sources) or a raw
    string (for unrecognized formats).

    Note: the generated code runs inside the visualization sandbox, whose
    import whitelist excludes ``osprey`` itself, so the ``series`` branch is a
    self-contained pivot rather than a call to
    :func:`osprey.utils.timeseries.extract_channel_series`.
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
    # Archiver envelope: {query: ..., series: {channel: {timestamps, values}}}.
    # Pivot to a wide DataFrame, one column per channel. Guarded on every entry
    # carrying 'timestamps' so an unrelated top-level 'series' key falls
    # through to the generic handling below (matches _build_oversize_preview).
    _series = data.get('series') if isinstance(data, dict) else None
    if isinstance(_series, dict) and all(
        isinstance(_v, dict) and 'timestamps' in _v for _v in _series.values()
    ):
        _channel_cols = {}
        for _channel, _entry in _series.items():
            _timestamps = _entry.get('timestamps', [])
            _values = _entry.get('values', [])
            # Tolerate a malformed artifact with mismatched lengths
            if len(_timestamps) != len(_values):
                _shared = min(len(_timestamps), len(_values))
                _timestamps, _values = _timestamps[:_shared], _values[:_shared]
            # utc=True: pd.to_datetime([]) yields a tz-naive empty index, which
            # concat below cannot join with a populated channel's tz-aware one.
            # dtype float64 for the empty case: an object-dtype empty column
            # makes Plotly Express refuse the whole wide frame.
            _idx = pd.to_datetime(_timestamps, utc=True)
            _channel_cols[_channel] = pd.Series(
                _values, index=_idx, dtype='float64' if not _values else None
            )
        data = pd.concat(_channel_cols, axis=1, sort=True) if _channel_cols else pd.DataFrame()
    # Handle the legacy split-orient archiver envelope: {query: ..., dataframe: {columns, index, data}}
    if isinstance(data, dict) and 'dataframe' in data:
        data = data['dataframe']
    # Handle split-orient format: {columns, index, data}
    if isinstance(data, dict) and 'columns' in data and 'index' in data and 'data' in data:
        _idx = data['index']
        # Only str: pd.to_datetime on ints would silently coerce row IDs to epoch-1970 timestamps.
        if _idx and isinstance(_idx[0], str):
            try:
                _idx = pd.to_datetime(_idx)
            except (ValueError, TypeError):
                pass
        data = pd.DataFrame(data['data'], columns=data['columns'], index=_idx)
    elif isinstance(data, dict):
        data = pd.DataFrame(data)
    elif isinstance(data, list):
        data = pd.DataFrame(data)
elif _data_path.endswith(('.xls', '.xlsx')):
    data = pd.read_excel(_data_path)
elif _data_path.endswith('.parquet'):
    data = pd.read_parquet(_data_path)
elif _data_path.endswith('.npy'):
    # Channel artifact: the raw ndarray half of a channel_read image/waveform
    # save. numpy only -- 'osprey' is not importable in the viz sandbox. Left
    # as an ndarray instead of being coerced to a DataFrame so 2-D frames stay
    # usable with imshow() and higher-rank stacks survive at all. allow_pickle
    # stays off: these artifacts are plain numeric buffers, never objects.
    import numpy as _np
    data = _np.load(_data_path, allow_pickle=False)
else:
    # Try CSV as default
    try:
        data = pd.read_csv(_data_path)
    except Exception:
        with open(_data_path) as _f:
            data = _f.read()
if hasattr(data, 'shape'):
    # ndarray sources have a shape but no columns
    _detail = f"columns: {list(data.columns)}" if hasattr(data, 'columns') else f"dtype: {data.dtype}"
    print(f"data_source loaded: {type(data).__name__} with shape {data.shape}, {_detail}")
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

    Each artifact is tagged with the category its ``save_artifact()`` call
    named, or with ``category`` when it named none, and visualization metadata
    (code, stdout, data_source) is embedded in the artifact's ``metadata``
    dict — no separate JSON blob is created.
    """
    from osprey.stores.artifact_store import get_artifact_store

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

            # A category the code passed to save_artifact() wins; the tool's
            # own category is the fallback for artifacts saved without one.
            art_category = art.get("category") or category
            art_entry = store.save_file(
                file_content=art["path"].read_bytes(),
                filename=art["path"].name,
                artifact_type=art["artifact_type"],
                title=art["title"],
                description=art["description"],
                mime_type=art["mime_type"],
                tool_source=tool_source,
                metadata=viz_metadata or None,
                category=art_category,
            )

            if art_category:
                store.update_entry_metadata(art_entry.id, source_agent="data-visualizer")

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
            from osprey.mcp_server.http import gallery_url

            response["gallery_url"] = gallery_url()
        except Exception:
            pass
    return response
