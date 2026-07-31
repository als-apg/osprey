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
    - JSON with legacy OSPREY metadata envelope → unwrapped, then converted to DataFrame
    - JSON with the long-format archiver ``series`` envelope (current
      ``archiver_read`` output: ``{query: ..., series: {channel: {timestamps,
      values}}}``) → pivoted to a wide DataFrame: one column per channel, a
      ``DatetimeIndex`` that is the *union* of every channel's own real
      timestamps. A channel with no sample at another channel's timestamp gets
      ``NaN`` there — nothing is forward-filled or otherwise invented, it is
      the ordinary consequence of aligning independently-timed columns for
      plotting. A non-numeric (enum/status) channel's column keeps its own
      dtype and does not raise. A channel with zero samples in range (an
      empty ``timestamps``/``values`` pair — always present for every
      *requested* channel, per ``archiver_read``'s contract) is tolerated:
      its tz-naive empty index is built with ``utc=True`` so it can still be
      concatenated against a populated channel's tz-aware index. A channel
      whose ``timestamps`` and ``values`` differ in length (a malformed
      artifact) is tolerated the same way
      :func:`osprey.utils.timeseries.lttb_downsample_channel` and
      :func:`osprey.interfaces.artifacts.app._pivot_channel_series_to_table`
      already tolerate it. This branch only fires when ``series`` is
      genuinely shaped like the archiver envelope (a dict of per-channel
      dicts) — a coincidental top-level ``series`` key of some other shape
      falls through to the generic dict/list handling below instead of
      raising.
    - JSON with the legacy split-orient archiver ``dataframe`` envelope
      (``{query: ..., dataframe: {columns, index, data}}``) → unwrapped, then
      converted to DataFrame. Old artifacts on disk in this layout still load.

    After this code runs, **``data`` is always a pandas DataFrame** (for
    tabular sources) or a raw string (for unrecognized formats).

    Note: this can't simply call
    :func:`osprey.utils.timeseries.extract_channel_series` at runtime — the
    generated code executes inside the visualization sandbox
    (``osprey.mcp_server.workspace.execution.sandbox_executor``), whose
    AST-level import whitelist does not include ``osprey`` itself, so an
    ``import osprey...`` here would fail *every* JSON ``data_source`` load
    with "Import not allowed", not just archiver ones. The ``series`` branch
    below is a self-contained pivot instead.
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
    # Handle the long-format archiver envelope: {query: ..., series: {channel:
    # {timestamps, values}}}. Pivot to a wide DataFrame (one column per channel)
    # for plotting; each channel keeps its own real samples and its own dtype
    # (non-numeric/enum channels included) -- only the alignment for a shared
    # x-axis is new here, nothing is forward-filled.
    # Guarded on shape (a dict of per-channel dicts) so a coincidental
    # top-level 'series' key of some other shape (e.g. a list, or a dict of
    # bare lists) falls through to the generic dict/list handling below
    # instead of crashing on .items()/.get().
    _series = data.get('series') if isinstance(data, dict) else None
    if isinstance(_series, dict) and all(isinstance(_v, dict) for _v in _series.values()):
        _channel_cols = {}
        for _channel, _entry in _series.items():
            _timestamps = _entry.get('timestamps', [])
            _values = _entry.get('values', [])
            # A channel whose timestamps/values differ in length (a malformed
            # artifact) is tolerated the same way the chart downsampler and
            # the artifacts-app table pivot already tolerate it.
            if len(_timestamps) != len(_values):
                _paired = list(zip(_timestamps, _values, strict=False))
                _timestamps = [_t for _t, _ in _paired]
                _values = [_v for _, _v in _paired]
            # utc=True: an empty channel (always present for every requested
            # channel per archiver_read's contract) yields a tz-naive empty
            # index from pd.to_datetime([]), while a populated channel yields
            # a tz-aware one -- concat below would raise "Cannot join
            # tz-naive with tz-aware DatetimeIndex" without this.
            # dtype float64 for the empty case: pd.Series([]) has no values to
            # infer from and lands on object dtype, and Plotly Express refuses a
            # wide frame whose columns differ in type. Without this, "plot beam
            # current alongside a channel with no data in this window" raises
            # instead of drawing the channel that does have data. An empty column
            # carries no value whose real dtype this could contradict.
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
                store.update_entry_metadata(
                    art_entry.id, category=category, source_agent="data-visualizer"
                )

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
