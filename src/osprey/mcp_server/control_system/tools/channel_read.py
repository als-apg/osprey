"""MCP tool: channel_read — read one or more control-system channels."""

import json
import logging

from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.channel_read")

#: Per-value inline element budget when config.yml says nothing. 2000 is this
#: repo's calibrated inline budget for array data - the same number the Bluesky
#: bridge figure builder uses as DEFAULT_MAX_POINTS - so ordinary waveforms and
#: orbit vectors stay inline while camera frames take the artifact path.
DEFAULT_READ_INLINE_MAX_ELEMENTS = 2000

#: One channel_read call inlines at most this multiple of the per-value budget
#: across all requested channels, consumed in request order.
AGGREGATE_BUDGET_FACTOR = 4

#: Per-channel rolling window of unpinned read artifacts when config.yml says
#: nothing. 0 means keep everything.
DEFAULT_CHANNEL_READ_ARTIFACT_RETENTION = 20


def _config_int(path: str, default: int) -> int:
    """Read a non-negative integer config value, falling back to *default*.

    Config is not always loaded (unit tests, a bare MCP process without a
    project config.yml), and a mistyped value must not take down a read that the
    hardware answered. Anything unusable - missing, unloadable, non-numeric or
    negative - degrades to the documented default.
    """
    from osprey.utils.config import get_config_value

    try:
        raw = get_config_value(path, default)
    except Exception:
        return default

    if isinstance(raw, bool):
        # bool is an int subclass: YAML `false` would silently coerce to 0,
        # flipping retention to keep-everything instead of failing loudly.
        logger.warning("Ignoring boolean %s=%r; using %d", path, raw, default)
        return default

    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-integer %s=%r; using %d", path, raw, default)
        return default

    if value < 0:
        logger.warning("Ignoring negative %s=%r; using %d", path, raw, default)
        return default
    return value


def get_read_inline_max_elements() -> int:
    """Per-value element count above which a read value takes the artifact path.

    ``control_system.read_inline_max_elements`` in config.yml; defaults to
    :data:`DEFAULT_READ_INLINE_MAX_ELEMENTS`.
    """
    return _config_int("control_system.read_inline_max_elements", DEFAULT_READ_INLINE_MAX_ELEMENTS)


def get_read_aggregate_max_elements() -> int:
    """Total inline element budget for a single ``channel_read`` call.

    ``AGGREGATE_BUDGET_FACTOR`` times the per-value threshold, so a batch of
    individually-small arrays cannot add up to a context-flooding response.
    """
    return AGGREGATE_BUDGET_FACTOR * get_read_inline_max_elements()


def get_channel_read_artifact_retention() -> int:
    """How many unpinned read artifacts to keep per channel (0 = keep all).

    ``control_system.channel_read_artifact_retention`` in config.yml; defaults to
    :data:`DEFAULT_CHANNEL_READ_ARTIFACT_RETENTION`.
    """
    return _config_int(
        "control_system.channel_read_artifact_retention",
        DEFAULT_CHANNEL_READ_ARTIFACT_RETENTION,
    )


#: Why a value took the artifact path instead of being inlined.
ARTIFACT_REASON_PER_VALUE = "per_value_threshold"
ARTIFACT_REASON_AGGREGATE = "aggregate_budget"

#: What each artifact_reason means, reported alongside any withheld value so the
#: agent can tell "genuinely oversized" from "squeezed out by this batch".
ARTIFACT_REASON_MEANINGS = {
    ARTIFACT_REASON_PER_VALUE: (
        "This value alone exceeds inline_max_elements; it is too large to inline in any call."
    ),
    ARTIFACT_REASON_AGGREGATE: (
        "This value is under inline_max_elements on its own, but the call's total inline "
        "budget was already spent by earlier channels in the request. Re-read this channel "
        "in a smaller batch to get the value inline."
    ),
}


def _numpy():
    """Return the numpy module, or None when it is not importable.

    The size rule only ever applies to ``np.ndarray`` values, so a process
    without numpy simply inlines everything - exactly the old behaviour.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        return None
    return np


def _element_count(value, np) -> int | None:
    """Element count for the size rule, or None when the rule does not apply.

    Only ``np.ndarray`` values are measured. ``str``/``bytes`` and scalars are
    always inline no matter how long they are - a 10k-character string is one
    channel value, not 10k elements, and summarising it would lose it.
    """
    if np is None or not isinstance(value, np.ndarray):
        return None
    return int(value.size)


def _finite(number) -> float | None:
    """Coerce a numpy statistic to a JSON-safe float, or None when it is not finite."""
    import math

    try:
        result = float(number)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _array_summary(value, np, artifact_reason: str) -> dict:
    """Describe an array that is being withheld from the inline payload.

    Reports shape, dtype and element count always; ``min``/``max``/``mean`` only
    for numeric dtypes (``np.issubdtype(dtype, np.number)``) - a string or
    object array has no meaningful extremes, and inventing them would break the
    honest-reporting contract these summaries are read under.
    """
    summary: dict = {
        "value_withheld": True,
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "element_count": int(value.size),
        "artifact_reason": artifact_reason,
    }
    if value.size and np.issubdtype(value.dtype, np.number):
        stats = {
            "min": _finite(value.min()),
            "max": _finite(value.max()),
            "mean": _finite(value.mean()),
        }
        if any(stat is not None for stat in stats.values()):
            summary.update(stats)
    return summary


def _series_values(value, np) -> list:
    """JSON-safe Python list for a 1-D array, with non-finite floats as ``None``.

    ``NaN``/``inf`` are legal in a float waveform but not in JSON: the chart
    endpoint serialises with ``allow_nan=False`` and would fail on them. ``None``
    is the gap marker the timeseries helpers already understand
    (:func:`~osprey.utils.timeseries.is_numeric_channel` counts it as numeric and
    LTTB skips over it), so a gap renders as a gap instead of breaking the chart.
    """
    values: list = value.tolist()
    if np is not None and np.issubdtype(value.dtype, np.floating):
        import math

        return [None if not math.isfinite(item) else item for item in values]
    return values


def _series_payload(address: str, values: list) -> dict:
    """Channel-series JSON payload for a 1-D reading, in the archiver artifact shape.

    :func:`~osprey.utils.timeseries.extract_channel_series` passes ``series``
    through unchanged, so this is exactly what the gallery's chart endpoint
    serves. The x axis is the synthesized sample index (``0..N-1``), not
    wall-clock time - a single read has one timestamp for the whole array, and
    the sample ordinal is the only real axis the data has.
    """
    return {
        "query": {
            "channels": [address],
            "x_axis": "sample_index",
            "sample_count": len(values),
        },
        "series": {address: {"timestamps": list(range(len(values))), "values": values}},
    }


def _save_series_reading(store, address: str, value, summary: dict, np) -> dict:
    """Persist a 1-D reading as a timeseries artifact and return its handle fields.

    ``metadata.data_type = "timeseries"`` is what the gallery keys on to pick the
    interactive chart viewport, and ``metadata.channel`` is the key the retention
    sweep selects this channel's readings by; ``category`` must be
    ``channel_values`` for the same reason.
    """
    values = _series_values(value, np)
    store_summary = _reading_store_summary(address, summary)
    store_summary["sample_count"] = len(values)

    entry = store.save_data(
        tool="channel_read",
        data=_series_payload(address, values),
        title=f"{address} ({len(values)} samples)",
        description=f"Oversized 1-D reading of {address}: {len(values)} samples",
        summary=store_summary,
        access_details={
            "data_file_structure": {
                "root_keys": ["query", "series"],
                "series_format": (
                    "mapping of channel name to {'timestamps': [...], 'values': [...]}"
                ),
            },
            "access_pattern": (
                f'json_data["series"]["{address}"]["values"]  ->  the array, in element order'
            ),
            "schema": {
                "timestamps": (
                    "the sample index (0..N-1), NOT wall-clock time - one read of an "
                    "array carries a single timestamp, reported on the reading entry."
                ),
                "values": (
                    "one entry per array element, in element order. A float element "
                    "that was NaN or infinite is stored as JSON null."
                ),
            },
            "view_hint": "Open this artifact in the gallery to see the interactive chart.",
        },
        category="channel_values",
        metadata={"data_type": "timeseries", "channel": address},
    )

    return {
        "artifact_id": entry.id,
        "data_file": entry.data_file,
        "artifact_note": (
            "Full values are in the artifact, not here: open data_file with json.load() in "
            "execute, or view the interactive chart in the artifact gallery."
        ),
    }


#: NTNDArray ``colorMode`` values that describe a three-component colour frame,
#: mapped to the array axis the components live on (RGB1 = interleaved by pixel,
#: RGB2 = by row, RGB3 = by plane). The connector reports the mode as
#: ``raw_metadata["color_mode"]`` when the control system sends one. Any other
#: mode - Mono, the Bayer variants, YUV - is not a layout this renderer can draw
#: without inventing colours, so those frames take the data-only path.
_RGB_COLOR_AXIS = {"RGB1": 0, "RGB2": 1, "RGB3": 2}

#: areaDetector reports ``colorMode`` as an enumerated integer on the wire, and
#: the connector passes that value through untranslated. These are the codes the
#: standard defines; anything outside the table is left to shape inference.
_COLOR_MODE_CODES = {0: "Mono", 1: "Bayer", 2: "RGB1", 3: "RGB2", 4: "RGB3"}


def _color_mode(cv) -> str | None:
    """The ``colorMode`` the connector reported for this reading, if any.

    Two forms are trusted, both carried in a real ``dict`` of raw metadata: a
    non-empty string naming the mode, and a plain ``int`` holding the
    areaDetector enum code (0 Mono, 1 Bayer, 2 RGB1, 3 RGB2, 4 RGB3), which is
    translated to the matching name. Booleans are not codes, and an unknown code
    reads as nothing reported. Anything else - a placeholder, a mock, an empty
    string - leaves the layout to be inferred from the shape, which is strictly
    better than acting on a value the control system never sent.
    """
    raw = getattr(getattr(cv, "metadata", None), "raw_metadata", None)
    if not isinstance(raw, dict):
        return None
    mode = raw.get("color_mode")
    if type(mode) is int:
        return _COLOR_MODE_CODES.get(mode)
    return mode if isinstance(mode, str) and mode.strip() else None


def _rgb_axis(shape: tuple, color_mode: str | None) -> int | None:
    """Which axis of a 3-D frame holds the colour components, or None.

    A reported ``colorMode`` wins outright - the control system knows its own
    layout, and a frame it calls Mono or Bayer is not RGB however its dimensions
    happen to line up. With no mode reported, the axis is inferred from which
    dimension is 3, last first: ``(H, W, 3)`` is the common wire layout, and a
    tie is resolved in that order rather than guessed at.
    """
    if color_mode is not None:
        axis = _RGB_COLOR_AXIS.get(color_mode.strip().upper())
        if axis is None or shape[axis] != 3:
            return None
        return axis
    for axis in (2, 0, 1):
        if shape[axis] == 3:
            return axis
    return None


def _to_uint8(value, np):
    """Auto-scale numeric data onto the 0-255 display range.

    The scaling is global across the whole array, so an RGB frame keeps its
    colour balance instead of having each channel stretched separately. A frame
    with no spread - every pixel identical, or every pixel NaN - renders flat
    black rather than dividing by a zero range; there is no contrast to show and
    the summary already reports the constant value.
    """
    data = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)

    low = float(data[finite].min())
    high = float(data[finite].max())
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)

    # Non-finite pixels are pinned to the low end: they have no display value,
    # and leaving them as NaN would poison the cast to uint8.
    scaled = (np.where(finite, data, low) - low) * (255.0 / (high - low))
    return scaled.round().clip(0, 255).astype(np.uint8)


def _render_png(value, np, color_mode: str | None) -> bytes | None:
    """Render a readable PNG of a 2-D or 3-D frame, or None when the layout is not one.

    2-D numeric data becomes an auto-scaled grayscale image; a 3-D frame with a
    3-sized colour axis becomes RGB. Everything else - 4-D and up, a 3-D stack
    with no colour axis, complex or non-numeric data - returns None, and the
    caller stores the raw array without a preview instead of drawing something
    the data does not support.

    The image is written at the array's native pixel resolution, with no axes,
    frame or resampling, so what the viewer sees is the frame itself.
    """
    dtype = value.dtype
    if dtype.hasobject or np.issubdtype(dtype, np.complexfloating):
        return None
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        return None
    if 0 in value.shape:
        return None

    if value.ndim == 2:
        pixels = _to_uint8(value, np)
    elif value.ndim == 3:
        axis = _rgb_axis(value.shape, color_mode)
        if axis is None:
            return None
        pixels = np.moveaxis(_to_uint8(value, np), axis, 2)
    else:
        return None

    import io

    # matplotlib is a core dependency and already the repo's server-side image
    # writer; ``imsave`` writes the pixel grid straight out at native size.
    import matplotlib.image as mpimage

    buffer = io.BytesIO()
    if pixels.ndim == 2:
        mpimage.imsave(buffer, pixels, cmap="gray", vmin=0, vmax=255, format="png")
    else:
        mpimage.imsave(buffer, pixels, format="png")
    return buffer.getvalue()


def _npy_bytes(value, np) -> bytes:
    """Serialize an array to ``.npy`` bytes, never enabling pickle."""
    import io

    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _shape_text(value) -> str:
    """Human-readable shape, e.g. ``48x64``."""
    return "x".join(str(int(dim)) for dim in value.shape)


def _reading_store_summary(address: str, summary: dict) -> dict:
    """The artifact's own summary: the inline one, keyed to its channel.

    ``value_withheld`` is dropped - it describes the tool response, where the
    numbers are missing, not the artifact, which is where they are.
    """
    store_summary = {key: item for key, item in summary.items() if key != "value_withheld"}
    store_summary["channel"] = address
    return store_summary


def _save_image_reading(store, address: str, value, summary: dict, np, png_bytes: bytes) -> dict:
    """Persist a 2-D+ reading as a rendered PNG plus the raw array beside it.

    The PNG is the entry's primary file, which is what the gallery displays; the
    ``.npy`` is ``data_file``, which is what analysis code loads. The PNG is
    rendered by the caller before this runs, so nothing inside the store's
    locked section can fail halfway through a render.
    """
    entry = store.save_channel_reading(
        png_bytes=png_bytes,
        npy_bytes=_npy_bytes(value, np),
        title=f"{address} ({_shape_text(value)} {value.dtype})",
        description=(
            f"Oversized {value.ndim}-D reading of {address}: {_shape_text(value)} {value.dtype}"
        ),
        summary=_reading_store_summary(address, summary),
        access_details={
            "access_pattern": (
                "numpy.load(data_file)  ->  the raw array, exactly as it was read "
                "(same shape and dtype)"
            ),
            "schema": {
                "data_file": (
                    "the raw array as a .npy file - the values themselves, not a rendering."
                ),
                "image": (
                    "the PNG beside it is an auto-scaled preview for looking at. Pixel "
                    "brightness is relative to this frame's own min and max, so it carries "
                    "no absolute units - read data_file for values."
                ),
            },
            "view_hint": (
                "Use the Read tool on the PNG beside this entry's data_file (the same path "
                "with a .png extension) to view the image."
            ),
        },
        category="channel_values",
        metadata={
            "channel": address,
            "data_type": "ndarray",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        },
    )

    handle = {
        "artifact_id": entry.id,
        "data_file": entry.data_file,
        "artifact_note": (
            "Full values are in the artifact, not here: numpy.load(data_file) in execute "
            "returns the array with its original shape and dtype."
        ),
    }
    png_path = _agent_path_for(store, entry)
    if png_path:
        handle["view_hint"] = f"Use the Read tool on {png_path} to view the image."
    return handle


def _agent_path_for(store, entry) -> str | None:
    """Agent-facing path of the entry's primary file, or None when unresolvable.

    Reported the way the rest of the repo reports file paths - relative to the
    repo root, which is the agent's working directory. For an image reading the
    primary file is the PNG; for a data-only reading it is the ``.npy`` itself.
    """
    filepath = store.get_file_path(entry.id)
    if filepath is None:
        return None
    try:
        return str(filepath.relative_to(store.repo_root))
    except ValueError:
        return filepath.name


def _save_data_only_reading(store, address: str, value, reason: str) -> dict:
    """Persist a reading that has no preview: the raw array, no PNG.

    The layouts that land here - 4-D and up, a 3-D stack with no colour axis, a
    frame whose render failed - have no honest image, but they still have data,
    and losing it because it could not be drawn would be the worse failure. The
    array is stored as a plain ``.npy`` artifact carrying the same
    ``channel_values`` category and ``metadata.channel`` key the image path
    uses, so retention sweeps it on the same schedule.
    """
    entry = store.save_object(
        value,
        title=f"{address} ({_shape_text(value)} {value.dtype})",
        description=(
            f"Oversized {value.ndim}-D reading of {address}: "
            f"{_shape_text(value)} {value.dtype} (no preview: {reason})"
        ),
        tool_source="channel_read",
        category="channel_values",
        metadata={
            "channel": address,
            "data_type": "ndarray",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "render_skipped": reason,
        },
    )

    handle: dict = {"artifact_id": entry.id, "no_preview_reason": reason}
    data_file = _agent_path_for(store, entry)
    if data_file:
        handle["data_file"] = data_file
    handle["artifact_note"] = (
        f"No image preview was rendered ({reason}), so this entry is data only. The values "
        "are all there: numpy.load(data_file) in execute returns the array with its "
        "original shape and dtype."
    )
    return handle


def _unstored_reading_note() -> dict:
    """Handle fields for a withheld value that could not be stored at all.

    The one shape with nowhere to go: an object-dtype array. ``numpy.save``
    cannot write it without pickle, and the store never enables pickle, so there
    is no file to hand back and the summary is genuinely all there is.
    """
    return {
        "artifact_status": "not_stored",
        "artifact_note": (
            "This value's dtype cannot be stored without pickle, which is never enabled, so "
            "it cannot be retrieved - the summary above is all there is. Read fewer "
            "channels, or a channel with fewer elements, to get values inline."
        ),
    }


def _prune_channel_history(store, address: str) -> None:
    """Trim this channel's rolling window of unpinned read artifacts.

    Runs only after a save has succeeded. A failing sweep is logged and
    swallowed on purpose: the artifact it would have tidied up around is already
    on disk and already handed to the agent, and letting the error escape would
    strip that working handle off the entry over a housekeeping problem.
    """
    try:
        store.prune_channel_readings(address, get_channel_read_artifact_retention())
    except Exception as exc:  # noqa: BLE001 - the reading itself is already saved
        logger.warning("Retention sweep failed for %s: %s", address, exc)


def _persist_oversized_reading(
    address: str,
    value,
    summary: dict,
    *,
    artifact_reason: str,
    color_mode: str | None = None,
) -> dict:
    """Persist an oversized read value and return the handle fields for its entry.

    1-D values are saved as a channel-series JSON artifact that the gallery
    renders as an interactive chart. 2-D and 3-D frames are saved as a rendered
    PNG plus the raw array beside it. A layout with no honest rendering is still
    saved, just without the preview - only a dtype the store cannot write at all
    ends up with no artifact.

    Args:
        address: Channel address the value was read from. Stored as
            ``metadata.channel``, which is also the retention key.
        value: The numpy array being withheld from the inline payload.
        summary: The inline summary already computed for the value (shape, dtype,
            element count, numeric stats, artifact_reason) - the same dict the
            returned fields are merged into, and the basis of the artifact's own
            summary.
        artifact_reason: One of :data:`ARTIFACT_REASON_PER_VALUE` /
            :data:`ARTIFACT_REASON_AGGREGATE`.
        color_mode: ``colorMode`` the control system reported for the frame, when
            it reported one. Decides how a 3-D frame is laid out; ignored
            otherwise.

    Returns:
        Fields to merge into the reading entry: the artifact handle
        (``artifact_id``, ``data_file``, a short access note) when the value was
        stored. Raising is allowed: the caller degrades the entry to
        ``artifact_error`` and still reports the read as successful, because the
        machine did answer.
    """
    from osprey.stores.artifact_store import get_artifact_store

    np = _numpy()
    store = get_artifact_store()

    if value.ndim == 1:
        handle = _save_series_reading(store, address, value, summary, np)
    else:
        # Rendering happens here, outside the store's locked section, so a render
        # that fails costs a preview and not a half-written artifact.
        png_bytes: bytes | None = None
        reason = "this array layout has no image rendering"
        try:
            png_bytes = _render_png(value, np, color_mode)
        except Exception as exc:  # noqa: BLE001 - a preview is not worth the read
            logger.warning("Could not render a preview for %s: %s", address, exc)
            reason = f"rendering failed ({type(exc).__name__})"

        if png_bytes is not None:
            handle = _save_image_reading(store, address, value, summary, np, png_bytes)
        elif not value.dtype.hasobject:
            handle = _save_data_only_reading(store, address, value, reason)
        else:
            # Nothing to write: no save happened, so no retention sweep.
            return _unstored_reading_note()

    _prune_channel_history(store, address)
    return handle


async def _gather_readings(connector, channels: list[str]) -> tuple[dict, dict]:
    """Read every address individually, keeping successes and failures apart.

    The tool issues the individual ``read_channel`` calls itself - functionally
    what the EPICS batch method does internally - so that it holds every
    exception first-hand and can report its text. ``read_multiple_channels``
    drops failures silently by contract, which would let a bad address vanish
    from the response without a trace.

    Returns:
        ``(readings, failures)`` - address to ``ChannelValue`` for the reads that
        answered, and address to error text for the ones that did not. Both are
        in request order.
    """
    import asyncio

    results = await asyncio.gather(
        *(connector.read_channel(address) for address in channels),
        return_exceptions=True,
    )

    readings: dict = {}
    failures: dict = {}
    first_error: BaseException | None = None
    for address, result in zip(channels, results, strict=True):
        if isinstance(result, BaseException):
            failures[address] = f"{type(result).__name__}: {result}"
            if first_error is None:
                first_error = result
        else:
            readings[address] = result

    if failures:
        logger.warning("channel_read could not read %d channel(s): %s", len(failures), failures)

    # Nothing answered: that is a failed read, not a successful read of nothing.
    # Re-raising preserves the typed error envelope (connection, timeout, ...).
    if not readings and first_error is not None:
        raise first_error

    return readings, failures


@mcp.tool()
async def channel_read(
    channels: list[str],
    include_metadata: bool = True,
) -> str:
    """Read current values from one or more control-system channels.

    Array values come back inline as JSON lists. A value with more elements than
    the configured inline budget - or one that no longer fits the budget left for
    this call - is reported as a summary (shape, dtype, element count and, for
    numeric data, min/max/mean) instead of raw numbers.

    Args:
        channels: List of channel/PV addresses to read.
        include_metadata: If True, also report units, precision, alarm status and
            description for each channel, plus enum_label/enum_labels on the entries
            whose channel is enum-typed (an enum reads as its integer index, and the
            label names the state that index stands for). Use channel_limits for the
            bounds a write is checked against - the control system does not report
            those here.

    Returns:
        JSON with a summary of channel values, one entry per channel. Channels
        that could not be read are listed under channels_failed with their error.
    """
    if not channels:
        return make_error(
            "validation_error",
            "No channels provided.",
            ["Provide at least one channel address."],
        )

    async with connector_error_handler("channel_read"):
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        connector = await registry.control_system()

        requested = list(dict.fromkeys(channels))
        readings, failures = await _gather_readings(connector, requested)

        # Only fields the connectors actually populate. No connector reports the
        # channel's display range, so this tool does not claim to either — and a
        # write bound is a limits-database question, not a control-system read.
        metadata_fields = ("units", "precision", "alarm_status", "description")
        # Enum-only, and reported only when the control system had them: an
        # ordinary analogue channel would otherwise grow two permanently-null
        # keys in every reading, which reads as "this channel has no labels"
        # rather than "labels do not apply here".
        enum_metadata_fields = ("enum_label", "enum_labels")

        np = _numpy()
        inline_max = get_read_inline_max_elements()
        aggregate_max = get_read_aggregate_max_elements()
        aggregate_remaining = aggregate_max

        readings_summary: dict = {}
        withheld: list[str] = []
        for addr, cv in readings.items():
            entry: dict = {"timestamp": str(cv.timestamp)}
            value = cv.value
            count = _element_count(value, np)
            artifact_reason: str | None = None

            if count is None:
                # str/bytes/scalars are always inline, whatever their length.
                entry["value"] = value
            elif count > inline_max:
                artifact_reason = ARTIFACT_REASON_PER_VALUE
            elif count > aggregate_remaining:
                artifact_reason = ARTIFACT_REASON_AGGREGATE
            else:
                # The budget is spent in request order, so which channels inline
                # is deterministic for a given call.
                aggregate_remaining -= count
                entry["value"] = value.tolist()

            if artifact_reason is not None:
                summary_fields = _array_summary(value, np, artifact_reason)
                try:
                    summary_fields.update(
                        _persist_oversized_reading(
                            addr,
                            value,
                            summary_fields,
                            artifact_reason=artifact_reason,
                            color_mode=_color_mode(cv),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - the machine did answer
                    logger.warning("Could not store oversized reading for %s: %s", addr, exc)
                    summary_fields["artifact_error"] = f"{type(exc).__name__}: {exc}"
                entry.update(summary_fields)
                withheld.append(addr)

            if include_metadata:
                for field in metadata_fields:
                    entry[field] = getattr(cv.metadata, field, None)
                for field in enum_metadata_fields:
                    enum_value = getattr(cv.metadata, field, None)
                    if enum_value is not None:
                        entry[field] = enum_value
            readings_summary[addr] = entry

        summary: dict = {
            "channels_read": len(readings_summary),
            "readings": readings_summary,
        }
        description = f"Read {len(readings_summary)} channel(s)"
        if failures:
            summary["channels_failed"] = failures
            description += f", {len(failures)} failed"

        access_details: dict = {
            "fields_per_entry": (
                ["value", "timestamp"] + (list(metadata_fields) if include_metadata else [])
            ),
        }
        if include_metadata:
            access_details["enum_fields_per_entry"] = {
                "fields": list(enum_metadata_fields),
                "meaning": (
                    "Present only on enum-typed channels (EPICS mbbi/bi/bo and "
                    "equivalents). value is the state's integer index; enum_label is "
                    "that state's name and enum_labels every state in index order. An "
                    "entry without these keys is not an enum channel."
                ),
            }
        if withheld:
            access_details["values_withheld"] = {
                "channels": withheld,
                "inline_max_elements": inline_max,
                "aggregate_max_elements": aggregate_max,
                "artifact_reason": ARTIFACT_REASON_MEANINGS,
                "reporting": (
                    "Report only what the summary states - shape, dtype, element count and "
                    "the listed statistics. Never state an individual element value for a "
                    "withheld channel."
                ),
            }

        # Return ephemeral result (no persistent storage for channel reads)
        return json.dumps(
            {
                "status": "success",
                "description": description,
                "summary": summary,
                "access_details": access_details,
            },
            default=str,
        )
