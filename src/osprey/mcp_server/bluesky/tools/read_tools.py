"""MCP tools: read/allow-listed Bluesky bridge operations.

Each tool is a thin HTTP client of one endpoint of the facility-side Bluesky
bridge. All seven are safe to call without operator approval
(``permissions_allow``) — none of them can start motion; ``queue_start`` is
the sole path by which execution begins.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
get_run                    GET  /runs/{id}
list_plans             GET  /plans
list_devices               GET  /devices
list_runs                   GET  /runs
get_run_data               GET  /runs/{id}/data
get_run_figure             GET  /runs/{id}/figure
get_plan_source            GET  /plans/{name}/source
==========================  =================================================

The HTTP primitive (``_http_get_json``) and the
``bridge_error_message`` / ``UNKNOWN_RUN_HINTS`` error-envelope helpers live in
``osprey.mcp_server.bluesky.server_context`` so tests can patch the network
boundary and every tool renders identical error shapes. A
connection-level failure there already raises the standard
``bluesky_bridge_unreachable`` error envelope, so the tools below only need to
translate non-2xx bridge responses (404/409/etc.) into ``make_error`` calls.

``get_run_figure`` is the one tool that does not relay its body verbatim: the
route serves the panel's figure, whose heatmaps are unbounded by design (a
facility-scale response matrix is a quarter of a megabyte of cells), so this
module projects it into a fixed point budget first. That projection lives in
the "Figure projection" section at the bottom of this file.
"""

import json
from collections.abc import Sequence
from urllib.parse import quote, urlencode

import anyio
from pydantic import ValidationError

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import (
    UNKNOWN_RUN_HINTS,
    _http_get_json,
    bridge_error_message,
)
from osprey.mcp_server.errors import make_error
from osprey.services.bluesky_bridge.figure import (
    DEFAULT_MAX_POINTS,
    BarsMark,
    Figure,
    HeatmapMark,
    LinesMark,
    Panel,
    Point,
    Series,
    decimate,
)


# ---------------------------------------------------------------------------
# Tool 1: get run
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_run(run_id: str) -> str:
    """Get one run's current lifecycle status.

    A run is a queued item seen from the run side: the manager's queue,
    running item and history, keyed by the OSPREY run id. Its lifecycle runs
    ``pending`` -> ``running`` -> ``completed`` | ``stopped`` | ``error``.

    Args:
        run_id: Run id returned by queue_add or list_runs.

    Returns:
        JSON run record. Four keys are always present: ``"id"`` (this run id),
        ``"status"`` (exactly one of "pending", "running", "completed",
        "stopped", "error"), ``"plan_name"``, and ``"plan_args"`` (the plan's
        parameter fields, ``{}`` when it takes none).

        Four more appear only when they are true of this run:
        ``"item_uid"`` (the queue item's handle), ``"run_uid"`` (the
        acquisition's own uid, ABSENT while pending or running because it does
        not exist until the worker starts the plan — read that as "not yet",
        never as "unknown"), ``"error"`` (present if and only if status is
        "error", explaining what the manager reported), and ``"progress"``
        (``{"rows_seen", "expected_points", "fraction", "complete"}``, ABSENT
        until the run actually starts, because the denominator is the point
        count the run declares in its own opening metadata and there is nothing
        to read before then — read an absent ``"progress"`` as "not started
        yet", never as a fabricated 0%. Once present, ``"fraction"`` is null
        whenever the run declared no point count, so report it as "N points so
        far" rather than a percentage).

        "stopped" means a human stopped it, by any route. An item that left the
        queue without a cleanly recorded finish reads as "error" by design, so
        an unrecognized ending is never mistaken for a successful run.

    Refusals:
        - unknown_run: this run id is not in the manager's queue or its
          retained history — typically rotated out of history. Note this does
          NOT mean the run's data is gone: get_run_data can still serve that
          same id from durable storage, so never infer "no data" from this.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 3: list plans
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_plans() -> str:
    """List the plans registered on the bridge.

    Each plan entry carries ``metadata`` (the plan's authoring-declared
    ``PLAN_METADATA`` — name, description and ``writes`` — or ``null`` for a
    built-in that doesn't author one) and ``provenance`` (its trust tier:
    ``shipped``/``preset``/``facility``/``session``/``unreviewed``, ascending
    ephemerality). Prefer a higher-provenance plan.

    What a plan touches is in its ``schema``, not its metadata. Every parameter
    that carries channel names declares its own role there — whether the plan
    drives that channel to a value or only records it — so the schema is where
    you read which channels a plan moves and which it reads. Match those roles
    against what was actually asked for (a request to measure something should
    not land on a plan that drives channels) before staging a plan into the
    draft (set_draft) for a future ``queue_add``.

    Returns:
        JSON ``{"status": "success", "plans": [...]}``, each entry shaped
        like ``{"name", "description", "schema", "metadata", "provenance"}``.
        An empty list means the facility has not injected a plan module (or
        this bridge version does not yet support plan discovery).
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, "/plans")
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps({"status": "success", "plans": body})


# ---------------------------------------------------------------------------
# Tool 4: list devices
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_devices(prefix: str = "", limit: int | None = None, offset: int = 0) -> str:
    """List the device names this deployment's plans accept, one page at a time.

    Plan parameters carry devices as strings, and this is the set those strings
    must come from — a name that is not here is not a device the worker has,
    and the run fails on its first step rather than at queue time. A plan's
    ``schema`` says which of its parameters carry device names, and whether the
    plan drives each one or only records it, so put a name in the parameter
    whose declared role matches how it is meant to be used. Read this before
    staging any device name into the draft; never invent or guess one.

    One page comes back per call, alongside ``total`` — how many names matched
    before the page was cut — so "that is all of them" is always
    distinguishable from "there is more". A facility-scale namespace is far
    larger than one page: when a specific device is wanted, ``prefix`` is the
    way to reach it, not paging to the end.

    Args:
        prefix: Return only names starting with this, matched literally and
            CASE-SENSITIVELY, exactly as the worker spells them. Empty (the
            default) returns every name.
        limit: Maximum entries in this page. ``None`` (the default) asks for
            the bridge's full page size. Values outside 1..page-size are
            clamped by the bridge, never rejected.
        offset: Index into the matching names to start this page at, counted
            from 0. Clamped up to at least 0, likewise never rejected.

    Returns:
        JSON ``{"status": "success", "devices": [...], "total", "offset",
        "limit"}``. Each entry is ``{"name"}`` plus whichever of
        ``is_movable``/``is_readable``/``is_flyable`` the worker reported —
        ``is_movable`` marks a device that can be driven as a setpoint,
        ``is_readable`` one that can be read as a readback. A missing flag
        means the worker did not say, not "no".

        ``total`` counts the matches before the page cut; ``offset`` and
        ``limit`` are the EFFECTIVE values the bridge used after clamping, not
        necessarily the ones asked for. A ``"note"`` appears whenever this page
        holds fewer than ``total`` entries, saying how many are missing and how
        to reach them — relay that rather than describing the page as the whole
        set. ``devices: []`` with ``total: 0`` means nothing matched: for an
        empty ``prefix`` the worker built no devices at all, and otherwise no
        name starts with it (check the spelling and the case).
    """
    params: dict[str, str | int] = {}
    if prefix:
        params["prefix"] = prefix
    if limit is not None:
        params["limit"] = limit
    if offset:
        params["offset"] = offset
    path = "/devices" + (f"?{urlencode(params)}" if params else "")
    status, body = await anyio.to_thread.run_sync(_http_get_json, path)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    # The route clamps; this tool reports what came back rather than what was
    # asked for, so `offset`/`limit` below are the bridge's effective values.
    devices = body["devices"]
    total = body["total"]
    effective_offset = body["offset"]
    result = {
        "status": "success",
        "devices": devices,
        "total": total,
        "offset": effective_offset,
        "limit": body["limit"],
    }
    if len(devices) < total:
        if effective_offset >= total:
            result["note"] = (
                f"This page is empty because offset {effective_offset} is past the end of "
                f"the namespace, which holds {total} matching devices — read again from a "
                "lower offset."
            )
        else:
            result["note"] = (
                f"{total - effective_offset - len(devices)} of the {total} matching devices "
                "are not on this page — narrow the search with prefix, or read the next page "
                "with offset."
            )
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 5: list runs
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_runs(limit: int = 20) -> str:
    """List runs the queue knows about — queued, running and recent — newest first.

    Covers what OSPREY enqueued: an item put on the queue by some other route,
    without an OSPREY run id, is absent from this list entirely. queue_list is
    the complete view of what the manager actually holds, so reach for that
    when the question is "what is this machine about to do" rather than "what
    did I start".

    Args:
        limit: Maximum number of runs to return (the bridge clamps this to
            the range [1, 100]).

    Returns:
        JSON ``{"status": "success", "runs": [...]}`` — each entry has the
        same shape as get_run's response.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs?limit={limit}")
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps({"status": "success", "runs": body})


# ---------------------------------------------------------------------------
# Tool 6: get run data (bounded)
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_run_data(
    run_id: str, max_rows: int = 100, offset: int | None = None, tail: bool = False
) -> str:
    """Read a bounded window of a run's data.

    Row-bounded by design: this never returns an unbounded table. Backed by
    the bridge's in-process live-row buffer (``GET /runs/{id}/data``), so
    reads work with no Tiled server — ``row_count``/``truncated`` describe
    the run's *true* total vs. what this window actually returned.

    Args:
        run_id: Run id returned by queue_add or list_runs.
        max_rows: Maximum number of rows to return (bridge-enforced cap).
        offset: Row offset to start from (``None`` = start from the
            beginning, or from the end if ``tail`` is true).
        tail: When true, return the most recent ``max_rows`` rows instead of
            the earliest ``max_rows`` rows.

    Returns:
        JSON ``{"run_uid", "columns", "rows", "row_count", "truncated",
        "analysis"[, "partial"]}``. ``partial: true`` means the run is still in
        progress and more rows will arrive; an empty/never-started buffer
        returns ``{"columns": [], "rows": []}``. ``run_uid`` is always present
        as a key but is null when the rows came from the live buffer, and
        populated when they came from durable storage — a null there says which
        path served the read, not that anything is missing.

        ``analysis`` is the run's peak statistics, computed over the WHOLE run
        rather than this window, and is always present with the same six keys
        either way: ``{"available", "reason", "x_channel", "x_column",
        "points", "channels"}``. Read ``available`` first.

        - ``available: true`` — ``x_channel`` is the channel the plan declared
          it drives (the independent variable) and ``x_column`` the data column
          it resolved to; ``points`` is how many rows were read. Each entry of
          ``channels`` is ``{"channel", "column", "points", "center", "width",
          "center_of_mass", "reason"}``, where that ``points`` is the number of
          x/y pairs actually used, which is lower than the run's when readings
          dropped out. ``center`` is the midpoint between the outermost
          half-maximum crossings, ``width`` the distance between them, and
          ``center_of_mass`` the readings' weighted mean position — all three
          in the x channel's own units. A null ``center``/``width`` on an
          analyzed channel (``reason`` null) is an ANSWER, not a skip: it says
          the readings never cross their own half-maximum, as a monotonic or
          flat channel does not. A null ``center_of_mass`` likewise means it
          came back non-finite.
        - ``available: false`` — ``reason`` says why, ``channels`` is empty and
          ``x_channel``/``x_column`` are null. ``run_in_progress`` (statistics
          are computed once the run settles; read again later),
          ``plan_identity_unavailable`` (the run cannot be tied back to a known
          plan), ``no_declared_movable`` / ``no_declared_readable`` (the plan
          declares no channel in that role, or the recorded parameters supplied
          none), ``multiple_movable_channels`` (the plan drove several
          channels, so the run has no single independent variable — expected
          for a response matrix, not a failure), ``movable_column_missing``
          (the declared channel never reached the run's data),
          ``insufficient_points`` (fewer than three rows),
          ``statistics_unavailable``.
        - A per-channel ``reason`` is null for an analyzed channel, or one of
          ``channel_column_missing`` (no column for it in the run's data),
          ``channel_is_the_x_axis`` (declared both driven and recorded — it is
          the axis, not a peak in it), ``insufficient_points`` (fewer than
          three usable pairs), ``statistics_unavailable`` (the fit raised for
          this channel alone; the others still report).

        An unavailable analysis is never an error and never a reason to call
        this again unchanged: report the reason in plain terms. Every value is
        a plain JSON number or null — never state a statistic that is null.

        Readable for runs get_run no longer knows about: a run rotated out of
        the manager's history still has its data, so a 404 from get_run is
        never a reason to skip reading here.
    """
    params = f"max_rows={max_rows}"
    if offset is not None:
        params += f"&offset={offset}"
    if tail:
        params += "&tail=true"
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}/data?{params}")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Tool 7: get run figure (point-bounded)
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_run_figure(run_id: str) -> str:
    """Read a run's figure — the plan's own view of what it measured.

    Point-bounded by design: this never returns an unbounded figure. The
    budget is 2000 points for the WHOLE figure (not per series), so a 200k-row
    run and a 20-row run come back the same size. Panels share that budget:
    series split what is left of it evenly, a series needing less than its
    share hands the difference back to the others, and each one is re-strided
    keeping both endpoints, so the x range is never shortened. A stretch of
    the series that contained a gap is represented BY that gap, so missing
    data is never smoothed into a reading — but how many dropouts there were,
    and how wide they ran, does not survive the stride. Grids larger than 40x40 come back as a summary
    instead of cells (see ``"heatmap_summary"`` below). Backed by
    ``GET /runs/{id}/figure``, which serves a live run as it goes and a
    finished one from durable storage.

    Args:
        run_id: Run id returned by queue_add or list_runs.

    Returns:
        JSON ``{"partial", "source", "reason", "panels", "point_budget",
        "points_returned"}``.

        ``partial: true`` means the run was still producing rows when this was
        drawn: read it again for more rather than describing the figure as
        final. ``source`` is ``"live"`` (the bridge's in-process row buffer) or
        ``"tiled"`` (durable storage) — which store answered, not how good the
        data is. ``points_returned`` is what this projection actually spent of
        ``point_budget``.

        ``reason`` is null when the plan drew this figure itself. Any other
        value means the bridge drew its DEFAULT VIEW instead — every numeric
        column the run recorded, drawn against the one channel the plan
        declared it drives when there is exactly one, and against row order
        otherwise (a plan sweeping several channels, or stepping a grid, has no
        single independent variable, so the panel's ``x_label`` says which it
        is). A default view is real data honestly plotted, NOT an error and not
        an empty result:
        narrate it as "the default view" and give the reason in plain terms.

        - ``"no_render"``: this plan declares no view of its own, so the
          default view IS its view. Never report this as a failure.
        - ``"render_failed"``: the plan's own view raised and the default view
          stands in. The data is fine; the plan's drawing code is not.
        - ``"render_not_supported_for_session_plans"``: the plan came from the
          session/unreviewed layer, whose drawing code is never run.
        - ``"params_mismatch"``: the run's recorded parameters did not fit the
          plan's schema, so its view could not be built.
        - ``"plan_identity_unavailable"``: the run could not be tied back to a
          known plan at all.
        - ``"source_unavailable"``: reading the rows failed. ``panels`` is
          empty — that means "could not read", never "the run recorded
          nothing". Try again, and get_run_data as the second opinion.
        - Anything else: still a default view. Report the reason verbatim
          rather than guessing at it.

        A figure carries no numbers about the run itself. Peak center, width
        and center of mass live on get_run_data's ``analysis`` block, computed
        over the whole run once it settles — read them there rather than
        estimating any of them off the plotted points, and report a null one as
        the answer that block gives rather than measuring it off the picture.

        Each panel carries ``title``, ``x_label``/``y_label`` (plus optional
        ``x_units``/``y_units``), ``annotations`` — sentences naming what the
        panel does NOT show, always worth relaying — and one ``mark``:

        - ``"lines"``: ``series``, each ``{label, points: [{x, y}, ...],
          decimated, source_points}``. A ``y`` of null is a gap the run never
          measured, never a zero. ``decimated: true`` means points were
          dropped: say "N of ``source_points`` points shown", never "the run
          took N points". Read the nulls as "there were gaps here", never as
          how many readings were missed — on a thinned series one null can
          stand for a whole run of dropouts, and it is placed where the
          dropout was, not scaled to how long it lasted.
        - ``"bars"``: ``categories`` and positionally aligned ``values``
          (null = a missing bar), with the same ``decimated``/
          ``source_points`` truth.
        - ``"heatmap"``: ``x_labels``/``y_labels`` and row-major ``values``,
          where ``values[j][i]`` is the cell at ``y_labels[j]`` /
          ``x_labels[i]``. Only grids of 40x40 or smaller arrive this way.
        - ``"heatmap_summary"``: this tool's stand-in for a grid too big to
          return as cells. Carries ``rows``/``columns``/``cells``/
          ``empty_cells``; ``x_labels``/``y_labels`` as ``{count, first,
          last}`` objects — the axis vocabulary and its ends, NOT the label
          lists a ``"heatmap"`` mark carries; the ``min`` and ``max``
          cell, and ``largest_magnitude`` — up to five cells with the biggest
          absolute value, each ``{value, x_label, y_label}``. Those are the
          strongest readings, NOT an outlier test: do not call them anomalies,
          and never state a cell value that is not in the summary, because the
          other cells were not returned. Ask for get_run_data if a specific
          cell matters.

    Refusals:
        - unknown_run: neither the bridge's live buffer nor durable storage
          knows this run id. A run get_run no longer lists can still have a
          figure, so a 404 from get_run is never a reason to skip this.
    """
    status, body = await anyio.to_thread.run_sync(_http_get_json, f"/runs/{run_id}/figure")
    if status == 404:
        return make_error("unknown_run", bridge_error_message(body, status), UNKNOWN_RUN_HINTS)
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    try:
        # ValidationError subclasses ValueError, as does decimate()'s rejection
        # of a mark claiming to have grown under decimation. Either way the
        # bridge sent a figure this projection cannot bound, and relaying it
        # unprojected is the one thing this tool promises never to do.
        projected = _project_figure(Figure.model_validate(body))
    except ValueError as exc:
        return make_error(
            "bluesky_bridge_error",
            f"The bridge returned a figure in a shape this tool cannot read: "
            f"{_unreadable_figure_reason(exc)}",
            [
                "The bridge may be running a different OSPREY version than this agent.",
                "get_run_data still serves this run's rows as a table.",
            ],
        )
    return json.dumps(projected)


# ---------------------------------------------------------------------------
# Tool 8: get plan source
# ---------------------------------------------------------------------------

# The bridge's own bounds on ``max_chars`` (see ``_SOURCE_TRUNCATE_CHARS`` /
# ``_SOURCE_TRUNCATE_CHARS_MAX`` in ``services.bluesky_bridge.app``). Mirrored
# here so this tool clamps an out-of-range ask instead of letting the route
# answer it with a 422 the agent can do nothing useful with.
_SOURCE_DEFAULT_MAX_CHARS = 4000
_SOURCE_MIN_MAX_CHARS = 1
_SOURCE_MAX_MAX_CHARS = 200_000


@mcp.tool()
async def get_plan_source(name: str, max_chars: int = _SOURCE_DEFAULT_MAX_CHARS) -> str:
    """Read one plan's source text as it sits on disk.

    This is the plan as written, not a description of it: read it when the
    question is what a plan does step by step, which channels it touches, or
    whether it matches what was asked for. list_plans is where a plan is
    chosen; this is where the choice is checked. Reading source starts
    nothing — execution begins only at queue_start, after a human reviews the
    draft in the plan panel.

    Args:
        name: Plan name exactly as list_plans reports it.
        max_chars: Maximum characters of source text to return. Clamped into
            [1, 200000] before the request, so an over-large ask comes back
            truncated rather than refused.

    Returns:
        JSON ``{"name", "source", "provenance", "validated", "truncated"}``.
        ``source`` is the first ``max_chars`` characters of the file;
        ``truncated`` is true when the file is longer than that, in which case
        never describe the plan's ending — ask again with a larger
        ``max_chars``, up to the 200000 ceiling. If the ask was already
        200000, the rest of the file is not retrievable through this tool and
        must not be described at all. ``provenance`` is the directory layer
        the bridge's own source scan found the file in (``shipped``/
        ``preset``/``facility``/``session``), never something the file
        declares about itself. That scan runs in ascending trust order and
        stops at the first match, so for a name declared in more than one
        layer both ``source`` and ``provenance`` describe the LOWEST-trust
        layer declaring it — which need not be the file the loader would run.
        Treat a plan name you know to be declared in several layers as
        ambiguous here rather than authoritative. ``validated`` is a fresh
        check for a session-tier plan —
        recomputed from the file's current content, so a re-authored file
        reads false until it passes again — and is true by construction for
        the operator-trusted shipped/preset/facility tiers.

    Refusals:
        - plan_source_unavailable: the bridge located no source file for this
          name. That means EITHER the name is not a registered plan, OR it is
          a plan supplied through the legacy ``bluesky.plan_module`` contract,
          whose plans list_plans reports but whose source this endpoint's
          layer scan does not reach. So a 404 here is not proof the plan does
          not exist: check list_plans before telling anyone the name is wrong.
    """
    bounded = max(_SOURCE_MIN_MAX_CHARS, min(int(max_chars), _SOURCE_MAX_MAX_CHARS))
    path = f"/plans/{quote(name, safe='')}/source?max_chars={bounded}"
    status, body = await anyio.to_thread.run_sync(_http_get_json, path)
    if status == 404:
        return make_error(
            "plan_source_unavailable",
            f"The bridge has no source file for plan {name!r}: "
            f"{bridge_error_message(body, status)}",
            [
                "Confirm the name against list_plans — it must match exactly.",
                "A plan supplied through the legacy bluesky.plan_module config key appears "
                "in list_plans but has no source file this endpoint can reach, so it 404s "
                "here even though the plan is real and runnable.",
            ],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(body, status))
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Figure projection
#
# The panel and the agent read the same figure, but not at the same cost. The
# route bounds each line series to 2000 points and leaves heatmaps whole —
# right for a canvas, where a 100x70 response matrix is 7000 colored cells and
# 248 KB of JSON the browser streams once. Reading that into an agent's
# context buys almost nothing: it cannot see the picture, and the numbers it
# would need are a handful of extremes, not seven thousand cells.
#
# So the projection spends ONE budget across the whole figure:
#
#   1. Heatmaps claim their cells first, in panel order. A grid bigger than
#      40x40 in either direction is summarized instead (dimensions, extremes,
#      largest cells) and costs nothing; a grid that fits the limit but not
#      the budget left at its turn is summarized for that reason instead. A
#      heatmap is never partly returned — a grid with holes strided into it
#      reads as missing measurements rather than as omitted cells.
#   2. Whatever is left is divided evenly among the strideable units: every
#      line series, plus each bars mark (its values ride through `decimate` as
#      index-valued points, the route's own technique). A unit needing less
#      than its share keeps everything and returns the difference to the pool.
#   3. Units are re-strided through `figure.decimate` with their prior truth
#      passed back in, so an already-thinned series stays honest about the
#      count it started from rather than reporting this pass alone.
#
# Thinning is reported by each mark's ``decimated``/``source_points`` flags;
# structural loss — a summarized grid, an omitted series — is reported as a
# panel annotation, in sentences, because nothing in the shape would otherwise
# say it happened.
# ---------------------------------------------------------------------------

FIGURE_POINT_BUDGET = DEFAULT_MAX_POINTS
"""Points the projection spends across the whole figure, all panels together."""

HEATMAP_MAX_DIM = 40
"""Largest grid returned as cells. Beyond this, in either direction, a
heatmap is summarized: 40x40 is already 1600 cells, most of the budget."""

HEATMAP_TOP_CELLS = 5
"""How many largest-magnitude cells a heatmap summary names."""

_MAX_REASON_CHARS = 400
"""How much of an unrecognized figure's complaint the refusal quotes."""


def _unreadable_figure_reason(exc: ValueError) -> str:
    """Say why a figure could not be read, in a bounded number of characters.

    The refusal message is the last place a bound could leak: pydantic reports
    a validation failure once per offending node, so ONE extra field on
    `Point` in a 12x2000-point figure is 24000 errors and megabytes of text —
    handed to the agent as an error, which is the payload this tool exists to
    prevent. Only the count and the first few locations tell an operator
    anything anyway. A plain `ValueError` (`decimate` rejecting a mark that
    claims to have grown) is one short sentence and passes through whole.
    """
    if isinstance(exc, ValidationError):
        reported = exc.errors()
        first = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in reported[:3]
        )
        counted = f"{len(reported)} validation error{'' if len(reported) == 1 else 's'}"
        text = f"{counted}; first: {first}"
    else:
        text = str(exc)
    if len(text) <= _MAX_REASON_CHARS:
        return text
    return f"{text[:_MAX_REASON_CHARS]}... (truncated)"


def _heatmap_dims(mark: HeatmapMark) -> tuple[int, int, int]:
    """Return ``(rows, columns, cells)`` for a possibly ragged grid.

    ``columns`` is the widest row (the dimension check errs toward
    summarizing), while ``cells`` counts what is actually there (the budget
    charge errs toward spending less).
    """
    rows = len(mark.values)
    columns = max((len(row) for row in mark.values), default=0)
    cells = sum(len(row) for row in mark.values)
    return rows, columns, cells


def _label_at(labels: Sequence[str], index: int) -> str | None:
    """The label for ``index``, or ``None`` when the axis has none.

    Labels and values are separate lists on the wire, so a grid can carry more
    cells than names. An unnamed cell says so rather than borrowing a
    neighbour's name.
    """
    return labels[index] if 0 <= index < len(labels) else None


def _axis_summary(labels: Sequence[str]) -> dict:
    """The vocabulary of one heatmap axis: how many names, and its ends."""
    return {
        "count": len(labels),
        "first": labels[0] if labels else None,
        "last": labels[-1] if labels else None,
    }


def _summarize_heatmap(mark: HeatmapMark, note: str) -> tuple[dict, str]:
    """Project a heatmap to a fixed-size summary, with the annotation for it.

    The summary answers what an agent can actually use from a grid it cannot
    see: how big it is, how much of it was never measured, where the extremes
    sit, and which cells read largest. It is explicitly lossy, so it carries
    ``decimated: true`` and the cell count it came from, like every other
    mark in the spec.
    """
    rows, columns, cells = _heatmap_dims(mark)
    measured = [
        (value, j, i)
        for j, row in enumerate(mark.values)
        for i, value in enumerate(row)
        if value is not None
    ]

    def _cell(entry: tuple[float, int, int]) -> dict:
        value, j, i = entry
        return {
            "value": value,
            "x_label": _label_at(mark.x_labels, i),
            "y_label": _label_at(mark.y_labels, j),
        }

    largest = sorted(measured, key=lambda entry: (-abs(entry[0]), entry[1], entry[2]))
    summary = {
        "kind": "heatmap_summary",
        "rows": rows,
        "columns": columns,
        "cells": cells,
        "empty_cells": cells - len(measured),
        "value_label": mark.value_label,
        "value_units": mark.value_units,
        "x_labels": _axis_summary(mark.x_labels),
        "y_labels": _axis_summary(mark.y_labels),
        "min": _cell(min(measured, key=lambda entry: (entry[0], entry[1], entry[2])))
        if measured
        else None,
        "max": _cell(max(measured, key=lambda entry: (entry[0], -entry[1], -entry[2])))
        if measured
        else None,
        "largest_magnitude": [_cell(entry) for entry in largest[:HEATMAP_TOP_CELLS]],
        "decimated": True,
        "source_points": mark.source_points,
    }
    return summary, note


def _allocate(needs: Sequence[int], budget: int) -> list[int]:
    """Split ``budget`` across units of the given sizes, evenly and by need.

    Every unit gets the same share, except that a unit needing less than its
    share takes only what it needs and the remainder is re-divided among the
    rest — the whole budget reaches the series that can use it instead of
    being reserved for a five-point series that will never spend it.

    When the budget is smaller than the number of units, an even split would
    give everyone nothing. Those units instead get one point each in figure
    order for as far as the budget reaches, and the units past that get zero:
    omitted, and annotated where they were.
    """
    grants = [0] * len(needs)
    pool = budget
    order = sorted(range(len(needs)), key=lambda index: (needs[index], index))
    starved: list[int] = []
    for position, index in enumerate(order):
        share = pool // (len(order) - position)
        if share < 1:
            starved = order[position:]
            break
        grants[index] = min(needs[index], share)
        pool -= grants[index]
    # A unit with nothing to show must not spend one of the last points: it
    # would cost a series that has data its only place in the figure.
    for index in [index for index in sorted(starved) if needs[index] > 0][:pool]:
        grants[index] = 1
    return grants


def _restride_series(series: Series, budget: int) -> Series:
    """Re-stride one series to ``budget`` points, keeping its prior truth.

    The route may already have thinned this series; ``source_points`` and
    ``decimated`` are passed back in so the result reports what the run
    actually took, not what this pass was handed.
    """
    return Series(
        label=series.label,
        **decimate(
            series.points,
            budget,
            source_points=series.source_points,
            decimated=series.decimated,
        )._asdict(),
    )


def _restride_bars(mark: BarsMark, budget: int) -> BarsMark:
    """Re-stride a bars mark to ``budget`` bars, keeping category alignment.

    Values ride through `decimate` as index-valued points (the route's own
    technique), and the surviving indices select the category/value pairs, so
    the endpoints and any missing bar survive the same way they do on a line.
    """
    indexed = [Point(x=float(index), y=value) for index, value in enumerate(mark.values)]
    thinned = decimate(indexed, budget, source_points=mark.source_points, decimated=mark.decimated)
    kept = [int(point.x) for point in thinned.points]
    return BarsMark(
        label=mark.label,
        categories=[_label_at(mark.categories, index) or "" for index in kept],
        values=[mark.values[index] for index in kept],
        decimated=thinned.decimated,
        source_points=thinned.source_points,
    )


def _project_figure(figure: Figure) -> dict:
    """Project ``figure`` into the whole-figure point budget. See section note."""
    budget = FIGURE_POINT_BUDGET
    # Panel-order pass 1: heatmaps take their cells or become summaries, and
    # every other panel records what it would like to spend.
    marks: list[dict | None] = []
    notes: list[list[str]] = []
    # One entry per strideable unit, in panel order — the order pass 2 walks
    # them again to spend its grant.
    needs: list[int] = []
    for panel in figure.panels:
        mark = panel.mark
        panel_notes: list[str] = []
        if isinstance(mark, HeatmapMark):
            rows, columns, cells = _heatmap_dims(mark)
            if rows > HEATMAP_MAX_DIM or columns > HEATMAP_MAX_DIM:
                summary, note = _summarize_heatmap(
                    mark,
                    f"This grid is {rows}x{columns}, larger than the {HEATMAP_MAX_DIM}x"
                    f"{HEATMAP_MAX_DIM} this tool returns cell by cell, so only its size, "
                    "extremes and largest cells are shown here.",
                )
            elif cells > budget:
                summary, note = _summarize_heatmap(
                    mark,
                    f"This grid's {cells} cells do not fit what is left of the figure's "
                    "point budget, so only its size, extremes and largest cells are "
                    "shown here.",
                )
            else:
                budget -= cells
                marks.append(mark.model_dump())
                notes.append(panel_notes)
                continue
            marks.append(summary)
            notes.append([note])
            continue
        marks.append(None)
        notes.append(panel_notes)
        if isinstance(mark, LinesMark):
            needs.extend(len(series.points) for series in mark.series)
        elif isinstance(mark, BarsMark):
            needs.append(len(mark.values))

    # Pass 2: divide what the heatmaps left over, then re-stride.
    grants = _allocate(needs, budget)
    granted = iter(grants)
    for panel_index, panel in enumerate(figure.panels):
        if marks[panel_index] is not None:
            continue
        mark = panel.mark
        if isinstance(mark, LinesMark):
            kept: list[Series] = []
            omitted = 0
            for series in mark.series:
                budget_for_series = next(granted)
                if budget_for_series < 1 and series.points:
                    omitted += 1
                    continue
                # A series with no points costs nothing and is kept: a run that
                # has not produced rows yet has an empty series, and dropping
                # it would read as "this readback was left out".
                kept.append(_restride_series(series, max(budget_for_series, 1)))
            if omitted:
                notes[panel_index].append(
                    f"{omitted} of {len(mark.series)} series are not shown: the figure's "
                    "point budget did not reach them."
                )
            # Floored at the incoming mark's own aggregates: a projection may
            # RAISE the loss a figure reports, never lower it. A plan is free
            # to set mark-level truth its series do not individually carry
            # (rows it dropped before building them), and recomputing from the
            # series alone would quietly erase that.
            marks[panel_index] = LinesMark(
                series=kept,
                decimated=mark.decimated
                or bool(omitted)
                or any(series.decimated for series in kept),
                source_points=max(
                    mark.source_points,
                    sum(series.source_points for series in mark.series),
                ),
            ).model_dump()
        elif isinstance(mark, BarsMark):
            budget_for_bars = next(granted)
            if budget_for_bars < 1 and mark.values:
                notes[panel_index].append(
                    f"None of this panel's {len(mark.values)} bars are shown: the "
                    "figure's point budget did not reach them."
                )
                marks[panel_index] = BarsMark(
                    label=mark.label,
                    decimated=True,
                    source_points=mark.source_points,
                ).model_dump()
            else:
                marks[panel_index] = _restride_bars(mark, max(budget_for_bars, 1)).model_dump()
        else:
            # Unreachable today: `Mark` is a closed union, so a kind with no
            # rule here would have failed validation before reaching this.
            # It exists so that relaxing validation can never open a hole in
            # the bound — an unbudgetable mark is withheld, never relayed,
            # because its size is by definition unknown.
            marks[panel_index] = {"kind": mark.kind, "omitted": True}
            notes[panel_index].append(
                "This panel's contents are not shown: it carries a kind of mark this tool "
                "has no way to bound, and returning it whole could exceed the figure's "
                "point budget."
            )

    return {
        "partial": figure.partial,
        "source": figure.source,
        "reason": figure.reason,
        "panels": [
            _panel_dict(panel, mark, note)
            for panel, mark, note in zip(figure.panels, marks, notes, strict=True)
        ],
        "point_budget": FIGURE_POINT_BUDGET,
        "points_returned": sum(_points_in(mark) for mark in marks if mark is not None),
    }


def _panel_dict(panel: Panel, mark: dict | None, extra_notes: list[str]) -> dict:
    """One projected panel: the plan's own framing, plus what was left out."""
    return {
        "title": panel.title,
        "x_label": panel.x_label,
        "y_label": panel.y_label,
        "x_units": panel.x_units,
        "y_units": panel.y_units,
        "annotations": [*panel.annotations, *extra_notes],
        "mark": mark,
    }


def _points_in(mark: dict) -> int:
    """How many values a projected mark actually returned.

    A summarized heatmap counts zero: its handful of named cells are a
    description of the grid, not the grid.
    """
    kind = mark.get("kind")
    if kind == "lines":
        return sum(len(series.get("points", [])) for series in mark.get("series", []))
    if kind == "bars":
        return len(mark.get("values", []))
    if kind == "heatmap":
        return sum(len(row) for row in mark.get("values", []))
    return 0
