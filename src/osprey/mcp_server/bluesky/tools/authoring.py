"""MCP tools: author and validate a session-tier Bluesky plan file.

Both tools are thin HTTP clients of the bridge's authoring routes and reach
NO hardware — ``write_plan`` only writes a file (never imports or
execs it), and ``validate_plan``'s dry run drives mock devices only,
in a subprocess with ``EPICS_CA_*`` neutralized (see
``osprey.services.bluesky_bridge.plan_validation``). Both therefore work
identically whether ``control_system.writes_enabled`` is on or off, and
neither carries the kill-switch (``_WRITES_CHECK``) hook — see the ``bluesky``
``ServerDefinition`` in ``registry/mcp.py``. Both are still approval-gated
(``permissions_ask``) so a human sees every authored/validated plan body.

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
write_plan          POST /plans/session
validate_plan       POST /plans/validate
==========================  =================================================

Same conventions as ``read_tools.py``: ``async def``, JSON string return
(``json.dumps`` / ``make_error``), blocking HTTP dispatched via
``anyio.to_thread.run_sync``, and the shared ``_http_post_json`` /
``bridge_error_message`` helpers from ``bluesky/server_context.py``.
"""

from __future__ import annotations

import json

import anyio

from osprey.mcp_server.bluesky.server import mcp
from osprey.mcp_server.bluesky.server_context import _http_post_json, bridge_error_message
from osprey.mcp_server.errors import make_error


# ---------------------------------------------------------------------------
# Tool 1: write a session-tier plan file
# ---------------------------------------------------------------------------
@mcp.tool()
async def write_plan(
    name: str,
    category: str,
    required_devices: list[str],
    writes: bool,
    body: str,
    description: str = "",
) -> str:
    """Author a session-tier plan file on the bridge. Reaches NO hardware.

    The bridge assembles a `PLAN_METADATA = {...}` block from
    ``name``/``description``/``category``/``required_devices``/``writes`` and
    prepends it to ``body`` (your own ``PARAMS`` + ``build_plan`` source, per
    the layered directory catalog's file contract), writing the combined text
    as one file. The bridge NEVER imports or execs this file — it is inert
    until validate_plan passes it and, later, a human approves
    launching it. Re-authoring an existing ``name`` overwrites the file and
    invalidates any prior passing validation (its content hash changes).

    Args:
        name: Plan name — must be a valid Python identifier; doubles as the
            on-disk file stem and the generated metadata's ``name`` field.
        category: Free-text grouping shown to operators (e.g. "accelerator").
        required_devices: Names of the `PARAMS` fields naming devices the
            plan drives/reads (e.g. ``["correctors", "detectors"]``).
        writes: Whether this plan moves a device (vs. read-only). Authoring
            metadata only — has no effect on whether writes actually happen;
            that is governed entirely by ``control_system.writes_enabled``.
        body: Your plan's own source: a `PARAMS` pydantic model (optional)
            and a `build_plan(devices, params)` callable, exactly as a
            directory-layer plan file needs — see the orm
            exemplar plan for the expected shape.
        description: Human-readable summary of what the plan does.

    Returns:
        JSON ``{"name", "content_hash"}`` on success — pass ``name`` to
        validate_plan next. On rejection (e.g. an invalid ``name``),
        an error envelope naming what to fix.
    """
    payload = {
        "name": name,
        "description": description,
        "category": category,
        "required_devices": required_devices,
        "writes": writes,
        "body": body,
    }
    status, resp_body = await anyio.to_thread.run_sync(_http_post_json, "/plans/session", payload)
    if status != 200:
        return make_error(
            "plan_write_rejected",
            bridge_error_message(resp_body, status),
            [
                "Check name is a valid Python identifier (used as the file stem).",
                "Check category/required_devices/writes are present and well-typed.",
            ],
        )
    return json.dumps(resp_body)


# ---------------------------------------------------------------------------
# Tool 2: validate a session-tier plan file
# ---------------------------------------------------------------------------
@mcp.tool()
async def validate_plan(
    name: str,
    sample_args: dict | None = None,
    dry_run_timeout: float = 30.0,
) -> str:
    """Validate a session plan already written by write_plan. Reaches NO hardware.

    Runs the bridge's three-stage validator (static import/pattern allowlist,
    then a mock-device dry run) against the CURRENT on-disk content of the
    named session plan file — never a body you pass here directly, so the
    validated bytes are always exactly the file's bytes. The dry run drives
    in-process mock devices only, in a subprocess whose ``EPICS_CA_*``
    variables are neutralized; it never reaches a real device regardless of
    ``control_system.writes_enabled``. A passing validation is recorded by
    content hash so the plan becomes loadable/promotable (tasks 2.4/2.5);
    editing the file afterward changes its hash and drops the record, so it
    must be re-validated.

    Args:
        name: Session plan name, as passed to write_plan.
        sample_args: Sample `PARAMS` field values used to build the dry run's
            generator and mock devices (e.g. device names, point counts).
            Omit only for a `PARAMS` with no required fields.
        dry_run_timeout: Seconds the dry-run subprocess is given to drive the
            plan to completion.

    Returns:
        JSON ``{"passed", "reasons", "content_hash"}``. ``reasons`` is empty
        on a pass; on a failure it names every rejection from whichever stage
        stopped the plan.
    """
    payload = {"name": name, "sample_args": sample_args, "dry_run_timeout": dry_run_timeout}
    status, resp_body = await anyio.to_thread.run_sync(_http_post_json, "/plans/validate", payload)
    if status == 404:
        return make_error(
            "unknown_session_plan",
            bridge_error_message(resp_body, status),
            ["Call write_plan first to author this plan name."],
        )
    if status != 200:
        return make_error("bluesky_bridge_error", bridge_error_message(resp_body, status))
    return json.dumps(resp_body)
