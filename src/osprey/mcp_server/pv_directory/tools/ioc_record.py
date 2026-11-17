"""MCP tool: ioc_record — one IOC's facts from the control system's own directory.

With the ``pv_directory_fleet`` predicate true, the facts carry the IOC's
``fleet`` entry (its row in the IOC fleet snapshot of the session's target, or
an outcome). The tool then presents that entry as the IOC's state, findings and
read time, or the outcome as a plain statement, and its description says so in
place of telling the agent to read the IOC's heartbeat. With the predicate false
the description and the answer are the directory's facts, exactly as before.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.pv_directory import server as _server
from osprey.mcp_server.pv_directory.server import (
    get_tool_context,
    mcp,
    refuse_unavailable,
)

logger = logging.getLogger("osprey.mcp_server.pv_directory.tools.ioc_record")

TITLE = "Look up an IOC"

#: What a ``fleet`` entry's non-``ok`` outcome means for one IOC, in plain words;
#: none of these is evidence that the IOC is not running.
_STATEMENTS = {
    "no_snapshot": (
        "The IOC fleet service has no snapshot of target {target} yet, so the state of {ioc} "
        "is not known. This says nothing about whether it runs."
    ),
    "not_refreshed": (
        "The IOC fleet service does not refresh target {target}, so the state of {ioc} is not "
        "known. This says nothing about whether it runs."
    ),
    "cannot_list": (
        "The IOC fleet service cannot list the IOCs of target {target}, so the state of {ioc} "
        "is not known. This says nothing about whether it runs."
    ),
    "unavailable": (
        "The IOC fleet service gave no usable answer for target {target}, so the state of {ioc} "
        "is not known right now. This says nothing about whether it runs."
    ),
    "stale": (
        "The snapshot of target {target} is stale: the IOC fleet service stopped refreshing it "
        "(last taken {taken}), so the current state of {ioc} is not known. This says nothing "
        "about whether it runs."
    ),
    "not_in_snapshot": (
        "{ioc} is not in the IOC fleet snapshot of target {target}, so its state is not known. "
        "This says nothing about whether it runs."
    ),
}
_OTHER_OUTCOME = (
    "The IOC fleet service answered {outcome} for target {target}, so the state of {ioc} is not "
    "known. This says nothing about whether it runs."
)


def _iso(epoch: Any) -> str | None:
    if isinstance(epoch, bool) or not isinstance(epoch, int | float):
        return None
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def present_fleet(ioc: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The IOC's ``fleet`` entry as the agent reads it.

    An ``ok`` entry becomes the IOC's ``state``, ``since``, ``reason`` (why that
    state), ``findings``, ``read_at`` and ``read_time`` (when its liveness was
    read, epoch seconds and ISO UTC) and ``liveness_pv``, with ``statement``
    ``None``; any other outcome becomes a ``statement`` that no state is known,
    with the service's ``reason``. Both carry ``outcome``, ``target``,
    ``generated_at`` and ``snapshot_time`` (when the snapshot was taken).
    """
    outcome = str(entry.get("outcome"))
    target = entry.get("target")
    generated_at = entry.get("generated_at")
    taken = _iso(generated_at)
    body: dict[str, Any] = {
        "outcome": outcome,
        "target": target,
        "generated_at": generated_at,
        "snapshot_time": taken,
    }
    row = entry.get("row")
    if outcome == "ok" and isinstance(row, Mapping):
        liveness = row.get("liveness") or {}
        read_at = liveness.get("read_at")
        body.update(
            statement=None,
            state=row.get("state"),
            since=row.get("since"),
            reason=row.get("reason"),
            findings=list(row.get("findings") or ()),
            read_at=read_at,
            read_time=_iso(read_at),
            liveness_pv=liveness.get("pv"),
        )
        return body
    template = _STATEMENTS.get(outcome, _OTHER_OUTCOME)
    body["statement"] = template.format(
        ioc=ioc, target=target, taken=taken or "at an unknown time", outcome=outcome
    )
    body["reason"] = entry.get("reason")
    return body


async def ioc_record(ioc: str) -> str:
    """Look up one IOC in the control system's own directory.

    Answers the IOC's host, its ports, how many channels it serves (and how
    many are inactive, and how many the curated channel list declares, with
    ``curated_reason`` when that cannot be counted), its engineer and where
    that came from (``engineer_source``), and ``health_pvs``: the IOC's health
    channels, each its ``name`` (address) and ``role`` (``heartbeat``,
    ``uptime`` or ``engineer``). Names only, never values: to tell whether the
    IOC is alive, read its health channels with channel_read; a ``heartbeat``
    channel that advances between two reads means alive. On the simulator the
    engineer is the address of a channel; read it with channel_read for the name.
    With an alarm feed configured, ``alarms`` lists the IOC's channels in alarm
    now (``outcome``, ``total``, ``rows`` of pv, severity, status, latched,
    acknowledged and time, and ``counts`` per severity over the whole IOC), or
    ``unavailable`` with the feed's ``reason``.
    ``links.panel`` is the panel page showing this IOC. ``not_listed`` means
    the directory lists no channel served by that IOC name.

    Args:
        ioc: The IOC name, as the directory records it.

    Returns:
        JSON object: ``ioc``, ``outcome``, ``source``, ``host``, ``ports``,
        ``pv_count``, ``inactive_count``, ``curated_count``, ``engineer``,
        ``engineer_source``, ``health_pvs``, ``links``, ``alarms`` with an alarm
        feed configured, and the reasons ``curated_reason`` and
        ``health_pvs_reason`` where they apply.
    """
    name = ioc.strip()
    if not name:
        make_error("validation_error", "ioc_record needs an IOC name.", ["Name one IOC."])
    try:
        facts = await get_tool_context().pv_facts.ioc_facts(name)
        if facts.get("outcome") == "unavailable":
            refuse_unavailable("ioc_record", facts)
        if "fleet" in facts:
            facts = {**facts, "fleet": present_fleet(name, facts["fleet"])}
        return json.dumps(facts)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("ioc_record failed")
        make_error(
            "internal_error",
            f"ioc_record failed: {exc}",
            ["Check the MCP server logs for details."],
        )


#: The description with the IOC fleet configured: the ``fleet`` entry answers
#: whether the IOC is alive, so it asks for no channel read to find out.
FLEET_DESCRIPTION = """Look up one IOC in the control system's own directory.

Answers the IOC's host, its ports, how many channels it serves (and how
many are inactive, and how many the curated channel list declares, with
``curated_reason`` when that cannot be counted), its engineer and where
that came from (``engineer_source``), and ``health_pvs``: the IOC's health
channels, each its ``name`` (address) and ``role`` (``heartbeat``,
``uptime`` or ``engineer``). Names only, never values.
``fleet`` says whether the IOC is alive, from the IOC fleet snapshot of the
session's target: with ``outcome`` ``ok``, its ``state`` (``alive``,
``stalled``, ``silent``, ``unreached`` or ``unknown``), ``since``,
``reason``, ``findings``, and ``read_at``/``read_time`` (when its liveness
was read) with ``snapshot_time`` (when the snapshot was taken); with any
other outcome, a ``statement`` that no state is known and the ``reason``,
which is never evidence that the IOC is not running. Answer liveness from
``fleet``: the snapshot already read the IOC's channels. ``unreached`` and
``unknown`` mean the state could not be established, not that the IOC stopped.
On the simulator the engineer is the address of a channel; read it with
channel_read for the name.
With an alarm feed configured, ``alarms`` lists the IOC's channels in alarm
now (``outcome``, ``total``, ``rows`` of pv, severity, status, latched,
acknowledged and time, and ``counts`` per severity over the whole IOC), or
``unavailable`` with the feed's ``reason``.
``links.panel`` is the panel page showing this IOC. ``not_listed`` means
the directory lists no channel served by that IOC name.

Args:
    ioc: The IOC name, as the directory records it.

Returns:
    JSON object: ``ioc``, ``outcome``, ``source``, ``host``, ``ports``,
    ``pv_count``, ``inactive_count``, ``curated_count``, ``engineer``,
    ``engineer_source``, ``health_pvs``, ``fleet``, ``links``, ``alarms`` with
    an alarm feed configured, and the reasons ``curated_reason`` and
    ``health_pvs_reason`` where they apply."""


def register_described(server: FastMCP, *, fleet: bool) -> Any:
    """Register :func:`ioc_record` on *server*, described for the fleet predicate.

    With *fleet* false the description is the function's own docstring, as it
    always was; with it true, :data:`FLEET_DESCRIPTION`. Not named
    ``register``: that name marks a gated module, and this tool is unconditional.
    """
    extra: dict[str, Any] = {"description": FLEET_DESCRIPTION} if fleet else {}
    return server.tool(
        ioc_record,
        name="ioc_record",
        title=TITLE,
        annotations=ToolAnnotations(readOnlyHint=True),
        **extra,
    )


def _fleet_predicate() -> bool:
    """The installed tool context's ``pv_directory_fleet``; off with no context.

    ``create_server`` installs the context before ``register_tools`` imports
    this module, so the server's registration sees the deployment's predicate.
    """
    context = _server._context
    return bool(context is not None and context.predicates.get("pv_directory_fleet"))


register_described(mcp, fleet=_fleet_predicate())
