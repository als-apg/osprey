"""MCP tool: server_record — one server's facts from the control system's own directory.

With the ``channel_directory_fleet`` predicate true, the facts carry the server's
``fleet`` entry (its row in the server fleet snapshot of the session's target, or
an outcome). The tool then presents that entry as the server's state, findings and
read time, or the outcome as a plain statement, and its description says so in
place of telling the agent to read the server's heartbeat. With the predicate false
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

from osprey.mcp_server.channel_directory import server as _server
from osprey.mcp_server.channel_directory.server import (
    get_tool_context,
    mcp,
    refuse_unavailable,
)
from osprey.mcp_server.errors import make_error
from osprey.services.channel_directory.vocabulary import (
    NEUTRAL_WORDS,
    Vocabulary,
    target_vocabulary,
)

logger = logging.getLogger("osprey.mcp_server.channel_directory.tools.server_record")

TITLE = "Look up a server"

#: What a ``fleet`` entry's non-``ok`` outcome means for one server, in plain
#: words; none of these is evidence that the server is not running.
#: ``{servers}`` is the target's own word for its servers.
_STATEMENTS = {
    "no_snapshot": (
        "The server fleet service has no snapshot of target {target} yet, so the state of {server} "
        "is not known. This says nothing about whether it runs."
    ),
    "not_refreshed": (
        "The server fleet service does not refresh target {target}, so the state of {server} is not "
        "known. This says nothing about whether it runs."
    ),
    "cannot_list": (
        "The server fleet service cannot list the {servers} of target {target}, so the state of "
        "{server} is not known. This says nothing about whether it runs."
    ),
    "unavailable": (
        "The server fleet service gave no usable answer for target {target}, so the state of {server} "
        "is not known right now. This says nothing about whether it runs."
    ),
    "stale": (
        "The snapshot of target {target} is stale: the server fleet service stopped refreshing it "
        "(last taken {taken}), so the current state of {server} is not known. This says nothing "
        "about whether it runs."
    ),
    "not_in_snapshot": (
        "{server} is not in the server fleet snapshot of target {target}, so its state is not known. "
        "This says nothing about whether it runs."
    ),
}
_OTHER_OUTCOME = (
    "The server fleet service answered {outcome} for target {target}, so the state of {server} is not "
    "known. This says nothing about whether it runs."
)


def _iso(epoch: Any) -> str | None:
    if isinstance(epoch, bool) or not isinstance(epoch, int | float):
        return None
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def present_fleet(
    server: str, entry: Mapping[str, Any], words: Vocabulary = NEUTRAL_WORDS
) -> dict[str, Any]:
    """The server's ``fleet`` entry as the agent reads it, a statement in the target's *words*.

    An ``ok`` entry becomes the server's ``state``, ``since``, ``reason`` (why that
    state), ``findings``, ``read_at`` and ``read_time`` (when its liveness was
    read, epoch seconds and ISO UTC) and ``liveness_channel``, with ``statement``
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
            liveness_channel=liveness.get("channel"),
        )
        return body
    template = _STATEMENTS.get(outcome, _OTHER_OUTCOME)
    body["statement"] = template.format(
        server=server,
        servers=words.servers,
        target=target,
        taken=taken or "at an unknown time",
        outcome=outcome,
    )
    body["reason"] = entry.get("reason")
    return body


async def server_record(server: str) -> str:
    """Look up one server in the control system's own directory.

    A server is the process serving channels, as the control system's
    directory names it. Answers the server's host, its ports, how many channels it
    serves (and how many are inactive, and how many the curated channel list
    declares, with ``curated_reason`` when that cannot be counted), its
    engineer and where that came from (``engineer_source``), and
    ``health_channels``: the server's health channels, each its ``name``
    (address) and ``role`` (``heartbeat``, ``uptime`` or ``engineer``). Names
    only, never values: to tell whether the server is alive,
    read its health channels with channel_read; a ``heartbeat`` channel that
    advances between two reads means alive. On the simulator the engineer is
    the address of a channel; read it with channel_read for the name.
    With an alarm feed configured, ``alarms`` lists the server's channels in alarm
    now (``outcome``, ``total``, ``rows`` of channel, severity, status, latched,
    acknowledged and time, and ``counts`` per severity over the whole server), or
    ``unavailable`` with the feed's ``reason``.
    ``links.panel`` is the panel page showing this server. ``not_listed`` means
    the directory lists no channel served by that server name.

    Args:
        server: The server's name, as the directory records it.

    Returns:
        JSON object: ``server``, ``outcome``, ``source``, ``host``, ``ports``,
        ``channel_count``, ``inactive_count``, ``curated_count``, ``engineer``,
        ``engineer_source``, ``health_channels``, ``links``, ``alarms`` with an alarm
        feed configured, and the reasons ``curated_reason`` and
        ``health_channels_reason`` where they apply.
    """
    name = server.strip()
    if not name:
        make_error("validation_error", "server_record needs a server name.", ["Name one server."])
    try:
        context = get_tool_context()
        facts = await context.channel_facts.server_facts(name)
        if facts.get("outcome") == "unavailable":
            refuse_unavailable("server_record", facts)
        if "fleet" in facts:
            target = str(facts["source"]["target"])
            words = target_vocabulary(context.config.get("control_system"), target)
            facts = {**facts, "fleet": present_fleet(name, facts["fleet"], words)}
        return json.dumps(facts)
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("server_record failed")
        make_error(
            "internal_error",
            f"server_record failed: {exc}",
            ["Check the MCP server logs for details."],
        )


#: The description with the server fleet configured: the ``fleet`` entry answers
#: whether the server is alive, so it asks for no channel read to find out.
FLEET_DESCRIPTION = """Look up one server in the control system's own directory.

A server is the process serving channels, as the control system's
directory names it. Answers the server's host, its ports, how many channels it
serves (and how many are inactive, and how many the curated channel list
declares, with ``curated_reason`` when that cannot be counted), its
engineer and where that came from (``engineer_source``), and
``health_channels``: the server's health channels, each its ``name``
(address) and ``role`` (``heartbeat``, ``uptime`` or ``engineer``). Names
only, never values.
``fleet`` says whether the server is alive, from the server fleet snapshot of the
session's target: with ``outcome`` ``ok``, its ``state`` (``alive``,
``stalled``, ``silent``, ``unreached`` or ``unknown``), ``since``,
``reason``, ``findings``, and ``read_at``/``read_time`` (when its liveness
was read) with ``snapshot_time`` (when the snapshot was taken); with any
other outcome, a ``statement`` that no state is known and the ``reason``,
which is never evidence that the server is not running. Answer liveness from
``fleet``: the snapshot already read the server's channels. ``unreached`` and
``unknown`` mean the state could not be established, not that the server stopped.
On the simulator the engineer is the address of a channel; read it with
channel_read for the name.
With an alarm feed configured, ``alarms`` lists the server's channels in alarm
now (``outcome``, ``total``, ``rows`` of channel, severity, status, latched,
acknowledged and time, and ``counts`` per severity over the whole server), or
``unavailable`` with the feed's ``reason``.
``links.panel`` is the panel page showing this server. ``not_listed`` means
the directory lists no channel served by that server name.

Args:
    server: The server's name, as the directory records it.

Returns:
    JSON object: ``server``, ``outcome``, ``source``, ``host``, ``ports``,
    ``channel_count``, ``inactive_count``, ``curated_count``, ``engineer``,
    ``engineer_source``, ``health_channels``, ``fleet``, ``links``, ``alarms`` with
    an alarm feed configured, and the reasons ``curated_reason`` and
    ``health_channels_reason`` where they apply."""


def register_described(server: FastMCP, *, fleet: bool) -> Any:
    """Register :func:`server_record` on *server*, described for the fleet predicate.

    With *fleet* false the description is the function's own docstring, as it
    always was; with it true, :data:`FLEET_DESCRIPTION`. Not named
    ``register``: that name marks a gated module, and this tool is unconditional.
    """
    extra: dict[str, Any] = {"description": FLEET_DESCRIPTION} if fleet else {}
    return server.tool(
        server_record,
        name="server_record",
        title=TITLE,
        annotations=ToolAnnotations(readOnlyHint=True),
        **extra,
    )


def _fleet_predicate() -> bool:
    """The installed tool context's ``channel_directory_fleet``; off with no context.

    ``create_server`` installs the context before ``register_tools`` imports
    this module, so the server's registration sees the deployment's predicate.
    """
    context = _server._context
    return bool(context is not None and context.predicates.get("channel_directory_fleet"))


register_described(mcp, fleet=_fleet_predicate())
