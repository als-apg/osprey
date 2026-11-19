"""MCP tool: server_fleet — every server's liveness state on the control target, from the fleet snapshot.

Gated on the ``channel_directory_fleet`` predicate. The tool reads the snapshot the
server fleet service already took, through the fleet connection
(:mod:`osprey.server_fleet_connection`); it reads no channel and queries no
directory, so calling it costs the control system nothing.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from osprey import server_fleet_connection
from osprey.mcp_server.channel_directory.server import get_tool_context, panel_link
from osprey.mcp_server.errors import make_error
from osprey.server_fleet_connection import FleetAnswer, FleetOutcome
from osprey.services.channel_directory.fleet import states as _states
from osprey.services.channel_directory.fleet.states import Finding, State
from osprey.services.channel_directory.vocabulary import (
    NEUTRAL_WORDS,
    Vocabulary,
    target_vocabulary,
)

logger = logging.getLogger("osprey.mcp_server.channel_directory.tools.server_fleet")

#: The most rows one call answers.
SERVER_FLEET_MAX_ROWS = 1000

#: The Channels panel's server fleet view, the page every answer links to.
VIEW_LINK = {"panel": "/panel/channel-finder/#/servers"}

_STATE_BULLET = re.compile(r"^\* ``([a-z_]+)`` -- (.+?)(?=^\* |^\S|\Z)", re.MULTILINE | re.DOTALL)


def _state_meanings() -> dict[str, str]:
    """Each state's meaning in one line, read from the state model's own docstring."""
    found = {
        name: " ".join(text.split()) for name, text in _STATE_BULLET.findall(_states.__doc__ or "")
    }
    return {state.value: found[state.value] for state in State}


#: One line per state, in the state model's order; every answer carries it.
STATE_MEANINGS = _state_meanings()

#: What each non-``ok`` outcome means, as a plain statement about the snapshot
#: and never about any server: none of these is evidence that a server is not
#: running. ``{server}`` and ``{servers}`` are the target's own words.
_STATEMENTS = {
    FleetOutcome.NO_SNAPSHOT: (
        "The server fleet service has no snapshot of target {target} yet; no {server} state is "
        "known for it until its first refresh completes. This says nothing about whether any "
        "{server} runs."
    ),
    FleetOutcome.NOT_REFRESHED: (
        "The server fleet service does not refresh target {target}, so no {server} state is known "
        "for it. This says nothing about whether any {server} runs."
    ),
    FleetOutcome.CANNOT_LIST: (
        "The server fleet service cannot list the {servers} of target {target}, so no {server} "
        "state is known for it. This says nothing about whether any {server} runs."
    ),
    FleetOutcome.UNAVAILABLE: (
        "The server fleet service gave no usable answer for target {target}, so no {server} state "
        "is known right now. This says nothing about whether any {server} runs."
    ),
    FleetOutcome.STALE: (
        "The snapshot of target {target} is stale: the server fleet service stopped refreshing it. "
        "The states below are as of {taken}, not now."
    ),
}


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def _control_target(config: Mapping[str, Any]) -> str:
    """The deployment's control target, as the directory service finds it; no query."""
    from osprey_connectors import control_context
    from osprey_connectors.types import baseline_target

    record = control_context.read_record()
    if record is not None:
        return record.target
    return baseline_target(config.get("control_system"))


def _check(values: list[str] | None, allowed: type[State] | type[Finding], what: str) -> set[str]:
    words = {member.value for member in allowed}
    chosen = set(values or ())
    unknown = sorted(chosen - words)
    if unknown:
        make_error(
            "validation_error",
            f"server_fleet: unknown {what} {', '.join(unknown)}.",
            [f"A {what} is one of: {', '.join(member.value for member in allowed)}."],
        )
    return chosen


def _matches(row: Any, group_by: str | None, filters: dict[str, Any]) -> bool:
    if filters["state"] and row.state.value not in filters["state"]:
        return False
    if filters["finding"] and not {f.value for f in row.findings} & filters["finding"]:
        return False
    if filters["host"] is not None and row.host != filters["host"]:
        return False
    if filters["group"] is not None and (
        group_by is None or getattr(row, group_by) != filters["group"]
    ):
        return False
    text = filters["text"]
    if text:
        haystack = (row.name, row.host, row.class_name, row.group)
        if not any(text.casefold() in field.casefold() for field in haystack if field):
            return False
    return True


def _answer(
    answer: FleetAnswer, filters: dict[str, Any], limit: int, words: Vocabulary = NEUTRAL_WORDS
) -> dict[str, Any]:
    taken = _iso(answer.generated_at)
    body: dict[str, Any] = {
        "outcome": answer.outcome.value,
        "target": answer.target,
        "statement": None,
        "reason": answer.reason,
        "generated_at": answer.generated_at,
        "snapshot_time": taken,
        "meanings": dict(STATE_MEANINGS),
        "links": dict(VIEW_LINK),
    }
    if answer.outcome is not FleetOutcome.OK:
        body["statement"] = _STATEMENTS[answer.outcome].format(
            target=answer.target, taken=taken, server=words.server, servers=words.servers
        )
    snapshot = answer.snapshot
    if snapshot is None:
        return body

    document = snapshot.to_json()
    filtered = any(value for value in filters.values())
    body["read_path"] = document["read_path"]
    body["group_by"] = snapshot.group_by
    if not filtered:
        body["counts"] = document["counts"]
        body["total"] = len(snapshot.servers)
        body["sources"] = document["sources"]
    else:
        body["filters"] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in filters.items()
            if value
        }
    chosen = [
        encoded
        for row, encoded in zip(snapshot.servers, document["servers"], strict=True)
        if _matches(row, snapshot.group_by, filters)
    ]
    body["matched"] = len(chosen)
    body["rows"] = [{**row, "links": panel_link("server", row["name"])} for row in chosen[:limit]]
    body["truncated"] = len(chosen) > limit
    return body


async def server_fleet(
    state: list[str] | None = None,
    finding: list[str] | None = None,
    group: str | None = None,
    host: str | None = None,
    text: str | None = None,
    limit: int = 100,
) -> str:
    """List the servers of the deployment's control target with their liveness state, from the fleet snapshot.

    A server is the process serving channels, as the control system's
    directory names it. Answers which servers are alive, stalled, silent,
    unreached or unknown, and which carry a finding, as of the snapshot the
    server fleet service took on its last refresh. Reads no channel and queries no
    directory: do not read heartbeat channels to answer a fleet question.
    ``unreached`` and ``unknown`` mean the state could not be established, not
    that a server stopped. Rows come in the view's order, most urgent first.

    Every answer carries ``outcome`` (``ok``, ``stale``, ``no_snapshot``,
    ``not_refreshed``, ``cannot_list`` or ``unavailable``), ``target``,
    ``statement`` (what a non-``ok`` outcome means, in plain words; ``None``
    on ``ok``), ``reason``, ``generated_at`` and ``snapshot_time`` (when the
    snapshot was taken, epoch seconds and ISO UTC), ``meanings`` (each state's
    meaning in one line) and ``links.panel`` (the server fleet view). A
    non-``ok`` outcome other than ``stale`` has no rows: no server state is
    known, which is never evidence that a server is not running. ``stale``
    keeps the rows of the last snapshot; they are its states, not the current
    ones.

    With a snapshot, the answer also carries ``read_path`` (``status`` ``ok``,
    ``down`` or ``unverified``, and its ``channel``; when ``down``, every server is
    ``unknown``), ``group_by``, ``matched`` (rows passing the filters),
    ``rows`` and ``truncated``. Without any filter it carries ``counts``
    (``by_state``, ``by_finding``, every state and finding keyed), ``total``
    and ``sources`` (each source of the snapshot with its ``outcome`` and
    ``reason``); with one, ``filters`` echoes them. A row holds ``name``,
    ``host``, ``class_name``, ``group``, ``state``, ``since``, ``reason``,
    ``findings``, ``directory_status``, ``listed``, ``sources``,
    ``liveness``, ``health`` and ``links.panel``, the server's page.

    Args:
        state: Only servers in one of these states.
        finding: Only servers carrying one of these findings (``degraded``,
            ``expected_running_not_alive``, ``expected_stopped_alive``,
            ``alive_unexpected``).
        group: Only servers whose value of the snapshot's grouping field
            (``group_by``: host, class or roster group) is this.
        host: Only servers on this host.
        text: Only servers whose name, host, class or group contains this,
            case-insensitive.
        limit: The most rows answered, 0 to 1000; a cut sets ``truncated``.

    Returns:
        JSON object as described above.
    """
    if not 0 <= limit <= SERVER_FLEET_MAX_ROWS:
        make_error(
            "validation_error",
            f"server_fleet takes a limit of 0 to {SERVER_FLEET_MAX_ROWS} (got {limit}).",
            [f"Ask for at most {SERVER_FLEET_MAX_ROWS} rows per call, or filter."],
        )
    filters = {
        "state": _check(state, State, "state"),
        "finding": _check(finding, Finding, "finding"),
        "group": group or None,
        "host": host or None,
        "text": (text or "").strip() or None,
    }
    try:
        context = get_tool_context()
        config = context.config
        answer = await server_fleet_connection.fetch_snapshot(config, _control_target(config))
        words = target_vocabulary(config.get("control_system"), answer.target)
        return json.dumps(_answer(answer, filters, limit, words))
    except ToolError:
        raise
    except Exception as exc:
        logger.exception("server_fleet failed")
        make_error(
            "internal_error",
            f"server_fleet failed: {exc}",
            ["Check the MCP server logs for details."],
        )


def register(server: FastMCP) -> Any:
    """Register :func:`server_fleet` on *server*."""
    return server.tool(
        server_fleet,
        name="server_fleet",
        title="List the servers and their liveness",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
