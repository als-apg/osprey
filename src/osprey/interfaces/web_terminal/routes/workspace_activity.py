"""Activity-history rows for the agent's ``set_workspace`` calls on CANOPY.

CANOPY's MCP transport is reached by the agent through the terminal's own panel
proxy (the CANOPY sidecar's agent hop, see :mod:`.proxy`), and through nothing
else of OSPREY's: those calls never pass the audit middleware OSPREY's own MCP
servers carry. The hop is therefore the one point where OSPREY itself observes
a call, and this module turns what it observes there into history.

For every JSON-RPC ``tools/call`` of ``set_workspace`` in a request body, one
row goes into the agent-activity history buffer, through the same
:func:`~.agent_activity.record_activity` the panel routes use for an agent
``open_panel``. The row carries the commands as sent (each verb, and the key,
ref, tool, closed-group id or tile id it names — for a ``drop``, what its
``source`` moves) and the batch status the tool
*returned*: ``done``, ``pending`` (a batch the window had not answered within
the call's wait stays pending here; nothing chases it), or ``refused`` with its
reason. A call whose answer is a JSON-RPC error, a tool error, or anything with
no batch in it reads ``error``; a call the hop saw go out but whose answer it
never saw (the stream ended first, the upstream was unreachable, a redirect was
relayed instead) reads ``unanswered``. CANOPY knows nothing of this: the row is
derived from the request and the reply alone.

::

    target: {"kind": "panel", "panel": "canopy",
             "workspace": {"commands": [{"cmd": str, "key"?: str}, ...],
                           "more"?: int,
                           "status": str, "reason"?: str}}

Only a request that carries such a call is observed, and only its own reply is
read; every other request on the hop is relayed exactly as before. The reply is
read as it streams past and is never held back: an SSE reply is parsed event by
event and observation stops at the last answer it waits for, a single JSON
reply is copied as it goes and parsed when it ends. Rows are history only and
are not broadcast, like the panel routes' rows. Nothing in here may fail the
relay: a reply that cannot be read yields the row's fallback status instead.

The strings in a row are agent-chosen and are rendered as text, never markup;
they are capped here so a runaway call cannot bloat the buffer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import Request

from osprey.interfaces.web_terminal.routes.agent_activity import record_activity
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID

logger = logging.getLogger(__name__)

#: The CANOPY tool whose calls become rows, as the server registers it (the
#: client-side ``mcp__canopy__`` prefix never reaches the transport).
SET_WORKSPACE_TOOL = "set_workspace"

#: Most commands one row lists; CANOPY refuses a longer batch, and the row
#: says how many more there were rather than listing them.
MAX_ROW_COMMANDS = 20

#: Cap on any agent-chosen string a row carries.
MAX_ROW_TEXT = 256

#: Statuses the tool returns, and the two the hop supplies itself.
RETURNED_STATUSES = ("done", "pending", "refused")
ERROR_STATUS = "error"
UNANSWERED_STATUS = "unanswered"

#: A reason is a token like ``no_window``; anything else is not shown.
_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

#: Where an SSE event ends, whichever line ending the server uses.
_SSE_EVENT_END_RE = re.compile(rb"\r\n\r\n|\n\n|\r\r")

#: The fields of a command's ``args`` that name what it acts on, in the order
#: they are looked for: a tab ``key``, a screen or collection ``ref``, a tool
#: name, a closed-group ``id``, a ``tile`` id (``close_tile``).
_KEY_FIELDS = ("key", "ref", "tool", "id", "tile")

#: The fields of a ``drop``'s ``source`` that name what moves: an open ``tab``
#: (its key) or a whole ``tile`` (its id). A ``new`` source nests one level
#: deeper and names a screen ``ref`` or a ``tool``.
_SOURCE_FIELDS = ("tab", "tile")
_NEW_SOURCE_FIELDS = ("ref", "tool")

#: Most reply bytes held while waiting for the end of one event or one JSON
#: reply. Past it the reply is relayed on unread and the calls read
#: ``unanswered``.
MAX_HELD_BYTES = 8 * 1024 * 1024


def _text(value: object) -> str | None:
    """*value* as a capped string, or ``None`` when it is not a non-empty string."""
    if not isinstance(value, str) or not value:
        return None
    return value[:MAX_ROW_TEXT]


def _id_of(message_id: object) -> str:
    """A JSON-RPC id in a form that is hashable and keeps ``1`` apart from ``"1"``."""
    return json.dumps(message_id, sort_keys=True)


def set_workspace_calls(body: bytes) -> dict[str, Any]:
    """The ``set_workspace`` calls in a request body, by JSON-RPC id.

    Args:
        body: The request body as the agent's client sent it; a single message
            or a batch array.

    Returns:
        Each call's id (see :func:`_id_of`) mapped to the ``commands`` argument
        it sent, unread. Empty for any body that is not JSON-RPC or calls no
        ``set_workspace``.
    """
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {}
    messages = parsed if isinstance(parsed, list) else [parsed]
    calls: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("method") != "tools/call":
            continue
        if "id" not in message:
            continue
        params = message.get("params")
        if not isinstance(params, dict) or params.get("name") != SET_WORKSPACE_TOOL:
            continue
        arguments = params.get("arguments")
        commands = arguments.get("commands") if isinstance(arguments, dict) else None
        calls[_id_of(message["id"])] = commands
    return calls


def command_rows(commands: object) -> tuple[list[dict[str, str]], int]:
    """The commands a call sent, as a row lists them.

    Args:
        commands: The ``commands`` argument as sent; anything that is not a
            list lists nothing.

    Returns:
        ``(rows, more)``: one ``{"cmd", "key"?}`` per command, up to
        :data:`MAX_ROW_COMMANDS`, and how many commands past that were not
        listed. A verb that is not a string reads ``?``.
    """
    if not isinstance(commands, list):
        return [], 0
    rows: list[dict[str, str]] = []
    for command in commands[:MAX_ROW_COMMANDS]:
        if not isinstance(command, dict):
            rows.append({"cmd": "?"})
            continue
        row = {"cmd": _text(command.get("cmd")) or "?"}
        key = _subject(command.get("args"))
        if key is not None:
            row["key"] = key
        rows.append(row)
    return rows, max(0, len(commands) - MAX_ROW_COMMANDS)


def _first_text(fields: tuple[str, ...], args: dict[str, Any]) -> str | None:
    """The first of *fields* in *args* that holds a non-empty string, capped."""
    for field in fields:
        text = _text(args.get(field))
        if text is not None:
            return text
    return None


def _subject(args: object) -> str | None:
    """What a command acts on: a named field of its ``args``, else a ``drop``'s source.

    A ``drop`` names what moves inside ``source``: ``{tab: key}``,
    ``{tile: id}``, or ``{new: {ref}}`` / ``{new: {tool}}`` for an item not yet
    open.
    """
    if not isinstance(args, dict):
        return None
    subject = _first_text(_KEY_FIELDS, args)
    if subject is not None:
        return subject
    source = args.get("source")
    if not isinstance(source, dict):
        return None
    subject = _first_text(_SOURCE_FIELDS, source)
    if subject is not None:
        return subject
    new = source.get("new")
    return _first_text(_NEW_SOURCE_FIELDS, new) if isinstance(new, dict) else None


def _tool_document(result: dict[str, Any]) -> dict[str, Any] | None:
    """The dict a tool returned: its structured content, else its JSON text."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "batch" in structured:
        return structured
    content = result.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            document = json.loads(item.get("text") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(document, dict) and "batch" in document:
            return document
    return None


def batch_outcome(message: dict[str, Any]) -> dict[str, str]:
    """The batch status a JSON-RPC answer to ``set_workspace`` returned.

    Args:
        message: The JSON-RPC response whose id matched the call.

    Returns:
        ``{"status"}`` and, when the batch carried one, ``{"reason"}``. The
        status is the batch's own when it is one CANOPY returns, else
        ``error``.
    """
    result = message.get("result")
    if "error" in message or not isinstance(result, dict) or result.get("isError"):
        return {"status": ERROR_STATUS}
    document = _tool_document(result)
    batch = document.get("batch") if document else None
    if not isinstance(batch, dict) or batch.get("status") not in RETURNED_STATUSES:
        return {"status": ERROR_STATUS}
    outcome = {"status": batch["status"]}
    reason = batch.get("reason")
    if isinstance(reason, str) and _REASON_RE.match(reason):
        outcome["reason"] = reason
    return outcome


def workspace_target(commands: object, outcome: dict[str, str]) -> dict[str, Any]:
    """The activity target of one ``set_workspace`` row."""
    rows, more = command_rows(commands)
    workspace: dict[str, Any] = {"commands": rows}
    if more:
        workspace["more"] = more
    workspace.update(outcome)
    return {"kind": "panel", "panel": CANOPY_PANEL_ID, "workspace": workspace}


class SetWorkspaceObserver:
    """Reads one request's reply as it is relayed and records a row per call.

    Lifecycle: :meth:`begin` with the reply's content type, :meth:`feed` with
    every chunk relayed, :meth:`close` once the relay is over however it ended.
    Every call still unanswered at :meth:`close` is recorded as
    ``unanswered``, so each call yields exactly one row. All three are
    idempotent past :meth:`close` and never raise.
    """

    def __init__(self, request: Request, calls: dict[str, Any]) -> None:
        self._request = request
        self._calls = dict(calls)
        self._stream = False
        self._held = bytearray()
        self._reading = True
        self._closed = False

    def begin(self, content_type: str) -> None:
        """Take the reply's content type: an SSE stream, or one JSON reply."""
        self._stream = content_type.split(";")[0].strip().lower() == "text/event-stream"

    def feed(self, chunk: bytes) -> None:
        """Read one relayed chunk. The chunk itself is never altered or delayed."""
        if self._closed or not self._reading or not self._calls or not chunk:
            return
        try:
            self._held.extend(chunk)
            if len(self._held) > MAX_HELD_BYTES:
                self._stop_reading()
                return
            if self._stream:
                self._read_events()
        except Exception:  # observation must never fail the relay
            logger.debug("set_workspace reply not readable", exc_info=True)
            self._stop_reading()

    def close(self) -> None:
        """Finish: parse a JSON reply, then record every call still unanswered."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._reading and not self._stream and self._held:
                self._answer(bytes(self._held))
        except Exception:  # observation must never fail the relay
            logger.debug("set_workspace reply not readable", exc_info=True)
        self._held = bytearray()
        for commands in self._calls.values():
            self._record(commands, {"status": UNANSWERED_STATUS})
        self._calls.clear()

    def _stop_reading(self) -> None:
        self._reading = False
        self._held = bytearray()

    def _read_events(self) -> None:
        """Parse every complete SSE event held; keep the incomplete tail."""
        while self._calls:
            end = _SSE_EVENT_END_RE.search(self._held)
            if end is None:
                return
            event = bytes(self._held[: end.start()])
            del self._held[: end.end()]
            lines = event.decode("utf-8", errors="replace").splitlines()
            data = "\n".join(
                line[5:].removeprefix(" ") for line in lines if line.startswith("data:")
            )
            if data:
                self._answer(data)
        self._stop_reading()

    def _answer(self, payload: bytes | str) -> None:
        """Record a row for each awaited call this JSON payload answers."""
        try:
            parsed = json.loads(payload)
        except ValueError:
            return
        for message in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(message, dict) or "id" not in message:
                continue
            if "result" not in message and "error" not in message:
                continue
            key = _id_of(message["id"])
            if key in self._calls:
                self._record(self._calls.pop(key), batch_outcome(message))

    def _record(self, commands: object, outcome: dict[str, str]) -> None:
        try:
            record_activity(self._request, SET_WORKSPACE_TOOL, workspace_target(commands, outcome))
        except Exception:  # history must never fail the relay
            logger.warning("set_workspace activity row not recorded", exc_info=True)


def observe_set_workspace(
    request: Request, panel_id: str, body: bytes
) -> SetWorkspaceObserver | None:
    """An observer for this agent-hop request, or ``None`` when there is nothing to observe.

    Args:
        request: The admitted agent-hop request; its ``app.state`` holds the buffer.
        panel_id: The panel the hop is for; only CANOPY's calls are observed.
        body: The request body as it is forwarded.
    """
    if panel_id != CANOPY_PANEL_ID:
        return None
    calls = set_workspace_calls(body)
    if not calls:
        return None
    return SetWorkspaceObserver(request, calls)
