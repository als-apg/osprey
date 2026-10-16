"""The history page's view contract: one SQL file per query the page sends.

Every query reads the ``telemetry`` database only, never a ``system.*`` table,
and takes its values as the store's typed query parameters (``{name:Type}``),
sent as ``param_<name>`` beside the query, so no value is ever spliced into
SQL. The page sends each one as ``EXECUTE AS <view user> <query>``; the view
user's grants and row policies decide what it may read. Every query is bounded
by its own ``LIMIT``, and every row query de-duplicates with ``LIMIT 1 BY`` on
its natural key (the ledger feed is at-least-once).

``reconstruct_session.sql`` sits beside them: the readers' recipe, written to
run as a reader under ``reader_limits``.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping
from importlib import resources
from typing import Any

#: Every query the page sends, by file stem.
PAGE_QUERIES = (
    "session_list",
    "session_users",
    "conversation",
    "tool_log",
    "approvals",
    "preflight",
    "load_full_log",
    "load_full_ledger",
    "agents",
)
#: The readers' recipe, which the docs include verbatim.
RECIPE = "reconstruct_session"
QUERIES = (*PAGE_QUERIES, RECIPE)

#: The session list's default window, in days back from now. The "older"
#: control steps back one day at a time, up to the viewer's ``days_back``.
SESSION_LIST_WINDOW_DAYS = 3
#: Rows per page of the session list, and of every paged row query.
SESSION_LIST_LIMIT = 200
PAGE_LIMIT = 500
#: Characters of content or record a page row carries; the load-full queries
#: read one whole row.
PREVIEW_CHARS = 4096

#: The agent's conversation events and the attribute each one's text is in.
USER_PROMPT_EVENT = "user_prompt"
ASSISTANT_RESPONSE_EVENT = "assistant_response"
CONTENT_ATTRIBUTES = {USER_PROMPT_EVENT: "prompt", ASSISTANT_RESPONSE_EVENT: "response"}
#: What the agent records in place of the text when its content gate is off.
REDACTED = "<REDACTED>"

#: The cursor of the first page of a keyset-paged row query: older than any
#: row the store keeps (the gateway refuses rows older than the retention
#: window). Not the Unix epoch itself, which the store refuses to read as a
#: ``DateTime64`` parameter under ``EXECUTE AS``.
FIRST_CURSOR = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)

_PARAMETER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*):([^{}]+)\}")


class QueryError(ValueError):
    """A query name or its parameters do not fit the contract."""


def load(name: str) -> str:
    """The text of query *name* (a stem of :data:`QUERIES`)."""
    if name not in QUERIES:
        raise QueryError(f"no history query {name!r}; the queries are {', '.join(QUERIES)}")
    return resources.files(__name__).joinpath(f"{name}.sql").read_text(encoding="utf-8")


def parameters(name: str) -> dict[str, str]:
    """The parameters query *name* takes: name -> the store type it is sent as."""
    found: dict[str, str] = {}
    for param, kind in _PARAMETER.findall(load(name)):
        if found.setdefault(param, kind) != kind:
            raise QueryError(f"{name}.sql uses {{{param}}} as both {found[param]} and {kind}")
    return found


def _timestamp(value: object) -> str:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.UTC).replace(tzinfo=None)
        return value.strftime("%Y-%m-%d %H:%M:%S.%f") + "000"
    return str(value)


def _wire(kind: str, value: Any) -> str:
    """*value* as the store reads a ``param_<name>`` of type *kind*."""
    if kind.startswith("Array("):
        if not isinstance(value, list | tuple):
            raise QueryError(f"an {kind} parameter takes a list, got {value!r}")
        return "[" + ", ".join(_literal("String", str(item)) for item in value) + "]"
    if kind.startswith("DateTime"):
        return _timestamp(value)
    return str(value)


def query_parameters(name: str, values: Mapping[str, Any]) -> dict[str, str]:
    """The ``param_<name>`` pairs that send *values* with query *name*.

    Raises:
        QueryError: A parameter the query takes is missing, or one it does not
            take is given.
    """
    wanted = parameters(name)
    missing = sorted(set(wanted) - set(values))
    extra = sorted(set(values) - set(wanted))
    if missing or extra:
        raise QueryError(
            f"{name}.sql takes {', '.join(sorted(wanted)) or 'no parameters'}"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; not taken {', '.join(extra)}" if extra else "")
        )
    return {f"param_{key}": _wire(kind, values[key]) for key, kind in wanted.items()}


def _literal(kind: str, value: Any) -> str:
    if kind.startswith("Array("):
        return _wire(kind, value)
    if kind.startswith(("UInt", "Int")):
        return str(int(value))
    text = _wire(kind, value)
    escaped = "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if kind.startswith("DateTime64"):
        precision = kind[len("DateTime64(") : -1] or "3"
        return f"toDateTime64({escaped}, {precision}, 'UTC')"
    return escaped


def render(name: str, values: Mapping[str, Any]) -> str:
    """Query *name* with *values* written in as literals, for reading and testing.

    The page never sends this form: it sends :func:`load` with
    :func:`query_parameters`, so the store types every value itself.
    """
    query_parameters(name, values)
    return _PARAMETER.sub(lambda m: _literal(m.group(2), values[m.group(1)]), load(name))
