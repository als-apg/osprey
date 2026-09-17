"""The wire contract of the model RPC, shared by the server and its clients.

The Virtual Accelerator answers questions about its physics model -- and
takes the few writes the model surface allows -- over one PVAccess RPC
channel, :data:`RPC_PV`. This module is everything the two ends agree on: the
verbs, the request shape, the reply shape and the error strings a client may
be shown. It imports p4p and nothing from the serving runtime or the model,
so the server's handler and a client that has neither installed import one
definition and cannot drift apart.

**Request.** An NTURI on :data:`RPC_PV` whose query carries four fields:

``verb``    one of :data:`VERBS`.
``names``   the variable names the call is about.
``values``  a JSON object mapping variable name to a finite number -- the
            writes a ``set`` carries -- or the empty string for none.
``token``   the credential the server checks before a write.

:func:`build_request` always sends all four. A generic PVA client sends only
the fields it is given, so :func:`parse_request` reads an omitted ``names``,
``values`` or ``token`` as empty; a field that *is* present must carry its
wire type, and ``verb`` is required.

**Reply.** An NTScalar string whose value is a JSON document, either
``{"ok": true, "result": ...}`` or ``{"ok": false, "error": "..."}``. The
error text is written for a person, and a client surfaces it unchanged.
Results are not held to the finiteness rule requests are: they report what
the model holds, and a non-finite number travels as the ``NaN``/``Infinity``
token Python's :mod:`json` reads back.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from p4p import Value
from p4p.nt import NTURI, NTScalar

RPC_PV = "model_rpc"
"""The PVAccess channel the model RPC is served on."""

VERBS = ("info", "get", "diff", "status", "set", "reset")
"""Every verb the server answers; anything else is refused before dispatch."""

RPC_TIMEOUT_S = 30.0
"""How long the server waits for the run loop before answering :data:`ERR_TIMEOUT`."""

ERR_NOT_READY = "the server is still starting"
"""The refusal for a call that arrives before the server can run a job.

The serving runner's ``NOT_READY`` is this string: a PVA put in the same
window is refused with it too."""

ERR_TIMEOUT = f"the model did not answer within {RPC_TIMEOUT_S:g} s"
"""The reply to a call the run loop did not complete within :data:`RPC_TIMEOUT_S`."""

QUERY_FIELDS = (("verb", "s"), ("names", "as"), ("values", "s"), ("token", "s"))
REQUEST_TYPE = NTURI(list(QUERY_FIELDS))
REPLY_TYPE = NTScalar("s")


class ModelRpcError(Exception):
    """A request or reply that breaks the contract, or a refusal from the server.

    ``str(error)`` is the message, fit to show a person as-is.
    """


@dataclass(frozen=True)
class RpcRequest:
    """A request that satisfies the contract, as the server dispatches it."""

    verb: str
    names: tuple[str, ...] = ()
    values: Mapping[str, float] = field(default_factory=dict)
    token: str = ""


def build_request(
    verb: str,
    *,
    names: Iterable[str] = (),
    values: Mapping[str, float] | None = None,
    token: str = "",
) -> Value:
    """The NTURI request for ``verb``, checked against the contract before it is sent.

    Raises :class:`ModelRpcError` for anything :func:`parse_request` would
    refuse, so a malformed call fails at the caller rather than round-tripping.
    """
    checked_verb = _check_verb(verb)
    checked_names = _check_names(names)
    encoded = "" if values is None else json.dumps(_check_values(values), allow_nan=False)
    if not isinstance(token, str):
        raise ModelRpcError("the token must be a string")
    return REQUEST_TYPE.wrap(
        RPC_PV,
        kws={"verb": checked_verb, "names": list(checked_names), "values": encoded, "token": token},
    )


def parse_request(request: Value) -> RpcRequest:
    """The :class:`RpcRequest` an NTURI carries; :class:`ModelRpcError` if it breaks the contract."""
    query = request.get("query")
    if query is None:
        raise ModelRpcError("the request carries no query; expected an NTURI with a verb")
    verb = query.get("verb")
    if verb is None:
        raise ModelRpcError("the request names no verb")
    names = query.get("names")
    raw_values = query.get("values")
    token = query.get("token")
    if token is not None and not isinstance(token, str):
        raise ModelRpcError("the token must be a string")
    return RpcRequest(
        verb=_check_verb(verb),
        names=() if names is None else _check_names(names),
        values=_decode_values(raw_values),
        token=token or "",
    )


def ok_reply(result: Any) -> Value:
    """The reply carrying ``result``, which must be JSON-serializable.

    Array-likes (anything with a ``tolist()``, such as numpy arrays and
    scalars) are sent as their list or number.
    """
    return REPLY_TYPE.wrap(json.dumps({"ok": True, "result": result}, default=_tolist))


def error_reply(message: str) -> Value:
    """The reply refusing a call with ``message``."""
    return REPLY_TYPE.wrap(json.dumps({"ok": False, "error": str(message)}))


def parse_reply(reply: Value | str) -> Any:
    """The result a reply carries; :class:`ModelRpcError` with the server's message on a refusal.

    Accepts the NTScalar itself or the string a p4p client context unwraps it
    to.
    """
    text = reply if isinstance(reply, str) else reply.get("value")
    if not isinstance(text, str):
        raise ModelRpcError("the reply carries no JSON document")
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise ModelRpcError(f"the reply is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("ok"), bool):
        raise ModelRpcError("the reply is not a model RPC reply document")
    if document["ok"]:
        return document.get("result")
    error = document.get("error")
    if not isinstance(error, str):
        raise ModelRpcError("the reply refuses the call without an error message")
    raise ModelRpcError(error)


def _check_verb(verb: object) -> str:
    if not isinstance(verb, str) or verb not in VERBS:
        raise ModelRpcError(f"unknown verb {verb!r}; expected one of: {', '.join(VERBS)}")
    return verb


def _check_names(names: object) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise ModelRpcError("names must be a list of variable names, not one string")
    if not isinstance(names, Iterable):
        raise ModelRpcError("names must be a list of variable names")
    checked: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise ModelRpcError(f"names must be strings; got {name!r}")
        checked.append(name)
    return tuple(checked)


def _decode_values(raw: object) -> dict[str, float]:
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise ModelRpcError("values must be a JSON string")
    try:
        decoded = json.loads(raw)
    except ValueError as exc:
        raise ModelRpcError(f"values is not valid JSON: {exc}") from exc
    return _check_values(decoded)


def _check_values(values: object) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ModelRpcError("values must be a JSON object mapping variable name to a number")
    checked: dict[str, float] = {}
    for name, raw in values.items():
        if not isinstance(name, str):
            raise ModelRpcError(f"value names must be strings; got {name!r}")
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise ModelRpcError(f"the value for {name!r} is not a number: {raw!r}")
        try:
            number = float(raw)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):
            raise ModelRpcError(f"the value for {name!r} is not finite: {raw!r}")
        checked[name] = number
    return checked


def _tolist(obj: object) -> Any:
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


__all__ = [
    "ERR_NOT_READY",
    "ERR_TIMEOUT",
    "QUERY_FIELDS",
    "REPLY_TYPE",
    "REQUEST_TYPE",
    "RPC_PV",
    "RPC_TIMEOUT_S",
    "VERBS",
    "ModelRpcError",
    "RpcRequest",
    "build_request",
    "error_reply",
    "ok_reply",
    "parse_reply",
    "parse_request",
]
