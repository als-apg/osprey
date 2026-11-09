"""HTTP/IPC utilities for MCP tool code."""

import contextlib
import logging
import os
import socket
import struct
import sys
import time

logger = logging.getLogger("osprey.mcp_server.http")

#: Activity kind that means "this touched the control system" — the kind
#: ``channel_write``, ``execute`` and ``execute_file`` already emit. It is the
#: one kind whose events are stamped with the session's control-system target,
#: because it is the one kind for which "which machine?" is a real question.
CONTROL_ACTIVITY_KIND = "channel"

#: Tool name reported for a control-system target switch. The switch tool emits
#: through :func:`notify_target_switch_async` rather than spelling this itself.
TARGET_SWITCH_TOOL = "control_target_set"

#: Activity kind a switch is reported under. A switch changes what the session
#: is configured to talk to, so it rides the ``config`` kind the route already
#: accepts — a new kind would be rejected with a 422 by the web terminal and the
#: event would vanish.
TARGET_SWITCH_KIND = "config"

#: The two switch outcomes. Spelled here so the success and failure emits of the
#: switch tool cannot drift into two vocabularies.
SWITCH_OUTCOME_SUCCESS = "success"
SWITCH_OUTCOME_FAILURE = "failure"

#: Sentinel handed to ``resolve_control_target`` as its "no answer" value. That
#: reader returns the baseline it is given whenever the record is absent,
#: unreadable or corrupt; passing a value no target can equal is how this module
#: tells "nothing recorded" apart from "recorded as live", without restating the
#: degradation rule that reader owns.
_NO_TARGET = ""


def gallery_url() -> str:
    """Build the gallery base URL from config.

    Resolved through the framework's shared host/port derivation, so the port
    follows ``OSPREY_ARTIFACT_SERVER_PORT`` when the deployment sets it (the
    multi-user compose render exports one per user).
    """
    from osprey.registry.web import resolve_web_server_base_url

    return resolve_web_server_base_url("artifact")


def web_terminal_url() -> str:
    """Build the web terminal base URL from config.

    In containerized deployments the actual port is set via OSPREY_WEB_PORT
    (docker-compose env), which may differ from the default in config.yml.
    The env var takes precedence when present.

    With neither the env var nor ``web_terminal.port`` set, the port is the
    ``web`` slot at *this config's own* ``deployment.port_base`` — the port the
    terminal binds when nothing overrides it. Derived rather than fixed: on a
    host running two deployments, a frozen number would address the other one's
    terminal.
    """
    import os

    from osprey.port_layout import default_port, resolve_port_base
    from osprey_connectors.workspace import load_osprey_config

    config = load_osprey_config()
    wt = config.get("web_terminal", {})
    host = wt.get("host", "127.0.0.1")
    fallback = wt.get("port", default_port("web", 0, base=resolve_port_base(config)))
    port = int(os.environ.get("OSPREY_WEB_PORT", fallback))
    return f"http://{host}:{port}"


def phoebus_bridge_default(config: dict) -> str:
    """The bridge URL ``phoebus.host``/``phoebus.port`` configure.

    Stock ``http://127.0.0.1:7979`` when the config carries neither key
    (matches ``bridge_preferences.properties``). This is the ONE spelling of
    the config half of :func:`phoebus_bridge_url`, shared with the render
    (``config_derived_context``'s ``phoebus_bridge_default`` context key) so
    the ``.mcp.json`` fallback the build emits and the resolution the running
    server performs cannot fork — a fallback rendered from anything else pins
    the client to a port the backend does not serve (#829).
    """
    ph = config.get("phoebus", {}) or {}
    host = ph.get("host", "127.0.0.1")
    port = ph.get("port", 7979)
    return f"http://{host}:{port}"


def phoebus_bridge_url() -> str:
    """Build the Phoebus agent-bridge base URL from env or config.

    The bridge is the in-JVM HTTP server embedded in a running Phoebus product
    (default ``http://127.0.0.1:7979``). Resolution order:

    1. ``PHOEBUS_BRIDGE_URL`` env var (full URL) — set by the framework server
       definition; wins outright. The definition's rendered fallback is
       :func:`phoebus_bridge_default`, so on a rendered project this step
       already answers with the configured host/port.
    2. ``phoebus.host`` / ``phoebus.port`` in config.yml, each defaulting to
       ``127.0.0.1`` / ``7979`` (:func:`phoebus_bridge_default`) — what a
       server launched outside a rendered entry resolves to.
    """
    import os

    from osprey_connectors.workspace import load_osprey_config

    full = os.environ.get("PHOEBUS_BRIDGE_URL")
    if full:
        return full.rstrip("/")

    return phoebus_bridge_default(load_osprey_config())


_PANEL_TOKEN_LATCH: str | None = None


def reset_panel_token_latch() -> None:
    """Forget the last panel token :func:`_panel_auth_headers` saw.

    The latch is process state: one caller's token is otherwise sent by the next.
    """
    global _PANEL_TOKEN_LATCH
    _PANEL_TOKEN_LATCH = None


def _panel_auth_headers() -> dict[str, str]:
    """Build the ``Authorization`` header the terminal API's panel routes want.

    The MCP server is an agent-reachable process, so it carries the *panel
    token* — a low-privilege credential that unlocks only the panel-coordination
    routes — and never the operator secret that authorises a browser session.

    The value is read from ``OSPREY_PANEL_TOKEN`` on every call rather than
    captured at import: the carrier is published into the environment by the
    launching process, which may happen after this module is first imported.

    The last non-blank value seen is also latched, because this process can
    lose the carrier from under itself: saving an artifact auto-launches the
    artifact companion app (``osprey.stores.artifact_store`` ->
    ``ensure_web_server("artifact")``), and when that happens in-process the app's
    construction closes both credential carriers in ``os.environ`` — the
    same scrub every interface app performs so a child it spawns cannot
    inherit them.  Without the latch every panel call after that point would
    go out with no bearer and be refused.  The environment is still consulted
    first, so a value published after import (or rotated) wins over the latch.
    When neither the carrier nor the latch holds a value, the credentials this
    process already holds are consulted last: a gallery auto-launched
    in-thread was built from that same holder and verifies exactly its panel
    token, so it is the right bearer by construction — and without it the
    process would be refused by its own gallery.  The holder is only *peeked*
    (:func:`~osprey.interfaces.web_auth.peek_web_credentials`), never
    populated: in a process that never held a carrier,
    ``get_web_credentials()`` would *mint* a token the terminal does not
    recognise.

    A process behind the multi-user proxy holds no operator secret, so building
    an interface app in it refuses, and that refusal consumes the carrier before
    anything latched it — the holder stays empty too.  The token it was handed
    is still the right bearer, and
    :func:`~osprey.interfaces.web_auth.peek_supplied_panel_token` is where it is
    read, last.

    Returns:
        ``{"Authorization": "Bearer <token>"}`` when the carrier, the latch,
        this process's own credential holder, or the token it was handed
        yields a non-blank value, otherwise ``{}``.  Blank counts as
        absent — an uninterpolated compose variable arrives as ``""`` — and a
        bare ``"Bearer "`` would be a credential-shaped lie the route has to
        reject.
    """
    global _PANEL_TOKEN_LATCH
    import os

    from osprey.interfaces.web_auth import (
        PANEL_TOKEN_ENV,
        peek_supplied_panel_token,
        peek_web_credentials,
    )

    token = os.environ.get(PANEL_TOKEN_ENV, "").strip()
    if token:
        _PANEL_TOKEN_LATCH = token
    elif _PANEL_TOKEN_LATCH:
        token = _PANEL_TOKEN_LATCH
    elif (held := peek_web_credentials()) is not None:
        token = held.panel_token
    elif (supplied := peek_supplied_panel_token()) is not None:
        token = supplied
    return {"Authorization": f"Bearer {token}"} if token else {}


def post_json(url: str, payload: dict, *, timeout: int = 3) -> None:
    """Fire-and-forget JSON POST to a local HTTP endpoint.

    Carries the panel token as a bearer header when one is available
    (:func:`_panel_auth_headers`).

    Non-fatal: logs a warning if the target is unreachable *or* rejects the
    call.  That deliberately covers a ``401``: a credential regression makes
    panel notifications go quiet rather than crashing the tool that emitted
    them, and nothing here retries.
    Used by focus tools and rail-visibility notifications.
    """
    import json as _json
    import urllib.request

    try:
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **_panel_auth_headers()},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        logger.warning("POST %s failed (non-fatal): %s", url, exc)


def _post_json_with_response(url: str, payload: dict, *, timeout: int = 3) -> tuple[int, dict]:
    """POST JSON and return ``(status_code, parsed_body)``.

    Unlike :func:`post_json` this propagates connection-level exceptions so the
    caller can distinguish "server rejected" from "server unreachable".

    Carries the panel token as a bearer header when one is available
    (:func:`_panel_auth_headers`).  A ``401`` is a rejection like any other: it
    comes back as ``(401, body)`` for the caller to surface, never raised and
    never retried.

    Args:
        url: Full URL to POST to.
        payload: Dict that will be JSON-serialised as the request body.
        timeout: Socket timeout in seconds.

    Returns:
        A ``(status_code, body_dict)`` tuple.  On an ``HTTPError`` the body is
        parsed from the error response (best-effort; falls back to ``{}``).

    Raises:
        urllib.error.URLError: When the target is unreachable.
        OSError: On other socket-level failures.
    """
    import json as _json
    import urllib.error
    import urllib.request

    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_panel_auth_headers()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body: dict = {}
        with contextlib.suppress(Exception):
            body = _json.loads(exc.read())
        return exc.code, body


def fetch_panels(timeout: int = 3) -> dict | None:
    """Read the Web Terminal's panel inventory from ``GET /api/panels``.

    The read counterpart of the ``notify_*`` helpers, and the one place MCP
    tools go for live panel state: rail membership (``visible``), the active
    panel, the configured presets, the dock's ``layout`` and its ``rev``, and
    ``layout_age_s`` and ``layout_dock``, which say how stale that knowledge is.

    The three layout-report fields carry three distinct states, and a caller
    must not collapse them: all-``null`` means no client has ever reported;
    ``layout: null`` with a numeric age and ``layout_dock: false`` means a
    client is watching but runs without the dock shell and cannot report
    tiles; a tree with ``layout_dock: true`` is a real occupancy report.

    .. warning::
        This performs a **blocking** HTTP call (bounded by ``timeout``).
        Async tools MUST call it via ``await anyio.to_thread.run_sync(...)``
        rather than inline, which would stall the event loop.

    Args:
        timeout: Socket timeout in seconds.

    Returns:
        The parsed JSON payload, or ``None`` when the web terminal is
        unreachable (CLI-only mode), rejected the panel token (``401`` — a
        credential regression reads as "no panel state", not as a crash), or
        answered with something other than a JSON object — callers decide how
        to surface that.  Read individual keys with ``.get()``; servers
        predating the layout-report feature omit the three fields entirely.
    """
    import json as _json
    import urllib.request

    base = web_terminal_url()
    try:
        # Built inside the guard: assembling the credential header is itself
        # fallible, and a read helper whose contract is "None when the panel
        # state cannot be had" must not crash its callers on the way to asking.
        req = urllib.request.Request(f"{base}/api/panels", headers=_panel_auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read())
    except Exception as exc:
        logger.warning(
            "panel fetch failed (web terminal unreachable or rejected the panel token): %s",
            exc,
        )
        return None
    return data if isinstance(data, dict) else None


def fetch_code_review_switch(timeout: float = 2) -> dict | None:
    """Read code review's runtime switch from the web terminal that started this session.

    Only the socket the terminal published (``SWITCH_SOCKET_ENV``) names that terminal, with no
    fallback: a session no terminal started is under no terminal's switch.

    An answer counts only when the kernel says it was served by a process of this account that
    is this process's own ancestor (:func:`_served_by_an_ancestor`). The agent's code runs as
    the same account and can bind a socket at that path whenever the terminal is not on it;
    what it cannot do is become an ancestor of the hook or the executor asking. The check reads pids only, never
    another process's ``/proc`` entries, so a terminal that hides its memory still counts.

    .. warning::
        A **blocking** call bounded by ``timeout`` from the dial to the last byte read,
        however the answer trickles in; async callers run it in a thread.

    Returns:
        The parsed JSON object, or ``None`` when no terminal is named, it cannot be reached,
        is not this session's ancestor, or answers with anything but a JSON object.
    """
    import json as _json

    from osprey.services.code_review.switch import SWITCH_ANSWER_LIMIT, SWITCH_SOCKET_ENV

    path = os.environ.get(SWITCH_SOCKET_ENV, "").strip()
    if not os.path.isabs(path) or not hasattr(socket, "AF_UNIX"):
        return None
    deadline = time.monotonic() + timeout
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(path)
            if not _served_by_an_ancestor(sock):
                logger.warning("code review switch answered by a process outside this session")
                return None
            body = b""
            while chunk := _recv_by(sock, deadline, SWITCH_ANSWER_LIMIT + 1 - len(body)):
                body += chunk
                if len(body) > SWITCH_ANSWER_LIMIT:
                    raise ValueError("the code review switch answer is too long")
        data = _json.loads(body)
    except Exception as exc:
        logger.warning("code review switch read failed: %s", exc)
        return None
    return data if isinstance(data, dict) else None


def _recv_by(sock: socket.socket, deadline: float, nbytes: int) -> bytes:
    """Up to *nbytes* from *sock*, waiting only for what is left until *deadline*.

    A socket timeout bounds each receive alone, so a server dripping one byte at a time
    would keep the reader waiting as long as it liked.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("the code review switch read ran out of time")
    sock.settimeout(remaining)
    return sock.recv(nbytes)


def _served_by_an_ancestor(sock: socket.socket) -> bool:
    """Whether the process serving *sock* runs as this account and is one of our ancestors.

    The terminal spawns every session, so the session's hooks and MCP servers descend
    from it; whatever the agent starts is a descendant of the session, never an ancestor
    of them. Never raises; a peer or process table that cannot be read is no proof.
    """
    try:
        peer = _peer(sock)
        if peer is None or peer[1] != os.getuid():
            return False
        import psutil

        return peer[0] in {p.pid for p in psutil.Process().parents()}
    except Exception:
        logger.warning("could not check who answered the code review switch", exc_info=True)
        return False


def _peer(sock: socket.socket) -> tuple[int, int] | None:
    """The ``(pid, uid)`` the kernel recorded for the listening end of *sock*, or ``None``.

    Linux reads ``SO_PEERCRED`` (``struct ucred``); macOS reads ``LOCAL_PEERPID`` and the uid
    from ``LOCAL_PEERCRED`` (``struct xucred``). Any other platform names no peer.
    """
    if sys.platform == "linux":
        ucred = struct.Struct("3i")
        pid, uid, _gid = ucred.unpack(
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, ucred.size)
        )
        return pid, uid
    if sys.platform == "darwin":
        (pid,) = struct.unpack("i", sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERPID, 4))
        version, uid = struct.unpack_from("2I", sock.getsockopt(_SOL_LOCAL, _LOCAL_PEERCRED, 76))
        return (pid, uid) if version == _XUCRED_VERSION else None
    return None


#: ``SOL_LOCAL``, ``LOCAL_PEERCRED``, ``LOCAL_PEERPID`` and ``XUCRED_VERSION`` from macOS
#: ``<sys/un.h>`` and ``<sys/ucred.h>``; 76 is ``sizeof(struct xucred)``.
_SOL_LOCAL = 0
_LOCAL_PEERCRED = 0x001
_LOCAL_PEERPID = 0x002
_XUCRED_VERSION = 0


def notify_panel_visibility(panel: str, visible: bool) -> None:
    """Fire-and-forget POST to show or hide a panel in the Web Terminal.

    Non-fatal if the web terminal is not
    running (CLI-only mode).  Show/hide is always permitted server-side so
    there is no status to surface.

    Args:
        panel: Panel identifier string.
        visible: ``True`` to show the panel, ``False`` to hide it.
    """
    base = web_terminal_url()
    post_json(
        f"{base}/api/panel-visibility",
        {"panel": panel, "visible": visible, "source": "agent"},
        timeout=2,
    )


def notify_panel_register(
    panel_id: str,
    label: str,
    url: str,
    path: str = "/",
    health_endpoint: str | None = None,
) -> dict:
    """Register a new panel with the Web Terminal server and return the outcome.

    Unlike the fire-and-forget helpers, registration is gated and validated
    server-side (SSRF allowlist, ``web.allow_runtime_panels`` flag), so the
    real HTTP response must reach the caller so the MCP tool can surface it to
    the agent.

    Args:
        panel_id: Unique panel identifier.
        label: Human-readable panel name shown in the UI.
        url: Upstream URL the Web Terminal will proxy/embed.
        path: Sub-path to open inside *url* (default ``"/"``).
        health_endpoint: Optional health-check URL; ``None`` omits the field.

    Returns:
        A dict with the following keys:

        * ``ok`` (bool) — ``True`` on HTTP 200, ``False`` otherwise.
        * ``status`` (int | None) — HTTP status code, or ``None`` when the
          web terminal was unreachable.
        * ``data`` (dict) — Parsed response body on success (``ok=True`` only).
        * ``detail`` (str) — Server ``"detail"`` string on rejection, or
          ``"Web Terminal is not running."`` when unreachable (``ok=False`` only).
    """
    base = web_terminal_url()
    payload: dict = {
        "id": panel_id,
        "label": label,
        "url": url,
        "path": path,
        "health_endpoint": health_endpoint,
        "source": "agent",
    }
    try:
        status, body = _post_json_with_response(f"{base}/api/panels/register", payload, timeout=5)
    except Exception as exc:
        logger.warning("panel register POST failed (web terminal unreachable): %s", exc)
        return {"ok": False, "status": None, "detail": "Web Terminal is not running."}

    if status == 200:
        return {"ok": True, "status": 200, "data": body}
    detail = body.get("detail", "") if isinstance(body, dict) else str(body)
    return {"ok": False, "status": status, "detail": detail}


#: The write route of each dock op on the web terminal.
PANEL_OP_ROUTES: dict[str, str] = {
    "drop": "/api/panel-op/drop",
    "close": "/api/panel-op/close",
    "open": "/api/panel-op/open",
    "open_set": "/api/panel-op/open-set",
    "open_arrangement": "/api/panel-op/open-arrangement",
}

#: Socket timeout of :func:`panel_op`. Longer than the route's own 5 s wait for
#: the dock client, so a silent dock comes back as the route's named
#: ``no_dock_answer`` (outcome unknown) rather than as a dropped connection.
PANEL_OP_TIMEOUT_S = 8

#: Reason :func:`panel_op` reports when the web terminal cannot be reached.
NO_WEB_TERMINAL = "no_web_terminal"


def panel_op(op: str, args: dict, *, base_rev: int | None = None) -> dict:
    """Run one dock op through the web terminal and return its outcome.

    The one agent dock-write path: every MCP caller that opens, moves or closes
    panels (workspace tools, ARIEL, Phoebus) comes through here. The helper
    mints the ``request_id``; the route relays the op to the current dock client
    and answers once that client has applied (or refused) it.

    .. warning::
        This performs a **blocking** HTTP call of up to
        :data:`PANEL_OP_TIMEOUT_S` seconds. Async tools MUST call it via
        ``await anyio.to_thread.run_sync(...)``.

    Args:
        op: One of :data:`PANEL_OP_ROUTES` (``drop``, ``close``, ``open``,
            ``open_set``, ``open_arrangement``).
        args: The op's arguments, e.g. ``{"member": "ariel", "url": u}``
            for ``open``; ``open_set`` and ``open_arrangement`` spell ``as`` as is.
        base_rev: The ``rev`` of the agent's last read, for ``drop``, the one op
            that names tile ids; omitted from the body when ``None``.

    Returns:
        ``{"ok": True, "status": 200, "layout": {...}, "popouts": [...], "rev":
        <int>}`` when the dock applied the op (an ``open`` adds ``placed`` and,
        when it replaced a panel, ``replaced`` and possibly ``note``), else ``{"ok": False, "status": <int|None>,
        "reason": <str>, "detail": <str>}`` (plus ``rev`` when the server gave
        one). ``reason`` is the route's named refusal (``stale_base``,
        ``no_dock_client``, ``terminal_not_agent_movable``, ...) or
        ``no_dock_answer`` (the dock did not answer in time and may still apply
        the op), ``invalid_request`` for a body the route could not parse, or
        :data:`NO_WEB_TERMINAL` when the terminal is unreachable.

    Raises:
        ValueError: For an ``op`` that is not a dock op.
    """
    import uuid

    route = PANEL_OP_ROUTES.get(op)
    if route is None:
        raise ValueError(f"Unknown panel op: {op!r}")
    payload: dict = {**args, "request_id": uuid.uuid4().hex}
    if base_rev is not None:
        payload["base_rev"] = base_rev
    try:
        status, body = _post_json_with_response(
            f"{web_terminal_url()}{route}", payload, timeout=PANEL_OP_TIMEOUT_S
        )
    except Exception as exc:
        logger.warning("panel op %s failed (web terminal unreachable): %s", op, exc)
        return {
            "ok": False,
            "status": None,
            "reason": NO_WEB_TERMINAL,
            "detail": "Web Terminal is not running.",
        }
    body = body if isinstance(body, dict) else {}
    if status == 200:
        return {
            "ok": True,
            "status": 200,
            "layout": body.get("layout"),
            "popouts": body.get("popouts", []),
            "rev": body.get("rev"),
            # An open says how it placed the panel (placed / replaced / note).
            **{key: body[key] for key in ("placed", "replaced", "note") if key in body},
        }
    out: dict = {"ok": False, "status": status}
    if body.get("status") == "refused":
        out["reason"] = str(body.get("reason", ""))
        out["detail"] = str(body.get("detail", ""))
    else:
        detail = body.get("detail", "")
        out["reason"] = "invalid_request"
        out["detail"] = detail if isinstance(detail, str) else str(detail)
    if "rev" in body:
        out["rev"] = body["rev"]
    return out


def resolve_activity_target() -> str | None:
    """The control-system target to name on an activity event, or ``None``.

    Answers from the deployment's control-context record, through
    :func:`~osprey.mcp_server.control_system.target_banner.resolve_control_target`
    — the same reader the approval prompt, the Phoebus guard and the health row
    use, so no two surfaces can describe one deployment differently. That it is
    one record read the same way everywhere is what makes the reader
    holder-agnostic: this call is made from whichever MCP server process emitted
    the activity (the controls server for ``channel_write``, the python executor
    for ``execute``), and every one of them reads the same file.

    Returns ``None``, and never the deployment baseline, when nothing is
    published. The baseline is what the config selects, not where a session
    stands: stamping it on an activity event could tell an operator a simulated
    write touched the real machine, or the reverse. An unpublished target is an
    absent claim, not a baseline one.

    Never raises: a failure to answer degrades to an unstamped event, the same
    fail-closed direction every other reader of that record takes.
    """
    try:
        from osprey.mcp_server.control_system.target_banner import resolve_control_target

        target = resolve_control_target(_NO_TARGET)
    except Exception as exc:
        # Reachable: the resolver does not guard its record reader, so a reader
        # that raises lands here and the event goes out unstamped.
        logger.debug("Could not resolve the session control-system target: %s", exc)
        return None
    return target or None


def _stamp_target(detail: str | None, target: str | None) -> str | None:
    """Prefix *detail* with *target*, or return it unchanged.

    The prefix goes at the FRONT and before the route's length bound is applied,
    so a bulk write long enough to be truncated still names the machine it
    landed on. ``detail`` is the only field of the activity frame the web
    terminal carries through to the browser, which is why the target rides in it
    rather than in a key of its own (see :func:`notify_agent_activity`).
    """
    if target is None:
        return detail
    return f"[{target}] {detail}" if detail else f"[{target}]"


def build_target_switch_detail(
    from_target: str,
    to_target: str,
    outcome: str,
    generation: int | None = None,
    reason: str | None = None,
) -> str:
    """Render the one-line body of a target-switch activity event.

    Shape::

        live → va · success (generation 3)
        live → va · failure: probe channel never connected
        va → live · failure

    Both targets are always named — the operator needs to know what the session
    was on as much as what it moved to, and on a failure the first of those is
    still where it is. ``generation`` and ``reason`` are omitted when absent
    rather than rendered as ``None``.
    """
    line = f"{from_target} → {to_target} · {outcome}"
    if reason:
        line = f"{line}: {reason}"
    if generation is not None:
        line = f"{line} (generation {generation})"
    return line


def notify_target_switch(
    *,
    from_target: str,
    to_target: str,
    outcome: str,
    generation: int | None = None,
    reason: str | None = None,
) -> None:
    """Fire-and-forget POST reporting a control-system target switch.

    Emitted on **both** outcomes. A switch that succeeds moves every subsequent
    tool call to another machine, and a switch that fails leaves the session on
    the machine it started from — an operator who approved the attempt has to be
    able to see which of those happened without asking.

    Args:
        from_target: Target the session was on (``live`` / ``va``).
        to_target: Target the switch was asked for.
        outcome: :data:`SWITCH_OUTCOME_SUCCESS` or :data:`SWITCH_OUTCOME_FAILURE`.
        generation: Switch generation the deployment is on after the attempt,
            when
            the caller knows it. Omitted from the line when ``None``.
        reason: Short, operator-readable cause of a failure. Omitted when
            ``None`` — an unexplained failure is still reported.

    Never raises; blocking, so coroutines call :func:`notify_target_switch_async`.
    """
    notify_agent_activity(
        TARGET_SWITCH_TOOL,
        TARGET_SWITCH_KIND,
        detail=build_target_switch_detail(from_target, to_target, outcome, generation, reason),
    )


async def notify_target_switch_async(
    *,
    from_target: str,
    to_target: str,
    outcome: str,
    generation: int | None = None,
    reason: str | None = None,
) -> None:
    """Awaitable form of :func:`notify_target_switch` — the one the switch tool
    calls. The blocking POST runs off the loop.

    The emit itself is fire-and-forget and swallows every failure. The one thing
    outside that guarantee is the deferred ``anyio`` import on the first call,
    which is deferred to match the rest of this module and cannot fail in the
    MCP-server process this runs in (anyio is a hard dependency of the server).
    """
    import functools

    import anyio

    await anyio.to_thread.run_sync(
        functools.partial(
            notify_target_switch,
            from_target=from_target,
            to_target=to_target,
            outcome=outcome,
            generation=generation,
            reason=reason,
        )
    )


def notify_agent_activity(
    tool: str,
    kind: str,
    panel: str | None = None,
    detail: str | None = None,
) -> None:
    """Fire-and-forget POST reporting agent tool activity to the Web Terminal.

    Posts ``{"tool": tool, "target": {"kind": kind, "panel"?: ..., "detail"?: ...}}``
    to ``/api/agent-activity`` so the UI can highlight what the agent is
    touching.  ``panel``/``detail`` are omitted from the body when ``None``.
    Non-fatal if the web terminal is not running (CLI-only mode): all
    exceptions are swallowed and this function never raises.

    .. warning::
        This function performs a **blocking** HTTP call (bounded at 1s).
        Async tools MUST call it via ``await anyio.to_thread.run_sync(...)``
        — never inline in a coroutine.  Calling a sync ``post_json``-style
        helper directly from async code stalls the event loop (a known
        foot-gun with the existing sync-post-in-async pattern; do not copy it).

    Args:
        tool: Name of the tool the agent invoked.
        kind: Activity target kind (validated server-side; unknown kinds 422).
        panel: Optional panel identifier the activity targets.
        detail: Optional free-form detail (e.g. channel name, file path).
            Truncated to the route's 1024-char bound so an unbounded caller
            (e.g. a bulk channel write) cannot turn the emit into a silent 422.

    Control-system activity (``kind`` :data:`CONTROL_ACTIVITY_KIND`) is stamped
    with the recorded control target here, in the one place every emit site passes
    through, so ``channel_write``, ``execute`` and ``execute_file`` report the
    same machine without each resolving it — three resolutions is how three
    surfaces start disagreeing. The stamp rides in ``detail`` because that is
    the only free field the ``/api/agent-activity`` route carries through to the
    browser: its request model names ``kind``/``panel``/``detail`` and drops
    anything else, so a structured ``target`` key would be silently discarded
    server-side and never reach an operator's screen.
    """
    try:
        target: dict = {"kind": kind}
        if panel is not None:
            target["panel"] = panel
        if kind == CONTROL_ACTIVITY_KIND:
            detail = _stamp_target(detail, resolve_activity_target())
        if detail is not None:
            if len(detail) > 1024:
                detail = detail[:1023] + "…"
            target["detail"] = detail
        base = web_terminal_url()
        post_json(f"{base}/api/agent-activity", {"tool": tool, "target": target}, timeout=1)
    except Exception as exc:
        logger.warning("agent-activity notify failed (non-fatal): %s", exc)


async def notify_agent_activity_async(
    tool: str,
    kind: str,
    panel: str | None = None,
    detail: str | None = None,
) -> None:
    """Awaitable form of :func:`notify_agent_activity` for coroutine emit sites.

    The single thread-hop for async tools: the blocking (bounded ~1s) POST runs
    on a worker thread so the event loop is never stalled. Same fire-and-forget
    contract — never raises.
    """
    import functools

    import anyio

    await anyio.to_thread.run_sync(
        functools.partial(notify_agent_activity, tool, kind, panel=panel, detail=detail)
    )
