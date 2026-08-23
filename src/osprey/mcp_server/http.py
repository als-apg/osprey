"""HTTP/IPC utilities for MCP tool code."""

import logging

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

#: Sentinel handed to ``resolve_session_target`` as its "no answer" value. That
#: resolver returns the baseline it is given whenever the state is absent,
#: foreign-owned, ambiguous or corrupt; passing a value no target can equal is
#: how this module tells "nothing published" apart from "published as live",
#: without restating the ownership rule that resolver owns.
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
    """
    import os

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    wt = config.get("web_terminal", {})
    host = wt.get("host", "127.0.0.1")
    port = int(os.environ.get("OSPREY_WEB_PORT", wt.get("port", 8087)))
    return f"http://{host}:{port}"


def phoebus_bridge_url() -> str:
    """Build the Phoebus agent-bridge base URL from env or config.

    The bridge is the in-JVM HTTP server embedded in a running Phoebus product
    (default ``http://127.0.0.1:7979``). Resolution order:

    1. ``PHOEBUS_BRIDGE_URL`` env var (full URL) — set by the framework server
       definition; wins outright.
    2. ``PHOEBUS_BRIDGE_PORT`` env var overrides only the port.
    3. ``phoebus.host`` / ``phoebus.port`` in config.yml.
    4. ``127.0.0.1:7979`` default (matches ``bridge_preferences.properties``).
    """
    import os

    from osprey.utils.workspace import load_osprey_config

    full = os.environ.get("PHOEBUS_BRIDGE_URL")
    if full:
        return full.rstrip("/")

    config = load_osprey_config()
    ph = config.get("phoebus", {})
    host = ph.get("host", "127.0.0.1")
    port = int(os.environ.get("PHOEBUS_BRIDGE_PORT", ph.get("port", 7979)))
    return f"http://{host}:{port}"


def post_json(url: str, payload: dict, *, timeout: int = 3) -> None:
    """Fire-and-forget JSON POST to a local HTTP endpoint.

    Non-fatal: logs a warning if the target is unreachable.
    Used by focus tools and panel-focus notifications.
    """
    import json as _json
    import urllib.request

    try:
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as exc:
        logger.warning("POST %s failed (non-fatal): %s", url, exc)


def _post_json_with_response(url: str, payload: dict, *, timeout: int = 3) -> tuple[int, dict]:
    """POST JSON and return ``(status_code, parsed_body)``.

    Unlike :func:`post_json` this propagates connection-level exceptions so the
    caller can distinguish "server rejected" from "server unreachable".

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
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body: dict = {}
        try:
            body = _json.loads(exc.read())
        except Exception:
            pass
        return exc.code, body


def fetch_panels(timeout: int = 3) -> dict | None:
    """Read the Web Terminal's panel inventory from ``GET /api/panels``.

    The read counterpart of the ``notify_*`` helpers, and the one place MCP
    tools go for live panel state: rail membership (``visible``), the active
    panel, the configured presets, and the layout-report fields ``open_tiles``,
    ``open_tiles_age_s`` and ``open_tiles_dock`` that say what is actually on
    screen and how stale that knowledge is.

    The three layout-report fields carry three distinct states, and a caller
    must not collapse them: all-``null`` means no client has ever reported;
    ``open_tiles: null`` with a numeric age and ``open_tiles_dock: false``
    means a client is watching but runs without the dock shell and cannot
    report tile order; a list (including ``[]``, a genuinely empty workspace)
    with ``open_tiles_dock: true`` is a real occupancy report.

    .. warning::
        This performs a **blocking** HTTP call (bounded by ``timeout``).
        Async tools MUST call it via ``await anyio.to_thread.run_sync(...)``
        rather than inline, which would stall the event loop.

    Args:
        timeout: Socket timeout in seconds.

    Returns:
        The parsed JSON payload, or ``None`` when the web terminal is
        unreachable (CLI-only mode) or answered with something other than a
        JSON object — callers decide how to surface that.  Read individual
        keys with ``.get()``; servers predating the layout-report feature omit
        the three fields entirely.
    """
    import json as _json
    import urllib.request

    base = web_terminal_url()
    try:
        with urllib.request.urlopen(f"{base}/api/panels", timeout=timeout) as resp:
            data = _json.loads(resp.read())
    except Exception as exc:
        logger.warning("panel fetch failed (web terminal unreachable): %s", exc)
        return None
    return data if isinstance(data, dict) else None


def notify_panel_visibility(panel: str, visible: bool) -> None:
    """Fire-and-forget POST to show or hide a panel in the Web Terminal.

    Mirrors :func:`notify_panel_focus`.  Non-fatal if the web terminal is not
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


def notify_panel_close(panel: str) -> None:
    """Fire-and-forget POST to close a panel's tile in the Web Terminal.

    The on-screen half of the panel vocabulary, and deliberately separate from
    :func:`notify_panel_visibility`: closing a tile leaves rail membership
    alone, so the operator can reopen the panel in one click. Non-fatal if the
    web terminal is not running (CLI-only mode).

    Args:
        panel: Panel identifier string.
    """
    base = web_terminal_url()
    post_json(
        f"{base}/api/panel-close",
        {"panel": panel, "source": "agent"},
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


def notify_panel_arrange(
    tiles: list[str] | None = None,
    preset: str | None = None,
    focus: str | None = None,
) -> dict:
    """Request a workspace tile arrangement and return the server's verdict.

    Like :func:`notify_panel_register`, and unlike the fire-and-forget
    ``notify_*`` helpers, this captures the real HTTP response: the arrange
    route validates the tile ids, the preset name and the focus target
    server-side, so a rejection carries a detail string the calling tool must
    show the agent rather than swallow.  ``source: "agent"`` is always sent so
    the UI can flash the agent glow on the affected rail entries.

    Exactly one of ``tiles`` and ``preset`` is expected.  Supplying both or
    neither is left to the route to reject, so the agent sees one authoritative
    validation message instead of two divergent ones.

    Args:
        tiles: Panel ids to leave open, in left-to-right order.
        preset: Name of a configured layout (``web.presets``) to apply instead
            of an explicit tile list.
        focus: Optional panel to focus; the route requires it to be one of the
            resulting tiles.  Omitted from the body when ``None``.

    Returns:
        A dict with the following keys:

        * ``ok`` (bool) — ``True`` on HTTP 200, ``False`` otherwise.
        * ``status`` (int | None) — HTTP status code, or ``None`` when the web
          terminal was unreachable.
        * ``data`` (dict) — Parsed response body on success (``ok=True`` only):
          the applied ``tiles``, ``focus``, ``preset`` and ``prune_rail``.
        * ``detail`` (str) — Rejection text (``ok=False`` only): the server's
          ``"detail"``, or ``"Web Terminal is not running."`` when unreachable.
          Always text — FastAPI's own body-validation 422s carry a list of
          error dicts, which is stringified here.
    """
    base = web_terminal_url()
    payload: dict = {"source": "agent"}
    if tiles is not None:
        payload["tiles"] = tiles
    if preset is not None:
        payload["preset"] = preset
    if focus is not None:
        payload["focus"] = focus

    try:
        status, body = _post_json_with_response(f"{base}/api/panel-arrange", payload, timeout=5)
    except Exception as exc:
        logger.warning("panel arrange POST failed (web terminal unreachable): %s", exc)
        return {"ok": False, "status": None, "detail": "Web Terminal is not running."}

    if status == 200:
        return {"ok": True, "status": 200, "data": body}
    detail = body.get("detail", "") if isinstance(body, dict) else body
    return {
        "ok": False,
        "status": status,
        "detail": detail if isinstance(detail, str) else str(detail),
    }


def notify_panel_focus(panel_id: str, url: str | None = None) -> None:
    """Fire-and-forget POST to switch the Web Terminal's active panel.

    Non-fatal if the web terminal is not running (CLI-only mode).
    """
    base = web_terminal_url()
    payload: dict = {"panel": panel_id, "source": "agent"}
    if url is not None:
        payload["url"] = url
    post_json(f"{base}/api/panel-focus", payload, timeout=2)


def resolve_activity_target() -> str | None:
    """The control-system target to name on an activity event, or ``None``.

    Answers from the target-state file a controls server publishes, through
    :func:`~osprey.mcp_server.control_system.target_banner.resolve_session_target`
    — the same resolver the approval prompt, the Phoebus guard and the health
    row use, so no two surfaces can describe one session differently. The
    resolver is holder-agnostic on purpose: this call is made from whichever MCP
    server process emitted the activity (the controls server for
    ``channel_write``, the python executor for ``execute``), and the record it
    matches is the one owned by the same Claude Code parent.

    Returns ``None``, and never the deployment baseline, when nothing is
    published. The baseline of a ``mock`` deployment resolves to ``live``;
    stamping that on an activity event would tell an operator a simulated write
    touched the real machine. An unpublished target is an absent claim, not a
    ``live`` one.

    Never raises: a failure to answer degrades to an unstamped event, the same
    fail-closed direction every other reader of that state file takes.
    """
    try:
        from osprey.mcp_server.control_system.target_banner import resolve_session_target

        target = resolve_session_target(_NO_TARGET)
    except Exception as exc:  # pragma: no cover - defensive; resolver is total
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
        generation: Switch generation the session is on after the attempt, when
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
    with the session's target here, in the one place every emit site passes
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
