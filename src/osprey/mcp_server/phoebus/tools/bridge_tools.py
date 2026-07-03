"""MCP tools: native Phoebus interaction via the agent bridge.

Each tool is a thin validation + HTTP-delegation layer over one endpoint of the
Phoebus agent bridge (the in-JVM JSON/HTTP server embedded in a running Phoebus
product):

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
phoebus_open_panel          POST /open
phoebus_list_displays       GET  /displays
phoebus_perceive            GET  /perceive
phoebus_perceive_region     GET  /perceive/region
phoebus_snapshot            GET  /snapshot            (image/png)
phoebus_drive               POST /drive
==========================  =================================================

The HTTP primitives (``_http_get_json`` / ``_http_get_bytes`` /
``_http_post_drive`` / ``_http_post_open``) are module-level so tests can patch
the network boundary.

Display addressing
------------------
Every tool that accepts a ``display`` argument supports three forms:

* ``"active"`` — the currently focused display (default).
* A display name as returned by ``phoebus_list_displays``.
* ``"handle:<id>"`` — the deterministic handle returned by
  ``phoebus_open_panel`` (e.g. ``"handle:d-3"``).  Use this form when
  multiple displays are open and you need to target the one you just opened.

On a backend shared by multiple web terminals, ``"active"`` resolves the
process-global focused display — a race when two terminals perceive/drive
concurrently. Set ``PHOEBUS_REQUIRE_HANDLE=1`` (or ``phoebus.require_handle:
true`` in config.yml) to reject the implicit ``"active"`` fallback and force
callers to address a specific display (a handle or an explicit name). Off by
default — the standalone single-user demo is unaffected.

Panel name registry
-------------------
``phoebus_open_panel`` resolves logical panel names to ``.bob`` resources via:

1. ``phoebus.panels.<name>`` in config.yml — value may be an absolute path,
   a ``file:`` URL, or a path relative to the config file's directory.
2. Built-in registry (demo panels shipped with OSPREY, resolved relative to
   the osprey repository root when running from a development checkout).
"""

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import anyio
from fastmcp.exceptions import ToolError

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import _post_json_with_response, notify_panel_focus, phoebus_bridge_url
from osprey.mcp_server.phoebus.server import mcp
from osprey.utils.workspace import load_osprey_config, resolve_config_path

logger = logging.getLogger("osprey.mcp_server.tools.phoebus")

# Module-level alias so tests can patch bridge deadline timing without touching
# the global time.monotonic that anyio uses internally for its thread-pool logic.
_monotonic = time.monotonic

_TIMEOUT = 15  # seconds — snapshot/perceive marshal onto the Phoebus FX thread

_UNREACHABLE_HINTS = [
    "Start a Phoebus product with the agent bridge enabled (default port 7979).",
    "Check the PHOEBUS_BRIDGE_URL env var or phoebus.host/phoebus.port in config.yml.",
    "Launch the bundled demo with demos/phoebus/run_demo.sh, then retry.",
]


# ---------------------------------------------------------------------------
# HTTP boundary (patched in tests)
# ---------------------------------------------------------------------------
def _http_get_json(path: str) -> tuple[int, dict | list]:
    """GET ``path`` on the bridge and return ``(status, parsed_json)``."""
    url = f"{phoebus_bridge_url()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body: dict = {}
        try:
            body = json.loads(exc.read())
        except Exception:
            pass
        return exc.code, body


def _http_get_bytes(path: str) -> tuple[int, dict, bytes]:
    """GET ``path`` and return ``(status, headers, raw_body)`` (for image/png)."""
    url = f"{phoebus_bridge_url()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _http_post_drive(payload: dict) -> tuple[int, dict]:
    """POST a drive request and return ``(status, parsed_json)``."""
    return _post_json_with_response(f"{phoebus_bridge_url()}/drive", payload, timeout=_TIMEOUT)


def _http_post_open(payload: dict) -> tuple[int, dict]:
    """POST an open request and return ``(status, parsed_json)``."""
    return _post_json_with_response(f"{phoebus_bridge_url()}/open", payload, timeout=_TIMEOUT)


def _bridge_error_message(body: object, status: int) -> str:
    """Extract the bridge's JSON ``error`` message, falling back to the status."""
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return f"Phoebus bridge returned HTTP {status}."


def _snapshot_dir() -> Path:
    """Resolve the directory PNG snapshots are written to (config or default)."""
    config = load_osprey_config()
    out = Path(config.get("phoebus", {}).get("snapshot_dir", "./_agent_data/screenshots"))
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Require-handle addressing (off by default; see module docstring)
# ---------------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _require_handle() -> bool:
    """Whether implicit ``"active"`` display addressing must be rejected.

    Resolution order (mirrors ``osprey.interfaces.vendor.is_offline``):

    1. ``PHOEBUS_REQUIRE_HANDLE`` env var (truthy: 1/true/yes/on; falsy:
       0/false/no/off) — set by the framework server definition; wins outright.
    2. ``phoebus.require_handle`` in config.yml.
    3. ``False`` default — existing ``"active"`` fallback behavior is unchanged.
    """
    env = os.environ.get("PHOEBUS_REQUIRE_HANDLE", "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in _FALSY:
        return False
    config = load_osprey_config()
    return bool(config.get("phoebus", {}).get("require_handle", False))


def _check_explicit_display(display: str) -> None:
    """Reject the implicit ``"active"`` display when require-handle mode is on.

    ``"active"`` is the only value that resolves process-global focus — a
    handle or an explicit display name always addresses one specific display,
    so both are allowed regardless of this setting.
    """
    if display == "active" and _require_handle():
        make_error(
            "phoebus_handle_required",
            "Implicit 'active' display addressing is disabled on this backend; "
            "call phoebus_open_panel to obtain a handle and pass it as "
            "display='handle:<id>'.",
            [
                "Call phoebus_open_panel to obtain a handle, then pass "
                "display='handle:<id>' to target that display explicitly.",
                "Alternatively pass an explicit display name from phoebus_list_displays.",
            ],
        )


# ---------------------------------------------------------------------------
# Panel name → .bob resource resolution
# ---------------------------------------------------------------------------

# Built-in registry: NOT a framework requirement. The framework works fine with
# this empty — phoebus.panels.<name> in config.yml is the real deployment path
# (see the module docstring). The only entry ever added here is the demo panel
# shipped under demos/, and only when that soft/optional condition is met (see
# below) — a demos/-relative path must never be a *required* framework default.
_BUILTIN_PANELS: dict[str, str] = {}

# bridge_tools.py is at src/osprey/mcp_server/phoebus/tools/bridge_tools.py
# → parents[5] is the repo root (src/osprey/mcp_server/phoebus/tools → 5 up → repo root)
_OSPREY_REPO_ROOT: Path = Path(__file__).resolve().parents[5]

# Soft default for the standalone demo (run_demo.sh + the walkthrough): register
# "osprey_demo" only when the demos/ tree is actually present next to this
# checkout (true for a git clone, false for an installed package) — so the demo
# keeps working out of the box without the framework carrying a hardcoded
# demos/-relative path as a required default.
_DEMO_PANEL_RELATIVE = "demos/phoebus/panels/osprey_demo.bob"
if (_OSPREY_REPO_ROOT / _DEMO_PANEL_RELATIVE).exists():
    _BUILTIN_PANELS["osprey_demo"] = _DEMO_PANEL_RELATIVE

_OPEN_READY_TIMEOUT = 30  # seconds to wait for ready=True after /open


def _resolve_panel_resource(name: str) -> str:
    """Map a logical panel name to a bridge-ready resource string.

    Resolution order:

    1. ``phoebus.panels.<name>`` in config.yml — value may be an absolute
       path, a ``file:`` URL, or a path relative to the config file's
       directory (resolved to absolute before forwarding).
    2. Built-in registry (demo panels shipped with OSPREY), resolved relative
       to the osprey repository root when running from a development checkout,
       or relative to the config file's directory otherwise.

    Args:
        name: Logical panel name (e.g. ``"osprey_demo"``).

    Raises:
        ToolError(unknown_panel): When *name* is not found in either source.
    """
    config = load_osprey_config()
    panels: dict = config.get("phoebus", {}).get("panels", {})
    resource: str | None = panels.get(name)

    if resource is None:
        builtin_relative = _BUILTIN_PANELS.get(name)
        if builtin_relative is not None:
            # Prefer the development-checkout repo root; fall back to config dir.
            candidate = (_OSPREY_REPO_ROOT / builtin_relative).resolve()
            if candidate.exists():
                resource = str(candidate)
            else:
                config_path = resolve_config_path()
                if config_path.exists():
                    resource = str((config_path.parent / builtin_relative).resolve())
                else:
                    resource = builtin_relative  # pass through; bridge will report missing

    if resource is None:
        known = sorted(set(panels) | set(_BUILTIN_PANELS))
        make_error(
            "unknown_panel",
            f"Unknown panel name '{name}'.",
            [
                f"Known panels: {known}.",
                f"Add 'phoebus.panels.{name}: /path/to/file.bob' to config.yml to register a custom panel.",
                "Paths may be absolute or relative to the config.yml directory.",
            ],
        )

    # Resolve relative non-URL config paths against the config file's directory.
    assert resource is not None  # make_error above raises; assertion satisfies type checker
    if not resource.startswith("file:") and not Path(resource).is_absolute():
        config_path = resolve_config_path()
        if config_path.exists():
            resource = str((config_path.parent / resource).resolve())

    return resource


# ---------------------------------------------------------------------------
# Tool 1: open panel
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_open_panel(name: str, focus: bool = True) -> str:
    """Open a Phoebus display by logical panel name and return its handle.

    Sends a ``POST /open`` request to the bridge, which loads the named
    ``.bob`` file and returns a deterministic display handle (e.g.
    ``"handle:d-3"``).  The tool polls ``GET /displays`` until the display
    reports ``ready=true`` or a 30-second timeout elapses, so that
    subsequent ``phoebus_perceive`` / ``phoebus_drive`` calls on the same
    handle do not race against model loading.

    Pass the returned ``handle`` string as the ``display`` argument of
    ``phoebus_perceive``, ``phoebus_perceive_region``, ``phoebus_snapshot``,
    and ``phoebus_drive`` to target this display unambiguously even when
    multiple displays are open.

    Panel name resolution (in order):

    1. ``phoebus.panels.<name>`` in config.yml — accepts an absolute path, a
       ``file:`` URL, or a path relative to the config file's directory.
    2. Built-in registry (demo panels shipped with OSPREY).

    After a successful open the tool asks the Web Terminal (best-effort,
    never fatal) to focus this instance's panel tab, so the user sees the
    display they just asked for without hunting through tabs.

    Args:
        name: Logical panel name (e.g. ``"osprey_demo"``).
        focus: Switch the Web Terminal to this instance's panel tab after
            opening (default ``True``). Pass ``False`` for batch or
            background opens that should not steal the user's view.

    Returns:
        JSON ``{"status": "success", "handle": "handle:d-N", "id": "d-N",
        "resource": "<resolved-path>", "ready": <bool>, "focused": <bool>}``.
    """
    resource = _resolve_panel_resource(name)  # raises ToolError on unknown name

    try:
        status, body = await anyio.to_thread.run_sync(_http_post_open, {"resource": resource})
    except Exception as exc:
        make_error(
            "phoebus_unreachable",
            f"Could not reach the Phoebus bridge: {exc}",
            _UNREACHABLE_HINTS,
        )

    if status != 200:
        make_error(
            "phoebus_open_failed",
            _bridge_error_message(body, status),
            ["Check that the resource path exists and is readable by the Phoebus process."],
        )

    display_id: str = body.get("id", "")
    if not display_id:
        make_error(
            "phoebus_open_failed",
            "Phoebus bridge returned a success response with no display id.",
            ["This may indicate a bridge version mismatch; check the bridge logs."],
        )

    handle = f"handle:{display_id}"
    ready: bool = bool(body.get("ready", False))

    # Poll until ready=True so downstream perceive/drive calls don't race the
    # JavaFX model-loading phase.  Both the sleep and the GET /displays call are
    # non-blocking so other tool calls and keepalives proceed normally.
    if not ready:
        deadline = _monotonic() + _OPEN_READY_TIMEOUT
        while _monotonic() < deadline:
            await asyncio.sleep(0.5)
            try:
                lst_status, displays = await anyio.to_thread.run_sync(
                    _http_get_json, "/displays"
                )
            except Exception:
                break  # bridge vanished; return what we have
            if lst_status == 200 and isinstance(displays, list):
                for d in displays:
                    if d.get("id") == display_id:
                        ready = bool(d.get("ready", False))
                        break
            if ready:
                break

    # Best-effort UX signal: switch the Web Terminal to this instance's panel
    # tab (panel id == server name; OSPREY_SERVER_NAME is set per-instance by
    # the registry, so a phoebus2 clone focuses the PHOEBUS2 tab). The display
    # is already open at this point — a missing/unreachable web terminal
    # (CLI-only mode) must never turn the open into a failure.
    focused = False
    if focus:
        panel_id = os.environ.get("OSPREY_SERVER_NAME", "phoebus")
        try:
            await anyio.to_thread.run_sync(notify_panel_focus, panel_id)
            focused = True
        except Exception as exc:
            logger.debug("panel focus notification failed (non-fatal): %s", exc)

    return json.dumps(
        {
            "status": "success",
            "handle": handle,
            "id": display_id,
            "resource": body.get("resource", resource),
            "ready": ready,
            "focused": focused,
        }
    )


# ---------------------------------------------------------------------------
# Tool 2: list displays
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_list_displays() -> str:
    """List every currently-open Phoebus display.

    Returns:
        JSON ``{"status": "success", "displays": [{"name", "ready", "active"}, ...]}``.
        An empty list means Phoebus is running but has no display open (or the
        bridge could reach no display).
    """
    try:
        status, body = _http_get_json("/displays")
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error("phoebus_error", _bridge_error_message(body, status))
    return json.dumps({"status": "success", "displays": body})


# ---------------------------------------------------------------------------
# Tool 3: perceive
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_perceive(display: str = "active") -> str:
    """Perceive a Phoebus display — walk its widget tree and report each widget.

    For every widget the result carries its type, name, nesting depth, on-screen
    bounds, visibility, and live PV state (name, value, severity, writability).
    Use this to understand what is on screen before snapshotting or driving.

    Args:
        display: Display reference — ``"active"`` (default, the focused display),
                 a display name as returned by ``phoebus_list_displays``, or a
                 handle string returned by ``phoebus_open_panel`` (e.g.
                 ``"handle:d-3"``).

    Returns:
        JSON ``{"status": "success", "display": <ref>, "perception": {...}}`` where
        ``perception`` is the bridge's stable schema (``display`` + ``widgets``).
    """
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced
    path = f"/perceive?display={urllib.parse.quote(display)}"
    try:
        status, body = _http_get_json(path)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error(
            "phoebus_error",
            _bridge_error_message(body, status),
            ["Check the display reference; list displays with phoebus_list_displays."],
        )
    return json.dumps({"status": "success", "display": display, "perception": body})


# ---------------------------------------------------------------------------
# Tool 4: perceive region
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_perceive_region(
    x: float, y: float, w: float, h: float, display: str = "active"
) -> str:
    """Perceive only the widgets whose on-screen bounds intersect a rectangle.

    Args:
        x: Left edge of the query rectangle, in screen pixels.
        y: Top edge of the query rectangle, in screen pixels.
        w: Rectangle width in screen pixels (must be > 0).
        h: Rectangle height in screen pixels (must be > 0).
        display: Display reference (``"active"``, a display name, or a handle
                 string from ``phoebus_open_panel`` e.g. ``"handle:d-3"``).

    Returns:
        JSON ``{"status": "success", "display": <ref>, "perception": {...}}`` whose
        widget list is filtered to the rectangle and sorted top-left first.
    """
    if w <= 0 or h <= 0:
        return make_error("validation_error", f"w and h must be > 0 (got w={w}, h={h}).")
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced
    path = f"/perceive/region?display={urllib.parse.quote(display)}&x={x}&y={y}&w={w}&h={h}"
    try:
        status, body = _http_get_json(path)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error("phoebus_error", _bridge_error_message(body, status))
    return json.dumps({"status": "success", "display": display, "perception": body})


# ---------------------------------------------------------------------------
# Tool 5: snapshot
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_snapshot(
    widget: str, display: str = "active", highlight: bool = False, dpi: float = 1.0
) -> str:
    """Capture a PNG snapshot of one widget and save it for viewing.

    The widget must have a live on-screen representation. The PNG is written to
    the screenshots directory and registered as an artifact; the response carries
    its file path plus the screen-space registration headers (origin + scale) so
    pixels can be mapped back to screen coordinates.

    Args:
        widget: Widget reference — a walk-order index (e.g. ``"0"``) or a widget
                name (e.g. ``"Setpoint"``).
        display: Display reference (``"active"``, a display name, or a handle
                 string from ``phoebus_open_panel`` e.g. ``"handle:d-3"``).
        highlight: When true, overlay an orange-red border on the target widget.
        dpi: Scale factor — 1.0 native, 2.0 HiDPI. Must be in (0, 8].

    Returns:
        JSON artifact response including the saved file path. Use the Read tool on
        that path to view the snapshot.
    """
    if not (0 < dpi <= 8):
        return make_error("validation_error", f"dpi must be in (0, 8] (got {dpi}).")
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced

    path = (
        f"/snapshot?display={urllib.parse.quote(display)}"
        f"&widget={urllib.parse.quote(widget)}"
        f"&highlight={'true' if highlight else 'false'}&dpi={dpi}"
    )
    try:
        status, headers, content = _http_get_bytes(path)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        msg = f"Phoebus bridge returned HTTP {status}."
        try:
            msg = json.loads(content).get("error", msg)
        except Exception:
            pass
        return make_error(
            "phoebus_error",
            msg,
            [
                "Confirm the widget is on screen and rendered (headless widgets "
                "cannot be snapshotted)."
            ],
        )

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", widget)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    fpath = _snapshot_dir() / f"phoebus_{safe}_{ts}.png"
    fpath.write_bytes(content)

    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.save_data(
            tool="phoebus_snapshot",
            data={
                "display": display,
                "widget": widget,
                "filepath": str(fpath),
                "bytes": len(content),
                "origin_x": headers.get("X-Bridge-Origin-X"),
                "origin_y": headers.get("X-Bridge-Origin-Y"),
                "scale": headers.get("X-Bridge-Scale"),
            },
            title=f"Phoebus snapshot: {widget}",
            description=f"PNG snapshot of widget '{widget}' on display '{display}'",
            summary={
                "widget": widget,
                "file_size_kb": round(len(content) / 1024, 1),
                "filepath": str(fpath),
            },
            access_details={
                "file_format": "PNG",
                "view_hint": f"Use Read tool on {fpath} to view the snapshot.",
            },
            category="screenshot",
        )
        return json.dumps(entry.to_tool_response(), default=str)
    except ToolError:
        raise
    except Exception:
        # ArtifactStore unavailable (e.g. minimal context) — still return the path.
        logger.exception("phoebus_snapshot: artifact save failed; returning raw path")
        return json.dumps(
            {
                "status": "success",
                "filepath": str(fpath),
                "bytes": len(content),
                "view_hint": f"Use Read tool on {fpath} to view the snapshot.",
            }
        )


# ---------------------------------------------------------------------------
# Tool 6: drive
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_drive(
    widget: str,
    verb: str,
    display: str = "active",
    value: str | None = None,
    mode: str = "synthetic",
) -> str:
    """Drive a Phoebus widget — click an action control or type a value.

    Two drive modes:
      * ``synthetic`` (default) — inject the real GUI event the widget listens
        for, so confirm dialogs, enable/disable rules, and bound scripts all run,
        exactly as if an operator clicked.
      * ``semantic`` — write the widget's primary PV / trigger its action directly
        through the runtime, bypassing the GUI chain. ``semantic`` supports only
        the ``type`` verb (``click`` returns a bypass result).

    Args:
        widget: Widget reference — a walk-order index or a widget name.
        verb: ``"click"`` (fire an action control) or ``"type"`` (commit a value
              into a text control).
        display: Display reference (``"active"``, a display name, or a handle
                 string from ``phoebus_open_panel`` e.g. ``"handle:d-3"``).
        value: Value to commit for the ``"type"`` verb; ignored for ``"click"``.
        mode: ``"synthetic"`` or ``"semantic"``.

    Returns:
        JSON ``{"status": "success", "fired": <bool>, "detail": <str>}``. ``fired``
        is false when no interactive control was resolved or the mode bypassed it.
    """
    verb_l = verb.lower()
    mode_l = mode.lower()
    if verb_l not in ("click", "type"):
        return make_error(
            "validation_error", f"Invalid verb '{verb}'.", ["verb must be 'click' or 'type'."]
        )
    if mode_l not in ("synthetic", "semantic"):
        return make_error(
            "validation_error",
            f"Invalid mode '{mode}'.",
            ["mode must be 'synthetic' or 'semantic'."],
        )
    if verb_l == "type" and value is None:
        return make_error("validation_error", "verb 'type' requires a 'value'.")
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced

    payload = {
        "display": display,
        "widget": widget,
        "verb": verb_l,
        "value": value,
        "mode": mode_l,
    }
    try:
        status, body = _http_post_drive(payload)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error(
            "phoebus_rejected",
            _bridge_error_message(body, status),
            ["Check the widget reference and that verb/mode are valid."],
        )
    return json.dumps(
        {
            "status": "success",
            "fired": body.get("fired"),
            "detail": body.get("detail"),
        }
    )
