"""MCP tools: screen capture — screenshot, window listing, and window management.

Cross-platform tools that give Claude vision into operator displays.  Platform-
specific work is delegated to a :class:`ScreenCaptureBackend` (macOS or Linux).
The tool functions below are thin validation + delegation layers.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.mcp_server.workspace.tools.screen_capture_backends import (
    BackendUnavailableError,
    WindowNotFoundError,
    get_backend,
)
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.tools.screen_capture")


def _get_output_dir() -> Path:
    """Resolve the screenshots output directory from config or default."""
    config = load_osprey_config()
    sc_config = config.get("screen_capture", {})
    output_dir = Path(sc_config.get("output_dir", "./osprey-workspace/screenshots"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Tool 1: screenshot_capture
# ---------------------------------------------------------------------------
@mcp.tool()
async def screenshot_capture(
    mode: str = "full",
    target: str | None = None,
    filename: str | None = None,
) -> str:
    """Capture a screenshot and save it as a PNG.

    Args:
        mode: Capture mode — "full" (entire screen), "display" (specific display),
              "region" (x,y,w,h rectangle), or "window" (specific window by app
              name or window ID).
        target: Mode-dependent target. For "window": app name or numeric WID.
                For "region": "x,y,w,h". For "display": display number (e.g. "1").
                Ignored for "full".
        filename: Optional filename (without extension). Defaults to
                  capture_YYYYMMDD_HHMMSS.

    Returns:
        JSON with status, file path, and image dimensions. Use the Read tool
        on the returned path to view the screenshot.
    """
    valid_modes = ("full", "display", "region", "window")
    if mode not in valid_modes:
        return json.dumps(
            make_error(
                "validation_error",
                f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}.",
                [f"Use mode='{m}' for {m} capture." for m in valid_modes],
            )
        )

    # Validate mode-specific params before touching the backend
    if mode == "region" and not target:
        return json.dumps(
            make_error(
                "validation_error",
                "Region mode requires target in 'x,y,w,h' format.",
                ["Example: target='100,100,800,600'"],
            )
        )

    if mode == "window" and not target:
        return json.dumps(
            make_error(
                "validation_error",
                "Window mode requires a target (app name or window ID).",
                ["Use list_windows to find available windows and their IDs."],
            )
        )

    # Build filename
    if not filename:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{ts}"
    output_dir = _get_output_dir()
    filepath = output_dir / f"{filename}.png"

    try:
        backend = get_backend()

        if mode == "full":
            info = await backend.capture_full(str(filepath))
        elif mode == "display":
            display_num = target or "1"
            info = await backend.capture_display(display_num, str(filepath))
        elif mode == "region":
            parts = target.split(",")
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            info = await backend.capture_region(x, y, w, h, str(filepath))
        elif mode == "window":
            info = await backend.capture_window(target, str(filepath))

        # Save to ArtifactStore (unified)
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.save_data(
            tool="screenshot_capture",
            data={
                "mode": mode,
                "target": target,
                "filepath": info.filepath,
                "width": info.width,
                "height": info.height,
            },
            title=f"Screenshot ({mode}): {info.width}x{info.height}",
            description=f"Screenshot ({mode}): {info.width}x{info.height}",
            summary={
                "mode": mode,
                "dimensions": f"{info.width}x{info.height}",
                "file_size_kb": round(info.size_bytes / 1024, 1),
                "filepath": info.filepath,
            },
            access_details={
                "file_format": "PNG",
                "view_hint": f"Use Read tool on {info.filepath} to view the screenshot.",
            },
            category="screenshot",
        )

        return json.dumps(entry.to_tool_response(), default=str)

    except BackendUnavailableError as exc:
        return json.dumps(make_error("platform_error", str(exc), exc.suggestions))
    except WindowNotFoundError as exc:
        return json.dumps(
            make_error(
                "window_not_found",
                str(exc),
                [
                    "Use list_windows to see available windows.",
                    "Check that the application is running and visible.",
                ],
            )
        )
    except ValueError as exc:
        return json.dumps(make_error("validation_error", str(exc)))
    except RuntimeError as exc:
        return json.dumps(
            make_error(
                "capture_error",
                str(exc),
                ["Check that Screen Recording permission is granted to the terminal."],
            )
        )
    except Exception as exc:
        logger.exception("screenshot_capture failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Screenshot capture failed: {exc}",
                ["Check permissions and MCP server logs."],
            )
        )


# ---------------------------------------------------------------------------
# Tool 2: list_windows
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_windows(
    app_filter: str | None = None,
) -> str:
    """List visible windows with their IDs, apps, titles, and positions.

    Args:
        app_filter: Optional app name to filter by (case-insensitive substring match).

    Returns:
        JSON array of window objects with wid, app, title, x, y, width, height.
        Use the wid value with screenshot_capture(mode="window", target=<wid>).
    """
    try:
        backend = get_backend()
        windows = await backend.list_windows(app_filter=app_filter)

        return json.dumps(
            {
                "status": "success",
                "window_count": len(windows),
                "windows": [w.to_dict() for w in windows],
            }
        )

    except BackendUnavailableError as exc:
        return json.dumps(make_error("platform_error", str(exc), exc.suggestions))
    except Exception as exc:
        logger.exception("list_windows failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Window listing failed: {exc}",
                ["Ensure the terminal has Screen Recording permission."],
            )
        )


# ---------------------------------------------------------------------------
# Tool 3: manage_window
# ---------------------------------------------------------------------------
@mcp.tool()
async def manage_window(
    app: str,
    action: str,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Manage a window — bring to front, move, or resize.

    Args:
        app: Application name (e.g. "Phoebus", "Terminal").
        action: One of "bring_to_front", "move", or "resize".
        x: X position (required for "move").
        y: Y position (required for "move").
        width: Width (required for "resize").
        height: Height (required for "resize").

    Returns:
        JSON with status and action details.
    """
    valid_actions = ("bring_to_front", "move", "resize")
    if action not in valid_actions:
        return json.dumps(
            make_error(
                "validation_error",
                f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}.",
                [f"Use action='{a}' to {a.replace('_', ' ')}." for a in valid_actions],
            )
        )

    # Validate action-specific params before touching the backend
    if action == "move" and (x is None or y is None):
        return json.dumps(
            make_error(
                "validation_error",
                "Move action requires both x and y parameters.",
                ["Example: manage_window(app='Terminal', action='move', x=100, y=100)"],
            )
        )

    if action == "resize" and (width is None or height is None):
        return json.dumps(
            make_error(
                "validation_error",
                "Resize action requires both width and height parameters.",
                ["Example: manage_window(app='Terminal', action='resize', width=800, height=600)"],
            )
        )

    try:
        backend = get_backend()

        if action == "bring_to_front":
            await backend.bring_to_front(app)
        elif action == "move":
            await backend.move_window(app, x, y)
        elif action == "resize":
            await backend.resize_window(app, width, height)

        return json.dumps(
            {
                "status": "success",
                "app": app,
                "action": action,
                "details": {
                    k: v
                    for k, v in {"x": x, "y": y, "width": width, "height": height}.items()
                    if v is not None
                },
            }
        )

    except BackendUnavailableError as exc:
        return json.dumps(make_error("platform_error", str(exc), exc.suggestions))
    except ValueError as exc:
        return json.dumps(
            make_error(
                "validation_error",
                str(exc),
                ["Example: app='Phoebus' or app='Google Chrome'"],
            )
        )
    except Exception as exc:
        logger.exception("manage_window failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Window management failed: {exc}",
                ["Check permissions.", "Check MCP server logs."],
            )
        )
