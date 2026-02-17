"""Linux X11 screen capture backend using pure-Python pip-installable libraries.

Uses ``mss`` for screenshots (X11 SHM extension) and ``python-xlib`` for window
listing and management (EWMH protocol).  No sudo / system packages required.

All synchronous library calls are wrapped in ``asyncio.to_thread()`` to avoid
blocking the event loop.  Each sync helper opens its own X11 ``Display``
connection for thread safety.
"""

import asyncio
import os
import struct

from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    ImageInfo,
    ScreenCaptureBackend,
    WindowInfo,
    WindowNotFoundError,
)


class LinuxX11Backend(ScreenCaptureBackend):
    """Screen capture backend for Linux X11 using mss + python-xlib."""

    # ------------------------------------------------------------------
    # Screenshot helpers (mss)
    # ------------------------------------------------------------------

    def _capture_monitor_sync(self, monitor_index: int, filepath: str) -> ImageInfo:
        """Capture a monitor by index (0 = all monitors combined)."""
        import mss
        import mss.tools

        with mss.mss() as sct:
            mon = sct.monitors[monitor_index]
            img = sct.grab(mon)
            mss.tools.to_png(img.rgb, img.size, output=filepath)
            return ImageInfo(
                filepath=filepath,
                width=img.width,
                height=img.height,
                size_bytes=os.path.getsize(filepath),
            )

    def _capture_region_sync(self, x: int, y: int, w: int, h: int, filepath: str) -> ImageInfo:
        """Capture a rectangular region."""
        import mss
        import mss.tools

        region = {"left": x, "top": y, "width": w, "height": h}
        with mss.mss() as sct:
            img = sct.grab(region)
            mss.tools.to_png(img.rgb, img.size, output=filepath)
            return ImageInfo(
                filepath=filepath,
                width=img.width,
                height=img.height,
                size_bytes=os.path.getsize(filepath),
            )

    # ------------------------------------------------------------------
    # Window helpers (python-xlib / EWMH)
    # ------------------------------------------------------------------

    def _get_client_windows_sync(self) -> list[dict]:
        """Query _NET_CLIENT_LIST and return window metadata dicts."""
        from Xlib import X
        from Xlib import display as xdisplay

        d = xdisplay.Display()
        try:
            root = d.screen().root

            # Atoms
            net_client_list = d.intern_atom("_NET_CLIENT_LIST")
            net_wm_name = d.intern_atom("_NET_WM_NAME")
            # Get client list
            prop = root.get_full_property(net_client_list, X.AnyPropertyType)
            if prop is None:
                return []

            windows: list[dict] = []
            for wid in prop.value:
                try:
                    win = d.create_resource_object("window", wid)

                    # App name from WM_CLASS
                    wm_class = win.get_wm_class()
                    app = wm_class[1] if wm_class and len(wm_class) > 1 else ""

                    # Title from _NET_WM_NAME, falling back to WM_NAME
                    title = ""
                    name_prop = win.get_full_property(net_wm_name, X.AnyPropertyType)
                    if name_prop:
                        title = name_prop.value.decode("utf-8", errors="replace")
                    else:
                        wm_name = win.get_wm_name()
                        if wm_name:
                            title = wm_name

                    # Geometry (translated to root coordinates)
                    geom = win.get_geometry()
                    coords = win.translate_coords(root, 0, 0)
                    # translate_coords returns coordinates of (0,0) of the
                    # child in root coords, but with the sign flipped
                    abs_x = -coords.x
                    abs_y = -coords.y

                    w = geom.width
                    h = geom.height

                    # Filter tiny windows (match macOS 50x50 threshold)
                    if w < 50 or h < 50:
                        continue

                    windows.append(
                        {
                            "wid": hex(wid),
                            "app": app,
                            "title": title,
                            "x": abs_x,
                            "y": abs_y,
                            "width": w,
                            "height": h,
                        }
                    )
                except Exception:
                    continue

            return windows
        finally:
            d.close()

    def _find_window_by_target_sync(self, target: str) -> dict:
        """Find a window by WID (hex) or app name substring."""
        windows = self._get_client_windows_sync()

        # Check hex WID
        if target.startswith("0x"):
            for w in windows:
                if w["wid"] == target:
                    return w
            raise WindowNotFoundError(f"No window found with WID '{target}'.")

        # App name substring match (case-insensitive)
        matches = [w for w in windows if target.lower() in w["app"].lower()]
        if not matches:
            raise WindowNotFoundError(f"No window found for app '{target}'.")
        return matches[0]

    def _send_ewmh_event_sync(self, wid_int: int, message_type_name: str, data: list[int]) -> None:
        """Send a client message to the root window (EWMH)."""
        from Xlib import X
        from Xlib import display as xdisplay

        d = xdisplay.Display()
        try:
            root = d.screen().root
            msg_type = d.intern_atom(message_type_name)

            # Pad data to 5 ints (20 bytes)
            while len(data) < 5:
                data.append(0)

            # Build ClientMessage event manually using struct
            event_data = struct.pack(
                "=BBHIiIIIII",
                33,  # ClientMessage type code
                32,  # format (32-bit)
                0,  # sequence (filled by X server)
                wid_int,
                msg_type,
                data[0],
                data[1],
                data[2],
                data[3],
                data[4],
            )

            # Use the lower-level send_event on root
            root.send_event(
                event=event_data,
                event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
            )
            d.flush()
        finally:
            d.close()

    def _resolve_wid_int(self, target: str) -> int:
        """Resolve target to an integer WID."""
        info = self._find_window_by_target_sync(target)
        wid_str = info["wid"]
        return int(wid_str, 16) if isinstance(wid_str, str) else int(wid_str)

    # ------------------------------------------------------------------
    # ScreenCaptureBackend interface
    # ------------------------------------------------------------------

    async def capture_full(self, filepath: str) -> ImageInfo:
        return await asyncio.to_thread(self._capture_monitor_sync, 0, filepath)

    async def capture_display(self, display: str, filepath: str) -> ImageInfo:
        import mss as _mss

        index = int(display)
        with _mss.mss() as sct:
            if index < 1 or index >= len(sct.monitors):
                raise ValueError(
                    f"Invalid display '{display}'. Available: 1–{len(sct.monitors) - 1}"
                )
        return await asyncio.to_thread(self._capture_monitor_sync, index, filepath)

    async def capture_region(self, x: int, y: int, w: int, h: int, filepath: str) -> ImageInfo:
        return await asyncio.to_thread(self._capture_region_sync, x, y, w, h, filepath)

    async def capture_window(self, target: str, filepath: str) -> ImageInfo:
        info = await asyncio.to_thread(self._find_window_by_target_sync, target)
        return await asyncio.to_thread(
            self._capture_region_sync,
            info["x"],
            info["y"],
            info["width"],
            info["height"],
            filepath,
        )

    async def list_windows(self, app_filter: str | None = None) -> list[WindowInfo]:
        raw = await asyncio.to_thread(self._get_client_windows_sync)

        if app_filter:
            raw = [w for w in raw if app_filter.lower() in w["app"].lower()]

        return [
            WindowInfo(
                wid=w["wid"],
                app=w["app"],
                title=w["title"],
                x=w["x"],
                y=w["y"],
                width=w["width"],
                height=w["height"],
            )
            for w in raw
        ]

    async def bring_to_front(self, app: str) -> None:
        wid_int = await asyncio.to_thread(self._resolve_wid_int, app)
        await asyncio.to_thread(
            self._send_ewmh_event_sync,
            wid_int,
            "_NET_ACTIVE_WINDOW",
            [2, 0, 0],  # source=2 (pager), timestamp=0, requestor=0
        )

    async def move_window(self, app: str, x: int, y: int) -> None:
        from Xlib import display as xdisplay

        wid_int = await asyncio.to_thread(self._resolve_wid_int, app)

        def _move():
            d = xdisplay.Display()
            try:
                win = d.create_resource_object("window", wid_int)
                win.configure(x=x, y=y)
                d.flush()
            finally:
                d.close()

        await asyncio.to_thread(_move)

    async def resize_window(self, app: str, width: int, height: int) -> None:
        from Xlib import display as xdisplay

        wid_int = await asyncio.to_thread(self._resolve_wid_int, app)

        def _resize():
            d = xdisplay.Display()
            try:
                win = d.create_resource_object("window", wid_int)
                win.configure(width=width, height=height)
                d.flush()
            finally:
                d.close()

        await asyncio.to_thread(_resize)
