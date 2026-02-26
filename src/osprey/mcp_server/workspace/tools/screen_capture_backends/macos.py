"""macOS screen capture backend using native tools.

Uses ``screencapture`` for screenshots, inline Swift (CGWindowListCopyWindowInfo)
for window listing, ``sips`` for image dimensions, and ``osascript`` (AppleScript)
for window management.
"""

import asyncio
import json
import re

from osprey.mcp_server.workspace.tools.screen_capture_backends.base import (
    ImageInfo,
    ScreenCaptureBackend,
    WindowInfo,
    WindowNotFoundError,
)

# Swift one-liner for CGWindowListCopyWindowInfo (inline constant)
_SWIFT_LIST_WINDOWS = r"""
import Cocoa
import Foundation

let options = CGWindowListOption(arrayLiteral: .optionOnScreenOnly, .excludeDesktopElements)
guard let infoList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
    print("[]")
    exit(0)
}

var results: [[String: Any]] = []
for info in infoList {
    guard let wid = info[kCGWindowNumber as String] as? Int,
          let bounds = info[kCGWindowBounds as String] as? [String: Any],
          let x = bounds["X"] as? Double,
          let y = bounds["Y"] as? Double,
          let w = bounds["Width"] as? Double,
          let h = bounds["Height"] as? Double else { continue }
    let app = info[kCGWindowOwnerName as String] as? String ?? ""
    let title = info[kCGWindowName as String] as? String ?? ""
    if w < 50 || h < 50 { continue }
    results.append([
        "wid": wid,
        "app": app,
        "title": title,
        "x": Int(x),
        "y": Int(y),
        "width": Int(w),
        "height": Int(h),
    ])
}

let jsonData = try! JSONSerialization.data(withJSONObject: results, options: [])
print(String(data: jsonData, encoding: .utf8)!)
"""

# Security: valid app name pattern for AppleScript injection prevention
_APP_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9 ._-]+$")


class MacOSBackend(ScreenCaptureBackend):
    """Screen capture backend for macOS using native system tools."""

    async def _list_windows_raw(self) -> list[dict]:
        """Run the Swift CGWindowList script and return parsed JSON array."""
        proc = await asyncio.create_subprocess_exec(
            "swift",
            "-e",
            _SWIFT_LIST_WINDOWS,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Swift window listing failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

        return json.loads(stdout.decode())

    async def _get_image_info(self, filepath: str) -> ImageInfo:
        """Get image dimensions via sips and file size."""
        import os

        proc = await asyncio.create_subprocess_exec(
            "sips",
            "-g",
            "pixelWidth",
            "-g",
            "pixelHeight",
            filepath,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode()

        width = height = 0
        for line in text.splitlines():
            if "pixelWidth" in line:
                width = int(line.split(":")[-1].strip())
            elif "pixelHeight" in line:
                height = int(line.split(":")[-1].strip())

        size_bytes = os.path.getsize(filepath)
        return ImageInfo(filepath=filepath, width=width, height=height, size_bytes=size_bytes)

    async def _run_screencapture(self, args: list[str], filepath: str) -> ImageInfo:
        """Run screencapture with given args, verify output, return ImageInfo."""
        import os

        cmd = ["screencapture", "-x", *args, filepath]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"screencapture failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

        if not os.path.exists(filepath):
            raise RuntimeError("Screenshot file was not created.")

        return await self._get_image_info(filepath)

    async def _run_osascript(self, script: str) -> None:
        """Execute an AppleScript snippet via osascript."""
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"AppleScript failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

    def _validate_app_name(self, app: str) -> None:
        """Validate app name to prevent AppleScript injection."""
        if not _APP_NAME_PATTERN.match(app):
            raise ValueError(
                "Invalid app name. Only alphanumeric characters, spaces, dots, "
                "underscores, and hyphens are allowed."
            )

    async def capture_full(self, filepath: str) -> ImageInfo:
        return await self._run_screencapture([], filepath)

    async def capture_display(self, display: str, filepath: str) -> ImageInfo:
        return await self._run_screencapture([f"-D{display}"], filepath)

    async def capture_region(self, x: int, y: int, w: int, h: int, filepath: str) -> ImageInfo:
        return await self._run_screencapture([f"-R{x},{y},{w},{h}"], filepath)

    async def capture_window(self, target: str, filepath: str) -> ImageInfo:
        if target.isdigit():
            wid = target
        else:
            windows = await self._list_windows_raw()
            matches = [w for w in windows if target.lower() in w.get("app", "").lower()]
            if not matches:
                raise WindowNotFoundError(f"No window found for app '{target}'.")
            wid = str(matches[0]["wid"])

        return await self._run_screencapture([f"-l{wid}", "-o"], filepath)

    async def list_windows(self, app_filter: str | None = None) -> list[WindowInfo]:
        raw = await self._list_windows_raw()

        if app_filter:
            raw = [w for w in raw if app_filter.lower() in w.get("app", "").lower()]

        return [
            WindowInfo(
                wid=w["wid"],
                app=w.get("app", ""),
                title=w.get("title", ""),
                x=w.get("x", 0),
                y=w.get("y", 0),
                width=w.get("width", 0),
                height=w.get("height", 0),
            )
            for w in raw
        ]

    async def bring_to_front(self, app: str) -> None:
        self._validate_app_name(app)
        await self._run_osascript(f'tell application "{app}" to activate')

    async def move_window(self, app: str, x: int, y: int) -> None:
        self._validate_app_name(app)
        await self._run_osascript(
            f'tell application "System Events" to tell process "{app}" '
            f"to set position of front window to {{{x}, {y}}}"
        )

    async def resize_window(self, app: str, width: int, height: int) -> None:
        self._validate_app_name(app)
        await self._run_osascript(
            f'tell application "System Events" to tell process "{app}" '
            f"to set size of front window to {{{width}, {height}}}"
        )
