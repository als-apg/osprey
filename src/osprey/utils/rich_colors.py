"""Rich color palette utilities.

Provides functions to convert Rich color names to hex values
using the official Rich library color definitions.

For STANDARD ANSI colors (0-15), can optionally query the terminal's
actual color palette for accurate matching.
"""

from __future__ import annotations

import logging
import re
import sys
import time

_logger = logging.getLogger(__name__)

# Cache for terminal colors (populated at startup if TTY available)
_terminal_colors: dict[int, str] = {}

# Regex to parse OSC 4 response: ESC ] 4 ; n ; rgb:RRRR/GGGG/BBBB (ST or BEL)
# Note: ESC prefix is optional because it may be consumed by previous read
_OSC4_RESPONSE_RE = re.compile(
    r"(?:\x1b)?\]4;(\d+);rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)(?:\x1b\\|\x07)?"
)


def query_terminal_color(color_index: int) -> tuple[int, str] | None:
    """Query terminal for actual RGB of ANSI color using OSC 4.

    Uses the OSC 4 escape sequence to query the terminal's configured
    color for the given palette index. Only works when running in a
    terminal with TTY access.

    Note: Due to terminal timing, the response may be for a previous query.
    The caller should use the returned response_index as the key.

    Args:
        color_index: ANSI color index to query (0-16, query one extra to get all 16)

    Returns:
        Tuple of (response_index, hex_color) or None if query fails.
        Use response_index as the key, not color_index.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    try:
        import select
        import termios
    except ImportError:
        # Not available on Windows
        return None

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        # Use cbreak mode: disable ICANON (line buffering) and ECHO
        # This is less aggressive than full raw mode
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
        new_settings[6][termios.VMIN] = 0  # Non-blocking
        new_settings[6][termios.VTIME] = 1  # 100ms timeout per read
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)

        try:
            # Drain any pending input to avoid reading stale responses
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)

            # Send OSC 4 query with BEL terminator (more widely supported)
            sys.stdout.write(f"\x1b]4;{color_index};?\x07")
            sys.stdout.flush()

            # Wait for terminal to process and respond
            time.sleep(0.1)  # 100ms delay for complete response

            # Read response with overall timeout
            response = ""
            for _ in range(50):  # Max 50 chars
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if not ready:
                    break
                char = sys.stdin.read(1)
                if not char:
                    break
                response += char
                # Check for complete response (ends with ST or BEL)
                if response.endswith("\x1b\\") or response.endswith("\x07"):
                    break

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Parse response with regex
        match = _OSC4_RESPONSE_RE.search(response)
        if match:
            # Use response_index as the key (may differ from queried index due to timing)
            response_index = int(match.group(1))

            # Convert 4-digit hex to 8-bit (take first 2 digits)
            r_val = int(match.group(2)[:2], 16)
            g_val = int(match.group(3)[:2], 16)
            b_val = int(match.group(4)[:2], 16)
            hex_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
            return (response_index, hex_color)

    except Exception as e:
        _logger.debug(f"OSC 4 query failed for color {color_index}: {e}")

    return None


def init_terminal_colors() -> None:
    """Query terminal colors for STANDARD range (0-15) at startup.

    Should be called once at application startup when running in a terminal.
    Results are cached for the session. If no TTY is available, this is a no-op.
    """
    global _terminal_colors

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        _logger.debug("No TTY available, using Rich default colors")
        return

    colors_loaded = {}
    # Query 0-16 (17 total) because responses are delayed by one position
    # Each query N typically returns the response for query N-1
    for i in range(17):
        result = query_terminal_color(i)
        if result:
            response_idx, hex_color = result
            # Only store indices 0-15 (standard ANSI colors)
            if 0 <= response_idx <= 15:
                _terminal_colors[response_idx] = hex_color
                colors_loaded[response_idx] = hex_color

    if colors_loaded:
        _logger.debug(f"Loaded {len(colors_loaded)} terminal colors")
    else:
        _logger.debug("Terminal color query not supported, using Rich defaults")


def get_rich_color_hex(color_name: str) -> str | None:
    """Convert a Rich color name to its hex value.

    For STANDARD colors (0-15), uses the terminal's actual palette if
    it was queried at startup. Otherwise uses Rich's truecolor representation.

    Args:
        color_name: Rich color name (e.g., 'sky_blue2', 'cyan')

    Returns:
        Hex color string (e.g., '#87afff') or None if invalid
    """
    try:
        from rich.color import Color, ColorType

        color = Color.parse(color_name)

        # For STANDARD colors, use terminal palette if available
        if color.type == ColorType.STANDARD and color.number in _terminal_colors:
            return _terminal_colors[color.number]

        # Otherwise use Rich's truecolor approximation
        triplet = color.get_truecolor()
        return f"#{triplet.red:02x}{triplet.green:02x}{triplet.blue:02x}"
    except Exception:
        return None
