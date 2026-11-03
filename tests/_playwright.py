"""Whether Playwright's Chromium is installed, for suites whose code under test launches it.

Run as ``python -m tests._playwright`` it is the browser suites' install probe:
exit 0 when the binary is there, 2 when it is not, and the last line says which.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The probe's exit status when Chromium is not installed (``check_kit.ABSENT_EXIT``).
ABSENT_EXIT = 2
INSTALLED = "chromium is installed"
ABSENT = "chromium is not installed"


def chromium_available() -> bool:
    """Whether the ``playwright`` package and its Chromium binary are both on this host.

    Probes the binary's path rather than launching it, so a suite can skip
    before the code under test starts its own browser.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    p = sync_playwright().start()
    try:
        return Path(p.chromium.executable_path).exists()
    finally:
        p.stop()


def main() -> int:
    """Print whether Chromium is installed and return the probe's exit status."""
    if chromium_available():
        print(INSTALLED)
        return 0
    print(ABSENT)
    return ABSENT_EXIT


if __name__ == "__main__":
    sys.exit(main())
