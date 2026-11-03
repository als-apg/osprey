"""Whether Playwright's Chromium is installed, for suites whose code under test launches it."""

from __future__ import annotations

from pathlib import Path


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
