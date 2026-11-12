"""Browser tests: a sidebar layout stored before the System group existed.

An operator's browser keeps its own sidebar layout; one stored by an older
version must show System holding Events, whichever group it held Events in.
Only a move made on the current version sticks.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_sidebar_groups_migration_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces._browser import wait_for_page_ready
from tests.interfaces._hub import hub_server

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.sync_api import Browser

try:
    from playwright.sync_api import expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_KEY = "osprey-sidebar-groups"


def _stored(placement: dict[str, str], version: int = 1) -> dict:
    """A layout as the build before System stored it: two groups, no System."""
    return {
        "version": version,
        "groups": [
            {"id": "workspace", "label": "Workspace"},
            {"id": "tools", "label": "Tools"},
        ],
        "placement": placement,
    }


def _groups_of(
    browser: Browser,
    tmp_path: Path,
    stored: dict,
) -> tuple[dict[str, str], list[str]]:
    """Load the hub with `stored` seeded; the group of Events and Bluesky, and all groups."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace
    )
    # Events is a URL-backed custom panel, as the demo preset has it.
    panels = [
        {"id": "events", "label": "Events", "url": "http://127.0.0.1:9/dashboard"},
        {"id": "bluesky", "label": "Bluesky", "url": "http://127.0.0.1:9/bluesky"},
    ]
    with hub_server(workspace, shared_root, custom_panels=panels) as (base_url, _app):
        page = browser.new_page()
        try:
            value = json.dumps(json.dumps(stored))
            page.add_init_script(
                f"try {{ localStorage.setItem({json.dumps(_KEY)}, {value}); }} catch (e) {{}}"
            )
            page.goto(base_url, wait_until="domcontentloaded")
            wait_for_page_ready(page)
            found: dict[str, str] = {}
            for panel_id in ("events", "bluesky"):
                row = page.locator(f'button.panel-rail-button[data-panel-id="{panel_id}"]')
                expect(row).to_be_attached(timeout=10_000)
                group = row.locator("xpath=ancestor::*[@data-group][1]")
                found[panel_id] = group.get_attribute("data-group") or ""
            groups = page.eval_on_selector_all(
                ".sidebar-group[data-group]", "els => els.map(e => e.dataset.group)"
            )
            return found, groups
        finally:
            page.close()


def test_untouched_old_layout_shows_events_in_system(tmp_path, chromium_browser):
    """An old default layout (Events in Workspace) gains System holding Events."""
    found, groups = _groups_of(chromium_browser, tmp_path, _stored({"events": "workspace"}))
    assert found["events"] == "system"
    assert "system" in groups


@pytest.mark.parametrize("version", [1, 2])
def test_old_layout_events_in_tools_custom_group_kept(tmp_path, chromium_browser, version):
    """The owner's layout: Events in Tools, a custom group holding Bluesky, System empty."""
    stored = _stored({"events": "tools", "bluesky": "g-0a1b2c3d"}, version=version)
    stored["groups"] += [
        {"id": "g-0a1b2c3d", "label": "test'"},
        {"id": "system", "label": "System"},
    ]
    found, groups = _groups_of(chromium_browser, tmp_path, stored)
    assert found == {"events": "system", "bluesky": "g-0a1b2c3d"}
    assert groups == ["workspace", "tools", "g-0a1b2c3d", "system"]


def test_current_version_events_in_tools_sticks(tmp_path, chromium_browser):
    """On the current version, Events moved to Tools is the operator's choice."""
    stored = _stored({"events": "tools"}, version=3)
    stored["groups"].append({"id": "system", "label": "System"})
    found, _ = _groups_of(chromium_browser, tmp_path, stored)
    assert found["events"] == "tools"
