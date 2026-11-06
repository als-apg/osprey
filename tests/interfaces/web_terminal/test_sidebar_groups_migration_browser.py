"""Browser tests: a sidebar layout stored before the System group existed.

An operator's browser keeps its own sidebar layout; one stored before System
existed must still show System holding Events, unless the operator moved Events
themselves.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_sidebar_groups_migration_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

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


def _group_of_events(browser: Browser, tmp_path: Path, stored: dict) -> tuple[str, list[str]]:
    """Load the hub with `stored` seeded; the group Events sits in, and all groups."""
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    shared_root = patch(
        "osprey_connectors.workspace.resolve_shared_data_root", return_value=workspace
    )
    # Events is a URL-backed custom panel, as the control-assistant preset has it.
    events = {"id": "events", "label": "Events", "url": "http://127.0.0.1:9/dashboard"}
    with hub_server(workspace, shared_root, custom_panels=[events]) as (base_url, _app):
        page = browser.new_page()
        try:
            value = json.dumps(json.dumps(stored))
            page.add_init_script(
                f"try {{ localStorage.setItem({json.dumps(_KEY)}, {value}); }} catch (e) {{}}"
            )
            page.goto(base_url, wait_until="domcontentloaded")
            events = page.locator('button.panel-rail-button[data-panel-id="events"]')
            expect(events).to_be_attached(timeout=10_000)
            group = events.locator("xpath=ancestor::*[@data-group][1]")
            groups = page.eval_on_selector_all(
                ".sidebar-group[data-group]", "els => els.map(e => e.dataset.group)"
            )
            return group.get_attribute("data-group") or "", groups
        finally:
            page.close()


def test_untouched_old_layout_shows_events_in_system(tmp_path, chromium_browser):
    """An old default layout (Events in Workspace) gains System holding Events."""
    group, groups = _group_of_events(chromium_browser, tmp_path, _stored({"events": "workspace"}))
    assert group == "system"
    assert "system" in groups


def test_empty_system_group_still_shows_events(tmp_path, chromium_browser):
    """A layout saved by an intermediate build (System present, empty) follows the default."""
    stored = _stored({"events": "workspace"})
    stored["groups"].append({"id": "system", "label": "System"})
    group, _ = _group_of_events(chromium_browser, tmp_path, stored)
    assert group == "system"


def test_current_version_events_in_workspace_sticks(tmp_path, chromium_browser):
    """On the current version, Events moved to Workspace is the operator's choice."""
    stored = _stored({"events": "workspace"}, version=2)
    stored["groups"].append({"id": "system", "label": "System"})
    group, _ = _group_of_events(chromium_browser, tmp_path, stored)
    assert group == "workspace"


def test_deliberately_moved_events_stays_put(tmp_path, chromium_browser):
    """An operator who moved Events into Tools keeps it there."""
    group, _ = _group_of_events(chromium_browser, tmp_path, _stored({"events": "tools"}))
    assert group == "tools"
