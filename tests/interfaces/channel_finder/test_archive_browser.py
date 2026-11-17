"""Browser test: the Channels panel's Archive view on the demo.

The build and the served panel are the All PVs browser test's: the demo preset
with the simulator's section, whose archive is the deployment's own MongoDB
store on ``va``. The build deploys no store, so the archiver connector the
panel builds is replaced by one whose store records the facility file's
channels: the directory is the simulator's, the join and every word on the
page are the panel's own.

Run:
    .venv/bin/pytest tests/interfaces/channel_finder/test_archive_browser.py -m browser -v
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import yaml

pytest.importorskip("osprey_connectors.service_answers")

from osprey.channel_roster import registered_channels
from osprey_connectors.factory import ConnectorFactory
from tests.interfaces.channel_finder.test_all_pvs_browser import (
    _open,
    _tabs,
    panel_server,  # noqa: F401  (fixture)
    panel_url,  # noqa: F401  (fixture)
    render,  # noqa: F401  (fixture)
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]


class _Store:
    """The deployment's store, recording *channels*."""

    def __init__(self, channels: frozenset[str]) -> None:
        self._channels = channels

    async def store_extent(self) -> Any:
        return SimpleNamespace(channels=self._channels, oldest=None, newest=None, reason=None)

    async def get_metadata(self, channel: str) -> Any:
        return SimpleNamespace(
            is_archived=channel in self._channels,
            archival_start=None,
            archival_end=None,
            sampling_period=None,
            description=None,
        )

    async def disconnect(self) -> None:
        pass


@pytest.fixture(scope="module", autouse=True)
def store(render: Path) -> Iterator[frozenset[str]]:  # noqa: F811
    """The facility file's channels, recorded by the store the panel reads."""
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    config["config_dir"] = str(render)
    channels = frozenset(record.address for record in registered_channels(config).records)
    assert channels

    async def create(_config: Any) -> _Store:
        return _Store(channels)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ConnectorFactory, "create_archiver_connector", create)
        yield channels


def test_the_archive_tab_shows_the_archived_count_and_the_configuration(
    chromium_browser: Browser,
    panel_url: str,  # noqa: F811
) -> None:
    host = _open(chromium_browser, panel_url, "#/archive")
    assert _tabs(host) == ["Curated", "All PVs", "Alarms", "Archive"]
    host.wait_for_selector('.archive-tile[data-tile="archived"]')

    answer = json.loads(host.evaluate("() => fetch('/api/directory/archive').then(r => r.text())"))
    archived = answer["archived"]
    assert archived["count"] > 0
    assert archived["count"] < archived["of"]

    tile = host.locator('.archive-tile[data-tile="archived"]')
    assert tile.locator(".archive-tile-value").inner_text() == f"{archived['count']:,}"
    left = archived["of"] - archived["count"]
    assert tile.locator(".archive-tile-sub").inner_text() == (
        f"of {archived['of']:,} · {left:,} not archived"
    )

    config = host.locator('[data-card="configuration"] pre').inner_text()
    assert "archiver:\n  type: mongodb_archiver" in config
    assert "va_archiver:" in config
    assert "password" not in config
