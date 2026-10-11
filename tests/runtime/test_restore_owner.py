"""The runtime owns the putting-back of a guarded run.

``journaled_run`` is the one place a run's displaced setpoints are written back:
an address is left alone only when it reads exactly its journaled value. Channels
are a dict served through ``osprey.runtime``; no control system is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import JOURNAL_FILE_NAME, guarded_run_dir, journaled_run
from osprey.runtime.journal import guarded_write
from osprey_connectors.workspace import reset_config_cache

#: The RF frequency the restore cases start from, in Hz.
F0 = 500.0e6


class _Channels:
    """A dict of channel values; every write is recorded and lands."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, Any]] = []

    def read_channels(self, addresses: list[str], **_kwargs: Any) -> list[Any]:
        return [self.values[a] for a in addresses]

    def write_channel(self, address: str, value: Any, **_kwargs: Any) -> None:
        self.writes.append((address, value))
        self.values[address] = value

    def channel_limits(self, _address: str) -> None:
        return None


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    config = {"control_system": {"type": "epics", "connector": {"epics": {}}}}
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.setenv(ENV_CONTROL_TARGET_GENERATION, "1")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.setattr(guarded_run, "_APPROVED", None)
    yield root
    reset_config_cache()


@pytest.fixture
def channels(monkeypatch: pytest.MonkeyPatch) -> _Channels:
    """``RF:freq`` at :data:`F0` and ``Q`` at 3.0."""
    fake = _Channels({"RF:freq": F0, "Q": 3.0})
    for name in ("read_channels", "write_channel", "channel_limits"):
        monkeypatch.setattr(osprey.runtime, name, getattr(fake, name))
    return fake


def _journal_path() -> Path:
    return guarded_run_dir("live") / JOURNAL_FILE_NAME


def _put(address: str, value: Any) -> None:
    """A guarded write of ``value`` to ``address``."""
    guarded_write([address], lambda: osprey.runtime.write_channel(address, value), lambda e: e)


class TestExactUnchanged:
    """An address is left alone only when it reads exactly its journaled value."""

    def test_a_100_hz_step_on_500_mhz_is_restored(self, channels: _Channels) -> None:
        with pytest.raises(ValueError, match="stop") as caught, journaled_run("live"):
            _put("RF:freq", F0 + 100)
            raise ValueError("stop")

        report = caught.value.restore_report
        assert report.restored == ["RF:freq"]
        assert report.unchanged == []
        assert channels.writes == [("RF:freq", F0 + 100), ("RF:freq", F0)]
        assert channels.values["RF:freq"] == F0

    def test_an_address_at_exactly_its_journaled_value_is_unchanged(
        self, channels: _Channels
    ) -> None:
        with pytest.raises(ValueError, match="stop") as caught, journaled_run("live"):
            _put("RF:freq", F0 + 100)
            channels.values["RF:freq"] = F0
            raise ValueError("stop")

        report = caught.value.restore_report
        assert report.unchanged == ["RF:freq"]
        assert report.restored == []
        assert channels.writes == [("RF:freq", F0 + 100)]
