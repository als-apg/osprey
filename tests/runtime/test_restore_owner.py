"""The runtime owns the putting-back of a guarded run.

``journaled_run`` is the one place a run's displaced setpoints are written back:
an address is left alone only when it reads exactly its journaled value. Channels
are a dict served through ``osprey.runtime``; no control system is involved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import osprey.runtime
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime import ENV_CONTROL_TARGET, ENV_CONTROL_TARGET_GENERATION
from osprey.runtime.guarded_run import JOURNAL_FILE_NAME, guarded_run_dir, journaled_run
from osprey.runtime.journal import OspreyWriteRefused, guarded_write, read_pending_journal
from osprey_connectors.dotenv import atomic_replace
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


class TestNoUndoValue:
    """A write whose pre-read could never be written back is refused before it happens."""

    @pytest.mark.parametrize(
        ("reading", "shown"),
        [
            (float("nan"), "nan"),
            (float("inf"), "inf"),
            (float("-inf"), "-inf"),
            ([1.0, float("nan")], "nan"),
        ],
    )
    def test_a_non_finite_pre_read_is_refused_and_nothing_is_journaled(
        self, channels: _Channels, reading: Any, shown: str
    ) -> None:
        channels.values["Q"] = reading
        with journaled_run("live"):
            with pytest.raises(OspreyWriteRefused) as caught:
                _put("Q", 4.0)
            lines = _journal_path().read_text(encoding="utf-8").splitlines()

        assert caught.value.channel_address == "Q"
        assert str(caught.value) == (
            f"OSPREY refused the write to 'Q': reads {shown} before the write, "
            "so a guarded run could not put it back"
        )
        assert [list(json.loads(line)) for line in lines] == [["header"]]
        assert read_pending_journal(_journal_path()) is None
        assert channels.writes == []

    def test_a_numpy_array_holding_nan_is_refused(self, channels: _Channels) -> None:
        channels.values["Q"] = np.array([1.0, np.nan])
        with journaled_run("live"), pytest.raises(OspreyWriteRefused, match="reads nan"):
            _put("Q", 4.0)
        assert channels.writes == []

    def test_a_finite_pre_read_journals_and_writes(self, channels: _Channels) -> None:
        with journaled_run("live"):
            _put("Q", 4.0)
            pending = read_pending_journal(_journal_path())

        assert pending is not None and pending.values == {"Q": 3.0}
        assert channels.writes == [("Q", 4.0)]


class TestAtomicReplace:
    """The journal rewrite and the executor's restore record replace a file one way."""

    def test_a_link_planted_at_every_predictable_name_is_never_followed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.journal"
        path.write_bytes(b"old\n")
        victim = tmp_path / "victim"
        victim.write_text("untouched", encoding="utf-8")
        planted = [
            path.with_name(name)
            for name in (
                f"{path.name}.tmp",
                f".{path.name}.tmp",
                f".{path.name}.{os.getpid()}.tmp",
                f"{path.name}.{os.getpid()}.tmp",
            )
        ]
        for link in planted:
            link.symlink_to(victim)

        atomic_replace(path, b"new\n", mode=0o664)

        assert path.read_bytes() == b"new\n"
        assert not path.is_symlink()
        assert victim.read_text(encoding="utf-8") == "untouched"
        assert all(link.is_symlink() for link in planted)
        leftovers = sorted(p for p in path.parent.iterdir() if p.name.endswith(".tmp"))
        assert leftovers == sorted(planted)

    def test_a_failing_replace_leaves_no_temporary_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "execution"
        folder.mkdir()
        path = folder / "restore.json"
        path.write_bytes(b"old")

        def refuse(*_args: Any) -> None:
            raise OSError("replace refused")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(OSError, match="replace refused"):
            atomic_replace(path, b"new", mode=0o644)

        assert path.read_bytes() == b"old"
        assert [p.name for p in folder.iterdir()] == ["restore.json"]

    @pytest.mark.parametrize("mode", [0o664, 0o644, 0o600])
    def test_the_result_has_the_mode_asked_for_whatever_the_umask(
        self, tmp_path: Path, mode: int
    ) -> None:
        path = tmp_path / "file"
        previous = os.umask(0o077)
        try:
            atomic_replace(path, b"data", mode=mode)
        finally:
            os.umask(previous)

        assert path.read_bytes() == b"data"
        assert path.stat().st_mode & 0o777 == mode
