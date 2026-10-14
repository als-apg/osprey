"""pyAML writes only inside a ``run_tool`` call; everywhere else they are refused.

``execute`` holds the target's run lock and nothing more, so a pyAML set inside it
is refused before anything is read or written. The same set inside ``run_tool``
has its prior value recorded, then is written. With no guarded-run directory the
run stops before the tool is called.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pyaml.accelerator import Accelerator

import osprey.runtime
from osprey.runtime.guarded_run import GuardedRunDirError, lock
from pyaml_cs_osprey import _run
from pyaml_cs_osprey.errors import OspreyReadFailed, OspreyWriteRefused
from pyaml_cs_osprey.run_tool import run_tool
from tests.pyaml_cs_osprey.conftest import DictConnector

MAGNET = "QF_001"
SETPOINT = "QF_001:Cm:set"
READBACK = "QF_001:Cm:rdbk"

CONFIG: dict[str, Any] = {
    "type": "pyaml.accelerator",
    "facility": "Test",
    "machine": "sr",
    "energy": 1.0e9,
    "controls": [{"type": "pyaml_cs_osprey.controlsystem", "name": "live"}],
    "devices": [
        {
            "type": "pyaml.magnet.quadrupole",
            "name": MAGNET,
            "model": {
                "type": "pyaml.magnet.identity_model",
                "physics": f"({READBACK}, {SETPOINT})[1/m]",
                "unit": "1/m",
            },
        }
    ],
}


class _CountingConnector(DictConnector):
    """A dict connector that records every read and every put it serves."""

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.reads: list[str] = []
        self.puts: list[tuple[str, Any]] = []
        self.read_error: Exception | None = None

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        self.reads.append(channel_address)
        if self.read_error is not None:
            raise self.read_error
        return await super().read_channel(channel_address, timeout)

    def _put(self, channel_address: str, value: Any) -> None:
        self.puts.append((channel_address, value))
        super()._put(channel_address, value)


@pytest.fixture(scope="module")
def sr() -> Accelerator:
    return Accelerator.from_dict(copy.deepcopy(CONFIG))


@pytest.fixture
def connector(
    monkeypatch: pytest.MonkeyPatch,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[_CountingConnector]:
    monkeypatch.setattr(osprey.runtime, "_limits_validator", None)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")
    machine = _CountingConnector({SETPOINT: 1.0, READBACK: 1.0})
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = machine
        yield machine


def _strength(sr: Accelerator) -> Any:
    return sr.live.magnet.get(name=MAGNET).strength


class TestExecuteIsReadOnly:
    """Under the run lock alone, as ``execute`` holds it, a pyAML set is refused."""

    def test_a_set_under_the_run_lock_is_refused_with_no_put(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        with lock("live"), pytest.raises(OspreyWriteRefused) as info:
            _strength(sr).set(1.5)

        assert info.value.reason == "pyAML writes run only inside pyaml_measure"
        assert info.value.channel_address == SETPOINT
        assert connector.puts == []
        assert connector.reads == []
        assert connector._state[SETPOINT] == 1.0

    def test_a_set_with_no_run_at_all_is_refused_the_same_way(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        with pytest.raises(OspreyWriteRefused, match="only inside pyaml_measure"):
            _strength(sr).set(1.5)
        assert connector.puts == []


class TestRunToolWrites:
    """Inside ``run_tool`` the same set has its prior value recorded, then is written."""

    def test_the_sets_first_value_is_recorded_before_it_is_written(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        recorded: list[dict[str, Any]] = []

        def put(channel_address: str, value: Any) -> None:
            # The write runs under the run's guard, so the record is read as it stands.
            assert _run._RUN is not None
            recorded.append(dict(_run._RUN.moved))
            _CountingConnector._put(connector, channel_address, value)

        connector._put = put  # type: ignore[method-assign]

        run = run_tool(lambda: _strength(sr).set(1.5))

        assert run.completed is True
        assert recorded == [{SETPOINT: 1.0}], "recorded before the put"
        assert run.moved == {SETPOINT: 1.0}
        assert connector.puts == [(SETPOINT, 1.5)]

    def test_no_guarded_run_directory_stops_the_run_before_the_tool(
        self,
        sr: Accelerator,
        connector: _CountingConnector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        called: list[str] = []

        def tool() -> None:
            called.append("ran")
            _strength(sr).set(1.5)

        with pytest.raises(GuardedRunDirError, match="guarded runs need var/guarded_run"):
            run_tool(tool)

        assert called == []
        assert connector.puts == []

    def test_a_failed_pre_write_read_is_a_read_failure_with_no_put(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        connector.read_error = TimeoutError("timed out")

        with lock("live"), _run.opened("live"), pytest.raises(OspreyReadFailed) as info:
            _strength(sr).set(1.5)

        assert SETPOINT in str(info.value)
        assert connector.puts == []
