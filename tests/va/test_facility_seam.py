"""The facility-seam regression gate.

Two halves, matching the seam's two promises:

* **The historical path is unchanged.** With none of the source env vars
  set, the entrypoint resolves to exactly the pre-seam behaviour: built-in
  generated manifest, PyAT lattice, bundled drive-limit/boot-value data.
  (The deep guarantee is the rest of ``tests/va`` -- every pre-seam test
  still runs against the default path -- so this half only pins the
  resolution itself.)

* **A file-backed facility boots without PyAT.** A manifest of three-part
  addresses (identity carried in the hierarchy keys, not the address text)
  boots the *real* ``entrypoint.main()`` -- records, engine source, iocInit,
  serving announcement -- in a subprocess whose import machinery makes any
  ``import at`` fatal. A change that sneaks a PyAT dependency onto the
  no-lattice path turns that subprocess boot into a crash, and this test
  red.

The subprocess split follows ``test_record_factory.py``: softioc state is
process-global and ``iocInit`` is once-per-process, so the boot runs in its
own dedicated process, never pytest's.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_seam_ioc_subprocess() -> None:
    """Subprocess entry point: make PyAT unimportable, then boot main().

    The blocker sits at the front of ``sys.meta_path``, so even an installed
    PyAT cannot be imported in this process -- equivalent to (and stricter
    than) running on a machine without PyAT.
    """
    import importlib.abc

    class ATBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "at" or fullname.startswith("at."):
                raise ImportError("SEAM VIOLATION: PyAT imported on the no-lattice path")
            return None

    sys.meta_path.insert(0, ATBlocker())

    from osprey.services.virtual_accelerator import entrypoint

    entrypoint.main()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--run-seam-ioc-subprocess":
    _run_seam_ioc_subprocess()
    sys.exit(0)


from osprey.services.virtual_accelerator import entrypoint  # noqa: E402
from osprey.services.virtual_accelerator.manifest import (  # noqa: E402
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_LONG_STRING,
)

# A three-part-address facility: the address text carries no six-level
# grammar; identity (SP<->RB pairing) rides entirely in the hierarchy keys.
_SEAM_CHANNELS = [
    {
        "address": "ZZSEAM:JET:PRESSURE",
        "ring": "ZZSEAM",
        "system": "JET",
        "family": "TARGET",
        "device": "01",
        "field": "PRESSURE",
        "subfield": "RB",
        "partition": PARTITION_STATIC_NOISY,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": True,
    },
    {
        "address": "ZZSEAM:STAGE:POS:SP",
        "ring": "ZZSEAM",
        "system": "STAGE",
        "family": "MOTOR",
        "device": "01",
        "field": "POS",
        "subfield": "SP",
        "partition": PARTITION_SP_ECHO,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    },
    {
        "address": "ZZSEAM:STAGE:POS",
        "ring": "ZZSEAM",
        "system": "STAGE",
        "family": "MOTOR",
        "device": "01",
        "field": "POS",
        "subfield": "RB",
        "partition": PARTITION_SP_ECHO,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    },
    {
        "address": "ZZSEAM:STATUS:MSG",
        "ring": "ZZSEAM",
        "system": "STATUS",
        "family": "MSG",
        "device": "01",
        "field": "TEXT",
        "subfield": "RB",
        "partition": PARTITION_STATIC_NOISY,
        "record_type": RECORD_TYPE_LONG_STRING,
        "noise": False,
    },
]


class TestDefaultResolutionIsThePreSeamPath:
    """ALS half: no env vars -> built-in manifest + built-in lattice."""

    def test_no_env_resolves_to_builtin_manifest_and_lattice(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VA_CHANNELS_FILE", raising=False)
        monkeypatch.delenv("VA_LATTICE", raising=False)
        channels_file = entrypoint._resolve_channels_file(tmp_path)
        assert channels_file is None
        assert entrypoint._resolve_lattice_mode(channels_file) == entrypoint.LATTICE_BUILTIN


class TestSetpointEchoEngineSync:
    """No-lattice physics coupling: sp-echo readback values are synced into
    the engine each tick, so a machine-file expression channel (the camera
    response) follows the latest accepted setpoint."""

    class _FakeRecord:
        def __init__(self, value: float) -> None:
            self._value = value

        def get(self) -> float:
            return self._value

        def set(self, value: float) -> None:
            self._value = value

    def test_expression_channel_follows_synced_setpoint(self, tmp_path: Path):
        from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
        from osprey.simulation.engine import SimulationEngine

        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps(
                {
                    "name": "sync-test",
                    "description": "engine-sync unit machine",
                    "channels": {
                        "ZZSYNC:STAGE:POS": {"value": 0.0, "noise": 0},
                        "ZZSYNC:CAM:INTENSITY": {
                            "expr": "1000 * exp(-((ch('ZZSYNC:STAGE:POS') - 2.0) ** 2))",
                            "noise": 0,
                        },
                    },
                }
            )
        )
        engine = SimulationEngine.from_file(machine)
        stage_rb = self._FakeRecord(0.0)
        intensity = self._FakeRecord(0.0)
        channels = [
            {
                "address": "ZZSYNC:CAM:INTENSITY",
                "partition": PARTITION_STATIC_NOISY,
                "record_type": RECORD_TYPE_ANALOG,
                "noise": False,
            }
        ]
        source = EngineSource(
            engine,
            channels,
            {"ZZSYNC:CAM:INTENSITY": intensity},
            tmp_path,
            setpoint_echo_records={
                "ZZSYNC:STAGE:POS": stage_rb,
                "ZZSYNC:NOT:SERVED": self._FakeRecord(1.0),  # dropped, engine-unknown
            },
        )

        source.poll_once()
        off_peak = intensity.get()

        stage_rb.set(2.0)  # the accepted setpoint echo moves the readback...
        source.poll_once()
        on_peak = intensity.get()

        # ...and the expression channel responds: on-peak beats off-peak by
        # the Gaussian's e^{-4} contrast.
        assert on_peak == pytest.approx(1000.0)
        assert off_peak == pytest.approx(1000.0 * 0.0183156, rel=1e-3)

    def test_without_sync_map_behaviour_is_unchanged(self, tmp_path: Path):
        from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
        from osprey.simulation.engine import SimulationEngine

        machine = tmp_path / "machine.json"
        machine.write_text(
            json.dumps(
                {
                    "name": "sync-default-test",
                    "description": "no sync map -> pure scenario source",
                    "channels": {"ZZSYNC2:TEMP:RB": {"value": 21.5, "noise": 0}},
                }
            )
        )
        engine = SimulationEngine.from_file(machine)
        temp = self._FakeRecord(0.0)
        channels = [
            {
                "address": "ZZSYNC2:TEMP:RB",
                "partition": PARTITION_STATIC_NOISY,
                "record_type": RECORD_TYPE_ANALOG,
                "noise": False,
            }
        ]
        EngineSource(engine, channels, {"ZZSYNC2:TEMP:RB": temp}, tmp_path).poll_once()
        assert temp.get() == pytest.approx(21.5)


class TestFileBackedBootWithoutPyat:
    """Facility half: real main() boot, file manifest, PyAT import fatal."""

    @pytest.fixture()
    def seam_data_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "channels_manifest.json").write_text(json.dumps({"channels": _SEAM_CHANNELS}))
        (tmp_path / "machine.json").write_text(
            json.dumps(
                {
                    "name": "seam-test facility",
                    "description": "minimal machine for the facility-seam boot test",
                    "channels": {
                        "ZZSEAM:JET:PRESSURE": {
                            "label": "jet backing pressure",
                            "value": 30.0,
                        }
                    },
                }
            )
        )
        return tmp_path

    def test_boots_serves_and_never_imports_pyat(self, seam_data_dir: Path):
        env = dict(os.environ)
        env.update(
            VA_DATA_DIR=str(seam_data_dir),
            VA_CHANNELS_FILE="channels_manifest.json",
            # VA_LATTICE deliberately unset: file-backed default must be "none".
            EPICS_CA_ADDR_LIST="127.0.0.1",
            EPICS_CA_AUTO_ADDR_LIST="NO",
            EPICS_CA_SERVER_PORT=str(_free_port()),
            EPICS_CA_REPEATER_PORT=str(_free_port()),
        )
        env.pop("VA_LATTICE", None)

        proc = subprocess.Popen(
            [sys.executable, __file__, "--run-seam-ioc-subprocess"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            lines: list[str] = []
            deadline = time.monotonic() + 60
            served = False
            while time.monotonic() < deadline:
                line = proc.stdout.readline()
                if not line:
                    break  # process exited (a crash -- asserted below)
                lines.append(line)
                if "serving PVs" in line:
                    served = True
                    break
            output = "".join(lines)
            assert served, f"IOC never reached the serving announcement:\n{output}"
            # The no-lattice notice proves the PhysicsBridge branch was
            # skipped by DEFAULT (VA_LATTICE unset), not by explicit opt-out.
            assert "PhysicsBridge skipped" in output
            # All four manifest channels (and nothing else) were built.
            assert f"serving PVs: {len(_SEAM_CHANNELS)} channels" in output
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
