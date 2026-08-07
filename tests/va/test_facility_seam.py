"""The facility-seam regression gate.

The seam is one sentence: **no accelerator-physics imports on the no-lattice
boot path.** ``VA_LATTICE=none`` describes a facility whose channels a
manifest lists and whose physics this process does not have, and the promise
that makes that mode worth having is that such a facility can boot the
virtual accelerator on a host where the physics stack is not installed at
all. So the guard is drawn around the physics, and only the physics: ``at``
(PyAT itself) and ``lume_pyat`` (the ring model built on it). ``lume_pyat``
has to be named separately because its ``__init__`` is PEP 562 lazy -- an
accidental import costs almost nothing and would not announce itself by
dragging anything heavy in.

``lume`` and ``h5py`` are **deliberately not blocked**, and that is a change
of position rather than a relaxation. The serving layer is built around a
:class:`~lume.model.LUMEModel`: the runner derives its PVA namespace and the
shape of its run loop from one, and the no-lattice boot serves the empty
:class:`~osprey.services.virtual_accelerator.serving.model_stub.NullModel`
rather than no model at all. ``lume`` is therefore a dependency of the
*serving* layer, not of the physics behind it, and blocking it would have
this gate assert something the design no longer claims. What the seam
protects is what this boot path is allowed to *depend on* -- not what its
dependencies cost.

It does cost, and the cost is worth naming here rather than discovering
later. Timing a ``VA_LATTICE=none`` boot from process spawn to the ``Loading
simulation engine`` line -- the last milestone both the pre-serving and the
serving entrypoint print, so the interval spans exactly the assembly this
seam governs and none of the transport behind it -- on darwin/arm64,
CPython 3.13, eleven paired runs interleaved between the two trees with one
discarded warm-up each, median of each arm:

    pre-serving entrypoint (no ``lume`` imported)    0.518 s
    serving entrypoint (``lume`` for NullModel)      1.216 s
    delta                                           +0.698 s

Nearly all of it is the single import: ``import lume.model`` alone costs
+0.916 s over a bare interpreter on the same host (median of seven), and the
boot pays less than that only because it shares the numpy that import needs
with code it was already loading. Two thirds of a second is a real number on
a boot whose remaining work is standing up a Channel Access server, and it
is not a regression to chase: it is what serving a LUMEModel costs, and this
mode serves one by design.

Two halves, matching the seam's two promises:

* **The historical path is unchanged.** With none of the source env vars
  set, the entrypoint resolves to exactly the pre-seam behaviour: built-in
  generated manifest, PyAT lattice, bundled drive-limit/boot-value data.
  (The deep guarantee is the rest of ``tests/va`` -- every pre-seam test
  still runs against the default path -- so this half only pins the
  resolution itself.)

* **A file-backed facility boots without PyAT.** A manifest of three-part
  addresses (identity carried in the hierarchy keys, not the address text)
  boots the *real* ``entrypoint.main()`` in a subprocess whose import
  machinery makes any ``import at`` fatal. A change that sneaks a physics
  dependency onto the no-lattice path turns that boot into a crash, and this
  test red.

The boot runs in a subprocess of its own because ``main()`` is a process
entry point in the literal sense: it installs SIGINT/SIGTERM handlers, binds
a Channel Access port and then blocks in the runner's loop until signalled.
None of that belongs in pytest's process.

**How far the boot gets depends on the host.** ``serving.runner`` imports the
Channel Access server extension, and a host without it cannot reach the
serving announcement no matter how clean the seam is. So the assertions are
split by what each proves: the seam itself and the whole no-lattice assembly
ahead of the transport are checked unconditionally, and the terminal outcome
is required to be either the serving announcement or a missing *server*
module -- never a physics one, and never anything else. On a host with the
extension this test proves the boot end to end; on one without it, it still
proves everything the seam is about.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osprey.interfaces._serving import free_port

# Every root the no-lattice path must stay clear of: PyAT, and the ring model
# built on it. See the module docstring for why ``lume`` and ``h5py`` are not
# here, and why ``lume_pyat`` has to be named rather than left to arrive with
# the heavy stack.
_BLOCKED_ROOTS = ("at", "lume_pyat")

# The marker the subprocess raises with, and the string the test greps its
# output for. Distinctive enough that no library's own ImportError can be
# mistaken for it.
_VIOLATION = "SEAM VIOLATION"

# Roots whose absence stops the boot at the transport rather than at the
# seam. ``serving.runner`` imports the Channel Access server extension at
# module scope, so a host without these gets as far as the runner import and
# no further -- a legitimate outcome for this test, and the only failure
# other than a full boot it accepts.
_SERVER_EXTENSION_ROOTS = ("pcaspy", "p4p", "lume_pva_apg")


def _run_seam_ioc_subprocess() -> None:
    """Subprocess entry point: make the physics unimportable, then boot main().

    The blocker sits at the front of ``sys.meta_path``, so even an installed
    PyAT cannot be imported in this process -- equivalent to (and stricter
    than) running on a machine without it.
    """
    import importlib.abc

    class ATBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root in _BLOCKED_ROOTS:
                raise ImportError(
                    f"{_VIOLATION}: {root} imported on the no-lattice path (via {fullname!r})"
                )
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

    # Assembly milestones the no-lattice boot must pass on its way to the
    # transport. Each is printed by ``main()`` itself, and together they
    # cover every decision the seam is about: the lattice branch not taken,
    # the manifest turned into a serving database, the scenario engine
    # loaded. Reaching all three with the physics unimportable IS the seam
    # holding, whether or not the host can stand a server up afterwards.
    MILESTONES = (
        "PhysicsBridge skipped",
        f"Built serving database: {len(_SEAM_CHANNELS)} channels",
        "Loading simulation engine from",
    )

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
        # Inherit the ambient environment MINUS the whole VA_ family, so this
        # boot is configured by exactly what is set below. An inherited
        # VA_BPM_ERRORS or VA_CORR_GAIN is a lattice-physics fault, and the
        # entrypoint refuses to serve one unless VA_LATTICE='builtin' -- which
        # a file-backed boot deliberately never sets.
        env = {k: v for k, v in os.environ.items() if not k.startswith("VA_")}
        env.update(
            VA_DATA_DIR=str(seam_data_dir),
            VA_CHANNELS_FILE="channels_manifest.json",
            # VA_LATTICE deliberately unset: file-backed default must be "none".
            EPICS_CA_ADDR_LIST="127.0.0.1",
            EPICS_CA_AUTO_ADDR_LIST="NO",
            EPICS_CA_SERVER_PORT=str(free_port()),
            EPICS_CA_REPEATER_PORT=str(free_port()),
        )

        served, output, returncode = self._boot(env)

        # The seam, and the only assertion here that is about the seam: with
        # the physics unimportable, nothing on this path reached for it. This
        # holds whether the boot completed or stopped at the transport, which
        # is why it is checked before anything else is.
        assert _VIOLATION not in output, f"the no-lattice boot imported physics:\n{output}"

        # ...and it got far enough for that to mean something. A process that
        # died before the lattice branch would satisfy the check above
        # vacuously.
        for milestone in self.MILESTONES:
            assert milestone in output, f"boot never reached {milestone!r}:\n{output}"

        if served:
            # A host with the Channel Access server extension: the boot ran to
            # completion and is serving the manifest whole, four channels and
            # nothing else.
            assert entrypoint._ready_line(len(_SEAM_CHANNELS)) in output
            assert returncode is None, f"the IOC exited instead of serving:\n{output}"
            return

        # A host without it. The boot is required to have stopped for exactly
        # that reason -- a missing *server* module -- and for no other. Any
        # other exception here is a real failure of the no-lattice path,
        # including a physics import that somehow evaded the blocker.
        assert returncode is not None, (
            f"the boot neither served nor exited within the deadline -- it hung:\n{output}"
        )
        missing = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", output)
        assert missing, (
            "the boot stopped short of serving without a missing-module error, "
            f"so something other than the absent server extension ended it:\n{output}"
        )
        root = missing.group(1).split(".")[0]
        assert root in _SERVER_EXTENSION_ROOTS, (
            f"the no-lattice boot stopped on a missing {root!r}, which is not one of the "
            f"server extensions {list(_SERVER_EXTENSION_ROOTS)} this host is allowed to lack"
        )

    def _boot(self, env: dict[str, str]) -> tuple[bool, str, int | None]:
        """Boot ``main()`` in a subprocess and drain it.

        Returns whether the serving announcement was reached, everything the
        process wrote (stderr merged in, so a traceback is part of the
        record), and its exit status -- ``None`` while it is still running,
        which is what a boot that reached the announcement looks like.

        Draining runs to the announcement or to EOF, never on a timer alone:
        the announcement means success and EOF means the process ended, and
        the deadline exists only so a boot that hangs without doing either
        fails this test rather than pytest's whole run.
        """
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
                    break  # process exited; its status says why
                lines.append(line)
                if entrypoint.READY_MARKER in line:
                    served = True
                    break
            if not served:
                # stdout hit EOF, or the deadline expired. Wait for the exit
                # status rather than sampling it: the pipe closes a moment
                # before the process is reapable, so ``poll()`` on its own
                # would report a crashed boot as still running. A timeout
                # here means it really is still running and silent -- a hang,
                # which the caller reports as one.
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            return served, "".join(lines), proc.poll()
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
