"""How the virtual accelerator is assembled, and what a boot with no physics serves.

Three things that only exist once the pieces are put together are proven here,
and nothing else: what the entrypoint hands the runner and in what order, what
the model stub a lattice-free boot serves does, and the two orderings a write
depends on that no single component can establish alone.

**Nothing here binds a port or creates a server.** The runner is the one module
in the assembly that imports the Channel Access server extension, and this host
has no working build of it, so the entrypoint is driven against a fake runner
injected in its place -- which is also what makes the boot order assertable as a
sequence rather than inferred from a live process's side effects. Live Channel
Access behaviour is proven against the deployed container, not here.

The write-path halves (:class:`TestHookBeforeEcho`,
:class:`TestPerWriteIsolation`) drive the real write path against a recording
driver, as the sibling suite does. What they add to it is the two things that
need more than one write, or more than the driver, to be visible: where the
physics hook falls relative to the values it produces, and that two writes in
flight at once stay two writes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable

from osprey.services.virtual_accelerator import entrypoint as ep
from osprey.services.virtual_accelerator.ioc import engine_source as engine_source_module
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
)
from osprey.services.virtual_accelerator.serving.model_stub import NullModel
from osprey.services.virtual_accelerator.serving.pvdb import (
    ServingRecords,
    build_serving_pvdb,
)
from osprey.services.virtual_accelerator.serving.write_path import (
    CohostWritePath,
    SetpointRoutedModel,
    physics_setpoint_addresses,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
# A floor rather than an equality: an added test is not a regression, and an
# exact count would make every future addition edit this line.
MIN_COLLECTED_TESTS = 45

#: The runner module the entrypoint imports last, and the one that reaches the
#: Channel Access server extension. Replaced wholesale in ``sys.modules`` for
#: the boot tests; see :func:`_fake_runner_module`.
RUNNER_MODULE = "osprey.services.virtual_accelerator.serving.runner"

# --- a facility, small enough to assert about in full ---------------------

RING = "LAB"

MAG1_SP = f"{RING}:MAG:HCM:01:CURRENT:SP"
MAG1_RB = f"{RING}:MAG:HCM:01:CURRENT:RB"
MAG2_SP = f"{RING}:MAG:HCM:02:CURRENT:SP"
MAG2_RB = f"{RING}:MAG:HCM:02:CURRENT:RB"
BPM_X = f"{RING}:BPM:BPM:01:X:RB"
#: The elements the deck carries those devices at. Every device has two
#: spellings -- one of these and an address above -- and which of them a
#: facility's people use to name it is theirs to decide.
BPM_ELEMENT = "BPM1"
MAG1_ELEMENT = "M1"
MAG2_ELEMENT = "M2"
VALVE_SP = f"{RING}:VAC:VALVE:01:POSITION:SP"
VALVE_RB = f"{RING}:VAC:VALVE:01:POSITION:RB"
GAUGE_RB = f"{RING}:VAC:GAUGE:01:PRESSURE:RB"

MAG_BAND = (-5.0, 5.0)
DRIVE_LIMITS = {MAG1_SP: MAG_BAND, MAG2_SP: MAG_BAND}
BOOT_VALUES = {MAG1_SP: 1.0, MAG1_RB: 1.0, MAG2_SP: 2.0, MAG2_RB: 2.0}

#: The file name a served tree's lattice carries, and so the only value
#: ``VA_LATTICE`` can take other than ``none``: the bindings beside it were
#: derived against that file, and it is the one the model reads.
SERVED_LATTICE = "lattice.json"

#: Every VA_* variable the entrypoint reads. Cleared before each test: a
#: developer's shell that happens to export one would otherwise change what
#: these tests assemble.
VA_ENV_VARS = (
    "VA_DATA_DIR",
    "VA_CHANNELS_FILE",
    "VA_LATTICE",
    "VA_STUCK_SETPOINTS",
    "VA_BPM_ERRORS",
    "VA_CORR_GAIN",
)


def _channel(
    address: str,
    *,
    subfield: str,
    partition: str,
    system: str,
    family: str,
    device: str,
    field_name: str,
    noise: float = 0.0,
) -> dict[str, Any]:
    """One manifest channel, carrying the full schema a file source must."""
    return {
        "address": address,
        "ring": RING,
        "system": system,
        "family": family,
        "device": device,
        "field": field_name,
        "subfield": subfield,
        "partition": partition,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": noise,
    }


def _magnet(device: str, setpoint: str, readback: str) -> list[dict[str, Any]]:
    return [
        _channel(
            setpoint,
            subfield="SP",
            partition=PARTITION_PYAT_COUPLED,
            system="MAG",
            family="HCM",
            device=device,
            field_name="CURRENT",
        ),
        _channel(
            readback,
            subfield="RB",
            partition=PARTITION_PYAT_COUPLED,
            system="MAG",
            family="HCM",
            device=device,
            field_name="CURRENT",
        ),
    ]


CHANNELS: list[dict[str, Any]] = [
    *_magnet("01", MAG1_SP, MAG1_RB),
    *_magnet("02", MAG2_SP, MAG2_RB),
    _channel(
        BPM_X,
        subfield="RB",
        partition=PARTITION_PYAT_COUPLED,
        system="BPM",
        family="BPM",
        device="01",
        field_name="X",
    ),
    _channel(
        VALVE_SP,
        subfield="SP",
        partition=PARTITION_SP_ECHO,
        system="VAC",
        family="VALVE",
        device="01",
        field_name="POSITION",
    ),
    _channel(
        VALVE_RB,
        subfield="RB",
        partition=PARTITION_SP_ECHO,
        system="VAC",
        family="VALVE",
        device="01",
        field_name="POSITION",
    ),
    _channel(
        GAUGE_RB,
        subfield="RB",
        partition=PARTITION_STATIC_NOISY,
        system="VAC",
        family="GAUGE",
        device="01",
        field_name="PRESSURE",
        noise=0.03,
    ),
]

PHYSICS_SETPOINTS = frozenset({MAG1_SP, MAG2_SP})


@pytest.fixture(autouse=True)
def _clean_va_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _identity_binding(setpoint: str, readback: str, element: str) -> dict[str, Any]:
    """One writable binding that serves the value written on its readback.

    The rules a document is parsed against live in the schema's own suite;
    what matters here is a document this facility's coupled setpoints are
    bound by, with a readback rule that needs no calibration curve -- so the
    boot tests exercise the wiring and not a conversion.
    """
    return {
        "kind": "strength",
        "family": "hcm",
        "setpoint_address": setpoint,
        "readback_address": readback,
        "readback": "identity",
        "element": element,
        "attribute": "PolynomB",
        "index": 1,
        "slices": [{"element": element, "weight": 1.0}],
        "owner": "hcm",
        "calibration": {"kind": "linear", "gain": 0.01, "offset": 0.0},
        "monitor_inverse": None,
        "nominal": 1.0,
        "energy_scaling": "brho",
        "energy_table": None,
    }


def _monitor_binding(address: str, element: str) -> dict[str, Any]:
    """One orbit reading, served on its own address.

    A monitor is read only and names no second address, so what it adds to a
    document is the pair a seeded readout error is resolved through: the
    address the reading is published on, and the element it is read at.
    """
    return {
        "kind": "monitor",
        "family": "bpm",
        "setpoint_address": address,
        "readback_address": None,
        "readback": "inverse",
        "element": element,
        "attribute": "x",
        "index": None,
        "slices": [{"element": element, "weight": 1.0}],
        "owner": "bpm",
        "calibration": {"kind": "linear", "gain": 1.0e-3, "offset": 0.0},
        "monitor_inverse": {"kind": "linear", "gain": 1.0e3, "offset": 0.0},
        "nominal": None,
        "energy_scaling": "none",
        "energy_table": None,
    }


BINDINGS_DOCUMENT: dict[str, Any] = {
    "system": "TestFacility",
    "energy_gev": 3.0,
    "lattice_sha256": "0" * 64,
    "bindings": [
        _identity_binding(MAG1_SP, MAG1_RB, MAG1_ELEMENT),
        _identity_binding(MAG2_SP, MAG2_RB, MAG2_ELEMENT),
        _monitor_binding(BPM_X, BPM_ELEMENT),
    ],
}


@pytest.fixture()
def facility(tmp_path: Path) -> Path:
    """A served data directory inside the facility tree that carries it.

    The whole file-backed source in one place, so a boot test says which
    facility it is booting and nothing about how the files are shaped -- laid
    out the way a built project's is: the served directory is the tree's
    ``simulation/``, holding the manifest, the machine file, the model and the
    limits copy ``osprey build`` writes in beside them, and the tree's own
    ``channel_limits.json`` sits at the data root, which is where the model
    reads its bands from.
    """
    root = tmp_path / "data"
    served = root / "simulation"
    served.mkdir(parents=True)
    (served / "channels.json").write_text(json.dumps({"channels": CHANNELS}))
    # Never loaded: every test that serves a lattice fakes the model. What the
    # boot path asks of it is that the name VA_LATTICE gives is really there.
    (served / SERVED_LATTICE).write_text("{}")
    (served / "va_bindings.json").write_text(json.dumps(BINDINGS_DOCUMENT))
    (served / "machine.json").write_text(
        json.dumps(
            {
                "name": "test-facility",
                "description": "a facility small enough to assert about",
                "channels": {
                    MAG1_SP: {"value": BOOT_VALUES[MAG1_SP]},
                    MAG1_RB: {"value": BOOT_VALUES[MAG1_RB]},
                    MAG2_SP: {"value": BOOT_VALUES[MAG2_SP]},
                    MAG2_RB: {"value": BOOT_VALUES[MAG2_RB]},
                    GAUGE_RB: {"value": 5e-8, "noise": 0.03},
                },
            }
        )
    )
    limits = json.dumps(
        {
            "defaults": {"writable": True},
            MAG1_SP: {"min_value": MAG_BAND[0], "max_value": MAG_BAND[1]},
            MAG2_SP: {"min_value": MAG_BAND[0], "max_value": MAG_BAND[1]},
        }
    )
    # Twice, as a built tree carries it: the authored file at the data root,
    # which is what the model weighs its nominals against, and the copy the
    # build writes in beside the manifest, which is what the IOC clamps
    # writes with -- the container mounts the served directory alone.
    (root / "channel_limits.json").write_text(limits)
    (served / "channel_limits.json").write_text(limits)
    (served / "active_scenarios").write_text("[]")
    return served


# --- the assembly, driven against a fake runner ---------------------------


@dataclass
class FakeWritePath:
    """``CohostWritePath``, reduced to the reader the entrypoint hands on."""

    stuck: frozenset[str] = frozenset()

    def stuck_setpoints(self) -> frozenset[str]:
        return self.stuck


@dataclass
class FakeRunner:
    """The runner, as the entrypoint sees it: constructed, then run.

    Records what it was handed rather than serving it. Constructing the real
    one creates two servers and binds a port; what the entrypoint owes it is
    an already-built database and a model, and that is what is asserted.
    """

    model: Any
    records: Any
    kwargs: dict[str, Any]
    journal: list[tuple[str, Any]]
    interrupt: bool = False
    #: The real runner's write path, reduced to the one reader the entrypoint
    #: hands on: which setpoints are stuck right now.
    write_path: Any = field(default_factory=lambda: FakeWritePath())

    def run(self) -> None:
        self.journal.append(("run", None))
        if self.interrupt:
            # What a stop signal looks like from here: the installed handler
            # raises KeyboardInterrupt on the thread blocked in run().
            raise KeyboardInterrupt("signal 15")


@dataclass
class FakeEngineSource:
    """The telemetry source, recorded rather than started."""

    engine: Any
    channels: list[dict[str, Any]]
    static_noisy: dict[str, Any]
    data_dir: Path
    state_dir: Path | None = None
    noise_level: float | None = None
    setpoint_echo_records: dict[str, Any] | None = None


@dataclass
class FakeBridge:
    """``PhysicsBridge``, minus the lattice.

    ``bind`` is the boot-value push, and the whole reason the entrypoint's
    order is a contract: it writes into PV specs, so it must land before the
    server copies them.
    """

    journal: list[tuple[str, Any]]
    model: Any
    bound: dict[str, Any] | None = None
    setpoints: frozenset[str] = frozenset()
    stuck_reader: Any = None

    def bind(
        self, records: dict[str, Any], *, physics_setpoints: frozenset[str] = frozenset()
    ) -> None:
        self.bound = records
        self.setpoints = physics_setpoints
        self.journal.append(("bind", records))

    def on_setpoint(self, address: str, value: float) -> None:  # pragma: no cover - identity only
        """Never called here: the fake runner enqueues nothing."""

    def refresh(self, changed: Any) -> None:  # pragma: no cover - identity only
        """Never called here: nothing writes the model through the surface."""

    def follow_stuck_setpoints(self, stuck: Any) -> None:
        self.stuck_reader = stuck
        self.journal.append(("follow-stuck", stuck))


@dataclass
class Boot:
    """One assembled process, and the order it was assembled in."""

    journal: list[tuple[str, Any]] = field(default_factory=list)

    def order(self) -> list[str]:
        return [step for step, _ in self.journal]

    def one(self, step: str) -> Any:
        """The payload of the single occurrence of ``step``."""
        payloads = [payload for name, payload in self.journal if name == step]
        assert len(payloads) == 1, f"{step} happened {len(payloads)} times"
        return payloads[0]

    def at(self, step: str) -> int:
        return self.order().index(step)

    @property
    def runner(self) -> FakeRunner:
        return self.one("runner")

    @property
    def engine_source(self) -> FakeEngineSource:
        return self.one("engine-source")


def _fake_runner_module(boot: Boot, *, interrupt: bool = False) -> types.ModuleType:
    """A stand-in for the module that imports the CA server extension.

    Injected into ``sys.modules`` under the real module's name, which is what
    the entrypoint's deferred ``from ... import CohostRunner`` resolves
    against. Injection rather than an attribute patch because the real module
    cannot be imported on a host without the extension -- there is nothing to
    patch an attribute onto.
    """
    module = types.ModuleType(RUNNER_MODULE)

    def cohost_runner(model: Any, records: Any, **kwargs: Any) -> FakeRunner:
        runner = FakeRunner(
            model=model,
            records=records,
            kwargs=kwargs,
            journal=boot.journal,
            interrupt=interrupt,
        )
        boot.journal.append(("runner", runner))
        return runner

    module.CohostRunner = cohost_runner  # type: ignore[attr-defined]
    return module


def _boot(
    monkeypatch: pytest.MonkeyPatch,
    facility: Path,
    *,
    lattice: str | None = None,
    interrupt: bool = False,
    env: dict[str, str] | None = None,
    orbit_solve_error: bool = False,
    unknown_device: bool = False,
    fake_model: bool = True,
    fake_bridge: bool = True,
    channels_file: str = "channels.json",
) -> Boot:
    """Assemble the virtual accelerator against fakes, and journal the order.

    Everything that would bind a port, start a thread, install a process-wide
    signal handler or solve a lattice is replaced. Everything else -- the
    manifest load, the serving database, the drive limits, the simulation
    engine -- is the real thing, because it is what the assembly is being
    asserted about.
    """
    boot = Boot()

    monkeypatch.setenv("VA_DATA_DIR", str(facility))
    monkeypatch.setenv("VA_CHANNELS_FILE", channels_file)
    if lattice is not None:
        monkeypatch.setenv("VA_LATTICE", lattice)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(
        "osprey.utils.logger.configure_logging",
        lambda *a, **k: boot.journal.append(("logging", None)),
    )
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, _fake_runner_module(boot, interrupt=interrupt))

    def engine_source(engine: Any, channels: Any, static: Any, data_dir: Any, **kwargs: Any) -> Any:
        source = FakeEngineSource(engine, channels, static, data_dir, **kwargs)
        boot.journal.append(("engine-source", source))
        return source

    monkeypatch.setattr(ep, "EngineSource", engine_source)
    monkeypatch.setattr(
        ep,
        "_start_engine_source",
        lambda source, interval: boot.journal.append(("engine-thread", (source, interval))),
    )
    monkeypatch.setattr(
        ep, "_install_shutdown_signals", lambda: boot.journal.append(("signals", None))
    )

    if lattice not in (None, ep.LATTICE_NONE):
        _install_fake_physics(
            monkeypatch,
            boot,
            orbit_solve_error=orbit_solve_error,
            unknown_device=unknown_device,
            fake_model=fake_model,
            fake_bridge=fake_bridge,
        )

    ep.main()
    return boot


def _install_fake_physics(
    monkeypatch: pytest.MonkeyPatch,
    boot: Boot,
    *,
    orbit_solve_error: bool,
    unknown_device: bool = False,
    fake_model: bool = True,
    fake_bridge: bool = True,
) -> None:
    """Replace the ring model and the bridge, keeping their real error type.

    The lattice-backed branch is being asserted for its *composition* -- one
    model, shared, bound before the runner exists -- and solving a real ring
    would establish none of that while costing a full closed-orbit solve.

    The bridge is replaced on every lattice-backed boot, real model or not.
    What the boot owes it is the composition above; what it does with a
    reading once it has one -- the readout errors it applies, the records it
    pushes into -- is its own suite's subject, and the fake's ``bind`` is
    exactly the boot-value push whose position in the order is the contract.

    ``fake_model=False`` keeps the real model, for the tests that are about
    the tree it is built from rather than about the assembly, and
    ``fake_bridge=False`` keeps the real one too, for the test that asks
    whether a facility tree's own monitors are readable by the bridge the
    deployment actually runs.
    """
    from osprey.services.virtual_accelerator.ioc import physics_bridge as bridge_module
    from osprey.services.virtual_accelerator.model import pyat as pyat_module

    def ring_model(
        data_root: Any, channels: Any, *, bpm_errors: Any = None, corrector_gains: Any = None
    ) -> Any:
        if orbit_solve_error:
            raise bridge_module.OrbitSolveError("no stable closed orbit")
        if unknown_device:
            # The one class the model raises for a binding, a misalignment and
            # a fault seed alike -- which is the point of the test below.
            raise bridge_module.UnknownDeviceError("the ring has no element named 'ABSENT'")
        model = NullModel()
        boot.journal.append(("model-sources", (data_root, channels)))
        boot.journal.append(("model-faults", (bpm_errors, corrector_gains)))
        boot.journal.append(("model", model))
        return model

    def physics_bridge(*, model: Any) -> FakeBridge:
        bridge = FakeBridge(journal=boot.journal, model=model)
        boot.journal.append(("bridge", bridge))
        return bridge

    if fake_model:
        monkeypatch.setattr(pyat_module, "PyATRingModel", ring_model)
    if fake_bridge:
        monkeypatch.setattr(bridge_module, "PhysicsBridge", physics_bridge)


class TestBootOrder:
    """The order the pieces are assembled in, which is a contract.

    Every value that must be on the wire at boot is pushed into a PV spec
    before the server copies it, and nothing writes into a spec afterwards.
    Both halves are orderings, so both are asserted as positions in one
    sequence.
    """

    def test_the_process_is_assembled_in_the_documented_order(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.order() == [
            "logging",
            "model-sources",
            "model-faults",
            "model",
            "bridge",
            "bind",
            "engine-source",
            "runner",
            "follow-stuck",
            "engine-thread",
            "signals",
            "run",
        ]

    def test_va_state_dir_points_engine_and_source_at_the_same_state(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path, tmp_path: Path
    ) -> None:
        """``VA_STATE_DIR`` is the runtime scenario-state mount, a different
        bind-mount from the build-owned data dir. The entrypoint must hand the
        SAME directory to the engine (which keys its state file there) and to
        the EngineSource (which polls it) -- pointing them at different
        directories is how a scenario switch goes silently dead."""
        state_dir = tmp_path / "va-state"
        state_dir.mkdir()
        boot = _boot(monkeypatch, facility, env={"VA_STATE_DIR": str(state_dir)})
        source = boot.engine_source
        assert source.state_dir == state_dir
        source.engine.set_active_scenarios([])
        assert (state_dir / "active_scenarios").is_file()

    def test_without_va_state_dir_the_state_lives_beside_the_data(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The documented fallback: no ``VA_STATE_DIR`` means the data dir is
        also the state dir, which is what every pre-relocation deploy shipped."""
        boot = _boot(monkeypatch, facility)
        source = boot.engine_source
        assert source.state_dir == facility

    def test_logging_is_configured_first(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Without it the serving and physics log records this process drives
        have no handler, including the ones a failed boot emits."""
        boot = _boot(monkeypatch, facility)
        assert boot.order()[0] == "logging"

    def test_boot_values_are_pushed_before_the_server_copies_the_specs(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """``bind`` writes the boot BPM readings into PV specs. The Channel
        Access server copies each spec when it creates the PV, so a push after
        the runner exists would never be served."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.at("bind") < boot.at("runner")

    def test_telemetry_starts_only_once_the_runner_exists(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The runner attaches every record to the live driver as its last
        act. A tick before that would edit boot specs behind the server."""
        boot = _boot(monkeypatch, facility)
        assert boot.at("runner") < boot.at("engine-thread")

    def test_the_telemetry_thread_polls_at_the_documented_interval(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility)
        source, interval = boot.one("engine-thread")
        assert source is boot.engine_source
        assert interval == engine_source_module.DEFAULT_POLL_INTERVAL_S

    def test_the_poll_interval_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A facility that wants livelier telemetry sets VA_POLL_INTERVAL_S."""
        boot = _boot(monkeypatch, facility, env={"VA_POLL_INTERVAL_S": "0.25"})
        _source, interval = boot.one("engine-thread")

        assert interval == 0.25

    def test_the_noise_level_reaches_the_engine_source(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """VA_NOISE_LEVEL is forwarded; before, the constructor never saw one."""
        boot = _boot(monkeypatch, facility, env={"VA_NOISE_LEVEL": "0.05"})

        assert boot.engine_source.noise_level == 0.05

    def test_the_noise_level_defaults_to_the_engine_source_constant(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """One place the number is written down, and the entrypoint reads it."""
        boot = _boot(monkeypatch, facility)

        assert boot.engine_source.noise_level == engine_source_module.DEFAULT_NOISE_LEVEL

    def test_an_unusable_poll_interval_refuses_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Refused, not clamped: a typo has to be visible in the container log."""
        with pytest.raises(SystemExit, match="VA_POLL_INTERVAL_S"):
            _boot(monkeypatch, facility, env={"VA_POLL_INTERVAL_S": "0"})

        with pytest.raises(SystemExit, match="VA_POLL_INTERVAL_S"):
            _boot(monkeypatch, facility, env={"VA_POLL_INTERVAL_S": "fast"})

    def test_a_negative_noise_level_refuses_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Zero noise is a legitimate ask; below zero is not a fraction."""
        with pytest.raises(SystemExit, match="VA_NOISE_LEVEL"):
            _boot(monkeypatch, facility, env={"VA_NOISE_LEVEL": "-0.1"})

        boot = _boot(monkeypatch, facility, env={"VA_NOISE_LEVEL": "0"})
        assert boot.engine_source.noise_level == 0.0

    def test_stop_signals_are_installed_only_once_the_servers_are_up(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Before that the process is still assembling, and dying immediately
        is the right answer to a stop signal."""
        boot = _boot(monkeypatch, facility)
        assert boot.at("runner") < boot.at("signals")

    def test_the_run_loop_is_entered_last(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility)
        assert boot.order()[-1] == "run"
        assert boot.at("signals") < boot.at("run")

    def test_the_ready_line_is_printed_after_the_servers_are_up(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Everything that waits on this boot greps for it, so it may not
        appear before the namespace is actually being served."""
        boot = _boot(monkeypatch, facility)
        out = capsys.readouterr().out
        assert ep.READY_MARKER in out
        assert boot.runner is not None

    def test_the_ready_line_counts_the_whole_served_namespace(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The count is read out of the line by the image boot check and the
        container fixtures, so it is the served channel count and not the
        manifest's."""
        boot = _boot(monkeypatch, facility)
        out = capsys.readouterr().out
        assert ep._ready_line(len(boot.runner.records.all)) in out
        assert len(boot.runner.records.all) == len(CHANNELS)


class TestRunnerHandoff:
    """What the entrypoint hands the runner."""

    def test_the_runner_gets_the_database_the_engine_source_drives(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """One database, not two: the telemetry source and the server must be
        pushing into and serving the same records."""
        boot = _boot(monkeypatch, facility)
        records = boot.runner.records
        assert isinstance(records, ServingRecords)
        assert boot.engine_source.static_noisy is records.static_noisy

    def test_setpoints_are_declared_asynchronous(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A synchronous setpoint would tell its client the write had landed
        before the solve behind it had started."""
        boot = _boot(monkeypatch, facility)
        records = boot.runner.records
        assert records.pvdb[MAG1_SP]["asyn"] is True
        assert records.pvdb[MAG1_RB].get("asyn", False) is False

    def test_the_drive_limits_reach_both_the_database_and_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The database publishes the band as display limits and the write
        path enforces it; they are the same numbers, from the same file."""
        boot = _boot(monkeypatch, facility)
        assert boot.runner.kwargs["drive_limits"] == DRIVE_LIMITS
        assert boot.runner.records.pvdb[MAG1_SP]["lolim"] == MAG_BAND[0]
        assert boot.runner.records.pvdb[MAG1_SP]["hilim"] == MAG_BAND[1]

    def test_boot_values_from_the_mounted_machine_file_reach_the_setpoints(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility)
        pvdb = boot.runner.records.pvdb
        assert pvdb[MAG1_SP]["value"] == BOOT_VALUES[MAG1_SP]
        assert pvdb[MAG2_RB]["value"] == BOOT_VALUES[MAG2_RB]

    def test_the_apply_fault_setpoints_reach_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, env={"VA_STUCK_SETPOINTS": f" {MAG1_SP} ,{MAG2_SP}"})
        assert boot.runner.kwargs["stuck_setpoints"] == frozenset({MAG1_SP, MAG2_SP})

    def test_no_apply_fault_means_an_empty_set_not_a_none(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility)
        assert boot.runner.kwargs["stuck_setpoints"] == frozenset()

    def test_the_entrypoint_never_attaches_the_driver_itself(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Attaching is the runner's last act, once the servers exist. Doing
        it here would point the records at a driver that does not yet serve
        them, and the boot values pushed before it would be lost."""
        attached: list[Any] = []
        monkeypatch.setattr(
            ServingRecords, "attach_driver", lambda self, driver: attached.append(driver)
        )
        _boot(monkeypatch, facility)
        assert attached == []


class TestNullLatticeBoot:
    """A boot whose facility has channels but no physics in this process."""

    def test_a_file_backed_manifest_defaults_to_no_lattice(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The default follows the channel source, so a facility supplying its
        own manifest never has to know PyAT exists."""
        boot = _boot(monkeypatch, facility)
        assert isinstance(boot.runner.model, NullModel)

    def test_the_served_model_is_the_stub(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)
        assert isinstance(boot.runner.model, NullModel)

    def test_there_is_no_physics_hook(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Which is what makes a pyat-coupled setpoint latch rather than
        propagate: the write path takes the absent hook as its signal."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)
        assert boot.runner.kwargs["on_setpoint"] is None

    def test_nothing_binds_a_physics_bridge(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)
        assert "bridge" not in boot.order()
        assert "bind" not in boot.order()

    def test_the_whole_manifest_is_still_served(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Same addresses, same count as a lattice-backed boot of the same
        facility: the difference is confined to what happens after a write."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)
        with_lattice = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert set(boot.runner.records.pvdb) == {channel["address"] for channel in CHANNELS}
        assert set(boot.runner.records.pvdb) == set(with_lattice.runner.records.pvdb)

    def test_accepted_setpoints_are_synced_into_the_engine(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """With no lattice the engine is the only physics in the process, so
        a machine-file expression channel can only respond to a setpoint if
        the sp-echo readbacks are fed back into it."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)
        echoes = boot.engine_source.setpoint_echo_records
        assert echoes is not None
        assert set(echoes) == {VALVE_RB}
        assert echoes[VALVE_RB] is boot.runner.records.all[VALVE_RB]

    def test_lattice_physics_faults_are_refused_without_a_lattice(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        with pytest.raises(SystemExit, match="VA_BPM_ERRORS"):
            _boot(
                monkeypatch,
                facility,
                lattice=ep.LATTICE_NONE,
                env={"VA_BPM_ERRORS": "BPM01:offset_x=1e-4"},
            )

    def test_corrector_gain_faults_are_refused_too(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        with pytest.raises(SystemExit, match="VA_CORR_GAIN"):
            _boot(
                monkeypatch,
                facility,
                lattice=ep.LATTICE_NONE,
                env={"VA_CORR_GAIN": "HCM01=1.5"},
            )


class TestLatticeBoot:
    """A boot with physics behind part of the namespace."""

    def test_one_model_serves_the_bridge_and_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A second model would be a second lattice, silently diverging from
        the one the BPM readings are solved out of."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        bridge = boot.one("bridge")
        assert bridge.model is boot.one("model")
        assert boot.runner.model is bridge.model

    def test_the_physics_hook_is_the_bridges(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.runner.kwargs["on_setpoint"] == boot.one("bridge").on_setpoint

    def test_the_bridge_is_bound_to_the_coupled_partition(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.one("bridge").bound is boot.runner.records.pyat_coupled

    def test_setpoints_are_not_also_synced_into_the_engine(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Physics coupling flows through the bridge; a second path into the
        engine would be a second thing moving the same channels."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.engine_source.setpoint_echo_records is None

    def test_the_fault_seeds_reach_the_model(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(
            monkeypatch,
            facility,
            lattice=SERVED_LATTICE,
            env={
                "VA_BPM_ERRORS": f"{BPM_ELEMENT}:offset_x=1e-4",
                "VA_CORR_GAIN": f"{MAG1_ELEMENT}=1.5",
            },
        )
        bpm_errors, corrector_gains = boot.one("model-faults")
        assert bpm_errors == {BPM_ELEMENT: {"offset_x": 1e-4}}
        assert corrector_gains == {MAG1_ELEMENT: {"factor": 1.5}}

    def test_no_faults_are_passed_as_none_rather_than_an_empty_map(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.one("model-faults") == (None, None)

    def test_a_lattice_with_no_stable_orbit_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Diagnosable rather than an opaque crash -- which is why the model
        itself never ends the process."""
        with pytest.raises(SystemExit, match="no stable closed orbit"):
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE, orbit_solve_error=True)

    def test_an_element_the_tree_does_not_carry_ends_the_boot_without_guessing_why(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A binding, a misalignment and a fault seed all raise the one class,
        so the headline names the condition and carries the model's own
        message. Naming a cause it cannot know would send an operator to the
        wrong file."""
        with pytest.raises(SystemExit) as excinfo:
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE, unknown_device=True)

        message = str(excinfo.value)
        assert "does not carry" in message
        assert "ABSENT" in message, "the model's own message has to survive"
        assert "fault seed" not in message, "the boot cannot know which of the three it was"

    def test_the_model_is_built_from_the_tree_around_the_served_directory(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The model reads a whole facility tree, not a lattice file.

        Its bindings and the lattice sit under the tree's ``simulation/`` --
        which is the mounted directory -- and the bands its nominals are
        weighed against at the root above it. So what the model is given is
        that root, and nothing here re-derives it.
        """
        root, _ = _boot(monkeypatch, facility, lattice=SERVED_LATTICE).one("model-sources")
        assert root == facility.parent

    def test_the_model_is_put_on_the_namespace_the_ioc_serves(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The channel list is passed in, not resolved a second time: two
        resolutions could land on two namespaces, and then the model would
        describe addresses the IOC does not serve."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        _, channels = boot.one("model-sources")
        assert [channel["address"] for channel in channels] == [
            channel["address"] for channel in CHANNELS
        ]

    def test_the_documents_readback_rules_reach_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """What a coupled setpoint serves on readback is the document's
        statement, and only the served path can apply it.

        Without this the runner derives the coupled set from the manifest
        alone and every readback is an echo of the value written -- so a
        facility that exported a reverse curve for a channel gets its channel
        served as though it had not.
        """
        bound = _boot(monkeypatch, facility, lattice=SERVED_LATTICE).runner.kwargs[
            "bound_setpoints"
        ]
        assert sorted(bound) == sorted(PHYSICS_SETPOINTS)
        assert {entry.rule for entry in bound.values()} == {"identity"}
        assert bound[MAG1_SP].readback == MAG1_RB

    def test_a_lattice_free_boot_declares_no_readback_rules(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """There is no document to read them from, and an empty map is what
        the write path treats as the lattice-free behaviour: every coupled
        readback echoes the value written."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)
        assert boot.runner.kwargs["bound_setpoints"] == {}


class TestTheModelSurfaceIsWiredAtBoot:
    """What the runner needs to answer for the model, and to arm a write.

    A model write reaches the model and no setpoint, so the one thing the
    boot owes the surface besides the token is the hook that re-serves what
    a written variable feeds.
    """

    def test_the_bridge_is_handed_the_setpoints_it_may_re_command(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A calibration write leaves a magnet delivering the wrong value for
        its standing command; the bridge can only put that right if it was
        told which records carry those commands."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

        assert boot.one("bridge").setpoints == frozenset(PHYSICS_SETPOINTS)

    def test_the_refresh_hook_is_the_bridges(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

        assert boot.runner.kwargs["refresh"] == boot.one("bridge").refresh

    def test_the_bridge_reads_the_stuck_set_from_the_write_path(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A stuck setpoint records what was written and hands the model
        nothing, so the bridge must not re-command from it on a calibration
        change. The set changes at runtime, so the bridge is handed the write
        path's reader rather than a copy of the boot set."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

        assert boot.one("bridge").stuck_reader == boot.runner.write_path.stuck_setpoints

    def test_the_stuck_reader_is_handed_over_only_once_the_runner_exists(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The write path is built with the server, and the bridge had to be
        bound before that -- so this is the earliest the reader exists."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

        assert boot.at("follow-stuck") > boot.at("runner")

    def test_a_lattice_free_boot_wires_no_stuck_reader(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """There is no bridge to hand it to."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)

        assert "follow-stuck" not in boot.order()

    def test_a_lattice_free_boot_passes_no_refresh_hook(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Nothing derives a reading from a model variable, so there is
        nothing to recompute and the runner's own inert default stands."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)

        assert "refresh" not in boot.runner.kwargs

    def test_the_status_verb_is_told_what_this_boot_built(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Both are boot facts nothing downstream can recover: the backend is
        whichever model was built, and the source is the file this boot
        resolved rather than the raw variable behind it."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

        assert boot.runner.kwargs["backend_name"] == type(boot.one("model")).__name__
        assert boot.runner.kwargs["lattice_source"].endswith(SERVED_LATTICE)

    def test_a_lattice_free_boot_says_it_serves_no_lattice(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_NONE)

        assert boot.runner.kwargs["lattice_source"] == ep.LATTICE_NONE

    def test_model_writes_are_refused_unless_the_deployment_arms_them(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The model RPC reaches past the served namespace into the physics,
        so an unset variable is a refusal rather than an open door."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

        assert boot.runner.kwargs["model_write_token"] is None

    def test_an_armed_token_reaches_the_runner_exactly_as_given(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Matched byte for byte against what a client presents, so nothing
        here rewrites it."""
        boot = _boot(
            monkeypatch,
            facility,
            lattice=SERVED_LATTICE,
            env={"VA_MODEL_WRITE_TOKEN": " s3cret "},
        )

        assert boot.runner.kwargs["model_write_token"] == " s3cret "

    def test_a_blank_token_is_unset_rather_than_a_token(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The compose passthrough sends an empty string when the host
        variable is absent, and an empty token is one an empty credential
        would match."""
        boot = _boot(
            monkeypatch, facility, lattice=SERVED_LATTICE, env={"VA_MODEL_WRITE_TOKEN": "   "}
        )

        assert boot.runner.kwargs["model_write_token"] is None


class TestBootRefusals:
    """Boots that must not reach the runner at all."""

    def test_a_missing_machine_file_names_the_mount(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("VA_DATA_DIR", str(tmp_path))
        with pytest.raises(SystemExit, match="no machine.json"):
            ep.main()

    def test_a_single_file_bind_mount_is_named_as_the_likely_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The commonest misconfiguration: mounting machine.json itself rather
        than the directory holding it."""
        monkeypatch.setenv("VA_DATA_DIR", str(tmp_path))
        with pytest.raises(SystemExit, match="DIRECTORY"):
            ep.main()

    def test_a_lattice_the_tree_does_not_carry_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """``VA_LATTICE`` names a source, so a name the mount does not carry
        is refused against the tree rather than deep inside a loader."""
        with pytest.raises(SystemExit, match="VA_LATTICE"):
            _boot(monkeypatch, facility, lattice="maybe.json")

    def test_an_unknown_lattice_name_is_refused_against_the_tree_it_searched(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The refusal has to say where it looked and where to go next.

        A deployment reading it knows only the name it set; the path it was
        resolved against is this process's own, and the value that serves the
        manifest without physics is the way out of the refusal.
        """
        with pytest.raises(SystemExit) as excinfo:
            _boot(monkeypatch, facility, lattice="maybe.json")
        message = str(excinfo.value)
        assert str(facility / "maybe.json") in message
        assert f"VA_LATTICE={ep.LATTICE_NONE}" in message

    def test_a_differently_cased_lattice_name_never_reaches_the_runner(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The name is looked up verbatim, so its case is part of it.

        Which refusal answers depends on the host: where the filesystem folds
        case the name finds the tree's own lattice and is refused as a
        different file, and where it does not the name finds nothing. Both
        name the spelling that was given, and neither serves a ring under a
        name the deployment did not write.
        """
        with pytest.raises(SystemExit) as excinfo:
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE.upper())
        assert SERVED_LATTICE.upper() in str(excinfo.value)

    def test_a_lattice_beside_a_different_name_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Serving one ring while modelling another is invisible from the
        wire: every channel is served and every write accepted, with the
        physics behind them belonging to a different lattice."""
        (facility / "other.json").write_text("{}")
        with pytest.raises(SystemExit, match="the tree's own lattice"):
            _boot(monkeypatch, facility, lattice="other.json")

    def test_a_served_directory_outside_a_facility_tree_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path, tmp_path: Path
    ) -> None:
        """The model resolves the tree around the mount, so the mount has to
        be the tree's ``simulation/`` directory; anything else would have it
        looking for its lattice somewhere the deployment never wrote one."""
        flat = tmp_path / "flat"
        flat.mkdir()
        for name in ("channels.json", "machine.json", SERVED_LATTICE, "va_bindings.json"):
            (flat / name).write_text((facility / name).read_text())
        with pytest.raises(SystemExit, match="simulation/ directory"):
            _boot(monkeypatch, flat, lattice=SERVED_LATTICE)

    def test_a_lattice_with_no_bindings_beside_it_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A lattice on its own models nothing any channel can reach."""
        (facility / "va_bindings.json").unlink()
        with pytest.raises(SystemExit, match="no bindings"):
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

    def test_a_coupled_channel_the_document_binds_to_nothing_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The manifest's coupled partition is derived from the document, so a
        disagreement means the two came from different builds.

        Named here, with the address and the file, because the model layer
        only ever sees the unbound channel as a variable of the wrong type --
        no address, no file.
        """
        document = {**BINDINGS_DOCUMENT, "bindings": BINDINGS_DOCUMENT["bindings"][:1]}
        (facility / "va_bindings.json").write_text(json.dumps(document))
        with pytest.raises(SystemExit, match=re.escape(MAG2_SP)):
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

    def test_a_readback_the_served_namespace_cannot_produce_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """A readback served through the facility's own inverse needs a model
        variable of that address to compute it.

        The model is built from the manifest's channels, so an address the
        manifest does not carry gets no variable and there is nothing to map
        back along -- the same two-builds disagreement the check above refuses
        from the other direction. Refused by name here, because all the write
        path can say is that the readback of some binding cannot be produced,
        and an unnamed refusal out of a container's PID 1 is a traceback.
        """
        unserved = f"{RING}:MAG:HCM:09:CURRENT:SP"
        bound_to_nothing = {
            **_identity_binding(unserved, f"{RING}:MAG:HCM:09:CURRENT:RB", "M9"),
            "readback": "inverse",
            "monitor_inverse": {"kind": "linear", "gain": 100.0, "offset": 0.0},
        }
        document = {
            **BINDINGS_DOCUMENT,
            "bindings": [*BINDINGS_DOCUMENT["bindings"], bound_to_nothing],
        }
        (facility / "va_bindings.json").write_text(json.dumps(document))
        with pytest.raises(SystemExit, match=re.escape(unserved)):
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE)

    def test_a_readback_is_bound_by_nothing_and_is_not_missed(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The serving layer mirrors a setpoint onto its readback record, so a
        coupled readback is not a model variable and no binding names it.

        The whole coupled partition here is two setpoints and three readbacks;
        a check that asked for a binding per coupled channel would refuse
        every tree ever exported.
        """
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert len(boot.runner.records.pyat_coupled) > len(PHYSICS_SETPOINTS)


class TestSeededReadoutErrorsAreResolvedAgainstTheDocument:
    """A seeded readout error is keyed by the monitor it perturbs.

    The grammar's device token is whatever the person seeding the fault knows
    the device by, and a facility publishes a monitor under two names: the
    address its reading goes out on and the element the deck reads it at. The
    bridge holds one error model per element, so the served document is what
    turns the first into the second -- and a token it knows under neither name
    ends the boot, because a machine that serves unperturbed readings while
    reporting a seeded fault is exactly the failure the seed exists to reveal.
    """

    def test_a_device_named_by_its_element_reaches_the_model(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(
            monkeypatch,
            facility,
            lattice=SERVED_LATTICE,
            env={"VA_BPM_ERRORS": f"{BPM_ELEMENT}:offset_x=1e-4,gain_y=1.05"},
        )
        assert boot.one("model-faults")[0] == {BPM_ELEMENT: {"offset_x": 1e-4, "gain_y": 1.05}}

    def test_a_device_named_by_its_address_reaches_the_same_monitor(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The spelling an operator reads off the control system, which is
        colon separated at every level -- so it survives the grammar whole and
        lands on the element the bridge keys by."""
        boot = _boot(
            monkeypatch,
            facility,
            lattice=SERVED_LATTICE,
            env={"VA_BPM_ERRORS": f"{BPM_X}:offset_x=1e-4"},
        )
        assert boot.one("model-faults")[0] == {BPM_ELEMENT: {"offset_x": 1e-4}}

    def test_a_device_the_document_knows_under_neither_name_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _boot(
                monkeypatch,
                facility,
                lattice=SERVED_LATTICE,
                env={"VA_BPM_ERRORS": "BPM99:offset_x=1e-4"},
            )
        message = str(excinfo.value)
        assert "VA_BPM_ERRORS" in message
        assert "BPM99" in message

    def test_one_monitor_seeded_twice_on_one_field_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Both spellings name one monitor, so seeding a field through each
        states two values for one readout error and which was meant could only
        be guessed."""
        with pytest.raises(SystemExit) as excinfo:
            _boot(
                monkeypatch,
                facility,
                lattice=SERVED_LATTICE,
                env={"VA_BPM_ERRORS": (f"{BPM_ELEMENT}:offset_x=1e-4;{BPM_X}:offset_x=2e-4")},
            )
        assert "VA_BPM_ERRORS" in str(excinfo.value)

    def test_seeding_nothing_leaves_the_model_unperturbed(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Resolution of an empty seed map is empty, and an empty map is the
        None the model reads as "every monitor reads the true position"."""
        boot = _boot(monkeypatch, facility, lattice=SERVED_LATTICE)
        assert boot.one("model-faults")[0] is None


class TestSeededCalibrationsAreResolvedTheSameWay:
    """A magnet is named on the same terms a monitor is.

    An operator reads a magnet's address off the same screen as a monitor's,
    and a scenario file renders whichever spelling it was written with. The
    model keys its calibrations by element, so the document turns one into the
    other here too -- and a name it knows neither way ends the boot rather
    than perturbing nothing.
    """

    def test_a_magnet_named_by_its_element_reaches_the_model(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(
            monkeypatch,
            facility,
            lattice=SERVED_LATTICE,
            env={"VA_CORR_GAIN": f"{MAG1_ELEMENT}=1.5"},
        )
        assert boot.one("model-faults")[1] == {MAG1_ELEMENT: {"factor": 1.5}}

    def test_a_magnet_named_by_its_setpoint_address_reaches_the_same_element(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The spelling the control system publishes. Without this a scenario
        that names its magnets the way an operator does would end the boot."""
        boot = _boot(
            monkeypatch, facility, lattice=SERVED_LATTICE, env={"VA_CORR_GAIN": f"{MAG1_SP}=1.5"}
        )
        assert boot.one("model-faults")[1] == {MAG1_ELEMENT: {"factor": 1.5}}

    def test_a_magnet_the_document_knows_under_neither_name_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _boot(monkeypatch, facility, lattice=SERVED_LATTICE, env={"VA_CORR_GAIN": "HCM99=1.5"})
        message = str(excinfo.value)
        assert "VA_CORR_GAIN" in message
        assert "HCM99" in message

    def test_one_magnet_seeded_twice_on_one_field_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _boot(
                monkeypatch,
                facility,
                lattice=SERVED_LATTICE,
                env={"VA_CORR_GAIN": f"{MAG1_ELEMENT}=1.5,{MAG1_SP}=2.0"},
            )
        assert "VA_CORR_GAIN" in str(excinfo.value)

    def test_a_monitor_is_not_a_magnet(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """The two rosters are resolved against different halves of the
        document, so a monitor's name is no more a magnet than a typo is."""
        with pytest.raises(SystemExit, match="VA_CORR_GAIN"):
            _boot(
                monkeypatch,
                facility,
                lattice=SERVED_LATTICE,
                env={"VA_CORR_GAIN": f"{BPM_ELEMENT}=1.5"},
            )


class TestShutdown:
    """How the process leaves."""

    def test_a_stop_signal_leaves_through_the_runners_own_exit(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``run()`` returns on a KeyboardInterrupt and nothing else; catching
        it again here covers a signal arriving between the loop's own try and
        this call."""
        _boot(monkeypatch, facility, interrupt=True)
        assert "virtual accelerator IOC stopped" in capsys.readouterr().out

    def test_both_stop_signals_raise_the_interrupt_the_runner_exits_on(self) -> None:
        """SIGTERM is what ``docker stop`` sends, and by default it kills the
        process outright rather than unwinding through the run loop."""
        import signal

        with pytest.raises(KeyboardInterrupt, match="15"):
            ep._raise_keyboard_interrupt(signal.SIGTERM, None)
        with pytest.raises(KeyboardInterrupt, match="2"):
            ep._raise_keyboard_interrupt(signal.SIGINT, None)


# --- the model a boot with no physics serves ------------------------------


class TestNullModel:
    """The stub is total, because the run loop calls every method of it."""

    @pytest.fixture()
    def model(self) -> NullModel:
        return NullModel()

    def test_it_is_the_model_the_runner_is_built_around(self, model: NullModel) -> None:
        assert isinstance(model, LUMEModel)

    def test_it_describes_no_variables(self, model: NullModel) -> None:
        """So the runner creates no PVA variable PV, and what a client sees is
        the co-hosted namespace alone."""
        assert model.supported_variables == {}

    def test_each_access_gets_its_own_dict(self, model: NullModel) -> None:
        """The base class hands this straight to callers; a shared one would
        let any of them mutate the model's declared namespace."""
        first = model.supported_variables
        first["ZZ:SOME:CHANNEL"] = object()
        assert model.supported_variables == {}

    def test_the_empty_read_back_is_empty_rather_than_an_error(self, model: NullModel) -> None:
        """The run loop reads every variable back at the end of each cycle."""
        assert model.get([]) == {}

    def test_reading_an_unknown_variable_still_raises(self, model: NullModel) -> None:
        """Totality is about the empty case, not about answering for channels
        this process has no physics for."""
        with pytest.raises(ValueError, match="not supported"):
            model.get([MAG1_SP])

    def test_the_empty_cycle_is_not_an_error(self, model: NullModel) -> None:
        """Every cycle in this configuration applies an empty batch; raising
        would take the loop down on its first tick."""
        assert model.set({}) is None

    def test_setting_an_unknown_variable_still_raises(self, model: NullModel) -> None:
        with pytest.raises(ValueError, match="not supported"):
            model.set({MAG1_SP: 1.0})

    def test_reset_is_a_no_op(self, model: NullModel) -> None:
        assert model.reset() is None

    def test_the_stub_pulls_in_no_server_or_physics_stack(self) -> None:
        """Its whole purpose is booting without them, so importing it must
        cost no more than ``lume`` does."""
        script = (
            "import sys;"
            "from osprey.services.virtual_accelerator.serving.model_stub import NullModel;"
            "heavy=[m for m in ('lume_pva_apg','pcaspy','p4p','at') if m in sys.modules];"
            "print(heavy);"
            "assert not heavy, heavy;"
            "assert NullModel().supported_variables == {}"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )

        assert result.returncode == 0, result.stderr


# --- the two orderings a write depends on ---------------------------------


class RecordingDriver:
    """The Channel Access driver, reduced to a journal.

    ``values`` is the served database: a one-shot read is answered from it.
    ``calls`` records every operation in order, and the physics hook and the
    PVA publisher record into the same list, so where the hook falls relative
    to the values it produces is one sequence rather than three.
    """

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = dict(values)
        self.calls: list[tuple[str, str, Any]] = []

    def setParam(self, reason: str, value: Any) -> None:  # noqa: N802 - driver contract
        self.calls.append(("setParam", reason, value))
        self.values[reason] = value

    def getParam(self, reason: str) -> Any:  # noqa: N802 - driver contract
        return self.values[reason]

    def updatePV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.calls.append(("updatePV", reason, None))

    def callbackPV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.calls.append(("callbackPV", reason, None))

    def setParamStatus(  # noqa: N802 - driver contract
        self, reason: str, alarm: Any, severity: Any
    ) -> None:
        self.calls.append(("setParamStatus", reason, (alarm, severity)))

    def sequence(self) -> list[tuple[str, str]]:
        return [(call, reason) for call, reason, _ in self.calls]


class TwoMagnetModel(LUMEModel):
    """A model of the two setpoints the physics hook writes through."""

    def __init__(self, *, refuse: frozenset[str] = frozenset()) -> None:
        self.refuse = refuse
        self.sets: list[dict[str, Any]] = []
        self._vars = {
            address: ScalarVariable(
                name=address,
                default_value=BOOT_VALUES[address],
                value_range=MAG_BAND,
                default_validation_config="none",
                read_only=False,
            )
            for address in (MAG1_SP, MAG2_SP)
        }

    @property
    def supported_variables(self) -> dict[str, ScalarVariable]:
        return self._vars

    def _get(self, names: list[str]) -> dict[str, Any]:
        return dict.fromkeys(names, 0.0)

    def _set(self, values: dict[str, Any]) -> None:
        self.sets.append(dict(values))
        refused = sorted(set(values) & self.refuse)
        if refused:
            raise RuntimeError(f"no stable closed orbit after writing {refused}")

    def reset(self) -> None:  # pragma: no cover - never reached here
        pass


@dataclass
class Arrangement:
    """One write path, its driver, and the queue standing in for the run loop.

    Assembled the way the runner assembles it: the hook is reached through the
    model wrapper, so it runs on whichever thread drains the queue, and the
    PVA publisher records into the driver's journal.
    """

    records: ServingRecords
    driver: RecordingDriver
    model: TwoMagnetModel
    routed: SetpointRoutedModel
    path: CohostWritePath
    queue: list[tuple[dict[str, Any], Any]]
    hook_journals: dict[str, list[tuple[str, str, Any]]]

    def drain(self) -> None:
        """One queued item is one cycle is one ``model.set`` -- the batching
        window is off, so the loop never merges two of them."""
        while self.queue:
            values, done = self.queue.pop(0)
            error = None
            try:
                self.routed.set({name: item["value"] for name, item in values.items()})
            except Exception as exc:  # noqa: BLE001 - the loop reports, never raises
                error = str(exc)
            if done is not None:
                done(error)


def _arrangement(*, refuse: frozenset[str] = frozenset()) -> Arrangement:
    records = build_serving_pvdb(
        CHANNELS,
        drive_limits=DRIVE_LIMITS,
        boot_values=BOOT_VALUES,
        async_setpoints=True,
    )
    driver = RecordingDriver({address: spec["value"] for address, spec in records.pvdb.items()})
    records.attach_driver(driver)
    model = TwoMagnetModel(refuse=refuse)
    queue: list[tuple[dict[str, Any], Any]] = []
    hook_journals: dict[str, list[tuple[str, str, Any]]] = {}

    def on_setpoint(address: str, value: float) -> None:
        # The journal as it stood when the hook was called: everything the
        # write is supposed to produce must still be absent from it.
        hook_journals[address] = list(driver.calls)
        driver.calls.append(("hook", address, value))
        model.set({address: value})

    def enqueue(values: dict[str, Any], done: Any = None, _reset: bool = False) -> None:
        queue.append((values, done))

    def pva_post(address: str, value: Any) -> None:
        # Only the model's own variables have a PVA channel; the paired :RB
        # has none, exactly as in the served namespace.
        if address in PHYSICS_SETPOINTS:
            driver.calls.append(("post", address, value))

    routed = SetpointRoutedModel(model, on_setpoint=on_setpoint, routed=PHYSICS_SETPOINTS)
    path = CohostWritePath(
        records,
        enqueue=enqueue,
        physics_setpoints=physics_setpoint_addresses(records),
        drive_limits=DRIVE_LIMITS,
        refusal_alarm=("WRITE_ALARM", "INVALID_ALARM"),
        pva_post=pva_post,
    )
    return Arrangement(
        records=records,
        driver=driver,
        model=model,
        routed=routed,
        path=path,
        queue=queue,
        hook_journals=hook_journals,
    )


class TestHookBeforeEcho:
    """Where the physics hook falls relative to the values it produces.

    A setpoint and its readback may carry only a value the model has taken,
    and whether it has is not known until the hook returns. So the hook is
    not merely *called* before the echo -- it has completed before anything
    at all has been committed, on either view.
    """

    def test_nothing_is_committed_when_the_hook_is_reached(self) -> None:
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 3.25)
        arrangement.drain()
        assert arrangement.hook_journals[MAG1_SP] == []

    def test_the_hook_precedes_both_views_and_the_completion(self) -> None:
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 3.25)
        arrangement.drain()
        assert arrangement.driver.sequence() == [
            ("hook", MAG1_SP),
            ("setParam", MAG1_SP),
            ("updatePV", MAG1_SP),
            ("post", MAG1_SP),
            ("setParam", MAG1_RB),
            ("updatePV", MAG1_RB),
            ("callbackPV", MAG1_SP),
        ]

    def test_the_hook_is_offered_the_post_clamp_value(self) -> None:
        """Clamp first, then physics: the value the model accepts is the value
        echoed, so the two can never disagree."""
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 700.0)
        arrangement.drain()
        assert arrangement.model.sets == [{MAG1_SP: MAG_BAND[1]}]
        assert arrangement.driver.values[MAG1_RB] == MAG_BAND[1]

    def test_a_put_reaches_the_hook_before_its_own_completion(self) -> None:
        """The PVA transport differs only in how the client is told the write
        finished, so the hook's position is the same."""
        arrangement = _arrangement()
        completions: list[str | None] = []
        arrangement.path.put(
            arrangement.driver, MAG1_SP, 3.25, done=lambda error: completions.append(error)
        )
        assert completions == []

        arrangement.drain()
        assert arrangement.hook_journals[MAG1_SP] == []
        assert completions == [None]

    def test_a_setpoint_with_no_hook_behind_it_echoes_nothing(self) -> None:
        """The hook is what an echo waits on, so a refusal reaching it leaves
        both addresses where they were -- for every reader."""
        arrangement = _arrangement(refuse=frozenset({MAG1_SP}))
        arrangement.path.write(arrangement.driver, MAG1_SP, 3.25)
        arrangement.drain()
        assert arrangement.driver.values[MAG1_SP] == BOOT_VALUES[MAG1_SP]
        assert arrangement.driver.values[MAG1_RB] == BOOT_VALUES[MAG1_RB]
        assert ("setParam", MAG1_SP) not in arrangement.driver.sequence()


class TestPerWriteIsolation:
    """Two writes in flight at once stay two writes.

    Setpoint writes are physically ordered events. The batching window is off
    (``update_rate`` 0.0) so that each queued write gets a ``model.set`` of its
    own; what that buys is asserted here, at the boundary where two writes
    could still be merged -- one client's value must never be applied, echoed
    or completed as part of another's.
    """

    def test_two_writes_are_two_queued_items(self) -> None:
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        assert [sorted(values) for values, _ in arrangement.queue] == [[MAG1_SP], [MAG2_SP]]

    def test_each_item_carries_its_own_completion(self) -> None:
        """A shared completion would end whichever asynchronous write happened
        to be in flight, not the one the item belongs to."""
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        first, second = (done for _, done in arrangement.queue)
        assert first is not None
        assert first is not second

    def test_each_item_is_applied_in_a_set_of_its_own(self) -> None:
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        arrangement.drain()
        assert arrangement.model.sets == [{MAG1_SP: 1.5}, {MAG2_SP: -2.5}]

    def test_neither_value_reaches_the_others_addresses(self) -> None:
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        arrangement.drain()
        assert arrangement.driver.values[MAG1_SP] == 1.5
        assert arrangement.driver.values[MAG1_RB] == 1.5
        assert arrangement.driver.values[MAG2_SP] == -2.5
        assert arrangement.driver.values[MAG2_RB] == -2.5

    def test_each_write_completes_exactly_once(self) -> None:
        arrangement = _arrangement()
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        arrangement.drain()
        completions = [
            reason for call, reason, _ in arrangement.driver.calls if call == "callbackPV"
        ]
        assert completions == [MAG1_SP, MAG2_SP]

    def test_a_refusal_does_not_withhold_the_other_writes_echo(self) -> None:
        """The two writes are independent events; one failing is not the
        machine-wide no-op a shared cycle would make of it."""
        arrangement = _arrangement(refuse=frozenset({MAG1_SP}))
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        arrangement.drain()
        assert arrangement.driver.values[MAG1_SP] == BOOT_VALUES[MAG1_SP]
        assert arrangement.driver.values[MAG2_SP] == -2.5

    def test_a_refusal_does_not_withhold_the_other_writes_completion(self) -> None:
        """A Channel Access write whose completion never fires postpones every
        later write to that PV for the life of the process."""
        arrangement = _arrangement(refuse=frozenset({MAG1_SP}))
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        arrangement.drain()
        completions = [
            reason for call, reason, _ in arrangement.driver.calls if call == "callbackPV"
        ]
        assert completions == [MAG1_SP, MAG2_SP]

    def test_the_alarm_lands_on_the_refused_setpoint_alone(self) -> None:
        arrangement = _arrangement(refuse=frozenset({MAG1_SP}))
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.write(arrangement.driver, MAG2_SP, -2.5)
        arrangement.drain()
        alarmed = [
            reason for call, reason, _ in arrangement.driver.calls if call == "setParamStatus"
        ]
        assert alarmed == [MAG1_SP]

    def test_a_put_and_a_write_in_flight_together_stay_separate(self) -> None:
        """Both transports enqueue onto the same loop, so the isolation has to
        hold across them and not only within one."""
        arrangement = _arrangement()
        completions: list[str | None] = []
        arrangement.path.write(arrangement.driver, MAG1_SP, 1.5)
        arrangement.path.put(
            arrangement.driver, MAG2_SP, -2.5, done=lambda error: completions.append(error)
        )
        arrangement.drain()
        assert arrangement.model.sets == [{MAG1_SP: 1.5}, {MAG2_SP: -2.5}]
        assert completions == [None]
        # The put ends no Channel Access asynchronous write of its own.
        assert [reason for call, reason, _ in arrangement.driver.calls if call == "callbackPV"] == [
            MAG1_SP
        ]


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_serving_entrypoint.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS


def test_every_host_side_serving_suite_still_guards_its_own_collection() -> None:
    """The floors above, for the suite as a whole rather than one module.

    A per-module floor cannot notice its own module being deleted, and the
    serving layer is proven almost entirely in process -- so a suite that
    quietly disappeared would take a large part of that proof with it and
    leave the remaining run green. This is the one check that spans the set:
    each host-side serving suite must still exist and must still declare a
    floor no lower than the one recorded here.

    A floor per suite rather than an exact total, and a lower bound rather
    than an equality: adding tests to any of them is not a regression, and
    only a suite shrinking or vanishing is.
    """
    recorded = {
        "test_serving_pvdb.py": 30,
        "test_serving_runner.py": 90,
        "test_serving_entrypoint.py": MIN_COLLECTED_TESTS,
        "test_engine_source_serving.py": 20,
    }
    here = Path(__file__).parent

    declared: dict[str, int | None] = {}
    for name in recorded:
        path = here / name
        if not path.is_file():
            declared[name] = None
            continue
        match = re.search(r"^MIN_COLLECTED_TESTS\s*=\s*(\d+)", path.read_text(), re.MULTILINE)
        declared[name] = int(match.group(1)) if match else None

    missing = sorted(name for name, floor in declared.items() if floor is None)
    assert not missing, f"host-side serving suite missing or without a collection floor: {missing}"

    lowered = {
        name: declared[name]
        for name, floor in recorded.items()
        if declared[name] < floor  # type: ignore[operator]
    }
    assert not lowered, f"collection floor lowered: {lowered}"
