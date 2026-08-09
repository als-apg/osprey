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
VALVE_SP = f"{RING}:VAC:VALVE:01:POSITION:SP"
VALVE_RB = f"{RING}:VAC:VALVE:01:POSITION:RB"
GAUGE_RB = f"{RING}:VAC:GAUGE:01:PRESSURE:RB"

MAG_BAND = (-5.0, 5.0)
DRIVE_LIMITS = {MAG1_SP: MAG_BAND, MAG2_SP: MAG_BAND}
BOOT_VALUES = {MAG1_SP: 1.0, MAG1_RB: 1.0, MAG2_SP: 2.0, MAG2_RB: 2.0}

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


@pytest.fixture()
def facility(tmp_path: Path) -> Path:
    """A mounted data directory: a manifest, a machine file and a limits file.

    The whole file-backed source in one place, so a boot test says which
    facility it is booting and nothing about how the files are shaped.
    """
    (tmp_path / "channels.json").write_text(json.dumps({"channels": CHANNELS}))
    (tmp_path / "machine.json").write_text(
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
    (tmp_path / "channel_limits.json").write_text(
        json.dumps(
            {
                "defaults": {"writable": True},
                MAG1_SP: {"min_value": MAG_BAND[0], "max_value": MAG_BAND[1]},
                MAG2_SP: {"min_value": MAG_BAND[0], "max_value": MAG_BAND[1]},
            }
        )
    )
    (tmp_path / "active_scenarios").write_text("[]")
    return tmp_path


# --- the assembly, driven against a fake runner ---------------------------


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
    bpm_errors: Any = None
    corrector_gains: Any = None
    bound: dict[str, Any] | None = None

    def bind(self, records: dict[str, Any]) -> None:
        self.bound = records
        self.journal.append(("bind", records))

    def on_setpoint(self, address: str, value: float) -> None:  # pragma: no cover - identity only
        """Never called here: the fake runner enqueues nothing."""


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
    monkeypatch.setenv("VA_CHANNELS_FILE", "channels.json")
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

    if lattice == ep.LATTICE_BUILTIN:
        _install_fake_physics(monkeypatch, boot, orbit_solve_error=orbit_solve_error)

    ep.main()
    return boot


def _install_fake_physics(
    monkeypatch: pytest.MonkeyPatch, boot: Boot, *, orbit_solve_error: bool
) -> None:
    """Replace the ring model and the bridge, keeping their real error type.

    The lattice-backed branch is being asserted for its *composition* -- one
    model, shared, bound before the runner exists -- and solving a real ring
    would establish none of that while costing a full closed-orbit solve.
    """
    from osprey.services.virtual_accelerator.ioc import physics_bridge as bridge_module
    from osprey.services.virtual_accelerator.model import pyat as pyat_module

    def ring_model() -> Any:
        if orbit_solve_error:
            raise bridge_module.OrbitSolveError("no stable closed orbit")
        model = NullModel()
        boot.journal.append(("model", model))
        return model

    def physics_bridge(*, model: Any, bpm_errors: Any, corrector_gains: Any) -> FakeBridge:
        bridge = FakeBridge(
            journal=boot.journal,
            model=model,
            bpm_errors=bpm_errors,
            corrector_gains=corrector_gains,
        )
        boot.journal.append(("bridge", bridge))
        return bridge

    monkeypatch.setattr(pyat_module, "PyATRingModel", ring_model)
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
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
        assert boot.order() == [
            "logging",
            "model",
            "bridge",
            "bind",
            "engine-source",
            "runner",
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
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
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
        assert interval == ep.ENGINE_POLL_INTERVAL_S

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
        with_lattice = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
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
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
        bridge = boot.one("bridge")
        assert bridge.model is boot.one("model")
        assert boot.runner.model is bridge.model

    def test_the_physics_hook_is_the_bridges(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
        assert boot.runner.kwargs["on_setpoint"] == boot.one("bridge").on_setpoint

    def test_the_bridge_is_bound_to_the_coupled_partition(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
        assert boot.one("bridge").bound is boot.runner.records.pyat_coupled

    def test_setpoints_are_not_also_synced_into_the_engine(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Physics coupling flows through the bridge; a second path into the
        engine would be a second thing moving the same channels."""
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
        assert boot.engine_source.setpoint_echo_records is None

    def test_the_fault_seeds_reach_the_bridge(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(
            monkeypatch,
            facility,
            lattice=ep.LATTICE_BUILTIN,
            env={"VA_BPM_ERRORS": "BPM01:offset_x=1e-4", "VA_CORR_GAIN": "HCM01=1.5"},
        )
        bridge = boot.one("bridge")
        assert bridge.bpm_errors == {"BPM01": {"offset_x": 1e-4}}
        assert bridge.corrector_gains == {"HCM01": {"factor": 1.5}}

    def test_no_faults_are_passed_as_none_rather_than_an_empty_map(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        boot = _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN)
        bridge = boot.one("bridge")
        assert bridge.bpm_errors is None
        assert bridge.corrector_gains is None

    def test_a_lattice_with_no_stable_orbit_ends_the_boot(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        """Diagnosable rather than an opaque crash -- which is why the model
        itself never ends the process."""
        with pytest.raises(SystemExit, match="no stable closed orbit"):
            _boot(monkeypatch, facility, lattice=ep.LATTICE_BUILTIN, orbit_solve_error=True)


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

    def test_an_unknown_lattice_mode_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch, facility: Path
    ) -> None:
        with pytest.raises(SystemExit, match="VA_LATTICE"):
            _boot(monkeypatch, facility, lattice="maybe")


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

    def enqueue(values: dict[str, Any], done: Any = None, reset: bool = False) -> None:
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
