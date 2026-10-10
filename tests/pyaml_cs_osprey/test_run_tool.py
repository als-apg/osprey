"""``run_tool`` restores what an interrupted pyAML tool moved and reports every address.

A completed run leaves the machine as the tool left it. An interrupted one - a
``KeyboardInterrupt``, a ``False`` return, or any exception - writes each moved
address back through ``osprey.runtime.write_channel``, walking back under
``max_step``, and never forces a refused channel. Every run is a journaled guarded
run on the target, so a second one is busy and a missing guarded-run directory
stops it before the tool is called.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyaml.accelerator import Accelerator
from pyaml.common.constants import Action
from pyaml.tuning_tools.measurement_tool import MeasurementTool
from pyaml_test_lattice import lattices

import osprey.runtime
from osprey.errors import ChannelReadFailedError
from osprey.mcp_server.python_executor.executor import RESTORE_REPORT_TAG
from osprey.runtime.guarded_run import (
    JOURNAL_FILE_NAME,
    GuardedRunDirError,
    OspreyRunBusy,
    lock,
)
from osprey.runtime.journal import active_journals, read_pending_journal
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig
from pyaml_cs_osprey.catalog import parse_reference
from pyaml_cs_osprey.device import OspreyDevice
from pyaml_cs_osprey.errors import OspreyWriteRefused
from pyaml_cs_osprey.run_tool import REPORT_TAG, RestoreReport, run_tool
from tests.pyaml_cs_osprey.conftest import FakeRuntime as _Machine

LATTICE = Path(lattices["fodo_1gev_6d.json"])


@pytest.fixture
def machine(
    monkeypatch: pytest.MonkeyPatch,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[_Machine]:
    m = _Machine({"A": 1.0, "B": 2.0, "C": 3.0})
    yield m.install(monkeypatch, ("read_channels", "write_channel", "channel_limits"))
    assert active_journals() == ()


def test_the_report_tag_is_the_one_the_executor_files() -> None:
    """The executor files exactly the lines this tag starts."""
    assert REPORT_TAG == RESTORE_REPORT_TAG


class TestGuardedRun:
    """The tool runs inside the target's journaled guarded run, or not at all."""

    def test_each_address_is_journaled_durably_before_its_first_write(
        self, machine: _Machine, guarded_repo: Path
    ) -> None:
        """While the tool runs, the run's durable journal already holds the old value."""
        journal_path = guarded_repo / "var" / "guarded_run" / "live" / JOURNAL_FILE_NAME
        seen: list[dict[str, Any]] = []

        def tool() -> None:
            machine.set("A", 5.0)
            pending = read_pending_journal(journal_path)
            assert pending is not None
            seen.append(dict(pending.values))

        run_tool(tool)
        assert seen == [{"A": 1.0}]
        assert read_pending_journal(journal_path) is None

    def test_a_second_run_on_the_target_is_busy_and_never_calls_the_tool(
        self, machine: _Machine
    ) -> None:
        """Another holder of the target's lock refuses the run before the tool starts."""
        taken = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with lock("live"):
                taken.set()
                release.wait(CHILD_TIMEOUT_S)

        thread = threading.Thread(target=holder)
        thread.start()
        try:
            assert taken.wait(CHILD_TIMEOUT_S)
            called: list[str] = []
            with pytest.raises(OspreyRunBusy) as info:
                run_tool(lambda: called.append("ran"))
        finally:
            release.set()
            thread.join(CHILD_TIMEOUT_S)
        assert called == []
        assert info.value.pid == os.getpid()
        assert f"(pid {os.getpid()}, since " in str(info.value)
        assert machine.writes == []

    def test_no_guarded_run_directory_raises_before_the_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outside a deployment repo there is no lock to take, so nothing runs."""
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        monkeypatch.delenv(osprey.runtime.ENV_CONTROL_TARGET, raising=False)
        called: list[str] = []
        with pytest.raises(GuardedRunDirError, match="guarded runs need var/guarded_run"):
            run_tool(lambda: called.append("ran"))
        assert called == []


def _tagged(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len(REPORT_TAG) + 1 :])
        for line in text.splitlines()
        if line.startswith(REPORT_TAG + " ")
    ]


class TestOutcomes:
    """The method's return value or exception decides whether the journal is restored."""

    @pytest.mark.parametrize("result", [None, True, 0, "done"])
    def test_a_completed_run_leaves_the_machine_as_the_tool_left_it(
        self, machine: _Machine, result: Any
    ) -> None:
        """Anything but ``False`` is success: nothing written back, empty report."""

        def tool() -> Any:
            machine.set("A", 5.0)
            return result

        report = run_tool(tool)
        assert report == RestoreReport(aborted=False)
        assert machine.values["A"] == 5.0
        assert machine.writes == [("A", 5.0, {})]

    def test_a_false_return_is_an_abort_and_restores(self, machine: _Machine) -> None:
        """``measure()`` reports an interrupted run with ``False``."""

        def tool() -> bool:
            machine.set("A", 5.0)
            return False

        report = run_tool(tool)
        assert report.aborted is True
        assert report.restored == ["A"]
        assert machine.values["A"] == 1.0

    def test_a_callback_stop_raised_as_keyboard_interrupt_returns_the_report(
        self, machine: _Machine
    ) -> None:
        """pyAML aborts on a falsy callback with ``KeyboardInterrupt``; it does not escape."""

        def tool(callback: Callable[..., Any]) -> None:
            machine.set("B", 7.0)
            if not callback(Action.APPLY, {}):
                raise KeyboardInterrupt

        report = run_tool(tool, callback=lambda _action, _data: False)
        assert report.aborted is True
        assert report.restored == ["B"]
        assert machine.values["B"] == 2.0

    def test_a_real_keyboard_interrupt_restores_and_propagates_with_the_report(
        self, machine: _Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``KeyboardInterrupt`` no callback asked for is a Ctrl-C: restored, then re-raised."""

        def tool(callback: Callable[..., Any]) -> None:
            machine.set("B", 7.0)
            assert callback(Action.APPLY, {})
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt) as info:
            run_tool(tool, callback=lambda _action, _data: True)
        report = info.value.restore_report  # type: ignore[attr-defined]
        assert report.aborted is True
        assert report.restored == ["B"]
        assert machine.values["B"] == 2.0
        assert _tagged(capsys.readouterr().err) == [json.loads(report.to_json())]

    def test_a_keyboard_interrupt_without_a_callback_propagates(self, machine: _Machine) -> None:
        """With no callback in the run, nothing but a Ctrl-C raises ``KeyboardInterrupt``."""

        def tool() -> None:
            machine.set("B", 7.0)
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_tool(tool)
        assert machine.values["B"] == 2.0

    def test_an_exception_restores_and_reraises_carrying_the_report(
        self, machine: _Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Any other exception restores, attaches ``restore_report``, prints to stderr."""

        def tool() -> None:
            machine.set("A", 5.0)
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom") as info:
            run_tool(tool)
        report = info.value.restore_report  # type: ignore[attr-defined]
        assert isinstance(report, RestoreReport)
        assert report.restored == ["A"]
        assert machine.values["A"] == 1.0
        out, err = capsys.readouterr()
        assert _tagged(err) == [json.loads(report.to_json())]
        assert _tagged(out) == []

    def test_a_refused_write_reraises_osprey_write_refused_with_the_report(
        self, machine: _Machine
    ) -> None:
        """A tool stopped by an ``OspreyWriteRefused`` still restores what it had moved."""

        def tool() -> None:
            machine.set("A", 5.0)
            raise OspreyWriteRefused("posture is read-only", "B")

        with pytest.raises(OspreyWriteRefused) as info:
            run_tool(tool)
        assert info.value.restore_report.restored == ["A"]  # type: ignore[attr-defined]

    @pytest.mark.parametrize("exc_type", [SystemExit, GeneratorExit])
    def test_exit_exceptions_restore_and_propagate(
        self, machine: _Machine, exc_type: type[BaseException]
    ) -> None:
        """``SystemExit`` and ``GeneratorExit`` restore, then propagate unchanged."""

        def tool() -> None:
            machine.set("C", 9.0)
            raise exc_type

        with pytest.raises(exc_type):
            run_tool(tool)
        assert machine.values["C"] == 3.0

    def test_arguments_and_the_callback_reach_the_method(self, machine: _Machine) -> None:
        """Positional and keyword arguments pass through; ``callback`` only when given."""
        seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def tool(*args: Any, **kwargs: Any) -> None:
            seen.append((args, kwargs))

        calls: list[Any] = []

        def cb(action: Action, _data: dict[str, Any]) -> bool:
            calls.append(action)
            return True

        run_tool(tool, 1, 2, n_step=3)
        run_tool(tool, callback=cb)
        assert seen[0] == ((1, 2), {"n_step": 3})
        assert seen[1][0] == () and list(seen[1][1]) == ["callback"]
        assert seen[1][1]["callback"](Action.APPLY, {}) is True
        assert calls == [Action.APPLY]
        assert machine.batch_reads == []


class TestRestore:
    """Restore reads once, skips unchanged addresses and never forces a refused one."""

    def test_addresses_already_back_are_unchanged_and_not_written(self, machine: _Machine) -> None:
        """One read decides; an address the tool put back itself is not written again."""

        def tool() -> bool:
            machine.set("A", 5.0)
            machine.set("B", 6.0)
            machine.set("A", 1.0)
            return False

        report = run_tool(tool)
        assert report.unchanged == ["A"]
        assert report.restored == ["B"]
        restore_writes = machine.writes[3:]
        assert restore_writes == [("B", 2.0, {})]
        assert machine.batch_reads[-1] == ["A", "B"]

    def test_a_distance_beyond_max_step_walks_back_in_three_equal_steps(
        self, machine: _Machine
    ) -> None:
        """Distance 2.5 with ``max_step`` 1.0 takes three steps, the last exactly home."""
        machine.limits["A"] = ChannelLimitsConfig("A", max_step=1.0, writable=True)

        def tool() -> bool:
            machine.set("A", 3.5)
            return False

        report = run_tool(tool)
        steps = [v for a, v, _ in machine.writes[1:] if a == "A"]
        assert steps == pytest.approx([3.5 - 2.5 / 3, 3.5 - 5.0 / 3, 1.0])
        assert steps[-1] == 1.0
        assert all(abs(b - a) <= 1.0 for a, b in zip([3.5, *steps], steps, strict=False))
        assert report.restored == ["A"]

    @pytest.mark.parametrize(("home", "moved"), [(0.7, 0.0), (0.0, 0.7), (1.7, 1.0), (1.0, 1.7)])
    def test_no_walk_back_step_exceeds_max_step_after_rounding(
        self, machine: _Machine, home: float, moved: float
    ) -> None:
        """Equal steps that round above ``max_step`` are resized, so the restore lands home."""
        machine.values["A"] = home
        machine.limits["A"] = ChannelLimitsConfig("A", max_step=0.1, writable=True)

        def tool() -> bool:
            machine.set("A", moved)
            machine.refuse["A"] = lambda value: abs(value - machine.values["A"]) > 0.1
            return False

        report = run_tool(tool)
        assert report.refused == []
        assert report.restored == ["A"]
        assert machine.values["A"] == home

    def test_a_refused_step_stops_that_address_and_reports_where_it_was_left(
        self, machine: _Machine
    ) -> None:
        """The walk-back stops at the first refusal; the other addresses still restore."""
        machine.limits["A"] = ChannelLimitsConfig("A", max_step=1.0, writable=True)

        def tool() -> bool:
            machine.set("A", 4.0)
            machine.set("B", 8.0)
            return False

        machine.refuse["A"] = lambda value: value < 2.5
        report = run_tool(tool)
        assert report.refused == [("A", "step too large", 3.0)]
        assert machine.values["A"] == 3.0
        assert report.restored == ["B"]
        assert machine.values["B"] == 2.0

    def test_restore_after_scaled_write_is_native(self, machine: _Machine) -> None:
        """A device write in SI journals and restores the native value, not the SI one."""
        dev = OspreyDevice(parse_reference("A[mm]"))

        def tool() -> bool:
            dev.set(5.0e-3)
            return False

        report = run_tool(tool)
        assert machine.writes == [("A", pytest.approx(5.0), {}), ("A", 1.0, {})]
        assert report.restored == ["A"]
        assert machine.values["A"] == 1.0

    def test_an_unconfirmed_write_back_is_failed(self, machine: _Machine) -> None:
        """A write back attempted and not confirmed lands in ``failed``."""

        def tool() -> bool:
            machine.set("C", 4.0)
            machine.fail.add("C")
            return False

        report = run_tool(tool)
        assert [address for address, _ in report.failed] == ["C"]
        assert report.restored == []

    def test_an_unreadable_address_costs_one_more_read_for_the_rest(
        self, machine: _Machine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failed channel is reported with its own cause; the rest are read once more."""
        batch_read = machine.read_channels

        def read_channels(addresses: Sequence[str], **kwargs: Any) -> list[Any]:
            if machine.values.get("B") is None and "B" in addresses:
                machine.batch_reads.append(list(addresses))
                raise ChannelReadFailedError(["B"], causes={"B": TimeoutError("timed out")})
            return batch_read(addresses, **kwargs)

        monkeypatch.setattr(osprey.runtime, "read_channels", read_channels)

        def tool() -> bool:
            machine.set("A", 5.0)
            machine.set("B", 6.0)
            machine.set("C", 7.0)
            machine.values["B"] = None
            return False

        report = run_tool(tool)
        assert machine.batch_reads[-2:] == [["A", "B", "C"], ["A", "C"]]
        assert report.failed == [("B", "Read failed for 1 channel(s): B (TimeoutError: timed out)")]
        assert report.restored == ["A", "C"]
        assert (machine.values["A"], machine.values["C"]) == (1.0, 3.0)

    def test_an_outer_abort_restores_a_nested_successful_write(self, machine: _Machine) -> None:
        """A nested ``run_tool`` journals into the outer run too."""

        def inner() -> None:
            machine.set("A", 5.0)

        def outer() -> bool:
            inner_report = run_tool(inner)
            assert inner_report.aborted is False
            assert machine.values["A"] == 5.0
            return False

        report = run_tool(outer)
        assert report.aborted is True
        assert report.restored == ["A"]
        assert machine.values["A"] == 1.0

    def test_the_tagged_line_is_printed_once_per_return(
        self, machine: _Machine, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each return prints exactly one ``OSPREY_PYAML_RESTORE`` line with the report."""

        def tool() -> bool:
            machine.set("B", 6.0)
            return False

        report = run_tool(tool)
        lines = _tagged(capsys.readouterr().out)
        assert lines == [json.loads(report.to_json())]
        assert lines[0] == {
            "restored": ["B"],
            "unchanged": [],
            "refused": [],
            "failed": [],
            "aborted": True,
            "deadline_guard": False,
        }


def _fodo_design() -> Accelerator:
    """A design-mode FODO accelerator with a tune response matrix and chromaticity tools."""
    quads = [f"QF_{i:03d}" for i in range(1, 17)]
    devices: list[dict[str, Any]] = [
        {
            "type": "pyaml.magnet.quadrupole",
            "name": q,
            "model": {"type": "pyaml.magnet.identity_model", "unit": "1/m", "physics": f"{q}:K"},
        }
        for q in quads
    ]
    devices += [
        {
            "type": "pyaml.rf.rf_plant",
            "name": "RF",
            "masterclock": "RF:freq",
            "transmitters": [
                {
                    "type": "pyaml.rf.rf_transmitter",
                    "name": "RFTRA",
                    "cavities": ["RF_001"],
                    "harmonic": 1,
                    "distribution": 1,
                    "voltage": "RF:volt",
                }
            ],
        },
        {
            "type": "pyaml.diagnostics.tune_monitor",
            "name": "BETATRON_TUNE",
            "tune_h": "TUNE:h",
            "tune_v": "TUNE:v",
        },
        {
            "type": "pyaml.tuning_tools.tune_response_matrix",
            "name": "DEFAULT_TUNE_RESPONSE_MATRIX",
            "quad_array_name": "QForTune",
            "betatron_tune_name": "BETATRON_TUNE",
            "quad_delta": 1e-3,
        },
        {
            "type": "pyaml.tuning_tools.chromaticity_monitor",
            "name": "CHROMATICITY_MONITOR",
            "betatron_tune_name": "BETATRON_TUNE",
            "rf_plant_name": "RF",
            "fit_order": 1,
            "n_avg_meas": 1,
            "n_step": 3,
            "sleep_between_meas": 0,
            "sleep_between_step": 0,
            "e_delta": 1e-3,
            "max_e_delta": 1e-2,
        },
        {
            "type": "pyaml.tuning_tools.chromaticity",
            "name": "DEFAULT_CHROMATICITY_CORRECTION",
            "sextu_array_name": "SX",
            "chromaticity_monitor_name": "CHROMATICITY_MONITOR",
            "response_matrix": "absent.json",
        },
    ]
    return Accelerator.from_dict(
        {
            "type": "pyaml.accelerator",
            "facility": "test",
            "machine": "sr",
            "energy": 1e9,
            "alphac": 0.0918,
            "simulators": [
                {"type": "pyaml.lattice.simulator", "lattice": str(LATTICE), "name": "design"}
            ],
            "arrays": [{"type": "pyaml.arrays.magnet", "name": "QForTune", "elements": quads}],
            "devices": devices,
        }
    )


@pytest.fixture(scope="module")
def sr() -> Accelerator:
    return _fodo_design()


def _stop(_action: Action, _data: dict[str, Any]) -> bool:
    return False


@pytest.mark.usefixtures("machine")
class TestPyamlTools:
    """Real pyAML tools in design mode abort through ``run_tool`` instead of raising."""

    def test_a_trm_measure_stopped_at_init_returns_an_aborted_report(self, sr: Accelerator) -> None:
        """``trm.measure`` sends INIT outside its ``try``; the interrupt becomes a report."""
        report = run_tool(sr.design.trm.measure, callback=_stop)
        assert report.aborted is True
        assert sr.design.trm._callback is None

    def test_a_chromaticity_measure_stopped_returns_an_aborted_report(
        self, sr: Accelerator
    ) -> None:
        """The chromaticity monitor reports its interrupted run with ``False``."""
        monitor = sr.design.get_chromaticity_monitor("CHROMATICITY_MONITOR")
        report = run_tool(monitor.measure, callback=_stop)
        assert report.aborted is True
        assert monitor._callback is None

    def test_a_later_plain_readback_is_unaffected_by_a_stale_callback(
        self, sr: Accelerator
    ) -> None:
        """The monitor sends INIT before registering; a cleared slot keeps that INIT inert."""
        monitor = sr.design.get_chromaticity_monitor("CHROMATICITY_MONITOR")
        run_tool(monitor.measure, callback=_stop)
        chroma = sr.design.chromaticity.readback()
        assert np.all(np.isfinite(chroma))


class _Monitor(MeasurementTool):
    """A measurement tool that only registers the callback it is given."""

    def measure(self, callback: Callable[..., Any] | None = None) -> bool:
        self._register_callback(callback)
        return True


class _ResponseMatrix(MeasurementTool):
    """Registers the callback on itself and on its chromaticity monitor, like ``crm``."""

    def __init__(self, name: str, monitor: _Monitor) -> None:
        super().__init__(name)
        self._monitor = monitor

    @property
    def chromaticity_monitor(self) -> _Monitor:
        return self._monitor

    def measure(self, callback: Callable[..., Any] | None = None) -> bool:
        self._register_callback(callback)
        return self.chromaticity_monitor.measure(callback=callback)


@pytest.mark.usefixtures("machine")
def test_the_chromaticity_monitor_slot_is_cleared_too() -> None:
    """A response matrix hands its callback to its monitor; neither slot survives the run."""
    monitor = _Monitor("CHROMATICITY_MONITOR")
    crm = _ResponseMatrix("CRM", monitor)
    report = run_tool(crm.measure, callback=lambda _action, _data: True)
    assert report.aborted is False
    assert crm._callback is None
    assert monitor._callback is None


#: Seconds a stub child may take to import and reach a phase, or to exit afterwards.
CHILD_TIMEOUT_S = 60.0

_SIGINT_CHILD = textwrap.dedent(
    """
    import json, os, sys, time

    import osprey.runtime
    from osprey.runtime.journal import journaled_write
    from pyaml_cs_osprey.run_tool import run_tool

    mode, workdir = sys.argv[1], sys.argv[2]
    values = {"A": 1.0, "B": 2.0, "C": 3.0}
    restoring = False

    def hold(name):
        # Announce the phase once, then wait for the parent's go (sent after SIGINT).
        ready = os.path.join(workdir, name + ".ready")
        if os.path.exists(ready):
            return
        open(ready, "w").close()
        go = os.path.join(workdir, name + ".go")
        while not os.path.exists(go):
            time.sleep(0.01)

    def read_channels(addresses, *, timeout=None):
        return [values[a] for a in addresses]

    def write_channel(address, value, **kwargs):
        if restoring:
            hold("restore")
        values[address] = value

    osprey.runtime.read_channels = read_channels
    osprey.runtime.write_channel = write_channel
    osprey.runtime.channel_limits = lambda address: None

    def put(address, value):
        journaled_write([address], lambda: osprey.runtime.write_channel(address, value))

    def span_tool(callback):
        # pyAML style: a falsy callback raises KeyboardInterrupt.
        put("A", 5.0)
        hold("span")
        if not callback("apply", {}):
            raise KeyboardInterrupt
        put("B", 6.0)
        return True

    def returning_tool(callback):
        # Chromaticity-monitor style: a falsy callback returns False.
        put("A", 5.0)
        hold("span")
        if not callback("apply", {}):
            return False
        put("B", 6.0)
        return True

    def plain_tool():
        put("A", 5.0)
        hold("span")
        put("B", 6.0)

    def restore_tool():
        global restoring
        put("A", 5.0)
        put("B", 6.0)
        restoring = True
        return False

    def outer_tool():
        put("A", 5.0)
        run_tool(span_inner)
        put("C", 9.0)
        return True

    def span_inner(callback):
        put("B", 6.0)
        hold("span")
        if not callback("apply", {}):
            raise KeyboardInterrupt
        return True

    tools = {
        "span": span_tool,
        "after": returning_tool,
        "plain": plain_tool,
        "restore": restore_tool,
        "nested": outer_tool,
    }
    try:
        report = run_tool(tools[mode])
        print("RETURNED", report.to_json(), flush=True)
        if mode == "after":
            osprey.runtime.write_channel("C", 99.0)
            print("AFTER", flush=True)
    except KeyboardInterrupt as exc:
        print("INTERRUPTED", exc.restore_report.to_json(), flush=True)
        raise
    finally:
        with open(os.path.join(workdir, "values.json"), "w") as handle:
            json.dump(values, handle)
    """
)


class _SigintChild:
    """A stub child running ``run_tool`` over a dict machine; the parent sends SIGINT.

    The child runs in a deployment repo of its own, so its guarded run has a
    target directory to lock.
    """

    def __init__(self, workdir: Path, mode: str) -> None:
        self.workdir = workdir
        script = workdir / "sigint_child.py"
        script.write_text(_SIGINT_CHILD, encoding="utf-8")
        repo = workdir / "repo"
        repo.mkdir(exist_ok=True)
        (repo / "profile.yml").write_text("name: probe\n", encoding="utf-8")
        env = {**os.environ, osprey.runtime.ENV_CONTROL_TARGET: "live"}
        for name in (
            "OSPREY_EXECUTION_DEADLINE",
            "OSPREY_EXECUTION_MODE",
            osprey.runtime.ENV_CONTROL_TARGET_GENERATION,
        ):
            env.pop(name, None)
        self.proc = subprocess.Popen(
            [sys.executable, str(script), mode, str(workdir)],
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def interrupt_at(self, phase: str) -> None:
        """Wait until the child reaches ``phase``, send SIGINT, then let it go on."""
        ready = self.workdir / f"{phase}.ready"
        deadline = time.monotonic() + CHILD_TIMEOUT_S
        while not ready.exists():
            if self.proc.poll() is not None:
                out, err = self.proc.communicate()
                pytest.fail(f"child exited {self.proc.returncode} before {phase}:\n{out}{err}")
            if time.monotonic() > deadline:
                pytest.fail(f"child did not reach {phase} in time")
            time.sleep(0.01)
        self.proc.send_signal(signal.SIGINT)
        (self.workdir / f"{phase}.go").touch()

    def finish(self) -> tuple[int, str, str, dict[str, float]]:
        """The child's exit status, stdout, stderr and final machine values."""
        out, err = self.proc.communicate(timeout=CHILD_TIMEOUT_S)
        values = json.loads((self.workdir / "values.json").read_text(encoding="utf-8"))
        return self.proc.returncode, out, err, values

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.communicate()


@pytest.fixture
def sigint_child(tmp_path: Path) -> Iterator[Callable[[str], _SigintChild]]:
    """Starts stub children; any still running at teardown is killed."""
    started: list[_SigintChild] = []

    def start(mode: str) -> _SigintChild:
        child = _SigintChild(tmp_path, mode)
        started.append(child)
        return child

    yield start
    for child in started:
        child.kill()


def _line(out: str, tag: str) -> dict[str, Any]:
    (line,) = [line for line in out.splitlines() if line.startswith(tag + " ")]
    return json.loads(line[len(tag) + 1 :])


HOME = {"A": 1.0, "B": 2.0, "C": 3.0}


class TestSigintAbort:
    """A Ctrl-C aborts the guarded run, restores it and ends the script."""

    def test_sigint_during_span_aborts(self, sigint_child: Callable[[str], _SigintChild]) -> None:
        """The chained callback stops the tool; restored, reported on stderr, re-raised."""
        child = sigint_child("span")
        child.interrupt_at("span")
        code, out, err, values = child.finish()
        assert code != 0
        report = _line(out, "INTERRUPTED")
        assert report["aborted"] is True
        assert report["restored"] == ["A"]
        assert values == HOME
        assert _tagged(err) == [report]
        assert _tagged(out) == []

    def test_sigint_during_restore_restores_all(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """A SIGINT while writing back is ignored: every address restored, one tagged line."""
        child = sigint_child("restore")
        child.interrupt_at("restore")
        code, out, err, values = child.finish()
        assert code == 0, err
        report = _line(out, "RETURNED")
        assert report["restored"] == ["A", "B"]
        assert values == HOME
        assert _tagged(out) + _tagged(err) == [report]

    def test_sigint_without_callback_restores_and_raises(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """A tool that takes no callback gets ``KeyboardInterrupt`` at once, then the restore."""
        child = sigint_child("plain")
        child.interrupt_at("span")
        code, out, err, values = child.finish()
        assert code != 0
        report = _line(out, "INTERRUPTED")
        assert report["aborted"] is True
        assert report["restored"] == ["A"]
        assert values == HOME
        assert _tagged(err) == [report]

    def test_sigint_in_nested_run_aborts_outer(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """A SIGINT inside a nested run aborts it and every enclosing run; nothing later runs."""
        child = sigint_child("nested")
        child.interrupt_at("span")
        code, out, err, values = child.finish()
        assert code != 0
        report = _line(out, "INTERRUPTED")
        assert report["restored"] == ["A"]
        assert report["unchanged"] == ["B"]
        assert values == HOME
        tagged = _tagged(err)
        assert len(tagged) == 2
        assert tagged[0]["restored"] == ["B"]
        assert tagged[1] == report

    def test_run_tool_off_main_thread_no_handler(self, machine: _Machine) -> None:
        """Off the main thread no handler is installed; on it, the previous one comes back."""
        before = signal.getsignal(signal.SIGINT)
        seen: list[Any] = []
        errors: list[BaseException] = []

        def tool(callback: Callable[..., Any]) -> None:
            seen.append(signal.getsignal(signal.SIGINT))
            seen.append(callback(Action.APPLY, {}))
            machine.set("A", 5.0)

        def worker() -> None:
            try:
                run_tool(tool)
            except BaseException as exc:  # surfaced by the assertion
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(CHILD_TIMEOUT_S)
        assert errors == []
        assert seen == [before, True]

        seen.clear()
        run_tool(tool)
        assert seen[0] is not before
        assert seen[1] is True
        assert signal.getsignal(signal.SIGINT) is before

    def test_sigint_abort_stops_script_after_restore(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """A tool that reports the stop with ``False`` still ends the script after the restore."""
        child = sigint_child("after")
        child.interrupt_at("span")
        code, out, err, values = child.finish()
        assert code != 0
        assert "AFTER" not in out
        assert "RETURNED" not in out
        report = _line(out, "INTERRUPTED")
        assert report["aborted"] is True
        assert report["restored"] == ["A"]
        assert values == HOME
        assert _tagged(err) == [report]
