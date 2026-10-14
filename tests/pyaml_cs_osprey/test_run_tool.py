"""``run_tool`` runs a pyAML tool under the run lock and lists every setpoint it moved.

The call holds the target's run lock, joining the process's hold, so a second
run is busy and a missing guarded-run directory stops it before the tool is
called. Before the first write of each address its value is recorded; nothing
is written back. A completed run, a stop, an exception and a Ctrl-C each report
the setpoints the run moved with their values before the call: on the returned
``ToolRun`` or, on an escaping exception, as ``moved_setpoints``. A Ctrl-C stops
every tool of the process's run that takes a callback, from any thread.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyaml.accelerator import Accelerator
from pyaml.common.constants import Action
from pyaml.tuning_tools.measurement_tool import MeasurementTool
from pyaml_test_lattice import lattices

import osprey.runtime
import osprey.runtime._run_state as _run_state
import pyaml_cs_osprey.run_tool as run_tool_mod
from osprey.runtime.guarded_run import GuardedRunDirError, OspreyRunBusy, lock
from pyaml_cs_osprey import _run
from pyaml_cs_osprey.errors import OspreyWriteRefused
from pyaml_cs_osprey.run_tool import ToolRun, run_tool
from tests.pyaml_cs_osprey.conftest import FakeRuntime as _Machine

LATTICE = Path(lattices["fodo_1gev_6d.json"])

#: Seconds a stub child may take to import and reach a phase, or to exit afterwards.
CHILD_TIMEOUT_S = 60.0

#: Setpoints of the dict machine before any tool ran.
HOME = {"A": 1.0, "B": 2.0, "C": 3.0}


@pytest.fixture
def machine(
    monkeypatch: pytest.MonkeyPatch,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[_Machine]:
    m = _Machine(dict(HOME))
    yield m.install(monkeypatch, ("read_channels", "write_channel", "channel_limits"))
    assert _run.moved() == {}, "no pyAML run outlives its call"


def test_run_tool_imports_no_private_runtime_name() -> None:
    """The wrapper reaches the runtime only through its public names."""
    tree = ast.parse(Path(run_tool_mod.__file__).read_text(encoding="utf-8"))
    private = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("osprey.runtime")
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert private == []


class TestRunLock:
    """The tool runs under the target's run lock, or not at all."""

    def test_each_addresss_first_value_is_recorded_before_its_first_write(
        self, machine: _Machine
    ) -> None:
        """While the tool runs, the run already holds each address's value before its write."""
        seen: list[dict[str, Any]] = []

        def tool() -> None:
            machine.set("A", 5.0)
            seen.append(_run.moved())
            machine.set("A", 6.0)
            machine.set("B", 7.0)

        run = run_tool(tool)

        assert seen == [{"A": 1.0}]
        assert run == ToolRun(completed=True, moved={"A": 1.0, "B": 2.0})
        assert list(run.moved) == ["A", "B"], "in first-write order"
        assert machine.values == {"A": 6.0, "B": 7.0, "C": 3.0}

    def test_a_second_run_on_the_target_is_busy_and_never_calls_the_tool(
        self, machine: _Machine, guarded_repo: Path
    ) -> None:
        """Another holder of the target's lock refuses the run before the tool starts."""
        child = textwrap.dedent(
            """
            import sys
            from osprey.runtime.guarded_run import lock

            with lock("live"):
                print("held", flush=True)
                sys.stdin.readline()
            """
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", child],
            cwd=guarded_repo,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        called: list[str] = []
        try:
            assert holder.stdout is not None and holder.stdin is not None
            assert holder.stdout.readline().strip() == "held", f"exit {holder.poll()}"
            with pytest.raises(OspreyRunBusy) as info:
                run_tool(lambda: called.append("ran"))
        finally:
            if holder.stdin is not None:
                holder.stdin.close()
            holder.wait(timeout=CHILD_TIMEOUT_S)
        assert called == []
        assert info.value.pid == holder.pid
        assert f"(pid {holder.pid}, since " in str(info.value)
        assert machine.writes == []

    def test_run_tool_from_a_thread_runs_under_the_callers_lock(self, machine: _Machine) -> None:
        """A tool run from a helper thread joins the caller's hold; nothing is written back."""
        holds: list[Any] = []
        outcomes: list[Any] = []

        def tool() -> None:
            holds.append(_run_state.STATE.hold)
            machine.set("A", 5.0)

        def failing() -> None:
            machine.set("B", 7.0)
            raise RuntimeError("the tool failed")

        def worker(method: Callable[[], None]) -> Callable[[], None]:
            def run() -> None:
                try:
                    outcomes.append(run_tool(method))
                except BaseException as exc:  # surfaced by the assertion
                    outcomes.append(exc)

            return run

        with lock("live"):
            hold = _run_state.STATE.hold
            for method in (tool, failing):
                thread = threading.Thread(target=worker(method))
                thread.start()
                thread.join(CHILD_TIMEOUT_S)
                assert not thread.is_alive()
            assert _run_state.STATE.hold is hold, "the threads released nothing"

        assert holds == [hold]
        completed, failed = outcomes
        assert completed == ToolRun(completed=True, moved={"A": 1.0})
        assert isinstance(failed, RuntimeError)
        assert failed.moved_setpoints == {"B": 2.0}  # type: ignore[attr-defined]
        assert machine.values == {"A": 5.0, "B": 7.0, "C": 3.0}

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


class _Tool:
    """A pyAML-shaped measurement: step, then callback; a falsy callback aborts."""

    def __init__(self, machine: _Machine, steps: list[tuple[str, float]]) -> None:
        self.machine = machine
        self.steps = steps
        self.returned: list[Any] = []

    def measure(self, callback: Callable[..., Any] | None = None) -> bool:
        for address, value in self.steps:
            self.machine.set(address, value)
            if callback is not None:
                ok = callback(Action.APPLY, {"source": self})
                self.returned.append(ok)
                if not ok:
                    raise KeyboardInterrupt
        return True


class TestOutcomes:
    """The method's return value or exception decides the outcome; nothing is written back."""

    @pytest.mark.parametrize("result", [None, True, 0, "done"])
    def test_a_completed_run_leaves_the_machine_as_the_tool_left_it(
        self, machine: _Machine, result: Any
    ) -> None:
        """Anything but ``False`` is success: nothing written back, what moved listed."""

        def tool() -> Any:
            machine.set("A", 5.0)
            return result

        run = run_tool(tool)
        assert run == ToolRun(completed=True, moved={"A": 1.0})
        assert machine.values["A"] == 5.0
        assert machine.writes == [("A", 5.0, {})]

    def test_a_false_return_is_a_stop_that_leaves_the_machine_where_it_stopped(
        self, machine: _Machine
    ) -> None:
        """``measure()`` reports an interrupted run with ``False``."""

        def tool() -> bool:
            machine.set("A", 5.0)
            return False

        assert run_tool(tool) == ToolRun(completed=False, moved={"A": 1.0})
        assert machine.values["A"] == 5.0
        assert machine.writes == [("A", 5.0, {})]

    def test_a_callback_stop_raised_as_keyboard_interrupt_returns_a_stopped_run(
        self, machine: _Machine
    ) -> None:
        """pyAML aborts on a falsy callback with ``KeyboardInterrupt``; it does not escape."""

        def tool(callback: Callable[..., Any]) -> None:
            machine.set("B", 7.0)
            if not callback(Action.APPLY, {}):
                raise KeyboardInterrupt

        run = run_tool(tool, callback=lambda _action, _data: False)
        assert run == ToolRun(completed=False, moved={"B": 2.0})
        assert machine.values["B"] == 7.0

    @pytest.mark.parametrize("says", [False, None, 0])
    def test_a_falsy_caller_result_stops_the_tool(self, machine: _Machine, says: Any) -> None:
        tool = _Tool(machine, [("A", 5.0), ("B", 6.0)])
        seen: list[Any] = []

        def caller(action: Any, data: Any) -> Any:
            seen.append((action, data["source"]))
            return says

        run = run_tool(tool.measure, callback=caller)

        assert seen == [(Action.APPLY, tool)]
        assert tool.returned == [says]
        assert run == ToolRun(completed=False, moved={"A": 1.0})
        assert machine.values == {**HOME, "A": 5.0}

    def test_a_truthy_caller_result_continues(self, machine: _Machine) -> None:
        tool = _Tool(machine, [("A", 5.0)])

        run = run_tool(tool.measure, callback=lambda _a, _d: "yes")

        assert tool.returned == ["yes"]
        assert run.completed is True

    def test_a_real_keyboard_interrupt_propagates_with_what_it_moved(
        self, machine: _Machine
    ) -> None:
        """A ``KeyboardInterrupt`` no callback asked for propagates; nothing is put back."""

        def tool(callback: Callable[..., Any]) -> None:
            machine.set("B", 7.0)
            assert callback(Action.APPLY, {})
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt) as info:
            run_tool(tool, callback=lambda _action, _data: True)
        assert info.value.moved_setpoints == {"B": 2.0}  # type: ignore[attr-defined]
        assert machine.values["B"] == 7.0

    def test_a_keyboard_interrupt_without_a_callback_propagates(self, machine: _Machine) -> None:
        """With no callback in the run, nothing but a Ctrl-C raises ``KeyboardInterrupt``."""

        def tool() -> None:
            machine.set("B", 7.0)
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_tool(tool)
        assert machine.values["B"] == 7.0

    def test_an_exception_reraises_carrying_what_it_moved(self, machine: _Machine) -> None:
        """Any other exception carries ``moved_setpoints``; nothing is written back."""

        def tool() -> None:
            machine.set("A", 5.0)
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom") as info:
            run_tool(tool)
        assert info.value.moved_setpoints == {"A": 1.0}  # type: ignore[attr-defined]
        assert machine.values["A"] == 5.0
        assert machine.writes == [("A", 5.0, {})]

    def test_a_refused_write_reraises_osprey_write_refused_with_what_it_moved(
        self, machine: _Machine
    ) -> None:
        """A tool stopped by an ``OspreyWriteRefused`` lists what it had moved."""

        def tool() -> None:
            machine.set("A", 5.0)
            raise OspreyWriteRefused("posture is read-only", "B")

        with pytest.raises(OspreyWriteRefused) as info:
            run_tool(tool)
        assert info.value.moved_setpoints == {"A": 1.0}  # type: ignore[attr-defined]
        assert machine.values["A"] == 5.0

    @pytest.mark.parametrize("exc_type", [SystemExit, GeneratorExit])
    def test_exit_exceptions_propagate_with_what_they_moved(
        self, machine: _Machine, exc_type: type[BaseException]
    ) -> None:
        """``SystemExit`` and ``GeneratorExit`` propagate carrying ``moved_setpoints``."""

        def tool() -> None:
            machine.set("C", 9.0)
            raise exc_type

        with pytest.raises(exc_type) as info:
            run_tool(tool)
        assert info.value.moved_setpoints == {"C": 3.0}  # type: ignore[attr-defined]
        assert machine.values["C"] == 9.0

    def test_a_nested_run_s_stop_does_not_stop_the_outer_run(self, machine: _Machine) -> None:
        """Each call reports its own outcome; the record is the process's one run."""
        inner_runs: list[ToolRun] = []

        def inner(callback: Callable[..., Any]) -> None:
            machine.set("B", 6.0)
            if not callback(Action.APPLY, {}):
                raise KeyboardInterrupt

        def outer() -> bool:
            machine.set("A", 5.0)
            inner_runs.append(run_tool(inner, callback=lambda _a, _d: False))
            machine.set("C", 9.0)
            return True

        run = run_tool(outer)

        assert inner_runs == [ToolRun(completed=False, moved={"A": 1.0, "B": 2.0})]
        assert run == ToolRun(completed=True, moved={"A": 1.0, "B": 2.0, "C": 3.0})
        assert machine.values == {"A": 5.0, "B": 6.0, "C": 9.0}

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

    def test_an_abort_request_is_a_field_of_the_pyaml_run(self, machine: _Machine) -> None:
        """A request from any thread reaches the run's opening thread; the next run starts clear."""
        assert _run.request_abort() is False
        seen: list[bool] = []

        def tool() -> None:
            requested: list[bool] = []
            helper = threading.Thread(target=lambda: requested.append(_run.request_abort()))
            helper.start()
            helper.join(CHILD_TIMEOUT_S)
            seen.extend([*requested, _run.abort_requested()])
            machine.set("A", 5.0)

        with pytest.raises(KeyboardInterrupt) as info:
            run_tool(tool)

        assert seen == [True, True]
        assert info.value.moved_setpoints == {"A": 1.0}  # type: ignore[attr-defined]
        assert run_tool(lambda: seen.append(_run.abort_requested())).completed is True
        assert seen[-1] is False


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
    """Real pyAML tools in design mode stop through ``run_tool`` instead of raising."""

    def test_a_trm_measure_stopped_at_init_returns_a_stopped_run(self, sr: Accelerator) -> None:
        """``trm.measure`` sends INIT outside its ``try``; the interrupt becomes a stopped run."""
        run = run_tool(sr.design.trm.measure, callback=_stop)
        assert run.completed is False
        assert sr.design.trm._callback is None

    def test_a_chromaticity_measure_stopped_returns_a_stopped_run(self, sr: Accelerator) -> None:
        """The chromaticity monitor reports its interrupted run with ``False``."""
        monitor = sr.design.get_chromaticity_monitor("CHROMATICITY_MONITOR")
        run = run_tool(monitor.measure, callback=_stop)
        assert run.completed is False
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
    run = run_tool(crm.measure, callback=lambda _action, _data: True)
    assert run.completed is True
    assert crm._callback is None
    assert monitor._callback is None


_SIGINT_CHILD = textwrap.dedent(
    """
    import json, os, sys, threading, time

    import osprey.runtime
    from pyaml_cs_osprey import _run
    from pyaml_cs_osprey.run_tool import run_tool

    mode, workdir, bound = sys.argv[1], sys.argv[2], float(sys.argv[3])
    values = {"A": 1.0, "B": 2.0, "C": 3.0}
    helper = {"falsy": False, "caught": False}

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
        values[address] = value

    osprey.runtime.read_channels = read_channels
    osprey.runtime.write_channel = write_channel
    osprey.runtime.channel_limits = lambda address: None

    def same(exc, *_addresses):
        return exc

    def put(address, value):
        _run.write(
            [address],
            lambda **kw: osprey.runtime.write_channel(address, value, **kw),
            same,
            map_read=same,
            confirm=None,
        )

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

    def helper_tool(callback):
        # Steps until its callback says stop, or writes C once the bound passes.
        put("B", 6.0)
        hold("span")
        until = time.monotonic() + bound
        while time.monotonic() < until:
            if not callback("apply", {}):
                helper["falsy"] = True
                raise KeyboardInterrupt
            time.sleep(0.01)
        put("C", 7.0)
        return True

    def in_helper():
        try:
            run_tool(helper_tool)
        except KeyboardInterrupt:
            helper["caught"] = True

    def main_tool(callback):
        put("A", 5.0)
        thread = threading.Thread(target=in_helper)
        thread.start()
        thread.join()
        if not callback("apply", {}):
            raise KeyboardInterrupt
        return True

    tools = {
        "span": span_tool,
        "after": returning_tool,
        "plain": plain_tool,
        "nested": outer_tool,
        "helper": main_tool,
    }
    try:
        run = run_tool(tools[mode])
        print("RETURNED", json.dumps({"completed": run.completed, "moved": run.moved}), flush=True)
        if mode == "after":
            osprey.runtime.write_channel("C", 99.0)
            print("AFTER", flush=True)
    except KeyboardInterrupt as exc:
        print("INTERRUPTED", json.dumps({"moved": exc.moved_setpoints}), flush=True)
        raise
    finally:
        print("HELPER", json.dumps(helper), flush=True)
        with open(os.path.join(workdir, "values.json"), "w") as handle:
            json.dump(values, handle)
    """
)


class _SigintChild:
    """A stub child running ``run_tool`` over a dict machine; the parent sends SIGINT.

    The child runs in a deployment repo of its own, so its run has a target
    directory to lock.
    """

    def __init__(self, workdir: Path, mode: str) -> None:
        self.workdir = workdir
        script = workdir / "sigint_child.py"
        script.write_text(_SIGINT_CHILD, encoding="utf-8")
        repo = workdir / "repo"
        repo.mkdir(exist_ok=True)
        (repo / "profile.yml").write_text("name: probe\n", encoding="utf-8")
        env = {**os.environ, osprey.runtime.ENV_CONTROL_TARGET: "live"}
        for name in ("OSPREY_EXECUTION_MODE", osprey.runtime.ENV_CONTROL_TARGET_GENERATION):
            env.pop(name, None)
        self.proc = subprocess.Popen(
            [sys.executable, str(script), mode, str(workdir), str(CHILD_TIMEOUT_S / 2)],
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


class TestSigintAbort:
    """A Ctrl-C stops every tool of the run and ends the script; nothing is put back."""

    def test_sigint_during_span_aborts(self, sigint_child: Callable[[str], _SigintChild]) -> None:
        """The chained callback stops the tool; re-raised with what it moved."""
        child = sigint_child("span")
        child.interrupt_at("span")
        code, out, err, values = child.finish()
        assert code != 0
        assert _line(out, "INTERRUPTED") == {"moved": {"A": 1.0}}
        assert values == {**HOME, "A": 5.0}
        assert "GUARDED_RUN_RESTORE" not in out + err

    def test_sigint_without_callback_raises_with_what_it_moved(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """A tool that takes no callback gets ``KeyboardInterrupt`` at once."""
        child = sigint_child("plain")
        child.interrupt_at("span")
        code, out, _err, values = child.finish()
        assert code != 0
        assert _line(out, "INTERRUPTED") == {"moved": {"A": 1.0}}
        assert values == {**HOME, "A": 5.0}

    def test_sigint_in_nested_run_aborts_outer(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """A SIGINT inside a nested run stops every enclosing run; nothing later writes."""
        child = sigint_child("nested")
        child.interrupt_at("span")
        code, out, _err, values = child.finish()
        assert code != 0
        assert _line(out, "INTERRUPTED") == {"moved": {"A": 1.0, "B": 2.0}}
        assert values == {"A": 5.0, "B": 6.0, "C": 3.0}

    def test_sigint_stops_a_helper_threads_tool_through_its_callback(
        self, sigint_child: Callable[[str], _SigintChild]
    ) -> None:
        """The run's abort flag reaches a tool on a helper thread; it never steps to its bound."""
        child = sigint_child("helper")
        child.interrupt_at("span")
        code, out, _err, values = child.finish()
        assert code != 0
        assert _line(out, "HELPER") == {"falsy": True, "caught": True}
        assert _line(out, "INTERRUPTED") == {"moved": {"A": 1.0, "B": 2.0}}
        assert values == {"A": 5.0, "B": 6.0, "C": 3.0}, "C was never written"

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

    def test_sigint_abort_stops_script(self, sigint_child: Callable[[str], _SigintChild]) -> None:
        """A tool that reports the stop with ``False`` still ends the script."""
        child = sigint_child("after")
        child.interrupt_at("span")
        code, out, _err, values = child.finish()
        assert code != 0
        assert "AFTER" not in out
        assert "RETURNED" not in out
        assert _line(out, "INTERRUPTED") == {"moved": {"A": 1.0}}
        assert values == {**HOME, "A": 5.0}
