"""Stepped measurements pyAML's shipped tools cannot express, run as one method call.

pyAML 0.3.1's tools step bipolar, one magnet at a time and with one scalar delta.
:class:`StepMeasurement` runs the other steppings: a unipolar orbit response with a
delta per corrector, a unipolar dispersion, and tune or chromaticity responses that
step a whole group of magnets as one knob.

Every write happens INSIDE :meth:`StepMeasurement.measure`: pySC's measurement
generators are drained within the call, so a run given to
:func:`pyaml_cs_osprey.run_tool.run_tool` writes nothing after ``run_tool`` returns,
and every write is journaled, locked and guarded by the deadline like any pyAML
tool's. ``measure`` reports progress the way pyAML's own tools do: ``Action.APPLY``
after each setpoint change, ``Action.MEASURE`` after each reading and
``Action.RESTORE`` after a stepped knob is put back. A callback that returns a falsy
value stops the run and ``measure`` returns ``False``; what it had moved is left for
the guarded run's journal to write back.

pySC is always called with ``skip_save=True``: its default writes result files
under the working directory's ``data/``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pySC
from pyaml.common.constants import Action
from pyaml.external.pySC_interface import pySCInterface
from pyaml.tuning_tools.measurement_tool import MeasurementTool
from pySC.apps import measure_dispersion, measure_ORM

if TYPE_CHECKING:
    from pyaml.common.holders.element_holder import ElementHolder

__all__ = ["KINDS", "RESPONSE_MATRIX_DATA", "StepMeasurement"]

#: The measurement kinds a :class:`StepMeasurement` runs.
KINDS = ("orm", "dispersion", "trm", "crm")

#: The measurement type recorded in ``latest_measurement["type"]``, the one pyAML's
#: response-matrix tools record.
RESPONSE_MATRIX_DATA = "pyaml.tuning_tools.response_matrix_data"

#: The callback action each pySC generator code is reported as, by code name (the
#: response and dispersion generators share the names); a code absent here
#: (initialised, done) is not reported.
_ACTIONS: dict[str, Action] = {
    "AFTER_SET": Action.APPLY,
    "AFTER_GET": Action.MEASURE,
    "AFTER_RESTORE": Action.RESTORE,
}

#: Pauses between a setpoint change and the next reading; tests replace it.
_sleep: Callable[[float], None] = time.sleep


class StepMeasurement(MeasurementTool):
    """One stepped measurement on a pyAML element holder (``sr.live``, ``sr.design``).

    Build one with :meth:`orm`, :meth:`dispersion`, :meth:`trm` or :meth:`crm`; call
    :meth:`measure` (through ``run_tool``) to run it. After a completed run
    ``latest_measurement`` holds ``type``, ``matrix`` (observables x variables, the
    response per unit step), ``variable_names`` and ``observable_names``.

    Attributes:
        kind: One of :data:`KINDS`.
    """

    def __init__(
        self,
        holder: ElementHolder,
        kind: str,
        *,
        bpm_array: str | None = None,
        correctors: Sequence[str] = (),
        deltas: Sequence[float] = (),
        rf_plant: str | None = None,
        groups: Sequence[str] = (),
        delta: float = 0.0,
        tune_monitor: str | None = None,
        chromaticity_monitor: str | None = None,
        set_wait_time: float = 0.0,
        n_avg_meas: int = 1,
        sleep_between_meas: float = 0.0,
    ) -> None:
        """Validate one measurement's arguments; prefer the per-kind constructors.

        Raises:
            ValueError: An argument the kind needs is missing or inconsistent.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown measurement kind {kind!r}; known: {', '.join(KINDS)}")
        super().__init__(f"STEP_{kind.upper()}")
        if kind in ("orm", "dispersion") and not bpm_array:
            raise ValueError(f"{kind} needs a BPM array")
        if kind == "orm":
            if not correctors:
                raise ValueError("orm needs at least one corrector")
            if len(deltas) != len(correctors):
                raise ValueError(
                    f"orm needs one delta per corrector: {len(correctors)} correctors, "
                    f"{len(deltas)} deltas"
                )
        if kind == "dispersion" and not rf_plant:
            raise ValueError("dispersion needs an RF plant")
        if kind in ("trm", "crm") and not groups:
            raise ValueError(f"{kind} needs at least one group")
        if kind == "trm" and not tune_monitor:
            raise ValueError("trm needs a tune monitor")
        if kind == "crm" and not chromaticity_monitor:
            raise ValueError("crm needs a chromaticity monitor")
        if kind in ("dispersion", "trm", "crm") and delta == 0:
            raise ValueError(f"{kind} needs a nonzero step")
        if any(d == 0 for d in deltas):
            raise ValueError("orm needs a nonzero delta for every corrector")
        if n_avg_meas < 1:
            raise ValueError("n_avg_meas must be at least 1")
        self.kind = kind
        self._holder = holder
        self._bpm_array = bpm_array
        self._correctors = list(correctors)
        self._deltas = [float(d) for d in deltas]
        self._rf_plant = rf_plant
        self._groups = list(groups)
        self._delta = float(delta)
        self._tune_monitor = tune_monitor
        self._chromaticity_monitor = chromaticity_monitor
        self.set_wait_time = float(set_wait_time)
        self.n_avg_meas = int(n_avg_meas)
        self.sleep_between_meas = float(sleep_between_meas)

    @classmethod
    def orm(
        cls,
        holder: ElementHolder,
        *,
        bpm_array: str,
        correctors: Sequence[str],
        deltas: Sequence[float],
        set_wait_time: float = 0.0,
        n_avg_meas: int = 1,
    ) -> StepMeasurement:
        """A unipolar orbit response: each corrector stepped by its own delta.

        Args:
            holder: The element holder the measurement runs on.
            bpm_array: The pyAML BPM array the orbit is read from.
            correctors: pyAML magnet names, in column order.
            deltas: The strength step of each corrector, paired with ``correctors``.
            set_wait_time: Seconds to wait after each setpoint change.
            n_avg_meas: Orbit readings averaged at each setting.
        """
        return cls(
            holder,
            "orm",
            bpm_array=bpm_array,
            correctors=correctors,
            deltas=deltas,
            set_wait_time=set_wait_time,
            n_avg_meas=n_avg_meas,
        )

    @classmethod
    def dispersion(
        cls,
        holder: ElementHolder,
        *,
        bpm_array: str,
        rf_plant: str,
        delta: float,
        set_wait_time: float = 0.0,
        n_avg_meas: int = 1,
    ) -> StepMeasurement:
        """A unipolar dispersion: the RF frequency stepped once by ``delta`` (Hz).

        Args:
            holder: The element holder the measurement runs on.
            bpm_array: The pyAML BPM array the orbit is read from.
            rf_plant: The pyAML RF plant whose frequency is stepped.
            delta: The frequency step in Hz.
            set_wait_time: Seconds to wait after each setpoint change.
            n_avg_meas: Orbit readings averaged at each setting.
        """
        return cls(
            holder,
            "dispersion",
            bpm_array=bpm_array,
            rf_plant=rf_plant,
            delta=delta,
            set_wait_time=set_wait_time,
            n_avg_meas=n_avg_meas,
        )

    @classmethod
    def trm(
        cls,
        holder: ElementHolder,
        *,
        groups: Sequence[str],
        delta: float,
        tune_monitor: str,
        set_wait_time: float = 0.0,
        n_avg_meas: int = 1,
        sleep_between_meas: float = 0.0,
    ) -> StepMeasurement:
        """A tune response with every member of a group stepped together.

        Args:
            holder: The element holder the measurement runs on.
            groups: pyAML magnet arrays, one knob each, in column order.
            delta: The strength step every member of a group takes.
            tune_monitor: The pyAML betatron tune monitor read.
            set_wait_time: Seconds to wait after each setpoint change.
            n_avg_meas: Tune readings averaged at each setting.
            sleep_between_meas: Seconds between two averaged readings.
        """
        return cls(
            holder,
            "trm",
            groups=groups,
            delta=delta,
            tune_monitor=tune_monitor,
            set_wait_time=set_wait_time,
            n_avg_meas=n_avg_meas,
            sleep_between_meas=sleep_between_meas,
        )

    @classmethod
    def crm(
        cls,
        holder: ElementHolder,
        *,
        groups: Sequence[str],
        delta: float,
        chromaticity_monitor: str,
        set_wait_time: float = 0.0,
    ) -> StepMeasurement:
        """A chromaticity response with every member of a group stepped together.

        The chromaticity at each setting is the chromaticity monitor's own RF
        sweep, run with its configured steps and averaging.

        Args:
            holder: The element holder the measurement runs on.
            groups: pyAML magnet arrays, one knob each, in column order.
            delta: The strength step every member of a group takes.
            chromaticity_monitor: The pyAML chromaticity monitor run at each setting.
            set_wait_time: Seconds to wait after each setpoint change.
        """
        return cls(
            holder,
            "crm",
            groups=groups,
            delta=delta,
            chromaticity_monitor=chromaticity_monitor,
            set_wait_time=set_wait_time,
        )

    @property
    def chromaticity_monitor(self) -> MeasurementTool | None:
        """The chromaticity monitor a ``crm`` hands its callback to, else ``None``."""
        if self._chromaticity_monitor is None:
            return None
        return self._holder.get_chromaticity_monitor(self._chromaticity_monitor)

    def measure(self, callback: Callable[..., Any] | None = None) -> bool:
        """Run the measurement, every write inside this call.

        Args:
            callback: Called as ``callback(action, data)`` after each setpoint
                change (``Action.APPLY``), each reading (``Action.MEASURE``) and
                each knob put back (``Action.RESTORE``); a falsy result stops the
                run.

        Returns:
            ``True`` when the run completed and ``latest_measurement`` holds its
            result, ``False`` when the callback stopped it.
        """
        self._register_callback(callback)
        try:
            self._init_measure(RESPONSE_MATRIX_DATA)
            if self.kind == "orm":
                result = self._measure_orm()
            elif self.kind == "dispersion":
                result = self._measure_dispersion()
            else:
                result = self._measure_groups()
        except KeyboardInterrupt:
            return False
        self.latest_measurement.update(result)
        return True

    def _interface(self, rf_plant: str | None = None) -> pySCInterface:
        """A pySC interface on the holder, waiting ``set_wait_time`` after each set."""
        interface = pySCInterface(
            element_holder=self._holder,
            bpm_array_name=self._bpm_array,
            rf_plant_name=rf_plant,
        )
        interface.set_wait_time = self.set_wait_time
        return interface

    def _drain(self, generator: Any) -> Any:
        """Run a pySC generator to its end, reporting each code; the last measurement."""
        measurement = None
        try:
            for idx, (code, current) in enumerate(generator):
                measurement = current
                action = _ACTIONS.get(code.name)
                if action is not None:
                    self.send_callback(action, {"idx": idx, "code": code.name})
        finally:
            generator.close()
        return measurement

    def _orbit_observables(self) -> list[str]:
        """The orbit observables in pySC's order: every BPM's x, then every BPM's y."""
        names = list(self._holder.bpms.get(self._bpm_array).names())
        return [f"{name}.x" for name in names] + [f"{name}.y" for name in names]

    def _measure_orm(self) -> dict[str, Any]:
        """Each corrector stepped once by its delta, the orbit read before and after."""
        pySC.disable_pySC_rich()
        measurement = self._drain(
            measure_ORM(
                self._interface(),
                corrector_names=list(self._correctors),
                delta=list(self._deltas),
                shots_per_orbit=self.n_avg_meas,
                bipolar=False,
                skip_save=True,
            )
        )
        return {
            "matrix": np.asarray(measurement.response_data.matrix).tolist(),
            "variable_names": list(self._correctors),
            "observable_names": self._orbit_observables(),
        }

    def _measure_dispersion(self) -> dict[str, Any]:
        """The RF frequency stepped once by the delta, the orbit read before and after."""
        pySC.disable_pySC_rich()
        measurement = self._drain(
            measure_dispersion(
                self._interface(rf_plant=self._rf_plant),
                delta=self._delta,
                shots_per_orbit=self.n_avg_meas,
                bipolar=False,
                skip_save=True,
            )
        )
        response_x, response_y = measurement.dispersion_data.frequency_response
        column = np.concatenate((np.ravel(response_x), np.ravel(response_y)))
        return {
            "matrix": [[float(value)] for value in column],
            "variable_names": [self._rf_plant],
            "observable_names": self._orbit_observables(),
        }

    def _read(self, group: str, idx: int) -> np.ndarray:
        """The tune or chromaticity at the current setting, as ``[x, y]``."""
        if self.kind == "trm":
            monitor = self._holder.get_betatron_tune_monitor(self._tune_monitor)
            total = np.zeros(2)
            for avg in range(self.n_avg_meas):
                tune = np.asarray(monitor.tune.get(), dtype=float)
                total += tune
                self.send_callback(
                    Action.MEASURE, {"idx": idx, "group": group, "avg_step": avg, "tune": tune}
                )
                if avg < self.n_avg_meas - 1:
                    _sleep(self.sleep_between_meas)
            return total / self.n_avg_meas
        monitor = self._holder.get_chromaticity_monitor(self._chromaticity_monitor)
        if not monitor.measure(callback=self._callback):
            raise KeyboardInterrupt
        chromaticity = np.asarray(monitor.chromaticity.get(), dtype=float)
        self.send_callback(
            Action.MEASURE, {"idx": idx, "group": group, "chromaticity": chromaticity}
        )
        return chromaticity

    def _measure_groups(self) -> dict[str, Any]:
        """Each group stepped once by the delta as one knob, read before and after."""
        observables = np.zeros((2, len(self._groups)))
        for idx, group in enumerate(self._groups):
            strengths = self._holder.magnets.get(group).strengths
            start = np.asarray(strengths.get(), dtype=float)
            before = self._read(group, idx)
            strengths.set(start + self._delta)
            self.send_callback(Action.APPLY, {"idx": idx, "group": group})
            _sleep(self.set_wait_time)
            after = self._read(group, idx)
            strengths.set(start)
            observables[:, idx] = (after - before) / self._delta
            self.send_callback(Action.RESTORE, {"idx": idx, "group": group})
            _sleep(self.set_wait_time)
        if self.kind == "trm":
            name = self._tune_monitor
        else:
            name = self._chromaticity_monitor
        return {
            "matrix": observables.tolist(),
            "variable_names": list(self._groups),
            "observable_names": [f"{name}.x", f"{name}.y"],
        }
