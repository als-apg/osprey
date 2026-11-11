"""The injection chain: a shot-based state machine the composite holds as a child.

:class:`InjectionModel` is a LUME model with no thread and no clock of its
own. Time reaches it only through :meth:`InjectionModel.advance` and a client's
request only through :meth:`InjectionModel.client_write`; ``set`` stores values
and starts nothing, since restore, rollback, journal replay and scenario writes
all reach it. The same calls with the same clock readings always yield the same
values.

A shot starts when a client writes the request channel while no shot is in
flight; a request during a shot is stored and ignored. The shot latches the
source charge and every stage efficiency at its start and applies its
:class:`Step` values at their offsets, in time order and, at equal offsets, in
declared order. Charge after a stage is the source charge times the product of
the stage efficiencies up to it; the current step raises the stored current by
the final charge times the current per charge, with no saturation and no decay.
The shot ends at its last step.

The cycle signal runs free of shots: it rises at the first ``advance`` after
construction or :meth:`InjectionModel.reset` and every period after that, and
falls a width after each rise.

Every address, offset, period and coefficient is the caller's data; this
module names none.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lume.model import LUMEModel
from lume.variables import ScalarVariable, Variable

__all__ = [
    "CHARGE",
    "COUNTER",
    "CURRENT",
    "EFFICIENCY",
    "STEP_KINDS",
    "InjectionModel",
    "Step",
]

#: A step that adds one to its channel.
COUNTER = "counter"
#: A step that shows the charge after its stage.
CHARGE = "charge"
#: A step that shows its stage's latched efficiency.
EFFICIENCY = "efficiency"
#: The step that raises the stored current; its channel refreshes by pass.
CURRENT = "current"
#: Every step kind.
STEP_KINDS = (COUNTER, CHARGE, EFFICIENCY, CURRENT)

_STAGED = (CHARGE, EFFICIENCY)
_HIGH = 1.0
_LOW = 0.0


@dataclass(frozen=True)
class Step:
    """One step of a shot.

    Attributes:
        address: The channel the step changes.
        kind: One of :data:`STEP_KINDS`.
        offset_s: Seconds after the shot starts.
        stage: For :data:`CHARGE` and :data:`EFFICIENCY`, the index of the
            stage into the model's ``efficiencies``; ``None`` otherwise.
    """

    address: str
    kind: str
    offset_s: float
    stage: int | None = None


@dataclass
class _Shot:
    start_s: float
    charge: float
    efficiencies: tuple[float, ...]
    next_step: int = 0


class InjectionModel(LUMEModel):
    """A shot-based injection chain driven by ``advance`` and ``client_write``.

    Args:
        request: The request setpoint, or ``None`` for a chain no client starts.
        source_charge: The setpoint holding the charge per shot at the source.
        efficiencies: One setpoint per stage, in stage order, each holding that
            stage's transfer efficiency.
        stored_current: The setpoint holding the stored current.
        steps: The steps of one shot.
        currents: The readings that show the stored current.
        scales: By reading of ``currents``, the factor from the stored
            current to that reading; a reading absent here shows it as is.
        busy: The readings of the cycle signal.
        period_s: The cycle's period.
        width_s: How long the cycle stays high after each rise.
        current_per_charge: The stored current one unit of final charge adds.
        start: Each channel's start value; a channel absent here starts at zero.

    Raises:
        ValueError: A step of unknown kind, a negative offset, a stage step
            whose stage is not an index into ``efficiencies``, a cycle whose
            width is not inside its period, or a scale on a channel that is
            no current reading.
    """

    def __init__(
        self,
        *,
        request: str | None,
        source_charge: str,
        efficiencies: Sequence[str],
        stored_current: str,
        steps: Iterable[Step],
        currents: Iterable[str] = (),
        scales: Mapping[str, float] | None = None,
        busy: Iterable[str] = (),
        period_s: float,
        width_s: float,
        current_per_charge: float,
        start: Mapping[str, float] | None = None,
    ) -> None:
        if not period_s > 0:
            raise ValueError(f"the cycle period must be positive, not {period_s}")
        if not 0 < width_s < period_s:
            raise ValueError(
                f"the cycle width must lie inside its period {period_s}, not {width_s}"
            )
        self._efficiencies = tuple(efficiencies)
        declared = list(steps)
        for step in declared:
            _check_step(step, len(self._efficiencies))
        # Time order; ``sorted`` is stable, so equal offsets keep declared order.
        self._steps = tuple(sorted(declared, key=lambda step: step.offset_s))
        self._request = request
        self._source = source_charge
        self._stored = stored_current
        self._currents = tuple(currents)
        self._scales = {name: float(scale) for name, scale in (scales or {}).items()}
        stray = sorted(set(self._scales) - set(self._currents))
        if stray:
            raise ValueError(f"a scale names no current reading: {', '.join(stray)}")
        self._busy = tuple(busy)
        self._period = float(period_s)
        self._width = float(width_s)
        self._per_charge = float(current_per_charge)

        setpoints = [
            *([request] if request is not None else []),
            source_charge,
            *self._efficiencies,
            stored_current,
        ]
        readings = [*(step.address for step in self._steps), *self._currents, *self._busy]
        self._variables: dict[str, Variable] = {}
        for name in setpoints:
            self._variables[name] = ScalarVariable(name=name, read_only=False)
        for name in readings:
            self._variables.setdefault(name, ScalarVariable(name=name, read_only=True))
        start = start or {}
        self._start = {name: float(start.get(name, _LOW)) for name in self._variables}

        self._values: dict[str, float] = {}
        self._anchor: float | None = None
        self._now: float | None = None
        self._edge = 0
        self._shot: _Shot | None = None
        self._emitted: dict[str, None] = {}
        self.reset()

    # -- the LUME model ----------------------------------------------------------

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """Every setpoint and reading of the chain."""
        return self._variables

    def reset(self) -> None:
        """Abandon any shot, return to the start values and drop the time anchor."""
        self._values = dict(self._start)
        self._anchor = None
        self._now = None
        self._edge = 0
        self._shot = None
        self._emitted = {}

    def _set(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            self._values[name] = float(value)

    def _get(self, names: list[str]) -> dict[str, Any]:
        stored = self._values[self._stored]
        return {
            name: stored * self._scales.get(name, 1.0)
            if name in self._currents
            else self._values[name]
            for name in names
        }

    # -- the time contract -------------------------------------------------------

    @property
    def in_shot(self) -> bool:
        """Whether a shot is in flight."""
        return self._shot is not None

    def advance(self, t_s: float) -> None:
        """Apply every cycle edge and step due by epoch second ``t_s``.

        The first call after construction or a reset anchors the model's time
        and raises the cycle. A reading earlier than the last one changes
        nothing.
        """
        if self._anchor is None:
            self._anchor = t_s
        if self._now is not None and t_s < self._now:
            return
        self._now = t_s
        self._apply_due()

    def _apply_due(self) -> None:
        while True:
            edge, step = self._edge_due(), self._step_due()
            if edge is None and step is None:
                return
            if step is None or (edge is not None and edge <= step):
                self._apply_edge()
            else:
                self._apply_step()

    def client_write(self, values: Mapping[str, Any], t_s: float) -> None:
        """Start a shot at ``t_s`` when ``values`` holds the request and none is in flight.

        The values are already stored by ``set``; this call only decides what
        the client's write starts. Steps due at once apply before it returns.
        """
        self.advance(t_s)
        if self._request is None or self._request not in values or self._shot is not None:
            return
        self._shot = _Shot(
            start_s=t_s,
            charge=self._values[self._source],
            efficiencies=tuple(self._values[name] for name in self._efficiencies),
        )
        self._end_shot_if_done()
        self._apply_due()

    def next_deadline(self) -> float | None:
        """The epoch second of the next cycle edge or step, or ``None`` before time is anchored."""
        if self._anchor is None:
            return None
        edge = self._edge_time(self._edge)
        step = self._step_time()
        return edge if step is None else min(edge, step)

    def emitted(self) -> list[str]:
        """The event channels changed since the last call, in first-change order, cleared."""
        names = list(self._emitted)
        self._emitted = {}
        return names

    # -- internals ---------------------------------------------------------------

    def _edge_time(self, edge: int) -> float:
        assert self._anchor is not None
        cycle, falling = divmod(edge, 2)
        return self._anchor + cycle * self._period + falling * self._width

    def _edge_due(self) -> float | None:
        if self._anchor is None or self._now is None:
            return None
        time = self._edge_time(self._edge)
        return time if time <= self._now else None

    def _step_time(self) -> float | None:
        if self._shot is None:
            return None
        return self._shot.start_s + self._steps[self._shot.next_step].offset_s

    def _step_due(self) -> float | None:
        time = self._step_time()
        return time if time is not None and self._now is not None and time <= self._now else None

    def _apply_edge(self) -> None:
        level = _HIGH if self._edge % 2 == 0 else _LOW
        self._edge += 1
        for name in self._busy:
            self._values[name] = level
            self._emitted[name] = None

    def _apply_step(self) -> None:
        shot = self._shot
        assert shot is not None
        step = self._steps[shot.next_step]
        shot.next_step += 1
        if step.kind == COUNTER:
            self._values[step.address] += 1
        elif step.kind == EFFICIENCY:
            assert step.stage is not None
            self._values[step.address] = shot.efficiencies[step.stage]
        elif step.kind == CHARGE:
            assert step.stage is not None
            self._values[step.address] = _charge_after(shot, step.stage)
        else:
            final = _charge_after(shot, len(shot.efficiencies) - 1)
            self._values[self._stored] += final * self._per_charge
        if step.kind != CURRENT:
            self._emitted[step.address] = None
        self._end_shot_if_done()

    def _end_shot_if_done(self) -> None:
        if self._shot is not None and self._shot.next_step >= len(self._steps):
            self._shot = None


def _charge_after(shot: _Shot, stage: int) -> float:
    charge = shot.charge
    for efficiency in shot.efficiencies[: stage + 1]:
        charge *= efficiency
    return charge


def _check_step(step: Step, stages: int) -> None:
    if step.kind not in STEP_KINDS:
        raise ValueError(f"{step.address}: unknown step kind {step.kind!r}")
    if step.offset_s < 0:
        raise ValueError(f"{step.address}: a step offset cannot be negative ({step.offset_s})")
    if step.kind in _STAGED and not (step.stage is not None and 0 <= step.stage < stages):
        raise ValueError(
            f"{step.address}: a {step.kind} step needs a stage index below {stages},"
            f" not {step.stage}"
        )
