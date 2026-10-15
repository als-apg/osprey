"""``ConnectorEnvironment``: an environment file, driven over OSPREY connectors.

The core's own environment. It knows nothing facility-specific: variables are
setpoint/readback channel pairs, observables are a channel or a reduction over
channels, and every read and write goes through :class:`ConnectorInterface`.
Facility plugins that need more (shot gating, composite physics objectives,
machine checks) implement ``EnvironmentProtocol`` themselves.

The control target is a property of the *call*, not of the environment: the
config dict carries ``control_target`` (``va``, ``standin`` or ``live``) and
optionally ``execution_mode: readonly``. Metadata (variables, observables,
objectives) needs neither, so listing and bounds resolution work without a
target; the first channel access without one is refused.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from osprey_optimizer.models.enums import EnvironmentType
from osprey_optimizer.models.exceptions import (
    EvaluationError,
    InterfaceError,
    ObservableNotFoundError,
    VariableOutOfBoundsError,
)
from osprey_optimizer.models.protocols import (
    InterfaceProtocol,
    ObservableMetadata,
    PVObservable,
    PVReading,
    ReducedObservable,
    VariableMetadata,
)

from .spec import EnvironmentFile

logger = logging.getLogger(__name__)

_READBACK_POLL_S = 0.1
_SETTLE_POLL_S = 0.2

LimitsLookup = Callable[[str | None], Mapping[str, Any] | None]
"""Per-channel limits for a target (``None``: the most restrictive across targets),
or ``None`` when limits checking is off. Entries carry ``min_value``,
``max_value`` and ``writable``, as ``ChannelLimitsConfig`` does."""


def deployment_limits(target: str | None) -> Mapping[str, Any] | None:
    """The deployment's channel limits for ``target``, from its config."""
    from osprey_connectors.control_system.limits_validator import LimitsValidator

    validator = (
        LimitsValidator.from_config(target=target)
        if target is not None
        else LimitsValidator.from_config_most_restrictive()
    )
    return None if validator is None else validator.limits


def within_channel_limits(
    variables: dict[str, VariableMetadata], limits: Mapping[str, Any] | None
) -> dict[str, VariableMetadata]:
    """Each variable's range narrowed to its setpoint channel's limits.

    The channel's ``max_step``, if any, is carried on the metadata so that
    resolving a run can refuse a search window the step limit would cut.

    The environment file says what the optimizer may explore; the limits
    database says what the channel may ever be set to. A run must satisfy both,
    so the range it sees is their intersection, and a request outside it fails
    when the job is created rather than at its first write. A channel the
    database marks read-only, or a range that does not meet the limits at all,
    is a misconfigured environment and is refused outright. Unlisted channels
    are left to the connector, whose policy decides them on every put.
    """
    if not limits:
        return variables
    out: dict[str, VariableMetadata] = {}
    for name, meta in variables.items():
        entry = limits.get(meta.setpoint_pv)
        if entry is None:
            out[name] = meta
            continue
        if not getattr(entry, "writable", True):
            raise ValueError(
                f"Variable {name!r}: its setpoint {meta.setpoint_pv} is read-only in the "
                "channel limits database"
            )
        lo = getattr(entry, "min_value", None)
        hi = getattr(entry, "max_value", None)
        max_step = getattr(entry, "max_step", None)
        new_min = meta.min_value if lo is None else max(meta.min_value, lo)
        new_max = meta.max_value if hi is None else min(meta.max_value, hi)
        if new_min > new_max:
            raise ValueError(
                f"Variable {name!r}: the environment's range [{meta.min_value}, {meta.max_value}] "
                f"does not meet the channel limits [{lo}, {hi}] of {meta.setpoint_pv}"
            )
        if (new_min, new_max) != (meta.min_value, meta.max_value):
            logger.info(
                "%s narrowed to the channel limits: [%s, %s] -> [%s, %s]",
                name,
                meta.min_value,
                meta.max_value,
                new_min,
                new_max,
            )
        out[name] = meta.model_copy(
            update={"min_value": new_min, "max_value": new_max, "max_step": max_step}
        )
    return out


def reduce_values(kind: str, values: list[float]) -> float:
    """The reduction a ``ReducedObservable`` names, over plain floats."""
    if kind == "rms":
        return math.sqrt(sum(v * v for v in values) / len(values))
    if kind == "mean":
        return sum(values) / len(values)
    if kind == "max":
        return max(values)
    raise ValueError(f"Unknown reduction {kind!r}")


def _numeric(reading: PVReading | None) -> float | None:
    if reading is None or not reading.connected:
        return None
    v = reading.value
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        return None
    return float(v)


class ConnectorEnvironment:
    """``EnvironmentProtocol`` over one environment file."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        interface_factory: Callable[[str, str | None], InterfaceProtocol] | None = None,
        limits_lookup: LimitsLookup = deployment_limits,
    ) -> None:
        spec = config.get("spec")
        if not isinstance(spec, EnvironmentFile):
            raise TypeError("ConnectorEnvironment needs its EnvironmentFile under config['spec']")
        self._spec = spec
        self._target: str | None = config.get("control_target")
        self._mode: str | None = config.get("execution_mode")
        self._all_variables = within_channel_limits(
            spec.variable_metadata(), limits_lookup(self._target)
        )
        self._observables = spec.observable_metadata()

        active = config.get("variables")
        if active is not None:
            unknown = [n for n in active if n not in self._all_variables]
            if unknown:
                raise ValueError(f"Unknown variable(s): {unknown}")
        self._active = tuple(active) if active is not None else None

        objective = config.get("objective_name") or spec.default_objective
        if objective is not None and objective not in self._observables:
            raise ValueError(f"Unknown default objective {objective!r}")
        self._default_objective = objective or self.available_objectives[0]

        self._interface_factory = interface_factory
        self._interface: InterfaceProtocol | None = None

    # -- metadata ---------------------------------------------------------

    @property
    def environment_type(self) -> EnvironmentType:
        return EnvironmentType.SIMULATION if self._target == "va" else EnvironmentType.PRODUCTION

    @property
    def variables(self) -> dict[str, VariableMetadata]:
        if self._active is None:
            return self._all_variables
        return {n: self._all_variables[n] for n in self._active}

    @property
    def observables(self) -> dict[str, ObservableMetadata]:
        return self._observables

    @property
    def available_objectives(self) -> list[str]:
        return [n for n, o in self._observables.items() if o.selectable_as_objective]

    @property
    def default_objective(self) -> str:
        return self._default_objective

    # -- channel access ---------------------------------------------------

    @property
    def interface(self) -> InterfaceProtocol:
        if self._interface is None:
            if self._target is None:
                raise InterfaceError(
                    "This call reaches the machine, so it must name a control_target "
                    "(va, standin or live)."
                )
            if self._interface_factory is not None:
                self._interface = self._interface_factory(self._target, self._mode)
            else:
                from .interface import ConnectorInterface

                self._interface = ConnectorInterface(self._target, self._mode)
        return self._interface

    def get_variables(self, variable_names: list[str]) -> dict[str, PVReading]:
        return self._read_variable_channels(variable_names, setpoints=False)

    def get_setpoints(self, variable_names: list[str]) -> dict[str, PVReading]:
        return self._read_variable_channels(variable_names, setpoints=True)

    def _read_variable_channels(
        self, variable_names: list[str], *, setpoints: bool
    ) -> dict[str, PVReading]:
        unknown = [n for n in variable_names if n not in self._all_variables]
        if unknown:
            raise VariableOutOfBoundsError(f"Unknown variable(s): {unknown}")
        metas = {n: self._all_variables[n] for n in variable_names}
        channels = {n: m.setpoint_pv if setpoints else m.readback_pv for n, m in metas.items()}
        readings = self.interface.read_many(list(dict.fromkeys(channels.values())))
        missing = PVReading(value=None, connected=False)
        return {name: readings.get(channel, missing) for name, channel in channels.items()}

    def set_variables(
        self,
        variable_inputs: dict[str, float],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> None:
        """Check bounds, write, wait for the readbacks to agree, then settle.

        The environment's bounds are checked here; the channel limits (and
        ``max_step``) are checked again by the connector on every put.
        """
        for name, value in variable_inputs.items():
            meta = self._all_variables.get(name)
            if meta is None:
                raise VariableOutOfBoundsError(f"Unknown variable: {name}")
            if not math.isfinite(value):
                raise VariableOutOfBoundsError(f"Variable {name} value {value} is not finite")
            if not meta.min_value <= value <= meta.max_value:
                raise VariableOutOfBoundsError(
                    f"Variable {name} value {value} is outside [{meta.min_value}, {meta.max_value}]"
                )
        writes = {self._all_variables[n].setpoint_pv: v for n, v in variable_inputs.items()}
        self.interface.set_values(writes)
        self._verify_readbacks(variable_inputs, check_cancellation)
        self._settle(check_cancellation)

    def _verify_readbacks(
        self, variable_inputs: dict[str, float], check_cancellation: Callable[[], None] | None
    ) -> None:
        pending = {
            self._all_variables[n].readback_pv: (n, v, self._spec.readback_tolerance(n))
            for n, v in variable_inputs.items()
        }
        deadline = time.monotonic() + self._spec.readback_timeout_seconds
        last: dict[str, float | None] = {}
        while pending:
            readings = self.interface.read_many(list(pending))
            for rb in list(pending):
                name, expected, tol = pending[rb]
                got = _numeric(readings.get(rb))
                last[name] = got
                if got is not None and abs(got - expected) <= tol:
                    del pending[rb]
            if not pending:
                return
            if time.monotonic() >= deadline:
                failed = ", ".join(
                    f"{n} (readback={last.get(n)}, expected={v})" for n, v, _ in pending.values()
                )
                raise EvaluationError(f"Readback verification timed out for: {failed}")
            if check_cancellation is not None:
                check_cancellation()
            time.sleep(_READBACK_POLL_S)

    def _settle(self, check_cancellation: Callable[[], None] | None) -> None:
        end = time.monotonic() + self._spec.settle_seconds
        while (remaining := end - time.monotonic()) > 0:
            if check_cancellation is not None:
                check_cancellation()
            time.sleep(min(_SETTLE_POLL_S, remaining))

    def get_observables(
        self,
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        """Read every channel the observables need in one batch, then reduce."""
        for name in observable_names:
            if name not in self._observables:
                raise ObservableNotFoundError(f"Unknown observable: {name}")
        if check_cancellation is not None:
            check_cancellation()
        channels: list[str] = []
        for name in observable_names:
            meta = self._observables[name]
            if isinstance(meta, ReducedObservable):
                channels.extend(meta.channels)
            elif isinstance(meta, PVObservable):
                channels.append(meta.readback_pv)
            else:
                raise ObservableNotFoundError(
                    f"Observable {name!r} has no channel this environment can read"
                )
        readings = self.interface.read_many(list(dict.fromkeys(channels)))

        out: dict[str, PVReading] = {}
        for name in observable_names:
            meta = self._observables[name]
            if isinstance(meta, PVObservable):
                out[name] = readings.get(meta.readback_pv, PVReading(value=None, connected=False))
                continue
            assert isinstance(meta, ReducedObservable)
            values = [_numeric(readings.get(c)) for c in meta.channels]
            if any(v is None for v in values):
                bad = [c for c, v in zip(meta.channels, values, strict=True) if v is None]
                logger.warning("%s: %d channel(s) unreadable, e.g. %s", name, len(bad), bad[:3])
                out[name] = PVReading(value=None, connected=False, units=meta.units or None)
            else:
                value = reduce_values(meta.reduce, [v for v in values if v is not None])
                out[name] = PVReading(
                    value=value, connected=True, units=meta.units or None, count=1
                )
        return out

    def evaluate(
        self,
        variable_inputs: dict[str, float],
        observable_names: list[str],
        *,
        check_cancellation: Callable[[], None] | None = None,
    ) -> dict[str, PVReading]:
        self.set_variables(variable_inputs, check_cancellation=check_cancellation)
        return self.get_observables(observable_names, check_cancellation=check_cancellation)

    def read_pvs(self, names: list[str]) -> dict[str, PVReading]:
        return self.interface.read_many(names)

    def check_connection(self) -> bool | None:
        first = next(iter(self.variables.values()), None)
        return self.interface.check_connection(first.readback_pv if first else None)

    def check_machine_settings(self) -> None:
        return None

    def __enter__(self) -> ConnectorEnvironment:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # The interface rides the process's pool bridge, which outlives any one environment.
        return None
