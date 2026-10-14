"""Environment files: an optimization environment declared in YAML, no Python.

One file per environment under the deployment's environments directory::

    name: va_vertical_orbit
    description: Vertical orbit correction with three correctors
    variables:
      SR:MAG:VCM:01:CURRENT:SP:
        readback: SR:MAG:VCM:01:CURRENT:RB
        min: -12.0
        max: 12.0
        step_size: 0.01     # required: suggestions snap to this grid
        units: A
    observables:
      orbit_rms_y:
        reduce: rms
        channels: ["SR:DIAG:BPM:{01..72}:POSITION:Y"]
        direction: MINIMIZE
        units: m
    default_objective: orbit_rms_y
    presets:
      gentle:
        description: Corrector 01 only, within 1 A of where it is
        variables: [SR:MAG:VCM:01:CURRENT:SP]
        variable_overrides:
          SR:MAG:VCM:01:CURRENT:SP: {window_abs: 1.0}
        algorithm: upper_confidence_bound
        n_iterations: 30

A variable's key is its setpoint channel unless ``setpoint`` says otherwise;
its readback defaults to the setpoint. ``{01..72}`` in a channel name expands
to one channel per number, zero-padded to the width written.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from osprey.services.optimizer.models.config import Preset
from osprey.services.optimizer.models.protocols import (
    ObservableMetadata,
    PVObservable,
    ReducedObservable,
    VariableMetadata,
)
from osprey.services.optimizer.models.types import Direction, validate_min_le_max

_RANGE = re.compile(r"\{(\d+)\.\.(\d+)\}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
# An environment file is operator-authored config, not data; anything this big is a mistake.
MAX_FILE_BYTES = 256 * 1024


def expand_channels(pattern: str) -> list[str]:
    """Expand every ``{a..b}`` range in ``pattern``, zero-padded to the width of ``a``.

    Several ranges expand as a product, left to right. A descending range is
    refused rather than silently reversed or emptied.
    """
    match = _RANGE.search(pattern)
    if match is None:
        return [pattern]
    first, last = match.group(1), match.group(2)
    lo, hi = int(first), int(last)
    if lo > hi:
        raise ValueError(f"Descending channel range {match.group(0)!r} in {pattern!r}")
    width = len(first)
    head, tail = pattern[: match.start()], pattern[match.end() :]
    return [
        expanded
        for n in range(lo, hi + 1)
        for expanded in expand_channels(f"{head}{n:0{width}d}{tail}")
    ]


class VariableSpec(BaseModel):
    """One knob: a setpoint channel the optimizer writes, and where to read it back."""

    model_config = ConfigDict(extra="forbid")

    setpoint: str | None = None
    readback: str | None = None
    min: float
    max: float
    step_size: float = Field(gt=0.0)
    """The finest change the optimizer makes: suggestions snap to this grid."""
    units: str = ""
    description: str = ""
    readback_tolerance: float | None = Field(default=None, ge=0.0)
    """How close the readback must come to the value written. Defaults to ``step_size``."""

    @model_validator(mode="after")
    def _bounds(self) -> VariableSpec:
        validate_min_le_max(self.min, self.max)
        return self


class ObservableSpec(BaseModel):
    """One measurement: a single channel, or a reduction over several."""

    model_config = ConfigDict(extra="forbid")

    channel: str | None = None
    reduce: Literal["rms", "mean", "max"] | None = None
    channels: list[str] | None = None
    direction: Direction = Direction.MAXIMIZE
    units: str = ""
    display_name: str = ""
    description: str = ""
    selectable_as_objective: bool = True
    target_value: float | None = None

    @model_validator(mode="after")
    def _one_source(self) -> ObservableSpec:
        if (self.channel is None) == (self.reduce is None):
            raise ValueError("an observable names either `channel`, or `reduce` with `channels`")
        if self.channel is not None and self.channels is not None:
            raise ValueError("`channels` goes with `reduce`, not with `channel`")
        if self.reduce is not None and not self.channels:
            raise ValueError("`reduce` needs a non-empty `channels` list")
        return self


class EnvironmentFile(BaseModel):
    """The whole file. Its ``name`` is the environment's name in the API."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=rf"^{_NAME.pattern}$")
    display_name: str = ""
    description: str = ""
    variables: dict[str, VariableSpec] = Field(min_length=1)
    observables: dict[str, ObservableSpec] = Field(min_length=1)
    default_objective: str | None = None
    settle_seconds: float = Field(default=0.0, ge=0.0)
    """Wait after every write, once readbacks agree, before reading observables."""
    readback_timeout_seconds: float = Field(default=5.0, gt=0.0)
    presets: dict[str, Preset] = Field(default_factory=dict)
    """Named run setups an operator or the agent can start from."""

    @model_validator(mode="after")
    def _objective_known(self) -> EnvironmentFile:
        if self.default_objective is not None:
            spec = self.observables.get(self.default_objective)
            if spec is None:
                raise ValueError(
                    f"default_objective {self.default_objective!r} is not an observable "
                    f"(have {sorted(self.observables)})"
                )
            if not spec.selectable_as_objective:
                raise ValueError(f"default_objective {self.default_objective!r} is not selectable")
        elif not any(o.selectable_as_objective for o in self.observables.values()):
            raise ValueError("no observable is selectable as an objective")
        return self

    @model_validator(mode="after")
    def _presets_known(self) -> EnvironmentFile:
        for key, preset in self.presets.items():
            if not _NAME.fullmatch(key):
                raise ValueError(f"preset name {key!r} is not a plain identifier")
            named = [*(preset.variables or []), *preset.variable_overrides]
            unknown = sorted({n for n in named if n not in self.variables})
            if unknown:
                raise ValueError(f"preset {key!r} names unknown variable(s) {unknown}")
            objective = preset.objective_name
            if objective is not None and not (
                objective in self.observables
                and self.observables[objective].selectable_as_objective
            ):
                raise ValueError(f"preset {key!r}: {objective!r} is not a selectable observable")
        return self

    def variable_metadata(self) -> dict[str, VariableMetadata]:
        """The variables in the engine's own metadata shape."""
        out: dict[str, VariableMetadata] = {}
        for name, v in self.variables.items():
            setpoint = v.setpoint or name
            out[name] = VariableMetadata(
                min_value=v.min,
                max_value=v.max,
                units=v.units,
                description=v.description,
                setpoint_pv=setpoint,
                readback_pv=v.readback or setpoint,
                step_size=v.step_size,
            )
        return out

    def observable_metadata(self) -> dict[str, ObservableMetadata]:
        """The observables in the engine's own metadata shape, channel ranges expanded."""
        out: dict[str, ObservableMetadata] = {}
        for name, o in self.observables.items():
            common: dict[str, object] = {
                "display_name": o.display_name,
                "description": o.description,
                "units": o.units,
                "direction": o.direction,
                "selectable_as_objective": o.selectable_as_objective,
                "target_value": o.target_value,
            }
            if o.reduce is not None:
                channels = [c for pattern in o.channels or [] for c in expand_channels(pattern)]
                out[name] = ReducedObservable.model_validate(
                    {**common, "reduce": o.reduce, "channels": channels}
                )
            else:
                assert o.channel is not None  # _one_source
                out[name] = PVObservable.model_validate({**common, "readback_pv": o.channel})
        return out

    def readback_tolerance(self, name: str) -> float:
        v = self.variables[name]
        return v.readback_tolerance if v.readback_tolerance is not None else v.step_size


def load_environment_file(path: Path) -> EnvironmentFile:
    """Parse and validate one environment file."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path} is larger than {MAX_FILE_BYTES} bytes")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a mapping")
    env = EnvironmentFile.model_validate(data)
    # Expand eagerly so a bad range fails at load, not at the first evaluation.
    env.observable_metadata()
    return env
