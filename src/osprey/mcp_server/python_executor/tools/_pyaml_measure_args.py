"""Server-side resolution of a ``pyaml_measure`` request.

:func:`resolve_measure_args` checks everything the tool can check before a
script is rendered or an approval is asked for, and returns the
:class:`MeasureArgs` the script builder reads. Every refusal is a
``validation_error`` raised through :func:`~osprey.mcp_server.errors.make_error`.

Every file is read from the RENDER root, the directory holding the rendered
``config.yml``: the facts record of each pyAML view
(``data/facility_facts.json``), the view itself (``data/pyaml/<model>/``) and
the facility file. A host render (``<repo>/build/``) and a container project,
whose root is its own render, therefore resolve the same relative path.

The checks:

* ``kind`` is one of :data:`KINDS`;
* ``model`` names a model with a pyAML view; it may be omitted only when
  exactly one model has one;
* the model's measurement file allows the kind; a kind it lacks is refused as
  ``<kind> needs <instrument or group>, which <model> does not have``;
* ``by`` is one of :data:`BYS`, taken by ``trm`` and ``crm`` alone, ``magnet``
  when omitted;
* ``modulation`` is one of :data:`MODULATIONS`; a group is stepped once from
  its present setting, so ``by`` ``group`` is ``unipolar`` when modulation is
  omitted and refuses ``bipolar``; every other omitted modulation is
  ``bipolar``;
* ``correctors`` are setpoint addresses of the model's ``hcor`` or ``vcor``
  group, each named once, taken by ``orm`` and ``dispersion`` alone;
* the view's configuration resolves under ``<render>/data/pyaml/`` (symlinks
  followed) and exists;
* the execution timeout (``python_executor.execution_timeout_seconds``) is a
  finite positive number, since the measurement's deadline guard needs a
  deadline to stop before.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from osprey.mcp_server.errors import make_error

__all__ = [
    "BYS",
    "BY_KINDS",
    "CORRECTOR_KINDS",
    "DEFAULT_BY",
    "DEFAULT_MODULATION",
    "GROUP",
    "GROUP_MODULATION",
    "KINDS",
    "MODULATIONS",
    "STEPPED",
    "TIMEOUT_KEY",
    "MeasureArgs",
    "resolve_measure_args",
]

#: The measurement kinds the tool runs.
KINDS = ("orm", "dispersion", "trm", "crm", "chromaticity_monitor")

#: How a step is taken about the setting the machine holds: on both sides of it,
#: or on one.
MODULATIONS = ("bipolar", "unipolar")
DEFAULT_MODULATION = "bipolar"

#: What one column of a tune or chromaticity response steps: one magnet, or
#: every magnet of the group as one knob.
BYS = ("magnet", "group")
DEFAULT_BY = "magnet"

#: The kinds that take ``by``.
BY_KINDS = ("trm", "crm")

#: The ``by`` that steps a whole group as one knob, and the one modulation it runs.
GROUP = "group"
GROUP_MODULATION = "unipolar"

#: The kinds that take ``correctors``.
CORRECTOR_KINDS = ("orm",)

#: The setpoints each kind writes: its group roles, and ``rf`` for the setpoint
#: the view's RF plant steps. The approval prompt lists every one of them.
STEPPED: dict[str, tuple[str, ...]] = {
    "orm": ("hcor", "vcor"),
    "dispersion": ("rf",),
    "trm": ("quad",),
    "crm": ("sext", "rf"),
    "chromaticity_monitor": ("rf",),
}

#: The group roles whose setpoints ``correctors`` may name.
_CORRECTOR_ROLES = ("hcor", "vcor")

#: The config key holding the sandbox run's timeout in seconds.
TIMEOUT_KEY = "python_executor.execution_timeout_seconds"

_REBUILD = "Run `osprey build`; the pyAML view is written by the build."


@dataclass(frozen=True)
class MeasureArgs:
    """A resolved measurement request.

    Attributes:
        kind: One of :data:`KINDS`.
        model: The model measured, one with a pyAML view.
        correctors: The corrector setpoint addresses given, in the order given;
            ``None`` when none were given.
        modulation: One of :data:`MODULATIONS`.
        by: One of :data:`BYS` for ``trm`` and ``crm``; ``None`` for every
            other kind.
        render_root: The directory holding the rendered ``config.yml``.
        config_path: Absolute, symlink-resolved path of the view's
            configuration, an existing file under ``<render>/data/pyaml/``.
        view_path: The configuration's path under the render, as the facts
            record states it.
        view_sha256: The configuration's digest, as the facts record states it.
        timeout: The sandbox run's timeout in seconds, finite and positive.
    """

    kind: str
    model: str
    correctors: tuple[str, ...] | None
    modulation: str
    by: str | None
    render_root: Path
    config_path: Path
    view_path: str
    view_sha256: str
    timeout: float


def _refuse(message: str, suggestion: str) -> NoReturn:
    make_error("validation_error", message, [suggestion])


def _one_of(name: str, value: object, allowed: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        _refuse(
            f"{name} {value!r} is not one of {', '.join(allowed)}",
            f"Pass {name} as one of: {', '.join(allowed)}.",
        )
    return value


def _timeout(timeout: object) -> float:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        _refuse(
            f"{TIMEOUT_KEY} is {timeout!r}; a measurement needs a finite, positive timeout "
            "so its deadline guard can stop and restore before the run is killed",
            f"Set {TIMEOUT_KEY} to a positive number of seconds.",
        )
    return float(timeout)


def _view_records(render_root: Path) -> dict[str, Mapping[str, Any]]:
    """Each model's pyAML view record, from the render's facts file."""
    from osprey.facility.views.facts import read_facts

    records = read_facts(render_root, "").get("measurement_models") or {}
    return {str(model): record for model, record in sorted(records.items())}


def _model(model: object, records: Mapping[str, Any]) -> str:
    listed = ", ".join(records)
    if not records:
        _refuse("this render carries no pyAML view, so there is no model to measure", _REBUILD)
    if model is None:
        if len(records) > 1:
            _refuse(
                f"model is omitted and more than one model has a pyAML view: {listed}",
                f"Pass model as one of: {listed}.",
            )
        return next(iter(records))
    if not isinstance(model, str) or model not in records:
        _refuse(
            f"model {model!r} has no pyAML view; the models that have one: {listed}",
            f"Pass model as one of: {listed}.",
        )
    return model


def _config_path(render_root: Path, model: str, record: Mapping[str, Any]) -> Path:
    from osprey.facility.views.pyaml import PYAML_DIR

    stated = record.get("path")
    if not isinstance(stated, str) or not stated:
        _refuse(f"the facts record of {model}'s pyAML view states no path", _REBUILD)
    resolved = (render_root / stated).resolve()
    view_root = (render_root / "data" / PYAML_DIR).resolve()
    if not resolved.is_relative_to(view_root):
        _refuse(
            f"the pyAML view of {model} is recorded at {resolved}, which is not under {view_root}",
            _REBUILD,
        )
    if not resolved.is_file():
        _refuse(
            f"the pyAML view of {model} is recorded at {resolved}, which does not exist", _REBUILD
        )
    return resolved


def _measurement(render_root: Path, model: str) -> Mapping[str, Any]:
    """The model's measurement record in the render's facility file."""
    from osprey.facility import FACILITY_FILE

    document = json.loads((render_root / FACILITY_FILE).read_text(encoding="utf-8"))
    for record in document.get("models") or []:
        if record.get("name") == model:
            found: Mapping[str, Any] = record.get("measurement") or {}
            return found
    return {}


def _refuse_lacking(
    kind: str,
    model: str,
    block: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> None:
    """Refuse a kind the model's view cannot run, naming what the model lacks."""
    from osprey.facility.views.pyaml import GROUP_ROLES, KIND_MEMBERS

    allowed = list(block.get("kinds") or {})
    if kind in allowed:
        return
    groups = block.get("groups") or {}
    instruments = measurement.get("instruments") or {}
    lacking = [
        member
        for member in KIND_MEMBERS[kind]
        if (member not in groups if member in GROUP_ROLES else member not in instruments)
    ]
    if lacking:
        _refuse(
            f"{kind} needs {' and '.join(lacking)}, which {model} does not have",
            f"Measure a kind {model} allows: {', '.join(allowed) or 'none'}.",
        )
    if kind not in allowed:
        _refuse(
            f"{kind} is not a kind measurement/{model}.yaml allows; it allows: "
            f"{', '.join(allowed) or 'none'}",
            f"Measure a kind {model} allows, or add {kind} to `kinds` and rebuild.",
        )


def _correctors(
    kind: str, correctors: object, model: str, groups: Mapping[str, Any]
) -> tuple[str, ...] | None:
    if correctors is None:
        return None
    if kind not in CORRECTOR_KINDS:
        why = (
            "dispersion steps the RF, not the correctors"
            if kind == "dispersion"
            else f"{kind} steps no named correctors"
        )
        _refuse(
            f"correctors are taken by {' and '.join(CORRECTOR_KINDS)} alone; {why}",
            f"Omit correctors for {kind}.",
        )
    if isinstance(correctors, str | bytes) or not isinstance(correctors, Sequence):
        _refuse(
            f"correctors must be a list of setpoint addresses, got {correctors!r}",
            "Pass correctors as a list of corrector setpoint addresses.",
        )
    if not correctors:
        _refuse(
            "correctors is empty",
            "Name the corrector setpoints to step, or omit correctors to step every one.",
        )
    known = {address for role in _CORRECTOR_ROLES for address in groups.get(role) or []}
    seen: set[str] = set()
    for address in correctors:
        if not isinstance(address, str):
            _refuse(
                f"correctors entry {address!r} is not a string",
                "Pass every corrector as its setpoint address.",
            )
        if address not in known:
            _refuse(
                f"{address} is no corrector setpoint of {model}'s "
                f"{' or '.join(_CORRECTOR_ROLES)} group",
                f"Use the setpoint addresses of {model}'s corrector groups.",
            )
        if address in seen:
            _refuse(
                f"corrector setpoint {address} appears more than once in correctors",
                "Name each corrector once.",
            )
        seen.add(address)
    return tuple(correctors)


def resolve_measure_args(
    kind: object,
    model: object,
    correctors: object,
    modulation: object,
    by: object,
    *,
    config_path: Path,
    timeout: object,
) -> MeasureArgs:
    """Resolve a ``pyaml_measure`` request against the render it runs in.

    Args:
        kind: The measurement kind, one of :data:`KINDS`.
        model: The model to measure, or ``None`` when exactly one model has a
            pyAML view.
        correctors: Corrector setpoint addresses for ``orm``, or ``None``.
        modulation: One of :data:`MODULATIONS`, or ``None`` for ``unipolar``
            with ``by`` ``group`` and ``bipolar`` otherwise.
        by: One of :data:`BYS` for ``trm`` or ``crm``, or ``None`` for
            ``magnet``.
        config_path: The rendered ``config.yml`` this process runs against; its
            directory is the render root every file is read from.
        timeout: The configured ``python_executor.execution_timeout_seconds``.

    Returns:
        The resolved :class:`MeasureArgs`.

    Raises:
        fastmcp.exceptions.ToolError: A ``validation_error`` envelope naming the
            argument, the model or the file at fault.
    """
    from osprey.facility.views.facts import hook_measurement

    render_root = Path(config_path).parent
    checked_kind = _one_of("kind", kind, KINDS)
    if by is not None and checked_kind not in BY_KINDS:
        _refuse(
            f"by is taken by {' and '.join(BY_KINDS)} alone; {checked_kind} has no magnets "
            "to group",
            f"Omit by for {checked_kind}.",
        )
    checked_by: str | None = None
    if checked_kind in BY_KINDS:
        checked_by = DEFAULT_BY if by is None else _one_of("by", by, BYS)
    if modulation is None:
        checked_modulation = GROUP_MODULATION if checked_by == GROUP else DEFAULT_MODULATION
    else:
        checked_modulation = _one_of("modulation", modulation, MODULATIONS)
    if checked_by == GROUP and checked_modulation != GROUP_MODULATION:
        _refuse(
            f"by group steps each group once from its present setting, so it is "
            f"{GROUP_MODULATION}; modulation is {checked_modulation}",
            f"Pass modulation {GROUP_MODULATION!r} with by 'group'.",
        )
    checked_timeout = _timeout(timeout)

    records = _view_records(render_root)
    checked_model = _model(model, records)
    record = records[checked_model]
    path = _config_path(render_root, checked_model, record)
    try:
        block = hook_measurement({"measurement_models": {checked_model: record}}, render_root)[
            checked_model
        ]
        measurement = _measurement(render_root, checked_model)
    except Exception as exc:
        _refuse(f"the pyAML view of {checked_model} cannot be read: {exc}", _REBUILD)
    _refuse_lacking(checked_kind, checked_model, block, measurement)
    checked_correctors = _correctors(
        checked_kind, correctors, checked_model, block.get("groups") or {}
    )
    return MeasureArgs(
        kind=checked_kind,
        model=checked_model,
        correctors=checked_correctors,
        modulation=checked_modulation,
        by=checked_by,
        render_root=render_root,
        config_path=path,
        view_path=str(record["path"]),
        view_sha256=str(record.get("sha256") or ""),
        timeout=checked_timeout,
    )
