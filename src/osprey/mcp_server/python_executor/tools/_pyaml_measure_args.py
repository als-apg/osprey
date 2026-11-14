"""Server-side resolution of a ``pyaml_measure`` request.

:func:`resolve_measure_args` checks everything the tool can check before a
script is rendered or an approval is asked for, and returns the
:class:`MeasureArgs` the script builder reads. Every refusal is a
``validation_error`` raised through :func:`~osprey.mcp_server.errors.make_error`.

Every file is read from the RENDER root, the directory holding the rendered
``config.yml``: the facts document (``data/facility_facts.json``), whose
``measurement_kinds`` table names every kind the deployment registers and whose
``measurement_models`` record each pyAML view with its model's kind block, and
the view itself (``data/pyaml/<model>/``). A host render (``<repo>/build/``)
and a container project, whose root is its own render, therefore resolve the
same relative path. Nothing here imports a kind's declaration or the pyAML
view writer: what a kind is and how it behaves is what the build wrote.

The checks:

* ``kind`` is one of :func:`kinds`, the table's kinds;
* ``model`` names a model with a pyAML view; it may be omitted only when
  exactly one model has one;
* the model's measurement file allows the kind; a kind whose instrument or
  group the model lacks is refused as
  ``<kind> needs <instrument or group>, which <model> does not have``, and a
  kind whose members exist but which ``kinds`` leaves out is refused as
  ``<kind> is not a kind measurement/<model>.yaml allows``;
* ``by`` is one of :data:`BYS`, taken by the kinds the table marks
  ``takes_by`` alone, ``magnet`` when omitted;
* ``modulation`` is one of :data:`MODULATIONS`; a group is stepped once from
  its present setting, so ``by`` ``group`` is ``unipolar`` when modulation is
  omitted and refuses ``bipolar``; every other omitted modulation is
  ``bipolar``;
* ``correctors`` are setpoint addresses of the kind's ``corrector_roles``
  groups, each named once, taken by the kinds that name corrector roles alone;
* the view's configuration resolves under ``<render>/data/pyaml/`` (symlinks
  followed) and exists;
* the execution timeout (``python_executor.execution_timeout_seconds``) is a
  finite positive number: a measurement moves the machine, the executor's kill
  at that timeout is its one time bound, and a measurement with no bound is
  not started.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from osprey.mcp_server.errors import make_error

__all__ = [
    "BYS",
    "DEFAULT_BY",
    "DEFAULT_MODULATION",
    "GROUP",
    "GROUP_MODULATION",
    "MODULATIONS",
    "TIMEOUT_KEY",
    "MeasureArgs",
    "by_kinds",
    "corrector_kinds",
    "kinds",
    "resolve_measure_args",
]

#: How a step is taken about the setting the machine holds: on both sides of it,
#: or on one.
MODULATIONS = ("bipolar", "unipolar")
DEFAULT_MODULATION = "bipolar"

#: What one column of a tune or chromaticity response steps: one magnet, or
#: every magnet of the group as one knob.
BYS = ("magnet", "group")
DEFAULT_BY = "magnet"

#: The ``by`` that steps a whole group as one knob, and the one modulation it runs.
GROUP = "group"
GROUP_MODULATION = "unipolar"

#: The config key holding the sandbox run's timeout in seconds.
TIMEOUT_KEY = "python_executor.execution_timeout_seconds"

_REBUILD = "Run `osprey build`; the pyAML view is written by the build."


@dataclass(frozen=True)
class MeasureArgs:
    """A resolved measurement request.

    Attributes:
        kind: A kind the model's measurement file allows.
        model: The model measured, one with a pyAML view.
        correctors: The corrector setpoint addresses given, in the order given;
            ``None`` when none were given.
        modulation: One of :data:`MODULATIONS`.
        by: One of :data:`BYS` for a kind that takes ``by``; ``None`` for
            every other kind.
        render_root: The directory holding the rendered ``config.yml``.
        config_path: Absolute, symlink-resolved path of the view's
            configuration, an existing file under ``<render>/data/pyaml/``.
        view_path: The configuration's path under the render, as the facts
            record states it.
        view_sha256: The configuration's digest, as the facts record states it.
        timeout: The sandbox run's timeout in seconds, finite and positive.
        plan: The kind's measure plan, as the build rendered it.
        matrix: Whether the kind's completed result is a response matrix.
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
    plan: Mapping[str, Any]
    matrix: bool


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
    """The measurement's time bound, in seconds.

    A measurement moves the machine. The executor kills it at
    ``python_executor.execution_timeout_seconds``, which is its one time
    bound, so a measurement with no bound is not started. The executor's own
    rule decides the value; this surface only words the refusal.
    """
    from osprey.mcp_server.python_executor.executor import execution_timeout_seconds

    try:
        return execution_timeout_seconds(timeout)
    except ValueError:
        _refuse(
            f"{TIMEOUT_KEY} is {timeout!r}; a measurement moves the machine and the "
            "executor's kill at this timeout is its one time bound",
            f"Set {TIMEOUT_KEY} to a finite, positive number of seconds; a measurement "
            "does not start without a time bound.",
        )


def _facts(render_root: Path) -> Mapping[str, Any]:
    from osprey.facility.views.facts import read_facts

    return read_facts(render_root, "")


def kinds(facts: Mapping[str, Any]) -> tuple[str, ...]:
    """Every kind the render's ``measurement_kinds`` table names, in its order."""
    return tuple(str(kind) for kind in facts.get("measurement_kinds") or {})


def by_kinds(facts: Mapping[str, Any]) -> tuple[str, ...]:
    """The kinds the table marks as taking ``by``."""
    table = facts.get("measurement_kinds") or {}
    return tuple(str(kind) for kind, row in table.items() if row.get("takes_by"))


def corrector_kinds(facts: Mapping[str, Any]) -> tuple[str, ...]:
    """The kinds the table gives corrector roles, the kinds that take ``correctors``."""
    table = facts.get("measurement_kinds") or {}
    return tuple(str(kind) for kind, row in table.items() if row.get("corrector_roles"))


def _view_records(facts: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Each model's pyAML view record, from the render's facts."""
    records = facts.get("measurement_models") or {}
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
    from osprey.facility.views.facts import PYAML_DIR

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


def _refuse_lacking(
    kind: str, model: str, row: Mapping[str, Any], record: Mapping[str, Any]
) -> Mapping[str, Any]:
    """The model's block of *kind*; refuse a kind it cannot run, naming what it lacks.

    A member the kind requires is lacking when the model's view gives its role
    no name and its measurement file names no instrument for it.
    """
    allowed = sorted(record.get("kinds") or {})
    if kind in allowed:
        found: Mapping[str, Any] = record["kinds"][kind]
        return found
    has = {*(record.get("names") or {}), *(record.get("instruments") or {})}
    lacking = [
        str(member)
        for member, requirement in (row.get("members") or {}).items()
        if requirement != "optional" and member not in has
    ]
    if lacking:
        _refuse(
            f"{kind} needs {' and '.join(lacking)}, which {model} does not have",
            f"Measure a kind {model} allows: {', '.join(allowed) or 'none'}.",
        )
    _refuse(
        f"{kind} is not a kind measurement/{model}.yaml allows; it allows: "
        f"{', '.join(allowed) or 'none'}",
        f"Measure a kind {model} allows, or add {kind} to `kinds` and rebuild.",
    )


def _correctors(
    kind: str, correctors: object, model: str, block: Mapping[str, Any], takers: Sequence[str]
) -> tuple[str, ...] | None:
    if correctors is None:
        return None
    roles = [str(role) for role in block.get("corrector_roles") or []]
    if not roles:
        stepped = list(block.get("stepped_roles") or [])
        why = (
            f"{kind} steps the RF, not the correctors"
            if stepped == ["rf"]
            else f"{kind} steps no named correctors"
        )
        _refuse(
            f"correctors are taken by {' and '.join(takers)} alone; {why}",
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
    members = block.get("members") or {}
    known = {
        str(address)
        for role in roles
        for address in (members.get(role) or {}).get("addresses") or []
    }
    seen: set[str] = set()
    for address in correctors:
        if not isinstance(address, str):
            _refuse(
                f"correctors entry {address!r} is not a string",
                "Pass every corrector as its setpoint address.",
            )
        if address not in known:
            _refuse(
                f"{address} is no corrector setpoint of {model}'s {' or '.join(roles)} group",
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
        kind: The measurement kind, one of the render's :func:`kinds`.
        model: The model to measure, or ``None`` when exactly one model has a
            pyAML view.
        correctors: Corrector setpoint addresses for a kind that takes them,
            or ``None``.
        modulation: One of :data:`MODULATIONS`, or ``None`` for ``unipolar``
            with ``by`` ``group`` and ``bipolar`` otherwise.
        by: One of :data:`BYS` for a kind that takes ``by``, or ``None`` for
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
    render_root = Path(config_path).parent
    facts = _facts(render_root)
    known = kinds(facts)
    if not known:
        _model(model, _view_records(facts))
        _refuse("this render's facts name no measurement kind", _REBUILD)
    checked_kind = _one_of("kind", kind, known)
    takers = by_kinds(facts)
    if by is not None and checked_kind not in takers:
        _refuse(
            f"by is taken by {' and '.join(takers) or 'no kind'} alone; {checked_kind} has no "
            "magnets to group",
            f"Omit by for {checked_kind}.",
        )
    checked_by: str | None = None
    if checked_kind in takers:
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

    records = _view_records(facts)
    checked_model = _model(model, records)
    record = records[checked_model]
    path = _config_path(render_root, checked_model, record)
    row = (facts.get("measurement_kinds") or {})[checked_kind]
    block = _refuse_lacking(checked_kind, checked_model, row, record)
    checked_correctors = _correctors(
        checked_kind, correctors, checked_model, block, corrector_kinds(facts)
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
        plan=dict(block.get("measure_plan") or {}),
        matrix=bool(block.get("matrix")),
    )
