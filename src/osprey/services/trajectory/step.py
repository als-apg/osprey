"""The correction tool the trajectory service computes its steps with.

The tool is pyAML's :class:`pyaml.tuning_tools.trajectory.Trajectory`, built
once at service start from the selected model's pyAML view: its
``trajectory_rm.json`` (the first-turn response matrix, rows named per plane,
columns every horizontal corrector then every vertical one) and the sidecar
``trajectory_rm.index.json`` beside it (``bpm_index``, ``hcorr_index``,
``vcorr_index`` and ``unsolved``). The tool's ``alpha``,
``n_discard`` and ``gain`` are the model's measurement-file values under
:data:`osprey.facility.views.pyaml.TOOL_KEYS` ``["trajectory"]``, read from the
render's ``facility.json``; the pyAML view carries no tool section for this
kind.

pyAML is imported inside :func:`build_tool`, never at module import: its
package reads its logging configuration when first imported, so the import
happens after the service configured logging.

Building never raises. A model with no response matrix or no sidecar, a
sidecar that does not describe the matrix, a tool that loaded no matrix (a
``None`` layout), or a ``PyAMLException`` (or ``PyAMLConfigException``) while
building gives a :class:`ToolUnavailable` naming the cause; the service starts
with it in its tool slot, and every step or apply that needs the tool is
refused through :func:`require_tool` with a :class:`StepRefused` that carries
that cause.

The step is computed from a shot and the current reference by
:func:`compute_step`: the tool's ``correct`` gives ``step`` in column order (every
horizontal corrector, then every vertical one), and the increment written is
``δ = −factor·step``, where ``factor`` is the largest uniform scale at most 1
that keeps every ``|δ|`` strictly inside its ``max_step``. A step solves on the
model matrix restricted to the shot's excluded BPMs and the SolveConfig's
excluded correctors (:func:`restricted_tool`), with the SolveConfig's
``alpha`` and ``gain`` and the discard count its ``keep_live`` stands for over
that matrix (:func:`n_discard_for`), all passed per call. The step's digest
binds the target, the model, the holder's ``client_id``, the shot id, the
reference's name and content sha256, the solve identity (:func:`solve_sha`)
and ``δ``, so an apply that carries it writes exactly the step it was shown.
The spectrum is the published solve's, never the step's. :class:`StepState`
holds the one pending step of a started service; a new shot or a reference
change replaces it, and a take-over voids it. ``correct`` runs off the event
loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.state import ShotResult, Step, StepReason

if TYPE_CHECKING:
    from osprey.services.trajectory.startup import TrajectoryStartup
    from osprey.services.trajectory.state import SolveConfig
    from osprey.services.trajectory.store import LoadedReference, TrajectoryStore

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_STEP_MARGIN",
    "REFERENCE_UNUSABLE",
    "RM_SOURCE",
    "SELECTION_UNKNOWN",
    "SHOT_MISMATCH",
    "STEP_FAILED",
    "TOOL_NAME",
    "TOOL_UNAVAILABLE",
    "RmSummary",
    "StepRefused",
    "StepState",
    "ToolUnavailable",
    "TrajectoryTool",
    "build_service_tool",
    "build_tool",
    "compute_step",
    "family_index",
    "keep_live_default",
    "max_steps",
    "model_of",
    "n_discard_for",
    "per_call_settings",
    "require_tool",
    "restricted_tool",
    "rm_summary",
    "solve_sha",
    "step_digest",
    "step_factor",
    "step_for_shot",
    "view_dir",
]

#: The name the built tool carries.
TOOL_NAME = "TRAJECTORY"

#: The refusal code of a step or apply the service has no tool for.
TOOL_UNAVAILABLE = "tool_unavailable"

#: The refusal code of a shot whose rows are not the tool's.
SHOT_MISMATCH = "shot_mismatch"

#: The refusal code of a current reference that cannot be matched to the shot.
REFERENCE_UNUSABLE = "reference_unusable"

#: The refusal code of a shot the tool refused to solve.
STEP_FAILED = "step_failed"

#: The refusal code of an exclusion naming no BPM or corrector of the build.
SELECTION_UNKNOWN = "selection_unknown"

#: The source the published response-matrix summary names: the build-time model
#: matrix, the service's only one.
RM_SOURCE = "model"

#: How many restricted tools are kept, the most recently used first.
_RESTRICTED_CACHE_SIZE = 16

#: The relative margin kept below every ``max_step``, so a step scaled to a
#: binding ``max_step`` passes the validator's strict ``>`` check.
MAX_STEP_MARGIN = 1e-9

#: The settings a ``correct`` call takes per call.
_SETTING_KEYS = ("alpha", "n_discard", "gain")

#: The sidecar's lists, each one entry per matrix row or per corrector column.
_SIDECAR_LISTS = ("bpm_index", "hcorr_index", "vcorr_index")


class StepRefused(Refused):
    """A step or apply the service cannot compute; nothing was written."""

    statuses = {
        TOOL_UNAVAILABLE: 503,
        SHOT_MISMATCH: 409,
        REFERENCE_UNUSABLE: 409,
        STEP_FAILED: 422,
        SELECTION_UNKNOWN: 409,
    }


@dataclass(frozen=True)
class TrajectoryTool:
    """The built tool and the sidecar facts its every ``correct`` call needs.

    Attributes:
        model: The model the response matrix was solved for.
        tool: The :class:`pyaml.tuning_tools.trajectory.Trajectory`.
        layout: The tool's :class:`~pyaml.tuning_tools.trajectory.TrajectoryLayout`,
            never ``None``.
        bpm_index: The lattice index of every matrix row, x rows then y rows,
            passed to ``correct`` verbatim.
        hcorr_index: The lattice index of every horizontal corrector column.
        vcorr_index: The lattice index of every vertical corrector column.
        unsolved: Each corrector whose response has no solve on the model, to
            the engine's text; its column is a column of zeros.
        alpha: The tool's configured ``alpha``.
        n_discard: The tool's configured ``n_discard``.
        gain: The tool's configured ``gain``.
    """

    model: str
    tool: Any
    layout: Any
    bpm_index: tuple[float, ...]
    hcorr_index: tuple[float, ...]
    vcorr_index: tuple[float, ...]
    unsolved: Mapping[str, str]
    alpha: float
    n_discard: int
    gain: float


@dataclass(frozen=True)
class ToolUnavailable:
    """Why the service has no tool; a step or apply is refused with it.

    Attributes:
        detail: The sentence an operator reads: what is missing or wrong, and where.
    """

    detail: str

    def refusal(self) -> StepRefused:
        """The refusal a step or apply that needs the tool raises."""
        return StepRefused(TOOL_UNAVAILABLE, self.detail)


def require_tool(slot: TrajectoryTool | ToolUnavailable | None) -> TrajectoryTool:
    """The built tool, or the refusal of a step or apply that needs one.

    Args:
        slot: The started service's tool slot.

    Returns:
        The built tool.

    Raises:
        StepRefused: ``tool_unavailable`` (503), with the cause the builder
            recorded, or naming the empty slot.
    """
    if isinstance(slot, TrajectoryTool):
        return slot
    if isinstance(slot, ToolUnavailable):
        raise slot.refusal()
    raise StepRefused(TOOL_UNAVAILABLE, "the service built no correction tool")


def model_of(startup: TrajectoryStartup) -> str:
    """The model the started service drives: its command set's, else its tool's.

    Raises:
        StepRefused: ``tool_unavailable`` with neither a command set nor a tool.
    """
    commands = startup.commands
    return commands.model if commands is not None else require_tool(startup.tool).model


def view_dir(render_root: Path, model: str) -> Path:
    """The directory of ``model``'s pyAML view under the render at ``render_root``."""
    return Path(render_root) / "data" / "pyaml" / model


def _tool_values(render_root: Path, model: str) -> dict[str, Any]:
    """The model's ``TOOL_KEYS["trajectory"]`` values from the render's ``facility.json``.

    Raises:
        ValueError: The file cannot be read, holds no record of ``model``, or
            the measurement file states a key with no value of its kind.
    """
    from osprey.facility import FACILITY_FILE
    from osprey.facility.views.pyaml import TOOL_KEYS

    path = Path(render_root) / FACILITY_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"the facility file {path} cannot be read: {exc}") from exc
    models = document.get("models") if isinstance(document, Mapping) else None
    record = next(
        (
            entry
            for entry in models or []
            if isinstance(entry, Mapping) and entry.get("name") == model
        ),
        None,
    )
    if record is None:
        raise ValueError(f"the facility file {path} holds no model {model!r}")
    measurement = record.get("measurement")
    measurement = measurement if isinstance(measurement, Mapping) else {}
    values: dict[str, Any] = {}
    for key in TOOL_KEYS["trajectory"]:
        value = measurement.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"measurement/{model}.yaml states no number for {key}, which the "
                "trajectory tool is built with"
            )
        if not math.isfinite(value):
            raise ValueError(f"measurement/{model}.yaml states {key} = {value!r}, not finite")
        values[key] = value
    if values["n_discard"] != int(values["n_discard"]):
        raise ValueError(
            f"measurement/{model}.yaml states n_discard = {values['n_discard']!r}, "
            "not a whole number"
        )
    values["n_discard"] = int(values["n_discard"])
    values["alpha"] = float(values["alpha"])
    values["gain"] = float(values["gain"])
    return values


def _sidecar(path: Path) -> dict[str, Any]:
    """The sidecar's lists and ``unsolved`` map, checked for kind only.

    Raises:
        ValueError: The file cannot be read, or a key is missing or of the wrong kind.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"the sidecar {path} cannot be read: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"the sidecar {path} is not a mapping")
    facts: dict[str, Any] = {}
    for key in _SIDECAR_LISTS:
        value = document.get(key)
        if not isinstance(value, list):
            raise ValueError(f"the sidecar {path} has no list {key!r}")
        if not all(
            isinstance(index, int | float) and not isinstance(index, bool) for index in value
        ):
            raise ValueError(f"the sidecar {path}: {key} must hold lattice indices")
        facts[key] = tuple(value)
    unsolved = document.get("unsolved")
    if not isinstance(unsolved, Mapping) or not all(
        isinstance(name, str) and isinstance(text, str) for name, text in unsolved.items()
    ):
        raise ValueError(f"the sidecar {path} has no map 'unsolved' of corrector to text")
    facts["unsolved"] = MappingProxyType(dict(unsolved))
    return facts


def _sidecar_mismatch(facts: Mapping[str, Any], layout: Any) -> str | None:
    """What of the sidecar does not describe the loaded matrix, or ``None``."""
    rows = len(layout.bpm_names_h) + len(layout.bpm_names_v)
    expected = {
        "bpm_index": rows,
        "hcorr_index": len(layout.hcorr_names),
        "vcorr_index": len(layout.vcorr_names),
    }
    for key, count in expected.items():
        if len(facts[key]) != count:
            return f"{key} has {len(facts[key])} entries, the matrix {count}"
    columns = {*layout.hcorr_names, *layout.vcorr_names}
    strays = sorted(name for name in facts["unsolved"] if name not in columns)
    if strays:
        return f"unsolved names {', '.join(strays)}, which are no column of the matrix"
    return None


def build_tool(render_root: Path, model: str | None) -> TrajectoryTool | ToolUnavailable:
    """Build ``model``'s trajectory tool from the render at ``render_root``.

    Args:
        render_root: The directory holding the rendered ``config.yml``; the
            pyAML view is under its ``data/pyaml/<model>``.
        model: The model the service commands, or ``None`` when the build has
            no trajectory model.

    Returns:
        The built tool, or why there is none. Never raises for a missing or
        bad input.
    """
    if model is None:
        return ToolUnavailable("the build has no trajectory model, so no response matrix")
    from osprey.facility.views.pyaml import TRAJECTORY_RM_FILE, TRAJECTORY_RM_INDEX_FILE

    directory = view_dir(render_root, model)
    matrix = directory / TRAJECTORY_RM_FILE
    index = directory / TRAJECTORY_RM_INDEX_FILE
    for path, what in ((matrix, "response matrix"), (index, "sidecar")):
        if not path.is_file():
            return _unavailable(
                f"model {model!r} has no trajectory {what}: {path} is missing; rebuild "
                "with the trajectory kind in its measurement file"
            )
    try:
        values = _tool_values(render_root, model)
        facts = _sidecar(index)
    except ValueError as exc:
        return _unavailable(f"model {model!r}: {exc}")

    from pyaml.common.exception import PyAMLConfigException, PyAMLException
    from pyaml.tuning_tools.trajectory import Trajectory

    try:
        tool = Trajectory(TOOL_NAME, str(matrix), **values)
    except (PyAMLException, PyAMLConfigException) as exc:
        message = getattr(exc, "message", None) or str(exc)
        return _unavailable(f"model {model!r}: the trajectory tool cannot be built: {message}")
    layout = tool.layout
    if layout is None:
        return _unavailable(f"model {model!r}: the trajectory tool loaded no response matrix")
    mismatch = _sidecar_mismatch(facts, layout)
    if mismatch is not None:
        return _unavailable(
            f"model {model!r}: the sidecar {index} does not describe {matrix}: {mismatch}"
        )
    logger.info(
        "trajectory service: tool built for model %s (%d BPMs, %d+%d correctors)",
        model,
        layout.n_bpm,
        layout.n_h,
        layout.n_v,
    )
    return TrajectoryTool(
        model=model,
        tool=tool,
        layout=layout,
        bpm_index=facts["bpm_index"],
        hcorr_index=facts["hcorr_index"],
        vcorr_index=facts["vcorr_index"],
        unsolved=facts["unsolved"],
        alpha=values["alpha"],
        n_discard=values["n_discard"],
        gain=values["gain"],
    )


def _unavailable(detail: str) -> ToolUnavailable:
    logger.warning("trajectory service: no correction tool: %s", detail)
    return ToolUnavailable(detail)


def build_service_tool(
    config: Mapping[str, Any], model: str | None
) -> TrajectoryTool | ToolUnavailable:
    """:func:`build_tool` for the render of the project ``config`` was loaded from.

    Args:
        config: The loaded project config.
        model: The model the service commands, or ``None``.

    Returns:
        As :func:`build_tool`.
    """
    from osprey_connectors.workspace import rendered_config_path, resolve_project_root

    return build_tool(rendered_config_path(resolve_project_root(config)).parent, model)


# -- the restricted model matrix ---------------------------------------------------


@dataclass(frozen=True)
class RmSummary:
    """The published summary of the response matrix a solve uses.

    Attributes:
        source: Where the matrix comes from; always :data:`RM_SOURCE`.
        rows: The matrix rows, x rows plus y rows.
        h_cols: The horizontal corrector columns.
        v_cols: The vertical corrector columns.
    """

    source: str
    rows: int
    h_cols: int
    v_cols: int

    def as_dict(self) -> dict[str, Any]:
        """The summary as the JSON object a solve embeds."""
        return {
            "source": self.source,
            "rows": self.rows,
            "h_cols": self.h_cols,
            "v_cols": self.v_cols,
        }


def rm_summary(tool: TrajectoryTool) -> RmSummary:
    """The summary of ``tool``'s matrix: the shared tool's, or a restricted one's."""
    layout = tool.layout
    return RmSummary(
        source=RM_SOURCE,
        rows=len(layout.bpm_names_h) + len(layout.bpm_names_v),
        h_cols=len(layout.hcorr_names),
        v_cols=len(layout.vcorr_names),
    )


def family_index(tool: TrajectoryTool) -> dict[str, dict[str, int]]:
    """The 1-based family index of every BPM and corrector of ``tool``'s matrix.

    Taken from the shared tool it is the build-time index the panel addresses:
    a BPM (named by its x row) by its position among the build's monitors, a
    corrector by its position in its plane. A restricted tool's names map into
    it with the excluded indices left as gaps.

    Returns:
        ``{"bpm": ..., "hcorr": ..., "vcorr": ...}``, each name to its index.
    """
    layout = tool.layout
    return {
        "bpm": {name: i for i, name in enumerate(layout.bpm_names_h, start=1)},
        "hcorr": {name: i for i, name in enumerate(layout.hcorr_names, start=1)},
        "vcorr": {name: i for i, name in enumerate(layout.vcorr_names, start=1)},
    }


def per_call_settings(
    tool: TrajectoryTool, *, alpha: float, n_discard: int, gain: float
) -> dict[str, Any]:
    """The ``alpha``, ``n_discard`` and ``gain`` a ``correct`` call on ``tool`` passes.

    The cutoff is a discard count held for the shared tool's spectrum; a
    restricted matrix may have a shorter one, so the count is clamped into
    ``[0, L]`` of ``tool``'s matrix rather than refused.
    """
    return {
        "alpha": float(alpha),
        "n_discard": max(0, min(int(n_discard), int(tool.layout.L))),
        "gain": float(gain),
    }


def keep_live_default(tool: TrajectoryTool) -> int:
    """The ``keep_live`` the tool's configured ``n_discard`` stands for: ``L − n_discard``.

    The measurement file states the cutoff as a discard count over the full
    matrix's spectrum; held as the count of positions kept, it survives an
    exclusion that shortens the spectrum.
    """
    return max(0, int(tool.layout.L) - int(tool.n_discard))


def n_discard_for(tool: TrajectoryTool, keep_live: int) -> int:
    """The discard count ``keep_live`` stands for over ``tool``'s matrix.

    ``L − min(keep_live, k_live)``, clamped into ``[0, L]``: the structural
    zeros are always discarded and the live values past ``keep_live`` with
    them.
    """
    layout = tool.layout
    length = int(layout.L)
    kept = min(max(int(keep_live), 0), int(layout.k_live))
    return max(0, min(length - kept, length))


def solve_sha(shared: TrajectoryTool, excluded: Iterable[str], settings: Mapping[str, Any]) -> str:
    """The identity of one solve: the shared matrix, the exclusions and the per-call settings.

    The sha256 hex digest of the canonical JSON object of the shared tool's
    matrix sha256, the sorted excluded names, and the ``alpha``, ``n_discard``
    and ``gain`` the solve passes (:func:`per_call_settings`), keys sorted, no
    whitespace, ASCII only. The published solve and every step solved on the
    same selection with the same settings carry the same identity.
    """
    canonical = json.dumps(
        {
            "alpha": float(settings["alpha"]),
            "excluded": sorted(set(excluded)),
            "gain": float(settings["gain"]),
            "matrix": _matrix_sha(shared),
            "n_discard": int(settings["n_discard"]),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


_RESTRICTED: OrderedDict[tuple[str, str, frozenset[str]], TrajectoryTool] = OrderedDict()
_MATRIX_SHA: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
_RESTRICTED_LOCK = threading.Lock()


def _matrix_sha(tool: TrajectoryTool) -> str:
    """The sha256 of ``tool``'s matrix and sidecar facts, computed once per tool."""
    try:
        cached = _MATRIX_SHA.get(tool.tool)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    data = tool.tool.response_matrix
    canonical = json.dumps(
        {
            "matrix": [[float(value) for value in row] for row in data.matrix],
            "observable_names": list(data.observable_names),
            "variable_names": list(data.variable_names),
            "observable_planes": list(data.observable_planes),
            "variable_planes": list(data.variable_planes),
            "bpm_index": [float(value) for value in tool.bpm_index],
            "hcorr_index": [float(value) for value in tool.hcorr_index],
            "vcorr_index": [float(value) for value in tool.vcorr_index],
            "unsolved": dict(tool.unsolved),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with contextlib.suppress(TypeError):
        _MATRIX_SHA[tool.tool] = sha
    return sha


def restricted_tool(shared: TrajectoryTool, excluded: Iterable[str]) -> TrajectoryTool:
    """The shared tool's model matrix restricted to the selection without ``excluded``.

    An excluded BPM, named by its x row as the shot names it, drops both its x
    row and its y row; an excluded corrector drops its column. The sidecar
    facts are sliced with the matrix, and ``unsolved`` keeps the correctors
    still selected. The restricted pyAML tool is built from a copy of the
    shared tool's matrix with neutral settings (``alpha`` 0, ``n_discard`` 0,
    ``gain`` 1): the operator's values are passed per call
    (:func:`per_call_settings`), and the configured values travel on the
    returned tool as on the shared one. The shared tool is never loaded nor
    changed. An empty exclusion is the shared tool itself; any other is cached
    by the exclusion set and the shared matrix's sha256.

    Args:
        shared: The service's built tool.
        excluded: The excluded BPM and corrector names.

    Returns:
        The tool a step on the selection solves with.

    Raises:
        StepRefused: ``selection_unknown`` naming each excluded name the build
            has no BPM or corrector of; ``step_failed`` when the restricted
            matrix cannot be solved with (no row, no column, or no live column).
    """
    names = frozenset(excluded)
    if not names:
        return shared
    layout = shared.layout
    bpms = tuple(layout.bpm_names_h)
    known = {*bpms, *layout.hcorr_names, *layout.vcorr_names}
    unknown = sorted(name for name in names if name not in known)
    if unknown:
        raise StepRefused(
            SELECTION_UNKNOWN,
            f"the build of model {shared.model!r} has no BPM or corrector "
            f"{', '.join(unknown)}; a selection excludes only the build's",
        )
    key = (shared.model, _matrix_sha(shared), names)
    with _RESTRICTED_LOCK:
        cached = _RESTRICTED.get(key)
        if cached is not None:
            _RESTRICTED.move_to_end(key)
            return cached
    restricted = _restrict(shared, names)
    with _RESTRICTED_LOCK:
        _RESTRICTED[key] = restricted
        _RESTRICTED.move_to_end(key)
        while len(_RESTRICTED) > _RESTRICTED_CACHE_SIZE:
            _RESTRICTED.popitem(last=False)
    return restricted


def _restrict(shared: TrajectoryTool, names: frozenset[str]) -> TrajectoryTool:
    """Build the restricted tool of :func:`restricted_tool`; ``names`` are all known."""
    layout = shared.layout
    n_bpm = len(layout.bpm_names_h)
    n_h = len(layout.hcorr_names)
    kept_bpms = [i for i, name in enumerate(layout.bpm_names_h) if name not in names]
    rows = [*kept_bpms, *(n_bpm + i for i in kept_bpms)]
    kept_h = [i for i, name in enumerate(layout.hcorr_names) if name not in names]
    kept_v = [i for i, name in enumerate(layout.vcorr_names) if name not in names]
    columns = [*kept_h, *(n_h + i for i in kept_v)]

    data = shared.tool.response_matrix
    matrix = [[data.matrix[r][c] for c in columns] for r in rows]
    fields = {
        "matrix": matrix,
        "observable_names": [data.observable_names[r] for r in rows],
        "variable_names": [data.variable_names[c] for c in columns],
        "rf_response": None,
        "observable_planes": [data.observable_planes[r] for r in rows],
        "variable_planes": [data.variable_planes[c] for c in columns],
    }

    from pyaml.common.exception import PyAMLConfigException, PyAMLException
    from pyaml.tuning_tools.trajectory import Trajectory

    try:
        tool = Trajectory(TOOL_NAME, fields, alpha=0.0, n_discard=0, gain=1.0)
    except (PyAMLException, PyAMLConfigException) as exc:
        message = getattr(exc, "message", None) or str(exc)
        raise StepRefused(
            STEP_FAILED,
            f"model {shared.model!r}: the matrix restricted to the selection cannot be "
            f"solved with: {message}",
        ) from exc
    columns_kept = {*fields["variable_names"]}
    return TrajectoryTool(
        model=shared.model,
        tool=tool,
        layout=tool.layout,
        bpm_index=tuple(shared.bpm_index[r] for r in rows),
        hcorr_index=tuple(shared.hcorr_index[i] for i in kept_h),
        vcorr_index=tuple(shared.vcorr_index[i] for i in kept_v),
        unsolved=MappingProxyType(
            {name: text for name, text in shared.unsolved.items() if name in columns_kept}
        ),
        alpha=shared.alpha,
        n_discard=shared.n_discard,
        gain=shared.gain,
    )


# -- the step ----------------------------------------------------------------------


def max_steps(validator: Any) -> dict[str, float]:
    """Every channel's ``max_step`` the limits ``validator`` states.

    Args:
        validator: The limits validator resolved for the service target, or
            ``None`` when limits are off.

    Returns:
        ``max_step`` by channel address, for the channels that state one.
    """
    if validator is None:
        return {}
    limits: Mapping[str, Any] = getattr(validator, "limits", None) or {}
    return {
        address: float(config.max_step)
        for address, config in limits.items()
        if getattr(config, "max_step", None) is not None
    }


def step_factor(step: Mapping[str, float], bounds: Mapping[str, float]) -> float:
    """The uniform scale of ``step`` that keeps every increment inside its ``max_step``.

    ``min(1, minᵢ max_stepᵢ·(1 − MAX_STEP_MARGIN)/|stepᵢ|)`` over the non-zero
    entries with a ``max_step``; 1 when there is none.

    Args:
        step: The tool's step by corrector name.
        bounds: ``max_step`` by corrector name.
    """
    factor = 1.0
    for name, value in step.items():
        bound = bounds.get(name)
        if bound is None or value == 0:
            continue
        factor = min(factor, bound * (1.0 - MAX_STEP_MARGIN) / abs(value))
    return max(factor, 0.0)


def step_digest(
    *,
    target: str,
    model: str,
    client_id: str,
    shot_id: str,
    reference: tuple[str, str] | None,
    solve_sha: str | None,
    deltas: Mapping[str, float],
) -> str:
    """The digest an apply carries to name exactly one step.

    The sha256 hex digest of the canonical JSON object of the target, the
    model, the holder's ``client_id``, the shot id, the reference's
    ``[name, sha256]`` (``null`` when solved to zero), the solve identity
    (:func:`solve_sha`, ``null`` for a step solved outside a published
    solve) and the deltas, keys sorted, no whitespace, ASCII only.

    Raises:
        ValueError: A delta is not finite.
    """
    canonical = json.dumps(
        {
            "client_id": client_id,
            "deltas": dict(deltas),
            "model": model,
            "reference": None if reference is None else list(reference),
            "shot_id": shot_id,
            "solve_sha": solve_sha,
            "target": target,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _plane(
    readings: Mapping[str, float | None], names: tuple[str, ...], plane: str, shot_id: str
) -> list[float]:
    """``readings`` in the tool's row order, ``None`` as NaN; refuses other names."""
    missing = [name for name in names if name not in readings]
    extra = sorted(set(readings) - set(names))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"lacks the tool's {plane} rows {', '.join(missing)}")
        if extra:
            parts.append(f"carries {plane} rows {', '.join(extra)} the tool does not have")
        raise StepRefused(SHOT_MISMATCH, f"shot {shot_id!r} {' and '.join(parts)}")
    return [math.nan if (value := readings[name]) is None else float(value) for name in names]


def _reference_rows(
    shot: ShotResult, reference: LoadedReference
) -> tuple[dict[str, float], dict[str, float]]:
    """The reference keyed by the shot's rows: each BPM's x row and y row.

    A shot lists its x rows and its y rows in its ``bpm_order``.
    """
    if not len(shot.bpm_order) == len(shot.x) == len(shot.y):
        raise StepRefused(
            SHOT_MISMATCH,
            f"shot {shot.shot_id!r} names {len(shot.bpm_order)} BPMs, {len(shot.x)} x rows "
            f"and {len(shot.y)} y rows; it needs one of each per BPM",
        )
    x = {
        row: reference.x.get(bpm, math.nan) for bpm, row in zip(shot.bpm_order, shot.x, strict=True)
    }
    y = {
        row: reference.y.get(bpm, math.nan) for bpm, row in zip(shot.bpm_order, shot.y, strict=True)
    }
    return x, y


def compute_step(
    tool: TrajectoryTool,
    shot: ShotResult,
    reference: LoadedReference | None,
    *,
    target: str,
    client_id: str,
    bounds: Mapping[str, float],
    solve_sha: str | None = None,
    settings: Mapping[str, Any] | None = None,
    ignore: Iterable[str] = (),
) -> Step:
    """The step for ``shot`` against ``reference``; blocking, run it off the loop.

    Args:
        tool: The tool the step solves with: the shared one, or one restricted
            to the selection.
        shot: The averaged shot, keyed by the tool's row names.
        reference: The current reference matched over the shot's BPMs, or
            ``None`` to solve to a zero trajectory.
        target: The service target.
        client_id: The holder's ``client_id`` the step is minted for.
        bounds: ``max_step`` by corrector name.
        solve_sha: The identity of the solve (:func:`solve_sha`) the digest
            binds; :func:`step_for_shot` always passes it.
        settings: The ``alpha``, ``n_discard`` and ``gain`` passed to
            ``correct`` (:func:`per_call_settings`); ``None`` passes none, so
            the tool solves with its configured values.
        ignore: Shot rows the tool does not solve with: the rows of the BPMs
            its matrix was restricted without.

    Returns:
        The step, ``δ = −factor·step`` by corrector name.

    Raises:
        StepRefused: ``shot_mismatch`` when the shot is of another model or its
            rows, less ``ignore``, are not the tool's; ``step_failed`` when the
            tool refuses the readings.
    """
    if shot.model != tool.model:
        raise StepRefused(
            SHOT_MISMATCH,
            f"shot {shot.shot_id!r} is of model {shot.model!r}; the tool solves {tool.model!r}",
        )
    layout = tool.layout
    skipped = frozenset(ignore)
    x = _plane(
        {name: value for name, value in shot.x.items() if name not in skipped},
        tuple(layout.bpm_names_h),
        "x",
        shot.shot_id,
    )
    y = _plane(
        {name: value for name, value in shot.y.items() if name not in skipped},
        tuple(layout.bpm_names_v),
        "y",
        shot.shot_id,
    )
    solve_to: list[list[float]] | None = None
    if reference is not None:
        ref_x, ref_y = _reference_rows(shot, reference)
        solve_to = [
            [ref_x[name] for name in layout.bpm_names_h],
            [ref_y[name] for name in layout.bpm_names_v],
        ]
    per_call = {} if settings is None else {key: settings[key] for key in _SETTING_KEYS}

    from pyaml.common.exception import PyAMLException

    try:
        result = tool.tool.correct(
            x,
            y,
            solve_to,
            bpm_index=list(tool.bpm_index),
            hcorr_index=list(tool.hcorr_index),
            vcorr_index=list(tool.vcorr_index),
            **per_call,
        )
    except PyAMLException as exc:
        message = getattr(exc, "message", None) or str(exc)
        raise StepRefused(
            STEP_FAILED, f"the tool cannot solve shot {shot.shot_id!r}: {message}"
        ) from exc

    raw = {name: float(value) for name, value in zip(result.names, result.step, strict=True)}
    factor = step_factor(raw, bounds)
    # Adding 0.0 turns a -0.0 into 0.0, so a zero increment has one digest form.
    deltas = {name: -factor * value + 0.0 for name, value in raw.items()}
    named = None if reference is None else (reference.load.name, reference.load.sha256)
    inverse = result.pseudo_inverse
    return Step(
        digest=step_digest(
            target=target,
            model=tool.model,
            client_id=client_id,
            shot_id=shot.shot_id,
            reference=named,
            solve_sha=solve_sha,
            deltas=deltas,
        ),
        deltas=deltas,
        factor=factor,
        reason=cast(StepReason, str(result.reason.value)),
        status=result.status,
        all_zero=bool(result.all_zero) or not any(deltas.values()),
        zeroed_h=list(result.zeroed_h),
        zeroed_v=list(result.zeroed_v),
        no_response_names=list(inverse.no_response_names),
        unsolved=list(tool.unsolved),
        shot_id=shot.shot_id,
        reference=None if reference is None else reference.load.name,
    )


def _solve_inputs(
    shared: TrajectoryTool, shot: ShotResult, config: SolveConfig | None
) -> tuple[TrajectoryTool, dict[str, Any] | None, str | None, frozenset[str]]:
    """The tool, settings, solve identity and ignored rows a step on ``shot`` solves with.

    With no SolveConfig and no excluded BPM the step is the shared tool's, at
    its configured values, outside any published solve. Blocking: a
    restriction not cached builds a pyAML tool.
    """
    if config is None and not shot.excluded_bpms:
        return shared, None, None, frozenset()
    if config is None:
        excluded_cms: list[str] = []
        alpha, gain, keep_live = shared.alpha, shared.gain, keep_live_default(shared)
    else:
        excluded_cms = list(config.excluded_cms)
        alpha, gain, keep_live = config.alpha, config.gain, config.keep_live
    excluded = frozenset({*shot.excluded_bpms, *excluded_cms})
    tool = restricted_tool(shared, excluded)
    settings = per_call_settings(
        tool, alpha=alpha, n_discard=n_discard_for(tool, keep_live), gain=gain
    )
    kept = {*tool.layout.bpm_names_h, *tool.layout.bpm_names_v}
    ignore = frozenset(
        name
        for name in (*shared.layout.bpm_names_h, *shared.layout.bpm_names_v)
        if name not in kept
    )
    return tool, settings, solve_sha(shared, excluded, settings), ignore


def _solved(
    shared: TrajectoryTool,
    shot: ShotResult,
    reference: LoadedReference | None,
    config: SolveConfig | None,
    *,
    target: str,
    client_id: str,
    bounds: Mapping[str, float],
) -> Step:
    tool, settings, sha, ignore = _solve_inputs(shared, shot, config)
    return compute_step(
        tool,
        shot,
        reference,
        target=target,
        client_id=client_id,
        bounds=bounds,
        solve_sha=sha,
        settings=settings,
        ignore=ignore,
    )


async def step_for_shot(
    startup: TrajectoryStartup,
    store: TrajectoryStore,
    shot: ShotResult,
    *,
    client_id: str,
    config: SolveConfig | None = None,
) -> Step:
    """The step for ``shot`` against the model's current reference, in a turn.

    Reads the current reference from ``store`` matched over the shot's BPMs,
    takes every ``max_step`` from the limits validator the turn resolved for
    the service target, restricts the model matrix to the shot's excluded BPMs
    and ``config``'s excluded correctors, and runs :func:`compute_step` with
    ``config``'s ``alpha``, ``gain`` and the discard count its ``keep_live``
    stands for, in a thread.

    Args:
        startup: The started service.
        store: Its references.
        shot: The shot to solve.
        client_id: The holder's ``client_id`` the step is minted for.
        config: The SolveConfig in force; ``None`` solves with the tool's
            configured values and no corrector excluded.

    Raises:
        StepRefused: ``tool_unavailable`` with no tool; ``reference_unusable``
            when the current reference is gone or names none of the shot's
            BPMs; ``selection_unknown`` or ``step_failed`` from
            :func:`restricted_tool`; as :func:`compute_step`.
    """
    from osprey.services.trajectory.store import NoMatchingBpms, ReferenceNotFound

    shared = require_tool(startup.tool)
    name = await store.current(shared.model)
    reference = None
    if name is not None:
        try:
            reference = await store.load_reference(shared.model, name, shot.bpm_order)
        except (ReferenceNotFound, NoMatchingBpms) as exc:
            raise StepRefused(REFERENCE_UNUSABLE, f"the current reference: {exc}") from exc
    bounds = max_steps(getattr(startup.connector, "_limits_validator", None))
    return await asyncio.to_thread(
        _solved,
        shared,
        shot,
        reference,
        config,
        target=startup.target,
        client_id=client_id,
        bounds=bounds,
    )


class StepState:
    """The one pending step of a started service, and the shot it came from.

    In memory only. An apply matches the step it writes by identity, so a
    replaced or voided step is never written, and a step is consumed once.
    """

    def __init__(self) -> None:
        self._step: Step | None = None
        self._shot: ShotResult | None = None
        self._consumed = False

    @property
    def step(self) -> Step | None:
        """The pending step, ``None`` when there is none."""
        return self._step

    @property
    def shot(self) -> ShotResult | None:
        """The shot the pending step was computed from, or the kept base of a void one.

        ``None`` when void without a kept shot.
        """
        return self._shot

    @property
    def consumed(self) -> bool:
        """Whether an apply has taken the pending step."""
        return self._consumed

    def replace(self, step: Step, shot: ShotResult) -> None:
        """Make ``step``, computed from ``shot``, the pending step."""
        self._step = step
        self._shot = shot
        self._consumed = False

    def void(self, *, keep_shot: bool = False) -> None:
        """Drop the pending step, and its shot unless ``keep_shot``.

        A kept shot is the base the next recompute solves from; with no step
        pending there is nothing to apply.
        """
        self._step = None
        if not keep_shot:
            self._shot = None
        self._consumed = False

    def consume(self, step: Step) -> bool:
        """Mark ``step`` taken when it is still the pending, unconsumed step.

        Returns:
            Whether ``step`` was the pending step and is now consumed.
        """
        if self._step is not step or self._consumed:
            return False
        self._consumed = True
        return True
