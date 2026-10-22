"""The SolveConfig: one piece of service state per target and model, with one writer.

A SolveConfig is ``{alpha, keep_live, gain, excluded_bpms, excluded_cms,
loss_cutoff}``. Its file defaults come from the build: ``alpha`` and ``gain``
are the tool's configured values, ``keep_live`` is ``L − n_discard`` of the
measurement file's discard count over the full matrix
(:func:`~osprey.services.trajectory.step.keep_live_default`), the loss cutoff
is the service's configured one, and nothing is excluded. The cutoff is held as
``keep_live`` only, so it survives an exclusion that shortens the spectrum; the
discard count it stands for over the current exclusions is derived
(:func:`~osprey.services.trajectory.step.n_discard_for`) and published beside it.

:meth:`SolveConfigStore.update` is the one writer. It is a plain coroutine,
not a holder call: every caller (the solve route, the setup) runs it inside a
holder action it already holds. Each caller sends only its own keys; an empty
update recomputes the solve (Calculate RM). An update validates every key it
carries — the names are the build's (a BPM by its x row, a corrector by its
column), at least one BPM and one live corrector per plane stay selected, the
numbers are in range — then recomputes the published :class:`~.state.Solve`
on the model matrix restricted to the selection
(:func:`~osprey.services.trajectory.step.restricted_tool`), persists the
operator overrides, and swaps both in, all in the same turn. A refused update
changes nothing.

Only overrides are persisted: a key whose value is its file default is not
one. Each override is written with the file default it overrode, under
``<target>/solve/<model>.json`` in the service's state directory. At load an
override is dropped with a notice when its file default changed, when it names
a BPM or corrector the build no longer has, or when it no longer validates;
the notices are published on the snapshot.

A loss-cutoff change reaches :attr:`SolveConfigStore.on_loss_cutoff`, the
extraction's hook, after it is persisted; the solve is not recomputed for it.
The hook :meth:`SolveConfigStore.derive_on_cutoff` installs re-extracts the
latest shot at the new cutoff as a derived shot
(:func:`~osprey.services.trajectory.acquisition.derive_shot`) and publishes it,
outside a continuous run only: during one the next iteration reads the new
cutoff. A derivation the hook refuses leaves the edit persisted; the refusal
answers for the derivation only.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.state import RmSummary, Solve, SolveConfig, SolveConfigPartial
from osprey.services.trajectory.step import (
    SELECTION_UNKNOWN,
    TrajectoryTool,
    keep_live_default,
    n_discard_for,
    per_call_settings,
    restricted_tool,
    rm_summary,
    solve_sha,
)

if TYPE_CHECKING:
    from osprey.services.trajectory.state import ShotResult
    from osprey.services.trajectory.store import TrajectoryStore

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_KEYS",
    "EXCLUSION_KEYS",
    "INVALID_SOLVE_CONFIG",
    "SELECTION_TOO_SMALL",
    "SOLVE_CONFIG_UNSAVED",
    "SOLVE_DIR",
    "SOLVE_FAILED",
    "TUNING_KEYS",
    "SolveConfigRefused",
    "SolveConfigStore",
    "SolveUpdate",
]

#: The keys a recompute of the step follows: a change re-solves the step's shot.
TUNING_KEYS = ("alpha", "keep_live", "gain")

#: The selection keys: a change voids the step.
EXCLUSION_KEYS = ("excluded_bpms", "excluded_cms")

#: Every key of a SolveConfig an update may carry.
CONFIG_KEYS = (*TUNING_KEYS, *EXCLUSION_KEYS, "loss_cutoff")

#: The directory, under the target's state directory, of the persisted overrides.
SOLVE_DIR = "solve"

#: The refusal code of a value out of its range or of the wrong kind.
INVALID_SOLVE_CONFIG = "invalid_solve_config"

#: The refusal code of a selection left without a BPM or a live corrector in a plane.
SELECTION_TOO_SMALL = "selection_too_small"

#: The refusal code of a selection and settings the tool cannot solve with.
SOLVE_FAILED = "solve_failed"

#: The refusal code of overrides that could not be written.
SOLVE_CONFIG_UNSAVED = "solve_config_unsaved"


class SolveConfigRefused(Refused):
    """A SolveConfig update the writer refuses; nothing changed."""

    statuses = {
        INVALID_SOLVE_CONFIG: 422,
        SELECTION_UNKNOWN: 409,
        SELECTION_TOO_SMALL: 409,
        SOLVE_FAILED: 422,
        SOLVE_CONFIG_UNSAVED: 503,
    }


@dataclass(frozen=True)
class SolveUpdate:
    """What one update did.

    Attributes:
        solve: The published solve after the update.
        changed: The keys whose value changed.
        recalculated: Whether the update carried no key (Calculate RM).
    """

    solve: Solve
    changed: frozenset[str]
    recalculated: bool


#: The hook a loss-cutoff change reaches, with the new cutoff.
LossCutoffHook = Callable[[float], Awaitable[None] | None]


def _number(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SolveConfigRefused(INVALID_SOLVE_CONFIG, f"{key} is a finite number, not {value!r}")
    return float(value)


def _names(key: str, value: Any) -> list[str]:
    if not isinstance(value, list | tuple) or not all(isinstance(name, str) for name in value):
        raise SolveConfigRefused(INVALID_SOLVE_CONFIG, f"{key} is a list of names, not {value!r}")
    return sorted(set(value))


class SolveConfigStore:
    """The SolveConfig of one target and model, and the solve published for it.

    Build it with :meth:`load`.

    Attributes:
        shared: The service's built tool; never changed.
        model: The model the config is for.
        defaults: The file default of every key.
        notices: What the load dropped, for the operator.
        on_loss_cutoff: The extraction's hook a loss-cutoff change reaches,
            ``None`` when nothing listens.
        cause: Who made the latest update (``load`` before any).
    """

    def __init__(
        self,
        shared: TrajectoryTool,
        *,
        loss_cutoff: float,
        directory: Path | None = None,
    ) -> None:
        self.shared = shared
        self.model = shared.model
        self.defaults: dict[str, Any] = {
            "alpha": float(shared.alpha),
            "keep_live": keep_live_default(shared),
            "gain": float(shared.gain),
            "excluded_bpms": [],
            "excluded_cms": [],
            "loss_cutoff": float(loss_cutoff),
        }
        self.notices: list[str] = []
        self.on_loss_cutoff: LossCutoffHook | None = None
        self.cause = "load"
        self._path = (
            None if directory is None else Path(directory) / SOLVE_DIR / f"{self.model}.json"
        )
        self._overrides: dict[str, Any] = {}
        self._solve: Solve | None = None

    # -- reading ---------------------------------------------------------------

    @property
    def solve(self) -> Solve | None:
        """The published solve, ``None`` when not even the file defaults solve."""
        return self._solve

    @property
    def values(self) -> dict[str, Any]:
        """Every key's value in force: its override, else its file default."""
        return {**self.defaults, **self._overrides}

    @property
    def overrides(self) -> dict[str, Any]:
        """The operator overrides, by key."""
        return dict(self._overrides)

    @property
    def config(self) -> SolveConfig:
        """The SolveConfig in force, with the discard count over the current exclusions."""
        if self._solve is not None:
            return self._solve.config
        values = self.values
        return SolveConfig(
            **values,
            n_discard=n_discard_for(self.shared, values["keep_live"]),
            overridden=sorted(self._overrides),
        )

    # -- the extraction's hook ---------------------------------------------------

    def derive_on_cutoff(
        self,
        sessions: Any,
        store: TrajectoryStore,
        publish: Callable[[ShotResult], Any],
    ) -> None:
        """Make a loss-cutoff change re-extract the latest shot as a derived shot.

        Outside a continuous run the hook derives the latest shot a holder's
        action made at the new cutoff and hands the derived shot to
        *publish*; with no latest shot, or one that kept no buffer, it does
        nothing. During a run it does nothing: the run's next iteration is
        built with the persisted cutoff.

        Args:
            sessions: The service's sessions: ``continuous_running``,
                ``latest_shot_id`` and ``layer.startup.settings``.
            store: Where the shots and their raw buffers are.
            publish: Called with each derived shot, once stored.

        The hook raises as
        :func:`~osprey.services.trajectory.acquisition.derive_shot`:
        ``raw_evicted`` or ``exclusions_changed``.
        """

        async def hook(cutoff: float) -> None:
            if sessions.continuous_running:
                return
            parent = sessions.latest_shot_id
            if parent is None:
                return
            from osprey.services.trajectory.acquisition import derive_shot

            try:
                derived = await derive_shot(
                    store,
                    self.model,
                    parent,
                    loss_cutoff=cutoff,
                    excluded_bpms=self.values["excluded_bpms"],
                    settings=sessions.layer.startup.settings,
                )
            except LookupError as exc:
                logger.info("trajectory service: no shot to re-extract (%s)", exc)
                return
            outcome = publish(derived)
            if inspect.isawaitable(outcome):
                await outcome

        self.on_loss_cutoff = hook

    # -- validation ------------------------------------------------------------

    def _check(self, key: str, value: Any) -> Any:
        """``value`` checked and normalized for ``key``.

        Raises:
            SolveConfigRefused: ``invalid_solve_config`` or ``selection_unknown``.
        """
        layout = self.shared.layout
        if key == "alpha":
            alpha = _number(key, value)
            if alpha < 0:
                raise SolveConfigRefused(INVALID_SOLVE_CONFIG, f"alpha is 0 or more, not {value!r}")
            return alpha
        if key == "gain":
            gain = _number(key, value)
            if gain <= 0:
                raise SolveConfigRefused(
                    INVALID_SOLVE_CONFIG, f"gain is greater than 0, not {value!r}"
                )
            return gain
        if key == "loss_cutoff":
            cutoff = _number(key, value)
            if not 0.0 <= cutoff <= 1.0:
                raise SolveConfigRefused(
                    INVALID_SOLVE_CONFIG,
                    f"loss_cutoff is a fraction of the injected-charge scale in [0, 1], not "
                    f"{value!r}",
                )
            return cutoff
        if key == "keep_live":
            length = int(layout.L)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= length:
                raise SolveConfigRefused(
                    INVALID_SOLVE_CONFIG,
                    f"keep_live is a whole number in [0, {length}], not {value!r}",
                )
            return int(value)
        names = _names(key, value)
        if key == "excluded_bpms":
            known, what = set(layout.bpm_names_h), "BPM"
        else:
            known, what = {*layout.hcorr_names, *layout.vcorr_names}, "corrector"
        unknown = [name for name in names if name not in known]
        if unknown:
            raise SolveConfigRefused(
                SELECTION_UNKNOWN,
                f"the build of model {self.model!r} has no {what} {', '.join(unknown)}; "
                f"{key} names only the build's",
            )
        return names

    def _check_minimums(self, values: Mapping[str, Any]) -> None:
        """Refuse a selection without a BPM, or without a live corrector in a plane."""
        layout = self.shared.layout
        bpms = set(values["excluded_bpms"])
        cms = set(values["excluded_cms"])
        dead = set(layout.no_response_names) | set(self.shared.unsolved)
        missing = []
        if not any(name not in bpms for name in layout.bpm_names_h):
            missing.append("a BPM")
        for plane, names in (("horizontal", layout.hcorr_names), ("vertical", layout.vcorr_names)):
            if not any(name not in cms and name not in dead for name in names):
                missing.append(f"a live {plane} corrector")
        if missing:
            raise SolveConfigRefused(
                SELECTION_TOO_SMALL,
                f"the selection keeps no {' and no '.join(missing)}; a solve needs at least one "
                "BPM and one live corrector per plane",
            )

    # -- solving ---------------------------------------------------------------

    def _publish(self, values: Mapping[str, Any], overridden: Iterable[str]) -> Solve:
        """The solve of ``values``; blocking, run it off the loop.

        Raises:
            SolveConfigRefused: ``selection_too_small`` or ``solve_failed``.
            StepRefused: As :func:`~osprey.services.trajectory.step.restricted_tool`.
        """
        self._check_minimums(values)
        excluded = frozenset({*values["excluded_bpms"], *values["excluded_cms"]})
        tool = restricted_tool(self.shared, excluded)
        n_discard = n_discard_for(tool, values["keep_live"])
        settings = per_call_settings(
            tool, alpha=values["alpha"], n_discard=n_discard, gain=values["gain"]
        )

        from pyaml.common.exception import PyAMLException

        try:
            inverse = tool.tool.pseudo_inverse(**settings)
        except PyAMLException as exc:
            message = getattr(exc, "message", None) or str(exc)
            raise SolveConfigRefused(
                SOLVE_FAILED,
                f"model {self.model!r}: the selection cannot be solved with alpha "
                f"{settings['alpha']!r}, keep_live {values['keep_live']} and gain "
                f"{settings['gain']!r}: {message}",
            ) from exc
        summary = rm_summary(tool)
        layout = tool.layout
        return Solve(
            sha=solve_sha(self.shared, excluded, settings),
            singular_values=[float(value) for value in inverse.singular_values],
            inverse_weights=[float(value) for value in inverse.inverse_weights],
            kind=[str(kind.value) for kind in inverse.spectrum_kind],  # type: ignore[misc]
            L=int(layout.L),
            live_rank=int(layout.k_live),
            rm=RmSummary(
                source=summary.source,  # type: ignore[arg-type]
                rows=summary.rows,
                h_cols=summary.h_cols,
                v_cols=summary.v_cols,
                computed_at=time.time(),
            ),
            config=SolveConfig(
                alpha=settings["alpha"],
                keep_live=values["keep_live"],
                n_discard=settings["n_discard"],
                gain=settings["gain"],
                excluded_bpms=list(values["excluded_bpms"]),
                excluded_cms=list(values["excluded_cms"]),
                loss_cutoff=values["loss_cutoff"],
                overridden=sorted(overridden),
            ),
        )

    # -- persistence -----------------------------------------------------------

    def _document(self, overrides: Mapping[str, Any]) -> bytes:
        return json.dumps(
            {
                "model": self.model,
                "overrides": {
                    key: {"value": value, "default": self.defaults[key]}
                    for key, value in sorted(overrides.items())
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode()

    def _write(self, overrides: Mapping[str, Any]) -> None:
        if self._path is None:
            return
        from osprey.services.trajectory.store import FILE_MODE
        from osprey_connectors.safety_file import atomic_replace

        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace(self._path, self._document(overrides), mode=FILE_MODE, sync="required")

    async def _persist(self, overrides: Mapping[str, Any]) -> None:
        try:
            await asyncio.to_thread(self._write, dict(overrides))
        except OSError as exc:
            raise SolveConfigRefused(
                SOLVE_CONFIG_UNSAVED, f"the SolveConfig overrides cannot be written: {exc}"
            ) from exc

    def _read(self) -> dict[str, Any]:
        """The persisted overrides, each ``{"value", "default"}``; ``{}`` with none."""
        if self._path is None or not self._path.is_file():
            return {}
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.notices.append(
                f"The SolveConfig overrides of model {self.model!r} could not be read ({exc}); "
                "the file defaults are in force."
            )
            return {}
        overrides = document.get("overrides") if isinstance(document, Mapping) else None
        return dict(overrides) if isinstance(overrides, Mapping) else {}

    def _reconciled(self, persisted: Mapping[str, Any]) -> dict[str, Any]:
        """The persisted overrides still valid against the build; a notice for each dropped."""
        kept: dict[str, Any] = {}
        for key, entry in persisted.items():
            if key not in CONFIG_KEYS or not isinstance(entry, Mapping):
                self.notices.append(f"The SolveConfig override {key!r} is not a key; dropped.")
                continue
            value, default = entry.get("value"), entry.get("default")
            if default != self.defaults[key]:
                self.notices.append(
                    f"The file default of {key} changed from {default!r} to "
                    f"{self.defaults[key]!r}; the override {key} = {value!r} was dropped."
                )
                continue
            try:
                checked = self._check(key, value)
            except SolveConfigRefused as exc:
                reason = (
                    "names what left the build"
                    if exc.code == SELECTION_UNKNOWN
                    else "no longer validates"
                )
                self.notices.append(
                    f"The override {key} = {value!r} {reason} ({exc.detail}); dropped."
                )
                continue
            if checked != self.defaults[key]:
                kept[key] = checked
        return kept

    @classmethod
    async def load(
        cls,
        shared: TrajectoryTool,
        *,
        loss_cutoff: float,
        directory: Path | None = None,
    ) -> SolveConfigStore:
        """The store of ``shared``'s model, its overrides read and its solve published.

        Never raises for persisted overrides: an override that no longer holds
        is dropped with a notice, and when the overrides together do not solve
        they are all dropped. When not even the file defaults solve, the store
        publishes no solve and says why.

        Args:
            shared: The service's built tool.
            loss_cutoff: The configured loss cutoff, the key's file default.
            directory: The target's state directory, ``None`` to keep the
                overrides in memory only.
        """
        store = cls(shared, loss_cutoff=loss_cutoff, directory=directory)
        persisted = await asyncio.to_thread(store._read)
        overrides = store._reconciled(persisted)
        values = {**store.defaults, **overrides}
        try:
            store._solve = await asyncio.to_thread(store._publish, values, overrides)
        except Refused as exc:
            if overrides:
                store.notices.append(
                    f"The SolveConfig overrides {', '.join(sorted(overrides))} do not solve "
                    f"together ({exc.detail}); dropped."
                )
            overrides = {}
            try:
                store._solve = await asyncio.to_thread(store._publish, store.defaults, ())
            except Refused as default_exc:
                store.notices.append(
                    f"The file defaults of model {shared.model!r} do not solve "
                    f"({default_exc.detail}); no solve is published."
                )
        store._overrides = overrides
        for notice in store.notices:
            logger.warning("trajectory service: %s", notice)
        if persisted and store.notices:
            try:
                await store._persist(overrides)
            except SolveConfigRefused as exc:
                logger.warning("trajectory service: %s", exc.detail)
        return store

    # -- the writer ------------------------------------------------------------

    async def update(
        self, partial: SolveConfigPartial | Mapping[str, Any], cause: str
    ) -> SolveUpdate:
        """Apply ``partial`` and publish the solve, in one turn; the one writer.

        Run it inside a holder action the caller already holds. Absent and
        ``None`` keys are left as they are; an empty partial recomputes the
        solve.

        Args:
            partial: The keys to set.
            cause: Who updates: ``operator`` for the solve route, ``setup``
                for the setup.

        Returns:
            The published solve and the keys that changed.

        Raises:
            SolveConfigRefused: A key does not validate, the selection is too
                small or does not solve, or the overrides cannot be written;
                nothing changed.
            StepRefused: As :func:`~osprey.services.trajectory.step.restricted_tool`.
        """
        if not isinstance(partial, SolveConfigPartial):
            try:
                partial = SolveConfigPartial.model_validate(dict(partial))
            except ValidationError as exc:
                raise SolveConfigRefused(
                    INVALID_SOLVE_CONFIG, f"the SolveConfig update does not validate: {exc}"
                ) from exc
        fields = {
            key: value
            for key, value in partial.model_dump(exclude_unset=True).items()
            if value is not None
        }
        current = self.values
        checked = {key: self._check(key, value) for key, value in fields.items()}
        changed = frozenset(key for key, value in checked.items() if value != current[key])
        values = {**current, **checked}
        overrides = {key: value for key, value in values.items() if value != self.defaults[key]}
        solve = self._solve
        if solve is None or not fields or changed - {"loss_cutoff"}:
            solve = await asyncio.to_thread(self._publish, values, overrides)
        elif changed:
            solve = solve.model_copy(
                update={
                    "config": solve.config.model_copy(
                        update={
                            "loss_cutoff": values["loss_cutoff"],
                            "overridden": sorted(overrides),
                        }
                    )
                }
            )
        if overrides != self._overrides:
            await self._persist(overrides)
        self._overrides = overrides
        self._solve = solve
        self.cause = cause
        logger.info(
            "trajectory service: SolveConfig of model %s updated by %s (%s)",
            self.model,
            cause,
            ", ".join(sorted(changed)) or "recomputed",
        )
        if "loss_cutoff" in changed and self.on_loss_cutoff is not None:
            outcome = self.on_loss_cutoff(values["loss_cutoff"])
            if inspect.isawaitable(outcome):
                await outcome
        return SolveUpdate(solve=solve, changed=changed, recalculated=not fields)
