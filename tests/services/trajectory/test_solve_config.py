"""The SolveConfig: its one writer, its persisted overrides, its solve and the step it drives.

Every test builds the shared tool from a render carrying the vendored golden
case, the way the service does; its measurement file states a discard count of
4, so the file default ``keep_live`` is ``L − 4``. The sessions run on a fake
action layer that runs each action at once with its precondition.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from osprey.services.trajectory import step
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.sessions import (
    CONTINUOUS_RUNNING,
    NOT_SESSION_HOLDER,
    SessionRefused,
    Sessions,
)
from osprey.services.trajectory.solve_config import (
    INVALID_SOLVE_CONFIG,
    SELECTION_TOO_SMALL,
    SOLVE_DIR,
    SolveConfigRefused,
    SolveConfigStore,
)
from osprey.services.trajectory.state import ShotResult, SolveConfigPartial
from osprey.services.trajectory.step import (
    SELECTION_UNKNOWN,
    TrajectoryTool,
    build_tool,
    n_discard_for,
    per_call_settings,
    restricted_tool,
    solve_sha,
)

GOLDENS = Path(__file__).parent / "data" / "trajectory_goldens.npz"
CASE = "example_sr_tikhonov"
MODEL = "SR"
TARGET = "va"
N_DISCARD_FILE = 4
LOSS_CUTOFF = 0.5


@pytest.fixture(scope="module")
def goldens() -> Any:
    with np.load(GOLDENS, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _case(goldens: Any, field: str) -> Any:
    return goldens[f"{CASE}/{field}"]


def _build(root: Path, goldens: Any, *, n_discard: int = N_DISCARD_FILE) -> TrajectoryTool:
    view = root / "data" / "pyaml" / MODEL
    view.mkdir(parents=True, exist_ok=True)
    rm = {
        "type": "pyaml.tuning_tools.orbit_response_matrix_data",
        "matrix": _case(goldens, "matrix").tolist(),
        "observable_names": _case(goldens, "observable_names").tolist(),
        "variable_names": _case(goldens, "variable_names").tolist(),
        "observable_planes": _case(goldens, "observable_planes").tolist(),
        "variable_planes": _case(goldens, "variable_planes").tolist(),
    }
    rows = _case(goldens, "bpm_index").tolist()
    index = {
        "bpm_index": [*rows, *rows],
        "hcorr_index": _case(goldens, "hcorr_index").tolist(),
        "vcorr_index": _case(goldens, "vcorr_index").tolist(),
        "unsolved": {},
    }
    (view / "trajectory_rm.json").write_text(json.dumps(rm), encoding="utf-8")
    (view / "trajectory_rm.index.json").write_text(json.dumps(index), encoding="utf-8")
    measurement = {
        "alpha": float(_case(goldens, "alpha")),
        "n_discard": n_discard,
        "gain": float(_case(goldens, "gain")),
    }
    facility = {"models": [{"name": MODEL, "measurement": measurement}]}
    (root / "facility.json").write_text(json.dumps(facility), encoding="utf-8")
    tool = build_tool(root, MODEL)
    assert isinstance(tool, TrajectoryTool), tool
    return tool


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    step._RESTRICTED.clear()


@pytest.fixture
def shared(tmp_path: Path, goldens: Any) -> TrajectoryTool:
    return _build(tmp_path / "build", goldens)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    return tmp_path / "state" / TARGET


async def _load(shared: TrajectoryTool, state: Path | None) -> SolveConfigStore:
    return await SolveConfigStore.load(shared, loss_cutoff=LOSS_CUTOFF, directory=state)


def _file(state: Path) -> dict[str, Any]:
    return json.loads((state / SOLVE_DIR / f"{MODEL}.json").read_text(encoding="utf-8"))


# -- the file defaults ---------------------------------------------------------


async def test_the_file_defaults_are_published_with_keep_live_for_the_discard_count(
    shared: TrajectoryTool, state: Path
) -> None:
    store = await _load(shared, state)
    solve = store.solve
    assert solve is not None
    length = int(shared.layout.L)

    assert solve.config.keep_live == length - N_DISCARD_FILE
    assert solve.config.n_discard == N_DISCARD_FILE
    assert solve.config.alpha == shared.alpha and solve.config.gain == shared.gain
    assert solve.config.excluded_bpms == [] and solve.config.excluded_cms == []
    assert solve.config.loss_cutoff == LOSS_CUTOFF
    assert solve.config.overridden == []
    assert solve.L == length and solve.live_rank == int(shared.layout.k_live)
    assert len(solve.singular_values) == len(solve.inverse_weights) == len(solve.kind) == length
    assert set(solve.kind[-N_DISCARD_FILE:]) <= {"DISCARDED", "STRUCTURAL"}
    assert "DISCARDED" not in solve.kind[:-N_DISCARD_FILE]
    assert (solve.rm.source, solve.rm.rows) == ("model", 2 * len(shared.layout.bpm_names_h))
    assert solve.rm.computed_at is not None
    assert store.notices == []
    assert not (state / SOLVE_DIR).exists()


async def test_the_published_identity_is_the_one_a_step_on_the_selection_binds(
    shared: TrajectoryTool,
) -> None:
    store = await _load(shared, None)
    settings = per_call_settings(
        shared, alpha=shared.alpha, n_discard=N_DISCARD_FILE, gain=shared.gain
    )

    assert store.solve is not None
    assert store.solve.sha == solve_sha(shared, (), settings)


async def test_the_spectrum_is_the_restricted_matrixs_pseudo_inverse(
    shared: TrajectoryTool,
) -> None:
    store = await _load(shared, None)
    hcm = shared.layout.hcorr_names[4]

    update = await store.update({"excluded_cms": [hcm], "alpha": 0.003}, "operator")

    restricted = restricted_tool(shared, {hcm})
    n_discard = n_discard_for(restricted, update.solve.config.keep_live)
    inverse = restricted.tool.pseudo_inverse(alpha=0.003, n_discard=n_discard, gain=shared.gain)
    assert update.solve.singular_values == pytest.approx(list(inverse.singular_values))
    assert update.solve.inverse_weights == pytest.approx(list(inverse.inverse_weights))
    assert update.solve.rm.h_cols == len(shared.layout.hcorr_names) - 1
    assert update.solve.L == int(restricted.layout.L)


# -- the writer ----------------------------------------------------------------


async def test_an_update_persists_only_the_overrides_with_the_default_they_overrode(
    shared: TrajectoryTool, state: Path
) -> None:
    store = await _load(shared, state)

    update = await store.update({"alpha": 0.002, "gain": shared.gain}, "operator")

    assert update.changed == {"alpha"}
    assert update.solve.config.overridden == ["alpha"]
    assert _file(state)["overrides"] == {
        "alpha": {"value": 0.002, "default": shared.alpha},
    }
    await store.update({"alpha": shared.alpha}, "operator")
    assert _file(state)["overrides"] == {}
    assert store.solve is not None and store.solve.config.overridden == []


async def test_each_caller_sends_only_its_own_keys(shared: TrajectoryTool) -> None:
    store = await _load(shared, None)
    bpm = shared.layout.bpm_names_h[2]
    await store.update({"alpha": 0.002}, "operator")

    update = await store.update(SolveConfigPartial(excluded_bpms=[bpm]), "setup")

    assert update.changed == {"excluded_bpms"}
    assert update.solve.config.alpha == 0.002
    assert update.solve.config.excluded_bpms == [bpm]
    assert store.cause == "setup"


async def test_an_empty_update_is_calculate_rm(shared: TrajectoryTool) -> None:
    store = await _load(shared, None)
    before = store.solve
    assert before is not None

    update = await store.update({}, "operator")

    assert update.recalculated and update.changed == frozenset()
    assert update.solve.sha == before.sha
    assert update.solve is not before


async def test_keep_live_survives_an_exclusion_that_shortens_the_spectrum(
    shared: TrajectoryTool,
) -> None:
    store = await _load(shared, None)
    await store.update({"keep_live": 130}, "operator")
    cms = [shared.layout.hcorr_names[0], shared.layout.vcorr_names[0]]

    update = await store.update({"excluded_cms": cms}, "operator")

    restricted = restricted_tool(shared, set(cms))
    assert update.solve.config.keep_live == 130
    assert update.solve.config.n_discard == int(restricted.layout.L) - 130
    assert update.solve.L == int(restricted.layout.L) == int(shared.layout.L) - 2


async def test_exclusions_are_held_sorted_and_once(shared: TrajectoryTool) -> None:
    store = await _load(shared, None)
    a, b = shared.layout.hcorr_names[3], shared.layout.hcorr_names[1]

    update = await store.update({"excluded_cms": [a, b, a]}, "operator")

    assert update.solve.config.excluded_cms == sorted({a, b})
    again = await store.update({"excluded_cms": [b, a]}, "operator")
    assert again.changed == frozenset()


@pytest.mark.parametrize(
    "partial",
    [
        {"alpha": -0.1},
        {"alpha": math.nan},
        {"gain": 0.0},
        {"gain": math.inf},
        {"keep_live": -1},
        {"keep_live": 145},
        {"loss_cutoff": 1.5},
        {"loss_cutoff": -0.01},
        {"keep_live": "ten"},
        {"n_discard": 2},
    ],
)
async def test_a_value_out_of_range_is_refused_and_changes_nothing(
    shared: TrajectoryTool, state: Path, partial: dict[str, Any]
) -> None:
    store = await _load(shared, state)
    before = store.solve

    with pytest.raises(SolveConfigRefused) as caught:
        await store.update(partial, "operator")

    assert caught.value.code == INVALID_SOLVE_CONFIG and caught.value.status == 422
    assert store.solve is before
    assert not (state / SOLVE_DIR).exists()


@pytest.mark.parametrize(
    ("key", "name"),
    [("excluded_bpms", "NOT:A:BPM"), ("excluded_cms", "NOT:A:CM"), ("excluded_cms", "bpm")],
)
async def test_a_name_the_build_lacks_is_refused(
    shared: TrajectoryTool, key: str, name: str
) -> None:
    store = await _load(shared, None)
    if name == "bpm":
        name = shared.layout.bpm_names_h[0]  # a BPM is no corrector

    with pytest.raises(SolveConfigRefused) as caught:
        await store.update({key: [name]}, "operator")

    assert caught.value.code == SELECTION_UNKNOWN and caught.value.status == 409
    assert store.overrides == {}


@pytest.mark.parametrize("what", ["bpms", "hcms", "vcms"])
async def test_a_selection_without_a_bpm_or_a_live_corrector_per_plane_is_refused(
    shared: TrajectoryTool, what: str
) -> None:
    store = await _load(shared, None)
    layout = shared.layout
    partial = {
        "bpms": {"excluded_bpms": list(layout.bpm_names_h)},
        "hcms": {"excluded_cms": list(layout.hcorr_names)},
        "vcms": {"excluded_cms": list(layout.vcorr_names)},
    }[what]

    with pytest.raises(SolveConfigRefused) as caught:
        await store.update(partial, "operator")

    assert caught.value.code == SELECTION_TOO_SMALL
    assert store.overrides == {}


async def test_a_selection_keeping_one_of_each_solves(shared: TrajectoryTool) -> None:
    store = await _load(shared, None)
    layout = shared.layout

    update = await store.update(
        {
            "excluded_bpms": list(layout.bpm_names_h[:-1]),
            "excluded_cms": [*layout.hcorr_names[1:], *layout.vcorr_names[1:]],
        },
        "operator",
    )

    assert (update.solve.rm.rows, update.solve.rm.h_cols, update.solve.rm.v_cols) == (2, 1, 1)


async def test_a_loss_cutoff_change_reaches_the_hook_and_keeps_the_solve(
    shared: TrajectoryTool, state: Path
) -> None:
    store = await _load(shared, state)
    before = store.solve
    assert before is not None
    seen: list[float] = []

    async def hook(cutoff: float) -> None:
        seen.append(cutoff)

    store.on_loss_cutoff = hook

    update = await store.update({"loss_cutoff": 0.25}, "operator")

    assert seen == [0.25]
    assert update.solve.sha == before.sha
    assert update.solve.rm.computed_at == before.rm.computed_at
    assert update.solve.config.loss_cutoff == 0.25
    assert update.solve.config.overridden == ["loss_cutoff"]
    assert _file(state)["overrides"] == {"loss_cutoff": {"value": 0.25, "default": LOSS_CUTOFF}}


# -- the load ------------------------------------------------------------------


async def test_overrides_survive_a_restart(shared: TrajectoryTool, state: Path) -> None:
    store = await _load(shared, state)
    cm = shared.layout.vcorr_names[2]
    await store.update({"alpha": 0.004, "excluded_cms": [cm], "keep_live": 100}, "operator")

    again = await _load(shared, state)

    assert again.notices == []
    assert again.solve is not None and store.solve is not None
    assert again.solve.config == store.solve.config
    assert again.solve.sha == store.solve.sha


async def test_an_override_whose_file_default_changed_is_dropped_with_a_notice(
    tmp_path: Path, goldens: Any, state: Path
) -> None:
    first = _build(tmp_path / "build", goldens, n_discard=4)
    store = await _load(first, state)
    await store.update({"keep_live": 100, "alpha": 0.004}, "operator")

    rebuilt = _build(tmp_path / "rebuilt", goldens, n_discard=6)
    again = await _load(rebuilt, state)

    assert again.solve is not None
    assert again.solve.config.keep_live == int(rebuilt.layout.L) - 6
    assert again.solve.config.alpha == 0.004
    assert again.solve.config.overridden == ["alpha"]
    assert len(again.notices) == 1 and "keep_live" in again.notices[0]
    assert set(_file(state)["overrides"]) == {"alpha"}


async def test_an_override_naming_what_left_the_build_is_dropped_with_a_notice(
    shared: TrajectoryTool, state: Path
) -> None:
    path = state / SOLVE_DIR / f"{MODEL}.json"
    path.parent.mkdir(parents=True)
    cm = shared.layout.hcorr_names[0]
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "overrides": {
                    "excluded_bpms": {"value": ["GONE:BPM"], "default": []},
                    "excluded_cms": {"value": [cm], "default": []},
                },
            }
        ),
        encoding="utf-8",
    )

    store = await _load(shared, state)

    assert store.solve is not None
    assert store.solve.config.excluded_bpms == []
    assert store.solve.config.excluded_cms == [cm]
    assert len(store.notices) == 1 and "GONE:BPM" in store.notices[0]
    assert set(_file(state)["overrides"]) == {"excluded_cms"}


async def test_an_unreadable_overrides_file_falls_back_to_the_defaults(
    shared: TrajectoryTool, state: Path
) -> None:
    path = state / SOLVE_DIR / f"{MODEL}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    store = await _load(shared, state)

    assert store.solve is not None and store.solve.config.overridden == []
    assert len(store.notices) == 1


# -- the step on the sessions --------------------------------------------------


class _Layer:
    def __init__(self, startup: Any) -> None:
        self.startup = startup
        self._stop = asyncio.Event()
        self.turn_hook: Any = None

    async def submit(self, owner: str, kind: str, action: Any, *, precondition: Any = None) -> Any:
        admission = Admission(owner=owner, kind=kind, target=TARGET, generation=0)
        if precondition is not None:
            precondition(admission)
        return await action(admission, self._stop)

    def control(self, owner: str, kind: str, fn: Any, *, admit: bool = True) -> Any:
        admission = Admission(owner=owner, kind=kind, target=TARGET, generation=0)
        return fn(admission if admit else None)

    def renew_stop_event(self) -> asyncio.Event:
        self._stop.set()
        self._stop = asyncio.Event()
        return self._stop


class _NoReferences:
    async def current(self, _model: str) -> None:
        return None


def _shot(tool: TrajectoryTool, shot_id: str = "shot-1") -> ShotResult:
    xs, ys = list(tool.layout.bpm_names_h), list(tool.layout.bpm_names_v)
    return ShotResult(
        shot_id=shot_id,
        model=MODEL,
        x={name: 1e-4 * (i % 7 - 3) for i, name in enumerate(xs)},
        y={name: 1e-4 * (i % 5 - 2) for i, name in enumerate(ys)},
        sum=dict.fromkeys(xs, 1.0),
        bpm_order=xs,
        lost=[],
        last_bpm=xs[-1],
        i0=0,
        window=(0, 10),
        per_shot_lost_at=[None],
        shots_included=1,
        shots_failed=[],
        commands_fired=[],
    )


async def _sessions(shared: TrajectoryTool) -> tuple[Sessions, str]:
    startup = SimpleNamespace(
        target=TARGET,
        connector=SimpleNamespace(_limits_validator=None),
        commands=SimpleNamespace(model=MODEL),
        tool=shared,
    )
    sessions = Sessions(_Layer(startup))  # type: ignore[arg-type]
    sessions.solve = await _load(shared, None)
    grant = await sessions.open("alice", "panel")
    return sessions, grant.client_id


async def _shoot(sessions: Sessions, client: str, shot: ShotResult) -> None:
    def build(hook: Any) -> Any:
        async def action(_admission: Admission, _stop: asyncio.Event) -> ShotResult:
            await hook(shot)
            return shot

        return action

    await sessions.shot("alice", "panel", client, _NoReferences(), build)  # type: ignore[arg-type]


async def _solve(sessions: Sessions, client: str, **partial: Any) -> Any:
    return await sessions.solve_config(
        "alice",
        "panel",
        client,
        _NoReferences(),  # type: ignore[arg-type]
        SolveConfigPartial(**partial),
    )


async def test_a_holders_step_binds_the_published_solve(shared: TrajectoryTool) -> None:
    sessions, client = await _sessions(shared)
    await _solve(sessions, client, alpha=0.003, gain=0.5)

    await _shoot(sessions, client, _shot(shared))

    pending = sessions.step.step
    assert pending is not None and sessions.solve is not None and sessions.solve.solve is not None
    assert "singular_values" not in pending.model_dump()
    assert pending.digest == step.step_digest(
        target=TARGET,
        model=MODEL,
        client_id=client,
        shot_id="shot-1",
        reference=None,
        solve_sha=sessions.solve.solve.sha,
        deltas=pending.deltas,
    )


async def test_a_knob_change_recomputes_the_step_from_its_shot(shared: TrajectoryTool) -> None:
    sessions, client = await _sessions(shared)
    shot = _shot(shared)
    await _shoot(sessions, client, shot)
    first = sessions.step.step
    assert first is not None

    update, recomputed = await _solve(sessions, client, gain=0.5)

    assert update.changed == {"gain"}
    assert recomputed is sessions.step.step and recomputed is not None
    assert recomputed.shot_id == "shot-1" and sessions.step.shot is shot
    assert recomputed.digest != first.digest
    for name, delta in first.deltas.items():
        assert recomputed.deltas[name] == pytest.approx(0.5 * delta, rel=1e-9, abs=1e-18)


async def test_a_bpm_exclusion_change_voids_the_step_and_its_shot(
    shared: TrajectoryTool,
) -> None:
    sessions, client = await _sessions(shared)
    await _shoot(sessions, client, _shot(shared))

    _update, after = await _solve(sessions, client, excluded_bpms=[shared.layout.bpm_names_h[5]])

    assert after is None
    assert sessions.step.step is None and sessions.step.shot is None


async def test_a_corrector_exclusion_change_voids_the_step_and_keeps_its_shot(
    shared: TrajectoryTool,
) -> None:
    sessions, client = await _sessions(shared)
    shot = _shot(shared)
    await _shoot(sessions, client, shot)
    cm = shared.layout.hcorr_names[3]

    _update, after = await _solve(sessions, client, excluded_cms=[cm])

    assert after is None and sessions.step.step is None
    assert sessions.step.shot is shot

    update, recomputed = await _solve(sessions, client)

    assert update.recalculated
    assert recomputed is not None and recomputed.shot_id == "shot-1"
    assert cm not in recomputed.deltas
    assert sessions.solve is not None and sessions.solve.solve is not None
    assert recomputed.digest == step.step_digest(
        target=TARGET,
        model=MODEL,
        client_id=client,
        shot_id="shot-1",
        reference=None,
        solve_sha=sessions.solve.solve.sha,
        deltas=recomputed.deltas,
    )


async def test_a_loss_cutoff_change_leaves_the_step(shared: TrajectoryTool) -> None:
    sessions, client = await _sessions(shared)
    await _shoot(sessions, client, _shot(shared))
    pending = sessions.step.step

    _update, after = await _solve(sessions, client, loss_cutoff=0.3)

    assert after is pending and sessions.step.step is pending


async def test_an_update_is_a_holder_call(shared: TrajectoryTool) -> None:
    sessions, _client = await _sessions(shared)

    with pytest.raises(SessionRefused) as caught:
        await _solve(sessions, "not-the-client", alpha=0.002)

    assert caught.value.code == NOT_SESSION_HOLDER
    assert sessions.solve is not None and sessions.solve.overrides == {}


async def test_during_a_continuous_run_only_the_loss_cutoff_may_change(
    shared: TrajectoryTool,
) -> None:
    sessions, client = await _sessions(shared)
    await _shoot(sessions, client, _shot(shared))
    pending = sessions.step.step
    sessions._continuous = SimpleNamespace(running=True)  # type: ignore[assignment]

    for partial in ({"alpha": 0.002}, {"alpha": 0.002, "loss_cutoff": 0.3}, {}):
        with pytest.raises(SessionRefused) as caught:
            await _solve(sessions, client, **partial)
        assert caught.value.code == CONTINUOUS_RUNNING and caught.value.status == 409

    update, after = await _solve(sessions, client, loss_cutoff=0.3)

    assert update.changed == {"loss_cutoff"} and after is pending
    with pytest.raises(SessionRefused) as caught:
        await _solve(sessions, "not-the-client", loss_cutoff=0.2)
    assert caught.value.code == NOT_SESSION_HOLDER
    assert sessions.solve is not None and sessions.solve.values["loss_cutoff"] == 0.3


async def test_a_take_over_keeps_the_solve_config(shared: TrajectoryTool) -> None:
    sessions, client = await _sessions(shared)
    await _solve(sessions, client, alpha=0.002)

    sessions.take_over("bob", "agent")

    assert sessions.solve is not None and sessions.solve.values["alpha"] == 0.002
