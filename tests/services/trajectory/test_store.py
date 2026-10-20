"""The trajectory store: shot files, references, names, NaN, matching, pruning, atomic writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
from pathlib import Path

import pytest

from osprey.services.trajectory.state import ReferenceLoad, ShotResult
from osprey.services.trajectory.store import (
    FILE_MODE,
    STATE_DIR_ENV_VAR,
    InvalidName,
    NoMatchingBpms,
    ReferenceNotFound,
    StoreUnconfigured,
    TrajectoryStore,
)

MODEL = "sr"


def _shot(shot_id: str, *, model: str = MODEL, x: float | None = 0.1) -> ShotResult:
    return ShotResult(
        shot_id=shot_id,
        model=model,
        x={"BPM1:X": x, "BPM2:X": 0.2},
        y={"BPM1:Y": 0.0, "BPM2:Y": None},
        sum={"BPM1:SUM": 5.0, "BPM2:SUM": 0.1},
        bpm_order=["BPM1", "BPM2"],
        lost=["BPM2"],
        last_bpm="BPM1",
        i0=3,
        window=(3, 10),
        per_shot_lost_at=["BPM2"],
        shots_included=1,
        shots_failed=[],
        commands_fired=["ARM"],
        timestamp=1.0,
    )


@pytest.fixture
def store(tmp_path: Path) -> TrajectoryStore:
    return TrajectoryStore(tmp_path, shots_per_model=3)


# --- configuration -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_state_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STATE_DIR_ENV_VAR, raising=False)


def test_the_store_is_rooted_at_the_state_dir_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(tmp_path))
    assert TrajectoryStore.from_env().root == tmp_path


@pytest.mark.parametrize("value", ["", "   "])
def test_an_unset_or_empty_state_dir_refuses_and_names_the_variable(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value:
        monkeypatch.setenv(STATE_DIR_ENV_VAR, value)
    with pytest.raises(StoreUnconfigured, match=STATE_DIR_ENV_VAR):
        TrajectoryStore.from_env()


def test_a_shot_count_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrajectoryStore(tmp_path, shots_per_model=0)


# --- names ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["", "a" * 65, "has space", "slash/name", "back\\slash", ".", "..", "naïve", "semi;colon"],
)
async def test_a_bad_reference_name_is_refused_before_any_file_is_written(
    store: TrajectoryStore, name: str
) -> None:
    with pytest.raises(InvalidName):
        await store.save_reference(MODEL, name, {"BPM1": (0.0, 0.0)})
    assert not (store.root / "references").exists()


async def test_the_reserved_current_name_is_refused_as_a_reference_name(
    store: TrajectoryStore,
) -> None:
    with pytest.raises(InvalidName, match="current"):
        await store.save_reference(MODEL, "current", {"BPM1": (0.0, 0.0)})


@pytest.mark.parametrize("name", ["a", "a" * 64, "Golden_2026-10-16.v2", "-x"])
async def test_every_name_the_pattern_allows_is_saved(store: TrajectoryStore, name: str) -> None:
    await store.save_reference(MODEL, name, {"BPM1": (0.0, 0.0)})
    assert (store.root / "references" / MODEL / f"{name}.json").is_file()


@pytest.mark.parametrize("model", ["", "../escape", "a/b", ".."])
async def test_a_bad_model_name_is_refused_for_shots_and_references(
    store: TrajectoryStore, model: str
) -> None:
    with pytest.raises(InvalidName):
        await store.save_shot(_shot("s1", model=model))
    with pytest.raises(InvalidName):
        await store.save_reference(model, "ref", {"BPM1": (0.0, 0.0)})
    with pytest.raises(InvalidName):
        await store.list_shots(model)


@pytest.mark.parametrize("shot_id", ["../x", "a b", ""])
async def test_a_bad_shot_id_is_refused(store: TrajectoryStore, shot_id: str) -> None:
    with pytest.raises(InvalidName):
        await store.save_shot(_shot(shot_id))
    with pytest.raises(InvalidName):
        await store.load_shot(MODEL, shot_id)


# --- shots ---------------------------------------------------------------------------


async def test_a_saved_shot_lands_in_its_model_directory_and_loads_back_equal(
    store: TrajectoryStore,
) -> None:
    shot = _shot("s1")
    path = await store.save_shot(shot)
    assert path == store.root / "shots" / MODEL / "s1.json"
    assert await store.load_shot(MODEL, "s1") == shot


async def test_a_shot_reading_nan_is_written_as_null_and_the_file_is_strict_json(
    store: TrajectoryStore,
) -> None:
    path = await store.save_shot(_shot("s1", x=None))
    data = json.loads(path.read_text(), parse_constant=pytest.fail)
    assert data["x"]["BPM1:X"] is None


async def test_every_file_the_store_writes_is_owner_written_and_group_read(
    store: TrajectoryStore,
) -> None:
    shot = await store.save_shot(_shot("s1"))
    await store.save_reference(MODEL, "gold", {"BPM1": (0.0, 0.0)})
    await store.set_current(MODEL, "gold")
    references = store.root / "references" / MODEL

    for path in (shot, references / "gold.json", references / "current.json"):
        assert path.stat().st_mode & 0o777 == FILE_MODE == 0o640, path


async def test_loading_a_missing_shot_raises_lookup_error(store: TrajectoryStore) -> None:
    with pytest.raises(LookupError):
        await store.load_shot(MODEL, "nope")


async def test_shots_are_pruned_to_the_per_model_count_oldest_first(
    store: TrajectoryStore,
) -> None:
    for index in range(5):
        await store.save_shot(_shot(f"s{index}"))
    listed = await store.list_shots(MODEL)
    assert [shot.shot_id for shot in listed] == ["s4", "s3", "s2"]
    files = sorted(p.name for p in (store.root / "shots" / MODEL).iterdir())
    assert files == ["s2.json", "s3.json", "s4.json"]


async def test_pruning_one_model_leaves_another_models_shots_alone(
    store: TrajectoryStore,
) -> None:
    await store.save_shot(_shot("other", model="booster"))
    for index in range(4):
        await store.save_shot(_shot(f"s{index}"))
    assert [s.shot_id for s in await store.list_shots("booster")] == ["other"]


async def test_resaving_a_shot_id_keeps_one_file_and_makes_it_newest(
    store: TrajectoryStore,
) -> None:
    for shot_id in ["a", "b", "c"]:
        await store.save_shot(_shot(shot_id))
    await store.save_shot(_shot("a", x=0.9))
    listed = await store.list_shots(MODEL)
    assert [s.shot_id for s in listed] == ["a", "c", "b"]
    assert listed[0].x["BPM1:X"] == 0.9


async def test_listing_a_model_with_no_shots_is_empty(store: TrajectoryStore) -> None:
    assert await store.list_shots(MODEL) == []


# --- references ----------------------------------------------------------------------


async def test_a_reference_exposes_the_sha256_of_its_file_content(
    store: TrajectoryStore,
) -> None:
    sha = await store.save_reference(MODEL, "gold", {"BPM1": (0.1, 0.2), "BPM2": (0.3, 0.4)})
    path = store.root / "references" / MODEL / "gold.json"
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = await store.load_reference(MODEL, "gold", ["BPM1", "BPM2"])
    assert loaded.load.sha256 == sha
    assert loaded.load.name == "gold"


async def test_different_reference_content_gives_a_different_sha(
    store: TrajectoryStore,
) -> None:
    first = await store.save_reference(MODEL, "gold", {"BPM1": (0.1, 0.2)})
    second = await store.save_reference(MODEL, "gold", {"BPM1": (0.1, 0.25)})
    assert first != second


async def test_a_full_match_returns_every_reading_and_no_missing_or_unused(
    store: TrajectoryStore,
) -> None:
    await store.save_reference(MODEL, "gold", {"BPM1": (0.1, 0.2), "BPM2": (0.3, 0.4)})
    loaded = await store.load_reference(MODEL, "gold", ["BPM1", "BPM2"])
    assert isinstance(loaded.load, ReferenceLoad)
    assert loaded.load.matched == ["BPM1", "BPM2"]
    assert loaded.load.missing_in_reference == []
    assert loaded.load.unused_in_reference == []
    assert loaded.load.nan_count == 0
    assert loaded.x == {"BPM1": 0.1, "BPM2": 0.3}
    assert loaded.y == {"BPM1": 0.2, "BPM2": 0.4}


async def test_a_partial_match_names_missing_and_unused_bpms_and_reads_nan_where_missing(
    store: TrajectoryStore,
) -> None:
    await store.save_reference(MODEL, "gold", {"BPM1": (0.1, 0.2), "BPM9": (0.5, 0.6)})
    loaded = await store.load_reference(MODEL, "gold", ["BPM1", "BPM2", "BPM3"])
    assert loaded.load.matched == ["BPM1"]
    assert loaded.load.missing_in_reference == ["BPM2", "BPM3"]
    assert loaded.load.unused_in_reference == ["BPM9"]
    assert list(loaded.x) == ["BPM1", "BPM2", "BPM3"]
    assert loaded.x["BPM1"] == 0.1
    assert math.isnan(loaded.x["BPM2"]) and math.isnan(loaded.y["BPM3"])
    assert "BPM9" not in loaded.x


async def test_nan_readings_are_stored_as_null_and_flagged_on_load(
    store: TrajectoryStore,
) -> None:
    await store.save_reference(
        MODEL, "gold", {"BPM1": (float("nan"), 0.2), "BPM2": (None, float("nan"))}
    )
    raw = (store.root / "references" / MODEL / "gold.json").read_text()
    json.loads(raw, parse_constant=pytest.fail)
    loaded = await store.load_reference(MODEL, "gold", ["BPM1", "BPM2"])
    assert loaded.load.nan_count == 3
    assert loaded.load.matched == ["BPM1", "BPM2"]
    assert math.isnan(loaded.x["BPM1"]) and loaded.y["BPM1"] == 0.2


async def test_an_infinite_reading_is_refused(store: TrajectoryStore) -> None:
    with pytest.raises(ValueError):
        await store.save_reference(MODEL, "gold", {"BPM1": (float("inf"), 0.0)})


async def test_a_reference_matching_no_selected_bpm_is_refused(
    store: TrajectoryStore,
) -> None:
    await store.save_reference(MODEL, "gold", {"BPM9": (0.1, 0.2)})
    with pytest.raises(NoMatchingBpms, match="gold"):
        await store.load_reference(MODEL, "gold", ["BPM1", "BPM2"])


async def test_an_empty_reference_is_refused_at_save(store: TrajectoryStore) -> None:
    with pytest.raises(ValueError):
        await store.save_reference(MODEL, "gold", {})


async def test_loading_an_unknown_reference_raises_reference_not_found(
    store: TrajectoryStore,
) -> None:
    with pytest.raises(ReferenceNotFound, match="gold"):
        await store.load_reference(MODEL, "gold", ["BPM1"])


async def test_references_are_listed_sorted_without_the_current_pointer(
    store: TrajectoryStore,
) -> None:
    await store.save_reference(MODEL, "b", {"BPM1": (0.0, 0.0)})
    await store.save_reference(MODEL, "a", {"BPM1": (0.0, 0.0)})
    await store.set_current(MODEL, "b")
    assert await store.list_references(MODEL) == ["a", "b"]
    assert await store.list_references("booster") == []


# --- current -------------------------------------------------------------------------


async def test_current_is_none_until_set_and_then_names_the_reference(
    store: TrajectoryStore,
) -> None:
    assert await store.current(MODEL) is None
    await store.save_reference(MODEL, "gold", {"BPM1": (0.0, 0.0)})
    await store.set_current(MODEL, "gold")
    assert await store.current(MODEL) == "gold"
    pointer = json.loads((store.root / "references" / MODEL / "current.json").read_text())
    assert pointer == {"name": "gold"}


async def test_current_can_be_cleared(store: TrajectoryStore) -> None:
    await store.save_reference(MODEL, "gold", {"BPM1": (0.0, 0.0)})
    await store.set_current(MODEL, "gold")
    await store.set_current(MODEL, None)
    assert await store.current(MODEL) is None


async def test_setting_current_to_an_unknown_reference_is_refused(
    store: TrajectoryStore,
) -> None:
    with pytest.raises(ReferenceNotFound):
        await store.set_current(MODEL, "gold")
    with pytest.raises(InvalidName):
        await store.set_current(MODEL, "../gold")


async def test_current_is_per_model(store: TrajectoryStore) -> None:
    await store.save_reference(MODEL, "gold", {"BPM1": (0.0, 0.0)})
    await store.set_current(MODEL, "gold")
    assert await store.current("booster") is None


# --- atomic, serialized writes -------------------------------------------------------


async def test_concurrent_reference_writes_never_leave_a_torn_file(
    store: TrajectoryStore,
) -> None:
    bpms = [f"BPM{i}" for i in range(400)]
    contents = [{bpm: (float(k), float(-k)) for bpm in bpms} for k in range(12)]
    await store.save_reference(MODEL, "gold", contents[0])
    path = store.root / "references" / MODEL / "gold.json"
    valid = {float(k) for k in range(12)}
    torn: list[str] = []
    done = threading.Event()

    def read_forever() -> None:
        while not done.is_set():
            data = json.loads(path.read_text())
            xs = {entry["x"] for entry in data["readings"].values()}
            if len(xs) != 1 or not xs <= valid:
                torn.append(repr(xs))

    reader = asyncio.create_task(asyncio.to_thread(read_forever))
    try:
        shas = await asyncio.gather(
            *(store.save_reference(MODEL, "gold", content) for content in contents)
        )
    finally:
        done.set()
        await reader
    assert torn == []
    assert len(set(shas)) == len(contents)
    final = await store.load_reference(MODEL, "gold", bpms)
    assert final.load.sha256 in shas
    assert [p.name for p in path.parent.iterdir()] == ["gold.json"]


async def test_concurrent_shot_writes_all_land_and_prune_to_the_count(
    tmp_path: Path,
) -> None:
    store = TrajectoryStore(tmp_path, shots_per_model=5)
    await asyncio.gather(*(store.save_shot(_shot(f"s{i:02d}")) for i in range(20)))
    listed = await store.list_shots(MODEL)
    assert len(listed) == 5
    names = sorted(p.name for p in (tmp_path / "shots" / MODEL).iterdir())
    assert len(names) == 5 and all(not n.startswith(".") for n in names)


async def test_a_failed_write_leaves_no_temporary_file_and_the_old_content(
    store: TrajectoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = await store.save_reference(MODEL, "gold", {"BPM1": (0.1, 0.2)})
    import osprey.services.trajectory.store as store_module

    def boom(*_args: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store_module.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        await store.save_reference(MODEL, "gold", {"BPM1": (0.9, 0.9)})
    monkeypatch.undo()
    directory = store.root / "references" / MODEL
    assert [p.name for p in directory.iterdir()] == ["gold.json"]
    assert (await store.load_reference(MODEL, "gold", ["BPM1"])).load.sha256 == sha


# --- layout keyed by target and model ------------------------------------------------


async def test_a_target_bound_store_keeps_its_files_under_the_target(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path, target="va")
    await store.save_shot(_shot("s1"))
    await store.save_reference(MODEL, "gold", {"BPM1": (0.0, 0.0)})
    await store.set_current(MODEL, "gold")
    assert store.directory == tmp_path / "va"
    assert (tmp_path / "va" / "shots" / MODEL / "s1.json").is_file()
    assert (tmp_path / "va" / "references" / MODEL / "gold.json").is_file()
    assert (tmp_path / "va" / "references" / MODEL / "current.json").is_file()
    assert not (tmp_path / "shots").exists()
    assert not (tmp_path / "references").exists()


async def test_two_targets_sharing_a_state_dir_never_see_each_others_files(
    tmp_path: Path,
) -> None:
    va = TrajectoryStore(tmp_path, target="va")
    standin = TrajectoryStore(tmp_path, target="standin")
    await va.save_shot(_shot("s1"))
    await va.save_reference(MODEL, "gold", {"BPM1": (0.0, 0.0)})
    await va.set_current(MODEL, "gold")
    assert await standin.list_shots(MODEL) == []
    assert await standin.list_references(MODEL) == []
    assert await standin.current(MODEL) is None
    with pytest.raises(LookupError):
        await standin.load_shot(MODEL, "s1")
    with pytest.raises(ReferenceNotFound):
        await standin.load_reference(MODEL, "gold", ["BPM1"])
    with pytest.raises(ReferenceNotFound):
        await standin.set_current(MODEL, "gold")


async def test_pruning_counts_one_target_and_model_at_a_time(tmp_path: Path) -> None:
    va = TrajectoryStore(tmp_path, target="va", shots_per_model=2)
    live = TrajectoryStore(tmp_path, target="live", shots_per_model=2)
    for i in range(3):
        await va.save_shot(_shot(f"v{i}"))
    await live.save_shot(_shot("l0"))
    assert [shot.shot_id for shot in await va.list_shots(MODEL)] == ["v2", "v1"]
    assert [shot.shot_id for shot in await live.list_shots(MODEL)] == ["l0"]


@pytest.mark.parametrize("target", ["", "../escape", "a/b", "..", "."])
def test_a_bad_target_name_is_refused_at_construction(tmp_path: Path, target: str) -> None:
    with pytest.raises(InvalidName, match="target"):
        TrajectoryStore(tmp_path, target=target)


def test_the_target_reaches_the_store_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(tmp_path))
    store = TrajectoryStore.from_env(target="standin")
    assert store.root == tmp_path
    assert store.target == "standin"
    assert store.directory == tmp_path / "standin"


def test_a_store_bound_to_no_target_keeps_its_files_at_the_root(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path)
    assert store.target is None
    assert store.directory == tmp_path
