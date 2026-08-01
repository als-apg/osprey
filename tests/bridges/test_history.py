"""Tests for :class:`osprey.bridges.core.history.HistoryStore`.

Covers the current turn shape ({question, answer, ts, run_id, artifacts}), the
grandfathering of pre-feature turns (PROPOSAL criterion 8: mixed old/new files
load and replay correctly), the char budget counting the full serialized turn,
the per-turn artifact cap, and the 90-day age-prune — all driven by an injected
clock so there are no real sleeps.
"""

from __future__ import annotations

import json

from osprey.bridges.core.history import (
    FAILED_TURN_ANSWER,
    MAX_AGE_SECONDS,
    MAX_ARTIFACTS_PER_TURN,
    HistoryStore,
)

DAY = 24 * 60 * 60


class FakeClock:
    """Monotone-ish injectable clock: reads return ``t``; tests set ``t``."""

    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _write(path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- basic round-trip -------------------------------------------------------


def test_append_and_recent_basic(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    h.append("k", "q1", "a1")
    h.append("k", "q2", "a2")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["q1", "q2"]
    assert [t["answer"] for t in turns] == ["a1", "a2"]
    # New turns carry the full shape.
    for t in turns:
        assert set(t) == {"question", "answer", "ts", "run_id", "artifacts"}
        assert t["run_id"] is None
        assert t["artifacts"] == []


def test_empty_key_is_noop(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    h.append("", "q", "a")
    assert h.recent("") == []


def test_recent_unknown_key(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    assert h.recent("nope") == []


# --- append signature: back-compat + new params -----------------------------


def test_append_three_arg_still_works(tmp_path):
    """Pre-existing 3-arg callers must keep working."""
    h = HistoryStore(str(tmp_path / "h.json"))
    h.append("k", "q", "a")  # no run_id / artifacts
    (turn,) = h.recent("k")
    assert turn["run_id"] is None
    assert turn["artifacts"] == []


def test_append_with_run_id_and_artifacts(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    arts = [
        {"entry_id": "e1", "filename": "plot.png", "delivered_mime": "image/png", "origin": "run"}
    ]
    h.append("k", "q", "a", run_id="run-123", artifacts=arts)
    (turn,) = h.recent("k")
    assert turn["run_id"] == "run-123"
    assert turn["artifacts"] == arts


def test_artifacts_capped_tail_kept(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    arts = [{"entry_id": f"e{i}"} for i in range(MAX_ARTIFACTS_PER_TURN + 5)]
    h.append("k", "q", "a", artifacts=arts)
    (turn,) = h.recent("k")
    assert len(turn["artifacts"]) == MAX_ARTIFACTS_PER_TURN
    # Tail retained (most recent).
    assert turn["artifacts"][0]["entry_id"] == "e5"
    assert turn["artifacts"][-1]["entry_id"] == f"e{MAX_ARTIFACTS_PER_TURN + 4}"


def test_descriptors_round_trip_opaquely(tmp_path):
    """The store must not inspect descriptor shape — arbitrary dicts survive."""
    h = HistoryStore(str(tmp_path / "h.json"))
    weird = [{"anything": {"nested": [1, 2, 3]}, "x": None}]
    h.append("k", "q", "a", artifacts=weird)
    (turn,) = h.recent("k")
    assert turn["artifacts"] == weird


# --- append_failed ----------------------------------------------------------


def test_append_failed_records_marker(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    h.append_failed("k", "q")
    (turn,) = h.recent("k")
    assert turn["question"] == "q"
    assert turn["answer"] == FAILED_TURN_ANSWER
    assert turn["run_id"] is None
    assert turn["artifacts"] == []


def test_append_failed_empty_question_noop(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"))
    h.append_failed("k", "")
    assert h.recent("k") == []


# --- caps: turn count and char budget ---------------------------------------


def test_turn_count_cap(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"), max_turns=3)
    for i in range(6):
        h.append("k", f"q{i}", f"a{i}")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["q3", "q4", "q5"]
    # Persisted window is also bounded.
    assert len(_read(tmp_path / "h.json")["k"]) == 3


def test_char_budget_drops_oldest(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"), max_chars=200)
    h.append("k", "x" * 100, "y" * 100)  # oldest, large
    h.append("k", "q2", "a2")
    h.append("k", "q3", "a3")
    turns = h.recent("k")
    # The big first turn is dropped by the char budget; newest survive.
    assert [t["question"] for t in turns] == ["q2", "q3"]


def test_char_budget_counts_full_serialized_turn(tmp_path):
    """The budget must charge for the whole serialized turn (descriptors
    included), not just question+answer — a turn heavy with artifact
    descriptors must be able to blow the budget on its own."""
    # Tiny q/a but a big artifact payload.
    big_art = [{"blob": "z" * 500}]
    small = HistoryStore(str(tmp_path / "s.json"))
    small.append("k", "q1", "a1")
    small.append("k", "q2", "a2", artifacts=big_art)
    # Budget large enough for two text-only turns but not once the 500-char
    # descriptor is counted: the oldest text turn is evicted.
    h = HistoryStore(str(tmp_path / "h.json"), max_chars=300)
    h.append("k", "q1", "a1")
    h.append("k", "q2", "a2", artifacts=big_art)
    turns = h.recent("k")
    # Only the newest (artifact-heavy) turn fits; it always survives as newest.
    assert [t["question"] for t in turns] == ["q2"]
    assert turns[0]["artifacts"] == big_art


def test_recent_always_keeps_newest_even_if_over_budget(tmp_path):
    h = HistoryStore(str(tmp_path / "h.json"), max_chars=1)
    h.append("k", "q1", "a1")
    h.append("k", "q2", "a2")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["q2"]


# --- 90-day age prune -------------------------------------------------------


def test_age_prune_on_append(tmp_path):
    clock = FakeClock(1_000_000.0)
    h = HistoryStore(str(tmp_path / "h.json"), now=clock)
    h.append("k", "old", "a")
    # Advance past the age ceiling, then append — old turn is pruned.
    clock.t += MAX_AGE_SECONDS + DAY
    h.append("k", "new", "a")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["new"]


def test_age_prune_keeps_within_window(tmp_path):
    clock = FakeClock(1_000_000.0)
    h = HistoryStore(str(tmp_path / "h.json"), now=clock)
    h.append("k", "t0", "a")
    clock.t += MAX_AGE_SECONDS - DAY  # still inside the window
    h.append("k", "t1", "a")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["t0", "t1"]


# --- grandfathering (PROPOSAL criterion 8) ----------------------------------


def test_grandfather_old_turns_stamps_load_time(tmp_path):
    path = tmp_path / "h.json"
    _write(path, {"k": [{"question": "old_q", "answer": "old_a"}]})
    clock = FakeClock(555.0)
    h = HistoryStore(str(path), now=clock)
    (turn,) = h.recent("k")
    assert turn["question"] == "old_q"
    assert turn["answer"] == "old_a"
    assert turn["ts"] == 555.0  # stamped at load time
    assert turn["run_id"] is None
    assert turn["artifacts"] == []


def test_grandfathered_ts_is_persisted(tmp_path):
    """The load-time ts must be written back, so a restart doesn't re-stamp to
    an ever-later 'now' and the age-prune reads a stable anchor."""
    path = tmp_path / "h.json"
    _write(path, {"k": [{"question": "old_q", "answer": "old_a"}]})
    clock = FakeClock(555.0)
    HistoryStore(str(path), now=clock)  # construction persists coercion
    on_disk = _read(path)
    assert on_disk["k"][0]["ts"] == 555.0
    assert on_disk["k"][0]["run_id"] is None
    assert on_disk["k"][0]["artifacts"] == []


def test_grandfathered_history_survives_first_append(tmp_path):
    """Regression for criterion 8: a pre-feature conversation must NOT be wiped
    as 'ancient' the first time a new turn is appended."""
    path = tmp_path / "h.json"
    _write(path, {"k": [{"question": "old_q", "answer": "old_a"}]})
    clock = FakeClock(1_000_000.0)
    h = HistoryStore(str(path), now=clock)
    # Small advance (well within 90d), then append.
    clock.t += DAY
    h.append("k", "new_q", "new_a")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["old_q", "new_q"]


def test_grandfathered_ancient_pruned_only_after_window(tmp_path):
    """Once the load-time anchor itself ages past the window, the old turn is
    pruned on append (it is not immortal)."""
    path = tmp_path / "h.json"
    _write(path, {"k": [{"question": "old_q", "answer": "old_a"}]})
    clock = FakeClock(1_000_000.0)
    h = HistoryStore(str(path), now=clock)  # old_q anchored at 1_000_000
    clock.t += MAX_AGE_SECONDS + DAY
    h.append("k", "new_q", "new_a")
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["new_q"]


def test_mixed_old_and_new_turns_load(tmp_path):
    """A file with both legacy (2-field) and current (5-field) turns loads and
    replays correctly, preserving order."""
    path = tmp_path / "h.json"
    _write(
        path,
        {
            "k": [
                {"question": "legacy_q", "answer": "legacy_a"},  # old shape
                {
                    "question": "new_q",
                    "answer": "new_a",
                    "ts": 2_000_000.0,
                    "run_id": "run-9",
                    "artifacts": [{"entry_id": "e1"}],
                },
            ]
        },
    )
    clock = FakeClock(1_500_000.0)
    h = HistoryStore(str(path), now=clock)
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["legacy_q", "new_q"]
    # Legacy grandfathered.
    assert turns[0]["ts"] == 1_500_000.0
    assert turns[0]["run_id"] is None
    assert turns[0]["artifacts"] == []
    # Current turn preserved verbatim.
    assert turns[1]["ts"] == 2_000_000.0
    assert turns[1]["run_id"] == "run-9"
    assert turns[1]["artifacts"] == [{"entry_id": "e1"}]


def test_current_format_file_not_marked_dirty(tmp_path):
    """A file already in the current shape must not be rewritten on load (no
    needless churn / no ts drift)."""
    path = tmp_path / "h.json"
    turn = {
        "question": "q",
        "answer": "a",
        "ts": 2_000_000.0,
        "run_id": "run-1",
        "artifacts": [{"entry_id": "e1"}],
    }
    _write(path, {"k": [turn]})
    before = path.read_text(encoding="utf-8")
    clock = FakeClock(9_999_999.0)
    HistoryStore(str(path), now=clock)
    # ts must NOT have been re-stamped to 9_999_999.
    assert _read(path)["k"][0]["ts"] == 2_000_000.0
    # File unchanged (not flushed).
    assert path.read_text(encoding="utf-8") == before


def test_oversized_artifacts_capped_on_load(tmp_path):
    path = tmp_path / "h.json"
    arts = [{"entry_id": f"e{i}"} for i in range(MAX_ARTIFACTS_PER_TURN + 3)]
    _write(
        path,
        {"k": [{"question": "q", "answer": "a", "ts": 5.0, "run_id": None, "artifacts": arts}]},
    )
    h = HistoryStore(str(path))
    (turn,) = h.recent("k")
    assert len(turn["artifacts"]) == MAX_ARTIFACTS_PER_TURN
    # And the cap is persisted.
    assert len(_read(path)["k"][0]["artifacts"]) == MAX_ARTIFACTS_PER_TURN


# --- malformed inputs -------------------------------------------------------


def test_malformed_conversation_dropped(tmp_path):
    path = tmp_path / "h.json"
    _write(path, {"good": [{"question": "q", "answer": "a"}], "bad": "not-a-list"})
    h = HistoryStore(str(path))
    assert [t["question"] for t in h.recent("good")] == ["q"]
    assert h.recent("bad") == []


def test_malformed_turn_dropped(tmp_path):
    path = tmp_path / "h.json"
    _write(path, {"k": ["not-a-dict", {"question": "q", "answer": "a"}]})
    h = HistoryStore(str(path))
    turns = h.recent("k")
    assert [t["question"] for t in turns] == ["q"]


def test_bool_ts_is_restamped(tmp_path):
    """A stray boolean ts (int subclass) must be re-stamped, not read as
    1.0/0.0 epoch (which would prune the turn instantly)."""
    path = tmp_path / "h.json"
    _write(path, {"k": [{"question": "q", "answer": "a", "ts": True}]})
    clock = FakeClock(777.0)
    h = HistoryStore(str(path), now=clock)
    (turn,) = h.recent("k")
    assert turn["ts"] == 777.0


# --- persistence across "restart" -------------------------------------------


def test_reload_preserves_turns(tmp_path):
    path = tmp_path / "h.json"
    h1 = HistoryStore(str(path))
    h1.append("k", "q", "a", run_id="r1", artifacts=[{"entry_id": "e1"}])
    h2 = HistoryStore(str(path))  # fresh instance = restart
    (turn,) = h2.recent("k")
    assert turn["question"] == "q"
    assert turn["run_id"] == "r1"
    assert turn["artifacts"] == [{"entry_id": "e1"}]
