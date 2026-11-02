"""The session hold record: one durable file per session that holds its writes.

:mod:`osprey_connectors.session_hold` owns the record's format, its reader and
its writer. The properties pinned here are the ones a write path relies on: the
reader never fails open under a root (a missing directory, a missing record and
a corrupt record all hold), nothing is ever held without a root, an answer
releases exactly the flags it names, and the cross-process lock keeps two
concurrent writers from losing each other's flag.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey_connectors import session_hold
from osprey_connectors.session_hold import HoldFlag
from osprey_connectors_tests._paths import PACKAGE_SRC

FIXTURES = Path(__file__).parent / "fixtures" / "session_hold"

KEY = "3f2b8c1e-6a0d-4c55-9a1e-0b7f2d4e9c11"  # gitleaks:allow - a test session id


@pytest.fixture(autouse=True)
def _no_ambient_root(monkeypatch):
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "session_holds"
    path.mkdir()
    return path


def _open(root: Path, flag_id: str, *, source="watchdog:a", prompt_ref=None, key=KEY) -> bool:
    return session_hold.open_flag(
        key,
        flag_id=flag_id,
        source=source,
        reason=f"reason for {flag_id}",
        prompt_ref=prompt_ref or f"ref-{flag_id}",
        prompt_excerpt=f"prompt for {flag_id}",
        root=root,
    )


def _record_path(root: Path, key: str = KEY) -> Path:
    return root / key / session_hold.RECORD_NAME


def _payload(root: Path, key: str = KEY) -> dict:
    return json.loads(_record_path(root, key).read_text(encoding="utf-8"))


# --- no root: nothing is held -------------------------------------------------


def test_no_root_is_never_held_and_open_flags_is_empty():
    assert session_hold.is_held(KEY) is False
    assert session_hold.open_flags(KEY) == []


def test_an_empty_env_root_is_no_store(monkeypatch):
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, "")
    assert session_hold.is_held(KEY) is False
    assert session_hold.open_flags(KEY) == []


def test_no_root_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_hold.ensure_session(KEY)
    assert (
        session_hold.open_flag(
            KEY, flag_id="f1", source="s", reason="r", prompt_ref="p", prompt_excerpt="e"
        )
        is False
    )
    assert list(tmp_path.iterdir()) == []


def test_the_env_root_is_used_when_no_argument(root, monkeypatch):
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(root))
    session_hold.ensure_session(KEY)
    assert session_hold.is_held(KEY) is False
    assert _open(None, "f1") is True  # root=None -> env
    assert session_hold.is_held(KEY) is True
    assert _record_path(root).is_file()


def test_an_argument_root_wins_over_the_env(root, tmp_path, monkeypatch):
    other = tmp_path / "other"
    monkeypatch.setenv(session_hold.HOLD_DIR_ENV_VAR, str(other))
    session_hold.ensure_session(KEY, root=root)
    assert _record_path(root).is_file()
    assert not other.exists()


# --- ensure_session -------------------------------------------------------------


def test_ensure_session_creates_an_empty_unheld_record(root):
    session_hold.ensure_session(KEY, root=root)
    payload = _payload(root)
    assert payload["schema"] == session_hold.SCHEMA_VERSION
    assert payload["session_key"] == KEY
    assert payload["flags"] == []
    assert session_hold.is_held(KEY, root=root) is False
    assert session_hold.open_flags(KEY, root=root) == []


def test_ensure_session_over_an_existing_record_with_open_flags_changes_nothing(root):
    session_hold.ensure_session(KEY, root=root)
    assert _open(root, "f1")
    before = _record_path(root).read_bytes()
    session_hold.ensure_session(KEY, root=root)
    assert _record_path(root).read_bytes() == before
    assert [f.flag_id for f in session_hold.open_flags(KEY, root=root)] == ["f1"]


def test_ensure_session_over_a_corrupt_record_changes_nothing(root):
    (root / KEY).mkdir()
    _record_path(root).write_text("{not json", encoding="utf-8")
    session_hold.ensure_session(KEY, root=root)
    assert _record_path(root).read_text(encoding="utf-8") == "{not json"
    assert session_hold.is_held(KEY, root=root) is True


# --- open / answer --------------------------------------------------------------


def test_open_then_answer_by_flag_ids_releases(root):
    session_hold.ensure_session(KEY, root=root)
    assert _open(root, "f1")
    assert session_hold.is_held(KEY, root=root) is True
    [flag] = session_hold.open_flags(KEY, root=root)
    assert isinstance(flag, HoldFlag)
    assert (flag.flag_id, flag.source, flag.reason, flag.prompt_ref) == (
        "f1",
        "watchdog:a",
        "reason for f1",
        "ref-f1",
    )
    assert flag.prompt_excerpt == "prompt for f1"
    assert flag.opened_at
    assert flag.stop_unconfirmed is False
    assert flag.started == ()

    released = session_hold.answer_flags(
        KEY, ["f1"], answer="continue", operator="alice", root=root
    )
    assert released == ("f1",)
    assert session_hold.is_held(KEY, root=root) is False
    assert session_hold.open_flags(KEY, root=root) == []


def test_answer_accepts_the_list_open_flags_returned(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    _open(root, "f2")
    ids = [f.flag_id for f in session_hold.open_flags(KEY, root=root)]
    released = session_hold.answer_flags(KEY, ids, answer="stop", operator="bob", root=root)
    assert sorted(released) == ["f1", "f2"]
    assert session_hold.is_held(KEY, root=root) is False


def test_only_named_ids_are_released(root):
    session_hold.ensure_session(KEY, root=root)
    for fid in ("f1", "f2", "f3"):
        _open(root, fid)
    released = session_hold.answer_flags(
        KEY, ("f2", "nope"), answer="continue", operator="alice", root=root
    )
    assert released == ("f2",)
    assert [f.flag_id for f in session_hold.open_flags(KEY, root=root)] == ["f1", "f3"]
    assert session_hold.is_held(KEY, root=root) is True


def test_answers_record_operator_and_time_and_closed_flags_stay(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    session_hold.answer_flags(KEY, ["f1"], answer="continue", operator="alice", root=root)
    [entry] = _payload(root)["flags"]
    assert entry["flag_id"] == "f1"
    assert entry["state"] == "closed"
    assert entry["answer"] == "continue"
    assert entry["operator"] == "alice"
    assert entry["answered_at"]


def test_answering_a_closed_id_again_releases_nothing_and_keeps_the_first_answer(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    session_hold.answer_flags(KEY, ["f1"], answer="continue", operator="alice", root=root)
    again = session_hold.answer_flags(KEY, ["f1"], answer="stop", operator="bob", root=root)
    assert again == ()
    [entry] = _payload(root)["flags"]
    assert (entry["answer"], entry["operator"]) == ("continue", "alice")


def test_an_unknown_answer_is_refused(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    with pytest.raises(ValueError):
        session_hold.answer_flags(KEY, ["f1"], answer="maybe", operator="a", root=root)
    assert session_hold.is_held(KEY, root=root) is True


def test_a_bare_string_of_ids_is_refused(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    with pytest.raises(TypeError):
        session_hold.answer_flags(KEY, "f1", answer="continue", operator="a", root=root)


def test_duplicate_source_and_prompt_ref_opens_once(root):
    session_hold.ensure_session(KEY, root=root)
    assert _open(root, "f1", source="watchdog:a", prompt_ref="p1") is True
    assert _open(root, "f2", source="watchdog:a", prompt_ref="p1") is False
    assert [f.flag_id for f in session_hold.open_flags(KEY, root=root)] == ["f1"]
    # A different source on the same prompt is a second flag.
    assert _open(root, "f3", source="watchdog:b", prompt_ref="p1") is True
    assert len(session_hold.open_flags(KEY, root=root)) == 2


def test_duplicate_is_checked_against_closed_flags_too(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1", prompt_ref="p1")
    session_hold.answer_flags(KEY, ["f1"], answer="continue", operator="a", root=root)
    assert _open(root, "f2", prompt_ref="p1") is False
    assert session_hold.is_held(KEY, root=root) is False


def test_the_excerpt_is_truncated_to_200_chars(root):
    session_hold.ensure_session(KEY, root=root)
    session_hold.open_flag(
        KEY,
        flag_id="f1",
        source="s",
        reason="r",
        prompt_ref="p",
        prompt_excerpt="x" * 5000,
        root=root,
    )
    [flag] = session_hold.open_flags(KEY, root=root)
    assert len(flag.prompt_excerpt) <= session_hold.EXCERPT_LIMIT == 200


def test_open_flag_without_ensure_creates_the_record_and_holds(root):
    assert _open(root, "f1") is True
    assert session_hold.is_held(KEY, root=root) is True


# --- stop unconfirmed ---------------------------------------------------------


def test_stop_unconfirmed_stays_open_names_started_and_skips_closed(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    _open(root, "f2")
    session_hold.answer_flags(KEY, ["f2"], answer="continue", operator="a", root=root)
    marked = session_hold.mark_stop_unconfirmed(
        KEY, ["f1", "f2", "nope"], operator="alice", started=("caput X 1",), root=root
    )
    assert marked == ("f1",)
    [flag] = session_hold.open_flags(KEY, root=root)
    assert flag.flag_id == "f1"
    assert flag.stop_unconfirmed is True
    assert flag.started == ("caput X 1",)
    assert session_hold.is_held(KEY, root=root) is True
    closed = next(e for e in _payload(root)["flags"] if e["flag_id"] == "f2")
    assert closed["stop_unconfirmed"] is False
    # A later Continue still releases it.
    assert session_hold.answer_flags(
        KEY, ["f1"], answer="continue", operator="alice", root=root
    ) == ("f1",)
    assert session_hold.is_held(KEY, root=root) is False


# --- fail closed under a root ---------------------------------------------------


def test_a_corrupt_record_holds_and_answering_its_synthetic_flag_rewrites_it(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    _record_path(root).write_text("garbage", encoding="utf-8")
    assert session_hold.is_held(KEY, root=root) is True
    [flag] = session_hold.open_flags(KEY, root=root)
    assert flag.flag_id == session_hold.UNREADABLE_FLAG_ID == "hold-record-unreadable"
    # Real ids cannot be answered against an unreadable record.
    assert session_hold.answer_flags(KEY, ["f1"], answer="continue", operator="a", root=root) == ()
    assert session_hold.is_held(KEY, root=root) is True

    released = session_hold.answer_flags(
        KEY, [session_hold.UNREADABLE_FLAG_ID], answer="continue", operator="alice", root=root
    )
    assert released == (session_hold.UNREADABLE_FLAG_ID,)
    payload = _payload(root)
    assert payload["schema"] == session_hold.SCHEMA_VERSION
    assert session_hold.is_held(KEY, root=root) is False
    [entry] = payload["flags"]
    assert entry["flag_id"] == session_hold.UNREADABLE_FLAG_ID
    assert (entry["state"], entry["operator"]) == ("closed", "alice")


def test_the_synthetic_id_on_a_readable_record_is_ignored(root):
    session_hold.ensure_session(KEY, root=root)
    _open(root, "f1")
    before = _record_path(root).read_bytes()
    assert (
        session_hold.answer_flags(
            KEY, [session_hold.UNREADABLE_FLAG_ID], answer="continue", operator="a", root=root
        )
        == ()
    )
    assert _record_path(root).read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": 99, "session_key": KEY, "flags": []},
        {"session_key": KEY, "flags": []},
        {"schema": 1, "session_key": "someone-else", "flags": []},
        {"schema": 1, "session_key": KEY, "flags": [{"flag_id": "f1"}]},
        {"schema": 1, "session_key": KEY, "flags": "nope"},
        [],
    ],
    ids=["other-schema", "no-schema", "other-key", "bad-flag", "bad-flags", "not-a-mapping"],
)
def test_a_malformed_record_holds(root, payload):
    (root / KEY).mkdir()
    _record_path(root).write_text(json.dumps(payload), encoding="utf-8")
    assert session_hold.is_held(KEY, root=root) is True
    assert [f.flag_id for f in session_hold.open_flags(KEY, root=root)] == [
        session_hold.UNREADABLE_FLAG_ID
    ]


def test_an_oversized_record_holds(root):
    session_hold.ensure_session(KEY, root=root)
    _record_path(root).write_text(" " * (session_hold._RECORD_READ_LIMIT + 10), encoding="utf-8")
    assert session_hold.is_held(KEY, root=root) is True


def test_a_removed_record_holds_under_a_root(root):
    session_hold.ensure_session(KEY, root=root)
    _record_path(root).unlink()
    assert session_hold.is_held(KEY, root=root) is True
    assert [f.flag_id for f in session_hold.open_flags(KEY, root=root)] == [
        session_hold.UNREADABLE_FLAG_ID
    ]


def test_a_removed_dir_holds_under_a_root_and_its_answer_recreates_it(root):
    session_hold.ensure_session(KEY, root=root)
    shutil.rmtree(root / KEY)
    assert session_hold.is_held(KEY, root=root) is True
    assert session_hold.answer_flags(
        KEY, [session_hold.UNREADABLE_FLAG_ID], answer="stop", operator="a", root=root
    ) == (session_hold.UNREADABLE_FLAG_ID,)
    assert session_hold.is_held(KEY, root=root) is False


def test_a_never_created_session_holds_under_a_root(root):
    assert session_hold.is_held(KEY, root=root) is True


def test_a_missing_root_holds(tmp_path):
    assert session_hold.is_held(KEY, root=tmp_path / "nowhere") is True


def test_a_symlinked_record_holds_and_is_not_followed(root, tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"schema": 1, "session_key": KEY, "flags": []}), encoding="utf-8")
    (root / KEY).mkdir()
    _record_path(root).symlink_to(good)
    assert session_hold.is_held(KEY, root=root) is True


def test_a_symlinked_session_dir_holds(root, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / session_hold.RECORD_NAME).write_text(
        json.dumps({"schema": 1, "session_key": KEY, "flags": []}), encoding="utf-8"
    )
    (root / KEY).symlink_to(elsewhere, target_is_directory=True)
    assert session_hold.is_held(KEY, root=root) is True
    with pytest.raises(OSError):
        session_hold.ensure_session(KEY, root=root)


def test_is_held_never_raises(root, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(session_hold, "_read_record", boom)
    assert session_hold.is_held(KEY, root=root) is True


# --- session keys -----------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "../escape", "a/b", "a\\b", "a\x00b", "-lead", "x" * 200]
)
def test_an_unsafe_key_is_rejected(root, bad):
    with pytest.raises(ValueError):
        session_hold.ensure_session(bad, root=root)
    with pytest.raises(ValueError):
        session_hold.open_flag(
            bad, flag_id="f", source="s", reason="r", prompt_ref="p", prompt_excerpt="e", root=root
        )
    with pytest.raises(ValueError):
        session_hold.answer_flags(bad, ["f"], answer="continue", operator="a", root=root)
    with pytest.raises(ValueError):
        session_hold.mark_stop_unconfirmed(bad, ["f"], operator="a", started=(), root=root)
    with pytest.raises(ValueError):
        session_hold.open_flags(bad, root=root)
    # The reader never raises; under a root an unsafe key holds.
    assert session_hold.is_held(bad, root=root) is True
    assert session_hold.is_held(bad) is False
    assert list(root.iterdir()) == []


def test_a_uuid_key_is_safe():
    assert session_hold.is_safe_session_key(KEY)


# --- concurrency -----------------------------------------------------------------


_CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from osprey_connectors import session_hold

    root, key, prefix, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    for i in range(n):
        assert session_hold.open_flag(
            key, flag_id=f"{prefix}-{i}", source=prefix, reason="r",
            prompt_ref=f"p-{i}", prompt_excerpt="e", root=root,
        )
    """
)


def test_concurrent_opens_from_two_processes_both_survive(root):
    session_hold.ensure_session(KEY, root=root)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PACKAGE_SRC), env.get("PYTHONPATH")]))
    env.pop(session_hold.HOLD_DIR_ENV_VAR, None)
    n = 25
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(root), KEY, prefix, str(n)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for prefix in ("a", "b")
    ]
    for proc in procs:
        _out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
    ids = {f.flag_id for f in session_hold.open_flags(KEY, root=root)}
    assert ids == {f"{p}-{i}" for p in ("a", "b") for i in range(n)}


# --- shared parity fixtures -------------------------------------------------------


def _cases() -> dict:
    return json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))["cases"]


def test_the_parity_fixtures_name_every_case():
    assert set(_cases()) == {"open", "closed", "corrupt", "dir-only", "absent", "unsafe-key"}


@pytest.mark.parametrize("case", sorted(json.loads((FIXTURES / "cases.json").read_text())["cases"]))
def test_the_package_agrees_with_every_parity_fixture(case, tmp_path):
    spec = _cases()[case]
    # Read a copy: the fixture tree is shared and must never be written.
    root = tmp_path / "root"
    shutil.copytree(FIXTURES / case / "root", root)
    key = spec["session_key"]
    assert session_hold.is_held(key, root=root) is spec["held"]
    assert session_hold.is_held(key) is False
    if spec["open_flag_ids"] is None:
        with pytest.raises(ValueError):
            session_hold.open_flags(key, root=root)
    else:
        assert [f.flag_id for f in session_hold.open_flags(key, root=root)] == spec["open_flag_ids"]


def test_only_a_session_with_nothing_on_disk_has_no_record(root):
    assert session_hold.has_no_record(KEY, root=root) is True
    assert session_hold.is_held(KEY, root=root) is True

    (root / KEY).mkdir(parents=True)
    assert session_hold.has_no_record(KEY, root=root) is False
    assert session_hold.is_held(KEY, root=root) is True


def test_a_dangling_link_at_the_session_is_not_an_absent_record(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / KEY).symlink_to(root / "nowhere")

    assert session_hold.has_no_record(KEY, root=root) is False


def test_has_no_record_refuses_an_unsafe_key(root):
    with pytest.raises(ValueError):
        session_hold.has_no_record("../x", root=root)
