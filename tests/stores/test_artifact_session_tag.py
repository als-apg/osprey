"""The session tag a save defaults to (contract C2, ``session_tag``).

Web saves carry the key, CLI saves ``cli:<transcript>`` from the launch's
pointer (so a ``/clear`` moves the tag), and dispatch, watchdog and keyless
processes save untagged. The tag reads the environment and the pointer only.
"""

from __future__ import annotations

import pytest

from osprey.stores import artifact_scope
from osprey.stores.artifact_scope import session_tag
from osprey.stores.artifact_store import ArtifactStore

KEY = "11111111-1111-4111-8111-111111111111"
LAUNCH = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
HOST_TERMINAL = "chat-ffffffff-ffff-4fff-8fff-ffffffffffff"
CID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
CID_NEW = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"

#: Every variable the tag reads; cleared so the developer's own shell never leaks in.
_SCOPE_ENV = (
    "OSPREY_SESSION_ID",
    "OSPREY_DISPATCH_RUN_ID",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_CLI_LAUNCH",
    "OSPREY_TELEMETRY_SESSION_ID",
    "CLAUDE_CODE_SESSION_ID",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "_agent_data"
    path.mkdir()
    return path


@pytest.fixture
def store(root):
    return ArtifactStore(workspace_root=root, auto_launch=False)


def _point(root, transcript):
    pointer = root / artifact_scope.CLI_POINTER_DIR / LAUNCH
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(transcript + "\n", encoding="utf-8")


def _cli_env(**extra):
    env = {
        "OSPREY_TERMINAL_SESSION": HOST_TERMINAL,
        "OSPREY_CLI_LAUNCH": LAUNCH,
        "CLAUDE_CODE_SESSION_ID": CID,
    }
    env.update(extra)
    return env


def _use_cli(monkeypatch, **extra):
    for name, value in _cli_env(**extra).items():
        monkeypatch.setenv(name, value)


class TestSessionTagByKind:
    def test_web_is_the_key(self, root):
        assert session_tag({"OSPREY_SESSION_ID": KEY}, root=root) == KEY

    def test_cli_is_the_pointer_transcript(self, root):
        _point(root, CID_NEW)
        assert session_tag(_cli_env(), root=root) == "cli:" + CID_NEW

    def test_cli_without_a_pointer_falls_back_to_the_harness_id(self, root):
        assert session_tag(_cli_env(), root=root) == "cli:" + CID

    def test_dispatch_run_id_is_untagged(self, root):
        env = {"OSPREY_SESSION_ID": KEY, "OSPREY_DISPATCH_RUN_ID": "run-1"}
        assert session_tag(env, root=root) == ""

    def test_headless_dispatch_flag_is_untagged(self, root):
        assert session_tag({"OSPREY_DISPATCH_RUN": "1", **_cli_env()}, root=root) == ""

    def test_watchdog_judge_is_untagged(self, root):
        _point(root, CID)
        env = _cli_env(OSPREY_TERMINAL_SESSION="watchdog-0123456789abcdef")
        assert session_tag(env, root=root) == ""

    def test_keyless_process_is_untagged(self, root):
        assert session_tag({"CLAUDE_CODE_SESSION_ID": CID}, root=root) == ""

    def test_inherited_terminal_session_is_untagged(self, root):
        _point(root, CID)
        assert session_tag(_cli_env(OSPREY_TERMINAL_SESSION="tmux-1"), root=root) == ""

    def test_the_mapping_beats_os_environ(self, root, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        assert session_tag({}, root=root) == ""

    def test_defaults_to_os_environ(self, root, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        assert session_tag(root=root) == KEY

    def test_never_reads_the_ledger(self, root, monkeypatch):
        def _boom():
            raise AssertionError("session_tag read the ledger")

        monkeypatch.setattr(artifact_scope, "read_ledger", _boom)
        assert session_tag({"OSPREY_SESSION_ID": KEY}, root=root) == KEY


class TestStoreDefaults:
    def test_web_saves_carry_the_key(self, store, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        assert store.save_file(b"x", "a.txt", "text", "t").session_id == KEY
        assert store.save_data("tool", {"a": 1}, "t").session_id == KEY

    def test_cli_saves_carry_the_cli_tag(self, store, root, monkeypatch):
        _point(root, CID)
        _use_cli(monkeypatch)
        assert store.save_file(b"x", "a.txt", "text", "t").session_id == "cli:" + CID
        assert store.save_data("tool", {"a": 1}, "t").session_id == "cli:" + CID

    def test_channel_reading_carries_the_cli_tag(self, store, root, monkeypatch):
        _point(root, CID)
        _use_cli(monkeypatch)
        entry = store.save_channel_reading(
            b"\x89PNG\r\n\x1a\n", b"npy", "reading", metadata={"channel": "SR:DCCT"}
        )
        assert entry.session_id == "cli:" + CID

    def test_after_a_clear_the_executor_result_carries_the_new_tag(self, store, root, monkeypatch):
        """The parent MCP process keeps its first harness id; the pointer moves."""
        _point(root, CID)
        _use_cli(monkeypatch)
        before = store.save_data("execute", {"r": 1}, "before")
        _point(root, CID_NEW)  # the turn-state hook rewrites it on /clear
        after = store.save_data("execute", {"r": 2}, "after")
        assert before.session_id == "cli:" + CID
        assert after.session_id == "cli:" + CID_NEW

    def test_after_the_pointer_moves_a_file_save_carries_the_new_id(self, store, root, monkeypatch):
        _point(root, CID)
        _use_cli(monkeypatch)
        store.save_file(b"1", "a.txt", "text", "first")
        _point(root, CID_NEW)
        assert store.save_file(b"2", "b.txt", "text", "second").session_id == "cli:" + CID_NEW

    def test_store_reads_its_own_workspace_pointer(self, root, tmp_path, monkeypatch):
        """A pointer under another root is not this store's."""
        other = tmp_path / "other"
        other.mkdir()
        _point(other, CID_NEW)
        _point(root, CID)
        _use_cli(monkeypatch)
        store = ArtifactStore(workspace_root=root, auto_launch=False)
        assert store.save_file(b"x", "a.txt", "text", "t").session_id == "cli:" + CID

    def test_sha256_save_is_tagged(self, store, root, monkeypatch):
        _point(root, CID)
        _use_cli(monkeypatch)
        from osprey.stores.artifact_scope import current_scope

        entry = store.save_or_touch_by_sha256(
            "abc",
            origin="",
            save_kwargs={
                "file_content": b"pic",
                "filename": "p.png",
                "artifact_type": "image",
                "title": "p",
                "metadata": {"sha256": "abc"},
            },
            scope=current_scope(root=root),
        )
        assert entry.session_id == "cli:" + CID

    def test_explicit_empty_session_id_stays_untagged(self, store, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        entry = store.save_file(b"x", "a.txt", "text", "t", session_id="")
        assert entry.session_id == ""

    def test_explicit_session_id_wins(self, store, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        other = "22222222-2222-4222-8222-222222222222"
        assert store.save_file(b"x", "a.txt", "text", "t", session_id=other).session_id == other

    def test_dispatch_saves_are_untagged(self, store, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "run-1")
        entry = store.save_data("tool", {"a": 1}, "t")
        assert (entry.session_id, entry.run_id) == ("", "run-1")

    def test_keyless_saves_are_untagged(self, store):
        assert store.save_file(b"x", "a.txt", "text", "t").session_id == ""

    def test_no_transcript_id_stamp(self, store, root, monkeypatch):
        _point(root, CID)
        _use_cli(monkeypatch)
        entry = store.save_file(b"x", "a.txt", "text", "t")
        assert "transcript_id" not in entry.metadata


def test_backfill_adopts_no_cli_tag(tmp_path):
    from osprey.interfaces.web_terminal.session_meta import backfill_rows

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / f"{CID}.jsonl").write_text('{"type": "user"}\n')
    (sessions / f"cli:{CID}.jsonl").write_text('{"type": "user"}\n')
    assert backfill_rows({}, sessions, ["cli:" + CID]) == []
