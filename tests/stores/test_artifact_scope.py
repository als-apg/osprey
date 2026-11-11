"""Tests for the one ownership predicate (``osprey.stores.artifact_scope``)."""

from __future__ import annotations

import dataclasses

import pytest

from osprey.audit.call import harness_session_id
from osprey.stores import session_ledger
from osprey.stores.artifact_scope import (
    CLI_POINTER_DIR,
    SessionScope,
    current_scope,
    owned_by,
    read_ledger,
    resolve_scope,
    scope_for,
)
from osprey.stores.artifact_store import (
    EXAMPLE_ORIGIN,
    WATCHDOG_ORIGIN,
    ArtifactEntry,
    owner_of,
)
from osprey.stores.session_ledger import EPOCH, Ledger, LedgerRow
from osprey_connectors import control_context, posture_store

KEY = "11111111-1111-4111-8111-111111111111"
OTHER_KEY = "22222222-2222-4222-8222-222222222222"
T1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"  # KEY's transcript before the /clear
T2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"  # KEY's transcript after the /clear
T_OTHER = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
CID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"  # a CLI conversation

BEFORE_CLEAR = "2026-11-05T09:00:00+00:00"
CLEAR_AT = "2026-11-05T10:00:00+00:00"
AFTER_CLEAR = "2026-11-05T11:00:00+00:00"

HUB_LEDGER = Ledger(
    rows=(
        LedgerRow(KEY, T1, EPOCH),
        LedgerRow(OTHER_KEY, T_OTHER, EPOCH),
        LedgerRow(KEY, T2, CLEAR_AT),
    )
)
WEB = SessionScope(kind="web", transcript=T2, ledger=HUB_LEDGER)
CLI = SessionScope(
    kind="cli", transcript=CID, ledger=Ledger(rows=(LedgerRow("cli:" + CID, CID, EPOCH),))
)


def _entry(
    *,
    session_id: str = KEY,
    timestamp: str = AFTER_CLEAR,
    run_id: str = "",
    origin: str = "",
    metadata: dict | None = None,
) -> ArtifactEntry:
    return ArtifactEntry(
        id="a1",
        artifact_type="json",
        title="t",
        description="",
        filename="f",
        mime_type="application/json",
        size_bytes=0,
        timestamp=timestamp,
        tool_source="",
        metadata=metadata or {},
        session_id=session_id,
        run_id=run_id,
        origin=origin,
    )


class TestSessionScope:
    def test_defaults(self):
        scope = SessionScope(kind="none", transcript=None, ledger=None)
        assert scope.reason == ""
        assert scope.ledger_ok is True

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            WEB.transcript = T1  # type: ignore[misc]


class TestOwnedByWeb:
    def test_entry_of_the_scopes_transcript_is_owned(self):
        assert owned_by(_entry(), WEB)

    def test_entry_saved_before_the_clear_is_not_owned(self):
        entry = _entry(timestamp=BEFORE_CLEAR)
        assert owner_of(entry, HUB_LEDGER) == T1
        assert not owned_by(entry, WEB)

    def test_first_ts_decides_over_timestamp(self):
        entry = _entry(timestamp=AFTER_CLEAR, metadata={"first_ts": BEFORE_CLEAR})
        assert not owned_by(entry, WEB)

    def test_another_keys_entry_is_not_owned(self):
        assert not owned_by(_entry(session_id=OTHER_KEY), WEB)

    def test_untagged_entry_is_not_owned(self):
        assert not owned_by(_entry(session_id=""), WEB)

    def test_unknown_key_is_not_owned(self):
        assert not owned_by(_entry(session_id="ffffffff-ffff-4fff-8fff-ffffffffffff"), WEB)

    def test_shipped_example_is_not_owned(self):
        assert not owned_by(_entry(session_id="", origin=EXAMPLE_ORIGIN), WEB)

    def test_watchdog_entry_is_not_owned(self):
        assert not owned_by(_entry(origin=WATCHDOG_ORIGIN), WEB)

    def test_dispatch_entry_is_not_owned(self):
        assert not owned_by(_entry(run_id="run-1"), WEB)

    def test_stamped_transcript_is_owned(self):
        entry = _entry(session_id="", metadata={"transcript_id": T2})
        assert owned_by(entry, WEB)


class TestUnresolvedScope:
    @pytest.mark.parametrize("ledger", [HUB_LEDGER, None])
    def test_no_transcript_owns_nothing(self, ledger):
        scope = SessionScope(kind="none", transcript=None, ledger=ledger, reason="no session")
        assert not owned_by(_entry(), scope)
        assert not owned_by(_entry(session_id=""), scope)
        assert not owned_by(_entry(metadata={"transcript_id": T2}), scope)

    def test_dispatch_scope_owns_no_run_entry(self):
        scope = SessionScope(kind="dispatch", transcript=None, ledger=None, reason="dispatch")
        assert not owned_by(_entry(session_id="", run_id="run-1"), scope)


class TestOwnedByCli:
    def test_cli_tagged_entry_is_owned_under_the_cli_scope(self):
        entry = _entry(session_id="cli:" + CID)
        assert owned_by(entry, CLI)

    def test_cli_tagged_entry_has_no_owner_under_the_hub_ledger(self):
        entry = _entry(session_id="cli:" + CID)
        assert owner_of(entry, HUB_LEDGER) is None
        assert not owned_by(entry, WEB)

    def test_web_entry_is_not_owned_under_the_cli_scope(self):
        assert not owned_by(_entry(), CLI)

    def test_other_cli_conversation_is_not_owned(self):
        assert not owned_by(_entry(session_id="cli:" + T_OTHER), CLI)


# ---- resolver ---------------------------------------------------------------

LAUNCH = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
HOST_TERMINAL = "chat-ffffffff-ffff-4fff-8fff-ffffffffffff"
CID_AFTER_CLEAR = "99999999-9999-4999-8999-999999999999"


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """A bound, empty state directory, so nothing reads the developer's ledger."""
    bound = tmp_path / "state"
    bound.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
    control_context.invalidate_cache()
    yield bound
    control_context.invalidate_cache()


@pytest.fixture
def root(tmp_path):
    return tmp_path / "data"


def _point(root, transcript: str, launch: str = LAUNCH) -> None:
    pointer = root / CLI_POINTER_DIR / launch
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(transcript + "\n", encoding="utf-8")


def _cli_env(**extra: str) -> dict[str, str]:
    return {"OSPREY_TERMINAL_SESSION": HOST_TERMINAL, "OSPREY_CLI_LAUNCH": LAUNCH, **extra}


class TestResolveScope:
    def test_a_key_with_a_ledger_row_is_its_current_transcript(self):
        scope = resolve_scope(key=KEY, ledger=HUB_LEDGER, ledger_ok=True)
        assert (scope.kind, scope.transcript, scope.reason) == ("web", T2, "")
        assert scope.ledger is HUB_LEDGER and scope.ledger_ok

    def test_a_key_without_a_ledger_row_is_unresolved(self):
        scope = resolve_scope(
            key="33333333-3333-4333-8333-333333333333", ledger=HUB_LEDGER, ledger_ok=True
        )
        assert (scope.kind, scope.transcript) == ("web", None)
        assert scope.reason and scope.ledger is HUB_LEDGER and scope.ledger_ok

    def test_an_unreadable_ledger_resolves_no_transcript(self):
        scope = resolve_scope(key=KEY, ledger=HUB_LEDGER, ledger_ok=False)
        assert (scope.kind, scope.transcript, scope.ledger_ok) == ("web", None, False)
        assert scope.reason and scope.ledger is HUB_LEDGER

    def test_dispatch_wins_over_a_key(self):
        scope = resolve_scope(key=KEY, ledger=HUB_LEDGER, ledger_ok=True, dispatch=True)
        assert (scope.kind, scope.transcript) == ("dispatch", None)
        assert scope.reason and scope.ledger is HUB_LEDGER

    def test_a_key_wins_over_a_cli_conversation(self):
        scope = resolve_scope(key=KEY, ledger=HUB_LEDGER, ledger_ok=True, cli_conversation=CID)
        assert (scope.kind, scope.transcript) == ("web", T2)

    def test_a_cli_scope_has_a_one_row_ledger_keyed_cli(self):
        scope = resolve_scope(key=None, ledger=HUB_LEDGER, ledger_ok=True, cli_conversation=CID)
        assert (scope.kind, scope.transcript, scope.reason) == ("cli", CID, "")
        assert tuple(scope.ledger.rows) == (LedgerRow("cli:" + CID, CID, EPOCH),)
        assert owned_by(_entry(session_id="cli:" + CID), scope)

    def test_nothing_named_is_none_with_a_reason(self):
        scope = resolve_scope(key="", ledger=HUB_LEDGER, ledger_ok=True, cli_conversation="")
        assert (scope.kind, scope.transcript) == ("none", None)
        assert scope.reason and scope.ledger is HUB_LEDGER and scope.ledger_ok


class TestScopeFor:
    def test_a_known_transcript_is_a_web_scope(self):
        scope = scope_for(T1, HUB_LEDGER, True)
        assert (scope.kind, scope.transcript, scope.reason) == ("web", T1, "")
        assert scope.ledger is HUB_LEDGER
        assert owned_by(_entry(timestamp=BEFORE_CLEAR), scope)

    def test_an_unknown_transcript_is_none(self):
        scope = scope_for(None, HUB_LEDGER, False)
        assert (scope.kind, scope.transcript, scope.ledger_ok) == ("none", None, False)
        assert scope.reason


class TestReadLedger:
    def test_reads_the_hubs_path(self, state_dir):
        session_ledger.append(state_dir, KEY, T1, EPOCH)
        ledger, ok = read_ledger()
        assert ok and ledger.current(KEY) == T1

    @pytest.mark.usefixtures("state_dir")
    def test_a_missing_ledger_is_readable_and_empty(self):
        ledger, ok = read_ledger()
        assert ok and tuple(ledger.rows) == ()

    def test_unavailable_when_state_dir_raises(self, monkeypatch):
        def _boom():
            raise RuntimeError("no identity")

        monkeypatch.setattr(posture_store, "state_dir", _boom)
        ledger, ok = read_ledger()
        assert not ok and tuple(ledger.rows) == ()

    def test_an_unreadable_ledger_is_unavailable(self, state_dir):
        (state_dir / session_ledger.STORE_FILENAME).write_text("{not json")
        _, ok = read_ledger()
        assert not ok


@pytest.mark.usefixtures("state_dir")
class TestCurrentScope:
    def test_a_web_key_reads_the_hub_ledger(self, state_dir, root):
        session_ledger.append(state_dir, KEY, T1, EPOCH)
        session_ledger.append(state_dir, KEY, T2, CLEAR_AT)
        scope = current_scope({"OSPREY_SESSION_ID": KEY}, root=root)
        assert (scope.kind, scope.transcript, scope.ledger_ok) == ("web", T2, True)

    def test_a_web_key_with_no_row_is_unresolved(self, root):
        scope = current_scope({"OSPREY_SESSION_ID": KEY}, root=root)
        assert (scope.kind, scope.transcript) == ("web", None) and scope.reason

    def test_a_web_key_over_an_unreadable_ledger_is_unavailable(self, state_dir, root):
        (state_dir / session_ledger.STORE_FILENAME).write_text("{not json")
        scope = current_scope({"OSPREY_SESSION_ID": KEY}, root=root)
        assert (scope.kind, scope.transcript, scope.ledger_ok) == ("web", None, False)

    def test_cli_reads_the_pointer(self, root):
        _point(root, CID)
        scope = current_scope(_cli_env(), root=root)
        assert (scope.kind, scope.transcript) == ("cli", CID)
        assert tuple(scope.ledger.rows) == (LedgerRow("cli:" + CID, CID, EPOCH),)

    def test_after_a_clear_the_pointer_beats_the_stale_harness_id(self, root):
        _point(root, CID)
        _point(root, CID_AFTER_CLEAR)  # the turn-state hook rewrote it on /clear
        scope = current_scope(_cli_env(CLAUDE_CODE_SESSION_ID=CID), root=root)
        assert (scope.kind, scope.transcript) == ("cli", CID_AFTER_CLEAR)

    def test_cli_without_a_pointer_falls_back_to_the_harness_id(self, root):
        scope = current_scope(_cli_env(CLAUDE_CODE_SESSION_ID=CID), root=root)
        assert (scope.kind, scope.transcript) == ("cli", CID)

    @pytest.mark.parametrize("content", ["", "\n  \n"])
    def test_an_empty_pointer_is_no_pointer(self, root, content):
        pointer = root / CLI_POINTER_DIR / LAUNCH
        pointer.parent.mkdir(parents=True)
        pointer.write_text(content)
        scope = current_scope(_cli_env(CLAUDE_CODE_SESSION_ID=CID), root=root)
        assert scope.transcript == CID

    def test_an_unreadable_pointer_is_no_pointer(self, root):
        (root / CLI_POINTER_DIR / LAUNCH).mkdir(parents=True)  # a directory, not a file
        scope = current_scope(_cli_env(CLAUDE_CODE_SESSION_ID=CID), root=root)
        assert scope.transcript == CID

    def test_an_unsafe_launch_name_reads_no_pointer(self, root, tmp_path):
        (tmp_path / "escape").write_text(T_OTHER)
        _point(root, CID)
        env = _cli_env(OSPREY_CLI_LAUNCH="../../escape")
        assert current_scope(env, root=root).kind == "none"

    def test_cli_with_neither_pointer_nor_harness_id_is_none(self, root):
        scope = current_scope(_cli_env(), root=root)
        assert (scope.kind, scope.transcript) == ("none", None) and scope.reason

    def test_another_launchs_pointer_is_not_read(self, root):
        _point(root, T_OTHER, launch="00000000-0000-4000-8000-000000000000")
        scope = current_scope(_cli_env(), root=root)
        assert scope.kind == "none"

    def test_a_cli_in_an_inherited_terminal_session_is_none(self, root):
        _point(root, CID)
        env = _cli_env(OSPREY_TERMINAL_SESSION="web-" + KEY, CLAUDE_CODE_SESSION_ID=CID)
        scope = current_scope(env, root=root)
        assert (scope.kind, scope.transcript) == ("none", None) and scope.reason

    def test_osprey_query_is_a_cli_launch(self, root):
        """``osprey query`` exports a host terminal session and a launch id like ``chat``."""
        _point(root, CID)
        env = _cli_env(OSPREY_TELEMETRY_SESSION_ID=CID)
        assert current_scope(env, root=root).transcript == CID

    @pytest.mark.parametrize(
        "env",
        [
            {"OSPREY_DISPATCH_RUN_ID": "run-1"},
            {"OSPREY_DISPATCH_RUN": "1"},
            {"OSPREY_DISPATCH_RUN": "1", "OSPREY_SESSION_ID": KEY},
            {"OSPREY_DISPATCH_RUN_ID": "run-1", **_cli_env(CLAUDE_CODE_SESSION_ID=CID)},
        ],
    )
    def test_dispatch(self, root, env):
        _point(root, CID)
        scope = current_scope(env, root=root)
        assert (scope.kind, scope.transcript) == ("dispatch", None) and scope.reason

    def test_a_dispatch_flag_other_than_one_is_not_dispatch(self, root):
        assert current_scope({"OSPREY_DISPATCH_RUN": "0"}, root=root).kind == "none"

    def test_a_watchdog_judge_is_none(self, root):
        _point(root, CID)
        env = {
            "OSPREY_SESSION_ID": "",
            "OSPREY_TERMINAL_SESSION": "watchdog-0123456789abcdef",
            "OSPREY_CLI_LAUNCH": LAUNCH,
            "CLAUDE_CODE_SESSION_ID": CID,
        }
        scope = current_scope(env, root=root)
        assert (scope.kind, scope.transcript) == ("none", None)

    def test_a_keyless_process_beside_a_newest_transcript_is_none(self, root, tmp_path):
        transcripts = tmp_path / "projects" / "repo"
        transcripts.mkdir(parents=True)
        (transcripts / f"{CID}.jsonl").write_text('{"type": "user"}\n')
        scope = current_scope({"CLAUDE_CODE_SESSION_ID": CID}, root=root)
        assert (scope.kind, scope.transcript) == ("none", None) and scope.reason

    def test_the_mapping_beats_os_environ(self, root, monkeypatch):
        monkeypatch.setenv("OSPREY_SESSION_ID", KEY)
        monkeypatch.setenv("OSPREY_TERMINAL_SESSION", HOST_TERMINAL)
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", T_OTHER)
        assert current_scope({}, root=root).kind == "none"
        scope = current_scope(_cli_env(CLAUDE_CODE_SESSION_ID=CID), root=root)
        assert (scope.kind, scope.transcript) == ("cli", CID)

    def test_none_reads_os_environ(self, root, monkeypatch):
        for name in ("OSPREY_SESSION_ID", "OSPREY_DISPATCH_RUN_ID", "OSPREY_DISPATCH_RUN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OSPREY_TERMINAL_SESSION", HOST_TERMINAL)
        monkeypatch.setenv("OSPREY_CLI_LAUNCH", LAUNCH)
        _point(root, CID)
        assert current_scope(root=root).transcript == CID


class TestHarnessSessionIdEnviron:
    def test_the_mapping_beats_os_environ(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "from-os")
        monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "forced-os")
        assert harness_session_id({}) is None
        assert harness_session_id({"CLAUDE_CODE_SESSION_ID": "mapped"}) == "mapped"
        mapped = {"OSPREY_TELEMETRY_SESSION_ID": "", "CLAUDE_CODE_SESSION_ID": "mapped"}
        assert harness_session_id(mapped) == "mapped"
        assert harness_session_id() == "forced-os"
