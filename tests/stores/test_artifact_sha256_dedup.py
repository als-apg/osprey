"""``save_or_touch_by_sha256``: one artifact per picture, and no hidden origin.

A caller that may see the same bytes many times (a logbook picture attached to
a session twice) saves them once: the second sighting refreshes the existing
entry's timestamp. No ``origin`` is hidden from a listing — the store keeps no
audit spills, so every entry it holds is something a run produced.

Reuse is per scope: a web or CLI save reuses only a match its scope owns, a
keyless save only an untagged operator-visible match, and a dispatch save never.
"""

from __future__ import annotations

import inspect
import threading
import time
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from osprey.stores import artifact_store
from osprey.stores.artifact_scope import SessionScope, owned_by, resolve_scope
from osprey.stores.artifact_store import (
    ArtifactStore,
    register_artifact_listener,
    unregister_artifact_listener,
)
from osprey.stores.session_ledger import EPOCH, Ledger, LedgerRow

_ORIGIN = "logbook"
_KEYLESS = resolve_scope(key=None, ledger=None, ledger_ok=True)


@pytest.fixture(autouse=True)
def _no_session_env(monkeypatch):
    for key in ("OSPREY_ARTIFACT_ORIGIN", "OSPREY_SESSION_ID", "OSPREY_DISPATCH_RUN_ID"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def store(agent_data_root) -> ArtifactStore:
    return ArtifactStore(workspace_root=agent_data_root, auto_launch=False)


def _picture(
    store: ArtifactStore,
    sha: str = "e" * 64,
    data: bytes = b"\x89PNG one",
    origin: str = _ORIGIN,
    *,
    scope: SessionScope = _KEYLESS,
    session_id: str | None = None,
    extra: dict | None = None,
):
    save_kwargs = {
        "file_content": data,
        "filename": "picture-0.png",
        "artifact_type": "image",
        "title": "logbook picture",
        "mime_type": "image/png",
        "tool_source": "attachment_to_artifact",
        "metadata": {"sha256": sha, **(extra or {})},
    }
    if session_id is not None:
        save_kwargs["session_id"] = session_id
    return store.save_or_touch_by_sha256(sha, origin=origin, save_kwargs=save_kwargs, scope=scope)


def _produced(store: ArtifactStore, title: str = "Beam profile"):
    return store.save_file(
        file_content=b"\x89PNG produced",
        filename="beam.png",
        artifact_type="image",
        title=title,
        mime_type="image/png",
        tool_source="execute",
    )


def test_the_audit_spill_origin_is_gone() -> None:
    assert not hasattr(artifact_store, "AUDIT_ORIGIN")
    assert "include_audit" not in inspect.signature(ArtifactStore.list_entries).parameters


def test_an_entry_of_the_old_spill_origin_is_listed_like_any_other(store) -> None:
    """``tool_call`` is no longer a reserved origin: nothing is filtered on it."""
    old = _picture(store, origin="tool_call")
    produced = _produced(store)

    assert {e.id for e in store.list_entries()} == {old.id, produced.id}
    assert {e.id for e in store.page_entries(limit=10).entries} == {old.id, produced.id}


def test_two_views_of_one_picture_are_one_artifact_with_a_refreshed_timestamp(store) -> None:
    first = _picture(store)
    stamp = first.timestamp
    time.sleep(0.01)

    second = _picture(store)

    assert second.id == first.id
    pictures = [e for e in store.list_entries() if e.origin == _ORIGIN]
    assert len(pictures) == 1
    assert pictures[0].timestamp > stamp
    # The refresh is on disk, not only in this process's copy.
    reread = ArtifactStore(workspace_root=store._workspace, auto_launch=False)
    assert reread.get_entry(first.id).timestamp == pictures[0].timestamp


def test_different_pictures_are_different_artifacts(store) -> None:
    a = _picture(store, "1" * 64, b"one")
    b = _picture(store, "2" * 64, b"two")
    assert a.id != b.id


def test_an_entry_of_another_origin_with_the_same_sha256_is_not_reused(store) -> None:
    other = store.save_file(
        file_content=b"x",
        filename="x.png",
        artifact_type="image",
        title="tool output",
        metadata={"sha256": "e" * 64},
    )
    picture = _picture(store)
    assert picture.id != other.id


def test_two_threads_saving_the_same_bytes_give_one_artifact(store) -> None:
    barrier = threading.Barrier(2)
    ids: list[str] = []

    def save() -> None:
        own = ArtifactStore(workspace_root=store._workspace, auto_launch=False)
        barrier.wait()
        ids.append(_picture(own).id)

    threads = [threading.Thread(target=save) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)

    assert len(ids) == 2 and ids[0] == ids[1]
    reread = ArtifactStore(workspace_root=store._workspace, auto_launch=False)
    assert len(reread.list_entries()) == 1


def test_listeners_and_the_gallery_fire_once_for_a_new_picture(tmp_path) -> None:
    seen = []

    def listener(entry) -> None:
        seen.append(entry.id)

    store = ArtifactStore(workspace_root=tmp_path, auto_launch=True)
    register_artifact_listener(listener)
    try:
        with patch.object(ArtifactStore, "_launch_gallery") as launch:
            first = _picture(store)
            _picture(store)
        assert seen == [first.id]
        launch.assert_called_once()
    finally:
        unregister_artifact_listener(listener)


def test_save_file_still_writes_and_notifies(store) -> None:
    seen = []
    register_artifact_listener(seen.append)
    try:
        entry = _produced(store)
    finally:
        unregister_artifact_listener(seen.append)
    assert seen == [entry]
    assert store.get_file_path(entry.id).read_bytes() == b"\x89PNG produced"


def test_scope_is_a_required_keyword() -> None:
    param = inspect.signature(ArtifactStore.save_or_touch_by_sha256).parameters["scope"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


# ---- reuse per scope (contract C4) -----------------------------------------

_KEY_A, _KEY_B = "key-a", "key-b"
_T_A, _T_B = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"


def _two_sessions() -> Ledger:
    return Ledger(rows=(LedgerRow(_KEY_A, _T_A, EPOCH), LedgerRow(_KEY_B, _T_B, EPOCH)))


def _web(key: str, ledger: Ledger, ledger_ok: bool = True) -> SessionScope:
    return resolve_scope(key=key, ledger=ledger, ledger_ok=ledger_ok)


def _shas(store: ArtifactStore, sha: str = "e" * 64) -> list:
    return [e for e in store.list_entries() if e.metadata.get("sha256") == sha]


class TestReuseIsPerSession:
    def test_another_transcripts_match_gives_a_new_entry_owned_by_this_one(self, store):
        """Req 12: bytes another transcript owns are saved again for this one."""
        ledger = _two_sessions()
        a, b = _web(_KEY_A, ledger), _web(_KEY_B, ledger)
        first = _picture(store, scope=a, session_id=_KEY_A)

        second = _picture(store, scope=b, session_id=_KEY_B)

        assert second.id != first.id
        assert owned_by(second, b) and not owned_by(second, a)
        assert owned_by(first, a)

    def test_a_same_transcript_match_is_reused(self, store):
        ledger = _two_sessions()
        a = _web(_KEY_A, ledger)
        first = _picture(store, scope=a, session_id=_KEY_A)
        stamp = first.timestamp
        time.sleep(0.01)

        again = _picture(store, scope=a, session_id=_KEY_A)

        assert again.id == first.id
        assert again.timestamp > stamp
        assert len(_shas(store)) == 1

    def test_a_save_after_a_clear_does_not_reuse_the_earlier_transcripts_entry(self, store):
        """One key, two transcripts: the new transcript owns nothing saved before it."""
        early = Ledger(rows=(LedgerRow(_KEY_A, _T_A, EPOCH),))
        first = _picture(store, scope=_web(_KEY_A, early), session_id=_KEY_A)
        time.sleep(0.01)
        moved = Ledger(rows=(*early.rows, LedgerRow(_KEY_A, _T_B, _now())))

        second = _picture(store, scope=_web(_KEY_A, moved), session_id=_KEY_A)

        assert second.id != first.id

    def test_a_then_b_then_b_again_reuses_bs_entry_and_holds_two(self, store):
        ledger = _two_sessions()
        a, b = _web(_KEY_A, ledger), _web(_KEY_B, ledger)
        from_a = _picture(store, scope=a, session_id=_KEY_A)
        from_b = _picture(store, scope=b, session_id=_KEY_B)

        again = _picture(store, scope=b, session_id=_KEY_B)

        assert from_b.id != from_a.id
        assert again.id == from_b.id
        assert {e.id for e in _shas(store)} == {from_a.id, from_b.id}

    def test_a_touch_refreshes_only_the_timestamp(self, store):
        ledger = _two_sessions()
        a = _web(_KEY_A, ledger)
        first = _picture(store, scope=a, session_id=_KEY_A)
        before = first.to_dict()
        time.sleep(0.01)

        again = _picture(store, data=b"other bytes", scope=a, session_id=_KEY_A)

        after = again.to_dict()
        assert after.pop("timestamp") > before.pop("timestamp")
        assert after == before
        assert store.get_file_path(first.id).read_bytes() == b"\x89PNG one"

    def test_a_cli_scope_reuses_only_what_it_owns(self, store):
        cli = resolve_scope(key=None, ledger=None, ledger_ok=True, cli_conversation=_T_A)
        other = _picture(store, extra={"transcript_id": _T_B})
        untagged = _picture(store)
        assert untagged.id != other.id

        mine = _picture(store, scope=cli, extra={"transcript_id": _T_A})
        again = _picture(store, scope=cli, extra={"transcript_id": _T_A})

        assert mine.id not in {other.id, untagged.id}
        assert again.id == mine.id

    def test_a_web_scope_with_no_transcript_never_reuses(self, store):
        """A key whose ledger is unreadable owns nothing, so it reuses nothing."""
        untagged = _picture(store)
        blind = _web(_KEY_A, Ledger(), ledger_ok=False)

        saved = _picture(store, scope=blind, session_id=_KEY_A)

        assert saved.id != untagged.id

    def test_the_newest_qualifying_match_is_reused(self, store):
        ledger = _two_sessions()
        a = _web(_KEY_A, ledger)
        older = _picture(store, scope=_KEYLESS)
        # A legacy index may hold two matches the scope owns; the newest wins.
        store.save_file(
            file_content=b"x",
            filename="a.png",
            artifact_type="image",
            title="first",
            origin=_ORIGIN,
            session_id=_KEY_A,
            metadata={"sha256": "e" * 64},
        )
        newest = store.save_file(
            file_content=b"y",
            filename="b.png",
            artifact_type="image",
            title="second",
            origin=_ORIGIN,
            session_id=_KEY_A,
            metadata={"sha256": "e" * 64},
        )

        reused = _picture(store, scope=a, session_id=_KEY_A)

        assert reused.id == newest.id != older.id


class TestDispatchNeverReuses:
    def test_a_dispatch_save_with_a_match_present_creates_a_new_entry(self, store):
        dispatch = resolve_scope(key=None, ledger=None, ledger_ok=True, dispatch=True)
        untagged = _picture(store)

        first = _picture(store, scope=dispatch)
        second = _picture(store, scope=dispatch)

        assert len({untagged.id, first.id, second.id}) == 3


class TestKeylessReusesOnlyUntagged:
    def test_an_untagged_match_is_reused(self, store):
        first = _picture(store)
        assert _picture(store).id == first.id

    @pytest.mark.parametrize("ledger_ok", [True, False])
    def test_a_session_owned_match_is_never_reused(self, store, ledger_ok):
        ledger = _two_sessions()
        owned = _picture(store, scope=_web(_KEY_A, ledger), session_id=_KEY_A)
        keyless = resolve_scope(key=None, ledger=ledger, ledger_ok=ledger_ok)

        saved = _picture(store, scope=keyless)

        assert saved.id != owned.id
        assert _picture(store, scope=keyless).id == saved.id

    def test_a_session_tagged_match_is_never_reused_even_without_a_ledger_row(self, store):
        tagged = _picture(store, session_id="unknown-key")
        assert _picture(store).id != tagged.id

    def test_a_cli_tagged_match_is_never_reused(self, store):
        cli_tagged = _picture(store, extra={"transcript_id": _T_A})
        assert _picture(store).id != cli_tagged.id

    @pytest.mark.parametrize(
        "hidden",
        [
            pytest.param({"run_id": "run-1"}, id="dispatch-run"),
            pytest.param({"origin": artifact_store.WATCHDOG_ORIGIN}, id="watchdog"),
        ],
    )
    def test_an_operator_hidden_match_is_never_reused(self, store, hidden):
        origin = hidden.get("origin", _ORIGIN)
        match = store.save_file(
            file_content=b"x",
            filename="x.png",
            artifact_type="image",
            title="hidden",
            session_id="",
            metadata={"sha256": "e" * 64},
            run_id=hidden.get("run_id", ""),
            origin=origin,
        )
        saved = _picture(store, origin=origin)
        assert saved.id != match.id


class TestOriginStillScopesReuse:
    @pytest.mark.parametrize("which", ["keyless", "web"])
    def test_another_origins_match_is_never_reused(self, store, which):
        ledger = _two_sessions()
        scope = _KEYLESS if which == "keyless" else _web(_KEY_A, ledger)
        session_id = "" if which == "keyless" else _KEY_A
        other = _picture(store, origin="upload", scope=scope, session_id=session_id)

        saved = _picture(store, scope=scope, session_id=session_id)

        assert saved.id != other.id


def _now() -> str:
    return datetime.now(UTC).isoformat()
