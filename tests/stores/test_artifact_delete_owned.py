"""``ArtifactStore.delete_owned`` deletes only what one session owns.

Ownership is :func:`owned_by` over the scope; ``form=None`` takes everything the
scope owns (run parts included), ``form="runs"`` takes each owned run head with
the owned members of its group, any other form or extension key takes the owned
entries listed under it. A scope with no transcript deletes nothing, and an
entry another session owns survives every call, even inside a split run group.
"""

from __future__ import annotations

import pytest

from osprey.stores.artifact_scope import SessionScope
from osprey.stores.artifact_store import (
    EXAMPLE_ORIGIN,
    RUN_RECORD_KEY,
    WATCHDOG_ORIGIN,
    ArtifactEntry,
    ArtifactStore,
)
from osprey.stores.session_ledger import EPOCH, Ledger, LedgerRow
from osprey.stores.type_registry import ExtensionType

KEY_A = "11111111-1111-4111-8111-111111111111"
KEY_B = "22222222-2222-4222-8222-222222222222"
T_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
T_A2 = "abababab-abab-4bab-8bab-abababababab"  # KEY_A's transcript after a /clear
T_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
CLEAR_AT = "2026-11-05T10:00:00+00:00"

LEDGER = Ledger(
    rows=(
        LedgerRow(KEY_A, T_A, EPOCH),
        LedgerRow(KEY_B, T_B, EPOCH),
        LedgerRow(KEY_A, T_A2, CLEAR_AT),
    )
)
SCOPE_A = SessionScope(kind="web", transcript=T_A, ledger=LEDGER)
SCOPE_A2 = SessionScope(kind="web", transcript=T_A2, ledger=LEDGER)
SCOPE_B = SessionScope(kind="web", transcript=T_B, ledger=LEDGER)
DISPATCH = SessionScope(kind="dispatch", transcript=None, ledger=None, reason="dispatch run")
NONE = SessionScope(kind="none", transcript=None, ledger=LEDGER, reason="no session")

LATTICE = {"lattice": ExtensionType("lattice", "Lattice", "tables")}
EXEC = "toolu_run1"


@pytest.fixture(autouse=True)
def _no_ambient_tags(monkeypatch):
    for var in ("OSPREY_SESSION_ID", "OSPREY_DISPATCH_RUN_ID", "OSPREY_ARTIFACT_ORIGIN"):
        monkeypatch.delenv(var, raising=False)


def _of(transcript: str | None, **extra) -> dict:
    meta = dict(extra)
    if transcript:
        meta["transcript_id"] = transcript
    return meta


def _doc(store: ArtifactStore, transcript: str | None, title: str = "Notes", **kw) -> ArtifactEntry:
    return store.save_file(
        b"# x", "x.md", "markdown", title, metadata=_of(transcript), session_id="", **kw
    )


def _table(store: ArtifactStore, transcript: str | None) -> ArtifactEntry:
    return store.save_file(
        b"<t/>", "t.html", "table_html", "Table", metadata=_of(transcript), session_id=""
    )


def _lattice(store: ArtifactStore, transcript: str | None) -> ArtifactEntry:
    return store.save_file(
        b"{}", "l.json", "json", "Lattice", metadata=_of(transcript), session_id="", kind="lattice"
    )


def _preview(store: ArtifactStore, transcript: str | None, exec_id: str = EXEC) -> ArtifactEntry:
    return store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        metadata=_of(transcript, run_part="preview", **{RUN_RECORD_KEY: True}),
        session_id="",
        exec_id=exec_id,
    )


def _output(store: ArtifactStore, transcript: str | None, exec_id: str = EXEC) -> ArtifactEntry:
    return store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure",
        mime_type="image/png",
        metadata=_of(transcript, run_part="output", figure=True),
        session_id="",
        exec_id=exec_id,
    )


def _result(store: ArtifactStore, transcript: str | None, exec_id: str = EXEC) -> ArtifactEntry:
    return store.save_data(
        tool="execute",
        data={"stdout": ""},
        title="fit",
        metadata=_of(
            transcript, run_part="result", run_status="succeeded", **{RUN_RECORD_KEY: True}
        ),
        exec_id=exec_id,
    )


def _ids(entries) -> set[str]:
    return {e.id for e in entries}


def _left(store: ArtifactStore) -> set[str]:
    return _ids(store.list_entries())


class TestEverything:
    def test_two_sessions_everything_deletes_only_this_ones(self, store):
        """Req 7: A's everything leaves B's, the example, watchdog and dispatch entries."""
        a_doc, a_table = _doc(store, T_A), _table(store, T_A)
        a_run = {_preview(store, T_A).id, _output(store, T_A).id, _result(store, T_A).id}
        b_doc = _doc(store, T_B)
        b_run = {
            _preview(store, T_B, "toolu_b").id,
            _result(store, T_B, "toolu_b").id,
        }
        example = _doc(store, None, origin=EXAMPLE_ORIGIN)
        judge = _doc(store, T_A, origin=WATCHDOG_ORIGIN)
        dispatched = _doc(store, T_A, run_id="run-1")

        removed = store.delete_owned(SCOPE_A, registry={})

        assert _ids(removed) == {a_doc.id, a_table.id} | a_run
        assert _left(store) == {b_doc.id, example.id, judge.id, dispatched.id} | b_run

    def test_ownership_follows_the_ledger_for_untagged_saves(self, store):
        """An entry with only a session key belongs to its key's transcript at save time."""
        before = store.save_file(
            b"# x",
            "x.md",
            "markdown",
            "before",
            session_id=KEY_A,
            metadata={"first_ts": "2026-11-05T09:00:00+00:00"},
        )
        after = store.save_file(
            b"# x",
            "x.md",
            "markdown",
            "after",
            session_id=KEY_A,
            metadata={"first_ts": "2026-11-05T11:00:00+00:00"},
        )
        assert _ids(store.delete_owned(SCOPE_A2, registry={})) == {after.id}
        assert _left(store) == {before.id}

    def test_nothing_owned_deletes_nothing(self, store):
        b_doc = _doc(store, T_B)
        assert store.delete_owned(SCOPE_A, registry={}) == []
        assert _left(store) == {b_doc.id}


class TestNoTranscript:
    @pytest.mark.parametrize("scope", [DISPATCH, NONE], ids=["dispatch", "none"])
    @pytest.mark.parametrize("form", [None, "runs", "documents"])
    def test_a_scope_without_a_transcript_deletes_nothing(self, store, scope, form):
        _doc(store, T_A)
        _doc(store, None)
        _preview(store, T_A)
        _result(store, T_A)
        before = _left(store)
        assert store.delete_owned(scope, form, registry={}) == []
        assert _left(store) == before

    def test_a_dispatch_scope_spares_dispatch_entries_too(self, store):
        dispatched = _doc(store, None, run_id="run-1")
        assert store.delete_owned(DISPATCH, registry={}) == []
        assert _left(store) == {dispatched.id}


class TestForms:
    def test_base_form_spares_an_extension_type_and_its_key_takes_it(self, store):
        """Req 11, delete half."""
        a_table, a_lat = _table(store, T_A), _lattice(store, T_A)
        b_table, b_lat = _table(store, T_B), _lattice(store, T_B)

        assert _ids(store.delete_owned(SCOPE_A, "tables", registry=LATTICE)) == {a_table.id}
        assert _left(store) == {a_lat.id, b_table.id, b_lat.id}

        assert _ids(store.delete_owned(SCOPE_A, "lattice", registry=LATTICE)) == {a_lat.id}
        assert _left(store) == {b_table.id, b_lat.id}

    def test_a_form_takes_only_owned_entries_listed_under_it(self, store):
        a_doc, a_table = _doc(store, T_A), _table(store, T_A)
        b_doc = _doc(store, T_B)
        assert _ids(store.delete_owned(SCOPE_A, "documents", registry={})) == {a_doc.id}
        assert _left(store) == {a_table.id, b_doc.id}

    def test_an_output_form_leaves_its_run(self, store):
        p, o, r = _preview(store, T_A), _output(store, T_A), _result(store, T_A)
        assert _ids(store.delete_owned(SCOPE_A, "plots", registry={})) == {o.id}
        assert _left(store) == {p.id, r.id}

    def test_a_form_never_takes_a_run_part(self, store):
        """A preview is a part of its run, not a document."""
        p, r = _preview(store, T_A), _result(store, T_A)
        assert store.delete_owned(SCOPE_A, "documents", registry={}) == []
        assert _left(store) == {p.id, r.id}


class TestRuns:
    def test_runs_take_owned_heads_with_their_owned_parts(self, store):
        run = {_preview(store, T_A).id, _output(store, T_A).id, _result(store, T_A).id}
        doc = _doc(store, T_A)
        b_run = {_preview(store, T_B, "toolu_b").id, _result(store, T_B, "toolu_b").id}

        assert _ids(store.delete_owned(SCOPE_A, "runs", registry={})) == run
        assert _left(store) == {doc.id} | b_run

    def test_a_run_split_by_a_clear_keeps_the_other_sessions_parts(self, store):
        """The head is A2's, the preview is A's: A2 takes its head and output only."""
        preview = _preview(store, T_A)
        output = _output(store, T_A2)
        result = _result(store, T_A2)

        assert _ids(store.delete_owned(SCOPE_A2, "runs", registry={})) == {output.id, result.id}
        assert _left(store) == {preview.id}

    def test_a_session_owning_only_a_part_owns_no_run(self, store):
        preview, result = _preview(store, T_A), _result(store, T_A2)
        assert store.delete_owned(SCOPE_A, "runs", registry={}) == []
        assert _left(store) == {preview.id, result.id}

    def test_everything_takes_owned_parts_of_a_split_run(self, store):
        preview, result = _preview(store, T_A), _result(store, T_A2)
        assert _ids(store.delete_owned(SCOPE_A, registry={})) == {preview.id}
        assert _left(store) == {result.id}


class TestMechanics:
    def test_files_are_unlinked_and_the_index_persists(self, store, agent_data_root):
        a_doc, b_doc = _doc(store, T_A), _doc(store, T_B)
        path = store.get_file_path(a_doc.id)
        assert path is not None and path.exists()

        store.delete_owned(SCOPE_A, registry={})

        assert not path.exists()
        assert store.get_file_path(b_doc.id).exists()
        reloaded = ArtifactStore(workspace_root=agent_data_root)
        assert _ids(reloaded.list_entries()) == {b_doc.id}

    def test_listeners_fire_once_per_removed_entry_after_the_lock(self, store):
        a1, a2 = _doc(store, T_A), _table(store, T_A)
        _doc(store, T_B)
        seen: list[tuple[str, int]] = []

        def listener(entry):
            # A listener may call back into the store: the lock is released.
            seen.append((entry.id, len(store.list_entries())))

        ArtifactStore.register_delete_listener(listener)
        try:
            store.delete_owned(SCOPE_A, registry={})
        finally:
            ArtifactStore.unregister_delete_listener(listener)

        assert sorted(i for i, _ in seen) == sorted([a1.id, a2.id])
        assert all(count == 1 for _, count in seen)

    def test_the_default_registry_is_used_when_none_is_given(self, store):
        a_doc = _doc(store, T_A)
        assert _ids(store.delete_owned(SCOPE_A, "documents")) == {a_doc.id}

    def test_scope_b_is_unaffected_by_scope_a(self, store):
        b_doc = _doc(store, T_B)
        _doc(store, T_A)
        store.delete_owned(SCOPE_A, registry={})
        assert _ids(store.delete_owned(SCOPE_B, registry={})) == {b_doc.id}
        assert store.list_entries() == []
