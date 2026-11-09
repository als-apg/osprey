"""A save at a stated time and under a stated id.

``save_file(at=...)`` backdates an entry: ``timestamp`` and ``metadata.first_ts``
both carry the stated time, so ``owner_of`` resolves the entry by the ledger row
in force then. ``save_file(entry_id=...)`` replaces the random id, refused when
the store already holds it or when it is not of the store's id shape.
"""

from __future__ import annotations

import re
import uuid
from collections import namedtuple
from datetime import UTC, datetime, timedelta

import pytest

from osprey.stores.artifact_store import ArtifactEntry, ArtifactStore, owner_of

Row = namedtuple("Row", "session_key transcript_id from_ts")

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


class _Ledger:
    def __init__(self, rows):
        self.rows = rows


@pytest.fixture
def store(agent_data_root) -> ArtifactStore:
    return ArtifactStore(workspace_root=agent_data_root, auto_launch=False)


def _save(store: ArtifactStore, **kwargs) -> ArtifactEntry:
    kwargs.setdefault("session_id", "K")
    return store.save_file(
        file_content=b"# body",
        filename="a.md",
        artifact_type="markdown",
        title="A",
        **kwargs,
    )


class TestStatedTime:
    def test_the_stated_time_lands_in_timestamp_and_first_ts(self, store):
        at = T0 + timedelta(minutes=7)
        entry = _save(store, at=at)
        assert datetime.fromisoformat(entry.timestamp) == at
        assert datetime.fromisoformat(entry.metadata["first_ts"]) == at

    def test_the_stated_time_survives_a_reload(self, store, agent_data_root):
        at = T0 + timedelta(minutes=7)
        saved = _save(store, at=at)
        reloaded = ArtifactStore(workspace_root=agent_data_root, auto_launch=False)
        entry = reloaded.get_entry(saved.id)
        assert entry is not None
        assert datetime.fromisoformat(entry.timestamp) == at
        assert datetime.fromisoformat(entry.metadata["first_ts"]) == at

    def test_owner_of_resolves_by_the_row_in_force_at_the_stated_time(self, store):
        ledger = _Ledger(
            [
                Row("K", "T1", T0.isoformat()),
                Row("K", "T2", (T0 + timedelta(hours=1)).isoformat()),
                Row("K", "T3", (T0 + timedelta(hours=2)).isoformat()),
            ]
        )
        # Saved now (long after every row) but stated inside T2's window.
        entry = _save(store, at=T0 + timedelta(hours=1, minutes=30))
        assert owner_of(entry, ledger) == "T2"
        # Without a stated time the same save belongs to the latest row.
        assert owner_of(_save(store), ledger) == "T3"

    def test_a_caller_first_ts_still_wins(self, store):
        caller = (T0 - timedelta(days=1)).isoformat()
        entry = _save(store, at=T0, metadata={"first_ts": caller})
        assert entry.metadata["first_ts"] == caller
        assert datetime.fromisoformat(entry.timestamp) == T0

    def test_a_naive_time_is_refused(self, store):
        with pytest.raises(ValueError, match="timezone"):
            _save(store, at=datetime(2026, 10, 1, 9, 0))
        assert store.list_entries() == []


class TestStatedId:
    def test_the_stated_id_replaces_the_random_one(self, store):
        entry = _save(store, entry_id="0123456789ab")
        assert entry.id == "0123456789ab"
        assert entry.filename.startswith("0123456789ab_")
        assert store.get_entry("0123456789ab") is entry
        assert (store.artifact_dir / entry.filename).read_bytes() == b"# body"

    def test_a_duplicate_id_is_refused(self, store):
        first = _save(store, entry_id="0123456789ab")
        with pytest.raises(ValueError, match="0123456789ab"):
            _save(store, entry_id="0123456789ab")
        assert [e.id for e in store.list_entries()] == [first.id]

    def test_a_duplicate_of_a_random_id_is_refused(self, store):
        first = _save(store)
        with pytest.raises(ValueError):
            _save(store, entry_id=first.id)
        assert len(store.list_entries()) == 1

    def test_a_duplicate_id_is_refused_across_store_instances(self, store, agent_data_root):
        _save(store, entry_id="0123456789ab")
        other = ArtifactStore(workspace_root=agent_data_root, auto_launch=False)
        with pytest.raises(ValueError):
            _save(other, entry_id="0123456789ab")

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(uuid.uuid4().hex, id="full-uuid-hex"),
            pytest.param("0123456789AB", id="uppercase"),
            pytest.param("0123456789a", id="11-chars"),
            pytest.param("0123456789abc", id="13-chars"),
            pytest.param("0123456789ag", id="non-hex"),
            pytest.param("", id="empty"),
            pytest.param("0123456789a\n", id="trailing-newline"),
            pytest.param("../escape000", id="path"),
        ],
    )
    def test_an_id_of_another_shape_is_refused(self, store, bad):
        with pytest.raises(ValueError):
            _save(store, entry_id=bad)
        assert store.list_entries() == []
        assert not any(store.artifact_dir.glob("*_a.md"))


class TestDefaults:
    def test_omitted_keywords_behave_as_today(self, store):
        before = datetime.now(UTC)
        entry = _save(store)
        after = datetime.now(UTC)
        assert re.fullmatch(r"[0-9a-f]{12}", entry.id)
        stamped = datetime.fromisoformat(entry.timestamp)
        assert before <= stamped <= after
        assert entry.metadata["first_ts"] == entry.timestamp

    def test_none_is_the_same_as_omitted(self, store):
        entry = _save(store, at=None, entry_id=None)
        assert re.fullmatch(r"[0-9a-f]{12}", entry.id)
        assert entry.metadata["first_ts"] == entry.timestamp

    def test_the_keywords_are_keyword_only(self, store):
        with pytest.raises(TypeError):
            store.save_file(  # type: ignore[misc]
                b"x", "a.md", "markdown", "A", "", "text/plain", "t", None, "", None, "", None, T0
            )
