"""Tests for the per-operator session ledger.

Pinned here:

* **Append-only, never repeating.** A row naming the key's latest transcript is
  not appended, and :func:`append` says so — which is what lets each write point
  send one frame per actual move.
* **A damaged file loads empty and is rewritten by the next append.**
* **The look-ups** (``keys``, ``current``, ``key_of``, ``transcripts``) answer
  from the append order, and the ledger satisfies ``owner_of``.
* **One path for the hub and the artifacts app**, in a container that binds
  ``OSPREY_CONTROL_CONTEXT_DIR`` and on a host that does not.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from osprey.stores import session_ledger
from osprey.stores.artifact_store import ArtifactEntry, owner_of
from osprey_connectors import posture_store

KEY = "aaaaaaaa-1111-2222-3333-444444444444"
OTHER_KEY = "bbbbbbbb-1111-2222-3333-444444444444"
T1 = "cccccccc-1111-2222-3333-444444444444"
T2 = "dddddddd-1111-2222-3333-444444444444"

AT_1 = "2026-11-01T10:00:00+00:00"
AT_2 = "2026-11-01T11:00:00+00:00"


class TestAppend:
    def test_append_writes_a_row_and_load_reads_it(self, tmp_path):
        assert session_ledger.append(tmp_path, KEY, KEY, session_ledger.EPOCH) is True
        ledger = session_ledger.load(tmp_path)
        assert list(ledger.rows) == [(KEY, KEY, session_ledger.EPOCH)]
        document = json.loads((tmp_path / session_ledger.STORE_FILENAME).read_text())
        assert document == {"rows": [[KEY, KEY, session_ledger.EPOCH]]}

    def test_the_ledger_is_readable_through_the_state_directory_group(self, tmp_path):
        # A dispatch worker's MCP processes read it under another uid, through the
        # state directory's setgid group, as they read the control-context record.
        session_ledger.append(tmp_path, KEY, T1, AT_1)
        mode = (tmp_path / session_ledger.STORE_FILENAME).stat().st_mode & 0o777
        assert mode == 0o640

    def test_a_repeat_of_the_latest_transcript_is_not_appended(self, tmp_path):
        assert session_ledger.append(tmp_path, KEY, T1, AT_1)
        path = tmp_path / session_ledger.STORE_FILENAME
        before = path.read_bytes()
        assert session_ledger.append(tmp_path, KEY, T1, AT_2) is False
        assert path.read_bytes() == before

    def test_returning_to_an_earlier_transcript_is_a_move(self, tmp_path):
        """Only the *latest* row is the no-repeat check, not every row."""
        session_ledger.append(tmp_path, KEY, KEY, session_ledger.EPOCH)
        session_ledger.append(tmp_path, KEY, T1, AT_1)
        assert session_ledger.append(tmp_path, KEY, KEY, AT_2) is True
        assert [r.transcript_id for r in session_ledger.load(tmp_path).rows] == [KEY, T1, KEY]

    def test_keys_are_independent(self, tmp_path):
        session_ledger.append(tmp_path, KEY, T1, AT_1)
        assert session_ledger.append(tmp_path, OTHER_KEY, T1, AT_1) is True

    def test_default_time_is_aware_now(self, tmp_path):
        session_ledger.append(tmp_path, KEY, T1)
        (row,) = session_ledger.load(tmp_path).rows
        assert datetime.fromisoformat(row.from_ts).tzinfo is not None

    def test_datetime_time_is_stored_as_aware_iso(self, tmp_path):
        session_ledger.append(tmp_path, KEY, T1, datetime(2026, 11, 1, 10, tzinfo=UTC))
        assert session_ledger.load(tmp_path).rows[0].from_ts == AT_1

    @pytest.mark.parametrize(
        ("key", "tid", "ts"),
        [("", T1, AT_1), (KEY, "", AT_1), (KEY, T1, "2026-11-01T10:00:00"), (KEY, T1, "junk")],
    )
    def test_blank_ids_and_naive_times_are_refused(self, tmp_path, key, tid, ts):
        assert session_ledger.append(tmp_path, key, tid, ts) is False
        assert not (tmp_path / session_ledger.STORE_FILENAME).exists()

    def test_no_location_appends_nothing(self):
        assert session_ledger.append(None, KEY, T1, AT_1) is False
        assert len(session_ledger.load(None)) == 0

    def test_append_creates_the_directory(self, tmp_path):
        state = tmp_path / "state" / "alice"
        assert session_ledger.append(state, KEY, T1, AT_1)
        assert session_ledger.exists(state)

    def test_append_many_applies_no_repeat_within_the_call(self, tmp_path):
        added = session_ledger.append_many(
            tmp_path, [(KEY, KEY, session_ledger.EPOCH), (KEY, KEY, AT_1), (KEY, T1, AT_1)]
        )
        assert [r.transcript_id for r in added] == [KEY, T1]


class TestCorruptFile:
    @pytest.mark.parametrize("text", ["{not json", "[]", '{"rows": 3}', ""])
    def test_a_damaged_file_loads_empty_and_is_logged(self, tmp_path, caplog, text):
        (tmp_path / session_ledger.STORE_FILENAME).write_text(text)
        with caplog.at_level("WARNING", logger="osprey.stores.session_ledger"):
            assert len(session_ledger.load(tmp_path)) == 0
        assert "session ledger" in caplog.text

    def test_the_next_append_rewrites_it(self, tmp_path):
        path = tmp_path / session_ledger.STORE_FILENAME
        path.write_text("{not json")
        assert session_ledger.append(tmp_path, KEY, T1, AT_1) is True
        assert json.loads(path.read_text()) == {"rows": [[KEY, T1, AT_1]]}

    def test_malformed_rows_are_dropped_one_by_one(self, tmp_path):
        rows = [[KEY, T1, AT_1], ["x"], [KEY, T2, "naive"], [KEY, 3, AT_2], [KEY, T2, AT_2]]
        (tmp_path / session_ledger.STORE_FILENAME).write_text(json.dumps({"rows": rows}))
        assert list(session_ledger.load(tmp_path).rows) == [(KEY, T1, AT_1), (KEY, T2, AT_2)]


class TestLoadChecked:
    """``load_checked``: the ledger plus whether it could be read at all.

    The hub's artifacts app and the agent's MCP processes share this reader;
    an unreadable document must answer *unavailable*, never an empty ledger
    that would silently re-own every artifact.
    """

    def test_no_location_is_unavailable(self):
        ledger, available = session_ledger.load_checked(None)
        assert list(ledger.rows) == []
        assert available is False

    def test_a_missing_file_is_an_empty_available_ledger(self, tmp_path):
        ledger, available = session_ledger.load_checked(tmp_path)
        assert list(ledger.rows) == []
        assert available is True

    @pytest.mark.parametrize("text", ["{not json", '{"other": []}', "[]", '{"rows": {}}'])
    def test_a_damaged_file_is_unavailable(self, tmp_path, text):
        (tmp_path / session_ledger.STORE_FILENAME).write_text(text)
        ledger, available = session_ledger.load_checked(tmp_path)
        assert list(ledger.rows) == []
        assert available is False

    def test_an_os_error_on_read_is_unavailable(self, tmp_path):
        # A directory where the file should be: read_text raises an OSError
        # (IsADirectoryError) that is not FileNotFoundError.
        (tmp_path / session_ledger.STORE_FILENAME).mkdir()
        ledger, available = session_ledger.load_checked(tmp_path)
        assert list(ledger.rows) == []
        assert available is False

    def test_a_good_file_answers_its_rows_and_available(self, tmp_path):
        session_ledger.append(tmp_path, KEY, T1, AT_1)
        session_ledger.append(tmp_path, KEY, T2, AT_2)
        ledger, available = session_ledger.load_checked(tmp_path)
        assert available is True
        assert list(ledger.rows) == [(KEY, T1, AT_1), (KEY, T2, AT_2)]

    def test_malformed_rows_in_a_good_document_stay_available(self, tmp_path):
        (tmp_path / session_ledger.STORE_FILENAME).write_text(
            json.dumps({"rows": [[KEY, T1, AT_1], ["bad"]]})
        )
        ledger, available = session_ledger.load_checked(tmp_path)
        assert available is True
        assert list(ledger.rows) == [(KEY, T1, AT_1)]

    def test_load_still_answers_the_same_ledger(self, tmp_path):
        session_ledger.append(tmp_path, KEY, T1, AT_1)
        assert session_ledger.load(tmp_path) == session_ledger.load_checked(tmp_path)[0]


class TestLookups:
    def _ledger(self, tmp_path):
        session_ledger.append_many(
            tmp_path,
            [
                (KEY, KEY, session_ledger.EPOCH),
                (KEY, T1, AT_1),
                (OTHER_KEY, T2, AT_1),
                (KEY, T2, AT_2),
            ],
        )
        return session_ledger.load(tmp_path)

    def test_keys_current_key_of_transcripts(self, tmp_path):
        ledger = self._ledger(tmp_path)
        assert ledger.keys() == [KEY, OTHER_KEY]
        assert ledger.current(KEY) == T2
        assert ledger.current(OTHER_KEY) == T2
        assert ledger.current("unknown") is None
        assert ledger.key_of(T2) == KEY  # the latest row naming it
        assert ledger.key_of(T1) == KEY
        assert ledger.key_of("unknown") is None
        assert ledger.transcripts() == [KEY, T1, T2]

    def test_the_ledger_satisfies_owner_of(self, tmp_path):
        ledger = self._ledger(tmp_path)
        entry = ArtifactEntry(
            id="a1",
            artifact_type="text",
            title="t",
            description="",
            filename="t.txt",
            mime_type="text/plain",
            size_bytes=1,
            timestamp=AT_2,
            tool_source="test",
            session_id=KEY,
        )
        entry.metadata = {"first_ts": "2026-11-01T10:30:00+00:00"}
        assert owner_of(entry, ledger) == T1
        entry.metadata = {"first_ts": "2026-10-01T00:00:00+00:00"}
        assert owner_of(entry, ledger) == KEY

    def test_removing_a_transcript_drops_its_rows_and_keeps_the_rest(self, tmp_path):
        self._ledger(tmp_path)
        assert session_ledger.remove_transcript(tmp_path, T2) == 2
        ledger = session_ledger.load(tmp_path)
        assert ledger.transcripts() == [KEY, T1]
        assert ledger.current(KEY) == T1 and ledger.current(OTHER_KEY) is None
        assert session_ledger.remove_transcript(tmp_path, T2) == 0


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    posture_store.invalidate_cache()
    yield monkeypatch
    posture_store.invalidate_cache()


class TestOnePathFromHubAndArtifactsApp:
    """The hub appends through ``session_meta``; the artifacts app loads by ``state_dir()``."""

    def _hub_appends_and_app_reads(self, expected_dir):
        from osprey.interfaces.web_terminal import session_meta

        app = SimpleNamespace(state=SimpleNamespace())
        assert session_meta.ledger_state_dir() == expected_dir
        assert session_meta.record_transcript_move(app, KEY, T1, AT_1)
        # The artifacts app's reader: the same rule, no hub imported.
        ledger = session_ledger.load(posture_store.state_dir())
        assert list(ledger.rows) == [(KEY, T1, AT_1)]
        assert (expected_dir / session_ledger.STORE_FILENAME).is_file()

    def test_container_bind_set(self, tmp_path, clean_env):
        bound = tmp_path / "bound-state"
        clean_env.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
        clean_env.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path / "elsewhere"))
        posture_store.invalidate_cache()
        self._hub_appends_and_app_reads(bound)

    def test_container_bind_unset(self, tmp_path, clean_env):
        root = tmp_path / "agent-data"
        root.mkdir()
        clean_env.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
        posture_store.invalidate_cache()
        expected = posture_store.state_dir()
        assert expected is not None and expected.is_relative_to(root)
        self._hub_appends_and_app_reads(expected)
