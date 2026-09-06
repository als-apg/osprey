"""Tests for the per-session-key transcript map.

The map exists because a session key and the transcript it points at diverge
the moment a ``/clear`` starts a fresh conversation under the same key. Four
properties are pinned here:

* **A key with no entry answers itself.** The common case is a session that has
  never cleared, and a caller resuming it must always have an id to resume.
* **A move survives a restart.** The point of persisting at all: a recreated
  container that answered the key would silently resume the wrong (empty)
  conversation and read as a session that lost its history.
* **The file sits beside the posture store**, under the same one path rule, so
  a deployment that resolves one of the two resolves both.
* **A map with no location still answers.** An unresolvable agent-data root
  costs durability, never a working view flip.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.interfaces.web_terminal import transcript_map
from osprey_connectors import session_store

# A session key: the bare UUID one browser tab keeps for its whole life.
KEY = "aaaaaaaa-1111-2222-3333-444444444444"
# A second key, so a stored entry can be shown not to answer for its neighbour.
OTHER_KEY = "bbbbbbbb-1111-2222-3333-444444444444"
# The transcript a ``/clear`` moved the first key to.
TRANSCRIPT = "cccccccc-1111-2222-3333-444444444444"


def _app() -> SimpleNamespace:
    """A bare app object — the store helpers need ``state`` and nothing else."""
    return SimpleNamespace(state=SimpleNamespace())


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    """Pin the agent-data root, the way a spawn does.

    ``OSPREY_AGENT_DATA_ROOT`` is the first half of the one resolution rule the
    map shares with the posture store, so setting it exercises the real
    derivation instead of patching around it.
    """
    root = tmp_path / "shared_agent_data"
    root.mkdir()
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    session_store.invalidate_cache()
    yield root
    session_store.invalidate_cache()


@pytest.fixture
def unresolvable_root(monkeypatch):
    """No stamp and no config derivation: a map with nowhere to write."""
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(session_store, "resolve_shared_data_root", _raise_no_root)
    session_store.invalidate_cache()
    yield
    session_store.invalidate_cache()


def _raise_no_root() -> Path:
    raise RuntimeError("no project root")


class TestAKeyAnswersItselfByDefault:
    def test_unknown_key_returns_itself(self, shared_root):
        """The ordinary case: a session that has never cleared has no entry."""
        assert transcript_map.get(_app(), KEY) == KEY

    def test_a_stored_entry_does_not_answer_for_another_key(self, shared_root):
        app = _app()
        transcript_map.set(app, KEY, TRANSCRIPT)

        assert transcript_map.get(app, OTHER_KEY) == OTHER_KEY

    def test_an_entry_naming_the_key_is_not_stored(self, shared_root):
        """``set(key, key)`` says what the default already says.

        Storing it would add one identity line per session to a file that is
        supposed to grow only per conversation that actually moved.
        """
        app = _app()
        transcript_map.set(app, KEY, KEY)

        assert app.state.transcript_map == {}
        assert transcript_map.get(app, KEY) == KEY

    def test_a_blank_transcript_id_is_ignored(self, shared_root):
        """There is no such transcript, and it would replace a usable answer."""
        app = _app()
        transcript_map.set(app, KEY, TRANSCRIPT)
        transcript_map.set(app, KEY, "   ")

        assert transcript_map.get(app, KEY) == TRANSCRIPT


class TestAMoveSurvivesARestart:
    def test_set_then_a_new_app_gets_the_mapped_id(self, shared_root):
        """The property the file exists for."""
        transcript_map.set(_app(), KEY, TRANSCRIPT)

        restarted = _app()
        transcript_map.load(restarted)

        assert transcript_map.get(restarted, KEY) == TRANSCRIPT

    def test_a_new_app_loads_lazily_without_a_lifespan(self, shared_root):
        """A helper reached before the lifespan load still reads the file."""
        transcript_map.set(_app(), KEY, TRANSCRIPT)

        assert transcript_map.get(_app(), KEY) == TRANSCRIPT

    def test_clearing_an_entry_survives_too(self, shared_root):
        app = _app()
        transcript_map.set(app, KEY, TRANSCRIPT)
        transcript_map.set(app, KEY, KEY)

        restarted = _app()
        transcript_map.load(restarted)

        assert transcript_map.get(restarted, KEY) == KEY

    def test_a_second_move_replaces_the_first(self, shared_root):
        second = "dddddddd-1111-2222-3333-444444444444"
        app = _app()
        transcript_map.set(app, KEY, TRANSCRIPT)
        transcript_map.set(app, KEY, second)

        restarted = _app()
        transcript_map.load(restarted)

        assert transcript_map.get(restarted, KEY) == second


class TestTheFileSitsBesideThePostureStore:
    def test_the_path_is_the_posture_store_s_directory(self, shared_root):
        posture_store = session_store.store_path()

        assert transcript_map.store_path() == posture_store.parent / "session-transcripts.json"

    def test_the_document_is_key_to_transcript_id(self, shared_root):
        transcript_map.set(_app(), KEY, TRANSCRIPT)

        written = json.loads(transcript_map.store_path().read_text(encoding="utf-8"))
        assert written == {KEY: TRANSCRIPT}

    def test_entries_that_cannot_name_a_transcript_are_dropped(self, shared_root):
        """A hand-edited file loses only the entries nobody could have used."""
        path = transcript_map.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({KEY: TRANSCRIPT, OTHER_KEY: "", "third": 7}),
            encoding="utf-8",
        )

        app = _app()
        transcript_map.load(app)

        assert app.state.transcript_map == {KEY: TRANSCRIPT}

    def test_a_corrupt_file_is_an_empty_map(self, shared_root):
        path = transcript_map.store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        assert transcript_map.get(_app(), KEY) == KEY


class TestAMapWithNoLocationStillAnswers:
    def test_set_and_get_work_in_memory(self, unresolvable_root):
        app = _app()
        transcript_map.set(app, KEY, TRANSCRIPT)

        assert transcript_map.store_path() is None
        assert transcript_map.get(app, KEY) == TRANSCRIPT

    def test_an_entry_made_during_the_outage_survives_the_recovery_load(
        self, unresolvable_root, tmp_path, monkeypatch
    ):
        """A later load must not undo what memory learned while it had nowhere to write."""
        app = _app()
        transcript_map.set(app, KEY, TRANSCRIPT)

        root = tmp_path / "recovered"
        root.mkdir()
        monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
        session_store.invalidate_cache()

        assert transcript_map.get(app, KEY) == TRANSCRIPT
