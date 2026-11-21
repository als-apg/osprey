"""One answer of ``POST /api/artifacts/session-summary`` is of one index state.

A run counts once (its head) whatever its status, so a summary that classified
the run groups from one read of the index and counted the entries of another
would count a run whose result landed between the two reads twice: its fresh
result beside its preview head.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.artifacts.app import create_app
from osprey.stores import session_ledger
from osprey.stores.artifact_forms import RUN_RECORD_KEY
from osprey_connectors import control_context, posture_store

EXEC = "toolu_run1"
KEY = "11111111-1111-4111-8111-111111111111"
T1 = KEY  # a key's first transcript id is the key itself


@pytest.fixture
def state_dir(tmp_path, monkeypatch) -> Path:
    """The ledger's state directory, bound so the app never reads a real one."""
    bound = tmp_path / "state"
    bound.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(bound))
    monkeypatch.delenv("OSPREY_SESSION_ID", raising=False)
    monkeypatch.delenv("OSPREY_ARTIFACT_ORIGIN", raising=False)
    control_context.invalidate_cache()
    yield bound
    control_context.invalidate_cache()


@pytest.fixture
def client(tmp_path, state_dir) -> TestClient:
    assert state_dir.is_dir()  # bound before the app reads any ledger
    return TestClient(create_app(workspace_root=tmp_path / "data"))


@pytest.fixture
def store(client):
    return client.app.state.artifact_store


def _preview(store):
    return store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True},
        exec_id=EXEC,
        session_id=KEY,
    )


def _result(store):
    entry = store.save_data(
        tool="execute",
        data={"stdout": ""},
        title="fit",
        metadata={"run_part": "result", RUN_RECORD_KEY: True, "run_status": "succeeded"},
        exec_id=EXEC,
    )
    store.update_entry_metadata(entry.id, session_id=KEY)
    return store.get_entry(entry.id)


def test_a_run_whose_result_lands_between_reads_counts_once(client, store, state_dir, monkeypatch):
    (state_dir / session_ledger.STORE_FILENAME).write_text(
        json.dumps({"rows": [[KEY, T1, session_ledger.EPOCH]]})
    )
    preview = _preview(store)
    result = _result(store)
    # The result is written to the index after the answer's first read of it.
    reads = iter([[preview], [preview, result]])
    monkeypatch.setattr(store, "list_entries", lambda *a, **kw: list(next(reads)))

    response = client.post("/api/artifacts/session-summary", json={"sessions": [T1]})

    assert response.status_code == 200, response.text
    assert response.json()["counts"][T1]["by_form"] == {"runs": 1}
