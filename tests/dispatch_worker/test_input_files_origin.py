"""Store-level provenance for ingested input files: the ``origin`` tag.

``ArtifactEntry.origin`` marks how an artifact entered a run — empty for the
agent's own output, ``"input"`` for a caller-supplied file the worker ingested
before the run. These tests pin the field default, its index round-trip, and
the ``save_file`` ``origin`` / explicit-``run_id`` parameters the worker uses to
write on behalf of a run whose id is not in its environment.
"""

from __future__ import annotations

import json

from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


def _save(store: ArtifactStore, **kw) -> object:
    return store.save_file(
        file_content=PNG_BYTES,
        filename="p.png",
        title="p",
        artifact_type="image",
        mime_type="image/png",
        **kw,
    )


def test_input_files_origin_defaults_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
    store = ArtifactStore(workspace_root=tmp_path)
    entry = _save(store)
    assert entry.origin == ""


def test_input_files_origin_set_to_input(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
    store = ArtifactStore(workspace_root=tmp_path)
    entry = _save(store, origin="input")
    assert entry.origin == "input"


def test_input_files_origin_explicit_run_id_overrides_env(tmp_path, monkeypatch):
    # The worker ingests before the agent starts, so the run id is not in its
    # environment — it passes run_id explicitly and it must win over the env.
    monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "env-run")
    store = ArtifactStore(workspace_root=tmp_path)
    entry = _save(store, run_id="explicit-run", origin="input")
    assert entry.run_id == "explicit-run"
    assert entry.origin == "input"


def test_input_files_origin_run_id_none_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OSPREY_DISPATCH_RUN_ID", "env-run")
    store = ArtifactStore(workspace_root=tmp_path)
    entry = _save(store)  # run_id defaults to None -> env
    assert entry.run_id == "env-run"


def test_input_files_origin_survives_index_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("OSPREY_DISPATCH_RUN_ID", raising=False)
    store = ArtifactStore(workspace_root=tmp_path)
    _save(store, run_id="run-1", origin="input")
    # A fresh store reads the persisted index from disk.
    reloaded = ArtifactStore(workspace_root=tmp_path)
    [entry] = reloaded.list_entries(run_filter="run-1")
    assert entry.origin == "input"


def test_input_files_origin_absent_in_old_index_loads_empty(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir(parents=True)
    (d / "artifacts.json").write_text(
        json.dumps(
            {
                "version": 1,
                "updated": "2026-01-01T00:00:00Z",
                "entry_count": 1,
                "entries": [
                    {
                        "id": "old1",
                        "artifact_type": "image",
                        "title": "t",
                        "description": "",
                        "filename": "t.png",
                        "mime_type": "image/png",
                        "size_bytes": 3,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "tool_source": "x",
                    }
                ],
            }
        )
    )
    store = ArtifactStore(workspace_root=tmp_path)
    [entry] = store.list_entries()
    assert entry.origin == ""
