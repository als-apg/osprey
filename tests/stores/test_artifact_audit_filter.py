"""Audit spills in the artifact store: saved once per picture, hidden by default.

The tool-call audit surface saves the bytes of a large image block as an
artifact with ``origin == AUDIT_ORIGIN``. Those are records of what a tool
returned, so every ordinary listing drops them, and the same picture seen twice
is one artifact whose timestamp moves forward.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from osprey.stores.artifact_store import (
    AUDIT_ORIGIN,
    ArtifactStore,
    register_artifact_listener,
    unregister_artifact_listener,
)


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return ArtifactStore(workspace_root=repo_root / "state" / "agent", auto_launch=False)


def _spill(store: ArtifactStore, sha: str = "e" * 64, data: bytes = b"\x89PNG one"):
    return store.save_or_touch_by_sha256(
        sha,
        origin=AUDIT_ORIGIN,
        save_kwargs={
            "file_content": data,
            "filename": "toolu_1-image-0.png",
            "artifact_type": "image",
            "title": "mcp__x__shot image",
            "mime_type": "image/png",
            "tool_source": "audit.tool_call",
            "metadata": {"tool_use_id": "toolu_1", "sha256": sha},
        },
    )


def _produced(store: ArtifactStore, title: str = "Beam profile"):
    return store.save_file(
        file_content=b"\x89PNG produced",
        filename="beam.png",
        artifact_type="image",
        title=title,
        mime_type="image/png",
        tool_source="execute",
    )


def test_the_origin_is_the_audit_surface_constant() -> None:
    from osprey.audit import tool_call

    assert AUDIT_ORIGIN == tool_call.ARTIFACT_ORIGIN == "tool_call"


def test_two_views_of_one_picture_are_one_artifact_with_a_refreshed_timestamp(store) -> None:
    first = _spill(store)
    stamp = first.timestamp
    time.sleep(0.01)

    second = _spill(store)

    assert second.id == first.id
    spills = [e for e in store.list_entries(include_audit=True) if e.origin == AUDIT_ORIGIN]
    assert len(spills) == 1
    assert spills[0].timestamp > stamp
    # The refresh is on disk, not only in this process's copy.
    reread = ArtifactStore(workspace_root=store._workspace, auto_launch=False)
    assert reread.get_entry(first.id).timestamp == spills[0].timestamp


def test_different_pictures_are_different_artifacts(store) -> None:
    a = _spill(store, "1" * 64, b"one")
    b = _spill(store, "2" * 64, b"two")
    assert a.id != b.id


def test_an_ordinary_artifact_with_the_same_sha256_is_not_reused(store) -> None:
    other = store.save_file(
        file_content=b"x",
        filename="x.png",
        artifact_type="image",
        title="tool output",
        metadata={"sha256": "e" * 64},
    )
    spill = _spill(store)
    assert spill.id != other.id


def test_the_gallery_listing_hides_audit_spills(store) -> None:
    spill = _spill(store)
    produced = _produced(store)

    assert [e.id for e in store.list_entries()] == [produced.id]
    assert [e.id for e in store.page_entries(limit=10).entries] == [produced.id]
    assert {e.id for e in store.list_entries(include_audit=True)} == {spill.id, produced.id}
    assert store.get_entry(spill.id) is not None


def test_two_threads_spilling_the_same_bytes_give_one_artifact(store) -> None:
    barrier = threading.Barrier(2)
    ids: list[str] = []

    def spill() -> None:
        own = ArtifactStore(workspace_root=store._workspace, auto_launch=False)
        barrier.wait()
        ids.append(_spill(own).id)

    threads = [threading.Thread(target=spill) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)

    assert len(ids) == 2 and ids[0] == ids[1]
    reread = ArtifactStore(workspace_root=store._workspace, auto_launch=False)
    assert len(reread.list_entries(include_audit=True)) == 1


def test_listeners_fire_once_for_a_new_spill_and_the_gallery_is_not_launched(tmp_path) -> None:
    seen = []

    def listener(entry) -> None:
        seen.append(entry.id)

    store = ArtifactStore(workspace_root=tmp_path, auto_launch=True)
    register_artifact_listener(listener)
    try:
        with patch.object(ArtifactStore, "_launch_gallery") as launch:
            first = _spill(store)
            _spill(store)
        assert seen == [first.id]
        launch.assert_not_called()
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
