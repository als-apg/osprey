"""``save_or_touch_by_sha256``: one artifact per picture, and no hidden origin.

A caller that may see the same bytes many times (a logbook picture attached to
a session twice) saves them once: the second sighting refreshes the existing
entry's timestamp. No ``origin`` is hidden from a listing — the store keeps no
audit spills, so every entry it holds is something a run produced.
"""

from __future__ import annotations

import inspect
import threading
import time
from unittest.mock import patch

import pytest

from osprey.stores import artifact_store
from osprey.stores.artifact_store import (
    ArtifactStore,
    register_artifact_listener,
    unregister_artifact_listener,
)

_ORIGIN = "logbook"


@pytest.fixture
def store(agent_data_root) -> ArtifactStore:
    return ArtifactStore(workspace_root=agent_data_root, auto_launch=False)


def _picture(
    store: ArtifactStore, sha: str = "e" * 64, data: bytes = b"\x89PNG one", origin: str = _ORIGIN
):
    return store.save_or_touch_by_sha256(
        sha,
        origin=origin,
        save_kwargs={
            "file_content": data,
            "filename": "picture-0.png",
            "artifact_type": "image",
            "title": "logbook picture",
            "mime_type": "image/png",
            "tool_source": "attachment_to_artifact",
            "metadata": {"sha256": sha},
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
