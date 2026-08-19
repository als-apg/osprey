"""``ArtifactStore.save_channel_reading`` — the two-file channel-reading save.

A multi-dimensional channel reading is stored twice: a rendered PNG the
gallery displays (the primary file) and the raw ``.npy`` payload beside it
(``data_file``). These tests pin the field mapping the gallery and the
retention prune both read, and the atomicity contract — a save that fails
partway leaves neither a file nor an index entry behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.stores.artifact_store import ArtifactStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n-rendered-frame"
NPY_BYTES = b"\x93NUMPY-raw-payload"
CHANNEL = "SIM:CAM1:ArrayData"


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    """An ArtifactStore with a real repo root, so ``data_file`` anchors resolve."""
    repo_root = tmp_path / "repo"
    (repo_root / "build").mkdir(parents=True)
    config_path = repo_root / "build" / "config.yml"
    config_path.write_text(
        f"project_root: {repo_root}\nagent_data:\n  base_dir: state/agent\n",
    )
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(repo_root)
    return ArtifactStore(workspace_root=repo_root / "state" / "agent")


def _save(store: ArtifactStore, **overrides):
    kwargs = {
        "png_bytes": PNG_BYTES,
        "npy_bytes": NPY_BYTES,
        "title": "Camera frame",
        "summary": {"shape": [2, 2]},
        "access_details": {"loader": "numpy.load"},
        "metadata": {"channel": CHANNEL},
    }
    kwargs.update(overrides)
    return store.save_channel_reading(**kwargs)


def _payload_files(store: ArtifactStore) -> list[Path]:
    """Every artifact file in the store directory (the index does not count)."""
    if not store.artifact_dir.exists():
        return []
    return sorted(p for p in store.artifact_dir.iterdir() if p.suffix in (".png", ".npy"))


class TestFieldMapping:
    def test_png_is_the_primary_file(self, store):
        entry = _save(store)

        assert entry.filename.endswith(".png")
        assert entry.mime_type == "image/png"
        assert entry.artifact_type == "image"
        assert store.get_file_path(entry.id).read_bytes() == PNG_BYTES

    def test_npy_is_the_companion_data_file(self, store):
        entry = _save(store)

        assert entry.data_file.endswith(".npy")
        assert not Path(entry.data_file).is_absolute(), "data_file is repo-root-relative"
        assert (store.repo_root / entry.data_file).read_bytes() == NPY_BYTES

        # The deletion path must see it as a real companion, not a self-pointer.
        companion = store._entry_data_file(entry)
        assert companion is not None
        assert companion != store.get_file_path(entry.id)

    def test_both_files_land_on_disk_and_share_the_artifact_id(self, store):
        entry = _save(store)
        files = _payload_files(store)

        assert [p.suffix for p in files] == [".npy", ".png"]
        assert all(p.name.startswith(f"{entry.id}_") for p in files)
        assert {p.read_bytes() for p in files} == {PNG_BYTES, NPY_BYTES}

    def test_channel_is_recorded_in_metadata(self, store):
        entry = _save(store)

        assert entry.metadata["channel"] == CHANNEL
        reopened = ArtifactStore(workspace_root=store._workspace)
        assert reopened.get_entry(entry.id).metadata["channel"] == CHANNEL

    def test_uses_the_reserved_channel_values_category(self, store):
        from osprey.stores.type_registry import valid_category_keys

        assert "channel_values" in valid_category_keys()
        assert _save(store).category == "channel_values"

    def test_summary_access_details_and_size_are_carried(self, store):
        entry = _save(store)

        assert entry.summary == {"shape": [2, 2]}
        assert entry.access_details == {"loader": "numpy.load"}
        assert entry.size_bytes == len(PNG_BYTES) + len(NPY_BYTES)

    def test_caller_metadata_is_not_mutated(self, store):
        metadata = {"channel": CHANNEL}
        _save(store, metadata=metadata)

        assert metadata == {"channel": CHANNEL}


class TestChannelIsRequired:
    @pytest.mark.parametrize("metadata", [None, {}, {"channel": ""}, {"other": "x"}])
    def test_missing_channel_raises(self, store, metadata):
        with pytest.raises(ValueError, match="channel"):
            _save(store, metadata=metadata)

    def test_missing_channel_writes_nothing(self, store):
        with pytest.raises(ValueError):
            _save(store, metadata={})

        assert _payload_files(store) == []
        assert store.list_entries() == []


class TestPartialSaveStrandsNothing:
    def test_png_write_failure_removes_the_npy(self, store, monkeypatch):
        """Failure between the two writes: the first file must not survive."""
        real_write_bytes = Path.write_bytes

        def failing_write(self: Path, data: bytes) -> int:
            if self.suffix == ".png":
                raise OSError("no space left on device")
            return real_write_bytes(self, data)

        monkeypatch.setattr(Path, "write_bytes", failing_write)

        with pytest.raises(OSError, match="no space left"):
            _save(store)

        monkeypatch.undo()
        assert _payload_files(store) == []
        assert store.list_entries() == []

    def test_index_write_failure_removes_both_files(self, store, monkeypatch):
        """Failure after both writes: neither file nor entry may remain."""

        def failing_save_index(self) -> None:
            raise OSError("index is read-only")

        monkeypatch.setattr(ArtifactStore, "_save_index", failing_save_index)

        with pytest.raises(OSError, match="index is read-only"):
            _save(store)

        monkeypatch.undo()
        assert _payload_files(store) == []
        assert store.list_entries() == []
        assert ArtifactStore(workspace_root=store._workspace).list_entries() == []

    def test_a_failed_save_does_not_disturb_earlier_entries(self, store, monkeypatch):
        kept = _save(store, title="Kept frame")

        def failing_save_index(self) -> None:
            raise OSError("index is read-only")

        monkeypatch.setattr(ArtifactStore, "_save_index", failing_save_index)
        with pytest.raises(OSError):
            _save(store, title="Doomed frame")
        monkeypatch.undo()

        assert [e.id for e in store.list_entries()] == [kept.id]
        assert store.get_file_path(kept.id).exists()
        assert [p.name for p in _payload_files(store)] == sorted(
            [store.get_file_path(kept.id).name, Path(kept.data_file).name]
        ), "only the surviving entry's own pair may remain"


class TestDeletionCollectsBothFiles:
    """The companion is only useful if the entry-aware delete path finds it."""

    def test_delete_entry_removes_the_pair(self, store):
        entry = _save(store)

        assert store.delete_entry(entry.id) is True
        assert _payload_files(store) == []


class TestGalleryServesThePng:
    @pytest.fixture
    def app_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        return TestClient(app)

    def test_serve_file_returns_the_image(self, app_client):
        entry = _save(app_client.app.state.artifact_store)

        resp = app_client.get(f"/files/{entry.id}/{entry.filename}")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content == PNG_BYTES
