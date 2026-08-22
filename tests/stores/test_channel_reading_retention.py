"""``ArtifactStore.prune_channel_readings`` — the per-channel rolling window.

An oversized channel read saves an artifact every time it runs; without a
window, a polling loop grows the gallery without bound. These tests pin the
retention semantics the tool layer depends on: the window counts unpinned
readings of one address, pinned frames are exempt in both directions, other
channels are untouched, and both files of a pruned two-file entry leave disk.

They also pin the two structural properties the prune has to keep — the sweep
is attributed to the ``"system"`` actor, and delete listeners fire outside the
index lock so a listener may call back into the store.
"""

from __future__ import annotations

import pytest

from osprey.stores.artifact_store import (
    ArtifactStore,
    current_artifact_mutation_actor,
    register_artifact_delete_listener,
    unregister_artifact_delete_listener,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n-rendered-frame"
NPY_BYTES = b"\x93NUMPY-raw-payload"
CHANNEL = "SIM:CAM1:ArrayData"
OTHER_CHANNEL = "SIM:CAM2:ArrayData"


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


@pytest.fixture
def delete_events():
    """Capture (entry, actor) for every delete listener call, then unregister."""
    events: list[tuple[str, str]] = []

    def listener(entry):
        events.append((entry.id, current_artifact_mutation_actor()))

    register_artifact_delete_listener(listener)
    try:
        yield events
    finally:
        unregister_artifact_delete_listener(listener)


def _save(store: ArtifactStore, channel: str = CHANNEL, title: str = "frame"):
    return store.save_channel_reading(
        png_bytes=PNG_BYTES,
        npy_bytes=NPY_BYTES,
        title=title,
        summary={"shape": [2, 2]},
        metadata={"channel": channel},
    )


def _ids(store: ArtifactStore) -> list[str]:
    return [e.id for e in store.list_entries()]


def _entry_files(store: ArtifactStore, entry) -> list:
    """Both files an entry owns: the primary PNG and the companion ``.npy``."""
    return [store.artifact_dir / entry.filename, store._entry_data_file(entry)]


class TestWindow:
    @pytest.mark.unit
    def test_oldest_beyond_the_window_are_pruned(self, store):
        """Saving N+k readings and keeping N drops the k oldest."""
        entries = [_save(store, title=f"frame {i}") for i in range(5)]

        pruned = store.prune_channel_readings(CHANNEL, keep=3)

        assert [e.id for e in pruned] == [entries[0].id, entries[1].id]
        assert _ids(store) == [e.id for e in entries[2:]]

    @pytest.mark.unit
    def test_window_not_yet_full_is_a_no_op(self, store):
        entries = [_save(store) for _ in range(2)]

        assert store.prune_channel_readings(CHANNEL, keep=3) == []
        assert _ids(store) == [e.id for e in entries]

    @pytest.mark.unit
    def test_exactly_at_the_window_is_a_no_op(self, store):
        entries = [_save(store) for _ in range(3)]

        assert store.prune_channel_readings(CHANNEL, keep=3) == []
        assert _ids(store) == [e.id for e in entries]

    @pytest.mark.unit
    def test_repeated_saves_hold_the_window(self, store):
        """The intended call shape: prune after every save keeps the window flat."""
        kept: list[str] = []
        for i in range(6):
            entry = _save(store, title=f"frame {i}")
            kept.append(entry.id)
            store.prune_channel_readings(CHANNEL, keep=2)
            kept = kept[-2:]
            assert _ids(store) == kept

    @pytest.mark.unit
    def test_keep_zero_keeps_everything(self, store):
        """0 means unlimited retention, not 'delete everything'."""
        entries = [_save(store) for _ in range(4)]

        assert store.prune_channel_readings(CHANNEL, keep=0) == []
        assert _ids(store) == [e.id for e in entries]

    @pytest.mark.unit
    def test_negative_keep_keeps_everything(self, store):
        entries = [_save(store) for _ in range(4)]

        assert store.prune_channel_readings(CHANNEL, keep=-1) == []
        assert _ids(store) == [e.id for e in entries]

    @pytest.mark.unit
    def test_unknown_channel_prunes_nothing(self, store):
        entries = [_save(store) for _ in range(3)]

        assert store.prune_channel_readings("SIM:NOPE", keep=1) == []
        assert _ids(store) == [e.id for e in entries]

    @pytest.mark.unit
    def test_prune_survives_reload(self, store):
        """The index is persisted once, so a fresh store sees the trimmed window."""
        entries = [_save(store) for _ in range(4)]
        store.prune_channel_readings(CHANNEL, keep=1)

        reloaded = ArtifactStore(workspace_root=store._workspace)
        assert _ids(reloaded) == [entries[-1].id]


class TestPinnedExemption:
    @pytest.mark.unit
    def test_pinned_entries_are_never_pruned(self, store):
        entries = [_save(store, title=f"frame {i}") for i in range(4)]
        store.set_pinned(entries[0].id, True)

        store.prune_channel_readings(CHANNEL, keep=1)

        assert entries[0].id in _ids(store), "a pinned reading must survive the sweep"

    @pytest.mark.unit
    def test_pinned_entries_do_not_occupy_the_window(self, store):
        """keep=2 with 1 pinned + 3 unpinned leaves 2 unpinned plus the pin."""
        pinned = _save(store, title="pinned")
        store.set_pinned(pinned.id, True)
        unpinned = [_save(store, title=f"frame {i}") for i in range(3)]

        pruned = store.prune_channel_readings(CHANNEL, keep=2)

        assert [e.id for e in pruned] == [unpinned[0].id]
        assert _ids(store) == [pinned.id, unpinned[1].id, unpinned[2].id]

    @pytest.mark.unit
    def test_an_all_pinned_channel_never_prunes(self, store):
        entries = [_save(store) for _ in range(4)]
        for entry in entries:
            store.set_pinned(entry.id, True)

        assert store.prune_channel_readings(CHANNEL, keep=1) == []
        assert _ids(store) == [e.id for e in entries]


class TestScoping:
    @pytest.mark.unit
    def test_other_channels_are_untouched(self, store):
        mine = [_save(store, title=f"mine {i}") for i in range(3)]
        theirs = [_save(store, OTHER_CHANNEL, title=f"theirs {i}") for i in range(3)]

        pruned = store.prune_channel_readings(CHANNEL, keep=1)

        assert [e.id for e in pruned] == [mine[0].id, mine[1].id]
        surviving = _ids(store)
        assert all(e.id in surviving for e in theirs)
        assert mine[2].id in surviving

    @pytest.mark.unit
    def test_other_categories_are_untouched(self, store):
        """Only ``channel_values`` entries are in scope, even for the same address."""
        other = store.save_data(
            tool="archiver_fetch",
            data={"values": [1, 2, 3]},
            title="archived trace",
            category="archiver_data",
            metadata={"channel": CHANNEL},
        )
        readings = [_save(store) for _ in range(3)]

        pruned = store.prune_channel_readings(CHANNEL, keep=1)

        assert other.id not in [e.id for e in pruned]
        assert other.id in _ids(store)
        assert readings[-1].id in _ids(store)

    @pytest.mark.unit
    def test_one_dimensional_json_readings_share_the_window(self, store):
        """A ``save_data`` series and a two-file frame count in the same window."""
        series = store.save_data(
            tool="channel_read",
            data={"values": [1, 2, 3]},
            title="waveform",
            category="channel_values",
            metadata={"channel": CHANNEL},
        )
        frame = _save(store)

        pruned = store.prune_channel_readings(CHANNEL, keep=1)

        assert [e.id for e in pruned] == [series.id]
        assert _ids(store) == [frame.id]


class TestFilesRemoved:
    @pytest.mark.unit
    def test_both_files_of_a_pruned_entry_leave_disk(self, store):
        doomed = _save(store, title="doomed")
        survivor = _save(store, title="survivor")

        doomed_files = _entry_files(store, doomed)
        survivor_files = _entry_files(store, survivor)
        assert all(p is not None and p.exists() for p in doomed_files)

        store.prune_channel_readings(CHANNEL, keep=1)

        assert not any(p.exists() for p in doomed_files), "PNG and .npy must both go"
        assert all(p.exists() for p in survivor_files)


class TestActorAttribution:
    @pytest.mark.unit
    def test_prune_deletes_are_attributed_to_the_system_actor(self, store, delete_events):
        entries = [_save(store) for _ in range(3)]

        store.prune_channel_readings(CHANNEL, keep=1)

        assert [e[0] for e in delete_events] == [entries[0].id, entries[1].id]
        assert {e[1] for e in delete_events} == {"system"}

    @pytest.mark.unit
    def test_an_ordinary_delete_is_still_the_agent(self, store, delete_events):
        """The sweep's attribution is scoped to the sweep, not leaked globally."""
        entry = _save(store)

        store.delete_entry(entry.id)

        assert delete_events == [(entry.id, "agent")]


class TestListenersFireOutsideTheLock:
    @pytest.mark.unit
    def test_a_listener_may_call_back_into_the_store(self, store):
        """A delete listener that re-enters the store must not deadlock.

        ``_with_index_lock`` takes an ``flock`` on a freshly opened descriptor
        and ``flock`` is per open-file-description, so a listener firing inside
        the critical section would block forever on its own store call.
        """
        seen: list[int] = []

        def reentrant_listener(entry):
            # A public, lock-taking call from inside the listener.
            seen.append(len(store.list_entries()))
            store.set_pinned(entry.id, False)

        register_artifact_delete_listener(reentrant_listener)
        try:
            for _ in range(3):
                _save(store)
            pruned = store.prune_channel_readings(CHANNEL, keep=1)
        finally:
            unregister_artifact_delete_listener(reentrant_listener)

        assert len(pruned) == 2
        # The listener ran after the index was already trimmed and persisted.
        assert seen == [1, 1]
