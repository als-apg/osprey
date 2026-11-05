"""Entry fields, save-path stamps, metadata merges, visibility and ownership.

An entry carries ``exec_id`` (its run) and ``kind`` (its extension type), both
empty on an index written before they existed. Every save stamps
``metadata.first_ts`` once (a sha256 touch never refreshes it), stamps
``metadata.call_id`` inside a tool call's scope, and takes ``origin`` from
``OSPREY_ARTIFACT_ORIGIN`` unless the caller named one. ``owner_of`` is the one
ownership rule, ``operator_visible`` the one visibility filter, and
``category`` on the save APIs is accepted, ignored and warned.
"""

from __future__ import annotations

import json
import warnings
from collections import namedtuple
from datetime import UTC, datetime, timedelta, timezone

import pytest

from osprey.audit.call import call_scope
from osprey.stores.artifact_store import (
    CATEGORY_DEPRECATION,
    WATCHDOG_ORIGIN,
    ArtifactEntry,
    ArtifactStore,
    in_window,
    operator_visible,
    owner_of,
)


def _save(store: ArtifactStore, title: str = "A", **kwargs) -> ArtifactEntry:
    kwargs.setdefault("artifact_type", "markdown")
    kwargs.setdefault("filename", "a.md")
    return store.save_file(file_content=b"# body", title=title, **kwargs)


@pytest.fixture(autouse=True)
def _no_origin_env(monkeypatch):
    for key in ("OSPREY_ARTIFACT_ORIGIN", "OSPREY_SESSION_ID", "OSPREY_DISPATCH_RUN_ID"):
        monkeypatch.delenv(key, raising=False)


class TestEntryFields:
    def test_exec_id_and_kind_round_trip_through_the_index(self, store, agent_data_root):
        saved = _save(store, exec_id="toolu_1", kind="lattice")
        reread = ArtifactStore(workspace_root=agent_data_root).get_entry(saved.id)
        assert (reread.exec_id, reread.kind) == ("toolu_1", "lattice")

    def test_they_default_to_empty(self, store):
        entry = _save(store)
        assert (entry.exec_id, entry.kind) == ("", "")

    def test_an_old_index_without_them_loads_with_empty_defaults(self, store, agent_data_root):
        legacy = {
            "id": "abc123def456",
            "artifact_type": "markdown",
            "title": "Old",
            "description": "",
            "filename": "abc123def456_old.md",
            "mime_type": "text/markdown",
            "size_bytes": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "tool_source": "execute",
            "metadata": {},
            "highlighted": True,
        }
        store.artifact_dir.mkdir(parents=True, exist_ok=True)
        (store.artifact_dir / "artifacts.json").write_text(json.dumps({"entries": [legacy]}))
        entry = ArtifactStore(workspace_root=agent_data_root).get_entry("abc123def456")
        assert (entry.exec_id, entry.kind, entry.origin) == ("", "", "")

    @pytest.mark.parametrize(
        "save",
        [
            pytest.param(
                lambda s: s.save_object("x", title="T", exec_id="e", kind="k"), id="object"
            ),
            pytest.param(
                lambda s: s.save_data(tool="t", data={}, title="T", exec_id="e", kind="k"),
                id="data",
            ),
            pytest.param(
                lambda s: s.save_channel_reading(
                    b"png", b"npy", title="T", metadata={"channel": "C"}, exec_id="e", kind="k"
                ),
                id="channel-reading",
            ),
            pytest.param(
                lambda s: s.save_or_touch_by_sha256(
                    "h",
                    origin="upload",
                    save_kwargs={
                        "file_content": b"x",
                        "filename": "x.txt",
                        "artifact_type": "text",
                        "title": "T",
                        "metadata": {"sha256": "h"},
                        "exec_id": "e",
                        "kind": "k",
                    },
                ),
                id="sha256",
            ),
        ],
    )
    def test_every_save_path_accepts_both(self, store, save):
        entry = save(store)
        assert (entry.exec_id, entry.kind) == ("e", "k")

    def test_save_from_path_accepts_both(self, store, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("hi")
        entry = store.save_from_path(src, title="T", exec_id="e", kind="k")
        assert (entry.exec_id, entry.kind) == ("e", "k")

    def test_they_are_keyword_only(self, store):
        with pytest.raises(TypeError):
            store.save_file(
                b"x",
                "x.md",
                "markdown",
                "T",
                "",
                "text/markdown",
                "t",
                None,
                "",
                None,
                "",
                None,
                "e",
            )


class TestFirstTs:
    def test_a_save_stamps_first_ts_as_its_timestamp(self, store):
        entry = _save(store)
        assert entry.metadata["first_ts"] == entry.timestamp

    def test_save_data_stamps_it_too(self, store):
        entry = store.save_data(tool="t", data={}, title="T")
        assert entry.metadata["first_ts"] == entry.timestamp

    def test_a_touch_keeps_first_ts_and_moves_the_timestamp(self, store):
        kwargs = {
            "file_content": b"x",
            "filename": "x.txt",
            "artifact_type": "text",
            "title": "T",
            "metadata": {"sha256": "h"},
        }
        first = store.save_or_touch_by_sha256("h", origin="upload", save_kwargs=kwargs)
        stamped = first.metadata["first_ts"]
        touched = store.save_or_touch_by_sha256("h", origin="upload", save_kwargs=kwargs)
        assert touched.id == first.id
        assert touched.metadata["first_ts"] == stamped
        assert touched.timestamp >= stamped


class TestCallId:
    def test_inside_a_call_scope_the_tool_use_id_is_stamped(self, store):
        with call_scope("toolu_abc", None):
            entry = _save(store)
        assert entry.metadata["call_id"] == "toolu_abc"

    def test_outside_a_call_scope_nothing_is_stamped(self, store):
        assert "call_id" not in _save(store).metadata

    def test_a_scope_without_an_id_stamps_nothing(self, store):
        with call_scope(None, None):
            assert "call_id" not in _save(store).metadata


class TestOrigin:
    def test_the_env_supplies_the_origin(self, store, monkeypatch):
        monkeypatch.setenv("OSPREY_ARTIFACT_ORIGIN", "watchdog")
        assert _save(store).origin == "watchdog"
        assert store.save_data(tool="t", data={}, title="T").origin == "watchdog"

    def test_an_explicit_origin_wins_over_the_env(self, store, monkeypatch):
        monkeypatch.setenv("OSPREY_ARTIFACT_ORIGIN", "watchdog")
        assert _save(store, origin="input").origin == "input"

    def test_no_env_no_origin(self, store):
        assert _save(store).origin == ""


class TestCategoryDeprecation:
    def test_a_category_warns_and_is_not_stored(self, store):
        with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
            entry = _save(store, category="plots")
        assert entry.category == ""

    @pytest.mark.parametrize(
        "save",
        [
            pytest.param(lambda s: s.save_object("x", title="T", category="c"), id="object"),
            pytest.param(
                lambda s: s.save_data(tool="t", data={}, title="T", category="c"), id="data"
            ),
            pytest.param(
                lambda s: s.save_channel_reading(
                    b"p", b"n", title="T", metadata={"channel": "C"}, category="c"
                ),
                id="channel-reading",
            ),
        ],
    )
    def test_every_save_path_warns_once(self, store, save):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entry = save(store)
        hits = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert [str(w.message) for w in hits] == [CATEGORY_DEPRECATION]
        assert entry.category == ""

    def test_no_category_no_warning(self, store):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _save(store)
            store.save_channel_reading(b"p", b"n", title="T", metadata={"channel": "C"})

    def test_channel_readings_still_prune_without_a_category(self, store):
        ids = [
            store.save_channel_reading(b"p", b"n", title=f"T{i}", metadata={"channel": "C"}).id
            for i in range(3)
        ]
        pruned = store.prune_channel_readings("C", keep=1)
        assert [e.id for e in pruned] == ids[:2]

    def test_the_in_script_save_artifact_warns_and_drops_it(self, tmp_path):
        from osprey.stores.artifact_manifest import SAVE_ARTIFACT_SOURCE

        scope: dict = {"_execution_dir": tmp_path}
        exec(SAVE_ARTIFACT_SOURCE, scope)
        with pytest.warns(DeprecationWarning):
            scope["save_artifact"]("hello", title="T", category="c")
        manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text())
        assert "category" not in manifest[0]


class TestMergeEntryMetadata:
    def test_two_instances_merge_without_losing_each_others_keys(self, store, agent_data_root):
        entry = _save(store, metadata={"keep": 1})
        other = ArtifactStore(workspace_root=agent_data_root)
        assert store.merge_entry_metadata({entry.id: {"settled": "ran"}}) == 1
        assert other.merge_entry_metadata({entry.id: {"transcript_id": "T1"}}) == 1
        final = ArtifactStore(workspace_root=agent_data_root).get_entry(entry.id).metadata
        assert final["keep"] == 1
        assert final["settled"] == "ran"
        assert final["transcript_id"] == "T1"
        assert "first_ts" in final

    def test_it_counts_only_changed_entries_and_ignores_unknown_ids(self, store):
        a = _save(store, metadata={"x": 1})
        b = _save(store)
        assert store.merge_entry_metadata({a.id: {"x": 1}, b.id: {"y": 2}, "nope": {"z": 3}}) == 1

    def test_one_index_write(self, store, monkeypatch):
        a, b = _save(store), _save(store)
        calls = []
        original = store._save_index
        monkeypatch.setattr(store, "_save_index", lambda: (calls.append(1), original())[1])
        store.merge_entry_metadata({a.id: {"k": 1}, b.id: {"k": 2}})
        assert calls == [1]

    def test_update_entry_metadata_stays_an_attribute_setter(self, store):
        entry = _save(store)
        assert store.update_entry_metadata(entry.id, pinned=True).pinned is True
        with pytest.raises(AttributeError):
            store.update_entry_metadata(entry.id, settled="ran")


class TestOperatorVisible:
    @pytest.mark.parametrize(
        ("kwargs", "visible"),
        [
            pytest.param({}, True, id="own-output"),
            pytest.param({"run_id": "r1"}, False, id="dispatch"),
            pytest.param({"origin": WATCHDOG_ORIGIN}, False, id="watchdog"),
            pytest.param({"origin": "upload"}, True, id="upload"),
        ],
    )
    def test_dispatch_and_watchdog_are_hidden(self, store, kwargs, visible):
        assert operator_visible(_save(store, **kwargs)) is visible


Row = namedtuple("Row", "session_key transcript_id from_ts")


class _Ledger:
    def __init__(self, rows):
        self.rows = rows


T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def _entry(store, *, first_ts=None, session_id="K", timestamp=None, **metadata):
    entry = _save(store, session_id=session_id)
    if first_ts is None:
        entry.metadata.pop("first_ts", None)
    else:
        entry.metadata["first_ts"] = first_ts.isoformat()
    if timestamp is not None:
        entry.timestamp = timestamp.isoformat()
    entry.metadata.update(metadata)
    return entry


class TestOwnerOf:
    LEDGER = _Ledger(
        [
            Row("K", "T1", T0.isoformat()),
            Row("K", "T2", (T0 + timedelta(hours=1)).isoformat()),
            ("K", "T3", T0 + timedelta(hours=2)),
            Row("OTHER", "X", T0.isoformat()),
        ]
    )

    def test_a_stamped_transcript_wins(self, store):
        entry = _entry(store, first_ts=T0, transcript_id="TU")
        assert owner_of(entry, self.LEDGER) == "TU"

    @pytest.mark.parametrize(
        ("offset", "owner"),
        [
            pytest.param(timedelta(minutes=30), "T1", id="first-row"),
            pytest.param(timedelta(hours=1), "T2", id="on-the-bound"),
            pytest.param(timedelta(hours=1, minutes=59), "T2", id="second-row"),
            pytest.param(timedelta(hours=5), "T3", id="latest-row"),
            pytest.param(timedelta(hours=-3), "T1", id="open-lower-bound"),
        ],
    )
    def test_key_and_first_ts_pick_the_row(self, store, offset, owner):
        assert owner_of(_entry(store, first_ts=T0 + offset), self.LEDGER) == owner

    def test_times_are_compared_as_datetimes_not_strings(self, store):
        # 13:30+01:00 is 12:30 UTC: still the first row, though it sorts later as text.
        first = datetime(2026, 11, 1, 13, 30, tzinfo=timezone(timedelta(hours=1)))
        assert owner_of(_entry(store, first_ts=first), self.LEDGER) == "T1"

    def test_a_legacy_entry_without_first_ts_uses_its_timestamp(self, store):
        entry = _entry(store, timestamp=T0 + timedelta(hours=1, minutes=5))
        assert "first_ts" not in entry.metadata
        assert owner_of(entry, self.LEDGER) == "T2"

    def test_an_unknown_key_has_no_owner(self, store):
        assert owner_of(_entry(store, first_ts=T0, session_id="NEVER"), self.LEDGER) is None

    def test_an_untagged_entry_has_no_owner(self, store):
        assert owner_of(_entry(store, first_ts=T0, session_id=""), self.LEDGER) is None

    def test_no_ledger_only_the_stamp_counts(self, store):
        assert owner_of(_entry(store, first_ts=T0), None) is None
        assert owner_of(_entry(store, first_ts=T0, transcript_id="TU"), None) == "TU"


class TestTimeWindow:
    def test_the_window_is_half_open(self, store):
        entry = _entry(store, first_ts=T0, timestamp=T0)
        assert in_window(entry, T0.isoformat(), None)
        assert not in_window(entry, None, T0.isoformat())
        assert in_window(entry, None, (T0 + timedelta(seconds=1)).isoformat())

    def test_a_naive_bound_is_refused(self, store):
        with pytest.raises(ValueError, match="aware"):
            in_window(_save(store), "2026-11-01T00:00:00", None)

    def test_list_entries_filters_by_window(self, store):
        old = _save(store)
        store.update_entry_metadata(old.id, timestamp=(T0 - timedelta(days=1)).isoformat())
        new = _save(store)
        ids = {e.id for e in store.list_entries(after=T0.isoformat())}
        assert ids == {new.id}


class TestWatchdogOverlay:
    def test_one_run_carries_the_watchdog_origin_and_one_terminal_session(self):
        from osprey.build.claude_code_telemetry import (
            DISPATCH_SESSION_PREFIX,
            HOST_SESSION_PREFIX,
            WATCHDOG_SESSION_PREFIX,
        )
        from osprey.services.watchdog.engines.agent import _run_env

        assert WATCHDOG_SESSION_PREFIX not in (DISPATCH_SESSION_PREFIX, HOST_SESSION_PREFIX)
        env = _run_env({"OSPREY_ARTIFACT_ORIGIN": "", "OSPREY_TERMINAL_SESSION": "chat-x"})
        assert env["OSPREY_ARTIFACT_ORIGIN"] == "watchdog"
        assert env["OSPREY_TERMINAL_SESSION"].startswith(WATCHDOG_SESSION_PREFIX)
        assert _run_env({}, "watchdog-run")["OSPREY_TERMINAL_SESSION"] == "watchdog-run"
        assert _run_env({})["OSPREY_TERMINAL_SESSION"] != env["OSPREY_TERMINAL_SESSION"]
