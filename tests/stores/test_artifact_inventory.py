"""Inventory helpers list a set of artifacts the way the gallery sidebar does.

``listed_rows`` classifies runs over the whole index and drops exactly the run
parts; ``entry_channels`` de-duplicates in order; ``form_totals`` keeps
first-seen order.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osprey.stores.artifact_inventory import (
    entry_channels,
    form_totals,
    listed_rows,
)
from osprey.stores.artifact_store import RUN_RECORD_KEY, ArtifactEntry, ArtifactStore, owner_of
from osprey.stores.type_registry import ExtensionType

EXEC = "toolu_inv1"


def _preview(store: ArtifactStore, **metadata) -> ArtifactEntry:
    return store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True, **metadata},
        exec_id=EXEC,
    )


def _notebook(store: ArtifactStore, **metadata) -> ArtifactEntry:
    return store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook: fit",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
        metadata={"run_part": "code", RUN_RECORD_KEY: True, **metadata},
        exec_id=EXEC,
    )


def _output(store: ArtifactStore, **metadata) -> ArtifactEntry:
    return store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure",
        mime_type="image/png",
        tool_source="execute",
        metadata={"run_part": "output", "figure": True, **metadata},
        exec_id=EXEC,
    )


def _result(store: ArtifactStore, **metadata) -> ArtifactEntry:
    return store.save_data(
        tool="execute",
        data={"stdout": ""},
        title="fit",
        metadata={
            "run_part": "result",
            RUN_RECORD_KEY: True,
            "run_status": "succeeded",
            **metadata,
        },
        exec_id=EXEC,
    )


def _ids(rows) -> list[str]:
    return [entry.id for entry, _key, _label in rows]


class TestListedRows:
    def test_a_runs_preview_and_notebook_are_dropped_head_and_outputs_stay(self, store):
        p, n, o, r = _preview(store), _notebook(store), _output(store), _result(store)
        notes = store.save_file(b"# x", "x.md", "markdown", "Notes")
        rows = listed_rows(store, [p, n, o, r, notes], registry={})
        assert _ids(rows) == [o.id, r.id, notes.id]
        assert [(key, label) for _e, key, label in rows] == [
            ("plots", "Plots"),
            ("runs", "Runs"),
            ("documents", "Documents"),
        ]

    def test_a_run_split_by_a_clear_gives_the_earlier_transcript_no_head(self, store):
        # The preview was saved in transcript A; the result after a /clear, in B.
        _preview(store, transcript_id="A")
        r = _result(store, transcript_id="B")
        index = store.list_entries()
        owned_a = [e for e in index if owner_of(e, None) == "A"]
        owned_b = [e for e in index if owner_of(e, None) == "B"]
        assert listed_rows(store, owned_a, registry={}) == []
        assert _ids(listed_rows(store, owned_b, registry={})) == [r.id]

    def test_runs_are_classified_over_the_whole_index_not_the_selection(self, store):
        p = _preview(store)
        _result(store)
        # Selected alone the preview would look like an in-flight head; the
        # index says the result heads the run, so the preview is a part.
        assert listed_rows(store, [p], registry={}) == []

    @pytest.mark.parametrize(
        "registry",
        [
            {"lattice": ExtensionType("lattice", "Lattice Model", "tables")},
            {"lattice": {"label": "Lattice Model", "base_form": "tables"}},
            {"lattice": SimpleNamespace(label="Lattice Model", base_form="tables")},
        ],
    )
    def test_an_extension_type_is_keyed_and_labelled_by_its_kind(self, store, registry):
        lat = store.save_file(b"{}", "l.json", "json", "Lattice", kind="lattice")
        assert listed_rows(store, [lat], registry=registry) == [(lat, "lattice", "Lattice Model")]

    def test_no_registry_uses_the_process_registry(self, store, monkeypatch):
        from osprey.stores import type_registry

        lat = store.save_file(b"{}", "l.json", "json", "Lattice", kind="lattice")
        registry = {"lattice": ExtensionType("lattice", "Lattice Model", "tables")}
        monkeypatch.setattr(type_registry, "get_extension_types", lambda: registry)
        assert listed_rows(store, [lat]) == [(lat, "lattice", "Lattice Model")]

    def test_an_empty_selection_has_no_rows(self, store):
        _preview(store)
        assert listed_rows(store, [], registry={}) == []


class TestEntryChannels:
    def test_channels_are_deduplicated_in_order_across_summary_and_access(self, store):
        e = store.save_data(
            tool="archiver_read",
            data={},
            title="t",
            summary={"channels": ["B", "A", "B"], "columns": ["C"]},
            access_details={"pvs": ["A", "D"], "channel_names": "not-a-list"},
        )
        assert entry_channels(e) == ["B", "A", "C", "D"]

    def test_an_entry_without_channels_has_none(self, store):
        e = store.save_file(b"# x", "x.md", "markdown", "Notes")
        assert entry_channels(e) == []


class TestFormTotals:
    def test_totals_keep_first_seen_order(self):
        pairs = [
            ("plots", "Plots"),
            ("runs", "Runs"),
            ("plots", "Plots"),
            ("lattice", "Lattice Model"),
            ("runs", "Runs"),
        ]
        assert form_totals(pairs) == [
            {"key": "plots", "label": "Plots", "count": 2},
            {"key": "runs", "label": "Runs", "count": 2},
            {"key": "lattice", "label": "Lattice Model", "count": 1},
        ]

    def test_totals_take_any_iterable_of_pairs(self):
        assert form_totals(iter([("runs", "Runs")])) == [
            {"key": "runs", "label": "Runs", "count": 1}
        ]
        assert form_totals([]) == []

    def test_totals_fed_from_listed_rows(self, store):
        _preview(store)
        o, r = _output(store), _result(store)
        rows = listed_rows(store, store.list_entries(), registry={})
        assert form_totals((key, label) for _e, key, label in rows) == [
            {"key": "plots", "label": "Plots", "count": 1},
            {"key": "runs", "label": "Runs", "count": 1},
        ]
        assert {o.id, r.id} == set(_ids(rows))
