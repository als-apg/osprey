"""Run groups are read from stored state only, per non-empty ``exec_id``.

``classify`` picks a group's head and status: a result record first, then a
preview settled ``not_run``/``failed``, else the run is in flight and its
preview (or notebook) heads it with no status. ``listed_form`` lists a head as
``runs``, hides its preview and notebook as parts, and lists an output by its
own form. Deleting a head deletes the group; deleting an output deletes only it.
"""

from __future__ import annotations

import pytest

from osprey.stores.artifact_store import (
    RUN_RECORD_KEY,
    ArtifactEntry,
    ArtifactStore,
    RunGroup,
    classify,
    listed_form,
    run_parts,
)
from osprey.stores.type_registry import ExtensionType

EXEC = "toolu_run1"


def _preview(store: ArtifactStore, exec_id: str = EXEC, **metadata) -> ArtifactEntry:
    return store.save_file(
        b"{}",
        "review.ipynb",
        "notebook",
        "Review: fit",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True, **metadata},
        exec_id=exec_id,
    )


def _notebook(store: ArtifactStore, exec_id: str = EXEC, **metadata) -> ArtifactEntry:
    return store.save_file(
        b"{}",
        "run.ipynb",
        "notebook",
        "Notebook: fit",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
        metadata={"run_part": "code", RUN_RECORD_KEY: True, **metadata},
        exec_id=exec_id,
    )


def _output(store: ArtifactStore, exec_id: str = EXEC, **kwargs) -> ArtifactEntry:
    kwargs.setdefault("metadata", {"run_part": "output", "figure": True})
    return store.save_file(
        b"png",
        "fig.png",
        "image",
        "Figure",
        mime_type="image/png",
        tool_source="execute",
        exec_id=exec_id,
        **kwargs,
    )


def _result(store: ArtifactStore, exec_id: str = EXEC, **metadata) -> ArtifactEntry:
    return store.save_data(
        tool="execute",
        data={"stdout": ""},
        title="fit",
        metadata={"run_part": "result", RUN_RECORD_KEY: True, **metadata},
        exec_id=exec_id,
    )


def _legacy_result(store: ArtifactStore, *, exec_id: str = "", failed: bool = False):
    """An executor result as written before ``run_part``: category ``code_output``."""
    entry = store.save_data(
        tool="execute",
        data={},
        title="old run",
        summary={"has_errors": failed, "status": "Failed" if failed else "Success"},
        metadata={RUN_RECORD_KEY: True},
        exec_id=exec_id,
    )
    store.update_entry_metadata(entry.id, category="code_output")
    return store.get_entry(entry.id)


class TestClassify:
    @pytest.mark.parametrize("status", ["succeeded", "failed", "not_run"])
    def test_a_result_heads_with_its_stored_status(self, store, status):
        p, n, o = _preview(store), _notebook(store), _output(store)
        r = _result(store, run_status=status)
        group = classify(store.run_members(EXEC))
        assert (group.head_id, group.run_status) == (r.id, status)
        assert group.roles == {p.id: "preview", n.id: "code", o.id: "output", r.id: "result"}

    @pytest.mark.parametrize("settled", ["not_run", "failed"])
    def test_a_settled_preview_heads_with_that_status(self, store, settled):
        p = _preview(store, settled=settled)
        _output(store)
        group = classify(store.run_members(EXEC))
        assert (group.head_id, group.run_status) == (p.id, settled)

    def test_an_unsettled_preview_is_in_flight(self, store):
        p = _preview(store)
        group = classify(store.run_members(EXEC))
        assert (group.head_id, group.run_status, group.in_flight) == (p.id, None, True)

    def test_a_preview_settled_ran_with_no_head_is_in_flight(self, store):
        p = _preview(store, settled="ran")
        _notebook(store)
        group = classify(store.run_members(EXEC))
        assert (group.head_id, group.run_status) == (p.id, None)

    def test_a_notebook_alone_heads_an_in_flight_run(self, store):
        n = _notebook(store)
        assert classify([n]) == RunGroup(n.id, None, {n.id: "code"})

    def test_a_result_wins_over_a_not_run_preview(self, store):
        _preview(store, settled="not_run")
        r = _result(store, run_status="succeeded")
        group = classify(store.run_members(EXEC))
        assert (group.head_id, group.run_status) == (r.id, "succeeded")

    @pytest.mark.parametrize(("failed", "status"), [(False, "succeeded"), (True, "failed")])
    def test_a_legacy_code_output_result_heads_by_what_it_says(self, store, failed, status):
        r = _legacy_result(store, exec_id=EXEC, failed=failed)
        _preview(store)
        group = classify(store.run_members(EXEC))
        assert (group.head_id, group.run_status) == (r.id, status)
        assert group.roles[r.id] == "result"

    def test_outputs_alone_have_no_head(self, store):
        o = _output(store)
        assert classify([o]) == RunGroup("", None, {o.id: "output"})


class TestRunIndex:
    def test_an_empty_exec_id_is_never_grouped(self, store):
        _preview(store, exec_id="")
        _output(store, exec_id="")
        a = _preview(store, exec_id="A")
        b = _output(store, exec_id="A")
        assert [e.id for e in store.run_members("A")] == [a.id, b.id]
        assert store.run_members("") == []
        assert store.run_group("") is None
        assert set(store.run_groups()) == {"A"}

    def test_run_parts_lists_every_member_but_the_head(self, store):
        p, n, o = _preview(store), _notebook(store), _output(store)
        r = _result(store, run_status="succeeded")
        parts = run_parts(store.run_group(EXEC), store.run_members(EXEC), registry={})
        assert [(x["id"], x["role"], x["form"]) for x in parts] == [
            (p.id, "preview", "documents"),
            (n.id, "code", "documents"),
            (o.id, "output", "plots"),
        ]
        assert r.id not in {x["id"] for x in parts}


class TestListedForm:
    def test_the_head_is_a_run_and_its_preview_and_notebook_are_parts(self, store):
        p, n, o = _preview(store), _notebook(store), _output(store)
        r = _result(store, run_status="succeeded")
        group = store.run_group(EXEC)
        assert listed_form(r, group) == "runs"
        assert listed_form(p, group) is None
        assert listed_form(n, group) is None
        assert listed_form(o, group) == "plots"

    def test_an_in_flight_preview_head_is_a_run(self, store):
        p = _preview(store)
        n = _notebook(store)
        group = store.run_group(EXEC)
        assert (listed_form(p, group), listed_form(n, group)) == ("runs", None)

    def test_a_kind_lists_under_its_base_form(self, store):
        o = _output(store, kind="lattice", metadata={"run_part": "output"})
        registry = {"lattice": ExtensionType("lattice", "Lattice", "tables")}
        assert listed_form(o, store.run_group(EXEC), registry) == "tables"
        assert listed_form(o, store.run_group(EXEC), {}) == "images"

    def test_a_legacy_triple_is_one_run_and_two_documents(self, store):
        preview = store.save_file(
            b"{}", "r.ipynb", "notebook", "Review", tool_source="osprey_approval"
        )
        notebook = store.save_file(
            b"{}",
            "n.ipynb",
            "notebook",
            "Notebook: fit",
            tool_source="execute",
            metadata={RUN_RECORD_KEY: True},
        )
        result = _legacy_result(store)
        assert all(e.exec_id == "" for e in (preview, notebook, result))
        forms = [listed_form(e, None, registry={}) for e in (preview, notebook, result)]
        assert sorted(forms) == ["documents", "documents", "runs"]
        assert listed_form(result, None, registry={}) == "runs"

    def test_form_filter_lists_by_listed_form(self, store):
        _preview(store)
        o = _output(store)
        r = _result(store, run_status="failed")
        other = store.save_file(b"# x", "x.md", "markdown", "Notes")
        assert [e.id for e in store.list_entries(form_filter="runs", registry={})] == [r.id]
        assert [e.id for e in store.list_entries(form_filter="plots", registry={})] == [o.id]
        assert [e.id for e in store.list_entries(form_filter="documents", registry={})] == [
            other.id
        ]
        page = store.page_entries(limit=10, form_filter="runs", registry={})
        assert (page.total, [e.id for e in page.entries]) == (1, [r.id])


class TestDelete:
    def test_deleting_a_head_deletes_the_group(self, store):
        _preview(store)
        _output(store)
        r = _result(store, run_status="succeeded")
        keep = _output(store, exec_id="OTHER")
        removed = store.delete_with_group(r.id)
        assert len(removed) == 3
        assert [e.id for e in store.list_entries()] == [keep.id]

    def test_deleting_an_in_flight_head_deletes_the_group(self, store):
        p = _preview(store)
        _notebook(store)
        assert len(store.delete_with_group(p.id)) == 2
        assert store.list_entries() == []

    def test_deleting_an_output_deletes_only_it(self, store):
        p = _preview(store)
        o = _output(store)
        assert [e.id for e in store.delete_with_group(o.id)] == [o.id]
        assert [e.id for e in store.list_entries()] == [p.id]

    def test_an_unknown_id_deletes_nothing(self, store):
        assert store.delete_with_group("nope") == []

    def test_delete_form_runs_takes_whole_groups(self, store):
        _preview(store)
        _output(store)
        _result(store, run_status="succeeded")
        doc = store.save_file(b"# x", "x.md", "markdown", "Notes")
        removed = store.delete_form("runs", registry={})
        assert len(removed) == 3
        assert [e.id for e in store.list_entries()] == [doc.id]

    def test_delete_form_of_outputs_leaves_their_run(self, store):
        p = _preview(store)
        _output(store)
        removed = store.delete_form("plots", registry={})
        assert len(removed) == 1
        assert [e.id for e in store.list_entries()] == [p.id]
