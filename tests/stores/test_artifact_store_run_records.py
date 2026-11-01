"""The run-record verdict and the listing filter built on it.

``is_run_record`` is the one server-side rule for which artifacts record a
script run rather than hold its result. The web terminal's sidebar lists those
under "Script runs" and everything else under Artifacts, so ``list_entries``
and ``page_entries`` have to split the store exactly along that rule — and a
page's ``total`` and ``next_cursor`` have to count and walk the filtered set,
not the whole store.
"""

from __future__ import annotations

from osprey.stores.artifact_store import RUN_RECORD_KEY, ArtifactStore, is_run_record


def _save(store: ArtifactStore, title: str, **kwargs):
    """Save a small markdown artifact; ``kwargs`` override what the test pins."""
    kwargs.setdefault("artifact_type", "markdown")
    kwargs.setdefault("filename", f"{title.replace(' ', '_').replace(':', '')}.md")
    return store.save_file(file_content=b"# body", title=title, **kwargs)


class TestIsRunRecord:
    def test_the_metadata_key_marks_a_run_record(self, store):
        entry = _save(store, "Run", metadata={RUN_RECORD_KEY: True})
        assert is_run_record(entry) is True

    def test_an_absent_key_is_not_a_run_record(self, store):
        assert is_run_record(_save(store, "Result")) is False

    def test_a_false_key_is_not_a_run_record(self, store):
        entry = _save(store, "Result", metadata={RUN_RECORD_KEY: False})
        assert is_run_record(entry) is False

    def test_a_truthy_non_boolean_key_is_not_a_run_record(self, store):
        """Only the literal ``True`` the writers set counts."""
        entry = _save(store, "Result", metadata={RUN_RECORD_KEY: "yes"})
        assert is_run_record(entry) is False

    def test_the_code_output_category_marks_a_run_record(self, store):
        assert is_run_record(_save(store, "Output", category="code_output")) is True

    def test_the_approval_tool_source_marks_a_run_record(self, store):
        assert is_run_record(_save(store, "Review", tool_source="osprey_approval")) is True

    def test_an_executor_notebook_marks_a_run_record(self, store):
        entry = _save(
            store,
            "Notebook: fit the tune",
            artifact_type="notebook",
            filename="run.ipynb",
            tool_source="execute",
        )
        assert is_run_record(entry) is True

    def test_an_execute_file_notebook_marks_a_run_record(self, store):
        entry = _save(
            store,
            "Notebook: scan.py",
            artifact_type="notebook",
            filename="scan.ipynb",
            tool_source="execute_file",
        )
        assert is_run_record(entry) is True

    def test_a_notebook_without_the_title_prefix_is_not_a_run_record(self, store):
        """A notebook the user saved from execute is a result when it is not titled as a run."""
        for tool_source in ("execute", "execute_file"):
            entry = _save(
                store,
                f"Analysis via {tool_source}",
                artifact_type="notebook",
                filename=f"{tool_source}.ipynb",
                tool_source=tool_source,
            )
            assert is_run_record(entry) is False

    def test_a_prefixed_notebook_from_another_tool_is_not_a_run_record(self, store):
        entry = _save(
            store,
            "Notebook: by hand",
            artifact_type="notebook",
            filename="hand.ipynb",
            tool_source="submit_response",
        )
        assert is_run_record(entry) is False

    def test_a_prefixed_title_on_a_non_notebook_is_not_a_run_record(self, store):
        entry = _save(store, "Notebook: notes", tool_source="execute")
        assert is_run_record(entry) is False

    def test_a_plain_markdown_is_not_a_run_record(self, store):
        entry = _save(store, "Findings", tool_source="submit_response")
        assert is_run_record(entry) is False


def _mixed_store(store: ArtifactStore) -> tuple[set[str], set[str]]:
    """Five run records and seven results, saved interleaved; return their id sets."""
    runs: set[str] = set()
    others: set[str] = set()
    for i in range(12):
        if i in (0, 2, 5, 7, 11):
            runs.add(_save(store, f"Run {i}", metadata={RUN_RECORD_KEY: True}).id)
        else:
            others.add(_save(store, f"Result {i}").id)
    assert (len(runs), len(others)) == (5, 7)
    return runs, others


class TestListEntriesRunRecord:
    def test_true_lists_only_run_records(self, store):
        runs, _ = _mixed_store(store)
        assert {e.id for e in store.list_entries(run_record=True)} == runs

    def test_false_lists_everything_else(self, store):
        _, others = _mixed_store(store)
        assert {e.id for e in store.list_entries(run_record=False)} == others

    def test_none_lists_all_of_them(self, store):
        runs, others = _mixed_store(store)
        assert {e.id for e in store.list_entries(run_record=None)} == runs | others
        assert {e.id for e in store.list_entries()} == runs | others

    def test_it_stacks_with_the_other_filters(self, store):
        _save(store, "Pinned run", metadata={RUN_RECORD_KEY: True})
        pinned = _save(store, "Pinned run two", metadata={RUN_RECORD_KEY: True})
        store.set_pinned(pinned.id, True)
        _save(store, "Result")
        assert [e.id for e in store.list_entries(run_record=True, pinned=True)] == [pinned.id]

    def test_it_is_not_the_dispatch_run_filter(self, store):
        """``run_filter`` names a dispatch run by id; a run record carries none."""
        run = _save(store, "Run", metadata={RUN_RECORD_KEY: True})
        assert store.list_entries(run_filter="no-such-run") == []
        assert [e.id for e in store.list_entries(run_record=True)] == [run.id]


class TestPageEntriesRunRecord:
    @staticmethod
    def _walk(store: ArtifactStore, run_record: bool | None, limit: int):
        """Collect every page of the filtered listing; return (ids, totals, page count)."""
        ids: list[str] = []
        totals: set[int] = set()
        cursor = None
        pages = 0
        while True:
            page = store.page_entries(limit=limit, cursor=cursor, run_record=run_record)
            pages += 1
            ids.extend(e.id for e in page.entries)
            totals.add(page.total)
            cursor = page.next_cursor
            if cursor is None:
                return ids, totals, pages

    def test_the_run_records_walk_to_exactly_their_own_set(self, store):
        runs, _ = _mixed_store(store)
        ids, totals, pages = self._walk(store, True, limit=3)
        assert len(ids) == len(set(ids)) == 5
        assert set(ids) == runs
        assert totals == {5}
        assert pages == 2

    def test_the_rest_walk_to_exactly_their_own_set(self, store):
        _, others = _mixed_store(store)
        ids, totals, pages = self._walk(store, False, limit=3)
        assert len(ids) == len(set(ids)) == 7
        assert set(ids) == others
        assert totals == {7}
        assert pages == 3

    def test_unfiltered_pages_still_walk_the_whole_store(self, store):
        runs, others = _mixed_store(store)
        ids, totals, _ = self._walk(store, None, limit=3)
        assert set(ids) == runs | others
        assert totals == {12}

    def test_a_page_that_ends_the_filtered_set_has_no_cursor(self, store):
        """The other kind's entries past the last match must not leave a cursor behind."""
        _mixed_store(store)
        page = store.page_entries(limit=5, run_record=True)
        assert len(page.entries) == 5
        assert page.total == 5
        assert page.next_cursor is None

    def test_pages_are_newest_first_within_the_filter(self, store):
        _mixed_store(store)
        ids, _, _ = self._walk(store, True, limit=2)
        entries = {e.id: e for e in store.list_entries()}
        keys = [(entries[i].timestamp, i) for i in ids]
        assert keys == sorted(keys, reverse=True)
