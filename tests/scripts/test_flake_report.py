"""Tests for the CI flake report's pure analysis functions.

The GitHub-facing layer is deliberately untested: it is a thin `gh api`
wrapper, and mocking it would only assert that the mock was called. The
logic worth protecting is the parsing, the flip rule, and the ranking --
all of which run on recorded log text.

Log excerpts below are verbatim from als-apg/osprey CI, ANSI codes and all.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "flake_report.py"
_spec = importlib.util.spec_from_file_location("flake_report", _MODULE_PATH)
flake_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flake_report)


class TestParseTestFailures:
    def test_xdist_progress_and_summary_lines_dedupe_to_one_node_id(self):
        """The same failure is printed twice; it must count once."""
        log = (
            "2026-08-16T05:29:33.0016730Z \x1b[31m[gw3] [ 43%] FAILED\x1b[0m "
            "tests/cli/test_phase_reporter_live.py::test_a_styled_line \n"
            "2026-08-16T05:35:12.7100440Z === short test summary info ===\n"
            "2026-08-16T05:35:13.1239170Z FAILED "
            "tests/cli/test_phase_reporter_live.py::test_a_styled_line - AssertionError: assert 1\n"
        )
        assert flake_report.parse_test_failures(log) == [
            "tests/cli/test_phase_reporter_live.py::test_a_styled_line"
        ]

    def test_assertion_message_is_not_captured_into_the_node_id(self):
        """A greedy match here would make every failure look unique."""
        log = (
            "2026-08-16T05:35:13Z FAILED tests/dispatch_worker/test_retention.py"
            "::test_retention_loop_runs_one_iteration - assert [123.0, 0.1] == [123.0]\n"
        )
        assert flake_report.parse_test_failures(log) == [
            "tests/dispatch_worker/test_retention.py::test_retention_loop_runs_one_iteration"
        ]

    def test_parametrised_id_containing_spaces_is_captured_whole(self):
        log = (
            "2026-08-16T05:35:13Z FAILED tests/cli/test_lifecycle_invariants.py"
            "::test_no_shipped_text[the quick brown] - AssertionError\n"
        )
        assert flake_report.parse_test_failures(log) == [
            "tests/cli/test_lifecycle_invariants.py::test_no_shipped_text[the quick brown]"
        ]

    def test_setup_error_and_collection_error_are_both_captured(self):
        log = (
            "2026-08-03T08:07:37Z \x1b[31mERROR\x1b[0m "
            "tests/integration/test_build_boot.py::test_build_paths[hello-world]\n"
            "2026-08-03T08:07:38Z ERROR tests/va/test_record_factory.py\n"
        )
        assert flake_report.parse_test_failures(log) == [
            "tests/integration/test_build_boot.py::test_build_paths[hello-world]",
            "tests/va/test_record_factory.py",
        ]

    def test_prose_mentioning_error_is_not_a_test_failure(self):
        """Only paths under tests/ or src/ count, so log chatter is ignored."""
        log = (
            "2026-08-17T22:49:28Z echo ERROR ALS-APG endpoint probe failed\n"
            "2026-08-17T23:05:15Z ##[error]Process completed with exit code 1.\n"
            "2026-08-16T05:27:40Z [gw2] [ 11%] PASSED tests/services/test_database.py::test_up\n"
        )
        assert flake_report.parse_test_failures(log) == []


class TestSplitFlakesAndPersistent:
    @staticmethod
    def _run(attempts, final_attempt=2, run_id=1, branch="feature/x"):
        return {
            "run_id": run_id,
            "branch": branch,
            "final_attempt": final_attempt,
            "attempts": attempts,
        }

    def test_failure_that_later_passes_is_a_flake(self):
        run = self._run({1: [(10, "E2E Tests", "failure")], 2: [(11, "E2E Tests", "success")]})
        flakes, persistent = flake_report.split_flakes_and_persistent([run])
        assert [f["job_id"] for f in flakes] == [10]
        assert persistent == []

    def test_failure_that_never_passes_is_not_a_flake(self):
        """This is the distinction that keeps real branch bugs out of the table."""
        run = self._run({1: [(10, "E2E Tests", "failure")], 2: [(11, "E2E Tests", "failure")]})
        flakes, persistent = flake_report.split_flakes_and_persistent([run])
        assert flakes == []
        assert [p["job_id"] for p in persistent] == [10]

    def test_aggregator_job_is_ignored(self):
        """It fails whenever anything else does, so it would top every ranking."""
        run = self._run(
            {
                1: [(10, flake_report.AGGREGATOR_JOB, "failure")],
                2: [(11, flake_report.AGGREGATOR_JOB, "success")],
            }
        )
        assert flake_report.split_flakes_and_persistent([run]) == ([], [])

    def test_final_attempt_failures_are_not_counted_as_flakes(self):
        """A failure on the last attempt has no later attempt to prove it flaky."""
        run = self._run(
            {1: [(10, "E2E", "success")], 2: [(11, "E2E", "failure")]},
            final_attempt=2,
        )
        flakes, persistent = flake_report.split_flakes_and_persistent([run])
        assert flakes == []
        assert persistent == []

    def test_three_attempt_run_counts_both_early_failures(self):
        run = self._run(
            {
                1: [(10, "E2E", "failure")],
                2: [(11, "E2E", "failure")],
                3: [(12, "E2E", "success")],
            },
            final_attempt=3,
        )
        flakes, _ = flake_report.split_flakes_and_persistent([run])
        assert sorted(f["job_id"] for f in flakes) == [10, 11]

    def test_job_absent_from_final_attempt_is_not_a_flake(self):
        """A skipped or removed job never proved it can pass."""
        run = self._run({1: [(10, "Removed Job", "failure")], 2: []})
        flakes, persistent = flake_report.split_flakes_and_persistent([run])
        assert flakes == []
        assert len(persistent) == 1


class TestGroupFixtureFailures:
    def test_three_or_more_from_one_file_collapse_to_a_fixture_entry(self):
        """test_build_boot.py produced 6 node IDs from one uv timeout."""
        node_ids = [
            "tests/integration/test_build_boot.py::test_a[hello-world]",
            "tests/integration/test_build_boot.py::test_a[control-assistant]",
            "tests/integration/test_build_boot.py::test_b[hello-world]",
        ]
        assert flake_report.group_fixture_failures(node_ids) == [
            "tests/integration/test_build_boot.py [fixture: 3 tests]"
        ]

    def test_two_from_one_file_stay_individual(self):
        node_ids = [
            "tests/e2e/test_scan.py::test_one",
            "tests/e2e/test_scan.py::test_two",
        ]
        assert flake_report.group_fixture_failures(node_ids) == node_ids

    def test_many_collection_errors_collapse_to_one_row(self):
        """One bad import names ~100 modules; that must not bury the ranking."""
        node_ids = [f"tests/agent_runner/test_{n}.py" for n in ("a", "b", "c", "d")]
        assert flake_report.group_fixture_failures(node_ids) == [
            flake_report.COLLECTION_ERROR_LABEL
        ]

    def test_collection_label_is_independent_of_file_count(self):
        """A constant label lets recurring collection failures aggregate."""
        few = flake_report.group_fixture_failures([f"tests/x/t{n}.py" for n in range(3)])
        many = flake_report.group_fixture_failures([f"tests/y/t{n}.py" for n in range(40)])
        assert few == many == [flake_report.COLLECTION_ERROR_LABEL]

    def test_isolated_collection_error_keeps_its_filename(self):
        """A single module failing to import is a real, nameable signal."""
        assert flake_report.group_fixture_failures(["tests/va/test_record_factory.py"]) == [
            "tests/va/test_record_factory.py"
        ]

    def test_collection_errors_and_test_failures_coexist(self):
        node_ids = [
            "tests/a/t1.py",
            "tests/a/t2.py",
            "tests/a/t3.py",
            "tests/e2e/test_scan.py::test_one",
        ]
        assert flake_report.group_fixture_failures(node_ids) == [
            flake_report.COLLECTION_ERROR_LABEL,
            "tests/e2e/test_scan.py::test_one",
        ]

    def test_files_are_grouped_independently(self):
        node_ids = [
            "tests/a.py::t1",
            "tests/a.py::t2",
            "tests/a.py::t3",
            "tests/b.py::t1",
        ]
        assert flake_report.group_fixture_failures(node_ids) == [
            "tests/a.py [fixture: 3 tests]",
            "tests/b.py::t1",
        ]


class TestRankFailures:
    def test_ranked_by_distinct_runs_then_branches(self):
        records = [
            {"run_id": 1, "branch": "a", "job_name": "J", "tests": ["t_hot"]},
            {"run_id": 2, "branch": "b", "job_name": "J", "tests": ["t_hot", "t_cold"]},
            {"run_id": 3, "branch": "c", "job_name": "J", "tests": ["t_hot"]},
        ]
        ranked = flake_report.rank_failures(records)
        assert [r["test"] for r in ranked] == ["t_hot", "t_cold"]
        assert ranked[0] == {
            "test": "t_hot",
            "runs": 3,
            "branches": 3,
            "last_seen": "",
            "jobs": ["J"],
        }

    def test_last_seen_is_the_most_recent_occurrence(self):
        """Without this, a flake fixed weeks ago looks identical to a live one."""
        records = [
            {
                "run_id": 1,
                "branch": "a",
                "job_name": "J",
                "created_at": "2026-08-01T10:00:00Z",
                "tests": ["t"],
            },
            {
                "run_id": 2,
                "branch": "b",
                "job_name": "J",
                "created_at": "2026-08-16T09:00:00Z",
                "tests": ["t"],
            },
            {
                "run_id": 3,
                "branch": "c",
                "job_name": "J",
                "created_at": "2026-08-09T09:00:00Z",
                "tests": ["t"],
            },
        ]
        assert flake_report.rank_failures(records)[0]["last_seen"] == "2026-08-16"

    def test_same_test_twice_in_one_run_counts_as_one_run(self):
        """Two attempts of the same run are not two independent data points."""
        records = [
            {"run_id": 7, "branch": "a", "job_name": "J", "tests": ["t"]},
            {"run_id": 7, "branch": "a", "job_name": "J", "tests": ["t"]},
        ]
        ranked = flake_report.rank_failures(records)
        assert ranked[0]["runs"] == 1
        assert ranked[0]["branches"] == 1

    def test_branch_spread_breaks_ties_on_run_count(self):
        """Wider branch spread means less chance it is one branch's bug."""
        records = [
            {"run_id": 1, "branch": "a", "job_name": "J", "tests": ["spread", "narrow"]},
            {"run_id": 2, "branch": "b", "job_name": "J", "tests": ["spread"]},
            {"run_id": 3, "branch": "a", "job_name": "J", "tests": ["narrow"]},
        ]
        ranked = flake_report.rank_failures(records)
        assert [r["test"] for r in ranked] == ["spread", "narrow"]
        assert ranked[0]["branches"] == 2
        assert ranked[1]["branches"] == 1


class TestIterJsonDocuments:
    def test_concatenated_pages_are_all_yielded(self):
        """`gh api --paginate` writes pages back to back with no separator."""
        raw = '{"total_count":2,"workflow_runs":[{"id":1}]}{"total_count":2,"workflow_runs":[{"id":2}]}'
        ids = [
            r["id"] for doc in flake_report.iter_json_documents(raw) for r in doc["workflow_runs"]
        ]
        assert ids == [1, 2]

    def test_single_page_still_works(self):
        raw = '{"workflow_runs":[{"id":9}]}'
        assert [d["workflow_runs"][0]["id"] for d in flake_report.iter_json_documents(raw)] == [9]

    def test_whitespace_between_documents_is_tolerated(self):
        raw = '{"a":1}\n  {"a":2}\n'
        assert [d["a"] for d in flake_report.iter_json_documents(raw)] == [1, 2]

    def test_empty_input_yields_nothing(self):
        assert list(flake_report.iter_json_documents("")) == []

    def test_truncated_trailing_document_stops_cleanly(self):
        """A cut-off response should surrender the good pages, not raise."""
        raw = '{"a":1}{"a":'
        assert [d["a"] for d in flake_report.iter_json_documents(raw)] == [1]


class TestStripAnsi:
    def test_removes_colour_codes_and_keeps_text(self):
        assert flake_report.strip_ansi("\x1b[31mFAILED\x1b[0m x") == "FAILED x"

    def test_leaves_plain_text_untouched(self):
        assert flake_report.strip_ansi("no codes here") == "no codes here"


@pytest.mark.parametrize("empty", ["", "\n", "   \n\n"])
def test_empty_log_yields_no_failures(empty):
    assert flake_report.parse_test_failures(empty) == []
