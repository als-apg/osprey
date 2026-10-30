"""Tests for the main full-CI lookup: when to dispatch, the last green sha, the tracker issue.

Every decision runs on recorded JSON through a fake ``gh``. The jobs fixture is
a revalidation dispatch of ci.yml as the Actions API returned it (name, status
and conclusion kept); the runs around it vary only the fields the logic reads.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "full_ci_on_main.py"
_spec = importlib.util.spec_from_file_location("full_ci_on_main", _MODULE_PATH)
full_ci_on_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(full_ci_on_main)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "full_ci_on_main"
REVALIDATION_JOBS: list[dict[str, Any]] = json.loads(
    (_FIXTURES / "revalidation_run_jobs.json").read_text(encoding="utf-8")
)
REPO = "example/repo"
TITLE = "Full CI red on main"


def _revalidation_jobs(**conclusions: str) -> list[dict[str, Any]]:
    """The recorded jobs, every one run, with some conclusions overridden by name prefix."""
    jobs = copy.deepcopy(REVALIDATION_JOBS)
    for job in jobs:
        if job["conclusion"] == "skipped" and "Benchmarks" not in job["name"]:
            job["conclusion"] = "success"
        for prefix, conclusion in conclusions.items():
            if job["name"].startswith(prefix):
                job["conclusion"] = conclusion
    return jobs


def _all_skipped_spending_lanes() -> list[dict[str, Any]]:
    """A dispatch that ran no spending lane: a benchmark or an e2e_filter run."""
    prefixes = tuple(full_ci_on_main.SPENDING_LANES.values())
    jobs = _revalidation_jobs()
    for job in jobs:
        if job["name"].startswith(prefixes) and not job["name"].startswith("E2E Tests"):
            job["conclusion"] = "skipped"
    return jobs


def _run(
    run_id: int,
    sha: str,
    *,
    event: str = "workflow_dispatch",
    status: str = "completed",
    conclusion: str | None = "success",
) -> dict[str, Any]:
    return {
        "id": run_id,
        "event": event,
        "head_branch": "main",
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


class FakeGh:
    """Answers the ``gh`` calls the module makes from in-memory state."""

    def __init__(
        self,
        head: str,
        runs: list[dict[str, Any]],
        jobs: dict[int, list[dict[str, Any]]],
        issues: list[dict[str, Any]] | None = None,
    ) -> None:
        self.head = head
        self.runs = runs
        self.jobs = jobs
        self.issues = issues or []
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        if args[0] == "api":
            path = args[-1]
            if path == f"repos/{REPO}/branches/main":
                return json.dumps({"commit": {"sha": self.head}})
            if path.startswith(f"repos/{REPO}/actions/workflows/ci.yml/runs?branch=main"):
                return json.dumps({"workflow_runs": self.runs})
            if path.startswith(f"repos/{REPO}/actions/runs/"):
                run_id = int(path.split("/")[5])
                return json.dumps([{"jobs": self.jobs[run_id]}])
            if path.startswith(f"repos/{REPO}/issues?state=open"):
                return json.dumps([[i for i in self.issues if i["state"] == "open"]])
        if args[:2] == ["issue", "create"]:
            number = 100 + len(self.issues)
            self.issues.append(
                {
                    "number": number,
                    "title": args[args.index("--title") + 1],
                    "body": args[args.index("--body") + 1],
                    "state": "open",
                }
            )
            return f"https://github.com/{REPO}/issues/{number}\n"
        if args[:2] == ["issue", "edit"]:
            issue = self._issue(args[2])
            issue["body"] = args[args.index("--body") + 1]
            return ""
        if args[:2] == ["issue", "close"]:
            issue = self._issue(args[2])
            issue["state"] = "closed"
            issue["closing_comment"] = args[args.index("--comment") + 1]
            return ""
        raise AssertionError(f"unexpected gh call: {args}")

    def _issue(self, number: str) -> dict[str, Any]:
        return next(i for i in self.issues if str(i["number"]) == number)

    def writes(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "issue"]


def _tool(fake: FakeGh) -> Any:
    return full_ci_on_main.FullCi(REPO, gh=fake)


# --------------------------------------------------------------------------
# should-dispatch
# --------------------------------------------------------------------------


class TestShouldDispatch:
    def test_same_sha_skips(self) -> None:
        fake = FakeGh("aaa", [_run(1, "aaa")], {1: _revalidation_jobs()})
        assert _tool(fake).should_dispatch() == f"skip https://github.com/{REPO}/actions/runs/1"

    def test_in_progress_same_sha_skips(self) -> None:
        jobs = _revalidation_jobs()
        for job in jobs:
            if job["name"].startswith("E2E Tests"):
                job.update(status="in_progress", conclusion=None)
        fake = FakeGh("aaa", [_run(2, "aaa", status="in_progress")], {2: jobs})
        assert _tool(fake).should_dispatch().startswith("skip ")

    def test_new_sha_dispatches(self) -> None:
        fake = FakeGh("bbb", [_run(1, "aaa")], {1: _revalidation_jobs()})
        assert _tool(fake).should_dispatch() == "dispatch"

    def test_no_revalidation_run_at_all_dispatches(self) -> None:
        fake = FakeGh("aaa", [_run(1, "aaa", event="push")], {1: _revalidation_jobs()})
        assert _tool(fake).should_dispatch() == "dispatch"

    def test_a_benchmark_or_filtered_dispatch_is_not_a_revalidation_run(self) -> None:
        """e2e-tests runs on every non-benchmark dispatch, so it alone marks nothing."""
        fake = FakeGh(
            "aaa",
            [_run(2, "aaa"), _run(1, "old")],
            {2: _all_skipped_spending_lanes(), 1: _revalidation_jobs()},
        )
        assert _tool(fake).should_dispatch() == "dispatch"

    def test_a_cancelled_run_on_the_head_does_not_cover_it(self) -> None:
        fake = FakeGh(
            "aaa",
            [_run(2, "aaa", conclusion="cancelled"), _run(1, "old")],
            {2: _revalidation_jobs(), 1: _revalidation_jobs()},
        )
        assert _tool(fake).should_dispatch() == "dispatch"


# --------------------------------------------------------------------------
# last-green
# --------------------------------------------------------------------------


class TestLastGreen:
    def test_the_newest_fully_green_revalidation_run(self) -> None:
        fake = FakeGh(
            "ccc",
            [
                _run(3, "ccc", conclusion="failure"),
                _run(2, "bbb", event="push"),
                _run(1, "aaa"),
            ],
            {3: _revalidation_jobs(**{"E2E Tests": "failure"}), 1: _revalidation_jobs()},
        )
        assert _tool(fake).last_green() == "aaa"

    def test_a_green_run_with_a_skipped_spending_lane_is_not_green(self) -> None:
        fake = FakeGh(
            "bbb",
            [_run(2, "bbb"), _run(1, "aaa")],
            {
                2: _revalidation_jobs(**{"Scan Stack Agentic E2E": "skipped"}),
                1: _revalidation_jobs(),
            },
        )
        assert _tool(fake).last_green() == "aaa"

    def test_a_cancelled_run_is_skipped_for_the_next_completed_one(self) -> None:
        fake = FakeGh(
            "bbb",
            [_run(2, "bbb", conclusion="cancelled"), _run(1, "aaa")],
            {2: _revalidation_jobs(), 1: _revalidation_jobs()},
        )
        assert _tool(fake).last_green() == "aaa"

    def test_one_lane_reads_any_completed_main_run_where_it_ran(self) -> None:
        """A push run where the lane skipped says nothing about it."""
        push_jobs = _revalidation_jobs(**{"Scan Stack Agentic E2E": "skipped"})
        fake = FakeGh(
            "ccc",
            [_run(3, "ccc", event="push"), _run(2, "bbb", conclusion="failure"), _run(1, "aaa")],
            {
                3: push_jobs,
                2: _revalidation_jobs(**{"Tier 3": "failure"}),
                1: _revalidation_jobs(),
            },
        )
        assert _tool(fake).last_green("scan-agentic-e2e") == "bbb"

    def test_a_matrix_lane_matches_by_prefix_and_needs_every_cell(self) -> None:
        jobs = _revalidation_jobs()
        cell = next(j for j in jobs if j["name"] == "Tier 3 — Agentic flow (hello-world)")
        cell["conclusion"] = "failure"
        fake = FakeGh(
            "bbb",
            [_run(2, "bbb", conclusion="failure"), _run(1, "aaa")],
            {2: jobs, 1: _revalidation_jobs()},
        )
        assert _tool(fake).last_green("agentic-per-preset") == "aaa"

    def test_one_lane_skips_a_cancelled_run(self) -> None:
        fake = FakeGh(
            "bbb",
            [_run(2, "bbb", conclusion="cancelled"), _run(1, "aaa")],
            {2: _revalidation_jobs(), 1: _revalidation_jobs()},
        )
        assert _tool(fake).last_green("e2e-tests") == "aaa"

    def test_none_exits_one_with_a_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake = FakeGh("aaa", [_run(1, "aaa", conclusion="failure")], {1: _revalidation_jobs()})
        assert full_ci_on_main.main(["last-green"], gh=fake, repo=REPO) == 1
        assert "no green" in capsys.readouterr().err

    def test_an_unknown_lane_is_refused(self) -> None:
        fake = FakeGh("aaa", [], {})
        with pytest.raises(SystemExit, match="unknown lane"):
            full_ci_on_main.main(["last-green", "no-such-lane"], gh=fake, repo=REPO)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _red_fake(**extra: Any) -> FakeGh:
    return FakeGh(
        "bbb",
        [_run(2, "bbb", conclusion="failure"), _run(1, "aaa")],
        {
            2: _revalidation_jobs(
                **{"Scan Stack Agentic E2E": "failure", "E2E Tests": "timed_out"}
            ),
            1: _revalidation_jobs(),
        },
        **extra,
    )


class TestReport:
    def test_red_creates_one_issue_naming_the_sha_lanes_run_and_last_green(self) -> None:
        fake = _red_fake()
        _tool(fake).report()
        [issue] = fake.issues
        assert issue["title"] == TITLE
        body = issue["body"]
        assert "bbb" in body
        assert f"https://github.com/{REPO}/actions/runs/2" in body
        assert "Scan Stack Agentic E2E" in body and "E2E Tests" in body
        assert "last green aaa" in body
        assert "Tier 0" not in body

    def test_red_again_updates_the_open_issue_without_a_second(self) -> None:
        fake = _red_fake()
        _tool(fake).report()
        fake.runs.insert(0, _run(3, "ccc", conclusion="failure"))
        fake.jobs[3] = _revalidation_jobs(**{"Dispatch Deploy E2E": "failure"})
        _tool(fake).report()
        assert len(fake.issues) == 1
        assert "ccc" in fake.issues[0]["body"]
        assert "Dispatch Deploy E2E" in fake.issues[0]["body"]

    def test_the_same_red_twice_writes_nothing_the_second_time(self) -> None:
        fake = _red_fake()
        _tool(fake).report()
        writes = len(fake.writes())
        _tool(fake).report()
        assert len(fake.writes()) == writes

    def test_green_closes_the_issue_naming_the_green_sha(self) -> None:
        fake = _red_fake()
        _tool(fake).report()
        fake.runs.insert(0, _run(3, "ccc"))
        fake.jobs[3] = _revalidation_jobs()
        _tool(fake).report()
        [issue] = fake.issues
        assert issue["state"] == "closed"
        assert "ccc" in issue["closing_comment"]

    def test_green_with_no_open_issue_is_a_no_op(self) -> None:
        fake = FakeGh("aaa", [_run(1, "aaa")], {1: _revalidation_jobs()})
        _tool(fake).report()
        assert fake.writes() == []

    def test_an_issue_with_another_title_is_left_alone(self) -> None:
        other = {"number": 7, "title": "Full CI red on main (old)", "body": "", "state": "open"}
        fake = FakeGh("aaa", [_run(1, "aaa")], {1: _revalidation_jobs()}, issues=[other])
        _tool(fake).report()
        assert fake.writes() == []

    def test_an_in_progress_run_is_not_read(self) -> None:
        """The newest COMPLETED run decides; a run still going is no verdict."""
        fake = _red_fake()
        fake.runs.insert(0, _run(3, "ccc", status="in_progress"))
        fake.jobs[3] = _revalidation_jobs()
        _tool(fake).report()
        [issue] = fake.issues
        assert "bbb" in issue["body"]

    def test_a_cancelled_run_is_skipped_for_the_next_completed_one(self) -> None:
        fake = _red_fake()
        fake.runs.insert(0, _run(3, "ccc", conclusion="cancelled"))
        fake.jobs[3] = _revalidation_jobs(**{"E2E Tests": "cancelled"})
        _tool(fake).report()
        [issue] = fake.issues
        assert "bbb" in issue["body"] and "ccc" not in issue["body"]

    def test_no_revalidation_run_yet_is_a_no_op(self) -> None:
        fake = FakeGh("aaa", [_run(1, "aaa", event="push")], {1: _revalidation_jobs()})
        _tool(fake).report()
        assert fake.writes() == []


def test_the_spending_lane_map_has_the_ten_lanes() -> None:
    assert len(full_ci_on_main.SPENDING_LANES) == 10
