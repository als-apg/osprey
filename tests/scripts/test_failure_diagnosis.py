"""Tests for the CI failure summary: what is collected, and what may be posted.

Every GitHub call goes through a fake ``gh`` that answers reads from a table of
exact API paths and records the writes. The model's answer is a dict handed to
the renderer; the tests that matter most are the ones where that dict, or a
name someone pushed, is hostile.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "failure_diagnosis.py"
_spec = importlib.util.spec_from_file_location("failure_diagnosis", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)

REPO = "example/repo"
API = f"repos/{REPO}"
NODE = "tests/a/test_x.py::TestX::test_flaps[case 1]"
OTHER_NODE = "tests/a/test_y.py::test_breaks"
JOB = "unit, rest-a"
LOG = (
    "﻿2026-11-24T16:43:31.1234567Z \x1b[31mFAILED\x1b[0m "
    f"{NODE} - AssertionError\n"
    f"2026-11-24T16:43:32.0000000Z FAILED {OTHER_NODE} - boom\n"
)
MAIN_RUNS = f"{API}/actions/workflows/{fd.WORKFLOW}/runs?branch=main&event=push&status=completed&per_page=10"


class FakeGh:
    """Reads answer from ``routes`` by exact path; anything else is a recorded write."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.writes: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        key = self._read_key(args)
        if key is None:
            self.writes.append(args)
            return "{}"
        if key not in self.routes:
            raise AssertionError(f"unexpected gh read: {key}")
        answer = self.routes[key]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)

    @staticmethod
    def _read_key(args: list[str]) -> str | None:
        if args[:2] == ["pr", "diff"]:
            return f"pr diff {args[2]}{' names' if '--name-only' in args else ''}"
        if args[0] != "api" or "PATCH" in args:
            return None
        if "search/issues" in args:
            return "search/issues"
        rest = [a for a in args[1:] if a not in ("--paginate", "--slurp", fd.ESCAPES_FLAG)]
        return rest[0] if len(rest) == 1 else None


def _job(job_id: int, name: str, conclusion: str = "failure") -> dict[str, Any]:
    steps = [
        {"name": "Run unit tests", "conclusion": conclusion},
        {"name": "Post", "conclusion": "success"},
    ]
    return {"id": job_id, "name": name, "conclusion": conclusion, "steps": steps}


def _run(run_id: int, branch: str = "feature", event: str = "pull_request", **extra: Any) -> dict:
    return {
        "id": run_id,
        "run_attempt": 1,
        "event": event,
        "head_branch": branch,
        "head_sha": f"{run_id:040d}",
        "html_url": f"https://example.test/runs/{run_id}",
        "created_at": f"2026-11-24T00:00:{run_id:02d}Z",
        "conclusion": "failure",
        "pull_requests": [],
        **extra,
    }


def _jobs_path(run_id: int, attempt: int | None = None) -> str:
    scope = f"attempts/{attempt}/" if attempt else ""
    return f"{API}/actions/runs/{run_id}/{scope}jobs?per_page={fd.JOBS_PER_PAGE}"


def _routes(**overrides: Any) -> dict[str, Any]:
    routes: dict[str, Any] = {
        f"{API}/actions/runs/50": _run(50, pull_requests=[{"number": 7}]),
        _jobs_path(50, 1): [
            {"jobs": [_job(11, JOB), _job(12, "All CI Checks Passed")]},
            {"jobs": [_job(13, "lint", "success"), _job(14, "docs", "cancelled")]},
        ],
        f"{API}/actions/jobs/11/logs": LOG,
        MAIN_RUNS: {"workflow_runs": [_run(40, "main", "push", conclusion="success")]},
        "search/issues": {"items": []},
        "pr diff 7": "diff --git a/x b/x\n",
        "pr diff 7 names": "x\nsrc/y.py\n",
    }
    routes.update(overrides)
    return routes


def _collect(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return fd.Diagnosis(REPO, gh=FakeGh(_routes(**overrides))).collect(50, 1, tmp_path)


def _answer(verdict: str = "change", **extra: Any) -> dict[str, Any]:
    failure = {"job": JOB, "test": NODE, "verdict": verdict, "cause": "c", "evidence": "e"}
    return {"summary": "s", "failures": [{**failure, **extra}]}


def _render(evidence: dict[str, Any], answer: Any) -> str:
    return fd.render_comment(evidence, fd.normalise(answer))


# -- collect ---------------------------------------------------------------


def test_collect_reads_the_failed_jobs_and_not_the_aggregator_or_a_cancelled_one(
    tmp_path: Path,
) -> None:
    evidence = _collect(tmp_path)

    assert evidence["failed_job_count"] == 1
    (job,) = evidence["jobs"]
    assert job["name"] == JOB
    assert job["failed_steps"] == ["Run unit tests"]
    assert job["tests"] == sorted([NODE, OTHER_NODE])
    assert evidence["target"] == 7
    assert evidence["diff"] == "complete"
    assert evidence["changed_files"] == ["x", "src/y.py"]
    assert evidence["main"] is None
    assert json.loads((tmp_path / "evidence.json").read_text()) == evidence


def test_collect_saves_the_log_without_timestamps_colour_or_the_byte_order_mark(
    tmp_path: Path,
) -> None:
    _collect(tmp_path)

    assert (tmp_path / "logs" / "11.log").read_text().splitlines() == [
        f"FAILED {NODE} - AssertionError",
        f"FAILED {OTHER_NODE} - boom",
    ]


def test_a_log_gh_will_not_print_is_asked_for_again_with_the_flag() -> None:
    calls: list[list[str]] = []

    def gh(args: list[str]) -> str:
        calls.append(args)
        if fd.ESCAPES_FLAG not in args:
            raise fd.GhError(f"pass {fd.ESCAPES_FLAG} to output it")
        return LOG

    log = fd.Diagnosis(REPO, gh=gh)._log(11)

    assert log is not None and NODE in log
    assert len(calls) == 2


def test_an_unfetchable_log_leaves_the_job_in_the_evidence_without_one(tmp_path: Path) -> None:
    evidence = _collect(tmp_path, **{f"{API}/actions/jobs/11/logs": fd.GhError("HTTP 410")})

    assert evidence["jobs"][0]["log"] is None
    assert evidence["jobs"][0]["tests"] == []


def test_a_job_list_answered_502_is_asked_for_once_more() -> None:
    answers = [fd.GhError("gh: Server Error (HTTP 502)"), json.dumps([{"jobs": [_job(11, JOB)]}])]

    def gh(_args: list[str]) -> str:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert [j["id"] for j in fd.Diagnosis(REPO, gh=gh)._failed_jobs(50)] == [11]


@pytest.mark.parametrize(
    ("diff", "expected"),
    [("x" * (fd.MAX_DIFF_CHARS + 1), "truncated"), (fd.GhError("HTTP 406"), "unavailable")],
)
def test_the_evidence_says_when_the_diff_is_not_whole(
    tmp_path: Path, diff: Any, expected: str
) -> None:
    evidence = _collect(tmp_path, **{"pr diff 7": diff})

    assert evidence["diff"] == expected


def test_only_the_first_failed_jobs_are_read(tmp_path: Path) -> None:
    many = [_job(100 + i, f"lane {i}") for i in range(fd.MAX_JOBS + 3)]
    logs = {f"{API}/actions/jobs/{job['id']}/logs": "" for job in many}

    evidence = _collect(tmp_path, **{_jobs_path(50, 1): [{"jobs": many}]}, **logs)

    assert evidence["failed_job_count"] == fd.MAX_JOBS + 3
    assert len(evidence["jobs"]) == fd.MAX_JOBS
    assert "Only the first" in _render(evidence, _answer())


def test_what_is_also_red_on_the_newest_earlier_main_run_is_named(tmp_path: Path) -> None:
    main_runs = [
        _run(60, "main", "push"),  # later than the run under diagnosis
        _run(45, "main", "push", conclusion="cancelled"),
        _run(40, "main", "push"),
        _run(30, "main", "push", conclusion="success"),
    ]

    evidence = _collect(
        tmp_path,
        **{
            MAIN_RUNS: {"workflow_runs": main_runs},
            _jobs_path(40): [{"jobs": [_job(41, JOB), _job(42, "another lane")]}],
            f"{API}/actions/jobs/41/logs": f"FAILED {NODE} - AssertionError\nFAILED tests/z.py::t\n",
        },
    )

    assert evidence["main"] == {
        "sha": f"{40:040d}",
        "url": "https://example.test/runs/40",
        "jobs": [JOB],
        "tests": [NODE],
    }
    body = _render(evidence, _answer("base"))
    assert "This test also failed on main at" in body
    body = _render(evidence, _answer("base", test=""))
    assert "This job also failed on main at" in body


def test_a_red_main_that_shares_no_failed_job_says_nothing(tmp_path: Path) -> None:
    evidence = _collect(
        tmp_path,
        **{
            MAIN_RUNS: {"workflow_runs": [_run(40, "main", "push")]},
            _jobs_path(40): [{"jobs": [_job(42, "another lane")]}],
        },
    )

    assert evidence["main"] is None


def test_a_green_main_says_nothing(tmp_path: Path) -> None:
    assert _collect(tmp_path)["main"] is None
    assert "**Checked.**" not in _render(_collect(tmp_path), _answer())


def test_known_issues_are_a_maintainers_and_name_the_test_as_a_whole_word(tmp_path: Path) -> None:
    def issue(number: int, text: str, association: str = "MEMBER") -> dict[str, Any]:
        return {"number": number, "title": "t", "body": text, "author_association": association}

    found = {
        "items": [
            issue(40, "see test_flaps in the macOS lane"),
            issue(30, "test_flapsy is another test"),
            issue(20, "test_flaps test_flaps test_flaps", association="NONE"),
            {
                "number": 10,
                "title": "flaky: test_flaps",
                "body": None,
                "author_association": "OWNER",
            },
        ]
    }

    evidence = _collect(tmp_path, **{"search/issues": found})

    assert evidence["known_issues"] == {NODE: [10, 40]}
    assert "Open issue naming this test: #10, #40." in _render(evidence, _answer())


@pytest.mark.parametrize(
    ("event", "branch", "pulls", "issues", "expected"),
    [
        (
            "push",
            "main",
            [{"number": 9, "merged_at": "x", "state": "closed", "base": {"ref": "main"}}],
            [],
            9,
        ),
        ("push", "main", [], [], None),
        (
            "pull_request",
            "feature",
            [{"number": 5, "merged_at": None, "state": "open", "base": {"ref": "main"}}],
            [],
            5,
        ),
        (
            "workflow_dispatch",
            "main",
            [],
            [{"number": 50, "title": fd.TRACKER_TITLE, "user": {"login": fd.BOT}}],
            50,
        ),
        (
            "workflow_dispatch",
            "main",
            [],
            [{"number": 51, "title": fd.TRACKER_TITLE, "user": {"login": "someone"}}],
            None,
        ),
        (
            "workflow_dispatch",
            "dependabot/x",
            [],
            [{"number": 50, "title": fd.TRACKER_TITLE, "user": {"login": fd.BOT}}],
            None,
        ),
    ],
)
def test_where_the_comment_goes(
    tmp_path: Path,
    event: str,
    branch: str,
    pulls: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    expected: int | None,
) -> None:
    run = _run(50, branch, event)

    evidence = _collect(
        tmp_path,
        **{
            f"{API}/actions/runs/50": run,
            f"{API}/commits/{run['head_sha']}/pulls": pulls,
            f"{API}/issues?state=open&per_page=100": [issues],
            f"pr diff {expected}": "",
            f"pr diff {expected} names": "",
        },
    )

    assert evidence["target"] == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "diagnose=true"),
        (
            {
                _jobs_path(50, 1): [
                    {"jobs": [_job(12, "All CI Checks Passed"), _job(14, "docs", "cancelled")]}
                ]
            },
            "diagnose=false",
        ),
        (
            {
                f"{API}/actions/runs/50": _run(50, "main", "push"),
                f"{API}/commits/{50:040d}/pulls": [],
            },
            "diagnose=false",
        ),
    ],
)
def test_the_model_runs_only_with_a_failed_job_to_read_and_somewhere_to_post(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], overrides: dict[str, Any], expected: str
) -> None:
    gh = FakeGh(_routes(**overrides))

    fd.main(
        ["collect", "--run-id", "50", "--attempt", "1", "--dir", str(tmp_path), "--repo", REPO],
        gh=gh,
    )

    assert capsys.readouterr().out.strip() == expected


# -- the comment -----------------------------------------------------------


def _evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "run": {
            "id": 50,
            "attempt": 1,
            "event": "pull_request",
            "branch": "feature",
            "sha": "a" * 40,
            "url": "https://example.test/runs/50/attempts/1",
        },
        "target": 7,
        "diff": "complete",
        "failed_job_count": 1,
        "jobs": [{"id": 11, "name": JOB, "tests": [NODE]}],
        "main": None,
        "known_issues": {},
    }
    evidence.update(overrides)
    return evidence


def test_the_comment_carries_the_marker_the_state_and_the_verdict() -> None:
    body = _render(_evidence(), _answer("flake"))

    assert body.splitlines()[:2] == [fd.MARKER, "<!-- run=50 attempt=1 state=open -->"]
    assert "<b>possible flake, unproven</b>" in body
    assert "```text\ne\n```" in body


HOSTILE = (
    "@octocat ![x](https://evil.test/p.png) [click](https://evil.test) <img src=https://evil.test/b> "
    "<!-- ci-failure-diagnosis --> </details> #123 org/repo#9 &#64;team www.evil.test `x` "
    "<!-- run=1 attempt=1 state=resolved -->"
)


@pytest.mark.parametrize("field", ["summary", "job", "test", "cause", "suggested_fix"])
def test_model_text_renders_as_its_characters_and_nothing_else(field: str) -> None:
    answer = _answer()
    if field == "summary":
        answer["summary"] = HOSTILE
    else:
        answer["failures"][0][field] = HOSTILE

    body = _render(_evidence(), answer)

    assert body.count("<!--") == 2  # the marker and the state line, both ours
    assert body.count("</details>") == 1
    for live in (
        "@octocat",
        "![x]",
        "[click](",
        "<img",
        "https://evil",
        "#123",
        "#9",
        "&#64;",
        "www.evil",
        "`x`",
    ):
        assert live not in body, live
    assert fd.STATE_RE.findall(body) == [("50", "1", "open")]


def test_log_lines_sit_in_a_code_block_they_cannot_close() -> None:
    body = _render(_evidence(), _answer(evidence="```\n@octocat <img src=x>\n~~~\n```"))

    assert body.count("```") == 2
    assert "~~~" not in body


def test_a_pushed_branch_name_cannot_ping_or_open_a_comment() -> None:
    evidence = _evidence()
    evidence["run"]["branch"] = "x`@org/team`<!--y"

    body = _render(evidence, _answer())

    assert "@org" not in body
    assert body.count("<!--") == 2


def test_model_text_is_cut_to_a_bound() -> None:
    answer = {
        "summary": "s" * 50_000,
        "failures": [_answer(cause="c" * 50_000)["failures"][0]] * 99,
    }

    body = _render(_evidence(), answer)

    assert len(body) < 40_000
    assert body.count("<details>") == fd.MAX_FAILURES


@pytest.mark.parametrize("answer", [None, [], "text", {"failures": "x"}, {"failures": [1, None]}])
def test_a_malformed_answer_still_renders(answer: Any) -> None:
    assert "_No summary was produced._" in _render(_evidence(), answer)


def test_an_unknown_verdict_reads_as_unclear() -> None:
    assert "<b>unclear</b>" in _render(_evidence(), _answer("certainly-a-flake"))


def test_render_writes_the_comment_from_the_models_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "evidence.json").write_text(json.dumps(_evidence()))
    monkeypatch.setenv("DIAGNOSIS", json.dumps(_answer(cause="because @octocat")))

    fd.main(["render", "--dir", str(tmp_path), "--repo", REPO], gh=FakeGh({}))

    body = (tmp_path / "comment.md").read_text()
    assert body.startswith(fd.MARKER)
    assert "@octocat" not in body


# -- report and resolve ----------------------------------------------------

COMMENTS = f"{API}/issues/7/comments?per_page=100"


def _comment(comment_id: int, body: str | None, login: str | None = fd.BOT) -> dict[str, Any]:
    return {"id": comment_id, "body": body, "user": {"login": login} if login else None}


def test_the_first_summary_is_posted_as_a_new_comment() -> None:
    gh = FakeGh({COMMENTS: [[]]})

    fd.Diagnosis(REPO, gh=gh).report(7, "body")

    assert gh.writes == [["api", f"{API}/issues/7/comments", "-f", "body=body"]]


def test_a_later_summary_replaces_the_bots_own_comment_and_no_one_elses() -> None:
    comments = [
        _comment(1, None),
        _comment(2, f"{fd.MARKER}\nold"),
        _comment(3, f"{fd.MARKER}\nquoted by a person", login="someone"),
        _comment(4, f"{fd.MARKER}\nanother bot", login="other[bot]"),
        _comment(5, f"{fd.MARKER}\ndeleted account", login=None),
    ]
    gh = FakeGh({COMMENTS: [comments]})

    fd.Diagnosis(REPO, gh=gh).report(7, "new")

    assert gh.writes == [["api", "-X", "PATCH", f"{API}/issues/comments/2", "-f", "body=new"]]


def test_report_posts_the_rendered_file_on_the_collected_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "evidence.json").write_text(json.dumps(_evidence()))
    (tmp_path / "comment.md").write_text("rendered")
    gh = FakeGh({COMMENTS: [[]]})

    fd.main(["report", "--dir", str(tmp_path), "--repo", REPO], gh=gh)

    assert gh.writes[0][-1] == "body=rendered"
    assert "commented on #7" in capsys.readouterr().out


def _resolve(run: dict[str, Any], attempt: int, comments: list[dict[str, Any]]) -> FakeGh:
    gh = FakeGh({f"{API}/actions/runs/{run['id']}": run, COMMENTS: [comments]})
    fd.Diagnosis(REPO, gh=gh).resolve(run["id"], attempt)
    return gh


def _green(run_id: int, **extra: Any) -> dict[str, Any]:
    return _run(run_id, conclusion="success", pull_requests=[{"number": 7}], **extra)


def test_a_later_attempt_passing_on_the_same_commit_proves_the_flake() -> None:
    open_body = _render(_evidence(), _answer("change"))

    gh = _resolve(_green(50), 2, [_comment(2, open_body)])

    (write,) = gh.writes
    body = write[-1].removeprefix("body=")
    assert "**Flake, proven.**" in body
    assert "<!-- run=50 attempt=1 state=resolved -->" in body
    assert "<details><summary>Earlier summary</summary>" in body
    assert "caused by the change" in body
    assert body.count(fd.MARKER) == 1


def test_a_newer_green_run_of_the_pull_request_supersedes_the_summary() -> None:
    gh = _resolve(_green(51), 1, [_comment(2, _render(_evidence(), _answer()))])

    assert "**Superseded.**" in gh.writes[0][-1]


@pytest.mark.parametrize(
    ("run", "attempt", "body"),
    [
        pytest.param(_green(51), 1, None, id="no-comment"),
        pytest.param(_green(49), 1, "open", id="an-older-run-finishing-late"),
        pytest.param(_green(50), 1, "open", id="the-same-attempt"),
        pytest.param(_green(51), 1, "resolved", id="already-resolved"),
    ],
)
def test_a_green_run_that_proves_nothing_changes_nothing(
    run: dict[str, Any], attempt: int, body: str | None
) -> None:
    rendered = _render(_evidence(), _answer())
    if body == "resolved":
        rendered = fd.render_resolved(rendered, "note", 50, 1)
    comments = [] if body is None else [_comment(2, rendered)]

    assert _resolve(run, attempt, comments).writes == []


def test_a_resolved_summary_is_replaced_whole_by_the_next_failure() -> None:
    resolved = fd.render_resolved(_render(_evidence(), _answer()), "note", 50, 1)
    gh = FakeGh({COMMENTS: [[_comment(2, resolved)]]})

    fd.Diagnosis(REPO, gh=gh).report(7, "next failure")

    assert gh.writes[0][-1] == "body=next failure"
