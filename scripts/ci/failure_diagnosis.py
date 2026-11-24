#!/usr/bin/env python3
"""Summarise a red CI run on its pull request, and say so when it goes green.

``.github/workflows/ci-failure-diagnosis.yml`` runs this around a language
model. The model holds no GitHub token and has only file-reading tools;
everything written to GitHub is written here, in a fixed shape.

    collect   read-only. Saves each failed job's log, the failing pytest node
              IDs, which of those jobs and tests are also red on the newest
              earlier push run on main, the maintainers' open issues that name
              a failing test, and the pull request's diff. Prints
              ``diagnose=true`` only when there is a failed job to read and
              somewhere to post.
    render    turns the model's answer into the comment body, in the job that
              ran the model, so only escaped and bounded text leaves it.
    report    posts that body as one comment, updated in place: on the pull
              request the run belongs to (for a push to main, the pull request
              that merged), or on the open ``Full CI red on main`` issue for a
              dispatch on main.
    resolve   for a green run: when a later attempt of the diagnosed run
              passed on the same commit, the comment says the failures were
              flakes, which that proves; when a newer run of the pull request
              passed, it says the summary is superseded.

What the model says is a reading of the logs. The facts beside it are computed
here: red on main, a known issue, a pass on a later attempt. Nothing is filed
on the model's word.

The logs are written by whatever the run executed, and branch, job and test
names by whoever pushed, so all of it is escaped before it is posted.

Standard library only; every GitHub call goes through ``gh``, injected so the
logic runs on recorded JSON in tests.

Usage:
    python3 scripts/ci/failure_diagnosis.py collect --run-id 123 --dir ci-failure
    python3 scripts/ci/failure_diagnosis.py render --dir ci-failure
    python3 scripts/ci/failure_diagnosis.py report --dir ci-failure
    python3 scripts/ci/failure_diagnosis.py resolve --run-id 123
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# scripts/ci is not a package; flake_report owns the log parsing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flake_report

WORKFLOW = "ci.yml"
MAIN = "main"
TRACKER_TITLE = "Full CI red on main"
#: The login the workflow token writes as; only its comments and issues are ours.
BOT = "github-actions[bot]"
#: Who may say, by opening an issue, that a test is known to fail.
MAINTAINERS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
RED_CONCLUSIONS = frozenset({"failure", "timed_out"})
VERDICT_LABELS = {
    "change": "caused by the change",
    "base": "already failing on main",
    "flake": "possible flake, unproven",
    "infrastructure": "infrastructure",
    "unknown": "unclear",
}

#: Failed jobs whose logs are saved; a run with more is a broad break, and the
#: first few logs name it.
MAX_JOBS = 8
#: Failing tests looked up among the open issues.
MAX_ISSUE_LOOKUPS = 10
MAX_DIFF_CHARS = 300_000
MAX_FAILURES = 10
JOBS_PER_PAGE = 30
ESCAPES_FLAG = "--allow-escape-sequences"

MARKER = "<!-- ci-failure-diagnosis -->"
STATE_RE = re.compile(r"^<!-- run=(\d+) attempt=(\d+) state=(open|resolved) -->$", re.MULTILINE)
TIMESTAMP_RE = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT[\d:.]+Z ", re.MULTILINE)

Gh = Callable[[list[str]], str]


class GhError(RuntimeError):
    """``gh`` exited non-zero."""


def run_gh(args: list[str]) -> str:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if done.returncode:
        raise GhError(f"gh {' '.join(args[:3])} failed: {done.stderr.strip()}")
    return done.stdout


# --------------------------------------------------------------------------
# Text that is safe to post
# --------------------------------------------------------------------------


def clean_log(raw: str) -> str:
    """A job log without colour escapes or the runner's per-line timestamps."""
    return TIMESTAMP_RE.sub("", flake_report.strip_ansi(raw))


def short_name(node_id: str) -> str:
    """``tests/a.py::TestX::test_y[case]`` -> ``test_y``, the part an issue names."""
    return node_id.rsplit("::", 1)[-1].split("[", 1)[0]


def cut(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def inert(text: Any, limit: int) -> str:
    """Text for a Markdown or HTML context that renders as its characters and nothing else.

    No tag, entity, mention, link, image, autolink or issue reference survives:
    the markup characters become entities, and a zero-width space parts the
    sequences GitHub would otherwise link.
    """
    text = cut(text, limit).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("`", "'").replace("](", "]\u200b(").replace("![", "!\u200b[")
    text = text.replace("://", ":\u200b//").replace("www.", "www\u200b.")
    return re.sub(r"([@#])(?=\w)", "\\1\u200b", text)


def fenced(text: Any, limit: int) -> str:
    """Log lines as a code block, where nothing renders; the fence cannot be closed early."""
    return "```text\n" + cut(text, limit).replace("```", "'''").replace("~~~", "'''") + "\n```"


def normalise(diagnosis: Any) -> dict[str, Any]:
    """The model's answer forced into the shape the comment renders, every field bounded."""
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    failures = []
    for item in diagnosis.get("failures") or []:
        if not isinstance(item, dict):
            continue
        verdict = item.get("verdict")
        failures.append(
            {
                "job": cut(item.get("job"), 120),
                "test": cut(item.get("test"), 300),
                "verdict": verdict if verdict in VERDICT_LABELS else "unknown",
                "cause": cut(item.get("cause"), 1200),
                "evidence": cut(item.get("evidence"), 1500),
                "suggested_fix": cut(item.get("suggested_fix"), 800),
            }
        )
    return {"summary": cut(diagnosis.get("summary"), 1500), "failures": failures[:MAX_FAILURES]}


def state_line(run_id: int, attempt: int, state: str) -> str:
    return f"<!-- run={run_id} attempt={attempt} state={state} -->"


def facts(evidence: dict[str, Any], failure: dict[str, Any]) -> list[str]:
    """What the collector established about one failure, independent of the model."""
    lines = []
    main = evidence["main"]
    if main:
        where = f"[`{main['sha'][:9]}`]({main['url']})"
        if failure["test"] in main["tests"]:
            lines.append(f"This test also failed on main at {where}.")
        elif failure["job"] in main["jobs"]:
            lines.append(f"This job also failed on main at {where}.")
    known = evidence["known_issues"].get(failure["test"])
    if known:
        lines.append("Open issue naming this test: " + ", ".join(f"#{n}" for n in known[:3]) + ".")
    return lines


def render_comment(evidence: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    run = evidence["run"]
    count = evidence["failed_job_count"]
    lines = [
        MARKER,
        state_line(run["id"], run["attempt"], "open"),
        "### CI failure summary",
        "",
        f"[Run {run['id']}, attempt {run['attempt']}]({run['url']}) at `{run['sha'][:9]}` on "
        f"`{inert(run['branch'], 100)}`: {count} failed job{'s' if count != 1 else ''}.",
        "",
        inert(diagnosis["summary"], 1500) or "_No summary was produced._",
        "",
    ]
    for failure in diagnosis["failures"]:
        title = f"<b>{VERDICT_LABELS[failure['verdict']]}</b> · {inert(failure['job'], 120)}"
        if failure["test"]:
            title += f" · <code>{inert(failure['test'], 300)}</code>"
        lines += [f"<details><summary>{title}</summary>", ""]
        if failure["cause"]:
            lines += [f"**Cause.** {inert(failure['cause'], 1200)}", ""]
        if failure["evidence"]:
            lines += ["**From the log.**", "", fenced(failure["evidence"], 1500), ""]
        if failure["suggested_fix"]:
            lines += [f"**Fix.** {inert(failure['suggested_fix'], 800)}", ""]
        established = facts(evidence, failure)
        if established:
            lines += ["**Checked.** " + " ".join(established), ""]
        lines += ["</details>", ""]
    if len(evidence["jobs"]) < count:
        lines += [f"Only the first {len(evidence['jobs'])} failed jobs were read.", ""]
    lines.append(
        "<sub>Verdicts, causes and fixes are a language model's reading of this run's job logs "
        "and the diff, and can be wrong. The lines marked Checked are computed.</sub>"
    )
    return "\n".join(lines) + "\n"


def render_resolved(old: str, note: str, run_id: int, attempt: int) -> str:
    """The earlier comment folded away under a line saying what happened since."""
    earlier = "\n".join(
        line for line in old.splitlines() if line != MARKER and not STATE_RE.match(line)
    ).replace("### CI failure summary", "", 1)
    return "\n".join(
        [
            MARKER,
            state_line(run_id, attempt, "resolved"),
            "### CI failure summary",
            "",
            note,
            "",
            "<details><summary>Earlier summary</summary>",
            "",
            earlier.strip(),
            "",
            "</details>",
            "",
        ]
    )


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------


class Diagnosis:
    def __init__(self, repo: str, gh: Gh = run_gh) -> None:
        self.repo = repo
        self.gh = gh

    def _api(self, path: str, paginate: bool = False) -> Any:
        args = ["api", *(["--paginate", "--slurp"] if paginate else []), path]
        try:
            return json.loads(self.gh(args))
        except GhError as failed:
            # GitHub answers 502 now and then for the job list of a large run.
            if "HTTP 50" not in str(failed):
                raise
            return json.loads(self.gh(args))

    def _run(self, run_id: int) -> dict[str, Any]:
        return self._api(f"repos/{self.repo}/actions/runs/{run_id}")

    def _failed_jobs(self, run_id: int, attempt: int | None = None) -> list[dict[str, Any]]:
        scope = f"attempts/{attempt}/" if attempt else ""
        pages = self._api(
            f"repos/{self.repo}/actions/runs/{run_id}/{scope}jobs?per_page={JOBS_PER_PAGE}",
            paginate=True,
        )
        return [
            job
            for page in pages
            for job in page.get("jobs", [])
            if job.get("conclusion") in RED_CONCLUSIONS
            and job["name"] != flake_report.AGGREGATOR_JOB
        ]

    def _log(self, job_id: int) -> str | None:
        """The job's log, or ``None`` when it is gone or cannot be fetched."""
        args = ["api", f"repos/{self.repo}/actions/jobs/{job_id}/logs"]
        try:
            try:
                raw = self.gh(args)
            except GhError as refused:
                # Newer gh prints a coloured log only when told to; older gh
                # does not know the flag.
                if ESCAPES_FLAG not in str(refused):
                    raise
                raw = self.gh([*args, ESCAPES_FLAG])
        except GhError:
            return None
        return clean_log(raw)

    def _pull_request(self, run: dict[str, Any]) -> int | None:
        """The pull request a run speaks to: its own, or the one a push to main merged."""
        if run["event"] == "pull_request" and run.get("pull_requests"):
            return run["pull_requests"][0]["number"]
        if run["event"] not in ("pull_request", "push"):
            return None
        pulls = self._api(f"repos/{self.repo}/commits/{run['head_sha']}/pulls")
        if run["event"] == "push":
            pulls = [p for p in pulls if p.get("merged_at") and p["base"]["ref"] == MAIN]
        else:
            pulls = [p for p in pulls if p["state"] == "open"]
        return pulls[0]["number"] if pulls else None

    def _tracker(self) -> int | None:
        """The open tracker issue main-full-ci.yml keeps, which the bot itself opened."""
        pages = self._api(f"repos/{self.repo}/issues?state=open&per_page=100", paginate=True)
        numbers = [
            issue["number"]
            for page in pages
            for issue in page
            if issue["title"] == TRACKER_TITLE
            and "pull_request" not in issue
            and (issue.get("user") or {}).get("login") == BOT
        ]
        return min(numbers) if numbers else None

    def target(self, run: dict[str, Any]) -> int | None:
        """The issue or pull request number the comment goes on, if any."""
        number = self._pull_request(run)
        if number is None and run["event"] == "workflow_dispatch" and run["head_branch"] == MAIN:
            number = self._tracker()
        return number

    # -- collect -----------------------------------------------------------

    def issues_naming(self, node_id: str) -> list[int]:
        """Open issues a maintainer wrote that name the test as a whole word."""
        name = short_name(node_id)
        query = f'repo:{self.repo} is:issue is:open "{name}"'
        try:
            found = json.loads(self.gh(["api", "-X", "GET", "search/issues", "-f", f"q={query}"]))
        except GhError:
            return []
        whole = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
        return sorted(
            item["number"]
            for item in found.get("items", [])
            if item.get("author_association") in MAINTAINERS
            and whole.search(f"{item['title']}\n{item.get('body') or ''}")
        )

    def red_on_main(self, run: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Which of ``jobs`` and their tests the newest earlier push run on main also failed.

        A pull request run tests the merge with main, so what is red there is
        red here whatever the change does.
        """
        listing = self._api(
            f"repos/{self.repo}/actions/workflows/{WORKFLOW}/runs"
            f"?branch={MAIN}&event=push&status=completed&per_page=10"
        )
        earlier = [
            other
            for other in listing["workflow_runs"]
            if other["id"] != run["id"]
            and other["created_at"] <= run["created_at"]
            and other.get("conclusion") in ("success", "failure")
        ]
        if not earlier or earlier[0]["conclusion"] == "success":
            return None
        main = earlier[0]
        names = {job["name"] for job in jobs}
        ours = {node for job in jobs for node in job["tests"]}
        try:
            shared = [job for job in self._failed_jobs(main["id"]) if job["name"] in names]
        except GhError:
            return None
        if not shared:
            return None
        theirs: set[str] = set()
        for job in shared:
            theirs.update(flake_report.parse_test_failures(self._log(job["id"]) or ""))
        return {
            "sha": main["head_sha"],
            "url": main["html_url"],
            "jobs": sorted(job["name"] for job in shared),
            "tests": sorted(ours & theirs),
        }

    def collect(self, run_id: int, attempt: int | None, out: Path) -> dict[str, Any]:
        run = self._run(run_id)
        attempt = attempt or run["run_attempt"]
        failed = self._failed_jobs(run_id, attempt)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        jobs = []
        for job in failed[:MAX_JOBS]:
            log = self._log(job["id"])
            log_file = None
            if log is not None:
                log_file = f"logs/{job['id']}.log"
                (out / log_file).write_text(log, encoding="utf-8")
            jobs.append(
                {
                    "id": job["id"],
                    "name": job["name"],
                    "conclusion": job["conclusion"],
                    "failed_steps": [
                        step["name"]
                        for step in job.get("steps") or []
                        if step.get("conclusion") in RED_CONCLUSIONS
                    ],
                    "log": log_file,
                    "tests": flake_report.parse_test_failures(log or ""),
                }
            )
        number = self.target(run)
        diff, changed = "none", []
        if number is not None and run["event"] != "workflow_dispatch":
            show = ["pr", "diff", str(number), "--repo", self.repo]
            try:
                text = self.gh(show)
                changed = self.gh([*show, "--name-only"]).split()
            except GhError:  # too large for the API, or gone
                diff = "unavailable"
            else:
                diff = "truncated" if len(text) > MAX_DIFF_CHARS else "complete"
                (out / "pr.diff").write_text(text[:MAX_DIFF_CHARS], encoding="utf-8")
        node_ids = sorted({node for job in jobs for node in job["tests"] if "::" in node})
        known = {node: self.issues_naming(node) for node in node_ids[:MAX_ISSUE_LOOKUPS]}
        evidence = {
            "run": {
                "id": run_id,
                "attempt": attempt,
                "event": run["event"],
                "branch": run.get("head_branch") or "?",
                "sha": run["head_sha"],
                "url": f"{run['html_url']}/attempts/{attempt}",
            },
            "target": number,
            "diff": diff,
            "changed_files": changed,
            "failed_job_count": len(failed),
            "jobs": jobs,
            "main": self.red_on_main(run, jobs) if jobs else None,
            "known_issues": {node: issues for node, issues in known.items() if issues},
        }
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        return evidence

    # -- report and resolve ------------------------------------------------

    def _comment(self, number: int) -> dict[str, Any] | None:
        """The bot's own summary comment on an issue or pull request."""
        pages = self._api(f"repos/{self.repo}/issues/{number}/comments?per_page=100", paginate=True)
        mine = [
            comment
            for page in pages
            for comment in page
            if (comment.get("body") or "").startswith(MARKER)
            and (comment.get("user") or {}).get("login") == BOT
        ]
        return mine[-1] if mine else None

    def _edit(self, comment: dict[str, Any], body: str) -> None:
        path = f"repos/{self.repo}/issues/comments/{comment['id']}"
        self.gh(["api", "-X", "PATCH", path, "-f", f"body={body}"])

    def report(self, number: int, body: str) -> str:
        comment = self._comment(number)
        if comment:
            self._edit(comment, body)
            return f"updated the comment on #{number}"
        self.gh(["api", f"repos/{self.repo}/issues/{number}/comments", "-f", f"body={body}"])
        return f"commented on #{number}"

    def resolve(self, run_id: int, attempt: int | None) -> str:
        """Tell the comment a green run answers what that run proves about it."""
        run = self._run(run_id)
        attempt = attempt or run["run_attempt"]
        number = self._pull_request(run)
        comment = self._comment(number) if number else None
        state = STATE_RE.search(comment["body"]) if comment else None
        if comment is None or state is None or state[3] != "open":
            return "no open summary to resolve"
        url = f"{run['html_url']}/attempts/{attempt}"
        if int(state[1]) == run_id and attempt > int(state[2]):
            note = (
                f"**Flake, proven.** [Attempt {attempt}]({url}) of the same run passed on the "
                f"same commit, so the failures of attempt {state[2]} below did not come from "
                "the code."
            )
        elif run["event"] == "pull_request" and run_id > int(state[1]):
            note = f"**Superseded.** [A later run]({url}) passed at `{run['head_sha'][:9]}`."
        else:
            return "the green run says nothing about the summary"
        self._edit(comment, render_resolved(comment["body"], note, int(state[1]), int(state[2])))
        return f"resolved the comment on #{number}"


def main(argv: list[str] | None = None, gh: Gh = run_gh) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "render", "report", "resolve"):
        command = commands.add_parser(name)
        command.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
        if name in ("collect", "resolve"):
            command.add_argument("--run-id", type=int, required=True)
            command.add_argument("--attempt", type=int)
        if name != "resolve":
            command.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo or GITHUB_REPOSITORY is required")
    tool = Diagnosis(args.repo, gh=gh)

    if args.command == "resolve":
        print(tool.resolve(args.run_id, args.attempt))
        return 0
    if args.command == "collect":
        evidence = tool.collect(args.run_id, args.attempt, args.dir)
        worth_it = bool(evidence["jobs"]) and evidence["target"] is not None
        print(f"diagnose={'true' if worth_it else 'false'}")
        return 0
    evidence = json.loads((args.dir / "evidence.json").read_text(encoding="utf-8"))
    if args.command == "render":
        raw = json.loads(os.environ.get("DIAGNOSIS") or "null")
        body = render_comment(evidence, normalise(raw))
        (args.dir / "comment.md").write_text(body, encoding="utf-8")
        return 0
    body = (args.dir / "comment.md").read_text(encoding="utf-8")
    print(body)
    print(f"- {tool.report(evidence['target'], body)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
