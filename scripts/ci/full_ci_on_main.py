#!/usr/bin/env python3
"""Full CI on main: when to start it, the last green sha, and where a red goes.

A push to main runs only the always-on lanes. Every other lane runs on main
through the revalidation dispatch of ci.yml (``revalidate_secret_lanes``),
which ``.github/workflows/main-full-ci.yml`` starts at most once a day, and only
when main has moved since the last one.

A revalidation run is recognised by its jobs: a ``workflow_dispatch`` run of
ci.yml on main in which a spending lane other than ``e2e-tests`` did not skip.
``e2e-tests`` runs on every dispatch that is not a benchmark run, so it alone
cannot tell a revalidation run from an ``e2e_filter`` one. A run that was
cancelled holds no verdict and is passed over everywhere.

Subcommands:

    should-dispatch   prints ``dispatch``, or ``skip <run url>`` when a
                      revalidation run (any status) already covers main's head
    last-green [lane] prints the newest main sha where a revalidation run and
                      every spending lane succeeded; with a lane, the newest
                      completed main run where that lane ran and succeeded
    report            opens, updates or closes the one ``Full CI red on main``
                      issue from the newest completed revalidation run

Standard library only; every GitHub call goes through ``gh``, injected so the
selection logic runs on recorded JSON in tests.

Usage:
    python3 scripts/ci/full_ci_on_main.py should-dispatch
    python3 scripts/ci/full_ci_on_main.py last-green scan-agentic-e2e
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from typing import Any

#: Spending-lane job id -> prefix of its rendered job name. A matrix lane
#: renders one job per cell, so a prefix is the only stable handle.
#: tests/deployment/test_ci_workflow_wiring.py pins this map against ci.yml.
SPENDING_LANES = {
    "agentic-per-preset": "Tier 3 — Agentic flow",
    "e2e-tests": "E2E Tests",
    "dispatch-deploy-e2e": "Dispatch Deploy E2E",
    "single-user-narrowing-e2e": "Single-User Narrowing E2E",
    "host-record-e2e": "Host Record E2E",
    "dispatch-overlay-e2e": "Dispatch Overlay E2E",
    "scan-agentic-e2e": "Scan Stack Agentic E2E",
    "nextcloud-talk-bridge-e2e": "Nextcloud Talk Bridge E2E",
    "gchat-bridge-e2e": "Google Chat Bridge E2E",
    "target-switch-agentic-e2e": "Control-Target Switch Agentic E2E",
}
#: Lanes that run on a dispatch only when ``revalidate_secret_lanes`` is set.
_REVALIDATION_MARKERS = tuple(
    prefix for lane, prefix in SPENDING_LANES.items() if lane != "e2e-tests"
)

BRANCH = "main"
WORKFLOW = "ci.yml"
TRACKER_TITLE = "Full CI red on main"
RED_CONCLUSIONS = frozenset({"failure", "cancelled", "timed_out"})
RUNS_PER_PAGE = 100

Gh = Callable[[list[str]], str]


def run_gh(args: list[str]) -> str:
    """Run ``gh`` and return its stdout; a failed call ends the program."""
    done = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if done.returncode:
        sys.exit(f"gh {' '.join(args[:3])} failed: {done.stderr.strip()}")
    return done.stdout


class NoGreen(LookupError):
    """No run on record proves what was asked."""


class FullCi:
    """The three questions, asked of one repository through one ``gh``."""

    def __init__(self, repo: str, gh: Gh = run_gh) -> None:
        self.repo = repo
        self.gh = gh
        self._jobs: dict[int, list[dict[str, Any]]] = {}

    # -- GitHub reads ------------------------------------------------------

    def _api(self, path: str, paginate: bool = False) -> Any:
        args = ["api", *(["--paginate", "--slurp"] if paginate else []), path]
        return json.loads(self.gh(args))

    def head(self) -> str:
        return self._api(f"repos/{self.repo}/branches/{BRANCH}")["commit"]["sha"]

    def runs(self) -> list[dict[str, Any]]:
        """ci.yml runs on main, newest first."""
        path = (
            f"repos/{self.repo}/actions/workflows/{WORKFLOW}/runs"
            f"?branch={BRANCH}&per_page={RUNS_PER_PAGE}"
        )
        return [r for r in self._api(path)["workflow_runs"] if r["head_branch"] == BRANCH]

    def jobs(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = run["id"]
        if run_id not in self._jobs:
            pages = self._api(
                f"repos/{self.repo}/actions/runs/{run_id}/jobs?per_page=100", paginate=True
            )
            self._jobs[run_id] = [job for page in pages for job in page.get("jobs", [])]
        return self._jobs[run_id]

    # -- classification ----------------------------------------------------

    def is_revalidation(self, run: dict[str, Any]) -> bool:
        if run["event"] != "workflow_dispatch":
            return False
        return any(
            job["name"].startswith(_REVALIDATION_MARKERS) and job.get("conclusion") != "skipped"
            for job in self.jobs(run)
        )

    def revalidation_runs(self, completed: bool) -> Iterator[dict[str, Any]]:
        """Revalidation runs with a verdict to give, newest first."""
        for run in self.runs():
            if run.get("conclusion") == "cancelled":
                continue
            if completed and run["status"] != "completed":
                continue
            if self.is_revalidation(run):
                yield run

    def _lane_succeeded(self, run: dict[str, Any], prefix: str) -> bool:
        cells = [j for j in self.jobs(run) if j["name"].startswith(prefix)]
        return bool(cells) and all(j.get("conclusion") == "success" for j in cells)

    # -- the three subcommands --------------------------------------------

    def should_dispatch(self) -> str:
        head = self.head()
        newest = next(self.revalidation_runs(completed=False), None)
        if newest is not None and newest["head_sha"] == head:
            return f"skip {newest['html_url']}"
        return "dispatch"

    def last_green(self, lane: str | None = None) -> str:
        if lane is None:
            for run in self.revalidation_runs(completed=True):
                if run["conclusion"] == "success" and all(
                    self._lane_succeeded(run, prefix) for prefix in SPENDING_LANES.values()
                ):
                    return run["head_sha"]
            raise NoGreen(f"no green revalidation run among the newest {RUNS_PER_PAGE} on main")
        prefix = SPENDING_LANES[lane]
        for run in self.runs():
            if run["status"] != "completed" or run.get("conclusion") == "cancelled":
                continue
            if run["event"] not in ("push", "workflow_dispatch"):
                continue
            if self._lane_succeeded(run, prefix):
                return run["head_sha"]
        raise NoGreen(f"no green {lane} among the newest {RUNS_PER_PAGE} runs on main")

    def open_tracker(self) -> dict[str, Any] | None:
        pages = self._api(f"repos/{self.repo}/issues?state=open&per_page=100", paginate=True)
        issues = [
            issue
            for page in pages
            for issue in page
            if issue["title"] == TRACKER_TITLE and "pull_request" not in issue
        ]
        return min(issues, key=lambda issue: issue["number"]) if issues else None

    def red_body(self, run: dict[str, Any]) -> str:
        failed = [j for j in self.jobs(run) if j.get("conclusion") in RED_CONCLUSIONS]
        try:
            green = f"last green {self.last_green()}"
        except NoGreen:
            green = "last green: none on record"
        lines = [
            f"Full CI on main is red at {run['head_sha']}.",
            "",
            f"Run: {run['html_url']}",
            "",
            "Failed jobs:",
            *(f"- {job['name']} ({job['conclusion']})" for job in failed),
            "",
            green,
        ]
        return "\n".join(lines) + "\n"

    def report(self) -> str:
        run = next(self.revalidation_runs(completed=True), None)
        if run is None:
            return "no completed revalidation run on main"
        issue = self.open_tracker()
        if run["conclusion"] == "success":
            if issue is None:
                return f"green at {run['head_sha']}"
            self.gh(
                [
                    "issue",
                    "close",
                    str(issue["number"]),
                    "--repo",
                    self.repo,
                    "--comment",
                    f"Green at {run['head_sha']}: {run['html_url']}",
                ]
            )
            return f"green at {run['head_sha']}; closed #{issue['number']}"
        body = self.red_body(run)
        if issue is None:
            self.gh(
                ["issue", "create", "--repo", self.repo, "--title", TRACKER_TITLE, "--body", body]
            )
            return f"red at {run['head_sha']}; opened the tracker issue"
        if issue.get("body") != body:
            self.gh(["issue", "edit", str(issue["number"]), "--repo", self.repo, "--body", body])
            return f"red at {run['head_sha']}; updated #{issue['number']}"
        return f"red at {run['head_sha']}; #{issue['number']} is current"


def _default_repo(gh: Gh) -> str:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    return json.loads(gh(["repo", "view", "--json", "nameWithOwner"]))["nameWithOwner"]


def main(argv: list[str] | None = None, gh: Gh = run_gh, repo: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("should-dispatch")
    last = commands.add_parser("last-green")
    last.add_argument("lane", nargs="?")
    commands.add_parser("report")
    args = parser.parse_args(argv)

    if args.command == "last-green" and args.lane and args.lane not in SPENDING_LANES:
        sys.exit(f"unknown lane {args.lane!r}; one of: {', '.join(sorted(SPENDING_LANES))}")
    tool = FullCi(repo or _default_repo(gh), gh=gh)
    if args.command == "should-dispatch":
        print(tool.should_dispatch())
    elif args.command == "last-green":
        try:
            print(tool.last_green(args.lane))
        except NoGreen as missing:
            print(missing, file=sys.stderr)
            return 1
    else:
        print(tool.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
