#!/usr/bin/env python3
"""Refuse a release tag unless one ci.yml run proved the named lanes on its tree.

Both release workflows run this before they build anything:
``release.yml`` (framework ``v*`` tags) names the model-spending lanes, and
``release-connectors.yml`` (``osprey-connectors-v*`` tags) names the lanes that
prove the connectors package on its own. Nothing in either workflow runs a test,
so without this a tag can publish a tree whose lanes never ran: every
model-spending lane is gated on the ``full-ci`` label, and ``all-checks-passed``
counts a conditionally skipped lane as a pass.

What the lanes proved is a TREE, not a commit. A ``--merge`` merge commit has a
different sha from the PR head whose run carried the label, and a tag may sit a
commit or two above it, so every nearby commit with the tagged commit's tree is
an equally good place to find the run. Every successful run on such a commit is
checked, not only the newest: a later re-run without the label is green and
empty, and it must not hide the labeled run that actually did the work.

Lanes are given one per line as ``<ci.yml job id>=<rendered job-name prefix>``.
The gate matches a prefix because a matrix job (``test``,
``agentic-per-preset``) renders one job per cell, each named after its cell.
tests/deployment/test_ci_workflow_wiring.py pins every workflow's lane list
against ci.yml, so a renamed lane fails a unit test, not a release.

Usage (the workflow passes the context through ``env:``, never ``${{ }}``)::

    python3 scripts/ci/require_green_lanes.py --lanes "$REQUIRED_LANES" \\
        [--hint "<extra remedy line>" ...]

``--repo``, ``--sha`` and ``--tag`` default to ``$REPO``, ``$TAG_SHA`` and
``$TAG_NAME``; ``gh`` reads its token from ``$GH_TOKEN``.

Stdlib-only, and run with the runner's bare ``python3``: the gate has to report
even where nothing else is installed yet.

Exit codes: 0 a run proved every lane, 1 none did, 2 the arguments or the
GitHub API could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

#: Commits below the tag along its first-parent line whose tree is checked too.
NEARBY_FIRST_PARENT = 3

#: ``git <args>`` -> stdout, stripped. Injected by the tests.
Git = Callable[..., str]
#: ``gh api <path>`` -> parsed JSON. Injected by the tests.
Api = Callable[[str, bool], Any]


class GateError(RuntimeError):
    """Something the gate could not read: a malformed lane list, a failing API."""


def parse_lanes(text: str) -> dict[str, str]:
    """``"<job id>=<prefix>"`` lines -> ``{job id: prefix}``.

    Blank lines are skipped; a line without ``=``, an empty side, or a job id
    given twice is refused, because each is a lane list that would gate on
    something other than what it says.
    """
    lanes: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lane, sep, prefix = (part.strip() for part in line.partition("="))
        if not sep or not lane or not prefix:
            raise GateError(f"lane line {raw!r} is not <ci.yml job id>=<job-name prefix>")
        if lane in lanes:
            raise GateError(f"lane {lane!r} is listed twice")
        lanes[lane] = prefix
    if not lanes:
        raise GateError("no lanes given: a gate over nothing proves nothing")
    return lanes


def same_tree_candidates(sha: str, git: Git) -> list[str]:
    """The tagged commit and every nearby commit carrying its exact tree, deduplicated."""
    tag_tree = git("rev-parse", f"{sha}^{{tree}}")
    nearby = git("log", "--first-parent", "-n", str(NEARBY_FIRST_PARENT), "--format=%H", sha)
    nearby_shas = nearby.split() + git("rev-parse", f"{sha}^@").split()
    candidates: list[str] = []
    seen: set[str] = set()
    for raw in [sha, *nearby_shas]:
        full = git("rev-parse", f"{raw}^{{commit}}")  # full form, so dedupe works
        if full in seen:
            continue
        seen.add(full)
        if git("rev-parse", f"{full}^{{tree}}") == tag_tree:
            candidates.append(full)
    return candidates


def missing_lanes(jobs: Iterable[Mapping[str, Any]], lanes: Mapping[str, str]) -> dict[str, str]:
    """Lane ids whose jobs are absent, skipped, or not all successful, with why."""
    jobs = list(jobs)
    bad: dict[str, str] = {}
    for lane, prefix in lanes.items():
        matches = [job for job in jobs if str(job.get("name", "")).startswith(prefix)]
        if not matches:
            bad[lane] = "no such job in the run"
        elif any(job.get("conclusion") != "success" for job in matches):
            bad[lane] = ", ".join(sorted({str(job.get("conclusion")) for job in matches}))
    return bad


def find_green_run(
    repo: str, candidates: Sequence[str], lanes: Mapping[str, str], api: Api
) -> tuple[str | None, list[str]]:
    """``(url, report)``: the first successful ci.yml run proving every lane, or
    ``None`` and one report line per successful run that did not."""
    report: list[str] = []
    for sha in candidates:
        runs = api(f"repos/{repo}/actions/workflows/ci.yml/runs?head_sha={sha}&per_page=20", False)
        for run in runs.get("workflow_runs", []):
            if run.get("conclusion") != "success":
                continue
            pages = api(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100", True)
            jobs = [job for page in pages for job in page.get("jobs", [])]
            bad = missing_lanes(jobs, lanes)
            if not bad:
                return str(run["html_url"]), report
            report.append(
                f"  {run['html_url']} — " + "; ".join(f"{k}: {v}" for k, v in sorted(bad.items()))
            )
    return None, report


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True)
    if done.returncode:
        raise GateError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def _api(path: str, paginate: bool) -> Any:
    cmd = ["gh", "api", *(["--paginate", "--slurp"] if paginate else []), path]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        raise GateError(f"gh api {path} failed: {done.stderr.strip()}")
    return json.loads(done.stdout)


def main(argv: Sequence[str] | None = None, *, git: Git = _git, api: Api = _api) -> int:
    parser = argparse.ArgumentParser(prog="require_green_lanes.py", description=__doc__)
    parser.add_argument("--lanes", required=True, help="one <ci.yml job id>=<prefix> per line")
    parser.add_argument("--repo", default=os.environ.get("REPO", ""))
    parser.add_argument("--sha", default=os.environ.get("TAG_SHA", ""))
    parser.add_argument("--tag", default=os.environ.get("TAG_NAME", ""))
    parser.add_argument(
        "--hint",
        action="append",
        default=[],
        help="a remedy line printed on failure, in place of the plain ci.yml dispatch",
    )
    args = parser.parse_args(argv)
    try:
        if not (args.repo and args.sha and args.tag):
            raise GateError("--repo, --sha and --tag (or REPO, TAG_SHA, TAG_NAME) are required")
        lanes = parse_lanes(args.lanes)
        candidates = same_tree_candidates(args.sha, git)
        url, report = find_green_run(args.repo, candidates, lanes, api)
    except GateError as exc:
        print(f"::error::{exc}")
        return 2

    if url is not None:
        print(f"✅ every required lane passed on this tree: {url}")
        print("Lanes: " + ", ".join(lanes))
        return 0

    tag = args.tag
    print(f"::error::no CI run proves the required lanes on the tree tagged {tag}")
    print("Lanes: " + ", ".join(lanes))
    print("Trees checked: " + ", ".join(sha[:12] for sha in candidates))
    print("\n".join(report) if report else "  no successful ci.yml run on any of them")
    print("A lane reported `skipped` did not run; one reported `cancelled` was superseded.")
    print("Run the lanes against this tree:")
    # The caller's remedies replace the plain dispatch, which would not run a
    # lane its own `if:` keeps off a bare dispatch.
    for hint in args.hint or [f"gh workflow run ci.yml --ref {tag}"]:
        print(f"  {hint}")
    print("Then re-push the tag, which retriggers this workflow:")
    print(f"  git push --delete origin {tag} && git push origin {tag}")
    print(f"  (git tag -f {tag} <sha> in between, if the tag has to move)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
