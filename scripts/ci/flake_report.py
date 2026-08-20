#!/usr/bin/env python3
"""Find flaky CI tests from GitHub Actions re-run history.

A flake here is proven, not guessed. When a job fails on one attempt of a
workflow run and succeeds on a later attempt, both attempts ran identical
code against an identical commit -- the only thing that changed is luck.
Failures that never flipped to green are reported separately as persistent
failures, because those are branch bugs and hiding them among the flakes
would be worse than not running this at all.

Reads only. Never writes to GitHub.

Usage:
    python scripts/ci/flake_report.py --days 30
    python scripts/ci/flake_report.py --days 7 --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

# The aggregator gate fails whenever anything else does, so it carries no
# information of its own and would otherwise top every ranking.
AGGREGATOR_JOB = "All CI Checks Passed"

MAX_WORKERS = 12

# Below this many failures from one file in one job, treat them as individual
# test failures. At or above it, a shared fixture almost certainly broke.
FIXTURE_GROUP_THRESHOLD = 3

# Constant, so repeat collection failures aggregate rather than splitting on
# however many modules each one happened to take down.
COLLECTION_ERROR_LABEL = "[collection error: modules failed to import]"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# pytest prints "FAILED <nodeid> - <message>" in its summary and
# "[gw1] [ 43%] FAILED <nodeid>" under xdist. A parametrised node ID can
# contain spaces inside its brackets, so the ID is matched as "everything up
# to whitespace or an opening bracket" plus an optional bracketed suffix --
# which stops cleanly before the " - <message>" tail.
FAILURE_RE = re.compile(r"(?:FAILED|ERROR)\s+((?:tests|src)/[^\s\[]+(?:\[[^\]]*\])?)")


# --------------------------------------------------------------------------
# Pure functions -- these are what the tests cover.
# --------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    """Remove SGR colour escapes so log lines can be matched literally."""
    return ANSI_RE.sub("", text)


def iter_json_documents(raw: str):
    """Yield each JSON document from a concatenated stream.

    ``gh api --paginate`` writes one JSON body per page back to back with no
    separator between them -- not newline-delimited, not wrapped in an array.
    A plain ``json.loads`` on the whole buffer raises "Extra data", and
    splitting on newlines finds nothing because there are none.
    """
    decoder = json.JSONDecoder()
    index, end = 0, len(raw)
    while index < end:
        while index < end and raw[index].isspace():
            index += 1
        if index >= end:
            return
        try:
            document, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            return
        yield document


def parse_test_failures(log_text: str) -> list[str]:
    """Extract failing pytest node IDs from one job's log.

    Returns them sorted and de-duplicated: the same failure appears in both
    the xdist progress stream and the final summary.
    """
    found: set[str] = set()
    for raw_line in strip_ansi(log_text).splitlines():
        match = FAILURE_RE.search(raw_line.rstrip())
        if match:
            found.add(match.group(1).strip())
    return sorted(found)


def split_flakes_and_persistent(runs):
    """Split early-attempt job failures by whether they later went green.

    ``runs`` is an iterable of dicts with keys ``run_id``, ``branch``,
    ``attempts`` (mapping attempt number -> list of ``(job_id, job_name,
    conclusion)``) and ``final_attempt``.

    Returns ``(flakes, persistent)``, each a list of job-failure dicts. A
    failure is a flake only when the same-named job succeeded on the final
    attempt -- same commit, opposite outcome.
    """
    flakes, persistent = [], []
    for run in runs:
        final_jobs = {
            name: conclusion
            for _job_id, name, conclusion in run["attempts"].get(run["final_attempt"], [])
        }
        for attempt, jobs in sorted(run["attempts"].items()):
            if attempt >= run["final_attempt"]:
                continue
            for job_id, name, conclusion in jobs:
                if conclusion != "failure" or name == AGGREGATOR_JOB:
                    continue
                record = {
                    "run_id": run["run_id"],
                    "branch": run["branch"],
                    "attempt": attempt,
                    "job_id": job_id,
                    "job_name": name,
                    "created_at": run.get("created_at", ""),
                }
                if final_jobs.get(name) == "success":
                    flakes.append(record)
                else:
                    persistent.append(record)
    return flakes, persistent


def group_fixture_failures(node_ids: list[str]) -> list[str]:
    """Collapse bursts of related failures so one incident occupies one row.

    Two bursts matter, and both come from a single root cause:

    * A broken shared fixture fails every test it feeds, at setup. Those are
      collapsed per file.
    * A failed import fails whole modules at collection time, which pytest
      reports as bare paths with no ``::`` node. One bad import can name a
      hundred files, which would otherwise bury the real ranking.

    The collection label is deliberately constant so that recurring collection
    failures aggregate across runs instead of splitting on the file count.
    """
    with_test, bare = [], []
    for node_id in node_ids:
        (with_test if "::" in node_id else bare).append(node_id)

    grouped: list[str] = []

    if len(bare) >= FIXTURE_GROUP_THRESHOLD:
        grouped.append(COLLECTION_ERROR_LABEL)
    else:
        grouped.extend(bare)

    by_file: dict[str, list[str]] = defaultdict(list)
    for node_id in with_test:
        by_file[node_id.split("::", 1)[0]].append(node_id)
    for path, ids in by_file.items():
        if len(ids) >= FIXTURE_GROUP_THRESHOLD:
            grouped.append(f"{path} [fixture: {len(ids)} tests]")
        else:
            grouped.extend(ids)

    return sorted(grouped)


def rank_failures(records) -> list[dict]:
    """Rank test entries by distinct runs blocked, then distinct branches.

    Branch spread is the confidence signal: the same test failing across many
    unrelated branches cannot be any one branch's bug.

    Each entry also carries ``last_seen``, without which a flake fixed weeks
    ago is indistinguishable from one that bit this morning.
    """
    runs_by_test: dict[str, set] = defaultdict(set)
    branches_by_test: dict[str, set] = defaultdict(set)
    jobs_by_test: dict[str, set] = defaultdict(set)
    last_seen_by_test: dict[str, str] = defaultdict(str)

    for record in records:
        for node_id in record["tests"]:
            runs_by_test[node_id].add(record["run_id"])
            branches_by_test[node_id].add(record["branch"])
            jobs_by_test[node_id].add(record["job_name"])
            created = record.get("created_at", "")
            if created > last_seen_by_test[node_id]:
                last_seen_by_test[node_id] = created

    ranked = [
        {
            "test": node_id,
            "runs": len(run_ids),
            "branches": len(branches_by_test[node_id]),
            "last_seen": last_seen_by_test[node_id][:10],
            "jobs": sorted(jobs_by_test[node_id]),
        }
        for node_id, run_ids in runs_by_test.items()
    ]
    ranked.sort(key=lambda row: (-row["runs"], -row["branches"], row["test"]))
    return ranked


# --------------------------------------------------------------------------
# GitHub access
# --------------------------------------------------------------------------


def gh_api(path: str, paginate: bool = False) -> str | None:
    """Call ``gh api`` and return stdout, "" if gone, or None if unanswerable.

    Expired logs and deleted runs are normal in a 90-day window, so a 404 is
    data rather than an error and comes back as "". Everything else -- timeout,
    rate limiting, auth failure, network loss -- means the question could not
    be asked at all, and comes back as ``None`` so callers never mistake an
    outage for a clean answer. ``gh api`` names the HTTP status in stderr, so
    a non-zero exit is only treated as "gone" when stderr says HTTP 404;
    anything unrecognisable fails toward UNKNOWN, never toward healthy.
    """
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    cmd.append(path)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        return "" if "HTTP 404" in stderr else None
    return result.stdout.decode("utf-8", errors="replace")


def cache_dir(repo: str) -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    # Keyed outside the repo so every worktree shares one cache.
    path = Path(root) / "osprey-flake-report" / repo.replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_json(cache: Path | None, key: str, produce):
    """Memoise an immutable API result on disk.

    Completed job logs and attempt job lists never change, so caching them
    turns a repeat scan from minutes into seconds. ``None`` -- the producer
    could not ask -- is never written, or a transient API failure would be
    memoised as a permanent "nothing failed here".
    """
    if cache is None:
        return produce()
    entry = cache / f"{key}.json"
    if entry.exists():
        try:
            return json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    value = produce()
    if value is None:
        return None
    try:
        entry.write_text(json.dumps(value))
    except OSError:
        pass
    return value


def list_rerun_runs(repo: str, workflow: str, since: str) -> list[dict] | None:
    """Return completed runs in the window that someone re-ran.

    ``None`` means the listing itself could not be fetched -- there is no
    window to analyse, which is not the same as an empty one.
    """
    raw = gh_api(
        f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=100&created=%3E%3D{since}",
        paginate=True,
    )
    if raw is None:
        return None
    runs = []
    for payload in iter_json_documents(raw):
        for run in payload.get("workflow_runs", []):
            if run.get("run_attempt", 1) > 1 and run.get("status") == "completed":
                runs.append(
                    {
                        "run_id": run["id"],
                        "branch": run.get("head_branch") or "?",
                        "final_attempt": run["run_attempt"],
                        "created_at": run.get("created_at", ""),
                    }
                )
    return runs


def fetch_attempt_jobs(repo: str, run_id: int, attempt: int, cache: Path | None):
    """Return the attempt's jobs as tuples, or ``None`` if unqueryable."""

    def produce():
        raw = gh_api(f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")
        if raw is None:
            return None
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # A truncated or garbled body is a failed answer, not an empty one.
            return None
        return [[j["id"], j["name"], j.get("conclusion")] for j in payload.get("jobs", [])]

    rows = cached_json(cache, f"jobs_{run_id}_{attempt}", produce)
    if rows is None:
        return None
    return [tuple(row) for row in rows]


def fetch_job_failures(repo: str, job_id: int, cache: Path | None) -> list[str] | None:
    """Return the job's failing node IDs, or ``None`` if the log is unfetchable."""

    def produce():
        raw = gh_api(f"repos/{repo}/actions/jobs/{job_id}/logs")
        if raw is None:
            return None
        return parse_test_failures(raw)

    return cached_json(cache, f"log_{job_id}", produce)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def render_table(ranked: list[dict], limit: int) -> str:
    if not ranked:
        return "  (none)"
    rows = ranked[:limit]
    width = min(max(len(r["test"]) for r in rows), 100)
    lines = [
        f"  {'runs':>4}  {'brs':>4}  {'last seen':>10}  test",
        f"  {'-' * 4}  {'-' * 4}  {'-' * 10}  {'-' * width}",
    ]
    for row in rows:
        lines.append(
            f"  {row['runs']:>4}  {row['branches']:>4}  {row['last_seen']:>10}  {row['test']}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Find proven flaky CI tests from GitHub Actions re-run history.",
    )
    parser.add_argument("--days", type=int, default=30, help="lookback window (default: 30)")
    parser.add_argument("--repo", default="als-apg/osprey")
    parser.add_argument("--workflow", default="ci.yml")
    parser.add_argument("--limit", type=int, default=25, help="rows to show (default: 25)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--no-cache", action="store_true", help="bypass the on-disk log cache")
    args = parser.parse_args(argv)

    cache = None if args.no_cache else cache_dir(args.repo)
    since = (datetime.now(UTC) - timedelta(days=args.days)).strftime("%Y-%m-%d")

    def note(message: str) -> None:
        if not args.json:
            print(message, file=sys.stderr)

    note(f"Scanning {args.repo} {args.workflow} since {since} ...")
    runs = list_rerun_runs(args.repo, args.workflow, since)
    if runs is None:
        print("Could not list workflow runs -- the window is UNKNOWN, not clean.", file=sys.stderr)
        return 1
    note(f"  {len(runs)} completed runs were re-run at least once")
    if not runs:
        print("No re-run runs in window -- nothing to analyse.")
        return 0

    # Every attempt of every re-run run, fetched concurrently: serially this
    # takes minutes and times out interactive shells.
    tasks = [(run, a) for run in runs for a in range(1, run["final_attempt"] + 1)]

    def load_attempt(task):
        run, attempt = task
        return (
            run["run_id"],
            attempt,
            fetch_attempt_jobs(args.repo, run["run_id"], attempt, cache),
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(load_attempt, tasks))

    # A run with any unqueryable attempt cannot be classified -- it is
    # reported as UNKNOWN rather than analysed as if those failures never
    # happened.
    unknown_run_ids = {run_id for run_id, _attempt, jobs in results if jobs is None}
    attempts_by_run: dict[int, dict[int, list]] = defaultdict(dict)
    for run_id, attempt, jobs in results:
        if jobs is not None:
            attempts_by_run[run_id][attempt] = jobs
    known_runs = [run for run in runs if run["run_id"] not in unknown_run_ids]
    for run in known_runs:
        run["attempts"] = attempts_by_run.get(run["run_id"], {})

    flakes, persistent = split_flakes_and_persistent(known_runs)
    note(
        f"  {len(flakes)} proven flakes, {len(persistent)} persistent failures"
        + (f", {len(unknown_run_ids)} runs UNKNOWN" if unknown_run_ids else "")
    )

    if not flakes:
        print("No proven flakes in window.")
        if unknown_run_ids:
            print(f"UNKNOWN -- {len(unknown_run_ids)} runs could not be queried, not proven clean.")
        return 0

    note(f"  fetching logs for {len(flakes)} flaky jobs ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        node_id_lists = list(
            pool.map(lambda f: fetch_job_failures(args.repo, f["job_id"], cache), flakes)
        )

    unknown_logs: list[dict] = []
    infra: list[dict] = []
    enriched: list[dict] = []
    for record, node_ids in zip(flakes, node_id_lists, strict=True):
        if node_ids is None:
            # An unfetchable log is not "no test-level failure" -- the job is
            # still a proven flake, it just cannot be attributed to a test.
            unknown_logs.append(record)
        elif node_ids:
            enriched.append({**record, "tests": group_fixture_failures(node_ids)})
        else:
            infra.append(record)

    ranked = rank_failures(enriched)
    persistent_jobs = sorted({p["job_name"] for p in persistent})

    if args.json:
        print(
            json.dumps(
                {
                    "repo": args.repo,
                    "since": since,
                    "runs_rerun": len(runs),
                    "unknown_runs": len(unknown_run_ids),
                    "unknown_job_logs": len(unknown_logs),
                    "flaky_job_instances": len(flakes),
                    "persistent_job_instances": len(persistent),
                    "flakes": ranked,
                    "infra_flakes": [
                        {"job_name": r["job_name"], "run_id": r["run_id"], "branch": r["branch"]}
                        for r in infra
                    ],
                    "persistent_failure_jobs": persistent_jobs,
                },
                indent=2,
            )
        )
        return 0

    print()
    print(f"PROVEN FLAKES -- failed then passed on the same commit ({args.days}d)")
    print(render_table(ranked, args.limit))
    if len(ranked) > args.limit:
        print(f"  ... {len(ranked) - args.limit} more (raise --limit to see them)")

    if infra:
        print()
        print(f"INFRA FLAKES -- job flipped green with no test-level failure ({len(infra)})")
        counts: dict[str, int] = defaultdict(int)
        for record in infra:
            counts[record["job_name"]] += 1
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>4}  {name}")

    if persistent_jobs:
        print()
        print(
            f"NOT FLAKES -- never went green, these are real failures ({len(persistent)} instances)"
        )
        for name in persistent_jobs:
            print(f"        {name}")

    if unknown_run_ids or unknown_logs:
        print()
        print(
            f"UNKNOWN -- could not be queried, not proven clean "
            f"({len(unknown_run_ids)} runs, {len(unknown_logs)} job logs)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
