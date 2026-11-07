#!/usr/bin/env python
"""Render a self-contained HTML dashboard for the agentic benchmark matrix.

Reads results/<model>__seed<seed>.json (the summary schema emitted by
run_e2e_for_model.sh) and produces a single static HTML file (inline CSS, no
external deps) grouped by the tier each test's ``agentic`` marker carries
(tests/e2e/agentic_tiers.py):

  * Tier 1, basic: a model x seed table whose headline is time and tokens per
    task, with an all-green check, then one row per test;
  * Tier 2, benchmark: one row per test, its first-attempt result per seed and
    each model's total;
  * a methodology footer (override mechanism, proxy, cap, exclusions, judge).

Each test row shows the marker's title and expands to what the test asks and
what passes; the file name is small grey text under it.

Partial-data friendly: missing (model,seed) runs render as "pending". Usage:
    scripts/benchmark/matrix_dashboard.py --results-dir results --out results/dashboard.html
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import html
import json
import os
import re
import sys
import time
from collections import defaultdict

# Two provider lanes in one matrix:
#   * Open SUBJECTS (gpt-oss/gemma/cborg-coder/qwen) run through CBORG ×3 seeds.
#   * Anthropic REFERENCES run NATIVE als-apg (the Claude models never needed CBORG
#     translation, and CBORG's anthropic-direct path breaks safety/feedback tests
#     als-apg passes — validated 2026-06-18). Ref ids are the exact als-apg strings.
# weak -> strong within the open spread; qwen (CBORG-proxied) after the self-hosted.
MODEL_ORDER = [
    "gpt-oss-20b",
    "gemma-4",
    "cborg-coder",
    "gpt-oss-120b",
    "google/qwen-3-coder",
    "google/qwen-3",
    "kolibri-1",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]
# Anthropic Claude models as a control/ceiling reference, flagged "(ref)".
# Ordered weak->strong (haiku < sonnet < opus) to bracket the open subjects;
# the 4.x ids label earlier runs' results.
REFERENCE_MODELS = {
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
}
SEEDS = [1, 2, 3]
# The skip-reason prefix of a test that does not apply to the model under test
# (tests/e2e/agentic_tiers.py NOT_APPLICABLE).
NOT_APPLICABLE = "n/a: "
OUTCOME_COLORS = {
    "passed": "#1a7f37",
    "failed": "#cf222e",
    "skipped": "#9a6700",
    "error": "#8250df",
    "timeout": "#bc4c00",
    "pending": "#8c959f",
}


_LIVE_KEY = {
    "passed": "passed",
    "failed": "failed",
    "timeout": "timeout",
    "skipped": "skipped",
    "error": "errors",
}

# A partial run whose live stream has not advanced for longer than this is a
# dead/killed run, not one still filling. 2× the 1800s per-test hang-breaker, so
# a single legitimately-slow test can never trip it. Used as a backstop only —
# the authoritative dead-run signal is run_is_dead() below, since mtime alone
# cannot distinguish "died 5 min ago" from "a slow test 5 min into its budget".
STALE_SECS = 3600

_TIMEOUT_BANNER = re.compile(r"\+{3,}\s*Timeout\s*\+{3,}")


def run_is_dead(run_log_path: str) -> bool:
    """True when the per-test hang-breaker fired and aborted the session.

    pytest-timeout prints a ``+++ Timeout +++`` banner when it kills a hung
    test; if that banner is the last thing in the log with no further
    ``[run ...]`` line after it, the session was taken down mid-run and will
    never write its finalizing .json. This is deterministic — unlike a time
    threshold it does not depend on how soon after death the dashboard is
    rendered."""
    try:
        with open(run_log_path, errors="ignore") as fh:
            txt = fh.read()
    except OSError:
        return False
    banners = list(_TIMEOUT_BANNER.finditer(txt))
    if not banners:
        return False
    return banners[-1].start() > txt.rfind("[run")


def collapse_reruns(tests: list[dict]) -> list[dict]:
    """One entry per test (keyed by nodeid), keeping the LAST line seen.

    pytest-rerunfailures emits a ``rerun`` line for every failed attempt of a
    flaky test, then a terminal line (passed/failed/...); the live stream
    contains both. Without collapsing, each retry is counted as a separate
    result — inflating ``total`` past the suite size and (since ``rerun`` is not
    in ``_LIVE_KEY``) miscounting every retry as a purple "error". Keeping the
    last line per nodeid folds retries into the final outcome, matching the
    junit-derived completed-run summary. A test seen ONLY as ``rerun`` — the run
    was killed mid-retry before any terminal line — is dropped, since it never
    reached a conclusive outcome."""
    last: dict[str, dict] = {}
    for t in tests:
        last[t.get("name")] = t  # live stream is chronological; terminal wins
    return [t for t in last.values() if t.get("outcome") != "rerun"]


def load_exclusions(config_path: str) -> list[tuple[str, str]]:
    """(rel_path, reason) for every file the matrix excludes, read live from
    matrix_e2e_config.json. ``rel_path`` is the path RELATIVE TO ``tests/e2e/``
    (e.g. ``test_dispatch_tutorial.py`` or ``claude_code/test_proxy_full_chain_e2e.py``)
    — the EXACT key ``canon_name`` produces, so ``apply_exclusions`` matches nested
    files too. (Basenaming here is what historically let ``claude_code/*`` excluded
    tests leak into the grid: the basename ``test_x.py`` never equals canon_name's
    ``claude_code/test_x.py``.) The footer basenames these only for display."""
    try:
        with open(config_path) as fh:
            cfg = json.load(fh)
    except Exception as exc:
        # LOUD failure: silently returning [] disables EVERY exclusion and the
        # grid fills with non-model tests (dispatch_tutorial, proxy smokes, ...)
        # with no hint why. A missing/renamed config must not look like "clean".
        print(
            f"WARNING: could not read exclusion config {config_path!r} ({exc}); "
            "rendering with NO exclusions — every e2e file will show in the grid.",
            file=sys.stderr,
        )
        return []
    excluded = [
        (e["path"].replace("tests/e2e/", "", 1), e.get("reason", ""))
        for e in cfg.get("excluded_files", [])
    ]
    if not excluded:
        print(
            f"WARNING: exclusion config {config_path!r} has no 'excluded_files'; "
            "every e2e file will show in the grid.",
            file=sys.stderr,
        )
    return excluded


def load(results_dir: str) -> dict:
    runs = {}
    # completed runs: authoritative summary parsed from junit at end of run
    for f in glob.glob(os.path.join(results_dir, "*__seed*.json")):
        if f.endswith(".lanes.json"):
            continue  # per-cell lane manifest, not a run summary (load_tiers reads it)
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        d["_partial"] = False
        runs[(d["model"], int(d["seed"]))] = d
    # in-progress runs: build a PARTIAL summary from the live per-test stream so
    # the dashboard fills in as tests finish (completed .json always wins).
    for f in glob.glob(os.path.join(results_dir, "*__seed*.live.jsonl")):
        base = os.path.basename(f)[: -len(".live.jsonl")]
        if "__seed" not in base:
            continue
        model, seed_s = base.rsplit("__seed", 1)
        # Reverse the runner's slash-safe transform (SAFE="${MODEL//\//__}") so a
        # live key matches the completed summary's real id (d["model"] keeps the
        # slash). Without this, 'google/qwen-3-coder' would show as a separate
        # 'google__qwen-3-coder' column while live, then jump columns on finish.
        model = model.replace("__", "/")
        try:
            seed = int(seed_s)
        except ValueError:
            continue
        if (model, seed) in runs:
            continue
        raw = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(Exception):
                        raw.append(json.loads(line))
        if not raw:
            continue
        tests = collapse_reruns(raw)
        if not tests:
            continue
        cnt = {"passed": 0, "failed": 0, "timeout": 0, "skipped": 0, "errors": 0}
        for t in tests:
            cnt[_LIVE_KEY.get(t.get("outcome"), "errors")] += 1
        stalled = (
            run_is_dead(f[: -len(".live.jsonl")] + ".run.log")
            or (time.time() - os.path.getmtime(f)) > STALE_SECS
        )
        runs[(model, seed)] = {
            "model": model,
            "seed": seed,
            "provider": _provider_from_run_log(f[: -len(".live.jsonl")] + ".run.log"),
            "route": "",
            "pytest_rc": None,
            "total_duration_s": int(sum(t.get("duration_s", 0) for t in tests)),
            **cnt,
            "total": len(tests),
            "tests": tests,
            "_partial": True,
            "_stalled": stalled,
        }
    return runs


def _provider_from_run_log(run_log_path: str) -> str:
    """The provider a still-running cell builds with, from the worker's
    ``>> model=... provider=...`` banner ("" when the log has none yet)."""
    try:
        with open(run_log_path, errors="replace") as f:
            for line in f:
                m = re.match(r">> model=\S+ seed=\S+ provider=(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return ""


def _manifests(results_dir: str):
    """Every per-cell ``*.lanes.json`` manifest in the results dir, parsed."""
    for f in glob.glob(os.path.join(results_dir, "*.lanes.json")):
        try:
            with open(f) as fh:
                yield json.load(fh)
        except Exception:
            continue


def load_tiers(results_dir: str) -> dict[tuple, dict]:
    """(file, qualname) -> the agentic marker's ``tier``, ``title``, ``asks`` and
    ``passes_if``, unioned across every per-cell manifest. Manifests written
    before the tiers carry none, and the result is then empty."""
    tiers: dict[tuple, dict] = {}
    for manifest in _manifests(results_dir):
        for nodeid, entry in manifest.items():
            if isinstance(entry, dict) and entry.get("lane") == "agentic" and entry.get("tier"):
                tiers[canon_name(nodeid)] = entry
    return tiers


def load_query_timing(results_dir: str) -> dict[str, list[dict]]:
    """model -> every per-query timing record across its seeds.

    Each cell's ``*.queries.jsonl`` (tests/e2e/sdk_helpers.py,
    ``OSPREY_E2E_QUERY_LOG``) holds one line per agent query: the test it ran
    in, wall time, time spent waiting on the model, turns and tokens. Each
    record gains ``_seed``, the cell it came from. Cells run before the log
    existed simply contribute nothing."""
    out: dict[str, list[dict]] = defaultdict(list)
    for f in glob.glob(os.path.join(results_dir, "*__seed*.queries.jsonl")):
        model, seed_s = os.path.basename(f)[: -len(".queries.jsonl")].rsplit("__seed", 1)
        model = model.replace("__", "/")
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(Exception):
                        out[model].append({**json.loads(line), "_seed": seed_s})
    return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _keep_all(_key: tuple) -> bool:
    return True


def latency_stats(model: str, runs: dict, queries: dict[str, list[dict]], keep=_keep_all) -> dict:
    """Time and token figures for one model across its seeds, over the tests
    ``keep`` accepts (by canonical name).

    ``median_task_s`` is the median wall time of those tests, skips excluded.
    ``tokens_per_task`` is the median, over (seed, test), of the output tokens
    the test's queries spent. ``model_share`` is the share of the queries' span
    spent waiting on the model (``model_wait_ms`` over ``span_ms``, measured
    from the session transcripts with concurrent requests counted once; records
    from before that measurement carry neither and are left out).
    ``tokens_per_s`` is output tokens per second of summed request time
    (``duration_api_ms``), the per-request throughput. Each is None when its
    data is missing; a query record with no test name counts toward every
    selection."""
    durations = []
    for (m, _s), d in runs.items():
        if m != model:
            continue
        for t in d.get("tests", []):
            if t.get("outcome") in ("skipped", "error") or not keep(canon_name(t["name"])):
                continue
            durations.append(float(t.get("duration_s", 0) or 0))
    wait_ms = span_ms = api_ms = out_tok = 0
    per_task: dict[tuple, int] = defaultdict(int)
    for q in queries.get(model, []):
        test = q.get("test")
        if test and not keep(canon_name(test)):
            continue
        if test:
            per_task[(q.get("_seed"), canon_name(test))] += q.get("output_tokens") or 0
        if q.get("span_ms") and q.get("model_wait_ms") is not None:
            span_ms += q["span_ms"]
            wait_ms += q["model_wait_ms"]
        if q.get("duration_api_ms"):
            api_ms += q["duration_api_ms"]
            out_tok += q.get("output_tokens") or 0
    return {
        "median_task_s": _median(durations),
        "tokens_per_task": _median([v for v in per_task.values() if v]),
        "model_share": (wait_ms / span_ms) if span_ms else None,
        "tokens_per_s": (out_tok / (api_ms / 1000)) if api_ms and out_tok else None,
    }


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, sec = divmod(int(round(seconds)), 60)
    return f"{m}m {sec:02d}s" if m else f"{sec}s"


def fmt_tokens(n: float | None) -> str:
    if n is None:
        return "—"
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:.0f}"


def not_applicable_label(test: dict) -> str | None:
    """``n/a: <label>`` for a result skipped because the test does not apply to
    the model (a skip reason ``n/a: <label> (<detail>)``, tests/e2e/agentic_tiers.py),
    else None."""
    reason = test.get("reason") or ""
    if test.get("outcome") != "skipped" or not reason.startswith(NOT_APPLICABLE):
        return None
    return reason.split(" (", 1)[0]


def tier_of(test: dict, tiers: dict[tuple, dict]) -> int | None:
    """The tier a result belongs to: the manifest's, else the one the worker
    recorded with the result, else None (no tier: harness, or renamed since)."""
    meta = tiers.get(canon_name(test["name"]))
    return (meta or {}).get("tier") or test.get("tier")


def conclusive_where(d: dict, keep) -> tuple[int, int]:
    """The cell's score over the tests ``keep`` accepts by canonical name:
    (passed, denominator), a timeout counting as a failure. Skips and errors are
    excluded: an error is a test that died in setup or teardown, which says nothing
    about the model."""
    p = den = 0
    for t in d.get("tests", []):
        if not keep(canon_name(t["name"])):
            continue
        o = t.get("outcome")
        if o == "passed":
            p += 1
            den += 1
        elif o in ("failed", "timeout"):
            den += 1
    return p, den


def parse_progress(results_dir: str):
    """Read matrix.log -> (started, ended, matrix_done). A combo START-ed but not
    END-ed is currently running."""
    started, ended, done = set(), set(), False
    path = os.path.join(results_dir, "matrix.log")
    if not os.path.exists(path):
        return started, ended, done
    rx = re.compile(r">> (START|END)\s+(\S+)\s+seed(\d+)")
    with open(path) as fh:
        for line in fh:
            if "MATRIX_EXIT_0" in line:
                done = True
            m = rx.search(line)
            if m:
                (started if m.group(1) == "START" else ended).add((m.group(2), int(m.group(3))))
    return started, ended, done


def label(model: str) -> str:
    """Display name: strip the CBORG routing prefix (google/, lbl/, ...) so a
    proxied id like 'google/qwen-3-coder' renders as 'qwen-3-coder'. Data keys
    keep the full id; this only affects what the reader sees. (The provider line
    under each model now distinguishes references, so no '(ref)' suffix.)"""
    return model.split("/", 1)[1] if "/" in model else model


def provider_of(model: str, runs: dict) -> str:
    """The provider that ran this model, for the lighter [provider] line under
    each model name. Authoritative source is the run's recorded ``provider``
    field (written by run_e2e_for_model.sh). Legacy results predate that field;
    fall back to the only mapping the matrix used then — references via als-apg,
    everything else (proxied subjects) via cborg — so old dashboards still label.
    """
    for (mm, _s), d in runs.items():
        if mm == model and d.get("provider"):
            return d["provider"]
    if model in REFERENCE_MODELS:
        return "als-apg"
    if model.startswith("deepseek"):  # local ds4 server; labeled before its first json lands
        return "ds4"
    return "cborg"


def provider_display(provider: str) -> str:
    """Render the canonical provider for the [provider] line. ds4 is the keyless
    local-server entry, so flag it as self-hosted to set it apart from the
    remote providers."""
    if provider in ("ds4", "vllm"):
        return f"{provider} · self-hosted"
    return provider


def canon_name(name: str) -> tuple[str, str]:
    """Normalize a test id to a canonical (file, qualname) regardless of source.

    Completed runs are summarized from the JUnit XML, whose ``classname`` is the
    DOTTED module path with the class folded in
    (``tests.e2e.claude_code.test_agent_delegation.TestAgentDelegation::test_x``).
    In-progress runs come from the live per-test stream, which uses the pytest
    NODEID (``tests/e2e/claude_code/test_agent_delegation.py::TestAgentDelegation::test_x``).
    Both name the same test; without normalization the per-test table lists every
    test twice (once per format), which reads as two stacked tables. Collapse
    both to ``("claude_code/test_agent_delegation.py", "TestAgentDelegation::test_x")``.
    """
    parts = name.split("::")
    head, tail = parts[0], parts[1:]
    if head.endswith(".py"):
        file = head
        qual = tail
    else:
        segs = head.split(".")
        mod_idx = next(
            (i for i in range(len(segs) - 1, -1, -1) if segs[i].startswith("test_")), None
        )
        if mod_idx is None:
            file = head.replace(".", "/") + ".py"
            qual = tail
        else:
            file = "/".join(segs[: mod_idx + 1]) + ".py"
            qual = segs[mod_idx + 1 :] + tail  # trailing class segment(s) + method
    file = file.replace("tests/e2e/", "")
    short = "::".join(qual) if qual else file
    return file, short


def apply_exclusions(runs: dict, excluded_files: set[str]) -> None:
    """Drop tests belonging to excluded files from each run, recomputing counts.

    The matrix runner ignores excluded files at collection time, so a freshly run
    cell never contains them. But results collected BEFORE a file was added to the
    exclusion list still carry its tests — e.g. test_dispatch_tutorial, which is
    pinned to als-apg/haiku and hangs to the worker timeout under CBORG cells, was
    excluded only after the matrix had already run it. Honor the CURRENT exclusion
    list retroactively so the per-cell counts and the per-test grid match the
    footer's advertised exclusion set, without re-running the matrix. Mutates runs
    in place; cells with no excluded tests are left untouched.
    """
    for d in runs.values():
        tests = d.get("tests", [])
        kept = [t for t in tests if canon_name(t["name"])[0] not in excluded_files]
        if len(kept) == len(tests):
            continue
        cnt = {"passed": 0, "failed": 0, "timeout": 0, "skipped": 0, "errors": 0}
        for t in kept:
            cnt[_LIVE_KEY.get(t.get("outcome"), "errors")] += 1
        d["tests"] = kept
        d.update(cnt)
        d["total"] = len(kept)
        d["total_duration_s"] = int(sum(t.get("duration_s", 0) for t in kept))


def seeds_for(model: str, runs: dict | None = None) -> list[int]:
    """Study subjects run all SEEDS (missing ones render 'pending'). Reference
    models are run on demand (one or more seeds) — show exactly the seeds that
    have data, so a second (or third) reference pass renders alongside the
    first rather than being hidden. Drives cell-counting and seed columns."""
    if model not in REFERENCE_MODELS:
        return SEEDS
    if runs:
        present = sorted({s for (m, s) in runs if m == model})
        if present:
            return present
    return [1]


def heat(frac: float | None) -> str:
    """Green->red heatmap color for a pass fraction in [0,1]; grey if None."""
    if frac is None:
        return "#eaeef2"
    # hue 0 (red) .. 130 (green)
    h = 130 * frac
    return f"hsl({h:.0f}, 62%, 80%)"


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "—"


OUTCOME_ORDER = ["passed", "failed", "timeout", "skipped", "error"]
OUTCOME_GLYPH = {"passed": "✓", "failed": "✗", "timeout": "⧗", "skipped": "∅", "error": "!"}


def run_counts(d: dict) -> dict:
    return {
        "passed": d.get("passed", 0),
        "failed": d.get("failed", 0),
        "timeout": d.get("timeout", 0),
        "skipped": d.get("skipped", 0),
        "error": d.get("errors", 0),
    }


def stacked_bar(counts: dict, width: int = 130, height: int = 14) -> str:
    tot = sum(counts.values())
    if not tot:
        return f"<div style='width:{width}px;height:{height}px;background:#eaeef2;border-radius:3px'></div>"
    seg = [
        f"<span title='{o}: {counts[o]}' style='display:inline-block;height:{height}px;"
        f"width:{100 * counts[o] / tot:.2f}%;background:{OUTCOME_COLORS[o]}'></span>"
        for o in OUTCOME_ORDER
        if counts.get(o)
    ]
    return (
        f"<div style='width:{width}px;height:{height}px;border-radius:3px;overflow:hidden;"
        f"font-size:0;white-space:nowrap;border:1px solid #d8dee4'>" + "".join(seg) + "</div>"
    )


def counts_text(counts: dict) -> str:
    return " ".join(
        f"<span style='color:{OUTCOME_COLORS[o]}'>{counts[o]}{OUTCOME_GLYPH[o]}</span>"
        for o in OUTCOME_ORDER
        if counts.get(o)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/dashboard.html")
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix_e2e_config.json"),
        help="matrix_e2e_config.json — source of the excluded-file list shown in the footer",
    )
    ap.add_argument(
        "--artifact",
        action="store_true",
        help="render a timeless static artifact (for docs/download): drop the issue "
        "ref, progress counters, live status, timestamp, auto-refresh and running-now "
        "line — keep only the title and the data.",
    )
    args = ap.parse_args()

    runs = load(args.results_dir)
    # Honor the exclusion list retroactively: results collected before a file was
    # excluded still carry its tests, so strip them here to match the footer.
    exclusions = load_exclusions(args.config)  # (rel_path, reason); rel_path == canon_name key
    apply_exclusions(runs, {rel for rel, _ in exclusions})
    # Tier manifest (marker-derived, tests/e2e/agentic_tiers.py): each agentic
    # test's tier and plain-words description. Results that predate the tiers
    # carry none and render as one untiered section.
    tiers = load_tiers(args.results_dir)
    started, ended, matrix_done = parse_progress(args.results_dir)
    running = started - ended  # combos START-ed but not END-ed in matrix.log
    # A combo with a finalized .json is done even if the driver died before
    # writing its END marker — so a wedged-then-finalized cell isn't "running".
    running -= {c for c, dd in runs.items() if not dd.get("_partial")}
    # A combo can be START-ed-but-not-END-ed yet have a dead process (killed
    # before it wrote END / its finalizing .json). Split on live-stream
    # staleness so the dashboard never advertises a dead run as "running".
    stalled_combos = {c for c, d in runs.items() if d.get("_stalled")}
    running_active = running - stalled_combos
    # matrix.log can carry a MATRIX_EXIT_0 from an earlier batch while a later
    # batch left combos START-ed but never END-ed. "Complete" must mean nothing
    # is still in flight, so AND it with an empty running set.
    matrix_done = matrix_done and not running
    # MODEL_ORDER only orders the rows; a model gets a row when it has results
    # (or, on the live monitor, has started), never as an empty placeholder.
    present = (
        {m for (m, _) in runs}
        if args.artifact
        else {m for (m, _) in runs} | {m for (m, _) in started}
    )
    models = [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))
    queries = load_query_timing(args.results_dir)

    # union of all test names, grouped by file; (file,test) -> model -> {seed: outcome}
    all_tests: dict[str, set] = defaultdict(set)
    test_results: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    # (file,test) -> model -> label, for a test that does not apply to the model
    not_applicable: dict[tuple, dict] = defaultdict(dict)
    for (m, s), d in runs.items():
        for t in d.get("tests", []):
            file, short = canon_name(t["name"])
            all_tests[file].add(short)
            test_results[(file, short)][m][s] = t["outcome"]
            label_na = not_applicable_label(t)
            if label_na:
                not_applicable[(file, short)][m] = label_na

    done = sum(1 for d in runs.values() if not d.get("_partial"))
    live_n = sum(1 for d in runs.values() if d.get("_partial") and not d.get("_stalled"))
    stalled_n = sum(1 for d in runs.values() if d.get("_stalled"))
    total_cells = sum(
        len(seeds_for(m, runs)) for m in models
    )  # subjects: 3 seeds; refs: the seeds actually run

    # Model-driving test count, DERIVED (never frozen): a completed run collects
    # exactly the model-driving subset, so its `total` is authoritative — take the
    # max across completed runs (a run that errored at collection can't undercut
    # it). Before any run completes, fall back to the union of tests seen so far,
    # so the live progress denominator can never be exceeded (the old hardcoded
    # constant could, e.g. 53/33). None only when there is no data at all.
    completed_totals = [
        d["total"] for d in runs.values() if not d.get("_partial") and d.get("total")
    ]
    union_n = sum(len(v) for v in all_tests.values())
    model_driving_n = max(completed_totals) if completed_totals else (union_n or None)

    css = """
    body{font:14px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
         margin:0;background:#f6f8fa;color:#1f2328}
    .wrap{max-width:1180px;margin:0 auto;padding:28px}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:30px 0 10px}
    .sub{color:#636c76;margin:0 0 18px}
    table{border-collapse:collapse;background:#fff;border:1px solid #d0d7de;border-radius:8px;overflow:hidden}
    th,td{padding:7px 10px;border-bottom:1px solid #eaeef2;text-align:center;font-variant-numeric:tabular-nums}
    th{background:#f6f8fa;font-weight:600;position:sticky;top:0}
    td.l,th.l{text-align:left}
    tr.sep td{padding:0;height:0;border-bottom:2px solid #c8d1da}
    .card{display:inline-block;background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;margin:0 12px 12px 0;vertical-align:top}
    .big{font-size:26px;font-weight:700} .muted{color:#636c76;font-size:12px}
    .prov{color:#8c959f;font-weight:400;font-size:11px;margin-top:1px}
    .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;color:#fff}
    .legend span{margin-right:14px} .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
    .filerow td{background:#eef1f4;font-weight:600;text-align:left}
    .totrow td{background:#f6f8fa;font-weight:600}
    td.l details summary{cursor:pointer;font-weight:600}
    td.l .ask{color:#1f2328;font-size:12.5px;margin:4px 0 0 14px;max-width:420px}
    td.l .ask b{color:#636c76;font-weight:600}
    td.l .file{color:#8c959f;font-size:11px;margin-top:2px}
    .ok{color:#1a7f37;font-weight:700} .no{color:#cf222e;font-weight:700}
    footer{margin-top:34px;color:#636c76;font-size:12.5px;border-top:1px solid #d0d7de;padding-top:14px}
    code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px}
    """

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if matrix_done:
        status = "✓ matrix complete"
    elif running_active:
        status = f"● live — {len(running_active)} running"
    elif stalled_n:
        status = "⚠ incomplete"  # the "N stalled" clause carries the count
    else:
        status = "○ idle"
    # Auto-refresh only for the LIVE monitor (never the static artifact), and
    # only while something is filling — a dead page that reloads every 60s reads
    # as "still working" when nothing is.
    refresh = (
        ""
        if args.artifact
        else ('<meta http-equiv="refresh" content="60">' if running_active else "")
    )
    title = "OSPREY agentic benchmark"

    H = []
    H.append(
        f"<!doctype html><meta charset=utf-8>{refresh}<title>{title}</title><style>{css}</style>"
    )
    H.append("<div class=wrap>")
    if args.artifact:
        # Timeless artifact: no issue ref, progress counters, live status,
        # timestamp, auto-refresh or running-now line — just title and data.
        H.append(f"<h1>{title}</h1>")
        H.append(
            f"<p class=sub>Open-weight &amp; self-hosted models on OSPREY's agentic tests, "
            f"Tier 1 and Tier 2 · {len(models)} model{'s' if len(models) != 1 else ''} × {len(SEEDS)} seeds</p>"
        )
    else:
        H.append(f"<h1>{title}</h1>")
        H.append(
            f"<p class=sub>{len(models)} model{'s' if len(models) != 1 else ''} × {len(SEEDS)} seeds · "
            f"per-test hang-breaker 1800s · <b>{done}/{total_cells}</b> runs complete · "
            + (f"<b>{live_n} filling live</b> · " if live_n else "")
            + f"<b>{html.escape(status)}</b> · updated {now}"
            + (" · auto-refresh 60s" if running_active else "")
            + "</p>"
        )
        if running_active:
            H.append(
                "<p class=sub>running now: "
                + ", ".join(
                    f"<code>{html.escape(label(m))} seed{s}</code>"
                    for (m, s) in sorted(running_active)
                )
                + "</p>"
            )

    # legend (drop the live-only "pending" state in the static artifact)
    H.append("<div class=legend style='margin-bottom:18px'>")
    for k, c in OUTCOME_COLORS.items():
        if args.artifact and k == "pending":
            continue
        H.append(f"<span><span class=dot style='background:{c}'></span>{k}</span>")
    H.append("</div>")

    # A test's section: its tier, or "other" for a result no tier claims
    # (harness results from before the board ran agentic tests only, or a test
    # renamed since). Results with no tier anywhere render as one section.
    tier_by_key: dict[tuple, int] = {}
    for d in runs.values():
        for t in d.get("tests", []):
            tier = tier_of(t, tiers)
            if tier:
                tier_by_key[canon_name(t["name"])] = tier
    tiered = bool(tier_by_key)

    def in_tier(tier):
        return lambda key: tier_by_key.get(key) == tier

    def summary_table(keep) -> None:
        """Model x seed: passes over the tests ``keep`` accepts, the all-green
        check, and time and tokens per task over the same tests."""
        H.append("<table>")
        H.append(
            "<tr><th class=l>model</th>"
            + "".join(f"<th>seed {s}</th>" for s in SEEDS)
            + "<th>all green</th><th>median task</th><th>tokens / task</th>"
            "<th>in model</th><th>tok/s</th></tr>"
        )
        sep_done = False
        for m in models:
            # Light divider between the self-hosted/open models and the Anthropic
            # reference bracket (drawn once, before the first reference row).
            if not sep_done and m in REFERENCE_MODELS:
                H.append(f"<tr class=sep><td colspan='{len(SEEDS) + 6}'></td></tr>")
                sep_done = True
            H.append(
                f"<tr><td class=l>{html.escape(label(m))}"
                f"<div class=prov>[{html.escape(provider_display(provider_of(m, runs)))}]</div></td>"
            )
            tot_p = tot_den = tot_err = 0
            m_seeds = seeds_for(m, runs)
            for s in SEEDS:
                if s not in m_seeds:  # seed not planned for this model (e.g. single-seed)
                    H.append("<td style='background:#fafbfc'><span class=muted>N/A</span></td>")
                    continue
                d = runs.get((m, s))
                if d and d["total"]:
                    if args.artifact and d.get("_partial"):
                        # Static artifact: never show a partial cell's interim
                        # score as if it were final — mark it as still running.
                        H.append(
                            "<td style='background:#fafbfc'><span class=muted>running</span></td>"
                        )
                        continue
                    p_s, den_s = conclusive_where(d, keep)
                    if not d.get("_partial"):
                        tot_p += p_s  # the all-green check is over COMPLETED seeds only
                        tot_den += den_s
                        tot_err += sum(
                            1
                            for t in d["tests"]
                            if t.get("outcome") == "error" and keep(canon_name(t["name"]))
                        )
                    c = dict.fromkeys(OUTCOME_ORDER, 0)
                    for t in d["tests"]:
                        if keep(canon_name(t["name"])):
                            c[t.get("outcome") if t.get("outcome") in c else "error"] += 1
                    # Incomplete cell (still filling, or interrupted): its real
                    # partial bar with a plain progress count over the whole run.
                    tag = (
                        f"<div class=muted>{d['total']}/{model_driving_n or '?'}</div>"
                        if d.get("_partial")
                        else ""
                    )
                    H.append(
                        f"<td><div style='font-weight:600'>{p_s}/{den_s}</div>"
                        f"<div style='display:flex;justify-content:center;margin:3px 0'>"
                        f"{stacked_bar(c)}</div>"
                        f"<div class=muted>{counts_text(c)}</div>"
                        f"{tag}</td>"
                    )
                elif d and not d["total"]:
                    H.append(
                        "<td style='background:#ffd8c2'>err<div class=muted>0 tests</div></td>"
                    )
                elif (m, s) in running_active:
                    H.append("<td style='background:#fff3cd'>●<div class=muted>running</div></td>")
                else:
                    H.append("<td style='background:#fafbfc'><span class=muted>N/A</span></td>")
            if tot_den and tot_p < tot_den:
                green = (
                    f"<td style='background:{heat(tot_p / tot_den)}'><span class=no>✗</span>"
                    f"<div class=muted>{tot_den - tot_p} failed</div></td>"
                )
            elif tot_err:
                # Nothing the model did went wrong, and not everything was measured.
                green = f"<td>·<div class=muted>{tot_err} setup error(s)</div></td>"
            elif not tot_den:
                green = "<td>·</td>"
            else:
                green = f"<td style='background:{heat(1.0)}'><span class=ok>✓</span></td>"
            lat = latency_stats(m, runs, queries, keep)
            share = lat["model_share"]
            tps = lat["tokens_per_s"]
            H.append(
                green + f"<td><b>{fmt_duration(lat['median_task_s'])}</b></td>"
                f"<td><b>{fmt_tokens(lat['tokens_per_task'])}</b></td>"
                f"<td>{f'{100 * share:.0f}%' if share is not None else '—'}</td>"
                f"<td>{f'{tps:.0f}' if tps is not None else '—'}</td></tr>"
            )
        H.append("</table>")

    def row_label(file: str, short: str) -> str:
        """A test row's label: the marker's title, expanding to what the test
        asks and what passes, with the file name small and grey beneath."""
        meta = tiers.get((file, short))
        where = f"<div class=file>{html.escape(file)}::{html.escape(short)}</div>"
        if not meta or not meta.get("title"):
            return f"{html.escape(short)}<div class=file>{html.escape(file)}</div>"
        return (
            f"<details><summary>{html.escape(meta['title'])}</summary>"
            f"<div class=ask><b>asks</b> {html.escape(meta.get('asks') or '')}</div>"
            f"<div class=ask><b>passes if</b> {html.escape(meta.get('passes_if') or '')}</div>"
            f"</details>{where}"
        )

    def render_test_table(keys: list[tuple], totals: bool = False) -> None:
        """One row per test, one square per seed per model. With ``totals``, a
        last row counts each model's first-attempt passes."""
        H.append(
            "<table><tr><th class=l>test</th>"
            + "".join(f"<th>{html.escape(label(m))}</th>" for m in models)
            + "</tr>"
        )

        def order(key):
            meta = tiers.get(key) or {}
            return (meta.get("title") or "~", key)

        for file, short in sorted(keys, key=order):
            H.append(f"<tr><td class=l>{row_label(file, short)}</td>")
            for m in models:
                seeds_map = test_results[(file, short)].get(m, {})
                if m in not_applicable[(file, short)]:
                    # Not applicable to this model: said so, never hidden, and
                    # (as a skip) counted against neither passes nor all-green.
                    H.append(
                        f"<td class=muted>{html.escape(not_applicable[(file, short)][m])}</td>"
                    )
                    continue
                dots = []
                for s in seeds_for(m, runs):  # ref models render a single square, not 3
                    o = seeds_map.get(s)
                    col = OUTCOME_COLORS.get(o, "#eaeef2")
                    dots.append(
                        f"<span title='seed{s}: {o or 'pending'}' style='display:inline-block;"
                        f"width:13px;height:13px;border-radius:2px;margin:1px;background:{col}'></span>"
                    )
                H.append(f"<td>{''.join(dots)}</td>")
            H.append("</tr>")
        if totals:
            H.append("<tr class=totrow><td class=l>first-attempt passes</td>")
            for m in models:
                p = den = 0
                for key in keys:
                    for o in test_results[key].get(m, {}).values():
                        if o == "passed":
                            p += 1
                        if o in ("passed", "failed", "timeout"):
                            den += 1
                H.append(f"<td>{p}/{den}</td>" if den else "<td>·</td>")
            H.append("</tr>")
        H.append("</table>")

    keys_all = [(file, short) for file in all_tests for short in all_tests[file]]
    if tiered:
        t1 = [k for k in keys_all if tier_by_key.get(k) == 1]
        t2 = [k for k in keys_all if tier_by_key.get(k) == 2]
        other = [k for k in keys_all if k not in tier_by_key]
        H.append("<h2>Tier 1 · basic</h2>")
        H.append(
            "<p class=sub>Tasks a capable model should complete: the stack works, the model "
            "calls the right tools, a simple workflow goes through. All green is the check; "
            "the comparison is <i>median task</i> (wall time of a Tier 1 test) and "
            "<i>tokens / task</i> (output tokens a test's queries spent, median). <i>In "
            "model</i> is the share of agent time spent waiting on the model, concurrent "
            "requests counted once, <i>tok/s</i> "
            "output tokens per second of model time. Each cell counts first-attempt passes "
            "(a timeout counts as a failure; a skip or a setup error scores nothing).</p>"
        )
        summary_table(in_tier(1))
        H.append("<h2 style='font-size:15px'>Tier 1 tests — one square per seed</h2>")
        render_test_table(t1)
        H.append("<h2>Tier 2 · benchmark</h2>")
        H.append(
            "<p class=sub>Harder tasks a capable model may still miss. Each test on its own: "
            "one square per seed, first attempt only (green pass · red fail · orange timeout "
            "· yellow skip · purple error · grey pending). Too few tests for a single "
            "score.</p>"
        )
        render_test_table(t2, totals=True)
        if other:
            H.append("<h2 style='font-size:15px'>Other results (no tier)</h2>")
            H.append(
                "<p class=sub>Results no tier claims (harness checks from earlier runs, or "
                "tests renamed since). They count toward neither tier.</p>"
            )
            render_test_table(other)
    else:
        H.append("<h2>All tests</h2>")
        H.append(
            "<p class=sub>These results predate the agentic tiers, so they render as one "
            "section (a timeout counts as a failure; a skip or a setup error scores nothing).</p>"
        )
        summary_table(_keep_all)
        render_test_table(keys_all)

    # ---- methodology ----
    # Scope sentence is DERIVED: the model-driving count from the run data and the
    # excluded-file list (with reasons) from matrix_e2e_config.json — so it tracks
    # the suite instead of asserting frozen totals (was "36 of 92", long stale).
    n_txt = str(model_driving_n) if model_driving_n is not None else "—"
    if tiered:
        n_1 = sum(1 for v in tier_by_key.values() if v == 1)
        n_2 = sum(1 for v in tier_by_key.values() if v == 2)
        what = (
            f"The <b>{n_txt}</b> agentic e2e tests ({n_1} in Tier 1, {n_2} in Tier 2) — those "
            f"carrying an <code>agentic</code> marker, whose tier, title and description are "
            f"re-derived from live collection every cell — run once per seed with reruns off, "
            f"so every result is a first attempt. Tier 3, facility-scale tasks, is planned. "
        )
    else:
        what = f"The <b>{n_txt}</b> model-driving e2e tests. "
    if exclusions:
        # The names stay one click away: the page names tests by what they ask,
        # never by file.
        excl_items = "".join(
            f"<li><code>{html.escape(os.path.basename(rel))}</code> — {html.escape(reason)}</li>"
            for rel, reason in exclusions
        )
        scope = (
            f"{what}{len(exclusions)} e2e files that don't drive a model are not on the board. "
            f"The tests are forced"
        )
        excl_details = (
            f"<details><summary>The {len(exclusions)} files not on the board</summary>"
            f"<ul>{excl_items}</ul></details>"
        )
    else:
        scope = f"{what}The tests are forced"
        excl_details = ""
    # The judge each cell ran with, as the worker recorded it; results that
    # predate the field name none.
    judges = sorted({d["judge"] for d in runs.values() if d.get("judge")})
    judged = (
        "Answers the tests judge are graded by "
        + ", ".join(f"<code>{html.escape(j)}</code>" for j in judges)
        + ". "
        if judges
        else ""
    )
    H.append(
        "<footer><b>Methodology.</b> " + scope + " onto each model suite-wide via "
        "<code>OSPREY_E2E_FORCE_PROVIDER</code> + <code>OSPREY_E2E_FORCE_MODEL</code> (every model "
        "the agent uses collapses to one). OpenAI-protocol models route through OSPREY's "
        "Anthropic↔OpenAI translation proxy; Anthropic models connect directly. The per-test "
        "timeout is 1800s; a timeout <b>counts as a failure</b> (the model did not complete the "
        "task within the cap) and is shown as a distinct category so it can be told apart from a "
        "logic failure. A test that dies in setup never reached the model: it shows as an error "
        "and scores neither way. Anthropic Claude models, where present, are a reference ceiling: the "
        "models the tests were written for. "
        + judged
        + "Latency figures come from the agent SDK's own "
        "accounting of each query (<code>OSPREY_E2E_QUERY_LOG</code>) and from per-test wall time; "
        "a self-hosted model's speed depends on its hardware, named under each model with its "
        "provider." + excl_details + "</footer>"
    )
    H.append("</div>")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(H))
    print(f"wrote {args.out}  ({done}/{total_cells} runs, {len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
