#!/usr/bin/env python
"""Render a self-contained HTML dashboard for the CBORG e2e model matrix (#259).

Reads results/<model>__seed<seed>.json (the summary schema emitted by
run_e2e_for_model.sh) and produces a single static HTML file (inline CSS, no
external deps) with:

  * run metadata + methodology footer (override mechanism, proxy, cap, drops),
  * a model x seed summary matrix (pass-rate heatmap + counts + wall-clock),
  * a per-test heatmap (one row per model-driving test x model cols; cell = passes across seeds),
    grouped by test file, so you can see exactly which capabilities each model
    holds up on weak -> strong.

Partial-data friendly: missing (model,seed) runs render as "pending". Usage:
    scripts/benchmark/matrix_dashboard.py --results-dir results --out results/dashboard.html
"""

from __future__ import annotations

import argparse
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
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]
# Anthropic Claude models as a control/ceiling reference, flagged "(ref)".
# Ordered weak->strong (haiku < sonnet < opus) to bracket the open subjects.
REFERENCE_MODELS = {"claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"}
SEEDS = [1, 2, 3]
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
        txt = open(run_log_path, errors="ignore").read()
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
        cfg = json.load(open(config_path))
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
        try:
            d = json.load(open(f))
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
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    raw.append(json.loads(line))
                except Exception:
                    pass
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


def parse_progress(results_dir: str):
    """Read matrix.log -> (started, ended, matrix_done). A combo START-ed but not
    END-ed is currently running."""
    started, ended, done = set(), set(), False
    path = os.path.join(results_dir, "matrix.log")
    if not os.path.exists(path):
        return started, ended, done
    rx = re.compile(r">> (START|END)\s+(\S+)\s+seed(\d+)")
    for line in open(path):
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
    """Render the canonical provider for the [provider] line. ds4 is fully
    self-hosted (local DeepSeek V4 on the Mac Studio, no external API), so flag
    the host to distinguish it from the proxied/remote providers."""
    if provider == "ds4":
        return "ds4 · macstudio (self-hosted)"
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


def conclusive(d: dict):
    """Pass rate denominator EXCLUDES timeouts and skips, so a 'too slow' result
    is never scored as a failure (no false negatives from the timeout). Returns
    (passed, conclusive_total)."""
    p, f, e = d.get("passed", 0), d.get("failed", 0), d.get("errors", 0)
    return p, p + f + e


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
    models = list(MODEL_ORDER)
    for m in sorted({m for (m, _) in runs} | {m for (m, _) in started}):
        if m not in models:
            models.append(m)
    if args.artifact:
        # A published artifact shows only models that actually have results — no
        # empty "pending" rows for models in MODEL_ORDER that weren't part of this run.
        with_data = {m for (m, _s) in runs}
        models = [m for m in models if m in with_data]

    # union of all test names, grouped by file; (file,test) -> model -> {seed: outcome}
    all_tests: dict[str, set] = defaultdict(set)
    test_results: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for (m, s), d in runs.items():
        for t in d.get("tests", []):
            file, short = canon_name(t["name"])
            all_tests[file].add(short)
            test_results[(file, short)][m][s] = t["outcome"]

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
    title = "OSPREY model-capability benchmark" if args.artifact else "CBORG e2e model matrix"

    H = []
    H.append(
        f"<!doctype html><meta charset=utf-8>{refresh}<title>{title}</title><style>{css}</style>"
    )
    H.append("<div class=wrap>")
    if args.artifact:
        # Timeless artifact: no issue ref, progress counters, live status,
        # timestamp, auto-refresh or running-now line — just title and data.
        H.append("<h1>OSPREY model-capability benchmark</h1>")
        H.append(
            f"<p class=sub>Open-weight &amp; self-hosted models across the full "
            f"<code>tests/e2e/</code> suite · {len(models)} models × {len(SEEDS)} seeds</p>"
        )
    else:
        H.append("<h1>CBORG open models — full OSPREY e2e suite</h1>")
        H.append(
            f"<p class=sub>Issue #259 · {len(models)} models × {len(SEEDS)} seeds · "
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

    # ---- model x seed matrix (full outcome breakdown per cell) ----
    H.append("<h2>Outcome breakdown by model × seed</h2>")
    H.append(
        "<p class=sub>Each cell shows the full mix for one run — pass% (of conclusive tests; "
        "timeouts &amp; skips excluded from the denominator, so a slow test is never a false negative) "
        "plus a stacked bar and counts (✓ pass · ✗ fail · ⧗ timeout · ∅ skip · ! error).</p>"
    )
    H.append("<table>")
    H.append(
        "<tr><th class=l>model</th>"
        + "".join(f"<th>seed {s}</th>" for s in SEEDS)
        + "<th>mean pass</th></tr>"
    )
    sep_done = False
    for m in models:
        # Light divider between the self-hosted/open models and the Anthropic
        # reference bracket (drawn once, before the first reference row).
        if not sep_done and m in REFERENCE_MODELS:
            H.append(f"<tr class=sep><td colspan='{len(SEEDS) + 2}'></td></tr>")
            sep_done = True
        H.append(
            f"<tr><td class=l>{html.escape(label(m))}"
            f"<div class=prov>[{html.escape(provider_display(provider_of(m, runs)))}]</div></td>"
        )
        fracs = []
        m_seeds = seeds_for(m, runs)
        for s in SEEDS:
            if s not in m_seeds:  # ref models: seed not planned, not "pending"
                H.append("<td style='background:#fafbfc'><span class=muted>n/a</span></td>")
                continue
            d = runs.get((m, s))
            if d and d["total"]:
                p_s, den_s = conclusive(d)
                fr = (p_s / den_s) if den_s else None
                if fr is not None and not d.get("_partial"):
                    fracs.append(fr)  # mean is over COMPLETED seeds only
                c = run_counts(d)
                if d.get("_partial"):
                    # Incomplete cell (still filling, or interrupted before all
                    # tests ran): show its REAL partial bar + pass% with a plain
                    # progress count. No live/died/stalled ops-language on a
                    # shared dashboard — just "<ran>/<total>".
                    tag = f"<div class=muted>{d['total']}/{model_driving_n or '?'}</div>"
                else:
                    tag = f"<div class=muted>{d['total']}t · {d['total_duration_s'] // 60}m</div>"
                H.append(
                    f"<td><div style='font-weight:600'>{pct(p_s, den_s)}</div>"
                    f"<div style='display:flex;justify-content:center;margin:3px 0'>{stacked_bar(c)}</div>"
                    f"<div class=muted>{counts_text(c)}</div>"
                    f"{tag}</td>"
                )
            elif d and not d["total"]:
                H.append("<td style='background:#ffd8c2'>err<div class=muted>0 tests</div></td>")
            elif (m, s) in running_active:
                H.append("<td style='background:#fff3cd'>●<div class=muted>running</div></td>")
            else:
                H.append("<td style='background:#f6f8fa'>·<div class=muted>pending</div></td>")
        mean = sum(fracs) / len(fracs) if fracs else None
        H.append(
            f"<td style='background:{heat(mean)}'><b>{(f'{100 * mean:.0f}%' if mean is not None else '·')}</b></td></tr>"
        )
    H.append("</table>")

    # ---- per-test heatmap (one square per seed) ----
    H.append("<h2>Per-test outcomes — one square per seed</h2>")
    H.append(
        "<p class=sub>For every test, the three squares are seed 1·2·3 colored by outcome "
        "(green pass · red fail · orange timeout · yellow skip · purple error · grey pending). "
        "Lets you spot a test a model passes once but times out on another seed.</p>"
    )
    H.append(
        "<table><tr><th class=l>test</th>"
        + "".join(f"<th>{html.escape(label(m))}</th>" for m in models)
        + "</tr>"
    )
    for file in sorted(all_tests):
        H.append(f"<tr class=filerow><td colspan={len(models) + 1}>{html.escape(file)}</td></tr>")
        for short in sorted(all_tests[file]):
            H.append(f"<tr><td class=l>{html.escape(short)}</td>")
            for m in models:
                seeds_map = test_results[(file, short)].get(m, {})
                m_seeds = seeds_for(m, runs)
                dots = []
                for s in m_seeds:  # ref models render a single square, not 3
                    o = seeds_map.get(s)
                    col = OUTCOME_COLORS[o] if o in OUTCOME_COLORS else "#eaeef2"
                    dots.append(
                        f"<span title='seed{s}: {o or 'pending'}' style='display:inline-block;"
                        f"width:13px;height:13px;border-radius:2px;margin:1px;background:{col}'></span>"
                    )
                H.append(f"<td>{''.join(dots)}</td>")
            H.append("</tr>")
    H.append("</table>")

    # ---- methodology ----
    # Scope sentence is DERIVED: the model-driving count from the run data and the
    # excluded-file list (with reasons) from matrix_e2e_config.json — so it tracks
    # the suite instead of asserting frozen totals (was "36 of 92", long stale).
    n_txt = str(model_driving_n) if model_driving_n is not None else "—"
    if exclusions:
        excl_html = ", ".join(
            f"<code title='{html.escape(reason)}'>{html.escape(os.path.basename(rel))}</code>"
            for rel, reason in exclusions
        )
        scope = (
            f"The <b>{n_txt}</b> model-driving e2e tests — the full <code>tests/e2e/</code> suite "
            f"minus the {len(exclusions)} files that don't call an LLM and/or don't route through the "
            f"model under test (hover for the reason: {excl_html}) — are forced"
        )
    else:
        scope = (
            f"The <b>{n_txt}</b> model-driving e2e tests (the full <code>tests/e2e/</code> suite minus "
            f"the files that don't call an LLM and/or don't route through the model under test) are forced"
        )
    H.append(
        "<footer><b>Methodology.</b> " + scope + " onto each "
        "CBORG model suite-wide via <code>OSPREY_E2E_FORCE_PROVIDER=cborg</code> + "
        "<code>OSPREY_E2E_FORCE_MODEL</code> (all tiers collapse to one model). Open models route through "
        "OSPREY's Anthropic↔OpenAI translation proxy. The per-test timeout is 1800s and serves only as a "
        "deadlock-breaker (~5× the slowest legitimate test observed); a timeout is recorded as its own "
        "<b>inconclusive</b> category and EXCLUDED from the pass-rate denominator, so a slow-but-capable "
        "model is never scored a false negative. Models were chosen from a 14-model lightweight probe to span "
        "weak/fast→big/slow across distinct open families (gpt-oss, gemma, cborg). "
        "<code>qwen-3</code> / <code>qwen-3-coder</code> are open-weight but CBORG-proxied "
        "(<code>google/</code> route, not <code>lbl/</code> self-hosted), included as an additional open "
        "family. Dropped at probe: <code>cborg-instant</code>, <code>cborg-instant-short</code> (could not "
        "relay a tool result; short context truncates the harness prompt). Closed-weight commercial models "
        "(GPT/Gemini/…) are out of scope as study subjects, but single-seed Anthropic Claude "
        "reference models — <code>claude-haiku-4-5</code> (the suite's literal default tier), "
        "<code>claude-sonnet-4-6</code>, and <code>claude-opus-4-6</code> — bracket the open models weak→strong "
        "as a control/ceiling, showing how they compare against the models the tests were written for. Open subjects "
        "run on the Mac Studio via CBORG; the Anthropic reference rows route natively via als-apg (see the "
        "<code>[provider]</code> line under each model).</footer>"
    )
    H.append("</div>")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(H))
    print(f"wrote {args.out}  ({done}/{total_cells} runs, {len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
