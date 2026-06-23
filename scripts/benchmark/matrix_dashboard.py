#!/usr/bin/env python
"""Render a self-contained HTML dashboard for the CBORG e2e model matrix (#259).

Reads results/<model>__seed<seed>.json (the summary schema emitted by
run_e2e_for_model.sh) and produces a single static HTML file (inline CSS, no
external deps) with:

  * run metadata + methodology footer (override mechanism, proxy, cap, drops),
  * a model x seed summary matrix (pass-rate heatmap + counts + wall-clock),
  * a per-test heatmap (92 test rows x model cols; cell = passes across seeds),
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
from collections import defaultdict

# Two provider lanes in one matrix:
#   * Open SUBJECTS (gpt-oss/gemma/cborg-coder/qwen) run through CBORG ×3 seeds.
#   * Anthropic REFERENCES run NATIVE als-apg (the Claude models never needed CBORG
#     translation, and CBORG's anthropic-direct path breaks safety/feedback tests
#     als-apg passes — validated 2026-06-18). Ref ids are the exact als-apg strings.
# weak -> strong within the open spread; qwen (CBORG-proxied) after the self-hosted.
MODEL_ORDER = ["gpt-oss-20b", "gemma-4", "cborg-coder", "gpt-oss-120b",
               "google/qwen-3-coder", "google/qwen-3",
               "claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"]
# Anthropic Claude models as a control/ceiling reference, flagged "(ref)".
# Ordered weak->strong (haiku < sonnet < opus) to bracket the open subjects.
REFERENCE_MODELS = {"claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"}
SEEDS = [1, 2, 3]
EXPECTED_TESTS = 33  # full e2e kit minus the 6 config exclusions (benchmark dropped)
OUTCOME_COLORS = {
    "passed": "#1a7f37", "failed": "#cf222e", "skipped": "#9a6700",
    "error": "#8250df", "timeout": "#bc4c00", "pending": "#8c959f",
}


_LIVE_KEY = {"passed": "passed", "failed": "failed", "timeout": "timeout",
             "skipped": "skipped", "error": "errors"}


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
        tests = []
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    tests.append(json.loads(line))
                except Exception:
                    pass
        if not tests:
            continue
        cnt = {"passed": 0, "failed": 0, "timeout": 0, "skipped": 0, "errors": 0}
        for t in tests:
            cnt[_LIVE_KEY.get(t.get("outcome"), "errors")] += 1
        runs[(model, seed)] = {
            "model": model, "seed": seed, "route": "", "pytest_rc": None,
            "total_duration_s": int(sum(t.get("duration_s", 0) for t in tests)),
            **cnt, "total": len(tests), "tests": tests, "_partial": True,
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
    keep the full id; this only affects what the reader sees. Reference models
    get a '(ref)' suffix so a baseline column is unmistakable."""
    name = model.split("/", 1)[1] if "/" in model else model
    return f"{name} (ref)" if model in REFERENCE_MODELS else name


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
    return f"{100*n/d:.0f}%" if d else "—"


OUTCOME_ORDER = ["passed", "failed", "timeout", "skipped", "error"]
OUTCOME_GLYPH = {"passed": "✓", "failed": "✗", "timeout": "⧗", "skipped": "∅", "error": "!"}


def run_counts(d: dict) -> dict:
    return {"passed": d.get("passed", 0), "failed": d.get("failed", 0),
            "timeout": d.get("timeout", 0), "skipped": d.get("skipped", 0),
            "error": d.get("errors", 0)}


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
    seg = [f"<span title='{o}: {counts[o]}' style='display:inline-block;height:{height}px;"
           f"width:{100*counts[o]/tot:.2f}%;background:{OUTCOME_COLORS[o]}'></span>"
           for o in OUTCOME_ORDER if counts.get(o)]
    return (f"<div style='width:{width}px;height:{height}px;border-radius:3px;overflow:hidden;"
            f"font-size:0;white-space:nowrap;border:1px solid #d8dee4'>" + "".join(seg) + "</div>")


def counts_text(counts: dict) -> str:
    return " ".join(f"<span style='color:{OUTCOME_COLORS[o]}'>{counts[o]}{OUTCOME_GLYPH[o]}</span>"
                    for o in OUTCOME_ORDER if counts.get(o))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/dashboard.html")
    args = ap.parse_args()

    runs = load(args.results_dir)
    started, ended, matrix_done = parse_progress(args.results_dir)
    running = started - ended  # combos in flight
    models = list(MODEL_ORDER)
    for m in sorted({m for (m, _) in runs} | {m for (m, _) in started}):
        if m not in models:
            models.append(m)

    # union of all test names, grouped by file; (file,test) -> model -> {seed: outcome}
    all_tests: dict[str, set] = defaultdict(set)
    test_results: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for (m, s), d in runs.items():
        for t in d.get("tests", []):
            name = t["name"]
            file = name.split("::")[0].replace("tests/e2e/", "")
            short = "::".join(name.split("::")[1:]) or name
            all_tests[file].add(short)
            test_results[(file, short)][m][s] = t["outcome"]

    done = sum(1 for d in runs.values() if not d.get("_partial"))
    live_n = sum(1 for d in runs.values() if d.get("_partial"))
    total_cells = sum(len(seeds_for(m, runs)) for m in models)  # subjects: 3 seeds; refs: the seeds actually run

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
    .card{display:inline-block;background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:14px 18px;margin:0 12px 12px 0;vertical-align:top}
    .big{font-size:26px;font-weight:700} .muted{color:#636c76;font-size:12px}
    .pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;color:#fff}
    .legend span{margin-right:14px} .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
    .filerow td{background:#eef1f4;font-weight:600;text-align:left}
    footer{margin-top:34px;color:#636c76;font-size:12.5px;border-top:1px solid #d0d7de;padding-top:14px}
    code{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px}
    """

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = ("✓ matrix complete" if matrix_done else
              (f"● live — {len(running)} running" if running else "○ idle"))
    refresh = "" if matrix_done else '<meta http-equiv="refresh" content="60">'

    H = []
    H.append(f"<!doctype html><meta charset=utf-8>{refresh}"
             f"<title>CBORG e2e model matrix</title><style>{css}</style>")
    H.append("<div class=wrap>")
    H.append("<h1>CBORG open models — full OSPREY e2e suite</h1>")
    H.append(f"<p class=sub>Issue #259 · {len(models)} models × {len(SEEDS)} seeds · "
             f"per-test hang-breaker 1800s · <b>{done}/{total_cells}</b> runs complete · "
             + (f"<b>{live_n} filling live</b> · " if live_n else "")
             + f"<b>{html.escape(status)}</b> · updated {now}"
             + (" · auto-refresh 60s" if not matrix_done else "") + "</p>")
    if running:
        H.append("<p class=sub>running now: " +
                 ", ".join(f"<code>{html.escape(label(m))} seed{s}</code>" for (m, s) in sorted(running)) + "</p>")

    # legend
    H.append("<div class=legend style='margin-bottom:18px'>")
    for k, c in OUTCOME_COLORS.items():
        H.append(f"<span><span class=dot style='background:{c}'></span>{k}</span>")
    H.append("</div>")

    # ---- model summary cards ----
    H.append("<h2>Per-model summary (aggregated over seeds)</h2>")
    for m in models:
        m_seeds = seeds_for(m, runs)
        ds = [runs[(m, s)] for s in m_seeds if (m, s) in runs]
        if not ds:
            H.append(f"<div class=card><div class=big>{html.escape(label(m))}</div>"
                     f"<div class=muted>pending</div></div>")
            continue
        ps = sum(d["passed"] for d in ds)
        denom = sum(conclusive(d)[1] for d in ds)
        dur = sum(d["total_duration_s"] for d in ds) / max(len(ds), 1)
        c = {o: sum(run_counts(d)[o] for d in ds) for o in OUTCOME_ORDER}
        # one stacked bar PER SEED so seed-to-seed consistency is visible
        bars = []
        for s in m_seeds:
            d = runs.get((m, s))
            if d and d["total"]:
                p_s, den_s = conclusive(d)
                bars.append(
                    f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
                    f"<span class=muted style='width:36px'>seed{s}</span>"
                    f"{stacked_bar(run_counts(d), width=150, height=13)}"
                    f"<span class=muted>{pct(p_s, den_s)}</span></div>")
            else:
                state = "● running" if (m, s) in running else "· pending"
                bars.append(
                    f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
                    f"<span class=muted style='width:36px'>seed{s}</span>"
                    f"<span class=muted>{state}</span></div>")
        H.append(
            f"<div class=card><div class=big>{html.escape(label(m))}</div>"
            f"<div><b style='color:{OUTCOME_COLORS['passed']}'>{pct(ps,denom)}</b> pass "
            f"<span class=muted>(conclusive; {ps}/{denom})</span></div>"
            f"<div style='margin:7px 0'>{''.join(bars)}</div>"
            f"<div class=muted>{counts_text(c)}</div>"
            f"<div class=muted>~{dur/60:.0f} min/run</div></div>"
        )

    # ---- model x seed matrix (full outcome breakdown per cell) ----
    H.append("<h2>Outcome breakdown by model × seed</h2>")
    H.append("<p class=sub>Each cell shows the full mix for one run — pass% (of conclusive tests; "
             "timeouts &amp; skips excluded from the denominator, so a slow test is never a false negative) "
             "plus a stacked bar and counts (✓ pass · ✗ fail · ⧗ timeout · ∅ skip · ! error).</p>")
    H.append("<table>")
    H.append("<tr><th class=l>model</th>" + "".join(f"<th>seed {s}</th>" for s in SEEDS) + "<th>mean pass</th></tr>")
    for m in models:
        H.append(f"<tr><td class=l>{html.escape(label(m))}</td>")
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
                tag = (f"<div class=muted style='color:#9a6700'>● live · {d['total']}/{EXPECTED_TESTS} done</div>"
                       if d.get("_partial")
                       else f"<div class=muted>{d['total']}t · {d['total_duration_s']//60}m</div>")
                H.append(f"<td><div style='font-weight:600'>{pct(p_s, den_s)}</div>"
                         f"<div style='display:flex;justify-content:center;margin:3px 0'>{stacked_bar(c)}</div>"
                         f"<div class=muted>{counts_text(c)}</div>"
                         f"{tag}</td>")
            elif d and not d["total"]:
                H.append("<td style='background:#ffd8c2'>err<div class=muted>0 tests</div></td>")
            elif (m, s) in running:
                H.append("<td style='background:#fff3cd'>●<div class=muted>running</div></td>")
            else:
                H.append("<td style='background:#f6f8fa'>·<div class=muted>pending</div></td>")
        mean = sum(fracs) / len(fracs) if fracs else None
        H.append(f"<td style='background:{heat(mean)}'><b>{(f'{100*mean:.0f}%' if mean is not None else '·')}</b></td></tr>")
    H.append("</table>")

    # ---- per-test heatmap (one square per seed) ----
    H.append("<h2>Per-test outcomes — one square per seed</h2>")
    H.append("<p class=sub>For every test, the three squares are seed 1·2·3 colored by outcome "
             "(green pass · red fail · orange timeout · yellow skip · purple error · grey pending). "
             "Lets you spot a test a model passes once but times out on another seed.</p>")
    H.append("<table><tr><th class=l>test</th>" + "".join(f"<th>{html.escape(label(m))}</th>" for m in models) + "</tr>")
    for file in sorted(all_tests):
        H.append(f"<tr class=filerow><td colspan={len(models)+1}>{html.escape(file)}</td></tr>")
        for short in sorted(all_tests[file]):
            H.append(f"<tr><td class=l>{html.escape(short)}</td>")
            for m in models:
                seeds_map = test_results[(file, short)].get(m, {})
                m_seeds = seeds_for(m, runs)
                dots = []
                for s in m_seeds:  # ref models render a single square, not 3
                    o = seeds_map.get(s)
                    col = OUTCOME_COLORS[o] if o in OUTCOME_COLORS else "#eaeef2"
                    dots.append(f"<span title='seed{s}: {o or 'pending'}' style='display:inline-block;"
                                f"width:13px;height:13px;border-radius:2px;margin:1px;background:{col}'></span>")
                H.append(f"<td>{''.join(dots)}</td>")
            H.append("</tr>")
    H.append("</table>")

    # ---- methodology ----
    H.append(
        "<footer><b>Methodology.</b> The 36 model-driving e2e tests (of 92 collected — docker-build, "
        "ariel-search, provider-plumbing, and build-integration tests are excluded because they don't call "
        "an LLM and/or don't route through the model under test) are forced onto each "
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
        "<b>(ref)</b> columns — <code>claude-haiku-4-5</code> (the suite's literal default tier), "
        "<code>claude-sonnet-4-6</code>, and <code>claude-opus-4-6</code> — bracket the open models weak→strong "
        "as a control/ceiling, showing how they compare against the models the tests were written for. Open subjects "
        "run on the Mac Studio via CBORG; the Anthropic <b>(ref)</b> columns route natively via als-apg.</footer>")
    H.append("</div>")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(H))
    print(f"wrote {args.out}  ({done}/{total_cells} runs, {len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
