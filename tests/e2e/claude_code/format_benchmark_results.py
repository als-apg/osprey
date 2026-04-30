"""Render channel-finder benchmark JSON results as a markdown table.

Pairs with the session-end dump in ``test_channel_finder_mcp_benchmarks.py``.
By default reads the most recent JSON in
``tests/e2e/claude_code/benchmark_results/``; pass a path to format a
specific run. Output is markdown on stdout.

Usage:
    uv run python tests/e2e/claude_code/format_benchmark_results.py
    uv run python tests/e2e/claude_code/format_benchmark_results.py path/to/run.json
    uv run python tests/e2e/claude_code/format_benchmark_results.py --diff old.json new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PIPELINE_LABELS = {
    "hierarchical": "Hierarchical",
    "middle_layer": "Middle Layer",
    "in_context": "In-Context",
}

PIPELINE_PREFIX = {
    "hier_": "hierarchical",
    "ml_": "middle_layer",
    "ic_": "in_context",
}


def _pipeline_for(qid: str) -> str:
    for prefix, name in PIPELINE_PREFIX.items():
        if qid.startswith(prefix):
            return name
    return "unknown"


def _status(f1: float) -> str:
    if f1 == 1.0:
        return "PERFECT"
    if f1 > 0.0:
        return "PARTIAL"
    return "MISS"


def _format_aggregates(aggregates: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## Pipeline aggregates", ""]
    lines.append(
        "| Pipeline | Perfect | Partial | Miss | Overall F1 | Perfect Match Rate | Total Cost |"
    )
    lines.append(
        "|----------|---------|---------|------|------------|--------------------|------------|"
    )
    for key, label in PIPELINE_LABELS.items():
        agg = aggregates.get(key)
        if not agg:
            continue
        lines.append(
            f"| {label} | {agg['perfect_count']} | {agg['partial_count']} | "
            f"{agg['no_match_count']} | {agg['overall_f1']:.3f} | "
            f"{agg['perfect_match_rate']:.0%} | ${agg['total_cost']:.4f} |"
        )
    return lines


def _format_per_query(results: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["", "## Per-query results", ""]
    lines.append("| Query | Status | F1 | Precision | Recall | Turns | Cost |")
    lines.append("|-------|--------|----|-----------|--------|-------|------|")

    by_pipeline: dict[str, list[dict[str, Any]]] = {p: [] for p in PIPELINE_LABELS}
    for qid, entry in results.items():
        pipeline = _pipeline_for(qid)
        if pipeline in by_pipeline:
            by_pipeline[pipeline].append(entry)

    for pipeline, entries in by_pipeline.items():
        if not entries:
            continue
        lines.append(f"| **— {PIPELINE_LABELS[pipeline]} —** | | | | | | |")
        for entry in sorted(entries, key=lambda e: e["query_id"]):
            cost = entry.get("cost_usd") or 0.0
            lines.append(
                f"| `{entry['query_id']}` | {_status(entry['f1'])} | "
                f"{entry['f1']:.2f} | {entry['precision']:.2f} | "
                f"{entry['recall']:.2f} | {entry['num_turns']} | ${cost:.4f} |"
            )
    return lines


def _format_misses(results: dict[str, dict[str, Any]]) -> list[str]:
    misses = [e for e in results.values() if e["f1"] < 1.0]
    if not misses:
        return ["", "_All queries scored a perfect F1._"]
    lines = ["", "## Imperfect queries — predicted vs. expected", ""]
    for entry in sorted(misses, key=lambda e: (e["f1"], e["query_id"])):
        lines.append(f"### `{entry['query_id']}` — F1 {entry['f1']:.2f}")
        lines.append(f"_Query:_ {entry['user_query']}")
        lines.append("")
        lines.append(f"- **Expected:** `{entry['expected']}`")
        lines.append(f"- **Predicted:** `{entry['predicted']}`")
        meta = entry.get("eval_meta", {})
        if meta.get("evaluation"):
            lines.append(f"- **Eval mode:** {meta['evaluation']}")
        lines.append("")
    return lines


def _latest_json(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("*.json"))
    return candidates[-1] if candidates else None


def render(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    ts = payload.get("run_timestamp_utc", "(unknown)")
    lines.append(f"# Channel Finder MCP Benchmark — {ts}")
    lines.append("")
    lines.extend(_format_aggregates(payload.get("aggregates", {})))
    lines.extend(_format_per_query(payload.get("results", {})))
    lines.extend(_format_misses(payload.get("results", {})))
    return "\n".join(lines) + "\n"


def render_diff(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Side-by-side aggregate comparison + per-query F1 deltas."""
    lines = ["# Channel Finder Benchmark — diff", ""]
    lines.append(f"- **Old:** {old.get('run_timestamp_utc', '(unknown)')}")
    lines.append(f"- **New:** {new.get('run_timestamp_utc', '(unknown)')}")
    lines.append("")
    lines.append("## Aggregate deltas")
    lines.append("")
    lines.append("| Pipeline | Old F1 | New F1 | Δ F1 | Old Perfect% | New Perfect% | Δ |")
    lines.append("|----------|--------|--------|------|--------------|--------------|---|")
    old_agg = old.get("aggregates", {})
    new_agg = new.get("aggregates", {})
    for key, label in PIPELINE_LABELS.items():
        o = old_agg.get(key)
        n = new_agg.get(key)
        if not o or not n:
            continue
        df1 = n["overall_f1"] - o["overall_f1"]
        dpm = n["perfect_match_rate"] - o["perfect_match_rate"]
        lines.append(
            f"| {label} | {o['overall_f1']:.3f} | {n['overall_f1']:.3f} | "
            f"{df1:+.3f} | {o['perfect_match_rate']:.0%} | "
            f"{n['perfect_match_rate']:.0%} | {dpm:+.0%} |"
        )
    lines.append("")
    lines.append("## Per-query F1 deltas (only queries that changed)")
    lines.append("")
    lines.append("| Query | Old F1 | New F1 | Δ |")
    lines.append("|-------|--------|--------|---|")
    o_res = old.get("results", {})
    n_res = new.get("results", {})
    for qid in sorted(set(o_res) | set(n_res)):
        of1 = o_res.get(qid, {}).get("f1")
        nf1 = n_res.get(qid, {}).get("f1")
        if of1 is None or nf1 is None or of1 == nf1:
            continue
        delta = nf1 - of1
        lines.append(f"| `{qid}` | {of1:.2f} | {nf1:.2f} | {delta:+.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to a benchmark JSON. Defaults to the most recent in "
        "tests/e2e/claude_code/benchmark_results/.",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("OLD", "NEW"),
        type=Path,
        help="Compare two benchmark JSON files instead of formatting one.",
    )
    args = parser.parse_args(argv)

    if args.diff:
        old = json.loads(args.diff[0].read_text())
        new = json.loads(args.diff[1].read_text())
        sys.stdout.write(render_diff(old, new))
        return 0

    target = args.path or _latest_json(Path(__file__).parent / "benchmark_results")
    if target is None or not target.exists():
        print(
            "No benchmark JSON found. Run the suite first or pass a path.",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(target.read_text())
    sys.stdout.write(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
