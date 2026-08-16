"""Measure qmd retrieval latency against a running qmd daemon.

Part of the qmd scale probe. Answers one question: at real ALS logbook scale,
is qmd fast enough to sit behind an interactive search surface?

Latency is measured with *distinct* queries. qmd caches results, so replaying
one query would report cache-hit timings rather than retrieval timings.

Usage:
    python run_probe.py --base-url http://localhost:8181 \
        --collection ariel --out report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from qmd_mcp import QmdMcpClient, QmdMcpError

HERE = Path(__file__).parent


def percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values`` by nearest-rank.

    Args:
        values: Sample values; must be non-empty.
        pct: Percentile in the range 0-100.

    Returns:
        The percentile value.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1)
    return ordered[max(0, index)]


def host_specs() -> dict[str, Any]:
    """Collect host specifications for the probe report.

    Only aggregate hardware facts are gathered -- no corpus content.

    Returns:
        A mapping of host specification fields.
    """
    specs: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        if platform.system() == "Darwin":
            specs["cpu_model"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            mem = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            specs["memory_gb"] = round(int(mem) / 1024**3, 1) if mem.isdigit() else None
            specs["cpu_count"] = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
    except (OSError, ValueError):  # pragma: no cover - best effort
        pass
    return specs


def dir_bytes(path: Path) -> int:
    """Return the total size of ``path`` in bytes, or 0 if absent."""
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def run_pass(
    client: QmdMcpClient,
    queries: list[dict[str, str]],
    collection: str | None,
    rerank: bool,
    limit: int,
) -> dict[str, Any]:
    """Time one latency pass over the query batch.

    Args:
        client: Connected MCP client.
        queries: Query batch; each entry supplies a ``lex`` and ``vec`` form.
        collection: Restrict search to this collection, or None for all.
        rerank: Whether to enable qmd's LLM reranker.
        limit: Result limit per query.

    Returns:
        A mapping of latency statistics and per-query samples.
    """
    samples: list[float] = []
    failures = 0
    hits: list[int] = []

    for spec in queries:
        searches = [
            {"type": "lex", "query": spec["lex"]},
            {"type": "vec", "query": spec["vec"]},
        ]
        args: dict[str, Any] = {"searches": searches, "limit": limit, "rerank": rerank}
        if collection:
            args["collections"] = [collection]

        start = time.perf_counter()
        try:
            result = client.call_tool("query", args)
        except QmdMcpError:
            failures += 1
            continue
        samples.append((time.perf_counter() - start) * 1000.0)

        # Results live under structuredContent; the `content` block is a
        # human-readable rendering of the same thing.
        structured = result.get("structuredContent") or {}
        hits.append(len(structured.get("results") or []))

    if not samples:
        return {"error": "all queries failed", "failures": failures}

    return {
        "queries_run": len(samples),
        "failures": failures,
        "rerank": rerank,
        "limit": limit,
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(percentile(samples, 95), 1),
        "p99_ms": round(percentile(samples, 99), 1),
        "min_ms": round(min(samples), 1),
        "max_ms": round(max(samples), 1),
        "mean_ms": round(statistics.mean(samples), 1),
        "mean_results_returned": round(statistics.mean(hits), 1) if hits else 0,
        "samples_ms": [round(s, 1) for s in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8181")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--queries", type=Path, default=HERE / "queries.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mirror-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--index-seconds", type=float, default=None)
    parser.add_argument("--embed-seconds", type=float, default=None)
    parser.add_argument("--entries", type=int, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "probe_report.json")
    args = parser.parse_args()

    batch = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]

    client = QmdMcpClient(args.base_url)
    health = client.health()
    client.connect()

    # One throwaway query loads the models so the reported numbers describe a
    # warm daemon, which is how the sidecar actually serves traffic.
    cold_start = time.perf_counter()
    client.call_tool(
        "query",
        {"searches": [{"type": "lex", "query": "warmup"}], "limit": 1, "rerank": True},
    )
    cold_ms = (time.perf_counter() - cold_start) * 1000.0

    report: dict[str, Any] = {
        "corpus": {
            "entries": args.entries,
            "mirror_bytes": dir_bytes(args.mirror_dir) if args.mirror_dir else None,
            "index_bytes": dir_bytes(args.index_dir) if args.index_dir else None,
        },
        "build": {
            "bm25_index_seconds": args.index_seconds,
            "embed_seconds": args.embed_seconds,
        },
        "daemon": {"health": health, "cold_first_query_ms": round(cold_ms, 1)},
        "host": host_specs(),
        "qmd_version": "2.5.3",
        "passes": {},
    }

    for label, rerank in (("hybrid_reranked", True), ("hybrid_no_rerank", False)):
        print(f"  pass: {label} ...", flush=True)
        report["passes"][label] = run_pass(client, batch, args.collection, rerank, args.limit)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {args.out}")
    print(json.dumps({k: v for k, v in report.items() if k != "passes"}, indent=2))
    for label, stats in report["passes"].items():
        trimmed = {k: v for k, v in stats.items() if k != "samples_ms"}
        print(f"{label}: {json.dumps(trimmed)}")
    return 0


if __name__ == "__main__":
    if shutil.which("node") is None:
        print("warning: node not found; a qmd daemon must already be running")
    raise SystemExit(main())
