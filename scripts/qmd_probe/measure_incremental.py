"""Measure incremental reindex cost on an already-built qmd index.

The initial index build is a one-time startup cost. What determines whether the
sidecar's update loop is viable is the *steady-state* cost:

1. **No-op update** -- the loop runs ``qmd update`` on its interval whether or
   not anything changed. If a no-op costs more than the interval, the loop never
   sleeps and pegs a core forever.
2. **One-new-entry update** -- what a single new logbook entry actually costs
   before it becomes findable.
3. **Embedding one new entry** -- vectors for the new document only.

Usage:
    python measure_incremental.py --project <qmd project dir> \
        --mirror <mirror root> --qmd <path to qmd bin> [--repeat 3]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run_timed(cmd: list[str], cwd: Path) -> tuple[float, str]:
    """Run a command and return (elapsed_seconds, combined output).

    Args:
        cmd: Argument vector.
        cwd: Working directory.

    Returns:
        Tuple of elapsed wall-clock seconds and captured output.
    """
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    return elapsed, (proc.stdout + proc.stderr)


def parse_counts(output: str) -> dict[str, int]:
    """Extract qmd's 'Indexed: N new, N updated, ...' line.

    Args:
        output: Combined command output.

    Returns:
        Mapping of qmd's per-run counters, empty if the line is absent.
    """
    for line in output.splitlines():
        if "Indexed:" in line:
            counts: dict[str, int] = {}
            for part in line.split("Indexed:", 1)[1].split(","):
                bits = part.strip().split()
                if len(bits) >= 2 and bits[0].isdigit():
                    counts[bits[1]] = int(bits[0])
            return counts
    return {}


def write_entry(mirror: Path, tag: str) -> Path:
    """Write one synthetic new logbook entry into the mirror.

    Args:
        mirror: Mirror root directory.
        tag: Unique suffix distinguishing this entry.

    Returns:
        Path of the file written.
    """
    now = datetime.now(tz=UTC)
    shard = mirror / now.strftime("%Y/%m")
    shard.mkdir(parents=True, exist_ok=True)
    path = shard / f"probe-incremental-{tag}.md"
    path.write_text(
        f"# Probe entry {tag}\n\n"
        f"Logged by qmd-probe on {now:%Y-%m-%d %H:%M:%S} UTC.\n"
        "Category: Operations. Level: Info. Source: ALS logbook.\n\n"
        f"Synthetic entry {tag} written by the qmd scale probe to measure "
        "incremental reindex latency. Mentions a distinctive token "
        f"zzprobetoken{tag} so it can be retrieved unambiguously.\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--mirror", required=True, type=Path)
    parser.add_argument("--qmd", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("incremental_report.json"))
    args = parser.parse_args()

    qmd = str(args.qmd)
    report: dict[str, Any] = {"noop_updates": [], "new_entry_updates": [], "embeds": []}
    created: list[Path] = []

    # 1. No-op updates: the steady-state cost of the loop's interval tick.
    for i in range(args.repeat):
        elapsed, out = run_timed([qmd, "update"], args.project)
        report["noop_updates"].append(
            {"trial": i, "seconds": round(elapsed, 2), "counts": parse_counts(out)}
        )
        print(f"  noop update {i}: {elapsed:.2f}s {parse_counts(out)}", flush=True)

    # 2. One new entry, then update: what a new logbook entry actually costs.
    for i in range(args.repeat):
        path = write_entry(args.mirror, f"{i}")
        created.append(path)
        elapsed, out = run_timed([qmd, "update"], args.project)
        report["new_entry_updates"].append(
            {"trial": i, "seconds": round(elapsed, 2), "counts": parse_counts(out)}
        )
        print(f"  +1 entry update {i}: {elapsed:.2f}s {parse_counts(out)}", flush=True)

        # 3. Embedding the newly indexed document.
        elapsed_embed, out_embed = run_timed([qmd, "embed"], args.project)
        report["embeds"].append({"trial": i, "seconds": round(elapsed_embed, 2)})
        print(f"  embed after +1  {i}: {elapsed_embed:.2f}s", flush=True)

    for path in created:
        path.unlink(missing_ok=True)
    # Let the index drop the removed probe entries so the corpus is left clean.
    elapsed, out = run_timed([qmd, "update"], args.project)
    report["cleanup_update"] = {"seconds": round(elapsed, 2), "counts": parse_counts(out)}

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    if shutil.which("node") is None:
        print("warning: node not found")
    raise SystemExit(main())
