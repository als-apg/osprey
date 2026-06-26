#!/usr/bin/env python3
"""Generate the complete benchmark suite: 9 databases + 3 query sets.

Produces:
  data/benchmarks/cross_paradigm/
    tier1/  (in_context.json, hierarchical.json, middle_layer.json)
    tier2/  (in_context.json, hierarchical.json, middle_layer.json)
    tier3/  (in_context.json, hierarchical.json, middle_layer.json)
    queries/
      tier1_queries.json
      tier2_queries.json
      tier3_queries.json

Usage:
    python scripts/generate_benchmark_suite.py
    python scripts/generate_benchmark_suite.py --output-dir /tmp/benchmarks
    python scripts/generate_benchmark_suite.py --validate
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from osprey.services.channel_finder.benchmarks.generator import (
    TEMPLATE_DB_PATH,
    TIER_1,
    TIER_2,
    TIER_3,
    expand_hierarchy,
    format_hierarchical,
    format_in_context,
    format_middle_layer,
    validate_queries,
)

TIERS = [(1, TIER_1), (2, TIER_2), (3, TIER_3)]
QUERY_SOURCE_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "cross_paradigm" / "queries"
)


def generate_suite(output_dir: Path) -> None:
    """Generate all 9 benchmark databases."""
    print(f"Loading template from {TEMPLATE_DB_PATH}")
    tree_data = json.loads(TEMPLATE_DB_PATH.read_text(encoding="utf-8"))
    channels = expand_hierarchy(tree_data)
    print(f"Expanded {len(channels)} total channels")

    for tier_num, tier_spec in TIERS:
        tier_dir = output_dir / f"tier{tier_num}"
        tier_dir.mkdir(parents=True, exist_ok=True)

        # In-context
        ic_data = format_in_context(channels, tier_spec)
        (tier_dir / "in_context.json").write_text(json.dumps(ic_data, indent=2), encoding="utf-8")

        # Hierarchical
        hier_data = format_hierarchical(tree_data, tier_spec)
        (tier_dir / "hierarchical.json").write_text(
            json.dumps(hier_data, indent=2), encoding="utf-8"
        )

        # Middle layer
        ml_data = format_middle_layer(channels, tier_spec)
        (tier_dir / "middle_layer.json").write_text(json.dumps(ml_data, indent=2), encoding="utf-8")

        # Channel count from in-context (envelope format)
        count = len(ic_data["channels"])
        print(f"  Tier {tier_num}: {count} channels (3 databases)")

    # Copy query files
    queries_dir = output_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    for tier_num in (1, 2, 3):
        src = QUERY_SOURCE_DIR / f"tier{tier_num}_queries.json"
        dst = queries_dir / f"tier{tier_num}_queries.json"
        if src.exists():
            shutil.copy2(src, dst)
            queries = json.loads(src.read_text(encoding="utf-8"))
            print(f"  Copied {src.name} ({len(queries)} queries)")
        else:
            print(f"  WARNING: {src} not found, skipping")

    print(f"\nSuite generated in {output_dir}/")


def run_validation(output_dir: Path) -> bool:
    """Run per-tier validation against generated databases."""
    print("\nValidating query sets against databases...")

    queries_dir = output_dir / "queries"
    tier_queries: dict[int, Path] = {}

    for tier_num in (1, 2, 3):
        path = queries_dir / f"tier{tier_num}_queries.json"
        if path.exists():
            tier_queries[tier_num] = path

    if not tier_queries:
        print("  No query files found, skipping validation")
        return True

    result = validate_queries(tier_queries=tier_queries, output_dir=output_dir)

    print(f"  Total queries: {result['total_queries']}")
    print(f"  Total unique PVs: {result['total_pvs']}")

    if result["missing_databases"]:
        print(f"  Missing databases: {result['missing_databases']}")

    if result["missing"]:
        print(f"  Missing PVs: {len(result['missing'])}")
        for entry in result["missing"][:5]:
            print(f"    tier{entry['tier']}/{entry['format']}: {entry['pv']}")
        if len(result["missing"]) > 5:
            print(f"    ... and {len(result['missing']) - 5} more")

    if result["valid"]:
        print("  Validation PASSED")
    else:
        print("  Validation FAILED")

    return result["valid"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the complete benchmark suite (9 DBs + 3 query sets)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks/cross_paradigm"),
        help="Output directory (default: data/benchmarks/cross_paradigm/)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run per-tier validation after generation",
    )
    args = parser.parse_args()

    generate_suite(args.output_dir)

    if args.validate:
        valid = run_validation(args.output_dir)
        if not valid:
            sys.exit(1)


if __name__ == "__main__":
    main()
