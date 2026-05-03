#!/usr/bin/env python3
"""Generate cross-paradigm benchmark databases for all tier x format combinations.

Usage:
    uv run python scripts/generate_tier_databases.py --tier all --format all
    uv run python scripts/generate_tier_databases.py --tier 1 --format in_context
    uv run python scripts/generate_tier_databases.py --output-dir /tmp/bench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from osprey.services.channel_finder.benchmarks.generator import (
    TIER_1,
    TIER_2,
    TIER_3,
    TierSpec,
    filter_channels,
    format_hierarchical,
    format_in_context,
    format_middle_layer,
    load_template,
    validate_queries,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mapping from CLI tier names to tier spec objects
TIER_SPECS: dict[str, TierSpec] = {
    "1": TIER_1,
    "2": TIER_2,
    "3": TIER_3,
}

ALL_TIERS = ["1", "2", "3"]
ALL_FORMATS = ["in_context", "hierarchical", "middle_layer"]

DEFAULT_OUTPUT_DIR = str(
    _PROJECT_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_databases"
    / "tiers"
)
DEFAULT_QUERIES_PATH = str(
    _PROJECT_ROOT / "data" / "benchmarks" / "queries" / "benchmark_queries.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate cross-paradigm benchmark databases.",
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which tier(s) to generate (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["in_context", "hierarchical", "middle_layer", "all"],
        default="all",
        dest="fmt",
        help="Which format(s) to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Generate databases, then validate that all targeted PVs exist",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate (skip generation)",
    )
    return parser.parse_args(argv)


def generate(
    tiers: list[str],
    formats: list[str],
    output_dir: Path,
) -> None:
    """Load the template, generate databases, and write output files.

    Args:
        tiers: List of tier numbers as strings ("1", "2", "3").
        formats: List of format names ("in_context", "hierarchical",
            "middle_layer").
        output_dir: Root directory for writing output files.
    """
    # Load the hierarchical template and expand to flat channels
    tree_data, all_channels = load_template()

    summary_lines: list[str] = []

    for tier_key in tiers:
        tier_spec = TIER_SPECS[tier_key]
        tier_dir = output_dir / f"tier{tier_key}"
        tier_dir.mkdir(parents=True, exist_ok=True)

        # Channel count is the same regardless of output format
        tier_channels = filter_channels(all_channels, tier_spec)
        channel_count = len(tier_channels)
        generated_formats: list[str] = []

        for fmt in formats:
            if fmt == "in_context":
                data: list | dict = format_in_context(all_channels, tier_spec)
            elif fmt == "hierarchical":
                data = format_hierarchical(tree_data, tier_spec)
            elif fmt == "middle_layer":
                data = format_middle_layer(all_channels, tier_spec)
            else:
                continue

            out_path = tier_dir / f"{fmt}.json"
            out_path.write_text(
                json.dumps(data, indent=2) + "\n",
                encoding="utf-8",
            )
            generated_formats.append(fmt)

        summary_lines.append(
            f"  Tier {tier_key}: {channel_count} channels ({', '.join(generated_formats)})"
        )

    print("Generated databases:")
    for line in summary_lines:
        print(line)


def run_validation(output_dir: Path) -> bool:
    """Run query validation and print results.

    Returns:
        True if all PVs found, False otherwise.
    """
    queries_path = Path(DEFAULT_QUERIES_PATH)
    report = validate_queries(queries_path, output_dir)

    if report["valid"]:
        print(
            f"All {report['total_pvs']} PVs found in all 9 databases "
            f"({report['total_queries']} queries)"
        )
        return True

    missing_dbs = report.get("missing_databases", [])
    if missing_dbs:
        print(f"MISSING DATABASE FILES: {len(missing_dbs)} file(s)")
        for db_path in missing_dbs:
            print(f"  {db_path}")

    print(
        f"VALIDATION FAILED: {len(report['missing'])} missing entries "
        f"({report['total_queries']} queries, {report['total_pvs']} unique PVs)"
    )
    for entry in report["missing"]:
        print(
            f"  query {entry['query_id']}: {entry['pv']} "
            f"missing in tier{entry['tier']}/{entry['format']}"
        )
    return False


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI script."""
    args = parse_args(argv)

    output_dir = Path(args.output_dir)

    if args.validate and args.validate_only:
        print("Error: --validate and --validate-only are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    if args.validate_only:
        if not run_validation(output_dir):
            sys.exit(1)
        return

    tiers = ALL_TIERS if args.tier == "all" else [args.tier]
    formats = ALL_FORMATS if args.fmt == "all" else [args.fmt]
    generate(tiers, formats, output_dir)

    if args.validate:
        if not run_validation(output_dir):
            sys.exit(1)


if __name__ == "__main__":
    main()
