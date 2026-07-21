#!/usr/bin/env python3
"""Regenerate the cross-paradigm tier channel databases from the facility spec.

Thin wrapper over
:func:`osprey.services.channel_finder.tools.generate_from_spec.generate` -- the
single canonical regeneration of every committed channel-database artifact. That
pipeline grows the tier-3 cross-paradigm views (``in_context``, ``hierarchical``,
``middle_layer``) from the facility spec and derives tier-1's ``in_context``
subset from them.

Tier 2 is retired: only tiers 1 and 3 are published. Tier 1 ships the
``in_context`` paradigm only; tier 3 ships all three. The published tiers (and
the paradigms each ships) come from
:data:`osprey.services.channel_finder.benchmarks.generator.TIER_PARADIGMS`.
Because regeneration is spec-driven and atomic, all published tiers/paradigms
are always regenerated together -- there is no per-tier or per-format selection.

Usage:
    uv run python scripts/generate_tier_databases.py
    uv run python scripts/generate_tier_databases.py --output-dir /tmp/bench
    uv run python scripts/generate_tier_databases.py --validate
    uv run python scripts/generate_tier_databases.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add scripts/ to path for the sibling shared-helper import under direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tier_db_common import (  # noqa: E402
    PRESET_TIERS_DIR,
    collect_tier_queries,
    regenerate_tier_databases,
    validate_queries,
)


def generate_databases(output_dir: Path) -> None:
    """Regenerate the shipped tier databases into ``output_dir``.

    Delegates to the canonical spec-driven generator, reading the committed
    tier-3 seeds and writing the grown tier-3 paradigms plus the derived
    tier-1 ``in_context`` subset under ``output_dir``.  The default
    ``output_dir`` is the in-tree preset location, for which this is a
    byte-identical in-place regeneration.
    """
    written = regenerate_tier_databases(output_dir)

    print("Generated databases:")
    for name, path in written.items():
        print(f"  {name}: {path}")


def run_validation(output_dir: Path) -> bool:
    """Run per-tier query validation and print results.

    Validates each published tier's queries (from the canonical in-tree
    ``QUERY_SOURCE_DIR``) against that tier's databases under ``output_dir``.

    Returns:
        True if all targeted PVs are found, False otherwise.
    """
    tier_queries = collect_tier_queries()

    if not tier_queries:
        print("No query files found, skipping validation")
        return True

    report = validate_queries(tier_queries=tier_queries, output_dir=output_dir)

    if report["valid"]:
        print(
            f"All {report['total_pvs']} PVs found "
            f"({report['total_queries']} queries across tiers {sorted(tier_queries)})"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate the cross-paradigm tier databases from the facility spec.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRESET_TIERS_DIR,
        help="Output directory (default: the in-tree control_assistant preset tiers/ location)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Regenerate databases, then validate that all targeted PVs exist",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate (skip regeneration)",
    )
    return parser.parse_args(argv)


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

    generate_databases(output_dir)

    if args.validate:
        if not run_validation(output_dir):
            sys.exit(1)


if __name__ == "__main__":
    main()
