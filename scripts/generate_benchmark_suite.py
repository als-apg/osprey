#!/usr/bin/env python3
"""Regenerate the shipped tier channel databases via the canonical spec pipeline.

Thin wrapper over
:func:`osprey.services.channel_finder.tools.generate_from_spec.generate` -- the
single canonical regeneration of every committed channel-database artifact. That
pipeline grows the tier-3 cross-paradigm views (``in_context``, ``hierarchical``,
``middle_layer``) from the facility spec and derives tier-1's ``in_context``
subset from them. This script adds a query-integrity ``--validate`` pass over the
shipped per-tier query sets.

Tier 2 is retired: only tiers 1 and 3 are published. Tier 1 ships the
``in_context`` paradigm only; tier 3 ships all three. The published tiers (and
the paradigms each ships) come from
:data:`osprey.services.channel_finder.benchmarks.generator.TIER_PARADIGMS`.

Regeneration is merge-preserve and idempotent: regenerating in place (the
default) rewrites the committed artifacts byte-for-byte.

Usage:
    python scripts/generate_benchmark_suite.py
    python scripts/generate_benchmark_suite.py --output-dir /tmp/benchmarks
    python scripts/generate_benchmark_suite.py --validate
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


def generate_suite(output_dir: Path) -> None:
    """Regenerate the shipped tier databases into ``output_dir``.

    Delegates to the canonical spec-driven generator, reading the committed
    tier-3 seeds and writing the grown tier-3 paradigms plus the derived
    tier-1 ``in_context`` subset under ``output_dir``.  The default
    ``output_dir`` is the in-tree preset location, for which this is a
    byte-identical in-place regeneration.
    """
    written = regenerate_tier_databases(output_dir)
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(f"\nSuite generated in {output_dir}/")


def run_validation(output_dir: Path) -> bool:
    """Run per-tier validation against generated databases.

    Reads queries from the canonical in-tree location (``QUERY_SOURCE_DIR``)
    rather than from a subdir of ``output_dir``.  ``output_dir`` is the
    tier-DB tree being validated.  Only the published tiers (the keys of
    :data:`TIER_PARADIGMS`) are validated.
    """
    print("\nValidating query sets against databases...")

    tier_queries = collect_tier_queries()

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
        description="Regenerate the shipped tier databases (canonical spec pipeline) + validate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PRESET_TIERS_DIR,
        help=(
            "Output directory for the regenerated tier DBs "
            "(default: the in-tree control_assistant preset tiers/ location)"
        ),
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
