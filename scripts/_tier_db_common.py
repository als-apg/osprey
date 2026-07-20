"""Shared plumbing for the tier channel-database regeneration scripts.

Both :mod:`scripts.generate_tier_databases` and
:mod:`scripts.generate_benchmark_suite` are thin CLIs over the same canonical
spec pipeline
(:func:`osprey.services.channel_finder.tools.generate_from_spec.generate`).
The paths, the tier→query-file mapping, and the single generator call live here
so that rule is defined once; each script keeps only its own argument parsing
and console output.

Tier 2 is retired: only tiers 1 and 3 are published. Tier 1 ships the
``in_context`` paradigm only; tier 3 ships all three. The published tiers (and
the paradigms each ships) come from
:data:`osprey.services.channel_finder.benchmarks.generator.TIER_PARADIGMS`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Add src to path for direct script execution.
sys.path.insert(0, str(_REPO_ROOT / "src"))

from osprey.services.channel_finder.benchmarks.generator import (  # noqa: E402
    TIER_PARADIGMS,
    validate_queries,
)
from osprey.services.channel_finder.tools.generate_from_spec import generate  # noqa: E402

__all__ = [
    "PRESET_TIERS_DIR",
    "QUERY_SOURCE_DIR",
    "TIER_PARADIGMS",
    "collect_tier_queries",
    "regenerate_tier_databases",
    "validate_queries",
]

PRESET_TIERS_DIR = (
    _REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "channel_databases"
    / "tiers"
)
QUERY_SOURCE_DIR = (
    _REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "benchmarks"
    / "cross_paradigm"
    / "queries"
)


def regenerate_tier_databases(output_dir: Path) -> dict[str, Path]:
    """Regenerate the shipped tier databases into ``output_dir``.

    Delegates to the canonical spec-driven generator, reading the committed
    tier-3 seeds and writing the grown tier-3 paradigms plus the derived
    tier-1 ``in_context`` subset under ``output_dir``.  When ``output_dir`` is
    the in-tree preset location (the scripts' default) this is a byte-identical
    in-place regeneration.

    Returns:
        Mapping of artifact name to the path written.
    """
    return generate(source_dir=PRESET_TIERS_DIR / "tier3", dest_dir=output_dir / "tier3")


def collect_tier_queries() -> dict[int, Path]:
    """Map each published tier to its committed query file (present ones only).

    Reads from the canonical in-tree :data:`QUERY_SOURCE_DIR`, iterating the
    published tiers in :data:`TIER_PARADIGMS`.
    """
    tier_queries: dict[int, Path] = {}
    for tier_num in TIER_PARADIGMS:
        path = QUERY_SOURCE_DIR / f"tier{tier_num}_queries.json"
        if path.exists():
            tier_queries[tier_num] = path
    return tier_queries
