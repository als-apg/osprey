#!/usr/bin/env python3
"""Annotate benchmark queries with a `tier` field.

For each benchmark dataset shipped in the control_assistant preset, classify
every query into the smallest tier (1, 2, or 3) whose channel database
contains all of the query's targeted PVs. Writes the `tier` field back into
the JSON files in place.

Usage:
    uv run python scripts/annotate_benchmark_tiers.py
    uv run python scripts/annotate_benchmark_tiers.py --check    # report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRESET = _PROJECT_ROOT / "src/osprey/templates/apps/control_assistant/data"
_TIERS_DIR = _PRESET / "channel_databases/tiers"
_BENCHMARKS_DIR = _PRESET / "benchmarks/datasets"


def _addresses_in_in_context(db: dict) -> set[str]:
    return {ch["address"] for ch in db.get("channels", [])}


def _channels_in_in_context(db: dict) -> set[str]:
    return {ch["channel"] for ch in db.get("channels", [])}


def _addresses_in_hierarchical(db: dict) -> set[str]:
    """Walk the hierarchical tree and collect every concrete PV string.

    PVs are encoded as ChannelNames lists at the leaves (subfield level) when
    the format_hierarchical writer emits them; in the source template they
    appear under `_expansion`. We accept both shapes by recursing and
    collecting any string-valued list under a `ChannelNames` key.
    """
    out: set[str] = set()

    def recurse(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "ChannelNames" and isinstance(v, list):
                    out.update(s for s in v if isinstance(s, str))
                else:
                    recurse(v)
        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(db)
    return out


def _addresses_in_middle_layer(db: dict) -> set[str]:
    """Middle-layer DBs nest ChannelNames lists inside ring/family/field/subfield."""
    return _addresses_in_hierarchical(db)


def load_tier_address_sets() -> dict[int, set[str]]:
    """Return {tier_num: set of canonical colon-form addresses}."""
    sets: dict[int, set[str]] = {}
    for tier_num in (1, 2, 3):
        ic = json.loads((_TIERS_DIR / f"tier{tier_num}/in_context.json").read_text())
        sets[tier_num] = _addresses_in_in_context(ic)
    return sets


def load_tier_channel_sets() -> dict[int, set[str]]:
    """Return {tier_num: set of Title_Case channel names}."""
    sets: dict[int, set[str]] = {}
    for tier_num in (1, 2, 3):
        ic = json.loads((_TIERS_DIR / f"tier{tier_num}/in_context.json").read_text())
        sets[tier_num] = _channels_in_in_context(ic)
    return sets


def classify(pvs: list[str], tier_sets: dict[int, set[str]]) -> int:
    """Return the smallest tier number whose set is a superset of pvs.

    Raises ValueError if no tier covers all PVs (indicates a bad query).
    """
    pv_set = set(pvs)
    for tier_num in (1, 2, 3):
        if pv_set.issubset(tier_sets[tier_num]):
            return tier_num
    missing = pv_set - tier_sets[3]
    raise ValueError(f"PVs not in any tier: {sorted(missing)[:5]}...")


def annotate_file(path: Path, tier_sets: dict[int, set[str]], *, write: bool) -> dict:
    """Annotate one benchmark file. Returns {tier: count} histogram."""
    queries = json.loads(path.read_text(encoding="utf-8"))
    histogram: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for q in queries:
        tier = classify(q["targeted_pv"], tier_sets)
        q["tier"] = tier
        histogram[tier] += 1
    if write:
        # Preserve key order: insert `tier` after `targeted_pv`. Easiest path
        # is to rewrite each dict in a stable order.
        ordered = []
        for q in queries:
            new_q = {}
            for key in ("user_query", "targeted_pv", "tier", "details"):
                if key in q:
                    new_q[key] = q[key]
            for key in q:
                if key not in new_q:
                    new_q[key] = q[key]
            ordered.append(new_q)
        path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return histogram


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only classify and report; do not write files.",
    )
    args = parser.parse_args(argv)

    address_sets = load_tier_address_sets()
    channel_sets = load_tier_channel_sets()

    targets = [
        ("hierarchical_benchmark.json", address_sets),
        ("middle_layer_benchmark.json", address_sets),
        ("in_context_benchmark.json", channel_sets),
    ]

    print(f"Annotating benchmarks in {_BENCHMARKS_DIR.relative_to(_PROJECT_ROOT)}")
    for filename, sets in targets:
        path = _BENCHMARKS_DIR / filename
        try:
            hist = annotate_file(path, sets, write=not args.check)
        except ValueError as exc:
            print(f"  FAIL {filename}: {exc}", file=sys.stderr)
            sys.exit(1)
        action = "checked" if args.check else "annotated"
        print(
            f"  {filename}: {action} "
            f"(tier1={hist[1]}, tier2={hist[2]}, tier3={hist[3]}, total={sum(hist.values())})"
        )


if __name__ == "__main__":
    main()
