#!/usr/bin/env python3
"""Update channel_limits.json with DRVH/DRVL limits from live EPICS PVs.

Reads .DRVH (drive high) and .DRVL (drive low) fields from EPICS setpoint
PVs and merges them into the channel limits database. Manual overrides
(max_step, verification, writable) are preserved.

Merge priority (safety-first):
    final min_value = max(epics_DRVL, manual_min)   # tighter lower bound
    final max_value = min(epics_DRVH, manual_max)   # tighter upper bound

Usage:
    python scripts/update_channel_limits.py                  # update limits
    python scripts/update_channel_limits.py --dry-run        # preview changes
    python scripts/update_channel_limits.py --timeout 5      # slower network
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import epics
except ImportError:
    print("ERROR: pyepics is required. Install with: pip install pyepics")
    sys.exit(1)


# Suffixes that indicate writable setpoint PVs (candidates for DRVH/DRVL)
WRITABLE_SUFFIXES = ("AO", "AO.VAL")


def discover_setpoint_pvs(pv_database_path: Path) -> list[str]:
    """Extract writable setpoint PV addresses from the ITS PV database."""
    with open(pv_database_path) as f:
        db = json.load(f)

    addresses = []
    for entry in db.get("channels", []):
        addr = entry.get("address", "")
        if not addr:
            continue
        # Keep only analog output (setpoint) PVs
        if any(addr.endswith(suffix) for suffix in WRITABLE_SUFFIXES):
            addresses.append(addr)

    # Deduplicate and sort
    return sorted(set(addresses))


def read_epics_limits(pvs: list[str], timeout: float) -> dict:
    """Read .DRVH and .DRVL for each PV from EPICS.

    Returns dict: {pv: {"drvh": float, "drvl": float}} for successful reads.
    Skips PVs where DRVH == DRVL == 0 (limits not configured in IOC).
    """
    results = {}
    errors = []

    for pv in pvs:
        pv_drvh = f"{pv}.DRVH"
        pv_drvl = f"{pv}.DRVL"

        drvh = epics.caget(pv_drvh, timeout=timeout)
        drvl = epics.caget(pv_drvl, timeout=timeout)

        if drvh is None or drvl is None:
            errors.append(pv)
            continue

        # Skip if both are 0 — limits not configured in IOC
        if drvh == 0.0 and drvl == 0.0:
            continue

        results[pv] = {"drvh": float(drvh), "drvl": float(drvl)}

    return results, errors


def load_limits_file(path: Path) -> dict:
    """Load existing channel_limits.json."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def merge_limits(existing: dict, epics_limits: dict) -> tuple[dict, dict]:
    """Merge EPICS DRVH/DRVL into existing limits, preserving manual overrides.

    Returns (merged_limits, change_log).
    """
    merged = dict(existing)  # shallow copy of top-level
    changes = {"updated": [], "added": [], "skipped": []}

    for pv, lim in epics_limits.items():
        epics_min = lim["drvl"]
        epics_max = lim["drvh"]

        if pv in merged:
            # Existing entry — merge with safety-first priority
            entry = dict(merged[pv])  # copy to avoid mutating original
            old_min = entry.get("min_value")
            old_max = entry.get("max_value")

            # Tighter bound: max of lower limits, min of upper limits
            new_min = max(epics_min, old_min) if old_min is not None else epics_min
            new_max = min(epics_max, old_max) if old_max is not None else epics_max

            changed = False
            if old_min != new_min:
                entry["min_value"] = new_min
                changed = True
            if old_max != new_max:
                entry["max_value"] = new_max
                changed = True

            if changed:
                # Tag the source
                entry["_epics_drvl"] = epics_min
                entry["_epics_drvh"] = epics_max
                merged[pv] = entry
                changes["updated"].append({
                    "pv": pv,
                    "old_min": old_min, "old_max": old_max,
                    "epics_min": epics_min, "epics_max": epics_max,
                    "final_min": new_min, "final_max": new_max,
                })
            else:
                changes["skipped"].append(pv)
        else:
            # New channel — add with EPICS limits
            merged[pv] = {
                "_comment": f"Auto-populated from EPICS .DRVH/.DRVL",
                "min_value": epics_min,
                "max_value": epics_max,
                "_epics_drvl": epics_min,
                "_epics_drvh": epics_max,
            }
            changes["added"].append({
                "pv": pv,
                "min": epics_min,
                "max": epics_max,
            })

    return merged, changes


def print_summary(changes: dict, errors: list[str]):
    """Print a human-readable summary of changes."""
    print("\n" + "=" * 60)
    print("Channel Limits Update Summary")
    print("=" * 60)

    if changes["updated"]:
        print(f"\nUpdated ({len(changes['updated'])} channels):")
        for c in changes["updated"]:
            print(f"  {c['pv']}")
            print(f"    min: {c['old_min']} -> {c['final_min']}  (EPICS DRVL: {c['epics_min']})")
            print(f"    max: {c['old_max']} -> {c['final_max']}  (EPICS DRVH: {c['epics_max']})")

    if changes["added"]:
        print(f"\nAdded ({len(changes['added'])} new channels):")
        for c in changes["added"]:
            print(f"  {c['pv']}  [{c['min']}, {c['max']}]")

    if changes["skipped"]:
        print(f"\nUnchanged ({len(changes['skipped'])} channels):")
        for pv in changes["skipped"]:
            print(f"  {pv}")

    if errors:
        print(f"\nEPICS errors ({len(errors)} channels — could not read DRVH/DRVL):")
        for pv in errors:
            print(f"  {pv}")

    total = len(changes["updated"]) + len(changes["added"])
    print(f"\nTotal: {total} changes, {len(changes['skipped'])} unchanged, {len(errors)} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Update channel_limits.json from EPICS .DRVH/.DRVL fields"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing"
    )
    parser.add_argument(
        "--pv-database",
        default="data/channel_databases/its_pv_in_context.json",
        help="Path to PV database (default: data/channel_databases/its_pv_in_context.json)"
    )
    parser.add_argument(
        "--output",
        default="data/channel_limits.json",
        help="Path to channel_limits.json (default: data/channel_limits.json)"
    )
    parser.add_argument(
        "--timeout", type=float, default=3.0,
        help="EPICS caget timeout in seconds (default: 3.0)"
    )
    args = parser.parse_args()

    pv_db_path = Path(args.pv_database)
    limits_path = Path(args.output)

    # 1. Discover setpoint PVs
    print(f"Reading PV database: {pv_db_path}")
    pvs = discover_setpoint_pvs(pv_db_path)
    print(f"Found {len(pvs)} setpoint PVs")

    if not pvs:
        print("No writable setpoint PVs found. Nothing to do.")
        return

    # 2. Read EPICS limits
    print(f"Reading .DRVH/.DRVL from EPICS (timeout={args.timeout}s)...")
    epics_limits, errors = read_epics_limits(pvs, args.timeout)
    print(f"Got limits for {len(epics_limits)} PVs, {len(errors)} errors")

    if not epics_limits:
        print("No EPICS limits retrieved. Nothing to merge.")
        if errors:
            print("PVs that failed:")
            for pv in errors:
                print(f"  {pv}")
        return

    # 3. Load existing limits
    existing = load_limits_file(limits_path)

    # 4. Merge
    merged, changes = merge_limits(existing, epics_limits)

    # 5. Print summary
    print_summary(changes, errors)

    # 6. Write (unless dry-run)
    if args.dry_run:
        print("\n--dry-run: No files written.")
    else:
        # Backup
        if limits_path.exists():
            backup = limits_path.with_suffix(".json.bak")
            shutil.copy2(limits_path, backup)
            print(f"\nBackup saved: {backup}")

        with open(limits_path, "w") as f:
            json.dump(merged, f, indent=2)
            f.write("\n")
        print(f"Written: {limits_path}")


if __name__ == "__main__":
    main()
