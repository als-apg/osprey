"""Utility to expand srcorr.json into a flat list of PV names.

Examples:
  # All families, all sectors, all suffixes (base PVs excluded by default)
  python gen_srcorr_pvs.py

  # Only SR correctors in sectors 1 and 5–7, keep base PVs
  python gen_srcorr_pvs.py --types corrector --sectors 1,5-7 --include-base

  # Only setpoint/readback suffixes for fast correctors, no base PV
  python gen_srcorr_pvs.py --types fast --suffixes setpoint \"current readback\"

  # Quadrupoles without sector zero padding
  python gen_srcorr_pvs.py --types quadrupole --no-pad2
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

SECTOR_MIN = 1
SECTOR_MAX = 40  # change if needed

SECTION_MAP = {
    "fast": ("SR Fast Correctors", "Sector ${sector} fast correctors"),
    "corrector": ("SR Correctors", "Sector ${sector} correctors"),
    "skew": ("SR Skew Quadrupoles", "Sector ${sector} skew quadrupoles"),
    "quadrupole": ("SR Quadrupoles", "Sector ${sector} quadrupoles"),
    "sextupole": ("SR Sextupoles", "Sector ${sector} sextupoles"),
    "dipole": ("SR Dipoles", "Sector ${sector} dipoles"),
    "dipole_trim": ("SR Dipole Trim", "Sector ${sector} dipole trim"),
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _norm(text: str) -> str:
    return text.replace(" ", "").replace("_", "").lower()


def parse_sectors(raw: str | None) -> List[int]:
    """Parse comma-separated list of sectors with optional ranges (e.g., 1,5,10-12)."""
    if not raw:
        return list(range(SECTOR_MIN, SECTOR_MAX + 1))

    sectors: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            step = 1 if start <= end else -1
            sectors.extend(range(start, end + step, step))
        else:
            sectors.append(int(part))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sectors:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def filter_suffixes(suffixes: dict, wanted: Sequence[str] | None) -> List[str]:
    """Return suffix values filtered by the provided labels (case/space-insensitive)."""
    if not wanted:
        return list(suffixes.values())
    wanted_norm = {_norm(w) for w in wanted}
    return [val for name, val in suffixes.items() if _norm(name) in wanted_norm]


def device_matches(device: dict, patterns: Sequence[str] | None) -> bool:
    """Check if a device matches any provided patterns against channel/tags."""
    if not patterns:
        return True
    candidates = [_norm(device.get("channel", ""))]
    for tag in device.get("tags", []) or []:
        candidates.append(_norm(tag))
    for pattern in patterns:
        p = _norm(pattern)
        for cand in candidates:
            if p == cand or p in cand or cand in p:
                return True
    return False


def generate_section(
    data: dict,
    section_key: str,
    devices_key: str,
    sectors: Iterable[int],
    *,
    include_base: bool,
    pad2: bool,
    suffix_labels: Sequence[str] | None,
    device_filters: Sequence[str] | None,
) -> List[str]:
    """Generate PVs for a given section in the JSON."""

    section = data.get(section_key, {})
    suffixes = filter_suffixes(section.get("suffix", {}), suffix_labels)
    devices = section.get(devices_key, [])

    pvs: List[str] = []

    for sector in sectors:
        sec_str = f"{sector:02d}" if pad2 else str(sector)

        for device in devices:
            if not device_matches(device, device_filters):
                continue

            allowed_sectors = device.get("sectors")
            if allowed_sectors is not None and sector not in allowed_sectors:
                continue

            base_pv = device.get("address_template", "").replace("${sector}", sec_str)
            if not base_pv:
                continue

            if include_base:
                pvs.append(base_pv)

            for suffix_val in suffixes:
                pvs.append(base_pv + suffix_val)

    return pvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PV names from srcorr.json.")
    parser.add_argument(
        "-d",
        "--database",
        default="srcorr.json",
        help="Path to srcorr.json (default: srcorr.json).",
    )
    parser.add_argument(
        "--sectors",
        help="Comma-separated list or ranges, e.g. '1,5,10-12'. Defaults to all sectors.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=SECTION_MAP.keys(),
        default=list(SECTION_MAP.keys()),
        help="Which device families to generate: fast, corrector, quadrupole. Defaults to all.",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        help="Suffix labels to include (case-insensitive). Defaults to all suffixes.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Filter devices by channel/tags (case-insensitive, partial match).",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        dest="include_base",
        help="Also emit the bare device PV (without suffix). Defaults to off.",
    )
    parser.add_argument(
        "--pad2",
        action="store_true",
        default=True,
        help="Pad sector numbers to 2 digits (01–40). Enabled by default.",
    )
    parser.add_argument(
        "--no-pad2",
        action="store_false",
        dest="pad2",
        help="Do not pad sector numbers (1–40).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Generate PVs
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    db_path = Path(args.database)
    with db_path.open("r") as f:
        data = json.load(f)

    sectors = parse_sectors(args.sectors)

    all_pvs: List[str] = []
    for type_key in args.types:
        section_key, devices_key = SECTION_MAP[type_key]
        all_pvs.extend(
            generate_section(
                data,
                section_key,
                devices_key,
                sectors,
                include_base=args.include_base,
                pad2=args.pad2,
                suffix_labels=args.suffixes,
                device_filters=args.devices,
            )
        )

    print(f"Generated {len(all_pvs)} PV names.\n")
    for pv in all_pvs:
        print(pv)


if __name__ == "__main__":
    main()
