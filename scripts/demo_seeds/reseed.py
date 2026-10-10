#!/usr/bin/env python3
"""Stamp the example facility's simulated motion from the per-signal rule table.

Every float readback of ``src/osprey/templates/facilities/example`` that carries
no ``linear`` key gets its ``noise`` and ``drift`` from the one row of
``rules.yaml`` that matches its device's class and its signal, converted into the
channel's unit; ``noise`` is written as ``{absolute: <sigma>}``. A channel no row matches stays still, unless its committed seed
moves, which is an error. Every other seed key (``nominal``, ``clamp``,
``linear``) is carried through as data.

The output is the committed ``seeds.yaml``: addresses sorted, each entry's keys
in the order nominal, noise, drift, clamp, linear, written with one canonical
``yaml.safe_dump``.

Usage::

    uv run python scripts/demo_seeds/reseed.py           # rewrite seeds.yaml
    uv run python scripts/demo_seeds/reseed.py --check   # exit 1 if it differs
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from osprey.facility.sources import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE = REPO_ROOT / "src/osprey/templates/facilities/example"
RULES = Path(__file__).resolve().parent / "rules.yaml"

#: Each unit a rule or a channel may carry: its dimension and its power of ten
#: against the dimension's base unit.
UNITS: dict[str, tuple[str, int]] = {
    "m": ("length", 0),
    "cm": ("length", -2),
    "mm": ("length", -3),
    "µm": ("length", -6),
    "A": ("current", 0),
    "mA": ("current", -3),
    "MHz": ("frequency", 6),
    "Hz": ("frequency", 0),
    "V": ("voltage", 0),
    "W": ("power", 0),
    "Pa": ("pressure", 0),
    "C": ("temperature", 0),
    "mrem/hr": ("dose rate", 0),
    "s": ("time", 0),
}

#: The seed keys, in the order an entry is written.
KEY_ORDER = ("nominal", "noise", "drift", "clamp", "linear")

#: The seed keys this generator owns.
MOTION_KEYS = ("noise", "drift")

#: A rule's keys.
RULE_KEYS = frozenset({"class", "signal", "machine", "device", "noise", "drift", "source"})

#: The literal a rule writes for a quantity that is absent.
NONE = "none"


class ReseedError(Exception):
    """The rule table cannot produce a seed for a channel."""


def load(path: Path) -> Any:
    """One YAML file as the facility loader parses it."""
    return read_yaml(path.read_text(encoding="utf-8"))


def load_rules(path: Path = RULES) -> list[dict[str, Any]]:
    """The rule table's rows, each checked for its keys."""
    rows: list[dict[str, Any]] = load(path)
    for row in rows:
        unknown = set(row) - RULE_KEYS
        if unknown:
            raise ReseedError(f"rule {row} has unknown keys {sorted(unknown)}")
        for key in ("class", "signal", "noise", "source"):
            if key not in row:
                raise ReseedError(f"rule {row} has no {key!r}")
    return rows


def convert(quantity: Any, unit: str) -> float | None:
    """A rule's ``{value, unit}`` in ``unit``; ``None`` for ``none``.

    The conversion is a decimal shift, so a value written in the channel's own
    unit comes back as the same float.
    """
    if quantity == NONE:
        return None
    source = quantity["unit"]
    if source not in UNITS or unit not in UNITS:
        raise ReseedError(f"no conversion from {source!r} to {unit!r}")
    (dimension, exponent), (target, target_exponent) = UNITS[source], UNITS[unit]
    if dimension != target:
        raise ReseedError(f"no conversion from {source!r} to {unit!r}")
    return float(Decimal(str(quantity["value"])).scaleb(exponent - target_exponent))


def specificity(row: dict[str, Any]) -> int:
    """How narrowly a row matches: by device, by machine, or by neither."""
    if "device" in row:
        return 2
    return 1 if "machine" in row else 0


def match(
    rules: list[dict[str, Any]], *, cls: str, signal: str, device: str
) -> dict[str, Any] | None:
    """The one row that matches a channel, or ``None`` when no row does.

    Args:
        rules: The rule table's rows.
        cls: The class of the channel's device.
        signal: The channel's signal.
        device: The channel's device id, ``<machine>/<name>``.

    Raises:
        ReseedError: Two rows match at the same specificity.
    """
    machine = device.split("/")[0]
    candidates = [
        row
        for row in rules
        if row["class"] == cls
        and row["signal"] == signal
        and row.get("machine", machine) == machine
        and row.get("device", device) == device
    ]
    if not candidates:
        return None
    best = max(specificity(row) for row in candidates)
    winners = [row for row in candidates if specificity(row) == best]
    if len(winners) > 1:
        raise ReseedError(f"{device} {signal}: {len(winners)} rules match")
    return winners[0]


def motion(row: dict[str, Any], unit: str) -> dict[str, Any]:
    """A row's ``noise`` and ``drift`` in the channel's ``unit``."""
    stamped: dict[str, Any] = {}
    noise = convert(row["noise"], unit)
    if noise is not None:
        stamped["noise"] = {"absolute": noise}
    drift = row.get("drift", NONE)
    if drift != NONE:
        stamped["drift"] = {
            "amplitude": convert(drift["amplitude"], unit),
            "period_s": convert(drift["period_s"], "s"),
        }
    return stamped


def moves(channel: dict[str, Any], seed: dict[str, Any]) -> bool:
    """A channel this generator stamps motion on: a float readback with no ``linear``."""
    return (
        channel.get("role", "readback") == "readback"
        and channel.get("value_type", "float") == "float"
        and "linear" not in seed
    )


def reseed(tree: Path = TREE, rules_path: Path = RULES) -> dict[str, dict[str, Any]]:
    """The seeds document with every channel's motion stamped from the rule table."""
    rules = load_rules(rules_path)
    channels = {record["id"]: record for record in load(tree / "records/channels.yaml")}
    devices = {record["id"]: record for record in load(tree / "records/devices.yaml")}
    committed: dict[str, dict[str, Any]] = load(tree / "seeds.yaml")
    unknown = sorted(set(committed) - set(channels))
    if unknown:
        raise ReseedError(f"seeds for addresses with no channel record: {unknown[:5]}")

    seeds: dict[str, dict[str, Any]] = {}
    for address in sorted(channels):
        channel, seed = channels[address], committed.get(address, {})
        extra = set(seed) - set(KEY_ORDER)
        if extra:
            raise ReseedError(f"{address}: unknown seed keys {sorted(extra)}")
        moved = any(key in seed for key in MOTION_KEYS)
        entry = {key: value for key, value in seed.items() if key not in MOTION_KEYS}
        if moves(channel, seed):
            device, signal = channel.get("on", {}).get("device"), channel.get("signal")
            row = None
            if device is not None and signal is not None:
                row = match(rules, cls=devices[device]["class"], signal=signal, device=device)
            if row is not None:
                entry.update(motion(row, channel.get("unit", "")))
            elif moved:
                raise ReseedError(f"{address}: moves today and no rule matches it")
        elif moved:
            raise ReseedError(f"{address}: carries motion but is not a float readback")
        if entry:
            seeds[address] = {key: entry[key] for key in KEY_ORDER if key in entry}
    return seeds


def render(tree: Path = TREE, rules_path: Path = RULES) -> str:
    """The text of the generated ``seeds.yaml``."""
    return yaml.safe_dump(reseed(tree, rules_path), sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the committed seeds.yaml differs"
    )
    args = parser.parse_args(argv)
    target = TREE / "seeds.yaml"
    text = render()
    if args.check:
        if target.read_text(encoding="utf-8") != text:
            print(f"{target.relative_to(REPO_ROOT)} differs; run scripts/demo_seeds/reseed.py")
            return 1
        print(f"{target.relative_to(REPO_ROOT)} matches")
        return 0
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
