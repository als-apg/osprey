#!/usr/bin/env python3
"""Freeze the demo's served facts as goldens, and check a golden against them.

Every mode recomputes one golden from the committed demo sources and either
writes it (``--write PATH``) or byte-compares it with a committed copy
(``--check PATH``, exit 1 on any difference). Output is deterministic: rows
sorted by address, UTF-8 JSON with two-space indentation and a final newline.
Each golden carries a ``_reproduce`` line naming the command that writes it.

Modes:

* default -- the demo fingerprint: one row per address of the demo TTL with
  its ``role`` (``setpoint`` where the binding ``writesSignal``, ``readback``
  where it ``readsSignal``), ``value_type`` (derived from the hierarchical
  database path), ``names`` (the tier-3 in_context ``channel`` strings for the
  address, source order) and ``description`` (the binding's own). The rows'
  sha256 is pinned in the golden.
* ``--standalone-addresses`` -- the expanded address set of the
  channel-finder-standalone preset's hierarchical database.
* ``--in-context-size`` -- the tier-1 in_context database's size and its
  (channel, address) rows.
* ``--cf-index-pre-line`` -- PATH is a directory: byte copies of the three
  tier-3 channel-finder indexes plus ``MANIFEST.json`` naming each copy's
  source and sha256.

Run it with the project interpreter (``uv run python``): parsing the TTL needs
``rdflib``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/facility_demo/fingerprint.py"

_APPS = "src/osprey/templates/apps"
_CA_DATA = f"{_APPS}/control_assistant/data"
DEMO_TTL = f"{_CA_DATA}/demo_machine.ttl"
TIER3_DIR = f"{_CA_DATA}/channel_databases/tiers/tier3"
TIER3_IN_CONTEXT = f"{TIER3_DIR}/in_context.json"
TIER3_HIERARCHICAL = f"{TIER3_DIR}/hierarchical.json"
TIER1_IN_CONTEXT = f"{_CA_DATA}/channel_databases/tiers/tier1/in_context.json"
#: The channel-finder pipelines that read a tier file; ``graph`` has none.
CF_INDEX_FILES = ("hierarchical.json", "in_context.json", "middle_layer.json")
STANDALONE_HIERARCHICAL = (
    f"{_APPS}/channel_finder_standalone/data/channel_databases/hierarchical.json"
)

GOLDEN_DIR = "tests/facility/golden"

_NARAD_PROPERTY = "https://narad.example.org/property/"

#: The fingerprint's word for each record type the hierarchical path derives.
_VALUE_TYPE_BY_RECORD_TYPE = {"ai": "float", "bi": "bool"}


class FingerprintError(RuntimeError):
    """The demo sources disagree with each other; the golden cannot be computed."""


def _rel(path: str) -> Path:
    return REPO_ROOT / path


def _dump(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    """The sha256 of ``rows`` as compact UTF-8 JSON, key order as written."""
    compact = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _reproduce(flag: str, golden: str) -> str:
    flags = f" {flag}" if flag else ""
    return f"uv run python {SCRIPT}{flags} --write {golden}"


def _ttl_bindings() -> dict[str, dict[str, str]]:
    """Address -> ``{role, description}`` for every binding of the demo TTL."""
    import rdflib

    prop = rdflib.Namespace(_NARAD_PROPERTY)
    graph = rdflib.Graph()
    graph.parse(_rel(DEMO_TTL), format="turtle")
    bindings: dict[str, dict[str, str]] = {}
    for subject, address in graph.subject_objects(prop.fullPv):
        writes = (subject, prop.writesSignal, None) in graph
        reads = (subject, prop.readsSignal, None) in graph
        if writes == reads:
            raise FingerprintError(
                f"{address}: a binding must either read or write its signal, not "
                f"{'both' if writes else 'neither'}"
            )
        if str(address) in bindings:
            raise FingerprintError(f"{address}: bound twice in {DEMO_TTL}")
        description = graph.value(subject, prop.description)
        bindings[str(address)] = {
            "role": "setpoint" if writes else "readback",
            "description": "" if description is None else str(description),
        }
    return bindings


def _in_context_names() -> dict[str, list[str]]:
    """Address -> the tier-3 in_context ``channel`` strings naming it, source order."""
    document = json.loads(_rel(TIER3_IN_CONTEXT).read_text(encoding="utf-8"))
    names: dict[str, list[str]] = {}
    for row in document["channels"]:
        names.setdefault(row["address"], []).append(row["channel"])
    return names


def _value_types() -> dict[str, str]:
    """Address -> value_type, derived from each address's hierarchical path."""
    from osprey.services.virtual_accelerator.manifest.classify import derive_record_type
    from osprey.services.virtual_accelerator.manifest.loaders import (
        load_hierarchical_channels,
    )

    value_types: dict[str, str] = {}
    for channel in load_hierarchical_channels():
        record_type, _noise = derive_record_type(channel.path)
        if record_type not in _VALUE_TYPE_BY_RECORD_TYPE:
            raise FingerprintError(f"{channel.address}: no value_type for {record_type!r}")
        value_types[channel.address] = _VALUE_TYPE_BY_RECORD_TYPE[record_type]
    return value_types


def _require_same_addresses(expected: set[str], label: str, actual: set[str]) -> None:
    if expected != actual:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise FingerprintError(
            f"{label} does not cover the TTL's addresses: missing {missing}, extra {extra}"
        )


def demo_fingerprint() -> bytes:
    """The demo fingerprint golden."""
    bindings = _ttl_bindings()
    names = _in_context_names()
    value_types = _value_types()
    _require_same_addresses(set(bindings), TIER3_IN_CONTEXT, set(names))
    _require_same_addresses(set(bindings), TIER3_HIERARCHICAL, set(value_types))
    rows = [
        {
            "address": address,
            "role": bindings[address]["role"],
            "value_type": value_types[address],
            "names": names[address],
            "description": bindings[address]["description"],
        }
        for address in sorted(bindings)
    ]
    roles: dict[str, int] = {}
    for row in rows:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
    return _dump(
        {
            "_reproduce": _reproduce("", f"{GOLDEN_DIR}/demo_fingerprint.json"),
            "_sources": [DEMO_TTL, TIER3_IN_CONTEXT, TIER3_HIERARCHICAL],
            "_sha256_of": "rows as compact UTF-8 JSON, key order as written",
            "count": len(rows),
            "roles": dict(sorted(roles.items())),
            "sha256": rows_sha256(rows),
            "rows": rows,
        }
    )


def standalone_addresses() -> bytes:
    """The channel-finder-standalone address-set golden."""
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    database = HierarchicalChannelDatabase(str(_rel(STANDALONE_HIERARCHICAL)))
    database.load_database()
    addresses = sorted({channel["address"] for channel in database.get_all_channels()})
    return _dump(
        {
            "_reproduce": _reproduce(
                "--standalone-addresses", f"{GOLDEN_DIR}/cf_standalone_addresses.json"
            ),
            "_sources": [STANDALONE_HIERARCHICAL],
            "count": len(addresses),
            "addresses": addresses,
        }
    )


def in_context_size() -> bytes:
    """The in_context size golden over the tier-1 database."""
    document = json.loads(_rel(TIER1_IN_CONTEXT).read_text(encoding="utf-8"))
    rows = document["channels"]
    declared = document.get("_metadata", {}).get("total_channels")
    if declared != len(rows):
        raise FingerprintError(
            f"{TIER1_IN_CONTEXT}: _metadata.total_channels {declared} != {len(rows)} rows"
        )
    return _dump(
        {
            "_reproduce": _reproduce("--in-context-size", f"{GOLDEN_DIR}/in_context_size.json"),
            "_sources": [TIER1_IN_CONTEXT],
            "size": len(rows),
            "rows": sorted(
                ({"address": row["address"], "channel": row["channel"]} for row in rows),
                key=lambda row: (row["address"], row["channel"]),
            ),
        }
    )


CF_INDEX_MANIFEST = "MANIFEST.json"


def cf_index_pre_line() -> dict[str, bytes]:
    """File name -> bytes for the pre-LINE channel-finder index goldens."""
    files = {name: _rel(f"{TIER3_DIR}/{name}").read_bytes() for name in CF_INDEX_FILES}
    manifest = {
        "_reproduce": _reproduce("--cf-index-pre-line", f"{GOLDEN_DIR}/cf_index_pre_line"),
        "files": {
            name: {
                "source": f"{TIER3_DIR}/{name}",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(files.items())
        },
    }
    return {**files, CF_INDEX_MANIFEST: _dump(manifest)}


_MODES: dict[str, Callable[[], bytes]] = {
    "fingerprint": demo_fingerprint,
    "standalone_addresses": standalone_addresses,
    "in_context_size": in_context_size,
}

#: Modes whose golden is a directory of files rather than one file.
_DIRECTORY_MODES: dict[str, Callable[[], dict[str, bytes]]] = {
    "cf_index_pre_line": cf_index_pre_line,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--standalone-addresses",
        dest="mode",
        action="store_const",
        const="standalone_addresses",
        help="the channel-finder-standalone address set",
    )
    mode.add_argument(
        "--in-context-size",
        dest="mode",
        action="store_const",
        const="in_context_size",
        help="the tier-1 in_context size and rows",
    )
    mode.add_argument(
        "--cf-index-pre-line",
        dest="mode",
        action="store_const",
        const="cf_index_pre_line",
        help="the tier-3 channel-finder indexes; PATH is a directory",
    )
    parser.set_defaults(mode="fingerprint")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--check", type=Path, metavar="PATH", help="byte-compare with PATH")
    target.add_argument("--write", type=Path, metavar="PATH", help="write the golden to PATH")
    return parser.parse_args(argv)


def _compute(mode: str) -> dict[str | None, bytes]:
    """Relative file name -> bytes; the key ``None`` is PATH itself."""
    if mode in _DIRECTORY_MODES:
        return dict(_DIRECTORY_MODES[mode]().items())
    return {None: _MODES[mode]()}


def _target(path: Path, name: str | None) -> Path:
    return path if name is None else path / name


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        golden = _compute(args.mode)
    except FingerprintError as exc:
        print(f"fingerprint: {exc}", file=sys.stderr)
        return 1
    if args.write is not None:
        for name, content in golden.items():
            target = _target(args.write, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return 0
    status = 0
    if None not in golden and args.check.is_dir():
        unexpected = sorted(
            entry.name for entry in args.check.iterdir() if entry.name not in golden
        )
        for name in unexpected:
            print(f"fingerprint: {args.check / name} is not part of the golden", file=sys.stderr)
            status = 1
    for name, content in golden.items():
        target = _target(args.check, name)
        if not target.is_file():
            print(f"fingerprint: {target} does not exist", file=sys.stderr)
            status = 1
        elif target.read_bytes() != content:
            print(f"fingerprint: {target} differs from the recomputed golden", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
