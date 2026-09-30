#!/usr/bin/env python3
"""Write the demo's ``data/facility/`` sources from its hand-written sources.

Run on demand; the written tree is committed and edited as authored sources
from then on. It writes, under ``--out``:

* ``identity.yaml`` -- ``code: ca``; ``--standalone`` adds the facility name;
* ``classes.yaml`` -- empty: every demo class is a vocabulary class;
* ``models.yaml`` -- the deck machine's model on the committed deck and its
  wiring, sorted by address;
* ``seeds.yaml`` -- how each channel no model wires starts and moves, and
  how each wired readback moves, from the machine file and the channel
  taxonomy;
* ``records/places.yaml`` -- the machines and the deck machine's sectors;
* ``records/devices.yaml`` -- every device, a hand place only where no model
  wires it;
* ``records/channels.yaml`` -- every channel of the demo TTL plus the
  fingerprint additions (the deck machine's tune and chromaticity readbacks);
* ``records/groups.yaml`` -- one group per machine family and per machine
  system;
* ``limits.yaml`` -- three records, each teaching one limits shape;
* ``measurement/SR.yaml`` -- the deck machine's measurement kinds, groups,
  instruments and pyAML step and settle keys.

Output is deterministic: records sorted by id (places in tree order), UTF-8
YAML, one scalar per line. Run it with the project interpreter
(``uv run python``): parsing the TTL needs ``rdflib``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_records() -> Any:
    """``_records.py`` beside this file, under a name no other module takes."""
    name = "facility_demo__records"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("_records.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_records = _load_records()


class _Dumper(yaml.SafeDumper):
    """A safe dumper that never writes anchors and writes scalar lists inline."""

    def ignore_aliases(self, _data: Any) -> bool:
        return True


def _represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.Node:
    flow = all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow or None)


_Dumper.add_representer(list, _represent_list)


def dump(document: Any) -> str:
    """One source file's text for ``document``."""
    return yaml.dump(
        document,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=2**16,
    )


def _load_sibling(stem: str) -> Any:
    """``<stem>.py`` beside this file, under a name no other module takes."""
    name = f"facility_demo_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{stem}.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_limits = _load_sibling("_limits")
_models = _load_sibling("_models")
_measurement = _load_sibling("_measurement")
_seeds = _load_sibling("_seeds")


def files(records: Any) -> dict[str, str]:
    """Relative path -> text for every file the generator writes.

    Args:
        records: The demo's records, a ``_records.Records``.

    Returns:
        Each file's path relative to ``data/facility`` and its text, in write order.
    """
    models = _models.build_models()
    return {
        "identity.yaml": dump(records.identity),
        "classes.yaml": dump(records.classes),
        "models.yaml": dump(models),
        "seeds.yaml": dump(_seeds.build_seeds(records.channels, models)),
        "records/places.yaml": dump(records.places),
        "records/devices.yaml": dump(records.devices),
        "records/channels.yaml": dump(records.channels),
        "records/groups.yaml": dump(records.groups),
        "limits.yaml": _limits.text(),
        _measurement.path(): dump(_measurement.build_measurement()),
    }


def write(out: Path, *, standalone: bool = False) -> list[Path]:
    """Write the demo's sources under ``out``.

    Args:
        out: The ``data/facility`` directory to write.
        standalone: Write the standalone presets' identity.

    Returns:
        The paths written, in write order.
    """
    written = []
    for rel, text in files(_records.build_records(standalone=standalone)).items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR", help="the data/facility directory"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="write the standalone presets' identity"
    )
    args = parser.parse_args(argv)
    try:
        write(args.out, standalone=args.standalone)
    except _records.RecordsError as exc:
        print(f"generate: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
