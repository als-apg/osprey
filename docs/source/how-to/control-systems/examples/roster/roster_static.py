"""A roster source that reads a static list of IOCs from a YAML file.

Point ``channel_directory.fleet.roster_sources`` at ``roster_static:roster`` (with this
file importable) to use the list beside it, or construct ``StaticRoster`` on your
own file. The file is read on every refresh, so an edit shows on the next one; a
missing or malformed file makes ``read`` raise, and the view reports this source
unavailable while keeping the others.

The file::

    servers:
      - {name: SR01C-IOC1, host: ioc-sr01, class: hard, expected: running, group: storage-ring}
      - {name: BTS-SOFTIOC, host: srv3, expected: stopped}
      - {name: LAB-TEST}            # host, class, expected and group are optional
"""

from __future__ import annotations

from pathlib import Path

import yaml

from osprey.services.channel_directory.roster import RosterRow


class StaticRoster:
    """Every IOC listed under ``servers:`` in one YAML file."""

    def __init__(self, path: str | Path, name: str | None = None) -> None:
        self.path = Path(path)
        self.name = name or self.path.stem

    def read(self, deadline: float) -> list[RosterRow]:
        del deadline  # a local file read needs no time budget
        document = yaml.safe_load(self.path.read_text())
        entries = document.get("servers") if isinstance(document, dict) else None
        if not isinstance(entries, list):
            raise ValueError(f"{self.path}: expected a list under 'servers:'")
        return [
            RosterRow(
                entry["name"],
                entry.get("host"),
                entry.get("class"),
                entry.get("expected"),
                group=entry.get("group"),
            )
            for entry in entries
        ]


roster = StaticRoster(Path(__file__).with_name("roster_static.yml"), name="static-list")
