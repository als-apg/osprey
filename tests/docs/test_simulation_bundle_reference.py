"""The simulation bundle reference page matches the loader that reads the bundle.

Two invariants hold the page to the code. The worked example the page prints is
a bundle the real :class:`~osprey.simulation.SimulationEngine` accepts: it is
written to disk exactly as captioned, loaded, composed, applied and read. And the
page names the loader's whole vocabulary: every event shape and value key, every
position key, every ``texture`` key and kind, and every field of a ``physics``
block and a logbook entry appears on it as an inline literal.
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from osprey.simulation.engine import SimulationEngine, resolve_active_scenarios

PAGE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "source"
    / "reference"
    / "contracts"
    / "simulation-bundle.rst"
)

_DIRECTIVE = re.compile(r"^(?P<indent> *)\.\. code-block:: json\s*$")
_OPTION = re.compile(r"^ *:(?P<name>[\w-]+):(?: +(?P<value>.*))?$")


def _captioned_json_blocks(text: str) -> dict[str, str]:
    """``{caption: body}`` for every ``.. code-block:: json`` carrying a ``:caption:``."""
    lines = text.splitlines()
    blocks: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = _DIRECTIVE.match(lines[i])
        i += 1
        if match is None:
            continue
        indent = len(match.group("indent"))
        caption = None
        while i < len(lines) and (option := _OPTION.match(lines[i])):
            if option.group("name") == "caption":
                caption = (option.group("value") or "").strip()
            i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            i += 1
        if caption:
            blocks[caption] = textwrap.dedent("\n".join(body)).strip() + "\n"
    return blocks


def _worked_example() -> dict[str, str]:
    return _captioned_json_blocks(PAGE.read_text(encoding="utf-8"))


def _scenario_name(blocks: dict[str, str]) -> str:
    (scenario_path,) = [c for c in blocks if c.endswith("/scenario.json")]
    return PurePosixPath(scenario_path).parent.name


def test_the_worked_example_is_three_files_of_one_bundle() -> None:
    blocks = _worked_example()
    captions = [PurePosixPath(c) for c in blocks]
    assert PurePosixPath("data/simulation/machine.json") in captions
    scenario_files = [c for c in captions if c.name == "scenario.json"]
    logbook_files = [c for c in captions if c.name == "logbook.json"]
    assert len(blocks) == 3, sorted(blocks)
    assert len(scenario_files) == 1 and len(logbook_files) == 1, sorted(blocks)
    (scenario_file,) = scenario_files
    (logbook_file,) = logbook_files
    assert scenario_file.parent == logbook_file.parent
    assert scenario_file.parent.parent == PurePosixPath("data/simulation/scenarios")


def test_the_worked_example_is_a_bundle_the_loader_accepts(tmp_path: Path) -> None:
    blocks = _worked_example()
    for caption, body in blocks.items():
        target = tmp_path / caption
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")

    machine_path = tmp_path / "data" / "simulation" / "machine.json"
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    engine = SimulationEngine(machine, machine_path, state_dir=tmp_path / "state")
    name = _scenario_name(blocks)

    assert name in engine.list_scenarios()
    assert engine.validate_composition(resolve_active_scenarios([name])) == []
    assert engine.scenario_logbook(name)

    expr_channels = [pv for pv, spec in machine["channels"].items() if "expr" in spec]
    assert expr_channels, "the example machine declares no expr channel"
    engine.set_active_scenarios([name], anchor=datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    reading = engine.read(expr_channels[0])
    assert isinstance(reading.value, float)


def test_the_page_names_the_loaders_vocabulary() -> None:
    """Every key the loader reads appears on the page as an inline literal.

    The names are imported from the loader's private module on purpose: a key
    the loader gains is a key the page must document, and this is where that
    shows.
    """
    from dataclasses import fields

    from osprey_connectors.simulation.machine import (
        _ATTACHMENT_KEYS,
        _EVENT_VALUE_KEYS,
        _TEXTURE_KEYS,
        _TEXTURE_KINDS,
        BpmErrorSpec,
        PhysicsFault,
        ScenarioLogEntry,
    )

    vocabulary: set[str] = set()
    for shape, value_keys in _EVENT_VALUE_KEYS.items():
        vocabulary.add(shape)
        vocabulary.update(value_keys)
    vocabulary.update({"at", "at_offset", "at_time", "at_when", "until", "until_offset"})
    vocabulary.update(_TEXTURE_KEYS)
    vocabulary.update(_TEXTURE_KINDS)
    vocabulary.update(_ATTACHMENT_KEYS)
    for model in (BpmErrorSpec, PhysicsFault, ScenarioLogEntry):
        vocabulary.update(f.name for f in fields(model))
    vocabulary.update({"days_ago", "time"})

    text = PAGE.read_text(encoding="utf-8")
    missing = sorted(key for key in vocabulary if f"``{key}``" not in text)
    assert not missing, f"{PAGE.name} does not name {missing}"
