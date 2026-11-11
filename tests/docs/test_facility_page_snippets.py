"""The facility pages' snippets are facility files the build accepts.

``describe-your-facility.rst`` walks the tree ``osprey init --preset
demo`` writes, so each of its YAML blocks is checked against that
preset's bundled facility, ``templates/facilities/example``:

* a block that states whole records is pasted into one scratch copy of the
  preset, at the file it belongs to, and the copy must pass the facility
  validator (the checks of ``osprey facility validate``);
* a block that shows part of a preset record (the top-off seed, the ``LINE``
  model) must be a subset of that record as the preset states it;
* the channel list the page includes is imported the way ``osprey facility
  import list`` does it, into the preset with its bundled files replaced, and
  the page's fixes are applied on top; that facility must pass the validator;
* the shipped files the page includes exist and are bundled files.

``import-mml-export.rst`` shows fragments of an MML import's ``mapping.yaml``;
their keys are keys the mapping reads and their answers parse.

Every block on both pages is in one of these tables or in :data:`SKIPPED`, so
a new block has to be placed in one.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HOW_TO = REPO_ROOT / "docs/source/how-to"
FACILITY_PAGE = HOW_TO / "describe-your-facility.rst"
MML_PAGE = HOW_TO / "import-mml-export.rst"
TEMPLATE_ROOT = REPO_ROOT / "src/osprey/templates"
PRESET = TEMPLATE_ROOT / "facilities/example"

#: A block with no facility file to check, keyed by page name and first line.
SKIPPED: dict[tuple[str, str], str] = {
    ("describe-your-facility.rst", "data/facility/"): "a directory tree, not a file",
    ("describe-your-facility.rst", "osprey facility validate"): "commands, not a facility file",
    ("import-mml-export.rst", "osprey facility import mml --print-exporter > mml_export.m"): (
        "a command; needs MATLAB"
    ),
    (
        "import-mml-export.rst",
        "osprey facility import mml mymachine.storagering.ao.json mymachine.booster.ao.json",
    ): "a command; needs MATLAB export files",
    ("import-mml-export.rst", "osprey facility validate"): "a command, not a facility file",
}

_DIRECTIVE = re.compile(
    r"^(?P<indent>\s*)\.\. (?P<kind>code-block|literalinclude):: (?P<arg>.+?)\s*$"
)
_OPTION = re.compile(r"^\s+:(?P<name>[\w-]+):\s*(?P<value>.*)$")


@dataclass(frozen=True)
class Block:
    """One code block or ``literalinclude`` of a page."""

    page: str
    line: int
    kind: str
    arg: str
    options: dict[str, str]
    text: str

    @property
    def first(self) -> str:
        """The block's first non-blank line, its key in the tables below."""
        return next((line for line in self.text.splitlines() if line.strip()), "")

    @property
    def data(self) -> Any:
        return yaml.safe_load(self.text)


def _blocks(page: Path) -> list[Block]:
    """Every ``code-block`` and ``literalinclude`` of *page*, its body dedented."""
    lines = page.read_text(encoding="utf-8").splitlines()
    found: list[Block] = []
    index = 0
    while index < len(lines):
        match = _DIRECTIVE.match(lines[index])
        if not match:
            index += 1
            continue
        start, indent = index, len(match["indent"])
        options: dict[str, str] = {}
        index += 1
        while index < len(lines) and (option := _OPTION.match(lines[index])):
            options[option["name"]] = option["value"]
            index += 1
        body: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > indent
        ):
            body.append(lines[index])
            index += 1
        depth = min((len(x) - len(x.lstrip()) for x in body if x.strip()), default=0)
        text = "\n".join(x[depth:] for x in body).strip("\n") + "\n"
        found.append(Block(page.name, start + 1, match["kind"], match["arg"], options, text))
    return found


def _block(page: Path, first: str) -> Block:
    matches = [block for block in _blocks(page) if block.first == first]
    assert len(matches) == 1, f"{page.name}: {len(matches)} blocks open with {first!r}"
    return matches[0]


# --- pasting into the preset ------------------------------------------------------


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _update(path: Path, block: Any) -> None:
    """A mapping file with *block*'s keys replacing the file's."""
    data = _load(path) if path.exists() else {}
    data.update(block)
    _dump(path, data)


def _by(key: str) -> Callable[[Path, Any], None]:
    """A list file with each of *block*'s records replacing the one of the same *key*."""

    def paste(path: Path, block: Any) -> None:
        records = _load(path) if path.exists() else []
        for record in block:
            records = [row for row in records if row.get(key) != record[key]]
            records.append(record)
        _dump(path, records)

    return paste


#: Blocks that state whole records, by first line: the preset file they belong
#: to and how they go into it.
PASTED: dict[str, tuple[str, Callable[[Path, Any], None]]] = {
    "code: ca": ("identity.yaml", _update),
    "- id: SR/SECT1": ("records/places.yaml", _by("id")),
    "- id: SR:MAG:QF:01:CURRENT:SP": ("records/channels.yaml", _by("id")),
    "- class: Undulator": ("classes.yaml", _by("class")),
    "BR:DIAG:DCCT:01:CURRENT:RB:": ("seeds.yaml", _update),
    "SR:MAG:SEPTUM:01:CURRENT:WAVEFORM:": ("seeds.yaml", _update),
    "GTL:DIAG:BPM:01:SUM:RB:": ("seeds.yaml", _update),
}

#: Blocks that show part of a preset record, by first line: the preset file.
SUBSETS: dict[str, str] = {
    "SR:DIAG:DCCT:01:CURRENT:RB:": "seeds.yaml",
    "- name: LINE": "models.yaml",
}

#: The fixes block, applied on top of the imported channel list.
FIXES = "schema: osprey.facility.fixes/1"

#: The channel list the page includes, relative to the page.
CHANNEL_LIST = "examples/facility/channels.csv"


def _validates(facility_dir: Path) -> str:
    """The validator's output for *facility_dir*; empty-handed when it passes."""
    from osprey.facility.validate import validate

    out = StringIO()
    code = validate(facility_dir, project_name="ca", file=out)
    return "" if code == 0 else out.getvalue() or f"exit {code}"


def _subset(part: Any, whole: Any) -> bool:
    """Whether *part* states only what *whole* states.

    A mapping is a subset when each of its keys is in *whole* with a subset
    value; a list when each of its items is a subset of some item of *whole*.
    """
    if isinstance(part, dict):
        return isinstance(whole, dict) and all(
            key in whole and _subset(value, whole[key]) for key, value in part.items()
        )
    if isinstance(part, list):
        return isinstance(whole, list) and all(
            any(_subset(item, candidate) for candidate in whole) for item in part
        )
    return bool(part == whole)


def _facility_blocks() -> list[Block]:
    return [block for block in _blocks(FACILITY_PAGE) if block.kind == "code-block"]


def test_every_block_on_the_facility_page_is_checked() -> None:
    placed = {*PASTED, *SUBSETS, FIXES}
    skipped = {first for page, first in SKIPPED if page == FACILITY_PAGE.name}
    unplaced = [
        f"{block.line}: {block.first}"
        for block in _facility_blocks()
        if block.first not in placed | skipped
    ]
    assert unplaced == []
    present = {block.first for block in _facility_blocks()}
    assert sorted((placed | skipped) - present) == []


@pytest.mark.slow  # validating the whole preset takes a few seconds
def test_the_pasted_blocks_build_on_the_preset(tmp_path: Path) -> None:
    facility = tmp_path / "facility"
    shutil.copytree(PRESET, facility)
    for first, (relative, paste) in PASTED.items():
        paste(facility / relative, _block(FACILITY_PAGE, first).data)
    assert _validates(facility) == ""


@pytest.mark.parametrize("first", sorted(SUBSETS))
def test_a_partial_block_is_a_subset_of_the_preset_record(first: str) -> None:
    block = _block(FACILITY_PAGE, first).data
    preset = _load(PRESET / SUBSETS[first])
    if isinstance(block, dict):
        assert _subset(block, preset), f"{first} states what the preset's record does not"
    else:
        for record in block:
            named = [row for row in preset if row.get("name") == record["name"]]
            assert len(named) == 1, f"the preset has no single model {record['name']}"
            assert _subset(record, named[0]), f"{first} states what the preset's model does not"


def test_the_channel_list_and_its_fixes_build_on_the_preset(tmp_path: Path) -> None:
    """``osprey facility import list`` replaces the bundled preset, then imports."""
    import click

    from osprey.cli.facility_cmd import _replace_or_refuse
    from osprey.facility.layers.list import plan_list
    from osprey.facility.roots import FacilityRoots

    facility = tmp_path / "facility"
    shutil.copytree(PRESET, facility)
    plan = plan_list(HOW_TO / CHANNEL_LIST)
    with click.Context(click.Command("list")) as ctx:
        _replace_or_refuse(
            ctx, "list", FacilityRoots.single(facility), tmp_path, lambda: plan.writes
        )

    plan.write(facility)
    _dump(facility / "fixes.yaml", _block(FACILITY_PAGE, FIXES).data)

    assert _validates(facility) == ""


def test_the_page_includes_the_channel_list() -> None:
    includes = [block.arg for block in _blocks(FACILITY_PAGE) if block.kind == "literalinclude"]
    assert CHANNEL_LIST in includes


def _includes() -> Iterator[Block]:
    for block in _blocks(FACILITY_PAGE):
        if block.kind == "literalinclude" and block.arg != CHANNEL_LIST:
            yield block


@pytest.mark.parametrize("block", list(_includes()), ids=lambda block: block.arg)
def test_an_included_preset_file_is_a_bundled_file(block: Block) -> None:
    """The page shows the shipped file after the line that marks it bundled."""
    from osprey.facility.bundled import bundled_origin, header_line

    path = (HOW_TO / block.arg).resolve()
    assert path.is_relative_to(PRESET), f"{block.arg} is not a file of the preset"
    assert bundled_origin(path) == "example"
    marker = header_line("example").split(". ", 1)[1]
    assert block.options.get("start-after") == marker


# --- the MML mapping fragments ---------------------------------------------------


def test_the_mapping_fragments_hold_keys_the_mapping_reads() -> None:
    from osprey.facility.layers.mml import mapping

    families = _block(MML_PAGE, "families:").data["families"]
    for raw, body in families.items():
        path = f"families.{raw}"
        assert set(body) <= mapping._FAMILY_REQUIRED | mapping._FAMILY_OPTIONAL, path
        mapping._devices(body, path)

    models = _block(MML_PAGE, "models:").data["models"]
    for raw, body in models.items():
        path = f"models.{raw}"
        assert set(body) <= mapping._MODEL_REQUIRED | mapping._MODEL_OPTIONAL, path
        mapping._places(body["places"], f"{path}.places")


def test_every_block_on_the_mml_page_is_checked() -> None:
    placed = {"families:", "models:"}
    skipped = {first for page, first in SKIPPED if page == MML_PAGE.name}
    firsts = [block.first for block in _blocks(MML_PAGE)]
    assert sorted(set(firsts) - placed - skipped) == []
    assert sorted((placed | skipped) - set(firsts)) == []
