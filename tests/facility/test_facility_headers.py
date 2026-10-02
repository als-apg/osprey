"""Every file the facility outputs write names its document and version.

A JSON file carries ``"schema": "osprey.facility.<doc>/1"`` at its top level; a
YAML or Markdown file starts with the line ``schema: osprey.facility.<doc>/1``.
The files checked are the ones ``render_facility_outputs`` writes in each render
of a control-assistant build: the facility file, and every view it writes.

Exempt, each for a stated reason: the files whose format is fixed outside
OSPREY (the limits view keeps its ``_version`` key, the graph view's TTL keeps a
comment header), the deck copies (pyAT's own lattice format) and the binary
files on ``BINARY_EXEMPTIONS``.
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

# xdist_group("built_control_assistant"): every module reading the session's one
# control-assistant build shares a worker, so the build runs once per run.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("built_control_assistant")]

#: A header value: the document's name and version 1.
HEADER = re.compile(r"osprey\.facility\.[a-z_]+/1")

#: Files, by pattern relative to the render root, whose format is fixed outside
#: OSPREY and so carries no ``schema`` header.
FORMAT_EXEMPTIONS: dict[str, str] = {
    "data/channel_limits.json": 'the limits view keeps `_version: "4.0"`',
    "*.ttl": "the graph view's TTL keeps a comment header",
    "*decks/*.json": "a deck copy is pyAT's own lattice format",
}

#: Binary files a view writes, relative to the render root; each must exist in
#: the main render.
BINARY_EXEMPTIONS: tuple[str, ...] = ()


def _exempt(relative: str) -> bool:
    return relative in BINARY_EXEMPTIONS or any(
        fnmatch(relative, pattern) for pattern in FORMAT_EXEMPTIONS
    )


def header_problem(relative: str, content: bytes) -> str | None:
    """What is wrong with one written file's header.

    Args:
        relative: The file's path relative to its render root.
        content: The file's bytes.

    Returns:
        The problem, or ``None`` when the file carries a header.
    """
    suffix = Path(relative).suffix
    if suffix == ".json":
        document = json.loads(content)
        value = document.get("schema") if isinstance(document, dict) else None
        if not isinstance(value, str) or not HEADER.fullmatch(value):
            return f"top-level `schema` is {value!r}"
        return None
    if suffix in (".yaml", ".yml", ".md"):
        first = content.decode("utf-8").split("\n", 1)[0]
        if not first.startswith("schema: ") or not HEADER.fullmatch(first[len("schema: ") :]):
            return f"first line is {first!r}"
        return None
    return f"no header rule for {suffix or 'a file without a suffix'}"


def test_every_written_file_carries_its_header(built_control_assistant: BuiltProject) -> None:
    outputs = built_control_assistant.outputs
    assert outputs
    problems = {
        relative: problem
        for render in outputs
        for relative, content in sorted(render.files.items())
        if not _exempt(relative) and (problem := header_problem(relative, content))
    }
    assert problems == {}


def test_the_facility_file_is_written_in_every_render_with_its_header(
    built_control_assistant: BuiltProject,
) -> None:
    from osprey.facility.render import FACILITY_FILE

    for render in built_control_assistant.outputs:
        assert json.loads(render.files[FACILITY_FILE])["schema"] == "osprey.facility.facility/1"


def test_the_bluesky_devices_view_is_checked(built_control_assistant: BuiltProject) -> None:
    content = built_control_assistant.outputs[0].files["data/bluesky_devices.yml"]
    assert header_problem("data/bluesky_devices.yml", content) is None


def test_the_in_context_index_is_checked(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.channel_finder import write_in_context

    inputs = ViewInputs(
        doc=built_control_assistant.facility,
        rendered_config={"channel_finder": {"pipeline_mode": "in_context"}},
        facility_dir=built_control_assistant.facility_dir,
        served=[],
    )
    (target,) = write_in_context(tmp_path, inputs)
    assert header_problem("data/channel_finder/in_context.json", target.read_bytes()) is None


def test_every_binary_exemption_is_written(built_control_assistant: BuiltProject) -> None:
    build_dir = built_control_assistant.build_dir
    assert [
        relative for relative in BINARY_EXEMPTIONS if not (build_dir / relative).is_file()
    ] == []


@pytest.mark.parametrize(
    ("relative", "content", "problem"),
    [
        ("facility.json", b'{"schema": "osprey.facility.facility/1"}', None),
        ("data/x/view.json", b'{"channels": []}', "top-level `schema` is None"),
        (
            "data/x/view.json",
            b'{"schema": "osprey.facility.x/2"}',
            "top-level `schema` is 'osprey.facility.x/2'",
        ),
        ("data/x/view.json", b"[]", "top-level `schema` is None"),
        ("data/x/view.yaml", b"schema: osprey.facility.x/1\nrows: []\n", None),
        (
            "data/x/view.yaml",
            b"# comment\nschema: osprey.facility.x/1\n",
            "first line is '# comment'",
        ),
        ("data/x/README.md", b"schema: osprey.facility.readme/1\n# Title\n", None),
        ("data/x/README.md", b"# Title\n", "first line is '# Title'"),
        ("data/x/index.db", b"\x00", "no header rule for .db"),
    ],
)
def test_the_header_rule(relative: str, content: bytes, problem: str | None) -> None:
    assert header_problem(relative, content) == problem


@pytest.mark.parametrize(
    ("relative", "exempt"),
    [
        ("data/channel_limits.json", True),
        ("data/graph/facility.ttl", True),
        ("data/simulator/decks/SR.json", True),
        ("facility.json", False),
        ("data/simulator/served_models.json", False),
    ],
)
def test_the_exemptions(relative: str, exempt: bool) -> None:
    assert _exempt(relative) is exempt
