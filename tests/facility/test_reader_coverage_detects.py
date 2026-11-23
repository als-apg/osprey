"""The reader-coverage scan fails on the readers it exists to catch.

``test_reader_coverage`` passes on today's tree; this module proves its scan would
fail on a module that parses ``facility.json`` unchecked. Each case plants a module
in a temporary ``src`` tree and runs the scan there; nothing here touches ``src/``.
The controls plant the allowed uses, so a failing case fails for its own use.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.facility.test_reader_coverage import (
    CHECKED_ENTRY_POINTS,
    HEADER_CHECKS,
    callers,
    flow_problems,
    name_sites,
    reaches,
)


def tree(root: Path, files: Mapping[str, str]) -> Path:
    """A ``src`` tree under ``root`` holding ``files``, every directory a package.

    The scan is cached by its source root and pytest reuses the directory of a passed
    test, so each tree gets a name of its own.
    """
    src = root / uuid.uuid4().hex / "src"
    for name, text in files.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        for parent in path.relative_to(src).parents:
            if parent != Path("."):
                init = src / parent / "__init__.py"
                if not init.exists():
                    init.touch()
    return src


PLANTED = "osprey/planted/reader.py"

#: Unchecked readers: (the planted module's source, a word the problem names).
UNCHECKED = [
    pytest.param(
        "import json\n"
        "from osprey.facility import FACILITY_FILE\n"
        "def f(p):\n"
        "    return json.loads(p.read_text())\n"
        "def g(root):\n"
        "    return f(root / FACILITY_FILE)\n",
        ".read_text",
        id="helper-takes-the-path-as-a-parameter",
    ),
    pytest.param(
        "import json\n"
        "from osprey.facility import FACILITY_FILE\n"
        "from osprey.facility.reading import load_facility\n"
        "def g(root):\n"
        "    p = root / FACILITY_FILE\n"
        "    load_facility(p)\n"
        "    return json.loads(p.read_text())\n",
        ".read_text",
        id="checks-and-also-parses-raw",
    ),
    pytest.param(
        "import json\n"
        "from osprey.facility import FACILITY_FILE\n"
        "def load_facility(p):\n"
        "    return json.loads(p.read_text())\n"
        "def g(root):\n"
        "    return load_facility(root / FACILITY_FILE)\n",
        ".read_text",
        id="a-local-load_facility-is-no-entry-point",
    ),
    pytest.param(
        "from osprey.facility import FACILITY_FILE\n"
        "def load_facility(p):\n"
        "    return p\n"
        "def g(root):\n"
        "    return load_facility(root / FACILITY_FILE)\n",
        "returned",
        id="a-local-load_facility-that-hands-it-on",
    ),
    pytest.param(
        "import json\n"
        "def g(root):\n"
        "    return json.loads((root / 'build/facility.json').read_text())\n",
        ".read_text",
        id="a-path-spelling-of-the-literal",
    ),
    pytest.param(
        "import json\n"
        "def g(root):\n"
        "    with open(root / 'facility.json') as handle:\n"
        "        return json.load(handle)\n",
        "open()",
        id="the-literal-through-open",
    ),
    pytest.param(
        "from osprey.facility.render import FACILITY_FILE as NAME\n"
        "def g(root):\n"
        "    return (root / NAME).read_bytes()\n",
        ".read_bytes",
        id="an-aliased-re-export",
    ),
    pytest.param(
        "import osprey.facility as facility\n"
        "def g(root):\n"
        "    return (root / facility.FACILITY_FILE).open()\n",
        ".open",
        id="a-module-attribute",
    ),
    pytest.param(
        "from osprey.facility.panels.document import FACILITY_FILE\n"
        "def g(root):\n"
        "    return (root / FACILITY_FILE).read_text()\n",
        ".read_text",
        id="the-panel-engine-s-name",
    ),
    pytest.param(
        "from osprey.channel_roster.sources import facility_file_path\n"
        "def g(config):\n"
        "    return facility_file_path(config).read_bytes()\n",
        ".read_bytes",
        id="facility_file_path",
    ),
    pytest.param(
        "from osprey.apps.sdk import facility_file\n"
        "def g():\n"
        "    path = facility_file()\n"
        "    return path.read_text()\n",
        ".read_text",
        id="the-sdk-facility_file",
    ),
    pytest.param(
        "import shutil\n"
        "from osprey.facility import FACILITY_FILE\n"
        "def g(root, out):\n"
        "    shutil.copy(root / FACILITY_FILE, out)\n",
        "passed to shutil.copy",
        id="handed-to-another-module",
    ),
    pytest.param(
        "from osprey.facility import FACILITY_FILE\n"
        "def g(root):\n"
        "    return root / FACILITY_FILE\n",
        "returned",
        id="returned-outside-path-only",
    ),
    pytest.param(
        "from osprey.facility import FACILITY_FILE\n"
        "def g(root):\n"
        "    paths = [root / FACILITY_FILE]\n"
        "    return [p.read_text() for p in paths]\n",
        "container",
        id="stored-in-a-container",
    ),
    pytest.param(
        "from osprey.facility import FACILITY_FILE\n"
        "def g(root):\n"
        "    path = root / FACILITY_FILE\n"
        "    def inner():\n"
        "        return path.read_text()\n"
        "    return inner()\n",
        ".read_text",
        id="read-in-a-closure",
    ),
    pytest.param(
        "from osprey.facility import FACILITY_FILE\n"
        "from osprey.facility.records import FacilityRecords\n"
        "class Holder:\n"
        "    def make(self, root):\n"
        "        return self.parse(root / FACILITY_FILE)\n"
        "    def parse(self, path):\n"
        "        return path.read_text()\n",
        ".read_text",
        id="a-method-helper",
    ),
]

#: The SDK package re-exports ``facility_file``, as the real one does.
SDK = {"osprey/apps/sdk/__init__.py": "from osprey.apps.sdk._facts import facility_file\n"}


@pytest.mark.parametrize(("source", "named"), UNCHECKED)
def test_an_unchecked_reader_fails_the_scan(tmp_path: Path, source: str, named: str):
    src = tree(tmp_path, {**SDK, PLANTED: source})
    problems = flow_problems(src)
    assert problems, "the planted reader passed the scan"
    assert any(named in str(problem) for problem in problems), problems


#: Allowed uses: none of these is a problem.
CHECKED = [
    pytest.param(
        "from osprey.facility import FACILITY_FILE\n"
        "from osprey.facility.reading import load_facility\n"
        "def g(root):\n"
        "    return load_facility(root / FACILITY_FILE)\n",
        id="load_facility",
    ),
    pytest.param(
        "from osprey.channel_roster.sources import facility_file_path\n"
        "from osprey.facility.views.facts import parsed_facility_file as parsed\n"
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def g(config):\n"
        "    path = facility_file_path(config)\n"
        "    if not path.exists():\n"
        "        logger.warning('no %s', path)\n"
        "        raise FileNotFoundError(f'{path} is not there')\n"
        "    return parsed(path)\n",
        id="probes-formats-and-an-aliased-entry-point",
    ),
    pytest.param(
        "from pathlib import Path\n"
        "from osprey.facility import FACILITY_FILE\n"
        "from osprey.facility.records import FacilityRecords\n"
        "def _read(path):\n"
        "    return FacilityRecords.from_file(path)\n"
        "def g(root):\n"
        "    return _read(Path(root) / FACILITY_FILE)\n",
        id="a-same-module-helper-that-checks",
    ),
    pytest.param(
        '"""Reads facility.json, the build\'s document."""\n'
        "def g():\n"
        '    """Never opens facility.json."""\n'
        "    return 'no facility.json here; run osprey build'\n",
        id="docstrings-and-prose",
    ),
]


@pytest.mark.parametrize("source", CHECKED)
def test_a_checked_reader_passes_the_scan(tmp_path: Path, source: str):
    src = tree(tmp_path, {**SDK, PLANTED: source})
    assert flow_problems(src) == []


def test_the_scan_sees_every_planted_name(tmp_path: Path):
    """The control for the cases: each planted spelling is found as a name."""
    sources = {f"osprey/planted/case{i}.py": case.values[0] for i, case in enumerate(UNCHECKED)}
    src = tree(tmp_path, {**SDK, **sources})
    found = {scope.module for scope, _, _ in name_sites(src)}
    assert found == {f"osprey.planted.case{i}" for i in range(len(UNCHECKED))}


READING = (
    "def check_header(document, source):\n"
    "    pass\n"
    "def load_facility(path):\n"
    "    document = path.read_text()\n"
    "    check_header(document, str(path))\n"
    "    return document\n"
)


@pytest.mark.parametrize(
    ("utils", "checks"),
    [
        pytest.param(
            "import json\n"
            "def facility_name(root, name):\n"
            "    return json.loads((root / 'x').read_text())['name']\n",
            False,
            id="parses-raw",
        ),
        pytest.param(
            "from osprey.facility.reading import load_facility\n"
            "def _recorded(root):\n"
            "    return load_facility(root)\n"
            "def facility_name(root, name):\n"
            "    return _recorded(root)['name']\n",
            True,
            id="through-a-private-helper",
        ),
        pytest.param(
            "def load_facility(path):\n"
            "    return {}\n"
            "def facility_name(root, name):\n"
            "    return load_facility(root)['name']\n",
            False,
            id="a-local-load_facility",
        ),
    ],
)
def test_an_entry_point_that_skips_the_header_check_is_caught(
    tmp_path: Path, utils: str, checks: bool
):
    src = tree(
        tmp_path,
        {"osprey/facility/reading.py": READING, "osprey/utils/facility.py": utils},
    )
    targets = HEADER_CHECKS | CHECKED_ENTRY_POINTS
    assert reaches(src, "osprey.utils.facility.facility_name", targets) is checks


def test_a_new_caller_of_an_entry_point_is_found(tmp_path: Path):
    src = tree(
        tmp_path,
        {
            "osprey/facility/reading.py": READING,
            PLANTED: "from osprey.facility import reading\n"
            "def g(path):\n"
            "    return reading.load_facility(path)\n",
        },
    )
    assert "osprey.planted.reader.g" in callers(src, CHECKED_ENTRY_POINTS)
