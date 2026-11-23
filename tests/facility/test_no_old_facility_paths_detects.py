"""The old-path sweep catches what it guards against and passes what it must.

A sweep that matches nothing passes on any tree, so these cases run
:func:`old_path_offenders` against a stand-in tree with each form of old-path
reference planted in it, beside the spellings that must stay legal: document
identifiers, the surviving framework modules, the new package and a lookalike
module name.
"""

from __future__ import annotations

from pathlib import Path

from tests.facility.test_no_old_facility_paths import old_path_offenders

MODULES = frozenset({"build", "kinds", "records", "scenarios"})
NAMES = MODULES | {"DEFAULT_ROLE"}


def _plant(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sweep(root: Path) -> list[str]:
    offenders, _ = old_path_offenders(
        root,
        ("src", "docs/source", "pyproject.toml", "scripts"),
        modules=MODULES,
        names=NAMES,
        exempt_files=frozenset(),
        line_allowlist={},
    )
    return offenders


def test_planted_old_paths_fail(tmp_path: Path) -> None:
    _plant(
        tmp_path,
        "src/pkg/mod.py",
        "import osprey.facility.build\n"
        "from osprey.facility.records import Record\n"
        "from osprey.facility import DEFAULT_ROLE, views\n"
        "from osprey.facility import (\n"
        "    views,\n"
        "    kinds,\n"
        ")\n"
        'importlib.import_module("osprey.facility.kinds")\n',
    )
    _plant(
        tmp_path,
        "pyproject.toml",
        '[project.entry-points."osprey.facility.kinds"]\nbeam = "osprey.facility.build:main"\n',
    )
    _plant(
        tmp_path,
        "docs/source/page.rst",
        "See :mod:`osprey.facility.build` and :func:`~osprey.facility.records.load`.\n"
        ".. automodule:: osprey.facility.scenarios\n"
        "The writer is ``src/osprey/facility/build.py``.\n",
    )
    _plant(tmp_path, "scripts/run.sh", "python scripts/facility_schema/json_schema.py\n")

    assert sorted(_sweep(tmp_path)) == sorted(
        [
            "src/pkg/mod.py:1: osprey.facility.build",
            "src/pkg/mod.py:2: osprey.facility.records",
            "src/pkg/mod.py:3: from osprey.facility import DEFAULT_ROLE",
            "src/pkg/mod.py:4: from osprey.facility import kinds",
            "src/pkg/mod.py:8: osprey.facility.kinds",
            "pyproject.toml:1: osprey.facility.kinds",
            "pyproject.toml:2: osprey.facility.build",
            "docs/source/page.rst:1: osprey.facility.build",
            "docs/source/page.rst:1: osprey.facility.records",
            "docs/source/page.rst:2: osprey.facility.scenarios",
            "docs/source/page.rst:3: osprey/facility/build",
            "scripts/run.sh:1: scripts/facility_schema",
        ]
    )


def test_identifiers_and_surviving_modules_pass(tmp_path: Path) -> None:
    _plant(
        tmp_path,
        "src/pkg/ok.py",
        'SCHEMA = "osprey.facility.facility/4"\n'
        'SCENARIOS = "osprey.facility.scenarios/3"\n'
        "# header osprey.facility.facility/N\n"
        "from osprey.facility.views import reader\n"
        "from osprey.facility import panels, settings\n"
        "from osprey.facility import other\n"
        "import osprey.facility\n"
        "import osprey_facility.build\n"
        "from osprey_facility.kinds import ENTRY_POINT_GROUP\n"
        'GROUP = "osprey_facility.kinds"\n'
        'LOOKALIKE = "osprey.facility.kinds_extra"\n',
    )
    _plant(tmp_path, "docs/source/ok.rst", "See :mod:`osprey_facility.build`.\n")

    assert _sweep(tmp_path) == []


def test_exempt_files_and_allowlisted_lines_pass(tmp_path: Path) -> None:
    _plant(tmp_path, "src/CHANGELOG.md", "- Removed ``osprey.facility.kinds``.\n")
    _plant(tmp_path, "src/frozen.json", '{"d": "osprey.facility.kinds entry points"}\n')
    _plant(tmp_path, "src/keep.py", 'OLD = "osprey.facility.kinds"  # proves it is ignored\n')

    offenders, used = old_path_offenders(
        tmp_path,
        ("src",),
        modules=MODULES,
        names=NAMES,
        exempt_files=frozenset({"src/frozen.json"}),
        line_allowlist={("src/keep.py", "proves it is ignored"): "negative case"},
    )

    assert offenders == []
    assert used == {("src/keep.py", "proves it is ignored")}
