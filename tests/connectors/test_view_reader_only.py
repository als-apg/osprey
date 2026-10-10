"""Only the simulator view's reader and its writer open a view file.

A view is opened through ``osprey_connectors.simulation.view.SimulatorView``,
which refuses a file of another schema; the build writes it through
``osprey.facility.views.simulator``. A file anywhere else that spells a view
file's name, joins a view file's constant onto a path, or joins the view's
``simulator`` directory onto a data root opens the view behind the reader's
back, so a schema change would reach it as a misread rather than a refusal.

The scan runs ``git grep`` over the tracked ``*.py`` files under
``SCAN_PATHS``: the product, the scripts and the two e2e harness directories.
Agent-facing text and the unit fixtures that write a view to read it back lie
outside it by design.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The trees the scan reads.
SCAN_PATHS: tuple[str, ...] = ("src", "packages", "scripts", "tests/e2e", "tests/va/e2e")

#: A view file's name as a string literal or as the last part of a path.
FILE_LITERAL = r"[\"'/](served_models|addresses|variables|seeds|scenarios)\.json[\"']"
#: A view file's name constant joined onto a path.
FILE_JOIN = r"/\s*(SERVED_MODELS|ADDRESSES|VARIABLES|SEEDS|SCENARIOS)_FILE\b"
#: The view's directory joined onto a data root, by ``/`` or as path parts.
DIR_JOIN = r"/\s*[\"']simulator[\"']|[\"']data[\"']\s*,\s*[\"']simulator[\"']"

TOKENS: tuple[str, ...] = (FILE_LITERAL, FILE_JOIN, DIR_JOIN)

#: The reader every consumer opens a view with.
READER = "packages/osprey-connectors/src/osprey_connectors/simulation/view.py"
#: The writer the build renders a view with.
WRITER = "src/osprey/facility/views/simulator.py"

#: Where a token matches without opening a view, each with why.
ZONES: dict[str, str] = {
    "src/osprey/facility/layers/**": (
        "the importers join their facility tree's own seeds.yaml as SEEDS_FILE, not the view"
    ),
    "src/osprey/interfaces/lattice_dashboard/catalog.py": (
        "names the view's variables file for the dashboard's change signature; "
        "the catalog reads through SimulatorView"
    ),
    "src/osprey/services/archiver_recorder/config.py": (
        "spells the addresses file only in its refusal; opens the view through SimulatorView"
    ),
    "src/osprey/services/virtual_accelerator/entrypoint.py": (
        "spells the addresses file only in its fatal message; opens the view through SimulatorView"
    ),
}


def scan(root: Path, allowed: Mapping[str, str] = ZONES) -> list[str]:
    """Every ``path:line`` under ``root`` that opens a view file.

    The reader, the writer and the ``allowed`` globs are left out.
    """
    includes = [f":(glob){path}/**/*.py" for path in SCAN_PATHS]
    excludes = [f":(exclude,glob){glob}" for glob in (READER, WRITER, *allowed)]
    patterns = [argument for token in TOKENS for argument in ("-e", token)]
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-P", *patterns, "--", *includes, *excludes],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return sorted(":".join(line.split(":", 2)[:2]) for line in result.stdout.splitlines() if line)


def _planted_repo(root: Path, files: Mapping[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


_CONNECTORS = "packages/osprey-connectors/src/osprey_connectors"


def test_a_join_onto_a_view_file_under_the_connectors_is_flagged(tmp_path: Path) -> None:
    root = _planted_repo(
        tmp_path,
        {
            f"{_CONNECTORS}/control_system/planted.py": 'document = view / "variables.json"\n',
            f"{_CONNECTORS}/control_system/joined.py": "path = root / SEEDS_FILE\n",
            f"{_CONNECTORS}/control_system/directory.py": 'view = data_root / "simulator"\n',
            f"{_CONNECTORS}/control_system/parts.py": 'view = Path(root, "data", "simulator")\n',
            f"{_CONNECTORS}/control_system/reader.py": "document = view.document(VARIABLES_FILE)\n",
            f"{_CONNECTORS}/control_system/docstring.py": '"""Reads ``addresses.json``."""\n',
            READER: 'ADDRESSES_FILE = "addresses.json"\n',
            WRITER: "target = root / VARIABLES_FILE\n",
            "src/osprey/facility/layers/mml/seed.py": "seeds = facility_dir / SEEDS_FILE\n",
            "tests/connectors/fixture.py": 'path = view / "variables.json"\n',
        },
    )

    assert [hit.split(":")[0] for hit in scan(root)] == [
        f"{_CONNECTORS}/control_system/directory.py",
        f"{_CONNECTORS}/control_system/joined.py",
        f"{_CONNECTORS}/control_system/parts.py",
        f"{_CONNECTORS}/control_system/planted.py",
    ]


def test_only_the_reader_and_the_writer_open_a_view_file() -> None:
    assert scan(REPO_ROOT) == []
