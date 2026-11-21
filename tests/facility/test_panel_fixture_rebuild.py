"""The panel tests' checked-in facility document is what the build writes.

``tests/facility/panels/fixture/facility.json`` stands in for building the
fixture tree in every panel session. These tests rebuild it from the tree and
compare byte for byte, so a change to the tree or to the build that leaves the
document behind fails here, naming the regeneration command. They live in the
root tests, beside the build, not under ``tests/facility/panels/``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from tests.facility.panels._fixture import FIXTURE_DOCUMENT, load_fixture_document

SCRIPT = Path(__file__).with_name("rebuild_panel_fixture.py")
COMMAND = "uv run python tests/facility/rebuild_panel_fixture.py"


def _script():
    spec = importlib.util.spec_from_file_location("rebuild_panel_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_checked_in_document_is_the_fixture_tree_built_byte_for_byte() -> None:
    script = _script()
    assert script.COMMAND == COMMAND
    assert FIXTURE_DOCUMENT.is_file(), f"{FIXTURE_DOCUMENT} is missing: run {COMMAND}"
    assert FIXTURE_DOCUMENT.read_bytes() == script.rebuilt(), (
        f"{FIXTURE_DOCUMENT.relative_to(SCRIPT.parents[2])} is stale: run {COMMAND}"
    )


def test_the_loader_reads_the_checked_in_document_fresh_each_time() -> None:
    first = load_fixture_document()
    assert first == json.loads(FIXTURE_DOCUMENT.read_text(encoding="utf-8"))
    first["devices"].clear()
    assert load_fixture_document()["devices"], "each call returns its own copy"


def test_a_stale_document_fails_the_check_naming_the_command_and_passes_once_regenerated(
    tmp_path: Path, capsys
) -> None:
    script = _script()
    stale = tmp_path / "facility.json"
    stale.write_bytes(FIXTURE_DOCUMENT.read_bytes().replace(b'"XR01:QF1"', b'"XR01:QF9"', 1))

    assert script.main(["--check", "--path", str(stale)]) == 1
    assert f"is stale; run {COMMAND}" in capsys.readouterr().out

    assert script.main(["--path", str(stale)]) == 0
    assert stale.read_bytes() == script.rebuilt()
    assert script.main(["--check", "--path", str(stale)]) == 0


def test_a_missing_document_fails_the_check_naming_the_command(tmp_path: Path, capsys) -> None:
    script = _script()
    assert script.main(["--check", "--path", str(tmp_path / "facility.json")]) == 1
    assert f"is missing; run {COMMAND}" in capsys.readouterr().out


def test_the_script_runs_as_the_command_names_it() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=SCRIPT.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
