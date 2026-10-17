"""Every stand-in for an upstream gap has a canary that fails when the gap closes.

A stand-in is code OSPREY carries only because a library it pins lacks a
behaviour; it is marked with a comment line that begins ``# Stand-in until``
and goes on to name that behaviour. Its canary sits beside that library's pin
test and passes only while the pinned release still lacks the behaviour, so
the pin that carries the fix fails the canary and the stand-in is deleted with
it. The table is keyed on the marker's sentence, so two stand-ins in one file
need two canaries. A marker with no canary, or a canary whose stand-in is gone,
fails here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests._tree_scan import product_sources

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Each stand-in's marker sentence, with the canary that watches its gap.
CANARIES: dict[str, str] = {
    "pyAML's MeasurementTool.save writes numpy arrays and scalars as JSON; "
    "plain() and save() go with it.": (
        "tests/pyaml_cs_osprey/test_pyaml_pin.py::test_pyaml_save_still_refuses_numpy_values"
    ),
    "lume-base states SolveError, the base of every LUME engine's no-solve exception.": (
        "tests/simulation/test_lume_pin.py::test_no_lume_base_class_is_behind_the_orbit_solve_error"
    ),
    "pyAML reads a dipole kick held in KickAngle; "
    "polynomial_kicks and its EnginePlugin row go with it.": (
        "tests/pyaml_cs_osprey/test_pyaml_pin.py::test_pyaml_still_reads_no_kick_angle"
    ),
}

_MARKER = re.compile(r"^\s*# Stand-in until (?P<sentence>.*)$", re.MULTILINE)


def _markers(root: Path) -> list[tuple[str, str]]:
    """Every stand-in marker in the shipped code under ``root``, as ``(file, sentence)``."""
    return [
        (relative, match["sentence"])
        for relative, text in product_sources(root)
        for match in _MARKER.finditer(text)
    ]


def _defines(node_id: str) -> bool:
    relative, _, name = node_id.partition("::")
    path = REPO_ROOT / relative
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body)


def test_every_stand_in_has_a_canary_and_every_canary_a_stand_in() -> None:
    markers = _markers(REPO_ROOT)
    unpaired = sorted(
        f"{relative}: {sentence}" for relative, sentence in markers if sentence not in CANARIES
    )
    gone = sorted(set(CANARIES) - {sentence for _, sentence in markers})
    assert not unpaired and not gone, (
        f"stand-ins with no canary: {unpaired}; canaries whose stand-in is gone: {gone}"
    )


def test_every_canary_named_here_exists() -> None:
    missing = sorted(node_id for node_id in CANARIES.values() if not _defines(node_id))
    assert not missing, f"no such test: {missing}"


def test_a_marker_in_a_package_source_is_found(tmp_path: Path) -> None:
    module = tmp_path / "packages" / "example" / "src" / "example" / "writer.py"
    module.parent.mkdir(parents=True)
    module.write_text("    # Stand-in until the library writes it.\nVALUE = 1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "plain.py").write_text('"""A stand-in for nothing."""\n', encoding="utf-8")

    assert _markers(tmp_path) == [
        ("packages/example/src/example/writer.py", "the library writes it.")
    ]
