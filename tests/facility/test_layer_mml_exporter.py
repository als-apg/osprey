"""The mml layer ships the MATLAB Middle Layer exporter.

The layer copy is the only exporter; ``osprey facility import mml
--print-exporter`` prints it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER_EXPORTER = REPO_ROOT / "src" / "osprey" / "facility" / "layers" / "mml" / "mml_export.m"


def test_the_layer_ships_the_exporter() -> None:
    assert LAYER_EXPORTER.is_file()
    assert LAYER_EXPORTER.read_text(encoding="utf-8").startswith("function ")
