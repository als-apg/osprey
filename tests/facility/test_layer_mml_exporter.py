"""The mml layer ships the MATLAB Middle Layer exporter.

The layer copy is the exporter; the control-assistant data template carries the
same file so ``osprey scaffold pull`` lands it beside its README. The two are one file
in two places and never differ by a byte.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER_EXPORTER = REPO_ROOT / "src" / "osprey" / "facility" / "layers" / "mml" / "mml_export.m"
TEMPLATE_EXPORTER = (
    REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "data"
    / "mml"
    / "mml_export.m"
)


def test_the_layer_ships_the_exporter() -> None:
    assert LAYER_EXPORTER.is_file()
    assert LAYER_EXPORTER.read_text(encoding="utf-8").startswith("function ")


def test_the_template_copy_is_the_layer_copy_byte_for_byte() -> None:
    assert TEMPLATE_EXPORTER.read_bytes() == LAYER_EXPORTER.read_bytes()
