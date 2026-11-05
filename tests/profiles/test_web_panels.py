"""No shipped preset carries the retired LATTICE tab.

The profile parser drops ``lattice`` from a document with a warning, so a
resolved preset would hide one still listing it. The raw files are read
instead: every preset, tier and app document the materializer ships.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile_document import RETIRED_PANEL_ID
from osprey.cli.build_profile_presets import _presets_dir

_DOCUMENTS = sorted(Path(_presets_dir()).rglob("*.yml"))


def test_the_preset_tree_is_found() -> None:
    assert any(path.name == "control-assistant.yml" for path in _DOCUMENTS)


@pytest.mark.parametrize("path", _DOCUMENTS, ids=lambda path: path.name)
def test_no_preset_document_selects_the_retired_panel(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    assert RETIRED_PANEL_ID not in (raw.get("web_panels") or [])
    assert raw.get("default_panel") != RETIRED_PANEL_ID
