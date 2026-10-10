"""The dev-wheel requirements manifest leaves workspace members to the wheel layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment import wheel_build as wb


@pytest.mark.parametrize(
    "line",
    [
        "pyaml-cs-osprey",
        "pyaml_cs_osprey>=0.1",
        "Pyaml.CS.Osprey",
        "PYAML--CS__osprey ; python_version >= '3.11'",
        "osprey_connectors==2026.9.1",
    ],
)
def test_every_spelling_of_a_member_name_is_a_workspace_requirement(line):
    assert wb._is_workspace_requirement(line)


@pytest.mark.parametrize(
    "line",
    ["pyaml-cs-osprey-extras", "pyaml", "osprey-connectors2>=1", "numpy>=1.24"],
)
def test_a_name_that_only_shares_a_prefix_is_not_a_workspace_requirement(line):
    assert not wb._is_workspace_requirement(line)


def test_the_manifest_drops_member_requirements_and_keeps_the_rest(monkeypatch, tmp_path):
    base = {
        "a.whl": ["numpy>=1.24", "Pyaml.CS.Osprey>=0.1"],
        "b.whl": ["osprey_connectors", "softioc>=4.5", "numpy>=1.24"],
    }
    monkeypatch.setattr(wb, "_wheel_base_requirements", lambda wheel: base[wheel.name])

    wb._write_local_requirements_manifest([Path("a.whl"), Path("b.whl")], str(tmp_path))

    manifest = (tmp_path / wb.LOCAL_REQUIREMENTS_FILENAME).read_text()
    assert manifest == "numpy>=1.24\nsoftioc>=4.5\n"
