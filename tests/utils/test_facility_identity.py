"""The facility identity a runtime reader takes from a render."""

from __future__ import annotations

import json
from pathlib import Path

from osprey.utils.facility import facility_identity


def _write_facility_file(render_root: Path, identity: object) -> None:
    render_root.mkdir(parents=True, exist_ok=True)
    document = {"schema": "osprey.facility.facility/1", "identity": identity}
    (render_root / "facility.json").write_text(json.dumps(document), encoding="utf-8")


def test_the_facility_file_names_the_identity(tmp_path: Path):
    _write_facility_file(
        tmp_path, {"code": "demo", "name": "Demo Lab", "description": "A demonstration."}
    )

    assert facility_identity(tmp_path, "other-project") == {
        "code": "demo",
        "name": "Demo Lab",
        "description": "A demonstration.",
    }


def test_a_file_identity_without_a_name_takes_the_project_name(tmp_path: Path):
    _write_facility_file(tmp_path, {"code": "demo"})

    assert facility_identity(tmp_path, "My Project") == {
        "code": "demo",
        "name": "My Project",
        "description": None,
    }


def test_a_file_identity_without_a_name_or_project_name_takes_its_code(tmp_path: Path):
    _write_facility_file(tmp_path, {"code": "demo"})

    assert facility_identity(tmp_path) == {"code": "demo", "name": "demo", "description": None}


def test_no_file_and_a_project_name_is_the_folded_identity(tmp_path: Path):
    identity = facility_identity(tmp_path, "1st-lab")

    assert identity == {"code": "x1st_lab", "name": "1st-lab", "description": None}
    assert identity is not None
    assert identity["code"] == "x1st_lab"


def test_no_file_and_no_project_name_is_no_identity(tmp_path: Path):
    assert facility_identity(tmp_path) is None
    assert facility_identity(tmp_path, None) is None
    assert facility_identity(tmp_path, "") is None


def test_an_unreadable_file_falls_back_to_the_project_name(tmp_path: Path):
    (tmp_path / "facility.json").write_text("{not json", encoding="utf-8")

    assert facility_identity(tmp_path, "demo") == {
        "code": "demo",
        "name": "demo",
        "description": None,
    }


def test_a_file_without_an_identity_code_falls_back_to_the_project_name(tmp_path: Path):
    _write_facility_file(tmp_path, {"name": "Demo Lab"})

    assert facility_identity(tmp_path, "demo") == {
        "code": "demo",
        "name": "demo",
        "description": None,
    }
