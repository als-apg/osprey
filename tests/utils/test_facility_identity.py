"""The facility identity a runtime reader takes from a render."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.utils.facility import FacilityFileError, facility_identity


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


def test_an_absent_file_falls_back_to_the_project_name(tmp_path: Path):
    assert not (tmp_path / "facility.json").exists()

    assert facility_identity(tmp_path, "demo") == {
        "code": "demo",
        "name": "demo",
        "description": None,
    }


def test_an_unreadable_file_is_refused(tmp_path: Path):
    (tmp_path / "facility.json").mkdir()

    with pytest.raises(FacilityFileError, match=r"facility\.json.*cannot be read"):
        facility_identity(tmp_path, "demo")


def test_a_file_that_is_not_json_is_refused(tmp_path: Path):
    (tmp_path / "facility.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(FacilityFileError, match=r"facility\.json.*not JSON"):
        facility_identity(tmp_path, "demo")


def test_a_file_without_an_identity_code_is_refused(tmp_path: Path):
    _write_facility_file(tmp_path, {"name": "Demo Lab"})

    with pytest.raises(FacilityFileError, match=r"facility\.json.*names no identity code"):
        facility_identity(tmp_path, "demo")


def test_a_file_that_is_not_an_object_is_refused(tmp_path: Path):
    (tmp_path / "facility.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FacilityFileError, match=r"facility\.json.*names no identity code"):
        facility_identity(tmp_path, "demo")


def test_reading_the_identity_leaves_the_facility_build_unimported(tmp_path: Path):
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from osprey.utils.facility import facility_identity\n"
        f"facility_identity(Path({str(tmp_path)!r}), 'demo')\n"
        "print('osprey.facility.build' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
