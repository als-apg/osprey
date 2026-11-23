"""The bundled example facility raises no place warning from the relation stage.

The rest of the relation stage's warnings are description tests in the facility
package's suite (``osprey_facility_tests/test_relation_warnings.py``). This one
stays here: the example's models run on the framework's ``pyat`` engine, which
only the framework registers.
"""

from __future__ import annotations

from pathlib import Path

from osprey_facility.build import LATER_STAGES
from osprey_facility.validate import run_stages
from tests._builds import FRAMEWORK_CHECKS

EXAMPLE = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"


def test_the_bundled_example_raises_no_place_warning() -> None:
    report = run_stages(
        EXAMPLE, project_name="demo", later=LATER_STAGES, extra_checks=FRAMEWORK_CHECKS
    )
    assert report.failed is None, [e.format_message() for e in report.errors]
    assert [w.line for w in report.validated.warnings if w.kind.startswith("place-")] == []
