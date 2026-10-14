"""What holds about the pyAML release OSPREY pins.

``osprey-framework`` declares ``accelerator-middle-layer`` directly, next to
``pyaml-cs-osprey``, because the framework's own code imports pyAML.
``pyaml-cs-osprey`` declares the same package for its control-system backend.
The two specifiers must be string-equal: a drift between them would let one
package be built and tested against a pyAML the other refuses.

The release's ``MeasurementTool.save`` cannot write a measurement holding numpy
values as JSON, which is why the measurement script carries its own writer; the
last test fails on the first pinned release that can.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRAMEWORK_PYPROJECT = REPO_ROOT / "pyproject.toml"
MEMBER_PYPROJECT = REPO_ROOT / "packages" / "pyaml-cs-osprey" / "pyproject.toml"
AML_PACKAGE = "accelerator-middle-layer"


def _aml_requirement(pyproject: pathlib.Path) -> str | None:
    """Return the ``accelerator-middle-layer`` entry of ``project.dependencies``.

    ``None`` when the package is not declared; several entries come back joined,
    so the comparison below fails with every one of them in its message.
    """
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    entries = [
        entry
        for entry in dependencies
        if canonicalize_name(Requirement(entry).name) == canonicalize_name(AML_PACKAGE)
    ]
    return "; ".join(entries) if entries else None


def test_framework_pin_equals_member_specifier() -> None:
    framework = _aml_requirement(FRAMEWORK_PYPROJECT)
    member = _aml_requirement(MEMBER_PYPROJECT)
    assert member is not None, (
        f"packages/pyaml-cs-osprey/pyproject.toml does not declare {AML_PACKAGE}; "
        f"framework pyproject.toml has {framework!r}"
    )
    assert framework == member, (
        f"{AML_PACKAGE} pins differ: framework pyproject.toml has {framework!r}, "
        f"packages/pyaml-cs-osprey/pyproject.toml has {member!r}"
    )


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_pyaml_save_still_refuses_numpy_values(tmp_path: pathlib.Path) -> None:
    import numpy as np
    from pyaml.tuning_tools.measurement_tool import MeasurementTool

    class Tool(MeasurementTool):
        def measure(self) -> bool:
            self._init_measure()
            self.latest_measurement.update({"fit": np.array([1.5, -2.5]), "n_step": np.int64(5)})
            return True

    tool = Tool("canary")
    tool.measure()
    try:
        tool.save(tmp_path / "measurement.json", with_type="json")
    except TypeError as exc:
        assert "not JSON serializable" in str(exc), (
            f"the pinned pyAML's save raised a TypeError that is not the JSON gap: {exc}"
        )
        return
    pytest.fail(
        "the pinned pyAML saves a measurement holding numpy arrays and scalars as JSON: the "
        "measurement script calls tool.save(path, with_type='json'), and plain() and save() in "
        "osprey/mcp_server/python_executor/tools/_pyaml_measure_script.py go with "
        "test_a_measurement_holding_arrays_is_saved_as_json_lists, this test and its row in "
        "tests/infrastructure/test_stand_in_canaries.py"
    )
