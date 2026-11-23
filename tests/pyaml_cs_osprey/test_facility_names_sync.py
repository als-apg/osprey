"""The facility description's pyAML words stay equal to what ``pyaml_cs_osprey`` offers."""

from __future__ import annotations

from osprey.facility import kinds
from pyaml_cs_osprey.measure import StepMeasurement
from pyaml_cs_osprey.names import CHROMATICITY_MONITOR_NAME


def test_step_factories_are_every_public_callable_on_step_measurement() -> None:
    offered = {
        name
        for name in dir(StepMeasurement)
        if not name.startswith("_") and callable(getattr(StepMeasurement, name))
    }
    assert kinds.STEP_FACTORIES == offered


def test_the_built_in_factories_are_offered() -> None:
    assert {"orm", "dispersion", "trm", "crm", "measure"} <= kinds.STEP_FACTORIES


def test_chromaticity_monitor_name_equals_the_pyaml_constant() -> None:
    assert kinds.CHROMATICITY_MONITOR_NAME == CHROMATICITY_MONITOR_NAME
