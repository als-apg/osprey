"""The wiring-keyed pyat model serves the same physics as the kind-keyed one.

``PyATLatticeModel``, built by the pyat engine from the SR deck and the SR
model's wiring, agrees with ``PyATRingModel``, built from the served bindings
document, at 1e-9 on every address that document binds and on the two
transverse tunes -- at the default operating point, after one corrector write
and after one quadrupole write.

The bindings document names each magnet's readback beside its setpoint under
an ``identity`` readback rule, and ``PyATRingModel`` declares no variable for
it: its oracle is the setpoint value that model holds. The wiring also carries
RF, tune and chromaticity addresses the bindings document does not bind; the
chromaticity and the RF frequency are held to ``at.get_optics`` and to the
cavity in ``test_pyat_build.py``, not here.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("at")

import numpy as np

from osprey.facility.views.simulator import simulator_wiring
from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.model.pyat import (
    OPTICS_NAMES,
    PyATRingModel,
)
from osprey.simulation.engines import pyat as engine
from osprey.simulation.engines.pyat_model import TUNES

MODEL = "SR"
CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def oracle() -> dict[str, str]:
    """Each address the bindings document binds -> the old model's variable for it.

    A setpoint or a monitor reading is its own variable; a readback is the
    setpoint its ``identity`` rule follows.
    """
    document = load_bindings(PACKAGE_PATHS.va_bindings)
    sources: dict[str, str] = {}
    for binding in document.bindings:
        sources[binding.setpoint_address] = binding.setpoint_address
        if binding.readback_address is not None:
            assert binding.readback == "identity", binding.readback_address
            sources[binding.readback_address] = binding.setpoint_address
    return dict(sorted(sources.items()))


@pytest.fixture(scope="module")
def models(built_control_assistant: Any) -> tuple[PyATRingModel, Any]:
    """The kind-keyed model over the packaged tree and the wiring-keyed one over SR."""
    facility = built_control_assistant.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    new = engine.build(
        MODEL,
        simulator_wiring(facility, MODEL),
        built_control_assistant.facility_dir / model["deck"],
        model.get("settings"),
        {},
    )
    old = PyATRingModel(PACKAGE_PATHS.data_root, build_manifest()["channels"])
    return old, new


def default_of(old: PyATRingModel, address: str) -> float:
    return float(old.supported_variables[address].default_value)


OPERATING_POINTS = {
    "default": lambda old: {},
    "corrector": lambda old: {CORRECTOR: 3.25},
    "quadrupole": lambda old: {QUADRUPOLE: default_of(old, QUADRUPOLE) * 1.01},
}


@pytest.fixture(params=sorted(OPERATING_POINTS))
def operating_point(request, models) -> tuple[PyATRingModel, Any]:
    """Both models reset to their defaults, then given the same single write."""
    old, new = models
    old.reset()
    new.reset()
    write = OPERATING_POINTS[request.param](old)
    if write:
        old.set(write)
        new.set(write)
    return old, new


def test_the_bindings_document_binds_840_addresses(oracle, models):
    old, _ = models
    assert len(oracle) == 840
    assert set(oracle.values()) <= set(old.supported_variables)
    assert TUNES in OPTICS_NAMES


def test_every_bound_address_agrees_at_1e_9(oracle, operating_point):
    old, new = operating_point
    expected = old.get(sorted(set(oracle.values())))
    served = new.get(list(oracle))
    np.testing.assert_allclose(
        [served[address] for address in oracle],
        [expected[source] for source in oracle.values()],
        rtol=0,
        atol=TOLERANCE,
    )


def test_the_two_transverse_tunes_agree_at_1e_9(operating_point):
    old, new = operating_point
    expected = old.get([TUNES])[TUNES]
    served = new.get([TUNES])[TUNES]
    assert expected.shape == (2,)
    assert served.shape == (3,)
    np.testing.assert_allclose(served[:2], expected, rtol=0, atol=TOLERANCE)


def test_each_write_moves_the_orbit(models):
    old, _ = models
    document = load_bindings(PACKAGE_PATHS.va_bindings)
    readings = [b.setpoint_address for b in document.bindings if b.kind == "monitor"]
    old.reset()
    nominal = old.get(readings)
    for write in (OPERATING_POINTS["corrector"](old), OPERATING_POINTS["quadrupole"](old)):
        old.reset()
        old.set(write)
        moved = old.get(readings)
        assert max(abs(moved[name] - nominal[name]) for name in readings) > TOLERANCE
    old.reset()
