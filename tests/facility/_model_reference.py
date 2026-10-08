"""The Middle Layer's own model answers, as the parity tests read them.

A 2.1 export writes a sixth file, ``<stem>.model.json``: what the Middle
Layer's model mode answered on the deck the export saved. It is kept for
verification only; nothing in a deployment reads it. Every parity test reads it
through this module, from ``tests/fixtures/mml/<tree>/``: a missing file fails
naming its path, and a section's refusal skips naming the Middle Layer's
message, the same way everywhere.

The parity tests make two checks, and their tolerances live here, one constant
each, so two tests comparing the same quantity cannot drift apart:

* **Check A, import fidelity.** pyAT replays the Middle Layer's exact
  model-mode recipe (``_mml_recipes``) on the deck and wiring the build wrote
  and meets the model file. Every tolerance is the numerical floor of that
  replay -- rounding, or a solver's convergence threshold divided by the step
  it differences over -- never a difference of method.
* **Check B, the measurement tool.** A measurement on the virtual accelerator
  meets pyAT's exact physics on the same lattice at the same operating point,
  taken at the tool's own steps. Its bands live beside its tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.mml._trees import exports_taking

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mml"

#: The periodic lattices whose model file a real Middle Layer wrote, as
#: ``(tree, stem)``: the supported exports that take check A.
MATLAB_RINGS = exports_taking("A", lines=False)

#: The transport lines whose model file a real Middle Layer wrote.
MATLAB_LINES = exports_taking("A", lines=True)

#: The synthetic machine: a made-up one whose model file no Middle Layer wrote,
#: so it takes check B only.
(SYNTHETIC,) = exports_taking("B", lines=False)

#: What brings a facility model file into the tree. It is the owner's step: no
#: test run writes a facility export file.
OWNER_STEP = (
    "owner step: run mml_export 2.1.0 once per sub-machine on the export host "
    "and commit all six files of each run"
)

# ---------------------------------------------------------------------------
# Check A tolerances
# ---------------------------------------------------------------------------

#: The circumference is a sum of element lengths; Matlab and numpy add them in
#: different orders, which leaves a few units in the last place (2e-14 seen).
CIRCUMFERENCE_RTOL = 1e-13

#: ``mcf`` differences the path length of two tracked particles; the port
#: repeats every step, so only the rounding of that difference is left (2e-14
#: seen).
MCF_RTOL = 1e-12

#: A tune rests on its closed orbit, which both sides solve only to the Newton
#: threshold ``ORBIT_CONVERGENCE``; through ``findm66`` that leaves about 1e-12
#: in a 6D tune (1.3e-12 seen), and far less in a 4D one (5e-15 seen).
TUNE_ATOL = 1e-11

#: Two tunes of one lattice a small step apart share their orbit and their
#: rounding, so their difference carries only the last few units of each
#: (1.2e-13 seen through ``findm66``, 1e-14 through ``findm44``). A chromaticity is
#: such a difference over a momentum step ``dp`` and is held to
#: ``2 * TUNE_DIFFERENCE_ATOL / dp``; a family response is a difference of two
#: of those, or of two tunes, over the family's step.
TUNE_DIFFERENCE_ATOL = 5e-13

#: The closed orbit at a monitor, in metres: each side stops its Newton
#: iteration once the update is below 1e-12, so the orbits may differ by that
#: much times the few iterations' residual gain (5e-13 seen).
ORBIT_ATOL_M = 1e-11

#: The dispersion is a difference of two closed orbits over the momentum span
#: between them, each orbit good to the Newton threshold; the band is
#: ``DISPERSION_ORBITS * ORBIT_CONVERGENCE / dp_span`` metres.
DISPERSION_ORBITS = 2.0

#: A transport line's dispersion is one tracked orbit over ``twissline``'s 1e-8
#: momentum step with no solver; rounding alone separates the two (7e-15 seen).
TRANSPORT_DISPERSION_ATOL_M = 1e-12

#: The orbit response, per entry, as a fraction of the matrix rms. The Linear
#: calculator is closed-form 4x4 algebra over ``findm44``'s matrices and the
#: transport calculator one tracked orbit per arm, so rounding through a few
#: thousand elements is all that separates the two (3e-13 seen).
ORM_RMS_FRACTION = 1e-10

#: The model file's hardware and physics answers to one measurement, related by
#: conversions it records at full precision, agree to the digits it writes.
CONVERSION_RTOL = 1e-9

#: Below this many metres a closed-orbit reading is rounding, not a position.
CONVERSION_ATOL_M = 1e-15


def model_path(tree: str, stem: str, *, root: Path = FIXTURES) -> Path:
    """The model file a tree commits for one stem."""
    return root / tree / f"{stem}.model.json"


def model_reference(tree: str, stem: str, *, root: Path = FIXTURES) -> dict[str, Any]:
    """Load the committed model file of one stem, or fail naming its path.

    Args:
        tree: The fixture directory under ``root``.
        stem: The export stem, ``<machine>.<submachine>``.
        root: The fixture root; the committed tree unless a test builds its own.

    Returns:
        The decoded model file.
    """
    path = model_path(tree, stem, root=root)
    if not path.is_file():
        pytest.fail(f"{tree}/{path.name} is not committed ({path}); {OWNER_STEP}")
    return json.loads(path.read_text(encoding="utf-8"))


def refusal(block: object) -> str | None:
    """The Middle Layer's message when ``block`` is a refusal, else ``None``."""
    if isinstance(block, dict) and set(block) == {"refused"}:
        return str(block["refused"])
    return None


def section(reference: dict[str, Any], name: str) -> dict[str, Any]:
    """One section of a model file, or skip naming the Middle Layer's refusal.

    Args:
        reference: A model file from :func:`model_reference`.
        name: One of the contract's section keys.

    Returns:
        The section's answer.
    """
    block = reference[name]
    message = refusal(block)
    if message is not None:
        pytest.skip(f"the model file refused {name}: {message}")
    return block


def refused_entries(reference: dict[str, Any]) -> dict[str, str]:
    """Every refusal anywhere in a model file, by dotted path, with its message.

    A whole section refused is its own name; a refusal inside a section (a
    transport line's ``state.mcf``, a cavity-less periodic deck's
    ``chromaticity.hardware``) is its dotted path.
    """
    found: dict[str, str] = {}

    def walk(node: object, where: str) -> None:
        message = refusal(node)
        if message is not None:
            found[where] = message
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")

    for name, block in reference.items():
        if name != "_export":
            walk(block, name)
    return found
