"""Write the synthetic MATLAB v7 ``.mat`` fixture with ``scipy.io.savemat``.

The committed ``quokka_booster.mat`` is the output of this script, so the loader
tests read a real MAT-file without MATLAB. The variables mirror what ``saveao``
leaves on disk for one invented sub-machine: an ``AO`` struct of families and an
``AD`` struct of machine scalars. The MATLAB value shapes the decoder must
handle are spelled out deliberately: char matrices (padded rows, one all-blank
row), cell arrays of strings, a single-row char matrix used as a broadcast
channel list, an empty double ``[]``, logical ``Status`` and a non-finite
``Range``. Function handles are absent because ``savemat`` cannot write them.

Run from the repository root to regenerate the fixture::

    .venv/bin/python tests/fixtures/mml/mat/build_mat.py [OUTPUT]

The MAT header embeds a creation timestamp, so a re-run changes a few header
bytes; the decoded content is identical and the fixture test compares that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

#: Default output path, beside this script.
DEFAULT_OUTPUT = Path(__file__).with_name("quokka_booster.mat")


def _char_matrix(rows: list[str]) -> np.ndarray:
    """Return a MATLAB char matrix: every row right-padded to the widest one."""
    width = max(len(row) for row in rows)
    return np.array([row.ljust(width) for row in rows])


def _cell(items: list[object]) -> np.ndarray:
    """Return a MATLAB cell array holding ``items`` in order."""
    cell = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _field(member_of: list[str], **values: object) -> dict[str, object]:
    """Return one AO field struct with the common simulator metadata."""
    return {
        "MemberOf": _cell(member_of),
        "Mode": "Simulator",
        "DataType": "Scalar",
        "Units": "Hardware",
        **values,
    }


def build_ao() -> dict[str, object]:
    """Return the ``AO`` struct of the invented booster."""
    return {
        "BPM": {
            "FamilyName": "BPM",
            "MemberOf": _cell(["BPM", "Diagnostics"]),
            "DeviceList": np.array([[1, 1], [1, 2], [2, 1]], dtype=float),
            "ElementList": np.array([1, 2, 3], dtype=float),
            "Status": np.array([True, True, False]),
            "CommonNames": _char_matrix(["bo-bpm-1", "bo-bpm-2", "bo-bpm-3"]),
            "Position": np.array([0.5, 12.0, 24.5]),
            "X": _field(
                ["BPM", "Monitor"],
                ChannelNames=_char_matrix(["BO:BPM1:X", "BO:BPM2:X", ""]),
                HWUnits="mm",
                PhysicsUnits="m",
            ),
            "Y": _field(
                ["BPM", "Monitor"],
                ChannelNames=_char_matrix(["BO:BPM1:Y", "BO:BPM2:Y", ""]),
                HWUnits=np.zeros((0, 0)),
                PhysicsUnits="m",
            ),
        },
        "HCM": {
            "FamilyName": "HCM",
            "MemberOf": _cell(["HCM", "Magnet"]),
            "DeviceList": np.array([[1, 1], [2, 1]], dtype=float),
            "CommonNames": _cell(["bo-hcm-1", "bo-hcm-2"]),
            "Setpoint": _field(
                ["HCM", "Setpoint"],
                ChannelNames=_char_matrix(["BO:HCM1:SP", "BO:HCM2:SP"]),
                HWUnits="Amps",
                PhysicsUnits="rad",
                Range=np.array([-np.inf, np.inf]),
            ),
            "Monitor": _field(
                ["HCM", "Monitor"],
                ChannelNames=_char_matrix(["BO:HCM:RB", "BO:HCM:RB"]),
                HWUnits=_cell(["Amps", "Amps"]),
                PhysicsUnits="rad",
            ),
        },
        "SF": {
            "FamilyName": "SF",
            "MemberOf": _cell(["SEXT", "Magnet"]),
            "DeviceList": np.array([[1, 1], [1, 2], [2, 1]], dtype=float),
            "Setpoint": _field(
                ["SEXT", "Setpoint"],
                ChannelNames="BO:SF:SP",
                HWUnits="Amps",
                PhysicsUnits="1/m^3",
            ),
        },
        "SEPTUM": {
            "FamilyName": "SEPTUM",
            "MemberOf": _cell(["SEPTUM", "Injection"]),
            "DeviceList": np.array([1, 1], dtype=float),
            "Monitor": _field(["SEPTUM", "Monitor"], HWUnits="kV"),
        },
    }


def build_ad() -> dict[str, object]:
    """Return the ``AD`` struct of the invented booster."""
    return {
        "Machine": "Quokka",
        "SubMachine": "BOOSTER",
        "OperationalMode": "Simulated ramp",
        "Energy": 2.4,
        "InjectionEnergy": 0.1,
        "Circumference": 75.0,
        "HarmonicNumber": 125.0,
        "MCF": 0.02,
        "ATModel": "quokka_booster_lattice",
    }


def build(path: Path = DEFAULT_OUTPUT) -> Path:
    """Write the fixture to ``path`` as a MATLAB v7 (MAT 5) file and return it."""
    savemat(path, {"AO": build_ao(), "AD": build_ad()}, format="5", oned_as="column")
    return path


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
