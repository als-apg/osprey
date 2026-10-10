"""A small facility with two measured models, and its pyAML view written in memory.

``measured_tree`` is a ``data/facility`` tree, in the shape
``tests/facility/_synthetic_trees.py`` writes, holding:

* model ``SR`` on a periodic deck: the split quadrupole ``SR/QF`` (slices QFA,
  QFB), the quadrupole ``SR/QD``, the BPM ``SR/BPM1`` read in both planes and the
  tune outputs ``SR:TUNE:X``/``Y``; its measurement file allows ``trm``;
* model ``LINE`` on a single-pass deck: the quadrupole ``LINE/Q1``, the BPM
  ``LINE/BPM1`` and the corrector ``LINE/COR1`` steered in both planes from two
  setpoints; its measurement file allows ``orm``.

:func:`with_rf`, :func:`with_correctors` and :func:`with_chromaticity` wire
more of SR's deck into the tree. :func:`write_view` builds a tree in memory and
writes its pyAML view.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.facility._synthetic_trees import Deck, write_tree

__all__ = [
    "LINE_GROUPS",
    "SR_GROUPS",
    "built_document",
    "measured_tree",
    "view_inputs",
    "with_chromaticity",
    "with_correctors",
    "with_rf",
    "write_view",
]

#: The group ids each model's measurement file names.
SR_GROUPS = {"bpm": "SR/BPM", "quad": "SR/Q"}
LINE_GROUPS = {"bpm": "LINE/BPM", "hcor": "LINE/HCM", "vcor": "LINE/VCM", "quad": "LINE/Q"}


def _sr_deck() -> Deck:
    """One stable bending cell: QFA, B1, QD, SX, BPM1, the corrector SCOR, B2, QFB, CAV."""
    return Deck(
        lambda at: [
            at.Marker("M1"),
            at.Quadrupole("QFA", 0.25, 1.0),
            at.Drift("D", 0.5),
            at.Dipole("B1", 1.0, math.pi / 8),
            at.Drift("D", 0.2),
            at.Quadrupole("QD", 0.5, -1.0),
            at.Drift("D", 0.2),
            at.Sextupole("SX", 0.1, 1.0),
            at.Monitor("BPM1"),
            at.Corrector("SCOR", 0.0, [0.0, 0.0]),
            at.Dipole("B2", 1.0, math.pi / 8),
            at.Drift("D", 0.5),
            at.Quadrupole("QFB", 0.25, 1.0),
            at.RFCavity("CAV", 0.0, 1.0e6, 500.0e6, 10, 3.0e9),
        ]
    )


def _line_deck() -> Deck:
    """M0, Q1 at 1.0-1.2, COR1 at 1.7, LBPM at 2.2."""
    return Deck(
        lambda at: [
            at.Marker("M0"),
            at.Drift("DL", 1.0),
            at.Quadrupole("Q1", 0.2, 0.9),
            at.Drift("DL2", 0.5),
            at.Corrector("COR1", 0.0, [0.0, 0.0]),
            at.Drift("DL3", 0.5),
            at.Monitor("LBPM"),
        ]
    )


def _setting(address: str, attribute: str, index: int, **where: Any) -> dict[str, Any]:
    return {
        "address": address,
        **where,
        "engine": {"attribute": attribute, "index": index},
        "calibration": {"curve": {"linear": {"gain": 0.01, "offset": 0.0}}},
    }


def _monitor(address: str, element: str, axis: str) -> dict[str, Any]:
    return {"address": address, "element": element, "engine": {"axis": axis}}


def measured_tree() -> dict[str, Any]:
    """The two measured models; every call returns a fresh tree a test may edit."""
    return {
        "records/places.yaml": [
            {"id": "SR", "span": {"model": "SR", "from_marker": "M1"}},
            {"id": "LINE", "span": {"model": "LINE", "from_marker": "M0"}},
        ],
        "records/devices.yaml": [
            {"id": "SR/QF", "class": "Quadrupole"},
            {"id": "SR/QD", "class": "Quadrupole"},
            {"id": "SR/BPM1", "class": "BeamPositionMonitor"},
            {"id": "LINE/Q1", "class": "Quadrupole"},
            {"id": "LINE/BPM1", "class": "BeamPositionMonitor"},
            {"id": "LINE/COR1", "class": "HCorrector"},
        ],
        "records/channels.yaml": [
            {"id": "QF:SP", "role": "setpoint", "pair": "QF:RB", "on": {"device": "SR/QF"}},
            {"id": "QF:RB", "on": {"device": "SR/QF"}},
            {"id": "QD:SP", "role": "setpoint", "on": {"device": "SR/QD"}},
            {"id": "BPM1:X", "unit": "mm", "on": {"device": "SR/BPM1"}},
            {"id": "BPM1:Y", "unit": "mm", "on": {"device": "SR/BPM1"}},
            {"id": "SR:TUNE:X", "on": {"place": "SR"}},
            {"id": "SR:TUNE:Y", "on": {"place": "SR"}},
            {"id": "LQ:SP", "role": "setpoint", "on": {"device": "LINE/Q1"}},
            {"id": "LBPM:X", "unit": "mm", "on": {"device": "LINE/BPM1"}},
            {"id": "LBPM:Y", "unit": "mm", "on": {"device": "LINE/BPM1"}},
            {"id": "LCOR:H:SP", "role": "setpoint", "unit": "A", "on": {"device": "LINE/COR1"}},
            {"id": "LCOR:V:SP", "role": "setpoint", "unit": "A", "on": {"device": "LINE/COR1"}},
        ],
        "records/groups.yaml": [
            {"id": "SR/BPM", "members": ["SR/BPM1"]},
            {"id": "SR/Q", "members": ["SR/QD", "SR/QF"]},
            {"id": "LINE/BPM", "members": ["LINE/BPM1"]},
            {"id": "LINE/HCM", "members": ["LINE/COR1"]},
            {"id": "LINE/VCM", "members": ["LINE/COR1"]},
            {"id": "LINE/Q", "members": ["LINE/Q1"]},
        ],
        "models.yaml": [
            {
                "name": "SR",
                "engine": "pyat",
                "deck": "decks/sr.json",
                "wiring": [
                    {
                        **_setting("QF:SP", "PolynomB", 1),
                        "slices": [{"element": "QFA"}, {"element": "QFB"}],
                    },
                    _setting("QF:RB", "PolynomB", 1, slices=[{"element": "QFA"}]),
                    _setting("QD:SP", "PolynomB", 1, element="QD"),
                    _monitor("BPM1:X", "BPM1", "x"),
                    _monitor("BPM1:Y", "BPM1", "y"),
                    {"address": "SR:TUNE:X", "engine": {"attribute": "tune", "axis": "x"}},
                    {"address": "SR:TUNE:Y", "engine": {"attribute": "tune", "axis": "y"}},
                ],
            },
            {
                "name": "LINE",
                "engine": "pyat",
                "deck": "decks/line.json",
                "settings": {
                    "pyat": {
                        "solve": "single_pass",
                        "twiss_in": {"beta": [5.0, 3.0], "alpha": [0.0, 0.0]},
                    }
                },
                "wiring": [
                    _setting("LQ:SP", "PolynomB", 1, element="Q1"),
                    _setting("LCOR:H:SP", "KickAngle", 0, element="COR1"),
                    _setting("LCOR:V:SP", "KickAngle", 1, element="COR1"),
                    _monitor("LBPM:X", "LBPM", "x"),
                    _monitor("LBPM:Y", "LBPM", "y"),
                ],
            },
        ],
        "measurement/SR.yaml": {
            "kinds": ["trm"],
            "groups": dict(SR_GROUPS),
            "instruments": {"tune": "SR:TUNE:X"},
            "quad_delta": 0.001,
            "n_step": 3,
        },
        "measurement/LINE.yaml": {
            "kinds": ["orm"],
            "groups": dict(LINE_GROUPS),
            "corrector_delta": 1.0e-5,
        },
        "decks/sr.json": _sr_deck(),
        "decks/line.json": _line_deck(),
    }


def with_rf(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with SR's cavity frequency ``RF:SP`` wired and named ``instruments.rf``."""
    tree["records/devices.yaml"].append({"id": "SR/CAV", "class": "AcceleratingCavity"})
    tree["records/channels.yaml"].append(
        {"id": "RF:SP", "role": "setpoint", "unit": "MHz", "on": {"device": "SR/CAV"}}
    )
    _sr(tree)["wiring"].append(
        {
            "address": "RF:SP",
            "element": "CAV",
            "engine": {"attribute": "Frequency"},
            "calibration": {"curve": {"linear": {"gain": 1.0e6, "offset": 0.0}}},
        }
    )
    tree["measurement/SR.yaml"]["instruments"]["rf"] = "RF:SP"
    return tree


def with_correctors(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with SR's corrector SCOR steered in both planes, its groups named."""
    tree["records/devices.yaml"].append({"id": "SR/COR1", "class": "HCorrector"})
    tree["records/channels.yaml"] += [
        {"id": "SCOR:H:SP", "role": "setpoint", "unit": "A", "on": {"device": "SR/COR1"}},
        {"id": "SCOR:V:SP", "role": "setpoint", "unit": "A", "on": {"device": "SR/COR1"}},
    ]
    tree["records/groups.yaml"] += [
        {"id": "SR/HCM", "members": ["SR/COR1"]},
        {"id": "SR/VCM", "members": ["SR/COR1"]},
    ]
    _sr(tree)["wiring"] += [
        _setting("SCOR:H:SP", "KickAngle", 0, element="SCOR"),
        _setting("SCOR:V:SP", "KickAngle", 1, element="SCOR"),
    ]
    tree["measurement/SR.yaml"]["groups"] |= {"hcor": "SR/HCM", "vcor": "SR/VCM"}
    return tree


def with_chromaticity(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with SR's chromaticity outputs wired and named ``instruments.chromaticity``."""
    tree["records/channels.yaml"] += [
        {"id": "SR:CHROM:X", "on": {"place": "SR"}},
        {"id": "SR:CHROM:Y", "on": {"place": "SR"}},
    ]
    _sr(tree)["wiring"] += [
        {"address": "SR:CHROM:X", "engine": {"attribute": "chromaticity", "axis": "x"}},
        {"address": "SR:CHROM:Y", "engine": {"attribute": "chromaticity", "axis": "y"}},
    ]
    tree["measurement/SR.yaml"]["instruments"]["chromaticity"] = "SR:CHROM:X"
    return tree


def _sr(tree: dict[str, Any]) -> dict[str, Any]:
    (record,) = [model for model in tree["models.yaml"] if model["name"] == "SR"]
    sr: dict[str, Any] = record
    return sr


def built_document(root: Path, tree: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """Write ``tree`` under ``root/data/facility`` and build its facility file in memory."""
    from osprey.facility.build import build_facility

    facility = write_tree(root / "data" / "facility", copy.deepcopy(dict(tree)))
    return build_facility(facility, project_name="demo"), facility


def view_inputs(root: Path, tree: Mapping[str, Any], served: list[str] | None = None) -> Any:
    """The view inputs of a render serving ``served`` (every model when ``None``)."""
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs

    document, facility = built_document(root, tree)
    config: dict[str, Any] = {} if served is None else {"simulation": {"models": served}}
    return ViewInputs(
        doc=document,
        rendered_config=config,
        facility_dir=facility,
        served=resolve_served(config, document),
    )


def write_view(
    root: Path, tree: Mapping[str, Any], served: list[str] | None = None
) -> tuple[Path, list[Path]]:
    """Build ``tree`` and write its pyAML view under ``root/render/data/pyaml``.

    Returns:
        The view's directory and the files written.
    """
    from osprey.facility.views.pyaml import write_pyaml_view

    inputs = view_inputs(root, tree, served)
    directory = root / "render" / "data" / "pyaml"
    return directory, write_pyaml_view(directory, inputs)
