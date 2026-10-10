"""The tune and chromaticity response matrices of a periodic model's design optics.

A periodic model whose measurement file allows ``trm`` gets ``trm.json`` and the
tune correction tool that loads it: column ``j`` is ``(+beta_x, -beta_y) / 4pi``
of the quadrupole, averaged over its slices by weight times length. A matrix
that cannot drive a correction (rank below two) is not written, and neither is
its tool; a single-pass model has no design optics and no matrix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from osprey.facility.views.pyaml import CONFIGURATION_FILE, TRM_FILE
from tests.facility._pyaml_trees import measured_tree, write_view

pytest.importorskip("pyaml")


def _trm(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((directory / "SR" / TRM_FILE).read_text())
    return loaded


def test_the_tune_response_is_the_slice_averaged_beta_over_4pi(tmp_path: Path) -> None:
    import at

    directory, _ = write_view(tmp_path, measured_tree())
    trm = _trm(directory)
    assert trm["type"] == "pyaml.tuning_tools.response_matrix_data"
    assert trm["observable_names"] == ["BETATRON_TUNE.x", "BETATRON_TUNE.y"]
    assert trm["variable_names"] == ["QD:SP", "QF:SP"]

    lattice = at.load_lattice(str(directory / "SR" / "lattice.json")).disable_6d(copy=True)
    _, beta, *_ = at.avlinopt(lattice, refpts=np.arange(len(lattice)))
    index = {element.FamName: i for i, element in enumerate(lattice)}
    qf = [index["QFA"], index["QFB"]]
    expected_qd = (beta[index["QD"], 0], -beta[index["QD"], 1])
    expected_qf = (beta[qf, 0].mean(), -beta[qf, 1].mean())
    matrix = np.asarray(trm["matrix"]) * 4 * math.pi
    assert matrix[:, 0] == pytest.approx(expected_qd, rel=1e-8)
    assert matrix[:, 1] == pytest.approx(expected_qf, rel=1e-8)


def test_the_tune_correction_tool_loads_the_matrix(tmp_path: Path) -> None:
    from pyaml.accelerator import Accelerator

    directory, _ = write_view(tmp_path, measured_tree())
    configuration = yaml.safe_load((directory / "SR" / CONFIGURATION_FILE).read_text())
    (tool,) = [d for d in configuration["devices"] if d["name"] == "DEFAULT_TUNE_CORRECTION"]
    assert tool["response_matrix"] == f"${{path:{TRM_FILE}}}"
    sr = Accelerator.load(str(directory / "SR" / CONFIGURATION_FILE))
    assert sr.design.tune.response_matrix.shape == (2, 2)


def test_a_rank_one_matrix_is_not_written_and_neither_is_its_tool(tmp_path: Path) -> None:
    tree = measured_tree()
    tree["records/groups.yaml"] = [
        group if group["id"] != "SR/Q" else {"id": "SR/Q", "members": ["SR/QD"]}
        for group in tree["records/groups.yaml"]
    ]
    directory, written = write_view(tmp_path, tree)
    assert directory / "SR" / TRM_FILE not in written
    configuration = yaml.safe_load((directory / "SR" / CONFIGURATION_FILE).read_text())
    names = [device["name"] for device in configuration["devices"]]
    assert "DEFAULT_TUNE_CORRECTION" not in names
    assert "DEFAULT_TUNE_RESPONSE_MATRIX" in names


def test_a_single_pass_model_has_no_matrix(tmp_path: Path) -> None:
    _, written = write_view(tmp_path, measured_tree())
    assert {path.name for path in written if path.parent.name == "LINE"} == {CONFIGURATION_FILE}
