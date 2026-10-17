"""The two first-turn response files of a model's pyAML view.

A model whose measurement file allows ``trajectory`` gets ``trajectory_rm.json``
and ``trajectory_rm.index.json`` beside its configuration. The first is a plain
dict of pyAML's orbit-response-matrix data type: the matrix its engine solves,
hardware reading per hardware setpoint, a row per turn-by-turn reading and a
column per corrector, each named by its channel address, with the plane of
every row and column and no RF response. A corrector whose step has no solve
keeps its column as a column of zeros. The second holds the lattice index of
every row and of every column, and each unsolved corrector with the engine's
text. Equal sources give equal bytes. An engine that states no
``response_matrix()``, refuses the response, or solves no corrector, stops the
build.
"""

from __future__ import annotations

import json
import types
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("at")

import numpy as np

from osprey.facility.errors import FacilityBuildError
from osprey.facility.views import pyaml as pyaml_view
from osprey.facility.views.pyaml import (
    CONFIGURATION_FILE,
    TRAJECTORY_RM_FILE,
    TRAJECTORY_RM_INDEX_FILE,
    measurement_groups,
    write_pyaml_view,
)
from osprey.facility.views.simulator import simulator_wiring
from osprey.simulation.engines import ENTRY_POINT_GROUP
from osprey.simulation.engines import pyat as engine
from tests.facility._pyaml_trees import (
    measured_tree,
    view_inputs,
    with_correctors,
    write_view,
)
from tests.facility._synthetic_trees import TRAJECTORY_KEYS, model, with_trajectory

#: Hardware per metre of a turn-by-turn record, and hardware per radian of a corrector.
SERIES_GAIN = 1.0e9
KICK_GAIN = 100.0

#: Each model with the element its one monitor device reads and its series and
#: corrector addresses in the files' order.
ROWS = {
    "SR": ["SR:BPM:TBT:X", "SR:BPM:TBT:Y"],
    "LINE": ["LINE:BPM:TBT:X", "LINE:BPM:TBT:Y"],
}
COLUMNS = {"SR": ["SCOR:H:SP", "SCOR:V:SP"], "LINE": ["LCOR:H:SP", "LCOR:V:SP"]}
#: Each model's one sum series, the sum reading of both its rows.
SUMS = {"SR": "SR:BPM:TBT:SUM", "LINE": "LINE:BPM:TBT:SUM"}
MODELS = ("SR", "LINE")

#: The sidecar's lists of lattice indices.
INDEX_LISTS = ("bpm_index", "hcorr_index", "vcorr_index")


def _with_series(tree: dict[str, Any], name: str, element: str) -> dict[str, Any]:
    """``tree`` with model ``name`` wiring its monitor's five signal channels on ``element``."""
    model(tree, name)["wiring"] += [
        {"address": f"{name}:BPM:ARM", "engine": {"attribute": "arm"}},
        {"address": f"{name}:BPM:ARMED", "engine": {"attribute": "armed"}},
    ]
    for axis in ("sum", "y", "x"):
        record: dict[str, Any] = {
            "address": f"{name}:BPM:TBT:{axis.upper()}",
            "element": element,
            "engine": {"axis": axis, "series": "turn"},
        }
        if axis != "sum":
            record["calibration"] = {
                "curve": {"linear": {"gain": 1.0 / SERIES_GAIN, "offset": 0.0}}
            }
        model(tree, name)["wiring"].append(record)
    return tree


def trajectory_tree() -> dict[str, Any]:
    """Both models measuring ``trajectory``: SR launched at its marker M1, LINE a single pass."""
    tree = with_trajectory(with_correctors(measured_tree()), "SR", "SR/BPM1", marker="M1")
    tree = with_trajectory(tree, "LINE", "LINE/BPM1")
    return _with_series(_with_series(tree, "SR", "BPM1"), "LINE", "LBPM")


def _files(directory: Path, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    folder = directory / name
    matrix: dict[str, Any] = json.loads((folder / TRAJECTORY_RM_FILE).read_text("utf-8"))
    index: dict[str, Any] = json.loads((folder / TRAJECTORY_RM_INDEX_FILE).read_text("utf-8"))
    return matrix, index


@pytest.fixture(scope="module")
def view(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Path]]:
    return write_view(tmp_path_factory.mktemp("view"), trajectory_tree())


# --- the files ----------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_a_model_measuring_the_kind_gets_both_files_beside_its_configuration(
    view: tuple[Path, list[Path]], name: str
) -> None:
    directory, written = view
    for file in (CONFIGURATION_FILE, TRAJECTORY_RM_FILE, TRAJECTORY_RM_INDEX_FILE):
        assert directory / name / file in written
        assert (directory / name / file).is_file()


def test_the_file_names_are_the_ones_the_readers_open() -> None:
    assert TRAJECTORY_RM_FILE == "trajectory_rm.json"
    assert TRAJECTORY_RM_INDEX_FILE == "trajectory_rm.index.json"


def test_a_model_not_measuring_the_kind_gets_neither_file(tmp_path: Path) -> None:
    directory, written = write_view(tmp_path, with_correctors(measured_tree()))
    assert written
    assert not list(directory.rglob("trajectory_rm*"))


def test_two_builds_of_equal_sources_write_equal_bytes(
    view: tuple[Path, list[Path]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = view
    # Solved afresh, so the bytes compared are two solves', not one solve's twice.
    monkeypatch.setattr(pyaml_view, "_SOLVED", {})
    second, _ = write_view(tmp_path, trajectory_tree())
    for name in MODELS:
        for file in (TRAJECTORY_RM_FILE, TRAJECTORY_RM_INDEX_FILE):
            assert (first / name / file).read_bytes() == (second / name / file).read_bytes()


@pytest.mark.parametrize("name", MODELS)
def test_the_files_are_indented_json_text_ending_in_one_newline(
    view: tuple[Path, list[Path]], name: str
) -> None:
    directory, _ = view
    for file in (TRAJECTORY_RM_FILE, TRAJECTORY_RM_INDEX_FILE):
        text = (directory / name / file).read_text("utf-8")
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"


# --- the matrix dict ----------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_the_matrix_file_is_a_plain_orbit_response_dict_with_planes_and_no_rf_response(
    view: tuple[Path, list[Path]], name: str
) -> None:
    matrix, _ = _files(view[0], name)
    assert list(matrix) == [
        "type",
        "matrix",
        "observable_names",
        "variable_names",
        "observable_planes",
        "variable_planes",
    ]
    assert matrix["type"] == "pyaml.tuning_tools.orbit_response_matrix_data"
    assert matrix["observable_names"] == ROWS[name]
    assert matrix["variable_names"] == COLUMNS[name]
    assert matrix["observable_planes"] == ["H", "V"]
    assert matrix["variable_planes"] == ["H", "V"]


@pytest.mark.parametrize("name", MODELS)
def test_every_name_is_a_channel_address_the_model_wires(
    view: tuple[Path, list[Path]], tmp_path: Path, name: str
) -> None:
    matrix, _ = _files(view[0], name)
    document = view_inputs(tmp_path, trajectory_tree()).doc
    channels = {str(channel["id"]) for channel in document["channels"]}
    wired = {str(entry["address"]): entry for entry in simulator_wiring(dict(document), name)}
    names = [*matrix["observable_names"], *matrix["variable_names"]]
    assert len(set(names)) == len(names)
    assert set(names) <= channels & set(wired)
    assert {wired[row]["role"] for row in matrix["observable_names"]} == {"turn_series"}
    groups = measurement_groups(document, name)
    assert matrix["variable_names"] == [*groups["hcor"], *groups["vcor"]]


@pytest.mark.parametrize("name", MODELS)
def test_the_index_lengths_match_the_rows_and_the_columns(
    view: tuple[Path, list[Path]], name: str
) -> None:
    matrix, index = _files(view[0], name)
    assert list(index) == ["bpm_index", "sum_names", "hcorr_index", "vcorr_index", "unsolved"]
    assert index["unsolved"] == {}
    assert all(type(entry) is int for key in INDEX_LISTS for entry in index[key])
    assert len(index["bpm_index"]) == len(matrix["matrix"]) == len(matrix["observable_names"])
    columns = len(index["hcorr_index"]) + len(index["vcorr_index"])
    assert columns == len(matrix["variable_names"])
    assert all(len(row) == columns for row in matrix["matrix"])
    assert matrix["variable_planes"] == ["H"] * len(index["hcorr_index"]) + ["V"] * len(
        index["vcorr_index"]
    )


@pytest.mark.parametrize("name", MODELS)
def test_the_matrix_is_the_engines_first_turn_response_at_the_stated_kick_step(
    view: tuple[Path, list[Path]], tmp_path: Path, name: str
) -> None:
    """Hardware reading per hardware setpoint, each corrector stepped by ``corrector_delta``."""
    matrix, index = _files(view[0], name)
    inputs = view_inputs(tmp_path, trajectory_tree())
    (record,) = [entry for entry in inputs.doc["models"] if entry["name"] == name]
    step = TRAJECTORY_KEYS["corrector_delta"] * KICK_GAIN
    answer = engine.response_matrix(
        inputs.facility_dir / record["deck"],
        simulator_wiring(dict(inputs.doc), name),
        record.get("settings"),
        COLUMNS[name],
        dict.fromkeys(COLUMNS[name], step),
        kind="trajectory",
    )
    assert answer.rows == matrix["observable_names"]
    assert answer.columns == matrix["variable_names"]
    solved = np.asarray(answer.matrix, dtype=float)
    written = np.asarray(matrix["matrix"], dtype=float)
    assert written.shape == solved.shape == (2, 2)
    np.testing.assert_allclose(written, solved, rtol=1.0e-8, atol=0.0)
    if name == "LINE":
        assert np.all(np.abs(np.diag(written)) > 0.0)
    else:
        # The particle launched at M1 meets SR's one monitor before its one corrector.
        assert not written.any()
    assert answer.unsolved == {}
    assert answer.index is not None
    horizontal, vertical = COLUMNS[name]
    assert index == {
        "bpm_index": [answer.index[row] for row in answer.rows],
        "sum_names": [SUMS[name]] * 2,
        "hcorr_index": [answer.index[horizontal]],
        "vcorr_index": [answer.index[vertical]],
        "unsolved": {},
    }


def test_the_matrix_values_are_rounded_to_nine_significant_digits(
    view: tuple[Path, list[Path]],
) -> None:
    for name in MODELS:
        matrix, _ = _files(view[0], name)
        for row in matrix["matrix"]:
            assert row == [float(f"{value:.9g}") for value in row]


def test_a_corrector_the_view_gives_no_magnet_is_still_a_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The columns are the measurement groups' correctors, whatever the view leaves out."""
    import yaml

    from osprey.facility.views import pyaml as view_module

    modelled = view_module._magnet_model

    def magnet_model(magnet: Any, *args: Any) -> Any:
        return None if magnet.address == "LCOR:V:SP" else modelled(magnet, *args)

    monkeypatch.setattr(view_module, "_magnet_model", magnet_model)
    directory, _ = write_view(tmp_path, trajectory_tree())
    configuration = yaml.safe_load((directory / "LINE" / CONFIGURATION_FILE).read_text("utf-8"))
    assert "LCOR:H:SP" in json.dumps(configuration)
    assert "LCOR:V:SP" not in json.dumps(configuration)
    matrix, index = _files(directory, "LINE")
    assert matrix["variable_names"] == COLUMNS["LINE"]
    assert len(index["vcorr_index"]) == 1


@pytest.mark.parametrize("name", MODELS)
def test_the_matrix_file_loads_as_pyamls_orbit_response_matrix_data(
    view: tuple[Path, list[Path]], name: str
) -> None:
    pytest.importorskip("pyaml")
    from pyaml.tuning_tools.orbit_response_matrix_data import OrbitResponseMatrixData

    matrix, _ = _files(view[0], name)
    loaded = OrbitResponseMatrixData.load(str(view[0] / name / TRAJECTORY_RM_FILE))
    assert type(loaded) is OrbitResponseMatrixData
    assert loaded.matrix == matrix["matrix"]
    assert loaded.observable_names == ROWS[name]
    assert loaded.variable_names == COLUMNS[name]
    assert loaded.observable_planes == ["H", "V"]
    assert loaded.variable_planes == ["H", "V"]
    assert loaded.rf_response is None


# --- the build stops ----------------------------------------------------------


class _Engines:
    """The entry points of the engine group, holding one module under one name."""

    def __init__(self, name: str, module: Any) -> None:
        self.names = {name}
        self._module = module

    def __getitem__(self, name: str) -> Any:
        module = self._module

        class _EntryPoint:
            def load(self) -> Any:
                return module

        return _EntryPoint()


def _stop_line(tmp_path: Path, tree: dict[str, Any], served: list[str] | None = None) -> str:
    inputs = view_inputs(tmp_path, tree, served)
    with pytest.raises(FacilityBuildError) as stopped:
        write_pyaml_view(tmp_path / "render" / "data" / "pyaml", inputs)
    assert not list((tmp_path / "render").rglob("trajectory_rm*"))
    return str(stopped.value.format_message())


def test_an_engine_without_the_function_stops_the_build_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = view_inputs(tmp_path, trajectory_tree(), ["LINE"])
    stub = types.ModuleType("stub_engine")
    stub.describe = engine.describe  # type: ignore[attr-defined]
    found = metadata.entry_points

    def entry_points(**selection: Any) -> Any:
        if selection.get("group") == ENTRY_POINT_GROUP:
            return _Engines("pyat", stub)
        return found(**selection)

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    with pytest.raises(FacilityBuildError) as stopped:
        write_pyaml_view(tmp_path / "render" / "data" / "pyaml", inputs)
    assert str(stopped.value.format_message()) == (
        "facility: engine-invalid: model LINE — engine pyat states no response_matrix(), so "
        "model LINE's trajectory response has no solve; fix: add "
        "response_matrix(kind='trajectory') to the engine plug-in"
    )


@pytest.mark.parametrize(
    "refusal",
    [ValueError("the corrector steers neither plane"), engine.SOLVE_ERRORS[0]("no first turn")],
    ids=["a request it refuses", "a step without a solve"],
)
def test_an_engine_refusal_stops_the_build_naming_the_function_and_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: Exception
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        assert kwargs == {"kind": "trajectory"}
        raise refusal

    monkeypatch.setattr(engine, "response_matrix", refuse)
    assert _stop_line(tmp_path, trajectory_tree(), ["LINE"]) == (
        "facility: engine-invalid: model LINE — engine pyat refuses "
        f"response_matrix(kind='trajectory') for model LINE: {refusal}; fix: give the model a "
        "deck, wiring and settings its first turn solves on"
    )


def test_a_build_stop_of_the_engines_own_is_the_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = FacilityBuildError(
        "engine-invalid", "LINE", ["models.yaml"], "state it", record_kind="model", detail="own"
    )

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise own

    monkeypatch.setattr(engine, "response_matrix", refuse)
    assert _stop_line(tmp_path, trajectory_tree(), ["LINE"]) == own.format_message()


def test_a_failure_that_is_no_refusal_is_not_turned_into_a_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a defect")

    monkeypatch.setattr(engine, "response_matrix", fail)
    inputs = view_inputs(tmp_path, trajectory_tree(), ["LINE"])
    with pytest.raises(RuntimeError, match="a defect"):
        write_pyaml_view(tmp_path / "render" / "data" / "pyaml", inputs)


def test_a_render_of_inputs_already_solved_is_not_solved_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every render of one build asks for the same response; an edited deck is solved afresh."""
    solved: list[str] = []
    real = engine.response_matrix

    def counted(*args: Any, **kwargs: Any) -> Any:
        solved.append(Path(args[0]).name)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "response_matrix", counted)
    inputs = view_inputs(tmp_path, trajectory_tree(), ["LINE"])
    first = tmp_path / "first"
    write_pyaml_view(first, inputs)
    write_pyaml_view(tmp_path / "second", inputs)
    assert solved == ["line.json"]
    assert (first / "LINE" / TRAJECTORY_RM_FILE).read_bytes() == (
        tmp_path / "second" / "LINE" / TRAJECTORY_RM_FILE
    ).read_bytes()

    deck = inputs.facility_dir / "decks" / "line.json"
    deck.write_text(deck.read_text("utf-8") + "\n", encoding="utf-8")
    write_pyaml_view(tmp_path / "third", inputs)
    assert solved == ["line.json", "line.json"]


def _answer(**changed: Any) -> Any:
    answer = {
        "rows": list(ROWS["LINE"]),
        "columns": list(COLUMNS["LINE"]),
        "matrix": np.eye(2),
        "unsolved": {},
        "index": dict.fromkeys(ROWS["LINE"], 6) | dict.fromkeys(COLUMNS["LINE"], 4),
    }
    return engine.ResponseMatrix(**(answer | changed))


@pytest.mark.parametrize(
    ("changed", "what"),
    [
        ({"matrix": np.array([[1.0, float("nan")], [0.0, 1.0]])}, "a value that is not finite"),
        ({"matrix": np.eye(3)}, "a matrix that is not 2 rows by 2 columns"),
        (
            {"index": {"LINE:BPM:TBT:X": 6, **dict.fromkeys(COLUMNS["LINE"], 4)}},
            "no index for LINE:BPM:TBT:Y",
        ),
        ({"index": dict.fromkeys(ROWS["LINE"], 6) | {"LCOR:H:SP": 4}}, "no index for LCOR:V:SP"),
        ({"index": None}, "no index for LINE:BPM:TBT:X, LINE:BPM:TBT:Y, LCOR:H:SP, LCOR:V:SP"),
        (
            {"columns": ["LCOR:V:SP", "LCOR:H:SP"]},
            "columns LCOR:V:SP, LCOR:H:SP and unsolved none, which are not the correctors "
            "asked for (LCOR:H:SP, LCOR:V:SP)",
        ),
        (
            {"columns": ["LCOR:H:SP"]},
            "columns LCOR:H:SP and unsolved none, which are not the correctors asked for "
            "(LCOR:H:SP, LCOR:V:SP)",
        ),
        (
            {"matrix": np.eye(2)[:, :1], "columns": ["LCOR:H:SP"], "unsolved": {"OTHER:SP": "x"}},
            "columns LCOR:H:SP and unsolved OTHER:SP, which are not the correctors asked for "
            "(LCOR:H:SP, LCOR:V:SP)",
        ),
        ({"matrix": [["high", 0.0], [0.0, 1.0]]}, "a value or an index that is no number"),
        (
            {"index": dict.fromkeys(ROWS["LINE"], None) | dict.fromkeys(COLUMNS["LINE"], 4)},
            "a value or an index that is no number",
        ),
        (
            {
                "rows": ["LINE:BPM:TBT:X", "LINE:BPM:TBT:SUM"],
                "index": dict.fromkeys(["LINE:BPM:TBT:X", "LINE:BPM:TBT:SUM"], 6)
                | dict.fromkeys(COLUMNS["LINE"], 4),
            },
            "row LINE:BPM:TBT:SUM, which the model wires in no transverse plane",
        ),
    ],
    ids=[
        "not finite",
        "matrix shape",
        "row index",
        "column index",
        "no index",
        "columns reordered",
        "column dropped",
        "unsolved not asked",
        "value no number",
        "index no number",
        "row without a plane",
    ],
)
def test_an_answer_that_is_no_response_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any], what: str
) -> None:
    monkeypatch.setattr(engine, "response_matrix", lambda *args, **kwargs: _answer(**changed))
    assert _stop_line(tmp_path, trajectory_tree(), ["LINE"]) == (
        "facility: engine-invalid: model LINE — engine pyat answers "
        f"response_matrix(kind='trajectory') for model LINE with {what}; fix: make "
        "response_matrix() answer a finite matrix with a plane and an index for every row "
        "and column"
    )


def test_an_unsolved_corrector_keeps_a_zero_column_and_is_named_in_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file keeps every corrector of the groups; the sidecar says which one has no solve."""
    lost = "particle lost at LBPM (while stepping LCOR:V:SP)"
    solve = engine.response_matrix

    def response_matrix(
        deck: Any, wiring: Any, settings: Any, correctors: Any, *rest: Any, **kwargs: Any
    ) -> Any:
        if correctors != COLUMNS["LINE"]:
            return solve(deck, wiring, settings, correctors, *rest, **kwargs)
        return _answer(
            matrix=np.array([[2.0], [3.0]]), columns=["LCOR:H:SP"], unsolved={"LCOR:V:SP": lost}
        )

    monkeypatch.setattr(engine, "response_matrix", response_matrix)
    directory, _ = write_view(tmp_path, trajectory_tree())
    matrix, index = _files(directory, "LINE")
    assert matrix["variable_names"] == COLUMNS["LINE"]
    assert matrix["matrix"] == [[2.0, 0.0], [3.0, 0.0]]
    assert index == {
        "bpm_index": [6, 6],
        "sum_names": [SUMS["LINE"]] * 2,
        "hcorr_index": [4],
        "vcorr_index": [4],
        "unsolved": {"LCOR:V:SP": lost},
    }


def test_an_answer_with_no_corrector_solved_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsolved = {"LCOR:H:SP": "lost (while stepping LCOR:H:SP)", "LCOR:V:SP": "lost too"}
    monkeypatch.setattr(
        engine,
        "response_matrix",
        lambda *args, **kwargs: _answer(matrix=np.empty((2, 0)), columns=[], unsolved=unsolved),
    )
    assert _stop_line(tmp_path, trajectory_tree(), ["LINE"]) == (
        "facility: engine-invalid: model LINE — engine pyat solves no corrector of model "
        "LINE's response_matrix(kind='trajectory'): lost (while stepping LCOR:H:SP); lost too; "
        "fix: give the model a deck, wiring and settings its first turn solves on"
    )


def test_an_answer_of_another_shape_is_no_refusal_of_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop points at the engine plug-in, not at the model's deck, wiring and settings."""
    monkeypatch.setattr(
        engine, "response_matrix", lambda *args, **kwargs: (list(ROWS["LINE"]), np.eye(2))
    )
    assert _stop_line(tmp_path, trajectory_tree(), ["LINE"]) == (
        "facility: engine-invalid: model LINE — engine pyat answers "
        "response_matrix(kind='trajectory') for model LINE with an answer that is not a "
        "response matrix; fix: make response_matrix() answer a finite matrix with a plane and "
        "an index for every row and column"
    )


def test_an_engine_whose_function_takes_no_kind_fails_as_python_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The view reads no signature: the call's own failure is the report."""

    def response_matrix(*request: Any) -> Any:
        raise AssertionError(f"not reached: {request}")

    monkeypatch.setattr(engine, "response_matrix", response_matrix)
    inputs = view_inputs(tmp_path, trajectory_tree(), ["LINE"])
    with pytest.raises(TypeError, match="unexpected keyword argument 'kind'"):
        write_pyaml_view(tmp_path / "render" / "data" / "pyaml", inputs)


# --- the order of the rows and columns ----------------------------------------

#: LINE's second monitor and second corrector of each plane, wired as copies of the first.
X1, Y1, S1 = "LINE:BPM:TBT:X", "LINE:BPM:TBT:Y", "LINE:BPM:TBT:SUM"
X2, Y2, S2 = "LINE:BPM2:TBT:X", "LINE:BPM2:TBT:Y", "LINE:BPM2:TBT:SUM"
H1, V1 = COLUMNS["LINE"]
H2, V2 = "LCOR2:H:SP", "LCOR2:V:SP"


def _widened(wiring: list[dict[str, Any]], *, drop: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """LINE's wiring with a second monitor ``LBPM2`` and a second corrector per plane."""
    by_address = {entry["address"]: entry for entry in wiring}
    added = [by_address[first] | {"address": second} for first, second in ((H1, H2), (V1, V2))] + [
        by_address[first] | {"address": second, "element": "LBPM2"}
        for first, second in ((X1, X2), (Y1, Y2), (S1, S2))
    ]
    return [entry for entry in [*wiring, *added] if entry["address"] not in drop]


def _write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: Any,
    hcor: list[str],
    vcor: list[str],
    drop: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """LINE's two files written from ``answer`` for ``hcor`` then ``vcor``, over the widened wiring."""
    from osprey.facility.views import simulator

    inputs = view_inputs(tmp_path, trajectory_tree(), ["LINE"])
    (record,) = [entry for entry in inputs.doc["models"] if entry["name"] == "LINE"]
    wired = simulator.simulator_wiring
    asked: list[list[str]] = []

    def response_matrix(*request: Any, **kind: Any) -> Any:
        asked.append(list(request[3]))
        return answer

    monkeypatch.setattr(
        simulator, "simulator_wiring", lambda *args: _widened(wired(*args), drop=drop)
    )
    monkeypatch.setattr(engine, "response_matrix", response_matrix)
    monkeypatch.setattr(pyaml_view, "_SOLVED", {})
    files = pyaml_view._trajectory_files(inputs, record, hcor, vcor)
    assert asked == [[*hcor, *vcor]]
    return json.loads(files[TRAJECTORY_RM_FILE]), json.loads(files[TRAJECTORY_RM_INDEX_FILE])


def _wide_answer(**changed: Any) -> Any:
    """Two monitors at 3 and 8, the correctors asked against their lattice order."""
    answer: dict[str, Any] = {
        "rows": [X1, X2, Y1, Y2],
        "columns": [H1, H2, V1, V2],
        "matrix": np.arange(16.0).reshape(4, 4) + 1.0,
        "unsolved": {},
        "index": {X1: 3, X2: 8, Y1: 3, Y2: 8, H1: 9, H2: 2, V1: 7, V2: 5},
    }
    return engine.ResponseMatrix(**(answer | changed))


def test_correctors_asked_out_of_lattice_order_are_written_in_it_per_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The solve is asked in the groups' order; the files are in lattice order from the injection."""
    matrix, index = _write(tmp_path, monkeypatch, _wide_answer(), [H1, H2], [V1, V2])
    assert matrix["variable_names"] == [H2, H1, V2, V1]
    assert matrix["variable_planes"] == ["H", "H", "V", "V"]
    solved = np.arange(16.0).reshape(4, 4) + 1.0
    assert matrix["matrix"] == solved[:, [1, 0, 3, 2]].tolist()
    assert index["hcorr_index"] == [2, 9]
    assert index["vcorr_index"] == [5, 7]
    assert matrix["observable_names"] == [X1, X2, Y1, Y2]
    assert index["bpm_index"] == [3, 8, 3, 8]
    assert index["sum_names"] == [S1, S2, S1, S2]


def test_an_unsolved_corrector_moves_with_its_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lost = "lost (while stepping LCOR:V:SP)"
    solved = np.arange(12.0).reshape(4, 3) + 1.0
    answer = _wide_answer(columns=[H1, H2, V2], matrix=solved, unsolved={V1: lost})
    matrix, index = _write(tmp_path, monkeypatch, answer, [H1, H2], [V1, V2])
    assert matrix["variable_names"] == [H2, H1, V2, V1]
    assert matrix["matrix"] == [[row[1], row[0], row[2], 0.0] for row in solved.tolist()]
    assert index["vcorr_index"] == [5, 7]
    assert index["unsolved"] == {V1: lost}


def test_correctors_at_one_index_keep_the_order_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    places = {X1: 3, X2: 8, Y1: 3, Y2: 8, H1: 4, H2: 4, V1: 4, V2: 4}
    matrix, index = _write(tmp_path, monkeypatch, _wide_answer(index=places), [H1, H2], [V1, V2])
    assert matrix["variable_names"] == [H1, H2, V1, V2]
    assert index["hcorr_index"] == index["vcorr_index"] == [4, 4]


def _stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: Any, drop: tuple[str, ...] = ()
) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        _write(tmp_path, monkeypatch, answer, [H1, H2], [V1, V2], drop)
    return str(stopped.value.format_message())


_ROWS_STOP = (
    "facility: engine-invalid: model LINE — engine pyat answers "
    "response_matrix(kind='trajectory') for model LINE with x rows {x} and y rows {y}, which "
    "do not read the same monitors in the same order; fix: wire every monitor the response "
    "reads with one x and one y turn series"
)


@pytest.mark.parametrize(
    ("rows", "x", "y"),
    [
        ([X1, X2, Y1], f"{X1}, {X2}", Y1),
        ([X1, Y1, Y2], X1, f"{Y1}, {Y2}"),
        ([X1, X2], f"{X1}, {X2}", "none"),
        ([X1, X2, Y2, Y1], f"{X1}, {X2}", f"{Y2}, {Y1}"),
        ([X1, Y2], X1, Y2),
        ([X1, Y1, X2, Y2], f"{X1}, {X2}", f"{Y1}, {Y2}"),
    ],
    ids=[
        "a monitor without y",
        "a monitor without x",
        "no y at all",
        "y in another order",
        "x and y of different monitors",
        "planes interleaved",
    ],
)
def test_x_and_y_rows_that_are_not_the_same_monitors_in_order_stop_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[str], x: str, y: str
) -> None:
    answer = _wide_answer(rows=rows, matrix=np.ones((len(rows), 4)))
    assert _stopped(tmp_path, monkeypatch, answer) == _ROWS_STOP.format(x=x, y=y)


@pytest.mark.parametrize(
    ("places", "half", "shown"),
    [
        ({X1: 8, X2: 3, Y1: 8, Y2: 3}, f"{X1}, {X2}", "8, 3"),
        ({X1: 3, X2: 3, Y1: 3, Y2: 3}, f"{X1}, {X2}", "3, 3"),
        ({X1: 3, X2: 8, Y1: 8, Y2: 8}, f"{Y1}, {Y2}", "8, 8"),
    ],
    ids=["decreasing", "equal", "equal in y alone"],
)
def test_a_half_whose_indices_do_not_increase_strictly_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, places: dict[str, int], half: str, shown: str
) -> None:
    index = places | {H1: 9, H2: 2, V1: 7, V2: 5}
    assert _stopped(tmp_path, monkeypatch, _wide_answer(index=index)) == (
        "facility: engine-invalid: model LINE — engine pyat answers "
        f"response_matrix(kind='trajectory') for model LINE with rows {half} at indices "
        f"{shown}, which do not increase strictly; fix: make response_matrix() answer each "
        "plane's rows in lattice order from the injection, one monitor per element"
    )


def test_a_row_whose_monitor_wires_no_sum_series_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _stopped(tmp_path, monkeypatch, _wide_answer(), drop=(S2,)) == (
        f"facility: engine-invalid: model LINE — row {X2} of model LINE's "
        "response_matrix(kind='trajectory') reads a monitor the model wires no sum series on; "
        "fix: wire a sum turn series on every monitor the response reads"
    )


@pytest.mark.parametrize("name", MODELS)
def test_each_row_names_the_sum_series_of_its_monitor(
    view: tuple[Path, list[Path]], name: str
) -> None:
    matrix, index = _files(view[0], name)
    assert index["sum_names"] == [SUMS[name]] * len(matrix["observable_names"])


# --- the kick step ------------------------------------------------------------

#: A corrector whose kick angle grows faster than its hardware value, with the way back.
_CURVED = {
    "curve": {"table": {"grid": [0.0, 1.0, 2.0], "values": [0.0, 0.01, 0.04]}},
    "inverse": {"table": {"grid": [0.0, 0.01, 0.04], "values": [0.0, 1.0, 2.0]}},
}


def test_the_hardware_step_spans_the_kick_step_about_the_held_value() -> None:
    """Held at 1, the kick is 0.01; 0.005 is hardware 0.5 and 0.015 is 1 + 0.005 / 0.03."""
    from osprey.facility.views.pyaml import hardware_width

    width = hardware_width({"default": 1.0, "calibration": _CURVED}, 0.01)
    assert width == pytest.approx(1.0 + 0.005 / 0.03 - 0.5, rel=1.0e-12)
    # About a held value of 0 the same span is hardware -0.5 to 0.5.
    assert hardware_width({"calibration": _CURVED}, 0.01) == pytest.approx(1.0, rel=1.0e-12)


def test_a_setpoint_without_a_calibration_steps_by_the_kick_step_itself() -> None:
    from osprey.facility.views.pyaml import hardware_width

    assert hardware_width({"default": 3.0}, 2.0e-5) == pytest.approx(2.0e-5, rel=1.0e-9)


@pytest.mark.parametrize(
    ("calibration", "text"),
    [
        (
            _CURVED | {"inverse": {"table": {"grid": [0.0, 0.04], "values": [1.0, 1.0]}}},
            "the hardware step is 0.0",
        ),
        ({"curve": {"table": {"grid": [-1.0, 1.0], "values": [-0.01, 0.01]}}}, "no inverse"),
    ],
    ids=["a flat way back", "no way back"],
)
def test_a_calibration_that_gives_no_hardware_step_is_refused(
    calibration: dict[str, Any], text: str
) -> None:
    from osprey.facility.views.pyaml import hardware_width

    with pytest.raises(ValueError, match=text):
        hardware_width({"default": 0.0, "calibration": calibration}, 0.01)


def test_a_corrector_whose_calibration_has_no_way_back_stops_the_build(tmp_path: Path) -> None:
    """The build refuses such a setpoint long before the view; the view never divides by it."""
    inputs = view_inputs(tmp_path, trajectory_tree(), ["LINE"])
    (record,) = [entry for entry in inputs.doc["models"] if entry["name"] == "LINE"]
    for wired in record["wiring"]:
        if wired["address"] == "LCOR:H:SP":
            wired["calibration"] = {
                "curve": {"table": {"grid": [-1.0, 1.0], "values": [-1.0e-2, 1.0e-2]}}
            }
    with pytest.raises(FacilityBuildError) as stopped:
        write_pyaml_view(tmp_path / "render" / "data" / "pyaml", inputs)
    assert str(stopped.value.format_message()) == (
        "facility: value-invalid: model LINE — corrector LCOR:H:SP of model LINE has no "
        "hardware step for `corrector_delta` 1e-05: a table calibration has no inverse; fix: "
        "state a calibration with a way back to hardware for the corrector"
    )
