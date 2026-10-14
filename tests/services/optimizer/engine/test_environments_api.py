"""Tests for the environments router in api/routers/environments.py.

Covers: /environments, /environments/{name}/status,
        /environments/{name}/variables (GET & PUT),
        /environments/{name}/observables (GET),
        /environments/{name}/evaluate (POST),
        /environments/{name} (GET details), /environments/{name}/pv (GET).
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from osprey.services.optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey.services.optimizer.models.exceptions import TimeoutError as OptimizerTimeoutError
from osprey.services.optimizer.models.exceptions import VariableOutOfBoundsError
from osprey.services.optimizer.models.protocols import PVReading

from ._fakes import MockEnvironment


class TestListEnvironmentsEndpoint:
    @pytest.mark.parametrize(
        ("empty", "expected_names"),
        [pytest.param(False, ["sim"], id="registered"), pytest.param(True, [], id="empty")],
    )
    def test_list_returns_registered(
        self, client: TestClient, mock_registry: MagicMock, empty: bool, expected_names: list[str]
    ) -> None:
        if empty:
            mock_registry.list_environments.return_value = []
        resp = client.get("/environments")
        assert resp.status_code == 200
        assert [e["name"] for e in resp.json()] == expected_names


class _BrokenEnv(MockEnvironment):
    """Env whose connection probe raises instead of returning a verdict."""

    def check_connection(self) -> bool | None:
        raise RuntimeError("EPICS timeout")


class TestEnvironmentStatusEndpoint:
    @pytest.mark.parametrize("check_result", [True, False], ids=["connected", "disconnected"])
    def test_reports_connection(
        self, client: TestClient, mock_env: MockEnvironment, check_result: bool
    ) -> None:
        mock_env._check_connection_result = check_result
        resp = client.get("/environments/sim/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["connected"] is check_result
        assert body["name"] == "sim"

    @pytest.mark.parametrize(
        ("load_side_effect", "expected_message"),
        [
            pytest.param(
                lambda name, config: _BrokenEnv(), "EPICS timeout", id="check_connection_raises"
            ),
            pytest.param(
                RuntimeError("plugin load fail"), "plugin load fail", id="load_environment_raises"
            ),
        ],
    )
    def test_probe_failure_reports_unknown_not_500(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        load_side_effect: Any,
        expected_message: str,
    ) -> None:
        """An exception inside check_connection, or out of load_environment itself,
        yields connected=None (the check could not complete) with the message set —
        never a 500."""
        mock_registry.load_environment.side_effect = load_side_effect
        resp = client.get("/environments/sim/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["connected"] is None
        assert expected_message in (body["message"] or "")


class TestSetEnvironmentVariablesEndpoint:
    def test_set_success(self, client: TestClient, mock_env: MockEnvironment) -> None:
        spy = MagicMock()
        mock_env.set_variables = spy  # type: ignore[method-assign]

        resp = client.put(
            "/environments/als_injector_sim/variables", json={"variables": {"var1": 3.14}}
        )
        assert resp.status_code == 204
        assert resp.content == b""
        spy.assert_called_once_with({"var1": 3.14})

    def test_out_of_bounds_returns_422(self, client: TestClient, mock_env: MockEnvironment) -> None:
        """An out-of-bounds / unknown / non-finite setpoint is a client error, not a 500."""
        mock_env.set_variables = MagicMock(  # type: ignore[method-assign]
            side_effect=VariableOutOfBoundsError(
                "Variable var1 value 99 is outside bounds [0.0, 10.0]"
            )
        )
        resp = client.put(
            "/environments/als_injector_sim/variables", json={"variables": {"var1": 99.0}}
        )
        assert resp.status_code == 422
        assert "outside bounds" in resp.json()["detail"]


class TestGetEnvironmentVariablesEndpoint:
    def test_unknown_name_returns_422(self, client: TestClient) -> None:
        """Unknown variable names are rejected up-front rather than surfacing as a 500."""
        resp = client.get("/environments/sim/variables", params={"names": ["ghost"]})
        assert resp.status_code == 422
        assert "ghost" in resp.json()["detail"]


class TestReadbackPrecision:
    """Variable readbacks and observables both cross the wire at full precision."""

    def test_get_variables_full_precision(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        mock_env.get_variables = MagicMock(return_value={"var1": PVReading(value=3.123456789)})  # type: ignore[method-assign]
        resp = client.get("/environments/sim/variables", params={"names": ["var1"]})
        assert resp.status_code == 200
        assert resp.json()["var1"]["value"] == 3.123456789

    def test_evaluate_full_precision(self, client: TestClient, mock_env: MockEnvironment) -> None:
        mock_env.get_variables = MagicMock(return_value={"var1": PVReading(value=3.123456789)})  # type: ignore[method-assign]
        mock_env.get_observables = MagicMock(return_value={"obs1": PVReading(value=10.987654321)})  # type: ignore[method-assign]
        resp = client.post(
            "/environments/als_injector_sim/evaluate",
            json={"variables": {"var1": 3.0}, "observables": ["obs1"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["variable_readbacks"]["var1"]["value"] == 3.123456789
        assert body["observables"]["obs1"]["value"] == 10.987654321

    def test_get_observables_full_precision(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        mock_env.get_observables = MagicMock(return_value={"obs1": PVReading(value=10.987654321)})  # type: ignore[method-assign]
        resp = client.get("/environments/sim/observables", params={"names": ["obs1"]})
        assert resp.status_code == 200
        assert resp.json()["obs1"]["value"] == 10.987654321


class TestGetEnvironmentObservablesEndpoint:
    def test_nan_serialized_as_null(self, client: TestClient) -> None:
        resp = client.get("/environments/sim/observables", params={"names": ["obs_faulted"]})
        assert resp.status_code == 200
        assert resp.json()["obs_faulted"]["value"] is None

    def test_unknown_name_returns_422(self, client: TestClient) -> None:
        resp = client.get("/environments/sim/observables", params={"names": ["nonexistent"]})
        assert resp.status_code == 422
        assert "nonexistent" in resp.json()["detail"]


class TestEvaluateEnvironmentEndpoint:
    def test_evaluate_nan_observable(self, client: TestClient) -> None:
        resp = client.post(
            "/environments/als_injector_sim/evaluate",
            json={"variables": {"var1": 3.0}, "observables": ["obs_faulted"]},
        )
        assert resp.status_code == 200
        assert resp.json()["observables"]["obs_faulted"]["value"] is None

    @pytest.mark.parametrize(
        "json_body",
        [
            pytest.param(
                {"variables": {"var1": 3.0}, "observables": ["nonexistent"]},
                id="unknown_observable",
            ),
            pytest.param(
                {"variables": {"bad_var": 3.0}, "observables": ["obs1"]}, id="unknown_variable"
            ),
            pytest.param({"variables": {"var1": 3.0}, "observables": []}, id="empty_observables"),
        ],
    )
    def test_request_validation_returns_422(
        self, client: TestClient, json_body: dict[str, Any]
    ) -> None:
        """Unknown observable, unknown variable, and an empty observable list are all
        rejected up-front by the evaluate route."""
        resp = client.post("/environments/als_injector_sim/evaluate", json=json_body)
        assert resp.status_code == 422

    def test_out_of_bounds_setpoint_returns_422(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        """A non-finite / out-of-bounds setpoint raised during evaluate maps to 422, not 500."""
        mock_env.evaluate = MagicMock(  # type: ignore[method-assign]
            side_effect=VariableOutOfBoundsError("Variable var1 value inf is not finite")
        )
        resp = client.post(
            "/environments/als_injector_sim/evaluate",
            json={"variables": {"var1": 3.0}, "observables": ["obs1"]},
        )
        assert resp.status_code == 422
        assert "not finite" in resp.json()["detail"]


# Cross-endpoint: 404 when environment is unknown / 500 on env-method failure


@pytest.mark.parametrize(
    ("method", "path", "json_body", "params", "expected_status"),
    [
        pytest.param("get", "/environments/unknown/status", None, None, 404, id="status"),
        # Guarded write routes: on the simulator the guard probes nothing, so the
        # unknown name reaches load_environment and is a 404 like the reads.
        pytest.param(
            "put",
            "/environments/nope/variables",
            {"variables": {"x": 1.0}},
            None,
            404,
            id="set_variables",
        ),
        pytest.param(
            "get", "/environments/nope/variables", None, {"names": ["x"]}, 404, id="get_variables"
        ),
        pytest.param(
            "get",
            "/environments/nope/observables",
            None,
            {"names": ["obs1"]},
            404,
            id="get_observables",
        ),
        pytest.param(
            "post",
            "/environments/nope/evaluate",
            {"variables": {"var1": 1.0}, "observables": ["obs1"]},
            None,
            404,
            id="evaluate",
        ),
    ],
)
def test_unknown_environment_returns_404(
    client: TestClient,
    mock_registry: MagicMock,
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    params: dict[str, Any] | None,
    expected_status: int,
) -> None:
    """Non-guarded endpoints return 404 for unknown env; guarded write routes return 412
    because the current-environment check fires before load_environment."""
    mock_registry.load_environment.side_effect = KeyError("nope")
    kwargs: dict[str, Any] = {}
    if json_body is not None:
        kwargs["json"] = json_body
    if params is not None:
        kwargs["params"] = params
    resp = getattr(client, method)(path, **kwargs)
    assert resp.status_code == expected_status


# Endpoint tuples: (method, path, json_body, params, broken_attr). Only the
# exception and expected status vary per row below.
_SET_VARS_1 = (
    "put",
    "/environments/als_injector_sim/variables",
    {"variables": {"var1": 1.0}},
    None,
    "set_variables",
)
_SET_VARS_3 = (
    "put",
    "/environments/als_injector_sim/variables",
    {"variables": {"var1": 3.0}},
    None,
    "set_variables",
)
_GET_VARS = ("get", "/environments/sim/variables", None, {"names": ["var1"]}, "get_variables")
_GET_OBS = ("get", "/environments/sim/observables", None, {"names": ["obs1"]}, "get_observables")
_EVALUATE = (
    "post",
    "/environments/als_injector_sim/evaluate",
    {"variables": {"var1": 3.0}, "observables": ["obs1"]},
    None,
    "evaluate",
)
_READ_PVS = ("get", "/environments/sim/pv", None, {"names": ["SR:DCCT"]}, "read_pvs")

_HW_FAULT = RuntimeError("hardware fault")
_WEDGED = OptimizerTimeoutError("worker wedged")
_UNREACHABLE = OptimizerConnectionError("Failed to connect to EPICS PV: V1:SP")


@pytest.mark.parametrize(
    ("method", "path", "json_body", "params", "broken_attr", "exception", "expected_status"),
    [
        pytest.param(*_SET_VARS_1, _HW_FAULT, 500, id="set_variables_500"),
        pytest.param(*_GET_VARS, _HW_FAULT, 500, id="get_variables_500"),
        pytest.param(*_GET_OBS, _HW_FAULT, 500, id="get_observables_500"),
        pytest.param(*_EVALUATE, RuntimeError("eval blew up"), 500, id="evaluate_500"),
        pytest.param(*_GET_VARS, _WEDGED, 503, id="get_variables_ca_worker_unresponsive"),
        pytest.param(*_GET_OBS, _WEDGED, 503, id="get_observables_ca_worker_unresponsive"),
        pytest.param(*_EVALUATE, _WEDGED, 503, id="evaluate_ca_worker_unresponsive"),
        pytest.param(
            *_READ_PVS,
            OptimizerTimeoutError("CA worker unresponsive"),
            503,
            id="pv_ca_worker_unresponsive",
        ),
        pytest.param(
            *_SET_VARS_3, _UNREACHABLE, 503, id="set_variables_control_system_unreachable"
        ),
        pytest.param(*_EVALUATE, _UNREACHABLE, 503, id="evaluate_control_system_unreachable"),
    ],
)
def test_environment_method_failure_status_mapping(
    client: TestClient,
    mock_env: MockEnvironment,
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
    params: dict[str, Any] | None,
    broken_attr: str,
    exception: Exception,
    expected_status: int,
) -> None:
    """An environment-method exception maps to the corresponding HTTP status:
    RuntimeError -> 500, OptimizerTimeoutError/OptimizerConnectionError -> 503."""
    setattr(mock_env, broken_attr, MagicMock(side_effect=exception))
    kwargs: dict[str, Any] = {}
    if json_body is not None:
        kwargs["json"] = json_body
    if params is not None:
        kwargs["params"] = params
    resp = getattr(client, method)(path, **kwargs)
    assert resp.status_code == expected_status


class TestGetEnvironmentDetailsEndpoint:
    """GET /environments/{name} happy path + error paths."""

    def test_details_include_observables_metadata(self, client: TestClient) -> None:
        """Happy path: the details response serializes observables metadata."""
        response = client.get("/environments/sim")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "sim"
        assert "variables" in data
        assert "observables" in data

        observables = data["observables"]
        assert "obs1" in observables
        meta = observables["obs1"]
        assert meta["units"] == "mA"
        assert meta["min_value"] == 0.0
        assert meta["max_value"] == 100.0
        assert meta["description"] == "Observable 1"

    def test_load_failure_returns_500(self, client: TestClient, mock_registry: MagicMock) -> None:
        """_load_env unexpected error -> 500."""
        mock_registry.load_environment.side_effect = RuntimeError("plugin import broke")
        resp = client.get("/environments/sim")
        assert resp.status_code == 500
        assert "Failed to load environment" in resp.json()["detail"]


class TestGetEnvironmentPVsEndpoint:
    """The env-scoped ``/pv`` endpoint reads arbitrary PV names through the env's
    CA context. Each env brings its own routing — multiple envs at different
    facilities can coexist without cross-talk.

    The response is a flat name→reading map in request order, with duplicate
    ``names`` values collapsed to a single entry (one read, not two). Readings
    the interface could not read come back ``connected=False`` rather than being
    omitted, and non-finite values serialize as JSON null — the bare token
    ``NaN`` is invalid JSON (some clients tolerate it, ``json.loads`` does not),
    so it is locked out at the response-text level. The endpoint bypasses the
    variables/observables metadata entirely: any PV name is accepted, and
    rejection happens at the CA layer, not the endpoint.
    """

    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            pytest.param(
                ["SR:DCCT"],
                {"SR:DCCT": {"value": 487.2, "connected": True, "units": "mA"}},
                id="scalar_with_metadata",
            ),
            pytest.param(
                ["SR:BPM:WF"],
                {"SR:BPM:WF": {"value": [0.1, -0.3, 0.5], "count": 3}},
                id="waveform_as_json_array",
            ),
            pytest.param(
                ["SR:MODE"],
                {"SR:MODE": {"value": 1, "enum_strings": ["Off", "User"]}},
                id="enum_with_labels",
            ),
            pytest.param(
                ["MISSING:PV"],
                {"MISSING:PV": {"value": None, "connected": False}},
                id="failed_read_serializes_as_disconnected",
            ),
            pytest.param(
                ["FAULT:NAN"],
                {"FAULT:NAN": {"value": None, "connected": True}},
                id="nan_value_serializes_as_null",
            ),
            pytest.param(
                ["SR:DCCT", "SR:DCCT"],
                {"SR:DCCT": {"value": 487.2}},
                id="duplicate_names_collapse_to_single_key",
            ),
            pytest.param(
                ["SR:DCCT", "BR:I", "MISSING:X"],
                {
                    "SR:DCCT": {"value": 487.2},
                    "BR:I": {"value": 487.2},
                    "MISSING:X": {"connected": False},
                },
                id="arbitrary_names_pass_through",
            ),
        ],
    )
    def test_pv_reading_shape(
        self, client: TestClient, names: list[str], expected: dict[str, Any]
    ) -> None:
        resp = client.get("/environments/sim/pv", params={"names": names})
        assert resp.status_code == 200
        assert "NaN" not in resp.text
        body = resp.json()
        assert list(body) == list(expected)  # request order, duplicates collapsed
        for pv_name, fields in expected.items():
            for field, value in fields.items():
                assert body[pv_name][field] == value

    def test_missing_names_query_returns_422(self, client: TestClient) -> None:
        """``names`` is a required query param — omitting it is a client error."""
        resp = client.get("/environments/sim/pv")
        assert resp.status_code == 422

    def test_env_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        mock_registry.load_environment.side_effect = KeyError("nope")
        resp = client.get("/environments/nope/pv", params={"names": ["SR:DCCT"]})
        assert resp.status_code == 404

    def test_unexpected_error_returns_500(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        mock_env.read_pvs = MagicMock(side_effect=RuntimeError("something else"))  # type: ignore[method-assign]
        resp = client.get("/environments/sim/pv", params={"names": ["SR:DCCT"]})
        assert resp.status_code == 500
