"""Every endpoint that reaches a machine names it; none has a default target.

The shared ``client`` fixture defaults a missing target to ``va`` for the older
tests, so these build their own client with the production dependency intact.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from osprey.services.optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey.services.optimizer.api.job_tracker import JobTracker
from osprey.services.optimizer.api.main import app
from osprey.services.optimizer.models.enums import EnvironmentType
from osprey.services.optimizer.models.protocols import PVObservable, PVReading, VariableMetadata

MACHINE_ROUTES = [
    pytest.param("get", "/environments/env/status", None, {}, id="status"),
    pytest.param("get", "/environments/env/variables", None, {"names": ["x"]}, id="get_variables"),
    pytest.param(
        "get", "/environments/env/observables", None, {"names": ["o"]}, id="get_observables"
    ),
    pytest.param("get", "/environments/env/pv", None, {"names": ["PV"]}, id="read_pvs"),
    pytest.param("get", "/environments/env/initial-bounds", None, {}, id="initial_bounds"),
    pytest.param(
        "put", "/environments/env/variables", {"variables": {"x": 1.0}}, {}, id="set_variables"
    ),
    pytest.param(
        "post",
        "/environments/env/evaluate",
        {"variables": {"x": 1.0}, "observables": ["o"]},
        {},
        id="evaluate",
    ),
]


@pytest.fixture
def production_client() -> Generator[tuple[TestClient, MagicMock], None, None]:
    env = MagicMock()
    env.__enter__.return_value = env
    env.variables = {
        "x": VariableMetadata(min_value=0, max_value=2, readback_pv="X:RB", setpoint_pv="X:SP")
    }
    env.observables = {"o": PVObservable(readback_pv="O")}
    env.environment_type = EnvironmentType.SIMULATION
    env.available_objectives = ["o"]
    env.default_objective = "o"
    env.get_variables.return_value = {"x": PVReading(value=1.0)}
    env.get_observables.return_value = {"o": PVReading(value=2.0)}
    env.read_pvs.return_value = {"PV": PVReading(value=3.0)}
    env.evaluate.return_value = {"o": PVReading(value=2.0)}
    env.check_connection.return_value = True
    registry = MagicMock()
    registry.load_environment.return_value = env
    registry.get_presets.return_value = {}
    tracker = MagicMock(spec=JobTracker)
    tracker.get_active_job.return_value = None
    app.dependency_overrides[get_environment_registry] = lambda: registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    try:
        with TestClient(app) as client:
            yield client, registry
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(("method", "path", "body", "params"), MACHINE_ROUTES)
def test_a_machine_route_without_a_target_is_422(production_client, method, path, body, params):
    client, registry = production_client
    resp = client.request(method, path, json=body, params=params)
    assert resp.status_code == 422
    assert any(err["loc"][-1] == "target" for err in resp.json()["detail"])
    registry.load_environment.assert_not_called()


@pytest.mark.parametrize(("method", "path", "body", "params"), MACHINE_ROUTES)
def test_an_unknown_target_is_422(production_client, method, path, body, params):
    client, registry = production_client
    resp = client.request(method, path, json=body, params={**params, "target": "production"})
    assert resp.status_code == 422
    registry.load_environment.assert_not_called()


@pytest.mark.parametrize("target", ["va", "standin", "live"])
@pytest.mark.parametrize("execution_mode", [None, "readonly"])
def test_the_named_target_reaches_the_environment(production_client, target, execution_mode):
    client, registry = production_client
    params = {"names": ["x"], "target": target}
    if execution_mode:
        params["execution_mode"] = execution_mode
    resp = client.get("/environments/env/variables", params=params)
    assert resp.status_code == 200
    registry.load_environment.assert_called_once_with(
        "env", {"control_target": target, "execution_mode": execution_mode}
    )


def test_metadata_routes_need_no_target(production_client):
    client, registry = production_client
    registry.list_environments.return_value = []
    assert client.get("/environments").status_code == 200
    assert client.get("/environments/env").status_code == 200


def test_creating_a_run_without_a_target_is_422(production_client):
    """A draft's bounds come from the machine the run will drive, so it names one at creation."""
    from osprey.services.optimizer.models.api import EnvironmentInfo

    client, registry = production_client
    registry.list_environments.return_value = [
        EnvironmentInfo(name="env", display_name="Env", valid=True)
    ]
    resp = client.post("/optimizations", json={"environment_name": "env", "objective_name": "o"})
    assert resp.status_code == 422
    assert "control_target" in resp.json()["detail"]
