"""Tests for current-environment ops (api/current_environment_ops.py,
api/environment_ops.py) and their HTTP surface: GET/PUT
/environments/current, and the guard_environment_write/start behavior on
the variables/evaluate/restore routes. The HTTP-level tests were moved
from the former test_api_endpoints.py.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from osprey.services.optimizer.api import current_environment_ops as ops
from osprey.services.optimizer.api import environment_ops
from osprey.services.optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey.services.optimizer.api.job_tracker import JobTracker
from osprey.services.optimizer.api.main import app
from osprey.services.optimizer.models.api import EnvironmentStatusResponse
from osprey.services.optimizer.models.enums import EnvironmentType

from ._fakes import MockEnvironment
from .conftest import make_run_dir, recording_registry


@pytest.fixture
def tracker():
    return MagicMock(spec=JobTracker)


@pytest.fixture
def registry():
    return MagicMock()


def test_select_rejects_when_job_active(tracker, registry, monkeypatch):
    tracker.set_current_environment_if_idle.return_value = False
    tracker.get_active_job.return_value = "job-9"
    monkeypatch.setattr(ops, "read_job_environment_name", lambda job_id: "als_injector_sim")
    with pytest.raises(HTTPException) as exc:
        ops.select_environment("als_injector_sim", tracker, registry)
    assert exc.value.status_code == 409
    assert "als_injector_sim" in exc.value.detail


def test_select_succeeds_on_red_health_returns_200_shape(tracker, registry, monkeypatch):
    tracker.set_current_environment_if_idle.return_value = True
    monkeypatch.setattr(ops, "environment_details", lambda name, registry: None)
    monkeypatch.setattr(
        ops,
        "_probe_and_cache_health",
        lambda name, registry: EnvironmentStatusResponse(
            name=name, connected=False, message="down"
        ),
    )
    resp = ops.select_environment("als_injector", tracker, registry)
    assert resp.name == "als_injector"
    assert resp.status is not None
    assert resp.status.connected is False  # selection still succeeded on red


def _stub_health_cache(monkeypatch, initial=None):
    """Back the health cache with a dict-like redis stub (pre-seeded with
    *initial*) and count ``probe_environment_health`` calls. Returns the counter."""
    store: dict[str, str] = dict(initial or {})
    redis_stub = MagicMock()
    redis_stub.get.side_effect = lambda k: store.get(k)
    redis_stub.set.side_effect = lambda k, v, ex=None: store.__setitem__(k, v)
    monkeypatch.setattr(
        "osprey.services.optimizer.utils.redis_client.get_redis_client", lambda **kw: redis_stub
    )
    calls = {"n": 0}

    def fake_probe(name, registry):
        calls["n"] += 1
        return EnvironmentStatusResponse(name=name, connected=True, message=None)

    monkeypatch.setattr(ops, "probe_environment_health", fake_probe)
    return calls


def test_build_response_serves_cached_health_on_second_call(tracker, registry, monkeypatch):
    tracker.get_current_environment.return_value = "als_injector"
    monkeypatch.setattr(ops, "environment_details", lambda name, registry: None)
    calls = _stub_health_cache(monkeypatch)

    ops.build_current_environment_response(tracker, registry)
    ops.build_current_environment_response(tracker, registry)
    assert calls["n"] == 1  # second call hit the cache, did not re-probe


# The tests above mock environment_details / probe_environment_health to isolate the
# callers; these exercise the environment_ops helpers directly against a stub env
# (load_environment / lookup_environment patched).
def _stub_env():
    env = MagicMock()
    env.environment_type = EnvironmentType.SIMULATION
    env.variables = {}
    env.observables = {}
    env.default_objective = "loss"
    env.available_objectives = ["loss"]
    return env


def test_environment_details_maps_env_attributes(registry, monkeypatch):
    env = _stub_env()
    monkeypatch.setattr(environment_ops, "load_environment", lambda name, registry: env)
    details = environment_ops.environment_details("als_injector_sim", registry)
    assert details.name == "als_injector_sim"
    assert details.environment_type == EnvironmentType.SIMULATION
    assert details.default_objective == "loss"
    env.__enter__.assert_called_once()  # opened the env context to read its schema


def test_probe_health_reports_connected(registry, monkeypatch):
    env = _stub_env()
    env.check_connection.return_value = True
    monkeypatch.setattr(environment_ops, "lookup_environment", lambda name, registry: env)
    status = environment_ops.probe_environment_health("als_injector", registry)
    assert status.connected is True and status.message is None


def test_probe_health_swallows_exception_returns_red(registry, monkeypatch):
    # Docstring contract: an unreachable env is selectable and reports red, not an error.
    env = _stub_env()
    env.check_connection.side_effect = RuntimeError("EPICS timeout")
    monkeypatch.setattr(environment_ops, "lookup_environment", lambda name, registry: env)
    status = environment_ops.probe_environment_health("als_injector", registry)
    assert status.connected is None
    assert status.message == "EPICS timeout"


def test_probe_health_reraises_httpexception(registry, monkeypatch):
    # A missing env (HTTPException from lookup) propagates unchanged, not swallowed to red.
    def _missing(name, registry):
        raise HTTPException(status_code=404, detail="no such env")

    monkeypatch.setattr(environment_ops, "lookup_environment", _missing)
    with pytest.raises(HTTPException) as exc:
        environment_ops.probe_environment_health("ghost", registry)
    assert exc.value.status_code == 404


def test_build_response_reprobes_when_cached_health_is_for_other_env(
    tracker, registry, monkeypatch
):
    # A cache entry for a different env is stale after a switch: ignore it and re-probe.
    tracker.get_current_environment.return_value = "env_b"
    monkeypatch.setattr(ops, "environment_details", lambda name, registry: None)
    stale = EnvironmentStatusResponse(
        name="env_a", connected=False, message="down"
    ).model_dump_json()
    calls = _stub_health_cache(monkeypatch, {ops.HEALTH_CACHE_KEY: stale})

    resp = ops.build_current_environment_response(tracker, registry)
    assert calls["n"] == 1  # cross-env cache ignored, re-probed for env_b
    assert resp.status is not None and resp.status.name == "env_b"


def test_environment_connected_reraises_httpexception(registry):
    # A missing env (404 from lookup) must propagate, not be swallowed to "not connected".
    registry.load_environment.side_effect = KeyError("ghost")
    with pytest.raises(HTTPException) as exc:
        environment_ops._environment_connected("ghost", registry)
    assert exc.value.status_code == 404


class TestWriteGuard:
    def _tracker(self, active=None, current=None):
        t = MagicMock(spec=JobTracker)
        t.get_active_job.return_value = active
        t.get_current_environment.return_value = current
        return t

    @pytest.mark.parametrize(
        "active,current,env_name,expected_status",
        [
            ("job-1", "als_injector_sim", "als_injector_sim", 409),
            (None, "als_injector", "als_injector_sim", 412),
            (None, None, "als_injector", 412),
        ],
        ids=["job_active", "pointer_mismatch", "pointer_unset"],
    )
    def test_rejects(self, active, current, env_name, expected_status):
        t = self._tracker(active=active, current=current)
        with pytest.raises(HTTPException) as exc:
            environment_ops.guard_environment_write(
                env_name, t, MagicMock(), override_unhealthy=False
            )
        assert exc.value.status_code == expected_status

    def test_rejects_unhealthy_nonsim_503(self, monkeypatch):
        t = self._tracker(active=None, current="als_injector")
        monkeypatch.setattr(environment_ops, "_environment_connected", lambda n, r: False)
        with pytest.raises(HTTPException) as exc:
            environment_ops.guard_environment_write(
                "als_injector", t, MagicMock(), override_unhealthy=False
            )
        assert exc.value.status_code == 503

    def test_allows_unhealthy_nonsim_with_override(self, monkeypatch):
        t = self._tracker(active=None, current="als_injector")
        monkeypatch.setattr(environment_ops, "_environment_connected", lambda n, r: False)
        environment_ops.guard_environment_write(
            "als_injector", t, MagicMock(), override_unhealthy=True
        )  # no raise

    def test_raising_probe_yields_503(self):
        # A raising check_connection is treated as "not connected" -> 503.
        t = self._tracker(active=None, current="als_injector")
        broken_env = MagicMock()
        broken_env.__enter__ = lambda s: s
        broken_env.__exit__ = lambda s, *a: None
        broken_env.check_connection.side_effect = RuntimeError("EPICS timeout")
        mock_reg = MagicMock()
        mock_reg.load_environment.return_value = broken_env
        with pytest.raises(HTTPException) as exc:
            environment_ops.guard_environment_write(
                "als_injector", t, mock_reg, override_unhealthy=False
            )
        assert exc.value.status_code == 503

    def test_skips_health_for_sim(self, monkeypatch):
        t = self._tracker(active=None, current="als_injector_sim")

        def _must_not_be_called(n, r):
            raise AssertionError("health probe must not be called for a sim env")

        monkeypatch.setattr(environment_ops, "_environment_connected", _must_not_be_called)
        environment_ops.guard_environment_write(
            "als_injector_sim", t, MagicMock(), override_unhealthy=False
        )  # probe not called, no raise

    def test_start_guard_skips_active_job_check(self, monkeypatch):
        # start does NOT check the active-job lock (the claim handles the slot).
        t = self._tracker(active="job-1", current="als_injector_sim")
        environment_ops.guard_environment_start(
            "als_injector_sim", t, MagicMock(), override_unhealthy=False
        )  # no raise


def test_switch_loses_to_active_job_atomically(redis_server):
    tracker = JobTracker()
    tracker._get_redis().flushdb()
    assert tracker.get_current_environment() is None  # unset until someone claims it
    assert tracker.set_current_environment_if_idle("als_injector_sim") is True
    tracker._get_redis().set("api:active_job", "job-1")  # a job becomes active
    assert tracker.set_current_environment_if_idle("als_injector") is False
    assert tracker.get_current_environment() == "als_injector_sim"  # pointer unchanged


@pytest.fixture
def client_with_tracker(
    mock_registry: MagicMock, seam_redis_mock: MagicMock
) -> Generator[tuple[TestClient, MagicMock], None, None]:
    seam_redis_mock.get.return_value = None  # no job config in Redis by default
    tracker = MagicMock(spec=JobTracker)
    app.dependency_overrides[get_environment_registry] = lambda: mock_registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    with TestClient(app) as c:
        yield c, tracker


def test_get_current_environment_unset_returns_null(
    client_with_tracker: tuple[TestClient, MagicMock],
) -> None:
    """Also pins that "/current" is not shadowed by the "/{name}" route: if it were
    parsed as "/{name}", the permissive mock_registry would serve
    EnvironmentDetails(name="current"); the null name proves the new route won.
    """
    client, tracker = client_with_tracker
    tracker.get_current_environment.return_value = None
    resp = client.get("/environments/current")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] is None  # the /{name} handler would return name="current"
    assert body["details"] is None and body["status"] is None  # fully-null unset response
    assert resp.headers.get("cache-control") == "no-store"


def test_put_current_environment_conflicts_when_job_active(
    client_with_tracker: tuple[TestClient, MagicMock],
) -> None:
    client, tracker = client_with_tracker
    tracker.set_current_environment_if_idle.return_value = False
    tracker.get_active_job.return_value = "job-3"
    resp = client.put("/environments/current", json={"name": "als_injector_sim"})
    assert resp.status_code == 409


@pytest.mark.parametrize(
    ("method", "path", "json_body", "active_job", "current_env", "expected_status"),
    [
        pytest.param(
            "put",
            "/environments/als_injector_sim/variables",
            {"variables": {"a": 1.0}},
            None,
            "als_injector",
            412,
            id="set_variables_not_current",
        ),
        pytest.param(
            "put",
            "/environments/als_injector_sim/variables",
            {"variables": {"a": 1.0}},
            "job-1",
            "als_injector_sim",
            409,
            id="set_variables_job_active",
        ),
        pytest.param(
            "post",
            "/environments/als_injector_sim/evaluate",
            {"variables": {"a": 1.0}, "observables": ["x"]},
            None,
            "als_injector",
            412,
            id="evaluate_not_current",
        ),
    ],
)
def test_guarded_routes_rejected(
    client_with_tracker: tuple[TestClient, MagicMock],
    mock_env: MockEnvironment,
    method: str,
    path: str,
    json_body: dict[str, object],
    active_job: str | None,
    current_env: str,
    expected_status: int,
) -> None:
    """Guard rejects set-variables/evaluate when the target env is not the current
    environment (412), or when a job is active (409).

    The status code alone would stay green if the guard ran *after* the hardware
    call, so also pin that no setpoint reached the environment.
    """
    client, tracker = client_with_tracker
    tracker.get_active_job.return_value = active_job
    tracker.get_current_environment.return_value = current_env
    set_variables, evaluate = MagicMock(), MagicMock()
    mock_env.set_variables = set_variables  # type: ignore[method-assign]
    mock_env.evaluate = evaluate  # type: ignore[method-assign]
    resp = getattr(client, method)(path, json=json_body)
    assert resp.status_code == expected_status
    set_variables.assert_not_called()
    evaluate.assert_not_called()


@pytest.mark.parametrize(
    "active_job,current_env,expected_status",
    [
        (None, "als_injector", 412),  # run's env is not the current environment
        ("job-1", "als_injector_sim", 409),  # a job is active
    ],
)
def test_restore_rejected_when_not_current_or_job_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    active_job: str | None,
    current_env: str,
    expected_status: int,
) -> None:
    """Guard rejects restore when the run's env is not the current environment,
    or when a job is active."""
    monkeypatch.chdir(tmp_path)
    run_id = make_run_dir(tmp_path, env_name="als_injector_sim")
    registry, recorded = recording_registry()
    tracker = MagicMock(spec=JobTracker)
    tracker.get_active_job.return_value = active_job
    tracker.get_current_environment.return_value = current_env
    app.dependency_overrides[get_environment_registry] = lambda: registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    try:
        with TestClient(app) as client:
            resp = client.post(f"/results/{run_id}/restore")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == expected_status
    assert recorded == {}


def test_put_variables_override_bypasses_unhealthy_503(
    client_with_tracker: tuple[TestClient, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """override_unhealthy=True bypasses the 503 health gate for a non-sim env."""
    client, tracker = client_with_tracker
    tracker.get_active_job.return_value = None
    tracker.get_current_environment.return_value = "als_injector"
    monkeypatch.setattr(environment_ops, "_environment_connected", lambda n, r: False)

    # Without override the unhealthy non-sim env is refused.
    resp = client.put("/environments/als_injector/variables", json={"variables": {"var1": 3.14}})
    assert resp.status_code == 503

    # With override the write proceeds (var1 is in the mock env's variables).
    resp = client.put(
        "/environments/als_injector/variables",
        json={"variables": {"var1": 3.14}, "override_unhealthy": True},
    )
    assert resp.status_code == 204
