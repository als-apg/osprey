"""Tests for the results router in api/routers/results.py.

Covers: GET /results, GET /results/{run_id}, POST /results/{run_id}/restore,
        GET /results/{run_id}/restore-preview.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from osprey_optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.api.main import app
from osprey_optimizer.api.routers import results as results_mod
from osprey_optimizer.models.api import ResultInfo

from .conftest import make_run_dir, recording_registry


@contextmanager
def restore_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_name: str, *, with_initial: bool = True
) -> Generator[tuple[TestClient, str, dict[str, float]], None, None]:
    """Install the registry + tracker overrides shared by the restore-flow tests.

    Writes a run dir under ``tmp_path`` for ``env_name`` (complete unless
    ``with_initial=False``, which omits the initial snapshot). The run records the
    virtual accelerator as its target when ``env_name`` ends in ``_sim`` and the
    live machine otherwise, keeping these tests' sim/real cases. Yields
    ``(client, run_id, recorded)`` where ``recorded`` accumulates ``set_variables``
    writes. Clears the dependency overrides on exit.
    """
    monkeypatch.chdir(tmp_path)
    target = "va" if env_name.endswith("_sim") else "live"
    run_id = make_run_dir(tmp_path, env_name=env_name, with_initial=with_initial, target=target)
    registry, recorded = recording_registry()
    tracker = MagicMock(spec=JobTracker)
    tracker.get_active_job.return_value = None
    app.dependency_overrides[get_environment_registry] = lambda: registry
    app.dependency_overrides[get_job_tracker] = lambda: tracker
    try:
        with TestClient(app) as client:
            yield client, run_id, recorded
    finally:
        app.dependency_overrides.clear()


class TestListResultsEndpoint:
    def test_success(self, client: TestClient) -> None:
        fake = [
            ResultInfo(
                run_id="env/2024-01-01_00-00-00",
                environment="env",
                timestamp="2024-01-01_00-00-00",
                objective_name="obj",
                has_model=False,
            ),
            ResultInfo(
                run_id="env/2024-01-02_00-00-00",
                environment="env",
                timestamp="2024-01-02_00-00-00",
                objective_name=None,
                has_model=True,
            ),
        ]
        with patch(
            "osprey_optimizer.api.routers.results.scan_result_infos",
            return_value=fake,
        ):
            resp = client.get("/results")
        assert resp.status_code == 200
        # Full ResultInfo wire shape, both rows, in scan order.
        assert resp.json() == [
            {
                "run_id": "env/2024-01-01_00-00-00",
                "environment": "env",
                "timestamp": "2024-01-01_00-00-00",
                "objective_name": "obj",
                "has_model": False,
                "job_id": None,
            },
            {
                "run_id": "env/2024-01-02_00-00-00",
                "environment": "env",
                "timestamp": "2024-01-02_00-00-00",
                "objective_name": None,
                "has_model": True,
                "job_id": None,
            },
        ]

    def test_exception(self, client: TestClient) -> None:
        with patch(
            "osprey_optimizer.api.routers.results.scan_result_infos",
            side_effect=OSError("disk error"),
        ):
            resp = client.get("/results")
        assert resp.status_code == 503


class TestGetResultDetailsEndpoint:
    def _make_load_return(self) -> tuple[pl.DataFrame, dict[str, float], dict[str, Any]]:
        data = pl.DataFrame({"var1": [1.0, 2.0], "obj": [0.5, 0.9]})
        best_point = {"var1": 2.0}
        config: dict[str, Any] = {"algorithm": "random", "snapshots": {"initial": {"var1": 1.0}}}
        return data, best_point, config

    def test_success(self, client: TestClient) -> None:
        data, best, cfg = self._make_load_return()
        with patch("osprey_optimizer.results.load_results", return_value=(data, best, cfg)):
            resp = client.get("/results/env/2024-01-01_00-00-00")
        assert resp.status_code == 200
        body = resp.json()
        assert body["timestamp"] == "env/2024-01-01_00-00-00"
        assert body["best_point"] == best
        assert body["initial_point"] == {"var1": 1.0}
        assert len(body["data_records"]) == 2

    def test_not_found(self, client: TestClient) -> None:
        with patch(
            "osprey_optimizer.results.load_results",
            side_effect=FileNotFoundError("gone"),
        ):
            resp = client.get("/results/env/missing")
        assert resp.status_code == 404

    def test_error(self, client: TestClient) -> None:
        with patch(
            "osprey_optimizer.results.load_results",
            side_effect=RuntimeError("corrupt"),
        ):
            resp = client.get("/results/env/bad")
        assert resp.status_code == 500

    def test_path_traversal_rejected(self, client: TestClient) -> None:
        """A traversal identifier is rejected before load_results is ever called."""
        with patch("osprey_optimizer.results.load_results") as mock_load:
            resp = client.get("/results/..%2f..%2f..%2f..%2fetc%2fpasswd")
        assert resp.status_code == 404
        mock_load.assert_not_called()


def test_restore_sim_env_proceeds_without_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D2 no-regression: a sim-env restore writes with NO confirm_token."""
    with restore_overrides(monkeypatch, tmp_path, "als_injector_sim") as (client, run_id, recorded):
        resp = client.post(f"/results/{run_id}/restore")

    assert resp.status_code == 200
    body = resp.json()
    assert body["environment"] == "als_injector_sim"
    assert body["restored"] == {"v1": 1.5, "v2": 3.0}  # objective dropped
    assert body["count"] == 2
    assert recorded == {"v1": 1.5, "v2": 3.0}


@pytest.mark.parametrize(
    ("query", "detail_token"),
    [
        pytest.param("", "restore-preview", id="without_token"),
        pytest.param("?confirm_token=deadbeef", None, id="wrong_token"),
    ],
)
def test_restore_real_env_unconfirmed_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, query: str, detail_token: str | None
) -> None:
    """D1: a non-sim restore with no confirm_token, or a stale/mismatched one, is
    rejected and writes nothing to the machine."""
    with restore_overrides(monkeypatch, tmp_path, "als_injector") as (client, run_id, recorded):
        resp = client.post(f"/results/{run_id}/restore{query}")
    assert resp.status_code == 428  # PRECONDITION_REQUIRED
    if detail_token is not None:
        assert detail_token in resp.json()["detail"]
    assert recorded == {}  # no write happened


def test_restore_real_env_matching_token_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D1: a non-sim restore WITH the previewed token passes the guard and writes."""
    with restore_overrides(monkeypatch, tmp_path, "als_injector") as (client, run_id, recorded):
        preview = client.get(f"/results/{run_id}/restore-preview")
        assert preview.status_code == 200
        token = preview.json()["preview_token"]
        assert token  # non-empty
        resp = client.post(f"/results/{run_id}/restore?confirm_token={token}")
    assert resp.status_code == 200
    assert resp.json()["restored"] == {"v1": 1.5, "v2": 3.0}
    assert recorded == {"v1": 1.5, "v2": 3.0}  # guard passed, write occurred (mocked env)


def test_restore_real_env_stale_token_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D1 stale-token guard: a token valid at preview time is rejected once the
    underlying restore values change, because the guard recomputes the hash from
    the CURRENT resolved values. No write happens."""
    with restore_overrides(monkeypatch, tmp_path, "als_injector") as (client, run_id, recorded):
        preview = client.get(f"/results/{run_id}/restore-preview")
        assert preview.status_code == 200
        stale_token = preview.json()["preview_token"]
        # Mutate the persisted snapshot so _resolve_restore_values now returns a
        # different value-set; the previously captured token no longer matches.
        run_dir = tmp_path / "results" / "als_injector" / "2025-01-15_10-30-00"
        (run_dir / "initial_point.json").write_text(
            json.dumps({"v1": 9.9, "v2": 8.8, "charge_ltb": 42.0})
        )
        resp = client.post(f"/results/{run_id}/restore?confirm_token={stale_token}")
    assert resp.status_code == 428
    assert recorded == {}  # no write happened


def test_restore_unknown_run_returns_404(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with restore_overrides(monkeypatch, tmp_path, "test_env") as (client, _, _):
        resp = client.post("/results/test_env/does-not-exist/restore")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "config_snapshots",
    [
        pytest.param(None, id="missing_snapshot"),
        pytest.param("corrupt", id="non_dict_snapshots_in_config"),
    ],
)
def test_restore_missing_or_non_dict_snapshot_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_snapshots: str | None
) -> None:
    """A run with no initial snapshot 409s; likewise a corrupt config.json whose
    'snapshots' key is not a dict must surface as the usual 409, not an
    AttributeError -> 500."""
    with restore_overrides(monkeypatch, tmp_path, "test_env", with_initial=False) as (
        client,
        run_id,
        _,
    ):
        if config_snapshots is not None:
            config = tmp_path / "results" / "test_env" / "2025-01-15_10-30-00" / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "environment_name": "test_env",
                        "objective_name": "charge_ltb",
                        "variables": ["v1", "v2"],
                        "snapshots": config_snapshots,
                    }
                )
            )
        resp = client.post(f"/results/{run_id}/restore")
    assert resp.status_code == 409


def test_restore_incomplete_snapshot_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with restore_overrides(monkeypatch, tmp_path, "test_env") as (client, run_id, _):
        initial = tmp_path / "results" / "test_env" / "2025-01-15_10-30-00" / "initial_point.json"
        initial.write_text(json.dumps({"v1": 1.5, "charge_ltb": 42.0}))  # v2 missing
        resp = client.post(f"/results/{run_id}/restore")
    assert resp.status_code == 409
    assert "v2" in resp.json()["detail"]


@pytest.mark.parametrize(
    "bad_value",
    ["1.5", float("nan"), float("inf"), True, None],
    ids=["string", "nan", "inf", "bool", "null"],
)
def test_restore_non_numeric_snapshot_value_returns_409_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_value: Any
) -> None:
    """Snapshot values come from json.load (Any); a value that is not a finite real
    number must be rejected with 409 BEFORE any hardware write, not forwarded to
    set_variables on the strength of the dict[str, float] annotation."""
    with restore_overrides(monkeypatch, tmp_path, "als_injector_sim") as (client, run_id, recorded):
        initial = (
            tmp_path / "results" / "als_injector_sim" / "2025-01-15_10-30-00" / "initial_point.json"
        )
        initial.write_text(json.dumps({"v1": bad_value, "v2": 3.0, "charge_ltb": 42.0}))
        resp = client.post(f"/results/{run_id}/restore")
    assert resp.status_code == 409
    assert "v1" in resp.json()["detail"]
    assert recorded == {}


def test_restore_non_dict_initial_snapshot_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An initial snapshot that parses to a non-dict (e.g. a list) is corrupt data,
    rejected with 409 before any value lookup."""
    with restore_overrides(monkeypatch, tmp_path, "als_injector_sim") as (client, run_id, recorded):
        initial = (
            tmp_path / "results" / "als_injector_sim" / "2025-01-15_10-30-00" / "initial_point.json"
        )
        initial.write_text(json.dumps([1.5, 3.0]))
        resp = client.post(f"/results/{run_id}/restore")
    assert resp.status_code == 409
    assert recorded == {}


def test_restore_preview_returns_tuned_values_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin that the preview sub-resource and the detail route stay disjoint.

    Detail is ``GET /results/{env}/{timestamp}`` (exactly two non-greedy segments)
    and preview is ``GET /results/{env}/{timestamp}/restore-preview`` (three).
    Because the converters are non-greedy ``str`` (not ``:path``), the three-segment
    preview URL can never be matched by the two-segment detail route, independent
    of declaration order. If anyone reintroduces a greedy ``{...:path}`` on detail,
    it would swallow this request and load a run named ``<id>/restore-preview``
    (404), and the preview body would disappear — this test catches that.
    """
    monkeypatch.chdir(tmp_path)
    run_id = make_run_dir(tmp_path)
    with TestClient(app) as client:
        resp = client.get(f"/results/{run_id}/restore-preview")
        # The preview token is a deterministic hash of the resolved values.
        second = client.get(f"/results/{run_id}/restore-preview")

    assert resp.status_code == 200
    body = resp.json()
    assert body["preview_token"] == second.json()["preview_token"]  # stable across calls
    # Preview shape (RestorePreviewResponse), not HistoricalRunDetails — the latter
    # would carry config/data_records and lack environment/values/count.
    assert set(body) == {"environment", "control_target", "values", "count", "preview_token"}
    assert body["environment"] == "test_env"
    assert body["control_target"] == "va"
    assert body["values"] == {"v1": 1.5, "v2": 3.0}  # objective dropped
    assert body["count"] == 2
    # The GET endpoint has no environment-registry dependency, so it cannot
    # write to the machine — the no-write guarantee is structural.


def test_restore_preview_unknown_run_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/results/test_env/does-not-exist/restore-preview")
    assert resp.status_code == 404


def test_restore_preview_missing_snapshot_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    run_id = make_run_dir(tmp_path, with_initial=False)
    with TestClient(app) as client:
        resp = client.get(f"/results/{run_id}/restore-preview")
    assert resp.status_code == 409


def test_restore_preview_incomplete_snapshot_returns_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "results" / "test_env" / "2025-01-15_10-30-00"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "environment_name": "test_env",
                "control_target": "va",
                "objective_name": "charge_ltb",
                "variables": ["v1", "v2"],
            }
        )
    )
    (run_dir / "initial_point.json").write_text(json.dumps({"v1": 1.5, "charge_ltb": 42.0}))
    with TestClient(app) as client:
        resp = client.get("/results/test_env/2025-01-15_10-30-00/restore-preview")
    assert resp.status_code == 409
    assert "v2" in resp.json()["detail"]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: results_mod._resolve_restore_values("../../etc/passwd"),
            id="resolve_restore_values",
        ),
        pytest.param(
            lambda: results_mod.get_result_details(env="..", timestamp=".."),
            id="get_result_details",
        ),
    ],
)
def test_path_traversal_rejected_before_any_read(call: Any) -> None:
    """A run identifier escaping the results dir is rejected as 404 BEFORE any
    filesystem read — each guard must short-circuit, not rely on load_results
    happening to fail (which it would for a nonexistent path anyway)."""
    with (
        patch.object(results_mod, "load_results") as load,
        pytest.raises(HTTPException) as ei,
    ):
        call()
    assert ei.value.status_code == 404
    load.assert_not_called()


def test_resolve_restore_values_missing_env_or_vars_409() -> None:
    """A run with an initial snapshot but no environment/variables -> 409."""
    cfg = {"snapshots": {"initial": {"v1": 1.0}}, "environment_name": None, "variables": None}
    with (
        patch.object(results_mod, "load_results", return_value=(None, None, cfg)),
        pytest.raises(HTTPException) as ei,
    ):
        results_mod._resolve_restore_values("env/run")
    assert ei.value.status_code == 409
    assert "missing environment or variables" in ei.value.detail
