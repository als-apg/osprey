"""Tests for the run config router in api/routers/run_config.py.

Covers: POST /environments/{name}/resolve-bounds,
        GET /environments/{name}/initial-bounds,
        GET /environments/{name}/presets.
"""

from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.services.optimizer.models.config import Preset
from osprey.services.optimizer.models.exceptions import TimeoutError as OptimizerTimeoutError
from osprey.services.optimizer.models.protocols import (
    PVReading,
    VariableMetadata,
)

from ._fakes import MockEnvironment


class TestResolveBoundsEndpoint:
    @pytest.mark.parametrize(
        "overrides,expected_min,expected_max,expected_clamped,expected_requested_min,expected_requested_max",
        [
            ({}, 0.0, 10.0, False, 0.0, 10.0),
            ({"var1": {"window_fraction": 0.1}}, 4.5, 5.5, False, 4.5, 5.5),
            ({"var1": {"window_abs": 2.0}}, 3.0, 7.0, False, 3.0, 7.0),
            ({"var1": {"min": 2.0}}, 2.0, 10.0, False, 2.0, 10.0),
            ({"var1": {"min": 20.0, "max": 30.0}}, 10.0, 10.0, True, 20.0, 30.0),
            ({"var1": {"min": -5.0, "max": 20.0}}, 0.0, 10.0, True, -5.0, 20.0),
        ],
        ids=[
            "no_overrides_returns_safety_bounds",
            "window_fraction_centers_on_current",
            "window_abs_centers_on_current",
            "min_only_falls_back_to_safety_max",
            "disjoint_override_collapses_to_safety_endpoint",
            "override_clamped_against_safety",
        ],
    )
    def test_resolve_bounds_response_shape(
        self,
        client: TestClient,
        overrides: dict[str, Any],
        expected_min: float,
        expected_max: float,
        expected_clamped: bool,
        expected_requested_min: float,
        expected_requested_max: float,
    ) -> None:
        """The response echoes what was asked for (``requested_*``) alongside what
        was granted, so a clamp against the safety range is visible to the caller."""
        resp = client.post(
            "/environments/sim/resolve-bounds",
            json={"overrides": overrides, "current_values": {"var1": 5.0}},
        )
        assert resp.status_code == 200
        assert "current_values" not in resp.json()  # the request's values are not echoed back
        bounds = resp.json()["bounds"]["var1"]
        assert bounds["min"] == pytest.approx(expected_min)
        assert bounds["max"] == pytest.approx(expected_max)
        assert bounds["was_clamped"] is expected_clamped
        assert bounds["requested_min"] == pytest.approx(expected_requested_min)
        assert bounds["requested_max"] == pytest.approx(expected_requested_max)

    def test_non_finite_override_bound_returns_serializable_422(self, client: TestClient) -> None:
        """A NaN bound (sent via the ``NaN`` JSON literal stdlib json accepts) is
        rejected by VariableOverride. The 422 body echoes the offending input, so
        without the non-finite sanitizer in the app's validation handler the
        response itself fails to serialize and degrades to a 500."""
        resp = client.post(
            "/environments/sim/resolve-bounds",
            content=b'{"overrides": {"var1": {"min": NaN}}, "current_values": {"var1": 5.0}}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        # Body must be valid JSON (the whole point — no 500 from an unserializable
        # nan) and report the finiteness failure.
        body = resp.json()
        assert any("finite" in str(err.get("msg", "")) for err in body["detail"])

    @pytest.mark.parametrize(
        "overrides,current_values,expected_prefix,expected_token",
        [
            ({"var1": {"min": 1.0}}, {}, "Missing current_value", "var1"),
            ({}, {"var1": 5.0, "ghost": 99.0}, "Unknown current_value", "ghost"),
            ({"ghost": {"min": 0.0, "max": 1.0}}, {"var1": 5.0}, "Unknown override", "ghost"),
        ],
        ids=[
            "override_without_current_value",
            "unknown_current_value_key",
            "unknown_override_variable",
        ],
    )
    def test_resolve_bounds_422_validation_errors(
        self,
        client: TestClient,
        overrides: dict[str, Any],
        current_values: dict[str, Any],
        expected_prefix: str,
        expected_token: str,
    ) -> None:
        resp = client.post(
            "/environments/sim/resolve-bounds",
            json={"overrides": overrides, "current_values": current_values},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail.startswith(expected_prefix)
        assert expected_token in detail

    def test_resolve_bounds_value_error_maps_to_422(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        """``_resolve_one_bound`` maps a ValueError out of resolve_bounds to 422."""
        var = next(iter(mock_env.variables))
        with patch(
            "osprey.services.optimizer.api.routers.run_config.resolve_bounds",
            side_effect=ValueError("bad window"),
        ):
            resp = client.post(
                "/environments/sim/resolve-bounds",
                json={
                    "overrides": {var: {"window_fraction": 0.1}},
                    "current_values": {var: 1.0},
                },
            )
        assert resp.status_code == 422
        assert "bad window" in resp.json()["detail"]

    def test_subset_of_multivariable_env_resolves_only_requested(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        """A single-variable cell edit need only supply that variable's current_value."""
        two_vars = {
            "var1": mock_env.variables["var1"],
            "var2": VariableMetadata(
                min_value=0.0,
                max_value=100.0,
                units="V",
                description="Var 2",
                readback_pv="V2:RBV",
                setpoint_pv="V2:SP",
            ),
        }
        with patch.object(
            type(mock_env), "variables", new_callable=PropertyMock, return_value=two_vars
        ):
            resp = client.post(
                "/environments/sim/resolve-bounds",
                json={
                    "overrides": {"var1": {"min": -5.0, "max": 20.0}},
                    "current_values": {"var1": 5.0},
                },
            )
        assert resp.status_code == 200
        bounds = resp.json()["bounds"]
        assert set(bounds) == {"var1"}
        assert bounds["var1"]["was_clamped"] is True

    def test_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        mock_registry.load_environment.side_effect = KeyError("nope")
        resp = client.post(
            "/environments/nope/resolve-bounds",
            json={"overrides": {}, "current_values": {"var1": 5.0}},
        )
        assert resp.status_code == 404


class TestInitialBoundsEndpoint:
    @pytest.mark.parametrize(
        ("preset", "expected_min", "expected_max", "expected_clamped"),
        [
            # mock env: var1 safety [0,10], current 5.0, fraction 0.1 → [4.5, 5.5]
            pytest.param(
                {"variable_overrides": {"var1": {"window_fraction": 0.1}}},
                4.5,
                5.5,
                False,
                id="preset_overrides_applied",
            ),
            pytest.param(None, 0.0, 10.0, False, id="no_preset_passes_safety_through"),
            pytest.param(
                {"variable_overrides": {"var1": {"min": 50.0, "max": 60.0}}},
                10.0,
                10.0,
                True,
                id="disjoint_override_collapses_to_safety_endpoint",
            ),
        ],
    )
    def test_initial_bounds_from_preset(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        preset: dict[str, Any] | None,
        expected_min: float,
        expected_max: float,
        expected_clamped: bool,
    ) -> None:
        """A preset's variable_overrides drive the window; without a preset the
        safety range passes through; an override entirely outside safety collapses
        to the nearest safety endpoint."""
        params = {}
        if preset is not None:
            mock_registry.get_presets.return_value = {"p": Preset.model_validate(preset)}
            params["preset"] = "p"
        resp = client.get("/environments/sim/initial-bounds", params=params)
        assert resp.status_code == 200
        body = resp.json()
        bounds = body["bounds"]["var1"]
        assert bounds["min"] == pytest.approx(expected_min)
        assert bounds["max"] == pytest.approx(expected_max)
        assert bounds["was_clamped"] is expected_clamped
        assert body["current_values"]["var1"]["value"] == 5.0

    def test_unknown_preset_returns_404(self, client: TestClient, mock_registry: MagicMock) -> None:
        mock_registry.get_presets.return_value = {}
        resp = client.get("/environments/sim/initial-bounds", params={"preset": "nope"})
        assert resp.status_code == 404
        assert "no preset 'nope'" in resp.json()["detail"]

    def test_preset_naming_an_unknown_variable_returns_422(
        self, client: TestClient, mock_registry: MagicMock
    ) -> None:
        mock_registry.get_presets.return_value = {"p": Preset(variables=["ghost"])}
        resp = client.get("/environments/sim/initial-bounds", params={"preset": "p"})
        assert resp.status_code == 422
        assert "ghost" in resp.json()["detail"]

    def test_unknown_env_returns_404(self, client: TestClient, mock_registry: MagicMock) -> None:
        mock_registry.load_environment.side_effect = KeyError("nope")
        resp = client.get("/environments/nope/initial-bounds")
        assert resp.status_code == 404

    def test_ca_worker_unresponsive_returns_503(
        self, client: TestClient, mock_env: MockEnvironment
    ) -> None:
        """An unreachable/slow control system surfaces as 503, not a generic 500."""
        mock_env.get_variables = MagicMock(side_effect=OptimizerTimeoutError("worker wedged"))  # type: ignore[method-assign]
        resp = client.get("/environments/sim/initial-bounds")
        assert resp.status_code == 503

    def test_initial_bounds_get_variables_error_500(
        self, client: TestClient, mock_registry: MagicMock, mock_env: MockEnvironment
    ) -> None:
        """An unexpected get_variables error while reading current values -> 500."""
        with patch.object(mock_env, "get_variables", side_effect=RuntimeError("read broke")):
            resp = client.get("/environments/sim/initial-bounds")
        assert resp.status_code == 500
        assert "Failed to read current values" in resp.json()["detail"]

    def test_unreadable_variable_falls_back_to_safety(
        self, client: TestClient, mock_registry: MagicMock, mock_env: MockEnvironment
    ) -> None:
        """A None readback (NaN) falls back to the safety range; the table is still usable."""
        mock_registry.get_presets.return_value = {
            "p": Preset(variable_overrides={"var1": {"window_fraction": 0.1}})
        }

        def _none_readback(variable_names: list[str]) -> dict[str, PVReading]:
            return {n: PVReading(value=None, connected=False) for n in variable_names}

        mock_env.get_variables = _none_readback  # type: ignore[method-assign]
        resp = client.get("/environments/sim/initial-bounds", params={"preset": "p"})
        assert resp.status_code == 200
        body = resp.json()
        bounds = body["bounds"]["var1"]
        assert bounds["min"] == 0.0
        assert bounds["max"] == 10.0
        assert bounds["was_clamped"] is False
        assert body["current_values"]["var1"]["value"] is None
        assert body["current_values"]["var1"]["connected"] is False


class TestPresetsEndpoint:
    def test_presets_are_returned_by_name(
        self, client: TestClient, mock_registry: MagicMock
    ) -> None:
        mock_registry.get_presets.return_value = {
            "phase_tuning": Preset(algorithm="extremum_seeking", n_iterations=100)
        }
        resp = client.get("/environments/sim/presets")
        assert resp.status_code == 200
        body = resp.json()
        assert list(body) == ["phase_tuning"]
        assert body["phase_tuning"]["algorithm"] == "extremum_seeking"
        assert body["phase_tuning"]["n_iterations"] == 100
        mock_registry.get_presets.assert_called_once_with("sim")

    def test_no_presets_is_an_empty_mapping(
        self, client: TestClient, mock_registry: MagicMock
    ) -> None:
        mock_registry.get_presets.return_value = {}
        resp = client.get("/environments/sim/presets")
        assert resp.status_code == 200
        assert resp.json() == {}

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            pytest.param(KeyError("nope"), 404, id="unknown_env"),
            pytest.param(ValueError("bad"), 422, id="bad_plugin_presets"),
        ],
    )
    def test_errors(
        self, client: TestClient, mock_registry: MagicMock, error: Exception, code: int
    ) -> None:
        mock_registry.get_presets.side_effect = error
        resp = client.get("/environments/nope/presets")
        assert resp.status_code == code
