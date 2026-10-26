"""Tests for the pluggable surrogate model system.

Covers SklearnBoTorchModel (wrapper) and SklearnModelConstructor (constructor).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim import optimize_acqf
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from osprey_optimizer.models.enums import SurrogateModelType
from osprey_optimizer.optimization.surrogate import constructor
from osprey_optimizer.optimization.surrogate.constructor import (
    SklearnModelConstructor,
    load_model_metadata,
    pack_model_bundle,
    unpack_model_bundle,
)
from osprey_optimizer.optimization.surrogate.wrapper import SklearnBoTorchModel


def _bump_mtime(path: Path, seconds: int = 1) -> None:
    """Advance a file's mtime so a (path, mtime) cache key is guaranteed to differ."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


@pytest.fixture
def train_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple training data: 20 samples, 3 features."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (20, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, 20)
    return X, y


@pytest.fixture
def fitted_rf(train_data: tuple[np.ndarray, np.ndarray]) -> RandomForestRegressor:
    X, y = train_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_et(train_data: tuple[np.ndarray, np.ndarray]) -> ExtraTreesRegressor:
    X, y = train_data
    model = ExtraTreesRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_gp(train_data: tuple[np.ndarray, np.ndarray]) -> GaussianProcessRegressor:
    X, y = train_data
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    model = GaussianProcessRegressor(kernel=kernel, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_gp_pipeline(train_data: tuple[np.ndarray, np.ndarray]) -> Pipeline:
    X, y = train_data
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("gp", gp)])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def fitted_model(
    request: pytest.FixtureRequest,
    fitted_rf: RandomForestRegressor,
    fitted_et: ExtraTreesRegressor,
    fitted_gp: GaussianProcessRegressor,
    fitted_gp_pipeline: Pipeline,
) -> RandomForestRegressor | ExtraTreesRegressor | GaussianProcessRegressor | Pipeline:
    """Indirect-parametrize entry point: select one of the fitted sklearn models by name."""
    return {
        "fitted_rf": fitted_rf,
        "fitted_et": fitted_et,
        "fitted_gp": fitted_gp,
        "fitted_gp_pipeline": fitted_gp_pipeline,
    }[request.param]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame(
        {
            "x0": rng.uniform(0, 1, n),
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(0, 1, n),
            "y": rng.uniform(0, 1, n),
        }
    )


class TestSklearnBoTorchModelPosterior:
    """Posterior shape/variance over RF, ExtraTrees, GP, and GP-pipeline backends."""

    @pytest.mark.parametrize(
        ("fitted_model", "variance_method"),
        [
            pytest.param("fitted_rf", "ensemble", id="rf"),
            pytest.param("fitted_et", "ensemble", id="extra_trees"),
            pytest.param("fitted_gp", "return_std", id="gp"),
            # a Pipeline is unwrapped to its inner estimator before the strategy is picked
            pytest.param("fitted_gp_pipeline", "return_std", id="gp_pipeline"),
        ],
        indirect=["fitted_model"],
    )
    def test_posterior_shape_and_variance(self, fitted_model: object, variance_method: str) -> None:
        model = SklearnBoTorchModel(fitted_model)
        assert model._variance_method == variance_method
        assert model.num_outputs == 1
        X = torch.rand(5, 3, dtype=torch.double)
        posterior = model.posterior(X)
        assert posterior.mean.shape == (5, 1)
        assert (posterior.variance > 0).all()


class TestSklearnBoTorchModelGradients:
    """Test that posterior is differentiable for acquisition optimization."""

    @pytest.mark.parametrize("fitted_model", ["fitted_rf", "fitted_gp"], indirect=True)
    def test_posterior_mean_has_grad_fn(self, fitted_model: object) -> None:
        """Posterior mean must be differentiable w.r.t. X (required by BoTorch optimizers)."""
        model = SklearnBoTorchModel(fitted_model)
        X = torch.rand(5, 3, dtype=torch.double, requires_grad=True)
        posterior = model.posterior(X)
        loss = posterior.mean.sum()
        loss.backward()
        assert X.grad is not None
        assert X.grad.shape == (5, 3)

    @pytest.mark.parametrize("fitted_model", ["fitted_rf", "fitted_gp"], indirect=True)
    def test_acquisition_optimization(self, fitted_model: object) -> None:
        """End-to-end: optimize_acqf must not raise with SklearnBoTorchModel."""
        model = SklearnBoTorchModel(fitted_model)
        acqf = LogExpectedImprovement(model=model, best_f=0.5)
        bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

        candidate, value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
        )
        assert candidate.shape == (1, 3)
        assert ((candidate >= bounds[0]) & (candidate <= bounds[1])).all()
        assert value.numel() == 1


class TestSklearnBoTorchModelRejection:
    """Test that unsupported models are rejected."""

    def test_unsupported_model_raises(self) -> None:
        class DummyModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X))

        with pytest.raises(ValueError, match="does not support uncertainty"):
            SklearnBoTorchModel(DummyModel())


class TestSklearnModelConstructor:
    """Tests for SklearnModelConstructor.build_model()."""

    @pytest.mark.parametrize(
        ("model_type", "expected_inner"),
        [
            (SurrogateModelType.RANDOM_FOREST, RandomForestRegressor),
            (SurrogateModelType.EXTRA_TREES, ExtraTreesRegressor),
            (SurrogateModelType.GAUSSIAN_PROCESS, GaussianProcessRegressor),
        ],
    )
    def test_build_model_fresh(
        self, sample_df: pd.DataFrame, model_type: SurrogateModelType, expected_inner: type
    ) -> None:
        constructor = SklearnModelConstructor(surrogate_model_type=model_type)
        model_list = constructor.build_model(
            input_names=["x0", "x1", "x2"],
            outcome_names=["y"],
            data=sample_df,
        )
        assert len(model_list.models) == 1
        # The REQUESTED estimator type was built (not whatever the default is).
        wrapper = model_list.models[0]
        assert isinstance(wrapper, SklearnBoTorchModel)
        assert type(wrapper._inner) is expected_inner

        X = torch.rand(3, 3, dtype=torch.double)
        posterior = wrapper.posterior(X)
        assert posterior.mean.shape == (3, 1)
        assert (posterior.variance > 0).all()

    def test_build_model_pretrained_bundle(
        self, tmp_path: Path, fitted_rf: RandomForestRegressor, sample_df: pd.DataFrame
    ) -> None:
        """A model_file is loaded by extracting the sklearn model from its bundle dict."""
        model_path = tmp_path / "model.joblib"
        joblib.dump(pack_model_bundle(fitted_rf, ["x0", "x1", "x2"]), model_path)

        constructor = SklearnModelConstructor(
            surrogate_model_type=SurrogateModelType.RANDOM_FOREST,
            model_file=str(model_path),
        )
        model_list = constructor.build_model(
            input_names=["x0", "x1", "x2"],
            outcome_names=["y"],
            data=sample_df,
        )
        assert len(model_list.models) == 1

        # The wrapped model IS the pretrained one, not a fresh fit on sample_df.
        wrapper = model_list.models[0]
        assert isinstance(wrapper, SklearnBoTorchModel)
        probe = np.zeros((3, 3))
        np.testing.assert_array_equal(
            wrapper.sklearn_model.predict(probe), fitted_rf.predict(probe)
        )

        X = torch.rand(3, 3, dtype=torch.double)
        posterior = wrapper.posterior(X)
        assert posterior.mean.shape == (3, 1)

    def test_pretrained_bundle_loaded_once_across_builds(
        self,
        tmp_path: Path,
        fitted_rf: RandomForestRegressor,
        sample_df: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """joblib.load runs once per (path, mtime), not once per build_model call."""
        model_path = tmp_path / "model.joblib"
        joblib.dump(pack_model_bundle(fitted_rf, ["x0", "x1", "x2"]), model_path)

        calls = []
        real_load = joblib.load

        def counting_load(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return real_load(*args, **kwargs)

        monkeypatch.setattr(joblib, "load", counting_load)
        constructor = SklearnModelConstructor(
            surrogate_model_type=SurrogateModelType.RANDOM_FOREST,
            model_file=str(model_path),
        )
        for _ in range(2):
            constructor.build_model(
                input_names=["x0", "x1", "x2"],
                outcome_names=["y"],
                data=sample_df,
            )
        assert len(calls) == 1

    def test_invalid_model_type_raises(self, sample_df: pd.DataFrame) -> None:
        constructor = SklearnModelConstructor(surrogate_model_type=SurrogateModelType.XOPT_GP)
        with pytest.raises(ValueError, match="No factory"):
            constructor.build_model(
                input_names=["x0", "x1", "x2"],
                outcome_names=["y"],
                data=sample_df,
            )

    def test_multiple_outcomes(self, sample_df: pd.DataFrame) -> None:
        """Each outcome gets its OWN fit — hoisting ``y`` out of the per-outcome
        loop would fit the constraint model on the objective column and steer BO
        toward constraint-violating setpoints."""
        sample_df["y2"] = sample_df["y"] * 2
        constructor = SklearnModelConstructor(surrogate_model_type=SurrogateModelType.RANDOM_FOREST)
        model_list = constructor.build_model(
            input_names=["x0", "x1", "x2"],
            outcome_names=["y", "y2"],
            data=sample_df,
        )
        assert len(model_list.models) == 2

        # y2 is exactly 2*y and the forests are seeded, so the second model must
        # predict exactly double the first — not the same column twice.
        objective_model = model_list.models[0]
        constraint_model = model_list.models[1]
        assert isinstance(objective_model, SklearnBoTorchModel)
        assert isinstance(constraint_model, SklearnBoTorchModel)
        probe = np.zeros((3, 3))
        np.testing.assert_allclose(
            constraint_model.sklearn_model.predict(probe),
            2 * objective_model.sklearn_model.predict(probe),
            rtol=1e-6,
        )


class TestPackUnpackModelBundle:
    @pytest.mark.parametrize(
        "bundle",
        [
            pytest.param(RandomForestRegressor(n_estimators=5), id="raw_sklearn_object"),
            pytest.param({"model": "fake"}, id="dict_without_variables_key"),
            pytest.param({"model": "fake", "variables": "not-a-list"}, id="variables_not_a_list"),
        ],
    )
    def test_unpack_rejects_non_bundle(self, bundle: Any) -> None:
        with pytest.raises(ValueError, match="Not a model bundle"):
            unpack_model_bundle(bundle)

    def test_roundtrip_via_joblib(self, tmp_path: Path, fitted_rf: RandomForestRegressor) -> None:
        """pack → joblib.dump → joblib.load → unpack preserves data."""
        bundle = pack_model_bundle(fitted_rf, ["x0", "x1", "x2"])
        path = tmp_path / "model.joblib"
        joblib.dump(bundle, path)

        loaded = joblib.load(path)
        model, variables = unpack_model_bundle(loaded)
        assert variables == ["x0", "x1", "x2"]
        # The round-tripped model predicts identically to the original.
        probe = np.zeros((3, 3))
        np.testing.assert_array_equal(model.predict(probe), fitted_rf.predict(probe))


class TestLoadModelMetadata:
    def test_returns_variables_and_objective_from_new_bundle(self, tmp_path: Path) -> None:
        """62-64: dict bundle yields both variable names and objective."""
        bundle = {"model": "fake", "variables": ["x", "y"], "objective": "charge"}
        path = tmp_path / "model.joblib"
        joblib.dump(bundle, path)

        variables, objective = load_model_metadata(str(path))
        assert variables == ["x", "y"]
        assert objective == "charge"

    def test_raw_sklearn_object_raises(self, tmp_path: Path) -> None:
        """A raw sklearn object is not a bundle — the load must fail loudly."""
        rf = RandomForestRegressor(n_estimators=3)
        path = tmp_path / "model.joblib"
        joblib.dump(rf, path)

        with pytest.raises(ValueError, match="Not a model bundle"):
            load_model_metadata(str(path))


class TestLoadModelMetadataSidecar:
    """Metadata is read from the .meta.json sidecar, never by unpickling (#120)."""

    def _write_bundle(
        self, path: Path, variables: list[str], objective: str | None = None
    ) -> dict[str, Any]:
        bundle: dict[str, Any] = {
            "model": RandomForestRegressor(n_estimators=3),
            "variables": variables,
        }
        if objective is not None:
            bundle["objective"] = objective
        joblib.dump(bundle, path)
        return bundle

    def _forbid_unpickling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make any bundle read fail loudly, so a sidecar hit is proven, not inferred."""

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("model bundle was unpickled")

        # Same module object constructor.py reads through.
        monkeypatch.setattr(joblib, "load", boom)

    def test_sidecar_serves_metadata_without_touching_the_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The models listing is polled every second; it must never unpickle an estimator."""
        path = tmp_path / "model.joblib"
        bundle = self._write_bundle(path, ["x", "y"], objective="charge")
        constructor.write_model_metadata(path, bundle)
        self._forbid_unpickling(monkeypatch)

        for _ in range(5):
            assert load_model_metadata(str(path)) == (["x", "y"], "charge")

    def test_legacy_bundle_is_read_once_then_backfilled(self, tmp_path: Path) -> None:
        """A bundle predating sidecars heals itself instead of unpickling on every poll."""
        path = tmp_path / "model.joblib"
        self._write_bundle(path, ["x", "y"])
        assert not constructor.model_metadata_path(path).exists()

        assert load_model_metadata(str(path)) == (["x", "y"], None)

        assert json.loads(constructor.model_metadata_path(path).read_text()) == {
            "variables": ["x", "y"],
            "objective": None,
        }

    def test_backfill_failure_still_returns_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read-only results directory must degrade to slow, not to broken."""
        path = tmp_path / "model.joblib"
        self._write_bundle(path, ["x"])

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(constructor, "write_model_metadata", refuse)

        assert load_model_metadata(str(path)) == (["x"], None)

    def test_sidecar_older_than_its_bundle_is_ignored(self, tmp_path: Path) -> None:
        """A model swapped on disk must not be described by the previous one's sidecar."""
        path = tmp_path / "model.joblib"
        bundle = self._write_bundle(path, ["stale"])
        constructor.write_model_metadata(path, bundle)

        self._write_bundle(path, ["x", "y", "z"])
        _bump_mtime(path)

        assert load_model_metadata(str(path))[0] == ["x", "y", "z"]

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("{not json", id="unparseable"),
            pytest.param('{"objective": "charge"}', id="variables_missing"),
            pytest.param('{"variables": "x"}', id="variables_not_a_list"),
            pytest.param('{"variables": [1, 2]}', id="variables_not_names"),
            pytest.param("[]", id="not_an_object"),
        ],
    )
    def test_unusable_sidecar_falls_back_to_the_bundle(self, tmp_path: Path, content: str) -> None:
        """An unusable sidecar is ignored in favour of the bundle, never served as empty metadata."""
        path = tmp_path / "model.joblib"
        self._write_bundle(path, ["x", "y"])
        constructor.model_metadata_path(path).write_text(content)

        assert load_model_metadata(str(path))[0] == ["x", "y"]

    def test_returned_variables_are_a_fresh_list(self, tmp_path: Path) -> None:
        """Callers own their list — xopt_backend filters its copy in place."""
        path = tmp_path / "model.joblib"
        bundle = self._write_bundle(path, ["x", "y"])
        constructor.write_model_metadata(path, bundle)

        first, _ = load_model_metadata(str(path))
        first.append("poison")

        assert load_model_metadata(str(path))[0] == ["x", "y"]

    def test_metadata_read_does_not_evict_the_estimator_cache(self, tmp_path: Path) -> None:
        """Scanning bundles for metadata must leave the running optimizer's model cached."""
        in_use = tmp_path / "in_use.joblib"
        self._write_bundle(in_use, ["x"])
        constructor_ = SklearnModelConstructor(model_file=str(in_use))
        constructor_._load_pretrained(str(in_use))
        hits_before = constructor._load_model_bundle.cache_info().hits

        # More bundles than the estimator cache holds (maxsize=8).
        for i in range(12):
            other = tmp_path / f"other{i}.joblib"
            self._write_bundle(other, ["x"])
            load_model_metadata(str(other))

        constructor_._load_pretrained(str(in_use))
        assert constructor._load_model_bundle.cache_info().hits == hits_before + 1


class TestPosteriorTransform:
    def test_posterior_applies_scalarized_transform(self) -> None:
        """A weight of -1 must flip the posterior mean (MINIMIZE convention).

        Silently ignoring the transform would make LogEI anti-optimize MINIMIZE
        objectives when a sklearn surrogate is attached.
        """
        rng = np.random.default_rng(0)
        X_train = rng.uniform(0, 1, (20, 2))
        y_train = X_train[:, 0] + 0.5 * X_train[:, 1]
        rf = RandomForestRegressor(n_estimators=10, random_state=0)
        rf.fit(X_train, y_train)
        model = SklearnBoTorchModel(rf)

        X = torch.rand(4, 1, 2, dtype=torch.double)
        base_mean = model.posterior(X).mean
        transform = ScalarizedPosteriorTransform(torch.tensor([-1.0], dtype=torch.double))
        flipped_mean = model.posterior(X, posterior_transform=transform).mean
        assert torch.allclose(flipped_mean, -base_mean)
