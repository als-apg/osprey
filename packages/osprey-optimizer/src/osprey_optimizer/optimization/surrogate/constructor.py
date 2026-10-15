"""ModelConstructor that builds SklearnBoTorchModel from sklearn estimators.

Integrates with xopt's ``BayesianGenerator.gp_constructor`` field so that
any sklearn model with native variance can replace the default GP surrogate.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from botorch.models.model import Model, ModelList
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xopt.generators.bayesian.base_model import ModelConstructor

from osprey_optimizer.models.enums import SurrogateModelType

from .wrapper import SklearnBoTorchModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _load_model_bundle(path: str, mtime_ns: int) -> tuple[Any, list[str]]:
    """Load and unpack a model bundle, cached per (path, mtime).

    xopt calls ``build_model`` on every ``generate()`` iteration; without the
    cache the full estimator is deserialized from disk each time. The mtime key
    keeps a file replaced on disk from being served stale.
    """
    del mtime_ns  # cache key only
    logger.info("Loading pretrained surrogate model from %s", path)
    obj = joblib.load(path)
    model, variables = unpack_model_bundle(obj)
    logger.info("Model was trained on variables: %s", variables)
    return model, variables


def pack_model_bundle(model: Any, variables: list[str]) -> dict[str, Any]:
    """Create a saveable bundle containing the sklearn model and its variable names."""
    return {"model": model, "variables": variables}


def unpack_model_bundle(obj: Any) -> tuple[Any, list[str]]:
    """Extract the sklearn model and variable names from a loaded object.

    Raises:
        ValueError: If *obj* is not a ``{"model": ..., "variables": [...]}``
            bundle as written by :func:`pack_model_bundle`.
    """
    if not (isinstance(obj, dict) and "model" in obj and isinstance(obj.get("variables"), list)):
        raise ValueError(
            "Not a model bundle: expected a dict with 'model' and 'variables' keys as written by pack_model_bundle."
        )
    return obj["model"], obj["variables"]


def model_metadata_path(model_file: str | Path) -> Path:
    """Sidecar path for a bundle: ``model.joblib`` -> ``model.meta.json``."""
    return Path(model_file).with_suffix(".meta.json")


def write_model_metadata(model_file: str | Path, bundle: dict[str, Any]) -> None:
    """Record a bundle's variables/objective in a sidecar beside it.

    ``GET /models`` reads metadata for every model on disk and is polled
    alongside ``GET /results``; unpickling a full sklearn estimator for two
    fields is what the sidecar avoids (#120). Written at save time, so the
    listing never has to derive what was already known when the file was made.
    """
    payload = {"variables": bundle["variables"], "objective": bundle.get("objective")}
    model_metadata_path(model_file).write_text(json.dumps(payload), encoding="utf-8")


def _read_model_metadata(model_file: Path) -> tuple[list[str], str | None] | None:
    """Return the sidecar's ``(variables, objective)``, or ``None`` to read the bundle instead.

    ``None`` means absent, older than the bundle (a model file swapped on disk
    must not be described by the previous one's sidecar), or malformed — every
    unusable case is logged rather than passed off as empty metadata.
    """
    sidecar = model_metadata_path(model_file)
    try:
        if sidecar.stat().st_mtime_ns < model_file.stat().st_mtime_ns:
            return None
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable model metadata %s: %s", sidecar, e)
        return None

    variables = payload.get("variables") if isinstance(payload, dict) else None
    if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
        logger.warning(
            "Ignoring malformed model metadata %s: 'variables' is not a list of names", sidecar
        )
        return None
    objective = payload.get("objective")
    return variables, objective if isinstance(objective, str) else None


def load_model_metadata(model_file: str) -> tuple[list[str], str | None]:
    """Load variable names and objective from a model bundle.

    Served from the ``.meta.json`` sidecar when one is usable. Bundles written
    before sidecars existed are unpickled once and then backfilled, so a legacy
    results directory heals itself instead of paying the unpickle on every poll.

    Returns ``(variables, objective)``; ``objective`` is ``None`` when the
    bundle has no ``objective`` key.

    Raises:
        ValueError: If the file is not a model bundle.
    """
    path = Path(model_file)
    sidecar = _read_model_metadata(path)
    if sidecar is not None:
        return sidecar

    obj = joblib.load(path)
    _, variables = unpack_model_bundle(obj)
    try:
        write_model_metadata(path, obj)
    except OSError as e:
        # Best-effort: a read-only results directory must not break the listing.
        logger.warning("Could not backfill model metadata for %s: %s", path, e)
    return variables, obj.get("objective")


def _build_gaussian_process() -> Pipeline:
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("gp", gp)])


_MODEL_FACTORIES: dict[SurrogateModelType, Any] = {
    SurrogateModelType.RANDOM_FOREST: partial(
        RandomForestRegressor, n_estimators=100, random_state=42, n_jobs=-1
    ),
    SurrogateModelType.EXTRA_TREES: partial(
        ExtraTreesRegressor, n_estimators=100, random_state=42, n_jobs=-1
    ),
    SurrogateModelType.GAUSSIAN_PROCESS: _build_gaussian_process,
}


class SklearnModelConstructor(ModelConstructor):  # type: ignore[misc]
    """Builds a SklearnBoTorchModel from sklearn estimators.

    When ``model_file`` is set, loads a pre-trained model via joblib.
    Otherwise, trains a fresh model from the provided data.
    """

    name: str = "sklearn"
    surrogate_model_type: SurrogateModelType = SurrogateModelType.RANDOM_FOREST
    model_file: str | None = None

    def build_model(
        self,
        input_names: list[str],
        outcome_names: list[str],
        data: pd.DataFrame,
        input_bounds: dict[str, list[float]] | None = None,
        dtype: torch.dtype = torch.double,
        device: torch.device | str = "cpu",
    ) -> ModelList:
        """Build a ModelList containing one SklearnBoTorchModel per outcome.

        Xopt passes bounds, dtype and device to every model constructor; an
        sklearn surrogate fits on the raw data and holds no tensors, so it
        needs none of them.
        """
        del input_bounds, dtype, device
        models: list[Model] = []
        for outcome in outcome_names:
            sklearn_model = (
                self._load_pretrained(self.model_file)
                if self.model_file
                else self._train_fresh(input_names, outcome, data)
            )
            models.append(SklearnBoTorchModel(sklearn_model))
        return ModelList(*models)

    def _load_pretrained(self, model_file: str) -> Any:
        """Load a pretrained model from disk (cached across BO iterations)."""
        return _load_model_bundle(model_file, Path(model_file).stat().st_mtime_ns)[0]

    def _train_fresh(
        self,
        input_names: list[str],
        outcome_name: str,
        data: pd.DataFrame,
    ) -> Any:
        """Train a fresh sklearn model on the provided data."""
        factory = _MODEL_FACTORIES.get(self.surrogate_model_type)
        if factory is None:
            raise ValueError(f"No factory for surrogate model type: {self.surrogate_model_type}")

        model = factory()
        X = data[input_names].to_numpy()
        y = data[outcome_name].to_numpy()

        logger.info(
            "Training %s surrogate on %d samples",
            self.surrogate_model_type.value,
            len(X),
        )
        model.fit(X, y)
        return model
