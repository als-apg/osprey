"""Results persistence for optimization runs.

This module handles saving optimization results to disk in parquet format
with timestamped directory structure.
"""

import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from osprey_optimizer.models.api import HistoricalRunDetails, ModelFileInfo, ResultInfo
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.types import BestPoint, ConfigDict

logger = logging.getLogger(__name__)

#: Where runs are written when the caller names no directory. The deployment
#: sets it to the mounted results volume; unset, runs land in ``./results``.
RESULTS_DIR_ENV = "OPTIMIZER_RESULTS_DIR"


def results_root() -> Path:
    """The base directory every run's results live under."""
    return Path(os.environ.get(RESULTS_DIR_ENV) or "results")


def model_file_under_results_root(model_file: str) -> str:
    """The resolved path of *model_file*, which must lie under :func:`results_root`.

    A model file is unpickled (``joblib.load``), so a run loads one only from
    where the service saves them: a run's ``model.joblib`` or an upload, both
    under the results root. A relative name is read under the root; a link
    that leads out of it is refused like any other path outside.

    Raises:
        ValueError: The path resolves outside the results root.
    """
    root = results_root().resolve()
    resolved = (root / model_file).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"model_file {model_file!r} is not under the results directory; "
            "pick a model from the models list or upload one."
        )
    return str(resolved)


def run_timestamp() -> str:
    """Filesystem-safe local-time timestamp for naming run directories."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented UTF-8 JSON to ``path``."""
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_json_if_absent(path: Path, obj: Any, label: str) -> None:
    """Write ``obj`` to ``path`` unless the file already exists, logging the write."""
    if not path.exists():
        _write_json(path, obj)
        logger.info("  - %s: %s", label, path)


def _read_json(path: Path) -> Any:
    """Read+parse JSON from ``path``; read/parse failures propagate."""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read+parse JSON from ``path``, requiring a JSON object at the top level.

    Raises:
        TypeError: If the file parses to anything other than an object.
    """
    parsed = _read_json(path)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object in {path.name}, got {type(parsed).__name__}")
    return parsed


def _read_json_or_warn(path: Path, label: str) -> dict[str, Any] | None:
    """Read+parse a JSON object from ``path``; log a warning and return ``None`` on failure.

    Failure covers read/parse errors and a top level that is not an object.
    """
    try:
        return _read_json_dict(path)
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to read %s %s: %s", label, path, e)
        return None


def initialize_results_dir(
    config: OptimizationConfig,
    job_id: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Initialize the results directory and save configuration/metadata immediately.

    Args:
        config: Optimization configuration
        job_id: Optional job identifier
        output_dir: Base output directory. Unset, the run goes under
            :func:`results_root`, in a subdirectory named for its environment.

    Returns:
        Path to the initialized results directory
    """
    if output_dir is None:
        output_dir = results_root()
        if config.environment_name:
            output_dir = output_dir / config.environment_name

    timestamp = run_timestamp()
    run_dir = output_dir / timestamp
    # The run-dir name is the run's public identity; the second-resolution
    # timestamp can collide (two starts in the same env within one second),
    # so claim the directory atomically and suffix on collision instead of
    # silently sharing an existing run's directory.
    suffix = 0
    while True:
        try:
            run_dir.mkdir(parents=True)
            break
        except FileExistsError:
            suffix += 1
            run_dir = output_dir / f"{timestamp}-{suffix}"

    logger.info("Initialized run directory: %s", run_dir)

    config_path = run_dir / "config.json"
    _write_json(config_path, config.model_dump())
    logger.info("  - Configuration: %s", config_path)

    if job_id:
        metadata_path = run_dir / "metadata.json"
        _write_json(metadata_path, {"job_id": job_id})
        logger.info("  - Metadata: %s", metadata_path)

    return run_dir


def save_early_results(
    run_dir: Path,
    initial_point: dict[str, float],
    env_metadata: dict[str, Any] | None = None,
    config: OptimizationConfig | None = None,
) -> None:
    """Save early-available optimization data to an existing run directory.

    When ``config`` is given, ``config.json`` is rewritten — the API start
    endpoint persists it before the worker resolves the objective direction,
    so this refresh records the direction the run actually used.
    """
    if config is not None:
        config_path = run_dir / "config.json"
        _write_json(config_path, config.model_dump())
        logger.info("  - Refreshed configuration: %s", config_path)

    if initial_point:
        path = run_dir / "initial_point.json"
        _write_json(path, initial_point)
        logger.info("  - Initial point: %s", path)

    if env_metadata:
        metadata_path = run_dir / "metadata.json"
        existing: dict[str, Any] = {}
        if metadata_path.exists():
            existing = _read_json(metadata_path)
        existing.update(env_metadata)
        _write_json(metadata_path, existing)
        logger.info("  - Updated metadata: %s", metadata_path)


def save_optimization_results(
    data: pl.DataFrame,
    best_point: BestPoint,
    config: OptimizationConfig,
    output_dir: Path | None = None,
    *,
    initial_point: dict[str, float] | None = None,
    job_id: str | None = None,
    model: Any | None = None,
    model_variables: list[str] | None = None,
) -> Path:
    """Save optimization results to disk.

    Args:
        data: Optimization results DataFrame
        best_point: Best point dictionary
        config: Optimization configuration used
        output_dir: Base output directory (default: :func:`results_root`). If this is a specific run dir, it's used directly.
        initial_point: The setpoints the run started from, if known.
        job_id: The job that produced the results, if any.
        model: A trained surrogate to save next to the results.
        model_variables: The model's input variables; required with ``model``.

    Returns:
        Path to the created results directory
    """
    if output_dir is not None and (
        (output_dir / "metadata.json").exists() or (output_dir / "config.json").exists()
    ):
        run_dir = output_dir
        logger.info("\nSaving results to existing directory: %s", run_dir)
    else:
        run_dir = initialize_results_dir(config, job_id, output_dir)
        logger.info("\nSaving results to: %s", run_dir)

    results_path = run_dir / "results.parquet"
    data.write_parquet(results_path)
    logger.info("  - Results: %s", results_path)

    best_point_path = run_dir / "best_point.json"
    _write_json(best_point_path, best_point)
    logger.info("  - Best point: %s", best_point_path)

    if initial_point:
        _write_json_if_absent(run_dir / "initial_point.json", initial_point, "Initial point")

    _write_json_if_absent(run_dir / "config.json", config.model_dump(), "Configuration")

    if job_id:
        _write_json_if_absent(run_dir / "metadata.json", {"job_id": job_id}, "Metadata")

    if model is not None:
        if model_variables is None:
            raise ValueError("model_variables is required to save a trained model bundle")
        try:
            import joblib

            from .optimization.surrogate.constructor import pack_model_bundle, write_model_metadata

            model_path = run_dir / "model.joblib"
            bundle = pack_model_bundle(model, model_variables)
            joblib.dump(bundle, model_path)
            # Sidecar first-class, not an optimization afterthought: the models
            # listing reads it instead of unpickling this estimator (#120).
            write_model_metadata(model_path, bundle)
            logger.info("  - Model: %s", model_path)
        except Exception:
            logger.warning("Failed to save trained model", exc_info=True)

    return run_dir


def load_results(
    identifier: str, output_dir: Path | None = None
) -> tuple[pl.DataFrame, BestPoint, ConfigDict]:
    """Load previously saved optimization results.

    Supports both new format (results.parquet) and legacy format
    (lhs_results.parquet + bo_results.parquet).

    Args:
        identifier: Run identifier: an absolute run directory, or a path
            relative to *output_dir* (``<env>/<timestamp>``)
        output_dir: Base directory for results

    Returns:
        Tuple of (data, best_point, config)
    """
    output_dir = results_root() if output_dir is None else output_dir
    # A relative identifier names a run under the results root, never one
    # relative to the working directory: the API's traversal guard checks it
    # against the root, so it must be read from there.
    path = Path(identifier)
    run_dir = path if path.is_absolute() else output_dir / identifier
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    results_path = run_dir / "results.parquet"
    lhs_path = run_dir / "lhs_results.parquet"
    bo_path = run_dir / "bo_results.parquet"
    best_point_path = run_dir / "best_point.json"
    config_path = run_dir / "config.json"

    if results_path.exists():
        data = pl.read_parquet(results_path)
    else:
        # Legacy fallback: concatenate lhs + bo
        frames = []
        if lhs_path.exists():
            frames.append(pl.read_parquet(lhs_path))
        if bo_path.exists():
            frames.append(pl.read_parquet(bo_path))
        data = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    best_point: BestPoint = {}
    if best_point_path.exists():
        best_point = _read_json_dict(best_point_path)

    config_dict: ConfigDict = {}
    if config_path.exists():
        config_dict = _read_json_dict(config_path)

    initial_point_path = run_dir / "initial_point.json"
    if initial_point_path.exists():
        initial_point = _read_json_or_warn(initial_point_path, "initial point")
        if initial_point is not None:
            if "snapshots" not in config_dict:
                config_dict["snapshots"] = {}
            config_dict["snapshots"]["initial"] = initial_point

    return data, best_point, config_dict


def build_run_details(identifier: str, output_dir: Path | None = None) -> HistoricalRunDetails:
    """Load a run and assemble its full details payload.

    Single source of truth for the shape served by the API and printed by the
    CLI's local mode.
    """
    data, best_point, config = load_results(identifier, output_dir)
    return HistoricalRunDetails(
        timestamp=identifier,
        config=config,
        best_point=best_point,
        initial_point=config.get("snapshots", {}).get("initial"),
        data_records=data.to_dicts(),
    )


def scan_results_directory(results_dir: Path | None = None) -> list[str]:
    """Scan the results directory for available optimization runs.

    This function scans environment subdirectories: results/env_name/timestamp.
    Legacy runs sitting directly under ``results_dir`` are not matched.

    Args:
        results_dir: Base directory containing results subdirectories.

    Returns:
        List of run identifier strings (format: "env_name/timestamp").
    """
    results_dir = results_root() if results_dir is None else results_dir
    runs = [f"{p.parent.parent.name}/{p.parent.name}" for p in results_dir.glob("*/*/config.json")]
    runs.sort(key=lambda x: Path(x).name, reverse=True)
    return runs


_StatToken = tuple[int, int] | None
"""``(mtime_ns, size)`` identity of a file, or ``None`` when it is not there."""

_RunMeta = tuple[str | None, str | None]
"""``(objective_name, job_id)`` — the parsed half of a run's ResultInfo."""

_RUN_META_CACHE: dict[Path, tuple[tuple[_StatToken, _StatToken], _RunMeta]] = {}
"""Read-through cache for the two JSON files :func:`scan_result_infos` parses per run.

``GET /results`` is polled once a second for as long as a job is running, and a
completed run's ``config.json``/``metadata.json`` never change again — without
this, every poll re-parses every run ever recorded (#120). Entries are keyed by
run directory and validated against each file's ``(mtime_ns, size)``, so a run
still being written is picked up as soon as its files change.

Concurrent ``list_results`` calls share it across the FastAPI threadpool; the
worst race is two threads parsing the same run and storing the same value.
"""


def _stat_token(path: Path) -> _StatToken:
    """Cheap change-detection identity for *path*: ``(mtime_ns, size)``, ``None`` if unreadable."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _run_metadata(run_dir: Path) -> _RunMeta:
    """Return ``(objective_name, job_id)`` for a run, re-parsing only on change."""
    config_path = run_dir / "config.json"
    metadata_path = run_dir / "metadata.json"
    token = (_stat_token(config_path), _stat_token(metadata_path))

    cached = _RUN_META_CACHE.get(run_dir)
    if cached is not None and cached[0] == token:
        return cached[1]

    meta = (_read_config_metadata(config_path), _read_run_job_id(metadata_path))
    _RUN_META_CACHE[run_dir] = (token, meta)
    return meta


def scan_result_infos(results_dir: Path | None = None) -> list[ResultInfo]:
    """Scan the results directory and return enriched ResultInfo for each run.

    Builds on scan_results_directory (list[str]) and adds per-run metadata.
    """
    results_dir = results_root() if results_dir is None else results_dir
    infos: list[ResultInfo] = []
    scanned: set[Path] = set()
    for run_id in scan_results_directory(results_dir):
        # scan_results_directory always yields exactly "<env>/<timestamp>"; let a
        # malformed id raise rather than silently skipping (no silent fallbacks).
        env, ts = run_id.rsplit("/", 1)
        run_dir = results_dir / run_id
        scanned.add(run_dir)
        objective_name, job_id = _run_metadata(run_dir)
        infos.append(
            ResultInfo(
                run_id=run_id,
                environment=env,
                timestamp=ts,
                objective_name=objective_name,
                # Deliberately uncached: model.joblib appears at the end of a run,
                # long after the JSON files settle. One stat, always live.
                has_model=(run_dir / "model.joblib").exists(),
                job_id=job_id,
            )
        )

    # Drop runs that vanished from disk. Run dirs are always "<results_dir>/<env>/<ts>",
    # so this only touches entries this scan could have produced — a scan of some
    # other results_dir (tests) must not evict them.
    for stale in [
        d for d in _RUN_META_CACHE if d.parent.parent == results_dir and d not in scanned
    ]:
        del _RUN_META_CACHE[stale]

    return infos


def _read_config_metadata(config_path: Path) -> str | None:
    """Read objective_name from a config.json.

    Returns None if the key is absent. Logs a warning and returns None if the
    file cannot be read or parsed.
    """
    cfg = _read_json_or_warn(config_path, "config metadata from")
    if cfg is None:
        return None
    objective_name = cfg.get("objective_name")
    if objective_name is not None and not isinstance(objective_name, str):
        raise TypeError(
            f"Expected str objective_name in {config_path}, got {type(objective_name).__name__}"
        )
    return objective_name


def _read_run_job_id(metadata_path: Path) -> str | None:
    """Read the job_id recorded in a run's metadata.json.

    Returns None for legacy runs without metadata.json. Logs a warning and
    returns None if the file exists but cannot be read/parsed.
    """
    if not metadata_path.exists():
        return None
    metadata = _read_json_or_warn(metadata_path, "run metadata")
    if metadata is None:
        return None
    job_id = metadata.get("job_id")
    return str(job_id) if job_id is not None else None


#: Bundles the model listing has already reported as unlisted (no sidecar).
_UNLISTED_MODELS: set[Path] = set()


def _load_model_metadata(model_path: Path) -> tuple[list[str] | None, str | None]:
    """Load variables and objective from a model bundle.

    Returns (None, None) and logs a warning if the file cannot be loaded or is
    not a valid model bundle; scan callers skip such files.
    """
    from .optimization.surrogate.constructor import (
        MissingModelMetadataError,
        load_model_metadata,
    )

    try:
        # Listing is a read anyone may call, so it never unpickles: a bundle
        # without its sidecar is skipped, not loaded.
        return load_model_metadata(str(model_path), unpickle=False)
    except MissingModelMetadataError:
        # Expected for a bundle written before sidecars existed, and the
        # listing is polled: said once per file, without a traceback.
        if model_path not in _UNLISTED_MODELS:
            _UNLISTED_MODELS.add(model_path)
            logger.warning("Not listing %s: it has no metadata sidecar", model_path)
        return None, None
    except Exception:
        logger.warning("Failed to load model metadata from %s", model_path, exc_info=True)
        return None, None


def scan_model_files(results_dir: Path | None = None) -> list[ModelFileInfo]:
    """Scan the results directory for saved surrogate model files.

    Includes models from both optimization runs and user uploads.

    Args:
        results_dir: Base directory containing results subdirectories.

    Returns:
        List of ModelFileInfo with label and absolute path for each model.
    """
    results_dir = results_root() if results_dir is None else results_dir
    models: list[ModelFileInfo] = []
    for run_id in scan_results_directory(results_dir):
        model_path = results_dir / run_id / "model.joblib"
        if model_path.exists():
            # Bundle variables are authoritative (they include fixed features);
            # a file that is not a valid bundle is skipped.
            bundle_vars, _ = _load_model_metadata(model_path)
            if bundle_vars is None:
                continue

            # Same cache the results listing fills: /models and /results are polled
            # together and read the same run directories.
            obj_name, _ = _run_metadata(model_path.parent)

            models.append(
                ModelFileInfo(
                    label=run_id,
                    path=str(model_path.resolve()),
                    variables=bundle_vars,
                    objective_name=obj_name,
                )
            )

    uploaded_dir = results_dir / "uploaded_models"
    if uploaded_dir.exists():
        for up_path in sorted(uploaded_dir.glob("*.joblib"), reverse=True):
            variables, objective = _load_model_metadata(up_path)
            if variables is None:
                continue
            models.append(
                ModelFileInfo(
                    label=f"upload: {up_path.name}",
                    path=str(up_path.resolve()),
                    variables=variables,
                    objective_name=objective,
                )
            )

    return models


def save_uploaded_model(
    filename: str, content: bytes, results_dir: Path | None = None
) -> ModelFileInfo:
    """Save an uploaded model file to the uploaded_models directory.

    Args:
        filename: Original filename (must end with .joblib).
        content: Raw file bytes.
        results_dir: Base results directory.

    Returns:
        ModelFileInfo with label and absolute path for the saved model.

    Raises:
        ValueError: If the filename does not end with .joblib, or the content
            is not a joblib model bundle with 'model' and 'variables' keys.
    """
    results_dir = results_root() if results_dir is None else results_dir
    if not filename.endswith(".joblib"):
        raise ValueError(f"Invalid file extension: expected .joblib, got '{filename}'")

    # Collapse to a bare filename so a crafted name (e.g. "../../evil.joblib")
    # can't escape the uploaded_models directory.
    safe_name = Path(filename).name
    if safe_name in ("", ".joblib"):
        raise ValueError(f"Invalid model filename: '{filename}'")

    import joblib

    from .optimization.surrogate.constructor import unpack_model_bundle, write_model_metadata

    try:
        obj = joblib.load(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Invalid model file '{filename}': not a readable joblib file") from exc
    unpack_model_bundle(obj)

    uploaded_dir = results_dir / "uploaded_models"
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    dest = uploaded_dir / safe_name
    if dest.exists():
        stem = Path(safe_name).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = uploaded_dir / f"{timestamp}_{stem}.joblib"

    dest.write_bytes(content)
    write_model_metadata(dest, obj)
    logger.info("Saved uploaded model: %s", dest)

    return ModelFileInfo(
        label=f"upload: {dest.name}",
        path=str(dest.resolve()),
    )
