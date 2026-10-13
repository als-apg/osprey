"""Tests for results.py — filesystem-based persistence of optimization data."""

import io
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import polars as pl
import pytest

from osprey.services.optimizer import results
from osprey.services.optimizer.models.config import OptimizationConfig
from osprey.services.optimizer.models.types import Direction
from osprey.services.optimizer.optimization.surrogate.constructor import unpack_model_bundle
from osprey.services.optimizer.results import (
    ResultMetadata,
    _load_model_metadata,
    _read_config_metadata,
    initialize_results_dir,
    load_results,
    save_early_results,
    save_optimization_results,
    save_uploaded_model,
    scan_model_files,
    scan_result_infos,
    scan_results_directory,
)

FIXED_TIMESTAMP = "2026-01-15_12-00-00"


def _joblib_bytes(obj: object) -> bytes:
    """Serialize an arbitrary object the way joblib writes it to disk."""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def _bundle_bytes(variables: list[str] | None = None) -> bytes:
    """Serialized {"model", "variables"} bundle, the only accepted model format."""
    return _joblib_bytes({"model": "fake", "variables": variables or ["x"]})


def _save(output_dir: Path, **overrides: Any) -> Path:
    """save_optimization_results over a standard one-row run; override only what
    the test varies (data, best_point, config, metadata)."""
    defaults: dict[str, Any] = {
        "data": pl.DataFrame({"x": [1.0]}),
        "best_point": {"x": 1.0},
        "config": OptimizationConfig(environment_name="env", control_target="va", n_iterations=1),
    }
    return save_optimization_results(output_dir=output_dir, **{**defaults, **overrides})


@pytest.fixture
def config() -> OptimizationConfig:
    return OptimizationConfig(environment_name="test_env", control_target="va", n_iterations=5)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "obj": [0.1, 0.2, 0.3]})


@pytest.fixture
def mock_now():
    """Patch datetime.now to return a fixed timestamp."""
    fixed = datetime(2026, 1, 15, 12, 0, 0)
    with patch("osprey.services.optimizer.results.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        yield mock_dt


class TestInitializeResultsDir:
    def test_creates_timestamped_dir_with_config_and_no_metadata(
        self, tmp_path: Path, config: OptimizationConfig, mock_now
    ):
        """A non-default output_dir gets a bare timestamp dir (environment_name is
        NOT appended) holding the serialized config; metadata.json is written only
        when a job_id is supplied."""
        run_dir = initialize_results_dir(config, output_dir=tmp_path / "results")
        assert run_dir.exists()
        assert run_dir == tmp_path / "results" / FIXED_TIMESTAMP
        config_path = run_dir / "config.json"
        assert config_path.exists()
        with config_path.open() as f:
            assert json.load(f) == config.model_dump()
        assert not (run_dir / "metadata.json").exists()

    def test_default_dir_appends_env_name(
        self, tmp_path: Path, monkeypatch, config: OptimizationConfig, mock_now
    ):
        """When output_dir is the default Path('results'), environment_name is appended."""
        monkeypatch.chdir(tmp_path)
        run_dir = initialize_results_dir(config)
        assert run_dir == Path("results") / "test_env" / FIXED_TIMESTAMP

    def test_same_second_starts_get_distinct_dirs(
        self, tmp_path: Path, config: OptimizationConfig, mock_now
    ):
        first = initialize_results_dir(config, "job-a", output_dir=tmp_path)
        second = initialize_results_dir(config, "job-b", output_dir=tmp_path)
        assert first != second
        assert first.name == FIXED_TIMESTAMP
        assert second.name == f"{FIXED_TIMESTAMP}-1"
        # Each dir keeps its own job link — no silent overwrite.
        assert json.loads((first / "metadata.json").read_text())["job_id"] == "job-a"
        assert json.loads((second / "metadata.json").read_text())["job_id"] == "job-b"


class TestSaveEarlyResults:
    def test_refreshes_config_json_with_resolved_direction(self, tmp_path: Path, mock_now):
        """The API path writes config.json before the worker resolves the
        objective direction; save_early_results must refresh it so the
        persisted config records the direction the run actually used."""
        config = OptimizationConfig(
            environment_name="test_env", objective_name="obj", n_iterations=5
        )
        run_dir = initialize_results_dir(config, "job-1", tmp_path)
        with (run_dir / "config.json").open() as f:
            assert json.load(f)["objective_direction"] is None

        config.objective_direction = Direction.MAXIMIZE
        save_early_results(run_dir, {"x": 1.0}, config=config)

        with (run_dir / "config.json").open() as f:
            assert json.load(f)["objective_direction"] == "MAXIMIZE"

    def test_config_json_untouched_without_config(self, tmp_path: Path, mock_now):
        """Without a config argument the existing config.json is left alone."""
        config = OptimizationConfig(
            environment_name="test_env", objective_name="obj", n_iterations=5
        )
        run_dir = initialize_results_dir(config, "job-1", tmp_path)
        before = (run_dir / "config.json").read_text()

        save_early_results(run_dir, {"x": 1.0})

        assert (run_dir / "config.json").read_text() == before

    def test_empty_initial_point_skips_file_and_writes_env_metadata(self, tmp_path: Path) -> None:
        """initial_point falsy + env_metadata branch with no prior metadata.json."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_early_results(run_dir, {}, env_metadata={"hardware": "sim"})

        assert not (run_dir / "initial_point.json").exists()
        with (run_dir / "metadata.json").open() as f:
            assert json.load(f) == {"hardware": "sim"}

    def test_env_metadata_merges_into_existing(self, tmp_path: Path) -> None:
        """env_metadata branch where metadata.json already exists (111-113 read+update)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "metadata.json").write_text(json.dumps({"job_id": "j-1"}))

        save_early_results(run_dir, {"x": 1.0}, env_metadata={"hardware": "sim"})

        with (run_dir / "metadata.json").open() as f:
            merged = json.load(f)
        assert merged == {"job_id": "j-1", "hardware": "sim"}
        assert (run_dir / "initial_point.json").exists()


class TestSaveOptimizationResults:
    def test_saves_all_required_files(
        self, tmp_path: Path, config: OptimizationConfig, sample_df: pl.DataFrame, mock_now
    ):
        run_dir = _save(
            tmp_path / "results", data=sample_df, best_point={"x": 2.0, "obj": 0.2}, config=config
        )
        assert (run_dir / "results.parquet").exists()
        assert (run_dir / "best_point.json").exists()
        assert (run_dir / "config.json").exists()
        # Without metadata there is no initial point to write.
        assert not (run_dir / "initial_point.json").exists()

    def test_saves_initial_point_when_provided(
        self, tmp_path: Path, config: OptimizationConfig, sample_df: pl.DataFrame, mock_now
    ):
        run_dir = _save(
            tmp_path / "results",
            data=sample_df,
            config=config,
            metadata=ResultMetadata(initial_point={"var1": 0.5}),
        )
        ip_path = run_dir / "initial_point.json"
        assert ip_path.exists()
        with ip_path.open() as f:
            assert json.load(f) == {"var1": 0.5}

    def test_pre_initialized_dir_reuses_existing(
        self, tmp_path: Path, config: OptimizationConfig, sample_df: pl.DataFrame, mock_now
    ):
        pre_dir = initialize_results_dir(config, output_dir=tmp_path / "results")
        run_dir = _save(pre_dir, data=sample_df, config=config)
        assert run_dir == pre_dir
        assert (run_dir / "results.parquet").exists()

    def test_writes_metadata_when_job_id_provided(
        self, tmp_path: Path, config: OptimizationConfig, sample_df: pl.DataFrame, mock_now
    ):
        run_dir = _save(
            tmp_path / "results",
            data=sample_df,
            config=config,
            metadata=ResultMetadata(job_id="job-456"),
        )
        with (run_dir / "metadata.json").open() as f:
            assert json.load(f)["job_id"] == "job-456"

    def test_preexisting_initial_point_not_overwritten(self, tmp_path: Path) -> None:
        """initial_point.json already present -> 158->162 (skip rewrite)."""
        run_dir = tmp_path / "results" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text("{}")  # marks pre-initialized
        (run_dir / "initial_point.json").write_text(json.dumps({"existing": 9.0}))

        _save(run_dir, metadata=ResultMetadata(initial_point={"x": 0.5}))
        with (run_dir / "initial_point.json").open() as f:
            assert json.load(f) == {"existing": 9.0}

    def test_writes_config_and_metadata_when_absent(self, tmp_path: Path) -> None:
        """Pre-initialized dir with only metadata.json -> config write and metadata
        write skipped; use a dir with only config.json so the metadata path runs."""
        run_dir = tmp_path / "results" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}")

        _save(run_dir, metadata=ResultMetadata(job_id="job-xyz"))
        # config.json left as-is; metadata written.
        with (run_dir / "metadata.json").open() as f:
            assert json.load(f) == {"job_id": "job-xyz"}

    def test_config_written_when_missing_in_preinitialized(self, tmp_path: Path) -> None:
        """Pre-initialized via metadata.json only -> config.json absent -> it gets written."""
        run_dir = tmp_path / "results" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(json.dumps({"job_id": "j"}))

        _save(run_dir)
        assert (run_dir / "config.json").exists()
        with (run_dir / "config.json").open() as f:
            assert json.load(f)["environment_name"] == "env"

    def test_model_saved_with_variables_bundle(self, tmp_path: Path) -> None:
        """meta.model + model_variables -> pack_model_bundle + dump (181-182, 185).

        Real round-trip: let pack_model_bundle/joblib.dump run so the on-disk
        model.joblib is the actual packed bundle, not a mock stand-in.
        """
        run_dir = tmp_path / "results" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}")

        # A distinguishable, picklable "model" so we can prove the RIGHT object landed.
        model = {"weights": [1.0, 2.0, 3.0]}
        run = _save(run_dir, metadata=ResultMetadata(model=model, model_variables=["x", "y"]))

        # The artifact lands at the pinned path and is a real, loadable bundle.
        model_path = run / "model.joblib"
        assert model_path == run_dir / "model.joblib"
        assert model_path.exists()

        loaded = joblib.load(model_path)
        assert loaded == {"model": {"weights": [1.0, 2.0, 3.0]}, "variables": ["x", "y"]}
        unpacked_model, unpacked_vars = unpack_model_bundle(loaded)
        assert unpacked_model == model
        assert unpacked_vars == ["x", "y"]

        # The listing reads this instead of unpickling the estimator (#120).
        assert json.loads((run / "model.meta.json").read_text()) == {
            "variables": ["x", "y"],
            "objective": None,
        }

    def test_model_save_failure_is_swallowed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """meta.model set but joblib.dump raises -> except branch swallows, but the
        failure is logged with the traceback (a silent `pass` must fail here)."""
        run_dir = tmp_path / "results" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}")

        with (
            patch("joblib.dump", side_effect=RuntimeError("disk full")),
            caplog.at_level("WARNING"),
        ):
            run = _save(run_dir, metadata=ResultMetadata(model=object(), model_variables=["x"]))
        assert not (run / "model.joblib").exists()
        [record] = [r for r in caplog.records if r.getMessage() == "Failed to save trained model"]
        assert record.exc_info is not None


class TestLoadResults:
    def test_loads_new_format(
        self, tmp_path: Path, config: OptimizationConfig, sample_df: pl.DataFrame, mock_now
    ):
        run_dir = _save(
            tmp_path / "results", data=sample_df, best_point={"x": 2.0, "obj": 0.2}, config=config
        )
        data, best_point, config_dict = load_results(str(run_dir))
        assert data.shape == sample_df.shape
        assert best_point == {"x": 2.0, "obj": 0.2}
        assert config_dict["environment_name"] == "test_env"

    @pytest.mark.parametrize(
        "legacy_files",
        [
            pytest.param(
                {"lhs_results.parquet": [1.0], "bo_results.parquet": [2.0]}, id="lhs_and_bo"
            ),
            pytest.param({"lhs_results.parquet": [1.0, 2.0]}, id="only_lhs"),
        ],
    )
    def test_legacy_parquet_fallback(self, tmp_path: Path, legacy_files: dict[str, list[float]]):
        """Pre-results.parquet runs are reassembled from the per-phase files."""
        run_dir = tmp_path / "legacy_run"
        run_dir.mkdir()
        for name, xs in legacy_files.items():
            pl.DataFrame({"x": xs}).write_parquet(run_dir / name)
        (run_dir / "config.json").write_text("{}")

        data, _, _ = load_results(str(run_dir))
        assert data.shape[0] == 2

    def test_empty_run_dir_returns_empty_df_and_best_point(self, tmp_path: Path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text("{}")

        data, best_point, _ = load_results(str(run_dir))
        assert data.shape == (0, 0)
        assert best_point == {}

    def test_missing_config_returns_empty_dict(self, tmp_path: Path):
        run_dir = tmp_path / "no_config"
        run_dir.mkdir()

        _, _, config_dict = load_results(str(run_dir))
        assert config_dict == {}

    def test_initial_point_merged_into_snapshots(
        self, tmp_path: Path, config: OptimizationConfig, mock_now
    ):
        run_dir = _save(
            tmp_path / "results",
            config=config,
            metadata=ResultMetadata(initial_point={"var1": 0.5, "var2": 1.0}),
        )
        _, _, config_dict = load_results(str(run_dir))
        assert config_dict["snapshots"]["initial"] == {"var1": 0.5, "var2": 1.0}

    def test_corrupt_initial_point_logged_not_crash(self, tmp_path: Path):
        run_dir = tmp_path / "corrupt_ip"
        run_dir.mkdir()
        (run_dir / "config.json").write_text('{"environment_name": "test"}')
        (run_dir / "initial_point.json").write_text("{invalid json")

        _, _, config_dict = load_results(str(run_dir))
        assert config_dict["environment_name"] == "test"
        assert "snapshots" not in config_dict

    def test_nonexistent_dir_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_results("nonexistent_dir_abc123")

    @pytest.mark.parametrize(
        ("filename", "content"),
        [
            pytest.param("best_point.json", "[1.0, 2.0]", id="best_point_list"),
            pytest.param("config.json", '"just a string"', id="config_str"),
        ],
    )
    def test_non_dict_json_raises_type_error(self, tmp_path: Path, filename: str, content: str):
        """A parseable-but-non-dict best_point/config fails loud at the read
        instead of flowing through typed as a dict."""
        run_dir = tmp_path / "bad_json"
        run_dir.mkdir()
        (run_dir / "config.json").write_text("{}")
        (run_dir / filename).write_text(content)

        with pytest.raises(TypeError, match=filename):
            load_results(str(run_dir))

    def test_initial_point_merged_when_snapshots_present(self, tmp_path: Path) -> None:
        """config already has snapshots -> 242->244 (skip creating snapshots dict)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps({"snapshots": {"other": {"a": 1}}}))
        (run_dir / "initial_point.json").write_text(json.dumps({"var1": 0.5}))

        _, _, config_dict = load_results(str(run_dir))
        assert config_dict["snapshots"]["initial"] == {"var1": 0.5}
        assert config_dict["snapshots"]["other"] == {"a": 1}


def test_read_config_metadata_non_str_objective_name_raises(tmp_path: Path) -> None:
    """A malformed (non-string) objective_name is corrupt data — raise instead of
    letting it flow through typed as str | None."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"objective_name": 123}')

    with pytest.raises(TypeError, match="objective_name"):
        _read_config_metadata(config_path)


def _bump_mtime(path: Path, seconds: int = 1) -> None:
    """Advance a file's mtime so a (mtime, size) cache key is guaranteed to differ."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def _make_run_dirs(base: Path, *rel_paths: str) -> None:
    """Create ``<base>/<rel>/config.json`` for each run path — what the scanners key on."""
    for rel in rel_paths:
        run_dir = base / rel
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}")


class TestScanResultsDirectory:
    def test_nonexistent_dir_returns_empty_list(self, tmp_path: Path):
        result = scan_results_directory(tmp_path / "nonexistent")
        assert result == []

    @pytest.mark.parametrize(
        ("run_dirs", "expected"),
        [
            pytest.param(
                ("env1/2026-01-01", "env1/2026-01-02"),
                ["env1/2026-01-02", "env1/2026-01-01"],
                id="env_timestamp_runs_newest_first",
            ),
            # Legacy format: timestamp dir directly under results with config.json
            pytest.param(("2026-01-01",), [], id="skips_legacy_dirs"),
        ],
    )
    def test_finds_env_timestamp_runs(
        self, tmp_path: Path, run_dirs: tuple[str, ...], expected: list[str]
    ):
        _make_run_dirs(tmp_path, *run_dirs)

        result = scan_results_directory(tmp_path)
        assert result == expected

    def test_multiple_envs(self, tmp_path: Path):
        _make_run_dirs(tmp_path, "env1/ts1", "env2/ts2")

        result = scan_results_directory(tmp_path)
        assert len(result) == 2
        assert "env1/ts1" in result
        assert "env2/ts2" in result

    def test_non_directory_entry_skipped(self, tmp_path: Path) -> None:
        """A plain file directly under results is ignored by the scan."""
        (tmp_path / "stray.txt").write_text("not a dir")
        _make_run_dirs(tmp_path, "env1/ts1")

        result = scan_results_directory(tmp_path)
        assert result == ["env1/ts1"]


class TestScanResultInfos:
    def _make_run(
        self,
        base: Path,
        env: str,
        ts: str,
        config: dict[str, object] | None = None,
        with_model: bool = False,
        job_id: str | None = None,
    ) -> None:
        run_dir = base / env / ts
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(config or {}))
        if with_model:
            (run_dir / "model.joblib").write_bytes(b"fake")
        if job_id is not None:
            (run_dir / "metadata.json").write_text(json.dumps({"job_id": job_id}))

    def test_nonexistent_dir_returns_empty_list(self, tmp_path: Path):
        result = scan_result_infos(tmp_path / "nonexistent")
        assert result == []

    def test_basic_fields_and_newest_first_order(self, tmp_path: Path):
        self._make_run(tmp_path, "env1", "2026-01-01", config={"objective_name": "my_obj"})
        self._make_run(tmp_path, "env1", "2026-01-02")
        result = scan_result_infos(tmp_path)
        assert len(result) == 2
        # scan_results_directory returns newest-first
        assert result[0].timestamp == "2026-01-02"
        info = result[1]
        assert info.run_id == "env1/2026-01-01"
        assert info.environment == "env1"
        assert info.timestamp == "2026-01-01"
        assert info.objective_name == "my_obj"
        assert info.has_model is False

    @pytest.mark.parametrize(
        ("run_kwargs", "attr", "expected"),
        [
            pytest.param({"with_model": True}, "has_model", True, id="has_model_true"),
            pytest.param({"config": {}}, "objective_name", None, id="objective_name_absent"),
            pytest.param({"job_id": "job-abc"}, "job_id", "job-abc", id="job_id_from_metadata"),
            pytest.param({}, "job_id", None, id="job_id_absent_legacy_run"),
        ],
    )
    def test_single_run_field(
        self, tmp_path: Path, run_kwargs: dict[str, Any], attr: str, expected: object
    ):
        self._make_run(tmp_path, "env1", "2026-01-01", **run_kwargs)
        result = scan_result_infos(tmp_path)
        assert len(result) == 1
        assert getattr(result[0], attr) == expected

    def test_job_id_none_for_corrupt_metadata(self, tmp_path: Path):
        self._make_run(tmp_path, "env1", "2026-01-01")
        (tmp_path / "env1" / "2026-01-01" / "metadata.json").write_text("{not json")
        result = scan_result_infos(tmp_path)
        assert result[0].job_id is None

    @pytest.mark.parametrize(
        ("filename", "content"),
        [
            pytest.param("config.json", "null", id="null_config"),
            pytest.param("metadata.json", "[]", id="list_metadata"),
        ],
    )
    def test_non_dict_json_yields_none_fields(
        self, tmp_path: Path, caplog, filename: str, content: str
    ):
        """Valid JSON that is not an object is as unusable as a corrupt file: the
        run still lists, with the field None and a warning logged (#83)."""
        self._make_run(
            tmp_path, "env1", "2026-01-01", config={"objective_name": "obj"}, job_id="job-abc"
        )
        (tmp_path / "env1" / "2026-01-01" / filename).write_text(content)
        result = scan_result_infos(tmp_path)
        assert len(result) == 1
        assert result[0].run_id == "env1/2026-01-01"
        if filename == "config.json":
            assert result[0].objective_name is None
            assert result[0].job_id == "job-abc"
        else:
            assert result[0].objective_name == "obj"
            assert result[0].job_id is None
        assert "Expected a JSON object" in caplog.text


class TestScanResultInfosCaching:
    """The per-run JSON parses are cached and invalidated on (mtime, size) (#120)."""

    def _make_run(self, base: Path, config: dict[str, object] | None = None) -> Path:
        run_dir = base / "env1" / "2026-01-01"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(config or {"objective_name": "obj"}))
        return run_dir

    def _spy_on_json_reads(self, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        """Record every JSON file the scan actually opens and parses."""
        reads: list[Path] = []
        real = results._read_json

        def spy(path: Path) -> Any:
            reads.append(path)
            return real(path)

        monkeypatch.setattr(results, "_read_json", spy)
        return reads

    def test_repeated_scans_reuse_the_parsed_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """GET /results is polled every second while a job runs; unchanged runs must not re-parse."""
        run_dir = self._make_run(tmp_path)
        (run_dir / "metadata.json").write_text(json.dumps({"job_id": "job-1"}))
        reads = self._spy_on_json_reads(monkeypatch)

        first = scan_result_infos(tmp_path)
        assert len(reads) == 2  # config.json + metadata.json, parsed once

        for _ in range(5):
            assert scan_result_infos(tmp_path) == first
        assert len(reads) == 2  # five polls later, still parsed only once

    def test_rewritten_config_is_reparsed(self, tmp_path: Path):
        """save_early_results refreshes config.json mid-run, so a rewrite must not be missed."""
        run_dir = self._make_run(tmp_path, config={"objective_name": "before"})
        assert scan_result_infos(tmp_path)[0].objective_name == "before"

        config_path = run_dir / "config.json"
        # Same byte length as the original: this exercises the mtime half of the key.
        config_path.write_text(json.dumps({"objective_name": "after!"}))
        _bump_mtime(config_path)

        assert scan_result_infos(tmp_path)[0].objective_name == "after!"

    def test_metadata_appearing_later_is_picked_up(self, tmp_path: Path):
        """A run scanned before metadata.json exists must not cache job_id=None forever."""
        run_dir = self._make_run(tmp_path)
        assert scan_result_infos(tmp_path)[0].job_id is None

        (run_dir / "metadata.json").write_text(json.dumps({"job_id": "job-9"}))
        assert scan_result_infos(tmp_path)[0].job_id == "job-9"

    def test_model_written_after_the_first_scan_is_seen(self, tmp_path: Path):
        """has_model flips when a finished run saves its surrogate — it is never cached."""
        run_dir = self._make_run(tmp_path)
        assert scan_result_infos(tmp_path)[0].has_model is False

        (run_dir / "model.joblib").write_bytes(b"fake")
        assert scan_result_infos(tmp_path)[0].has_model is True

    def test_deleted_run_is_dropped_from_the_cache(self, tmp_path: Path):
        """A removed run must not leak an entry in a long-lived server process."""
        run_dir = self._make_run(tmp_path)
        scan_result_infos(tmp_path)
        assert run_dir in results._RUN_META_CACHE

        shutil.rmtree(run_dir)
        assert scan_result_infos(tmp_path) == []
        assert run_dir not in results._RUN_META_CACHE

    def test_model_scan_shares_the_run_metadata_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """/models and /results are polled together over the same run dirs — one parse, not two."""
        run_dir = self._make_run(tmp_path)
        (run_dir / "model.joblib").write_bytes(_bundle_bytes(["x"]))
        reads = self._spy_on_json_reads(monkeypatch)

        scan_result_infos(tmp_path)
        scan_model_files(tmp_path)

        assert reads == [run_dir / "config.json"]

    def test_scanning_another_results_dir_keeps_the_first_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Eviction is scoped to the scanned root; a second root must not blow the first away."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        self._make_run(dir_a)
        self._make_run(dir_b)
        scan_result_infos(dir_a)
        reads = self._spy_on_json_reads(monkeypatch)

        scan_result_infos(dir_b)
        scan_result_infos(dir_a)

        assert reads == [dir_b / "env1" / "2026-01-01" / "config.json"]


class TestScanModelFiles:
    @pytest.mark.parametrize(
        "subdir", ["nonexistent", ""], ids=["nonexistent_dir", "empty_results_dir"]
    )
    def test_returns_empty_list_when_no_models(self, tmp_path: Path, subdir: str):
        result = scan_model_files(tmp_path / subdir)
        assert result == []

    def test_finds_runs_with_model_files(self, tmp_path: Path):
        run_dir = tmp_path / "env1" / "2026-01-01"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}")
        (run_dir / "model.joblib").write_bytes(_bundle_bytes(["x", "y"]))

        result = scan_model_files(tmp_path)
        assert len(result) == 1
        assert result[0].label == "env1/2026-01-01"
        assert result[0].path == str((run_dir / "model.joblib").resolve())
        assert result[0].variables == ["x", "y"]

    @pytest.mark.parametrize(
        ("with_model", "without_model"),
        [
            pytest.param(
                ("env1/2026-01-01",), ("env1/2026-01-02",), id="skips_run_without_model_file"
            ),
            pytest.param(("env1/ts1", "env2/ts2"), (), id="multiple_envs_with_models"),
        ],
    )
    def test_lists_exactly_the_runs_with_a_model(
        self, tmp_path: Path, with_model: tuple[str, ...], without_model: tuple[str, ...]
    ):
        _make_run_dirs(tmp_path, *with_model, *without_model)
        for rel in with_model:
            (tmp_path / rel / "model.joblib").write_bytes(_bundle_bytes())

        result = scan_model_files(tmp_path)
        assert {m.label for m in result} == set(with_model)

    def test_skips_invalid_bundle_with_warning(self, tmp_path: Path, caplog):
        """A model.joblib that is not a valid bundle is excluded from the listing."""
        run_dir = tmp_path / "env1" / "2026-01-01"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text("{}")
        (run_dir / "model.joblib").write_bytes(b"not a bundle")

        result = scan_model_files(tmp_path)
        assert result == []
        assert "Failed to load model metadata" in caplog.text

    def test_uploaded_models_scanned_and_invalid_skipped(self, tmp_path: Path, caplog) -> None:
        """Every valid bundle in uploaded_models/ is listed; a stale non-bundle
        file is excluded from the listing."""
        uploaded = tmp_path / "uploaded_models"
        uploaded.mkdir()
        (uploaded / "m1.joblib").write_bytes(_bundle_bytes(["a"]))
        (uploaded / "m2.joblib").write_bytes(_bundle_bytes(["b"]))
        (uploaded / "bad.joblib").write_bytes(b"raw pickle junk")

        result = scan_model_files(tmp_path)
        assert {m.label for m in result} == {"upload: m1.joblib", "upload: m2.joblib"}
        assert "Failed to load model metadata" in caplog.text


class TestReadConfigMetadata:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            pytest.param(
                '{"objective_name": "flux", "variables": ["x", "y"]}', "flux", id="reads_objective"
            ),
            pytest.param("{}", None, id="missing_key"),
        ],
    )
    def test_reads_objective_name(self, tmp_path: Path, content: str, expected: str | None):
        config_path = tmp_path / "config.json"
        config_path.write_text(content)
        assert _read_config_metadata(config_path) == expected

    @pytest.mark.parametrize(
        ("content", "filename"),
        [
            pytest.param("{invalid json", "config.json", id="corrupt"),
            pytest.param(None, "nonexistent.json", id="missing"),
        ],
    )
    def test_unreadable_config_logs_warning(
        self, tmp_path: Path, caplog, content: str | None, filename: str
    ):
        """A corrupt or absent config.json returns None and logs, never raises."""
        config_path = tmp_path / filename
        if content is not None:
            config_path.write_text(content)
        assert _read_config_metadata(config_path) is None
        assert "Failed to read config metadata" in caplog.text


@pytest.mark.parametrize(
    ("write_bytes", "filename"),
    [
        pytest.param(b"not a real joblib file", "model.joblib", id="corrupt"),
        pytest.param(None, "nonexistent.joblib", id="missing"),
    ],
)
def test_load_model_metadata_logs_and_returns_none(
    tmp_path: Path,
    caplog,
    write_bytes: bytes | None,
    filename: str,
):
    """The metadata loader returns (None, None) and logs when the file is corrupt or absent."""
    model_path = tmp_path / filename
    if write_bytes is not None:
        model_path.write_bytes(write_bytes)
    assert _load_model_metadata(model_path) == (None, None)
    assert "Failed to load model metadata" in caplog.text


class TestSaveUploadedModel:
    @pytest.mark.parametrize(
        ("filename", "expected_name"),
        [
            pytest.param("model.joblib", "model.joblib", id="plain_filename"),
            pytest.param(
                "../../../../evil.joblib", "evil.joblib", id="traversal_collapsed_to_basename"
            ),
        ],
    )
    def test_saved_inside_uploaded_models_dir(
        self, tmp_path: Path, filename: str, expected_name: str
    ) -> None:
        """Uploads land under uploaded_models/ under their basename — a crafted
        filename can't escape the directory."""
        payload = _bundle_bytes()
        info = save_uploaded_model(filename, payload, results_dir=tmp_path)

        saved = Path(info.path)
        assert saved == (tmp_path / "uploaded_models").resolve() / expected_name
        assert saved.read_bytes() == payload
        # The traversal target above the results dir must not exist.
        assert not (tmp_path.parent / expected_name).exists()
        # Uploads get the same sidecar as run-saved models, so listing them
        # never unpickles either (#120).
        assert json.loads(saved.with_suffix(".meta.json").read_text()) == {
            "variables": ["x"],
            "objective": None,
        }

    @pytest.mark.parametrize(
        ("filename", "content", "match"),
        [
            pytest.param(
                "model.txt", _bundle_bytes(), r"expected \.joblib", id="non_joblib_extension"
            ),
            pytest.param(
                "../.joblib", _bundle_bytes(), "Invalid model filename", id="empty_basename"
            ),
            pytest.param(
                "model.joblib", b"raw junk", "not a readable joblib file", id="unreadable_content"
            ),
            # A valid joblib file that is not a {"model", "variables"} bundle.
            pytest.param(
                "model.joblib",
                _joblib_bytes({"weights": [1.0]}),
                "Not a model bundle",
                id="non_bundle",
            ),
        ],
    )
    def test_rejects_invalid_upload(
        self, tmp_path: Path, filename: str, content: bytes, match: str
    ) -> None:
        """A rejected upload raises and never reaches uploaded_models/."""
        with pytest.raises(ValueError, match=match):
            save_uploaded_model(filename, content, results_dir=tmp_path)
        assert not (tmp_path / "uploaded_models" / Path(filename).name).exists()

    def test_duplicate_name_timestamped(self, tmp_path: Path) -> None:
        """Existing dest -> timestamped alternative name."""
        first, second = _bundle_bytes(["a"]), _bundle_bytes(["b"])
        save_uploaded_model("model.joblib", first, results_dir=tmp_path)
        info = save_uploaded_model("model.joblib", second, results_dir=tmp_path)

        saved = Path(info.path)
        assert saved.name != "model.joblib"
        assert saved.name.endswith("_model.joblib")
        assert saved.read_bytes() == second
