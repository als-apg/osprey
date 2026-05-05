"""Tests for the benchmark runner.

Tests cover:
  - BenchmarkRunner initialization
  - Config reading and query resolution
  - run_queries with mocked SDK calls
  - Query index filtering
  - JSON output saving when output_dir is provided
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.services.channel_finder.benchmarks.models import BenchmarkRun
from osprey.services.channel_finder.benchmarks.runner import (
    BenchmarkRunner,
    model_slug,
    read_db_path_from_config,
)

# Module path for patching
_RUNNER_MOD = "osprey.services.channel_finder.benchmarks.runner"
_SDK_BACKEND_MOD = "osprey.services.channel_finder.benchmarks.backends.sdk_backend"


# ---------------------------------------------------------------------------
# Test data & helpers
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    {
        "query_id": 0,
        "user_query": "Find the storage ring dipole current setpoint",
        "targeted_pv": ["SR:MAG:DIPOLE:B01:CURRENT:SP"],
    },
    {
        "query_id": 1,
        "user_query": "Show me all BPM horizontal positions",
        "targeted_pv": [
            "SR:DIAG:BPM:BPM01:POSITION:X",
            "SR:DIAG:BPM:BPM02:POSITION:X",
        ],
    },
]


@dataclass
class FakeSDKResult:
    """Minimal stand-in for SDKWorkflowResult."""

    text_blocks: list[str] = field(default_factory=list)
    tool_traces: list = field(default_factory=list)
    result: MagicMock | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return 0.01

    @property
    def num_turns(self) -> int:
        return 3


def _make_fake_sdk_result(pvs: list[str]) -> FakeSDKResult:
    """Create a fake SDK result that mentions the given PVs."""
    text = "The recommended channels are: " + ", ".join(pvs) + "."
    return FakeSDKResult(text_blocks=[text])


def _make_project_dir(
    tmp_path: Path,
    *,
    pipeline_mode: str = "in_context",
    queries: list[dict] | None = None,
    provider: str = "anthropic",
) -> Path:
    """Create a fake project directory with config.yml and benchmark queries.

    Always writes a ``claude_code.provider`` so ``BenchmarkRunner`` can
    resolve a ``ClaudeCodeModelSpec`` (the runner refuses to construct
    without one). Defaults to ``anthropic`` so the resolver picks up
    built-in tier→wire-id mappings without needing ``api.providers``.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)

    queries_path = project_dir / "data" / "benchmark_queries.json"
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(
        json.dumps(queries or SAMPLE_QUERIES, indent=2),
        encoding="utf-8",
    )

    config = {
        "claude_code": {"provider": provider},
        "channel_finder": {
            "pipeline_mode": pipeline_mode,
            "pipelines": {
                pipeline_mode: {
                    "database": {
                        "path": "data/channel_databases/channels.json",
                    },
                    "benchmark": {
                        "dataset_path": "data/benchmark_queries.json",
                    },
                }
            },
        },
    }
    (project_dir / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
    return project_dir


# Wire IDs that the anthropic provider's CLAUDE_CODE_PROVIDERS entry maps tiers to.
# Used by the assertions below — kept here so a model-ID change in the resolver
# is caught at one place rather than scattered across tests.
_HAIKU_WIRE = "claude-haiku-4-5-20251001"
_SONNET_WIRE = "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# TestBenchmarkRunnerInit
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerInit:
    """BenchmarkRunner stores parameters correctly."""

    def test_defaults(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir)
        assert runner.project_dir == project_dir
        assert runner.model_tier == "haiku"
        assert runner.model == _HAIKU_WIRE
        assert runner.max_turns == 25
        assert runner.max_budget_per_query == 2.0
        assert runner.max_concurrent == 5
        assert runner.verbose is False
        assert runner.queries_override is None

    def test_custom_params(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        override = tmp_path / "custom.json"
        runner = BenchmarkRunner(
            project_dir,
            model_tier="sonnet",
            max_turns=10,
            max_budget_per_query=5.0,
            max_concurrent=3,
            verbose=True,
            queries_override=override,
        )
        assert runner.model_tier == "sonnet"
        assert runner.model == _SONNET_WIRE
        assert runner.max_turns == 10
        assert runner.max_budget_per_query == 5.0
        assert runner.max_concurrent == 3
        assert runner.verbose is True
        assert runner.queries_override == override

    def test_unknown_tier_raises(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        with pytest.raises(KeyError, match="not configured for provider"):
            BenchmarkRunner(project_dir, model_tier="bogus-tier")

    def test_missing_provider_raises(self, tmp_path: Path):
        project_dir = tmp_path / "no-provider"
        project_dir.mkdir()
        # Config without claude_code.provider — runner must refuse to construct.
        (project_dir / "config.yml").write_text(
            yaml.dump({"channel_finder": {"pipeline_mode": "in_context"}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="No claude_code.provider configured"):
            BenchmarkRunner(project_dir)


# ---------------------------------------------------------------------------
# TestConfigReading
# ---------------------------------------------------------------------------


class TestConfigReading:
    """Verify config resolution methods."""

    def test_read_config(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir)
        config = runner._read_config()
        assert config["channel_finder"]["pipeline_mode"] == "in_context"

    def test_read_config_missing_file_after_delete(self, tmp_path: Path):
        # Construct against a valid config, then remove it to verify the
        # internal reader still raises when called later. (Construction
        # itself eagerly resolves the spec, so we can't construct without
        # config — that case is covered by ``test_missing_provider_raises``.)
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir)
        (project_dir / "config.yml").unlink()
        with pytest.raises(FileNotFoundError):
            runner._read_config()

    def test_resolve_pipeline_mode(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path, pipeline_mode="hierarchical")
        runner = BenchmarkRunner(project_dir)
        assert runner._resolve_pipeline_mode() == "hierarchical"

    def test_resolve_queries_path_from_config(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir)
        path = runner._resolve_queries_path()
        assert path == project_dir / "data" / "benchmark_queries.json"

    def test_resolve_queries_path_override(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        override = tmp_path / "custom_queries.json"
        runner = BenchmarkRunner(project_dir, queries_override=override)
        assert runner._resolve_queries_path() == override

    def test_load_queries(self, tmp_path: Path):
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir)
        queries = runner.load_queries()
        assert len(queries) == 2
        assert queries[0]["user_query"] == SAMPLE_QUERIES[0]["user_query"]
        assert queries[1]["targeted_pv"] == SAMPLE_QUERIES[1]["targeted_pv"]


# ---------------------------------------------------------------------------
# TestRunQueries
# ---------------------------------------------------------------------------


class TestRunQueries:
    """Mock run_sdk_query to verify run_queries logic.

    Tests force ``backend="sdk"`` so that patching ``sdk_backend.run_sdk_query``
    intercepts the dispatch path. Without this, the default ``backend="auto"``
    would observe ``pipeline_mode="in_context"`` and construct an
    ``InContextBackend`` that spawns a real MCP subprocess.
    """

    @pytest.mark.asyncio()
    async def test_run_queries_returns_benchmark_run(self, tmp_path: Path):
        """run_queries returns a BenchmarkRun with correct paradigm and model."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        fake_result_q0 = _make_fake_sdk_result(SAMPLE_QUERIES[0]["targeted_pv"])
        fake_result_q1 = _make_fake_sdk_result(SAMPLE_QUERIES[1]["targeted_pv"])

        mock_sdk = AsyncMock(side_effect=[fake_result_q0, fake_result_q1])

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            result = await runner.run_queries()

        assert isinstance(result, BenchmarkRun)
        assert result.paradigm == "in_context"
        assert result.tier is None
        assert result.model == _HAIKU_WIRE
        assert len(result.query_results) == 2

        for qr in result.query_results:
            assert qr.f1 == 1.0
            assert qr.precision == 1.0
            assert qr.recall == 1.0

        assert mock_sdk.call_count == 2

    @pytest.mark.asyncio()
    async def test_run_queries_partial_match(self, tmp_path: Path):
        """run_queries handles partial matches (F1 < 1.0)."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        fake_result = _make_fake_sdk_result(["SR:MAG:DIPOLE:B01:CURRENT:SP"])
        mock_sdk = AsyncMock(return_value=fake_result)

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (
                    [expected[0]] if expected else [],
                    {"stage": 1},
                ),
            ),
        ):
            result = await runner.run_queries()

        # Query 0 has 1 expected PV, predicted 1 -> F1=1.0
        assert result.query_results[0].f1 == 1.0
        # Query 1 has 2 expected PVs, predicted 1 -> partial
        qr1 = result.query_results[1]
        assert qr1.recall == 0.5
        assert qr1.f1 < 1.0

    @pytest.mark.asyncio()
    async def test_run_queries_with_indices(self, tmp_path: Path):
        """run_queries respects query_indices filter."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        fake_result = _make_fake_sdk_result(SAMPLE_QUERIES[1]["targeted_pv"])
        mock_sdk = AsyncMock(return_value=fake_result)

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            result = await runner.run_queries(query_indices=[1])

        assert len(result.query_results) == 1
        assert result.query_results[0].query_id == 1
        assert mock_sdk.call_count == 1

    @pytest.mark.asyncio()
    async def test_run_queries_out_of_range_index_skipped(self, tmp_path: Path):
        """run_queries silently skips out-of-range indices."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        fake_result = _make_fake_sdk_result(SAMPLE_QUERIES[0]["targeted_pv"])
        mock_sdk = AsyncMock(return_value=fake_result)

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            result = await runner.run_queries(query_indices=[0, 999])

        assert len(result.query_results) == 1
        assert mock_sdk.call_count == 1

    @pytest.mark.asyncio()
    async def test_run_queries_progress_callback(self, tmp_path: Path):
        """run_queries calls progress_callback for each query."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        fake_result = _make_fake_sdk_result(["SR:MAG:DIPOLE:B01:CURRENT:SP"])
        mock_sdk = AsyncMock(return_value=fake_result)
        callbacks = []

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            await runner.run_queries(progress_callback=callbacks.append)

        assert len(callbacks) == 2

    @pytest.mark.asyncio()
    async def test_run_queries_saves_per_query_json(self, tmp_path: Path):
        """run_queries saves per-query JSON when output_dir is set."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")
        output_dir = tmp_path / "results"

        fake_result = _make_fake_sdk_result(["SR:MAG:DIPOLE:B01:CURRENT:SP"])
        mock_sdk = AsyncMock(return_value=fake_result)

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            await runner.run_queries(output_dir=output_dir)

        assert output_dir.exists()
        json_files = list(output_dir.glob(f"query_*_{_HAIKU_WIRE}_sdk_r0.json"))
        assert len(json_files) == 2

    @pytest.mark.asyncio()
    async def test_repeat_idx_suffix_in_filenames(self, tmp_path: Path):
        """repeat_idx is reflected in per-query filenames and on the BenchmarkRun."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk", repeat_idx=2)
        output_dir = tmp_path / "results"

        fake_result = _make_fake_sdk_result(["SR:MAG:DIPOLE:B01:CURRENT:SP"])
        mock_sdk = AsyncMock(return_value=fake_result)

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            run = await runner.run_queries(output_dir=output_dir)

        assert run.repeat_idx == 2
        files_r2 = list(output_dir.glob(f"query_*_{_HAIKU_WIRE}_sdk_r2.json"))
        files_r0 = list(output_dir.glob(f"query_*_{_HAIKU_WIRE}_sdk_r0.json"))
        assert len(files_r2) == 2
        assert len(files_r0) == 0


# ---------------------------------------------------------------------------
# num_failed observability
# ---------------------------------------------------------------------------


class TestNumFailed:
    """Tests for exception-counting via num_failed."""

    @pytest.mark.asyncio()
    async def test_one_exception_returns_fewer_results(self, tmp_path: Path):
        """One failing query yields num_failed=1."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        fake_result = _make_fake_sdk_result(SAMPLE_QUERIES[0]["targeted_pv"])
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return fake_result

        with (
            patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", side_effect=_side_effect),
            patch(
                f"{_RUNNER_MOD}.evaluate_response",
                side_effect=lambda text, expected, **kwargs: (expected, {"stage": 1}),
            ),
        ):
            result = await runner.run_queries()

        assert len(result.query_results) == 1
        assert result.num_failed == 1

    @pytest.mark.asyncio()
    async def test_all_exceptions_returns_empty(self, tmp_path: Path):
        """All failing queries yields num_failed == total."""
        project_dir = _make_project_dir(tmp_path)
        runner = BenchmarkRunner(project_dir, backend="sdk")

        mock_sdk = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(f"{_SDK_BACKEND_MOD}.run_sdk_query", mock_sdk):
            result = await runner.run_queries()

        assert len(result.query_results) == 0
        assert result.num_failed == 2


# ---------------------------------------------------------------------------
# Helper tests (unchanged functions)
# ---------------------------------------------------------------------------


class TestModelSlug:
    """Tests for the model_slug helper."""

    def test_slash(self):
        assert model_slug("anthropic/claude-haiku") == "anthropic_claude-haiku"

    def test_no_slash(self):
        assert model_slug("claude-sonnet-4-5") == "claude-sonnet-4-5"

    def test_spaces(self):
        assert model_slug("some model/name") == "some_model_name"


class TestReadDbPathFromConfig:
    """Tests for read_db_path_from_config."""

    def test_reads_in_context_path(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        config = {
            "channel_finder": {
                "pipelines": {
                    "in_context": {
                        "database": {
                            "path": "data/channel_databases/tiers/tier1/in_context.json",
                        }
                    }
                }
            }
        }
        (project_dir / "config.yml").write_text(yaml.dump(config), encoding="utf-8")

        result = read_db_path_from_config(project_dir, "in_context")
        expected = (
            project_dir
            / "data"
            / "channel_databases"
            / "tiers"
            / "tier1"
            / "in_context.json"
        ).resolve()
        assert result == expected

    def test_missing_config_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_db_path_from_config(tmp_path, "in_context")

    def test_missing_key_raises(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "config.yml").write_text(yaml.dump({"channel_finder": {}}), encoding="utf-8")

        with pytest.raises(KeyError):
            read_db_path_from_config(project_dir, "in_context")
