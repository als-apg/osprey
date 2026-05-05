"""Benchmark runner for channel finder evaluation.

Runs benchmark queries against a single OSPREY project using the Claude
Agent SDK.  For each query it:
  1. Sends the query via ``run_sdk_query()``
  2. Evaluates the response with ``evaluate_response()`` + ``compute_f1()``
  3. Collects a ``QueryResult``
  4. Aggregates into a ``BenchmarkRun``

Public API:
    BenchmarkRunner  — main orchestrator class

Cross-paradigm matrix orchestration lives in the companion paper repository.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from osprey.services.channel_finder.benchmarks.backends import create_backend
from osprey.services.channel_finder.benchmarks.evaluation import (
    compute_f1,
    evaluate_response,
)
from osprey.services.channel_finder.benchmarks.models import (
    BenchmarkRun,
    QueryResult,
)

logger = logging.getLogger(__name__)

# Map paradigm names to their config key path for the database
PARADIGM_CONFIG_KEYS: dict[str, list[str]] = {
    "in_context": [
        "channel_finder",
        "pipelines",
        "in_context",
        "database",
        "path",
    ],
    "hierarchical": [
        "channel_finder",
        "pipelines",
        "hierarchical",
        "database",
        "path",
    ],
    "middle_layer": [
        "channel_finder",
        "pipelines",
        "middle_layer",
        "database",
        "path",
    ],
}


def read_db_path_from_config(project_dir: Path, paradigm: str) -> Path:
    """Read the database path from a project's config.yml.

    Args:
        project_dir: Root of the initialized OSPREY project.
        paradigm: One of ``in_context``, ``hierarchical``, ``middle_layer``.

    Returns:
        Resolved absolute Path to the database file.

    Raises:
        KeyError: If the config key path is missing.
        FileNotFoundError: If config.yml does not exist.
    """
    import yaml

    config_path = project_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yml not found in {project_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    keys = PARADIGM_CONFIG_KEYS[paradigm]
    node = config
    for key in keys:
        node = node[key]

    db_path = Path(str(node))
    if not db_path.is_absolute():
        db_path = project_dir / db_path
    return db_path.resolve()


def model_slug(model: str) -> str:
    """Convert a model identifier to a filename-safe slug.

    Examples:
        ``"anthropic/claude-haiku"`` -> ``"anthropic_claude-haiku"``
        ``"claude-sonnet-4-5"`` -> ``"claude-sonnet-4-5"``
    """
    return model.replace("/", "_").replace(" ", "_")


class BenchmarkRunner:
    """Runs benchmark queries against a single OSPREY project.

    The runner speaks model **tiers** (``haiku`` / ``sonnet`` / ``opus``).
    The concrete wire id and provider come from the project's ``config.yml``
    via ``ClaudeCodeModelResolver`` — the same resolution path the
    production Claude Code CLI uses. Each backend then formats the wire id
    for its own consumer (bare for the SDK CLI; provider-prefixed slug for
    LiteLLM).
    """

    def __init__(
        self,
        project_dir: Path,
        *,
        model_tier: str = "haiku",
        max_turns: int = 25,
        max_budget_per_query: float = 2.0,
        max_concurrent: int = 5,
        verbose: bool = False,
        queries_override: Path | None = None,
        backend: str = "auto",
        repeat_idx: int = 0,
        use_llm_judge: bool = False,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.model_tier = model_tier
        self._spec = self._resolve_model_spec()
        if model_tier not in self._spec.tier_to_model:
            available = ", ".join(sorted(self._spec.tier_to_model)) or "(none)"
            raise KeyError(
                f"Tier {model_tier!r} not configured for provider "
                f"{self._spec.provider!r}; available: {available}"
            )
        # Bare wire id (e.g. "claude-haiku-4-5-20251001"). Used for output
        # filename slugs and BenchmarkRun serialization. Each backend
        # converts this to its own grammar.
        self.model = self._spec.tier_to_model[model_tier]
        self.max_turns = max_turns
        self.max_budget_per_query = max_budget_per_query
        self.max_concurrent = max_concurrent
        self.verbose = verbose
        self.queries_override = queries_override
        self.repeat_idx = repeat_idx
        self.use_llm_judge = use_llm_judge
        self._backend = create_backend(
            backend,
            self.project_dir,
            self._spec,
            model_tier,
            max_turns=max_turns,
            max_budget_usd=max_budget_per_query,
        )
        self.backend_name = self._backend.name

    def _resolve_model_spec(self):
        """Resolve the project's Claude Code model spec from config.yml."""
        from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

        config = self._read_config()
        cc_config = config.get("claude_code", {})
        api_providers = config.get("api", {}).get("providers", {})
        spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
        if spec is None:
            raise ValueError(
                f"No claude_code.provider configured in {self.project_dir}/config.yml; "
                "BenchmarkRunner cannot resolve a model wire id."
            )
        return spec

    def _read_config(self) -> dict[str, Any]:
        """Read and return the project's config.yml as a dict."""
        import yaml

        config_path = self.project_dir / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yml not found in {self.project_dir}")
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def _count_channels(self) -> int:
        """Count channels in the active pipeline's database."""
        mode = self._resolve_pipeline_mode()
        try:
            db_path = read_db_path_from_config(self.project_dir, mode)
            if not db_path.exists():
                return 0
            data = json.loads(db_path.read_text(encoding="utf-8"))
            if isinstance(data, list):  # in_context format
                return len(data)
            if isinstance(data, dict) and "tree" in data:  # hierarchical
                from osprey.services.channel_finder.benchmarks.generator import (
                    expand_hierarchy,
                )

                return len(expand_hierarchy(data))
            # middle_layer — count ChannelNames
            from osprey.services.channel_finder.benchmarks.generator import (
                collect_middle_layer_pvs,
            )

            return len(collect_middle_layer_pvs(data))
        except Exception:
            return 0

    def _resolve_pipeline_mode(self) -> str:
        """Read the active pipeline mode from config."""
        config = self._read_config()
        return config["channel_finder"]["pipeline_mode"]

    def _resolve_queries_path(self) -> Path:
        """Resolve the benchmark queries file path.

        Uses ``queries_override`` if set; otherwise reads the path from
        ``channel_finder.pipelines.<mode>.benchmark.dataset_path`` in config.
        """
        if self.queries_override is not None:
            return Path(self.queries_override)

        config = self._read_config()
        mode = config["channel_finder"]["pipeline_mode"]
        pipeline_cfg = config["channel_finder"]["pipelines"][mode]
        dataset_path = pipeline_cfg["benchmark"]["dataset_path"]

        path = Path(dataset_path)
        if not path.is_absolute():
            path = self.project_dir / path
        return path

    def load_queries(self) -> list[dict[str, Any]]:
        """Load and return benchmark queries from the resolved path."""
        queries_path = self._resolve_queries_path()
        queries: list[dict[str, Any]] = json.loads(queries_path.read_text(encoding="utf-8"))
        return queries

    async def run_queries(
        self,
        *,
        query_indices: list[int] | None = None,
        output_dir: Path | None = None,
        progress_callback: Callable[[QueryResult], None] | None = None,
        progress_bar: Any | None = None,
    ) -> BenchmarkRun:
        """Run benchmark queries concurrently against the project.

        Args:
            query_indices: If provided, only run these query indices.
                Pass ``None`` to run all queries.
            output_dir: If provided, save per-query result JSONs here.
            progress_callback: Called with each ``QueryResult`` as it completes.
            progress_bar: Optional tqdm progress bar to update per query.

        Returns:
            BenchmarkRun with per-query results and aggregates.
        """
        all_queries = self.load_queries()

        if query_indices is not None:
            queries = [(i, all_queries[i]) for i in query_indices if i < len(all_queries)]
        else:
            queries = list(enumerate(all_queries))

        pipeline_mode = self._resolve_pipeline_mode()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Running accumulators for live progress display
        completed_count = 0
        total_f1 = 0.0
        total_cost = 0.0

        async def _run_one(idx: int, query: dict) -> QueryResult | None:
            nonlocal completed_count, total_f1, total_cost
            user_query = query["user_query"]
            expected = query["targeted_pv"]

            try:
                async with semaphore:
                    logger.info(
                        "Query %d: %s",
                        idx,
                        user_query[:80],
                    )

                    t0 = time.monotonic()
                    output = await self._backend.run_query(user_query, pipeline_mode)
                    latency = time.monotonic() - t0

                predicted, eval_meta = evaluate_response(
                    output.response_text,
                    expected,
                    use_llm_judge=self.use_llm_judge,
                )
                precision, recall, f1 = compute_f1(predicted, expected)

                # Serialize tool traces for post-hoc analysis
                tool_calls = [
                    {
                        "name": t.name,
                        "input": t.input,
                        "result": t.result,
                        "is_error": t.is_error,
                    }
                    for t in output.tool_traces
                ]

                qr = QueryResult(
                    query_id=idx,
                    user_query=user_query,
                    expected=expected,
                    predicted=predicted,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    cost_usd=output.cost_usd,
                    num_turns=output.num_turns,
                    latency_s=latency,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    eval_meta=eval_meta,
                    tool_calls=tool_calls,
                    response_text=output.response_text,
                )

                # Update running accumulators and progress bar
                completed_count += 1
                total_f1 += f1
                total_cost += qr.cost_usd
                if progress_bar is not None:
                    avg_f1 = total_f1 / completed_count
                    progress_bar.set_postfix(
                        F1=f"{avg_f1:.2f}",
                        cost=f"${total_cost:.3f}",
                        last=f"Q{idx}={f1:.2f}",
                    )
                    progress_bar.update(1)

                if self.verbose:
                    logger.info(
                        "  -> Q%d F1=%.2f P=%.2f R=%.2f (%.1fs)",
                        idx,
                        f1,
                        precision,
                        recall,
                        latency,
                    )

                if output_dir is not None:
                    slug = model_slug(self.model)
                    qr.to_json(
                        output_dir
                        / f"query_{idx}_{slug}_{self.backend_name}_r{self.repeat_idx}.json"
                    )

                if progress_callback is not None:
                    progress_callback(qr)

                return qr
            except Exception:
                logger.exception("Query %d failed", idx)
                if progress_bar is not None:
                    progress_bar.update(1)
                return None

        tasks = [_run_one(idx, q) for idx, q in queries]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]
        num_failed = len(raw_results) - len(results)

        channel_count = self._count_channels()

        return BenchmarkRun.from_query_results(
            paradigm=pipeline_mode,
            model=self.model,
            results=results,
            channel_count=channel_count,
            num_failed=num_failed,
            backend=self.backend_name,
            repeat_idx=self.repeat_idx,
        )
