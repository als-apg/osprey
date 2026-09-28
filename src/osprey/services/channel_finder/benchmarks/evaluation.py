"""Channel finder benchmark evaluation.

Stage 1 (always-on) — Programmatic recall check: case-insensitive substring
matching to determine which expected PVs appear anywhere in the agent's
response text. Returns ``(found, missing)``. Cheap, deterministic, no API
call.

Stage 2 (opt-in by passing a ``judge``) — LLM coverage judge: a model with
structured output decides which expected channels the agent's FINAL answer
covers — counting both literal mentions AND unambiguous shorthand (e.g. "all
96 BPMs", "BPM:01 through BPM:96"). Also returns any channels the agent
recommended outside the expected set, so precision can be measured. Runs
whenever the caller opts in, regardless of whether Stage 1 found everything.

The judge runs on a provider the project configures under ``api.providers``,
resolved once by :func:`resolve_judge`. A judge that cannot run is an error
(:class:`CoverageJudgeError`), never a silent fallback to Stage 1.

The opt-in default keeps single-paradigm benchmark runs free of upstream
LLM-judge cost; cross-paradigm research that wants shorthand-tolerant
scoring opts in explicitly.

Public API:
    programmatic_recall_check  — stage 1 only
    JudgeRoute                 — the provider, endpoint, key and model the judge calls
    resolve_judge              — a JudgeRoute from a project's configuration
    llm_judge_coverage         — stage 2 only
    evaluate_response          — pipeline (stage 1 default, stage 2 opt-in)
    compute_f1                 — precision / recall / F1 from predicted vs expected
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from osprey.models.config import main_model_id
from osprey.models.provider_registry import get_provider_registry
from osprey.services.channel_finder.benchmarks.project_env import (
    expand_api_providers,
    project_env,
)
from osprey.services.channel_finder.core.exceptions import CoverageJudgeError
from osprey_connectors.config import is_unresolved_placeholder

# ---------------------------------------------------------------------------
# Stage 1 — Programmatic recall
# ---------------------------------------------------------------------------


def programmatic_recall_check(text: str, expected: list[str]) -> tuple[list[str], list[str]]:
    """Check which expected channels appear in the response text.

    Case-insensitive substring match.

    Args:
        text: Full agent response text.
        expected: List of expected channel names (PV strings).

    Returns:
        Tuple of (found, missing) — lists of channel names.
    """
    text_lower = text.lower()
    found = [ch for ch in expected if ch.lower() in text_lower]
    missing = [ch for ch in expected if ch.lower() not in text_lower]
    return found, missing


# ---------------------------------------------------------------------------
# Stage 2 — LLM coverage judge
# ---------------------------------------------------------------------------


class ChannelExtractionResult(BaseModel):
    """Structured output for LLM coverage judging.

    Indices reference into the expected list (0-based), keeping output
    short — emitting integers instead of 30-character PV names cuts the
    judge's response from ~1.2k tokens (for 96 channels) to ~100 tokens
    and removes the can't-quite-spell-the-PV failure mode.
    """

    covered_expected_indices: list[int]
    extra_recommended: list[str]
    reasoning: str


@dataclass(frozen=True)
class JudgeRoute:
    """The provider, endpoint, key and model the coverage judge calls.

    Attributes:
        provider: The registered provider adapter's name.
        model_id: The model id the provider serves.
        base_url: The endpoint, or ``None`` for a provider that needs none.
        api_key: The key to send; kept out of ``repr`` so it never reaches a log.
        extra_body: Provider-specific request fields from ``api.providers``.
    """

    provider: str
    model_id: str
    base_url: str | None
    api_key: str | None = field(default=None, repr=False)
    extra_body: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """``provider/model_id``, naming the judge in messages and metadata."""
        return f"{self.provider}/{self.model_id}"


def resolve_judge(
    project_dir: Path, provider: str, *, judge_model: str | None = None
) -> JudgeRoute:
    """Resolve the coverage judge from the project's ``api.providers``.

    ``${VAR}`` references expand against the process environment over the
    project's ``.env``; the provider's registered adapter then resolves the
    endpoint (its override variable wins) and the key (a keyless adapter gets
    its placeholder).

    Args:
        project_dir: The benchmarked project's directory, holding ``config.yml``.
        provider: The provider to judge on; must be configured and registered.
        judge_model: The judge's model id. Omitted, the deployment's main model
            when ``claude_code.provider`` is this provider, else the entry's
            ``default_model``.

    Returns:
        The route :func:`llm_judge_coverage` calls.

    Raises:
        CoverageJudgeError: When the project's configuration cannot run the judge;
            the message names the provider and the config path.
    """
    config_path = project_dir / "config.yml"
    if not config_path.is_file():
        raise CoverageJudgeError(
            f"The coverage judge on '{provider}' needs {config_path}, and it does not exist."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    env = project_env(project_dir)
    providers = expand_api_providers(config, env)

    registry = get_provider_registry()
    adapter = registry.get_provider(provider)
    if adapter is None:
        known = ", ".join(sorted(registry.list_providers()))
        raise CoverageJudgeError(
            f"No provider adapter is registered under '{provider}', named for the coverage "
            f"judge by {config_path}. Name one of {known}."
        )

    entry = providers.get(provider)
    if not isinstance(entry, dict):
        configured = ", ".join(sorted(providers)) or "none"
        raise CoverageJudgeError(
            f"The coverage judge's provider '{provider}' is not configured under "
            f"api.providers in {config_path}. Configured: {configured}."
        )

    raw_key = entry.get("api_key")
    if is_unresolved_placeholder(raw_key):
        raise CoverageJudgeError(
            f"api.providers.{provider}.api_key in {config_path} is {raw_key}, and that "
            f"variable is set neither in the environment nor in {project_dir / '.env'}."
        )
    api_key = adapter.effective_api_key(raw_key or None)
    if api_key is None:
        raise CoverageJudgeError(
            f"api.providers.{provider}.api_key in {config_path} names no key, and the "
            f"'{provider}' provider requires one for the coverage judge."
        )

    raw_url = entry.get("base_url")
    try:
        base_url = adapter.resolve_base_url(raw_url)
    except ValueError as exc:
        quoted = (
            f" is {raw_url}, which is not set, and" if is_unresolved_placeholder(raw_url) else ""
        )
        raise CoverageJudgeError(
            f"api.providers.{provider}.base_url in {config_path}{quoted} names no endpoint "
            f"for the coverage judge: {exc}"
        ) from exc

    if judge_model:
        model_id = judge_model
    else:
        try:
            model_id = main_model_id(
                {"claude_code": config.get("claude_code") or {}, "api": {"providers": providers}},
                provider,
            )
        except ValueError as exc:
            raise CoverageJudgeError(
                f"The coverage judge on '{provider}' has no model in {config_path}: {exc}"
            ) from exc

    extra_body = entry.get("extra_body")
    return JudgeRoute(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        extra_body=extra_body if isinstance(extra_body, dict) else None,
    )


def llm_judge_coverage(
    response_text: str, expected: list[str], *, judge: JudgeRoute
) -> tuple[list[str], list[str]]:
    """Judge which expected channels the agent's final answer covers.

    Calls the judge's provider adapter with structured output. The judge
    decides coverage based on the agent's FINAL answer only — both literal
    enumeration and unambiguous shorthand ("all 96 BPMs", "BPM:01 through
    BPM:96") count as coverage. It also returns any channels the agent
    recommended outside the expected set, so precision can be measured.

    Args:
        response_text: Full agent response text.
        expected: Expected channel names — the canonical naming the judge
            scores coverage against.
        judge: The route resolved by :func:`resolve_judge`.

    Returns:
        Tuple of (covered_expected, extra_recommended). ``covered_expected``
        is a subset of ``expected``. ``extra_recommended`` is anything the
        agent named in its final answer that's not in ``expected``.

    Raises:
        CoverageJudgeError: When the judge call fails, or its reply carries no
            structured verdict.
    """
    # Number the expected list so the judge can refer to entries by index.
    expected_numbered = "\n".join(f"{i}: {ch}" for i, ch in enumerate(expected))
    prompt = (
        "Evaluate a channel finder agent's FINAL answer.\n\n"
        "Return two fields:\n"
        "1. covered_expected_indices: 0-based indices into the expected list "
        "below for channels the agent's final answer covers. Count both "
        "literal mentions AND unambiguous shorthand (e.g. 'all 96 BPMs', "
        "'BPM:01 through BPM:96'). Channels mentioned only during "
        "exploration do NOT count.\n"
        "2. extra_recommended: channels the agent recommends in its final "
        "answer that are NOT in the expected list (literal PV strings).\n\n"
        f"Expected channels (indexed):\n{expected_numbered}\n\n"
        f"Agent response:\n{response_text}"
    )

    adapter = get_provider_registry().get_provider(judge.provider)
    if adapter is None:
        raise CoverageJudgeError(
            f"The coverage judge {judge.label} failed: no provider adapter is registered "
            f"under '{judge.provider}'."
        )
    extra: dict[str, Any] = {"extra_body": judge.extra_body} if judge.extra_body else {}
    try:
        result = adapter().execute_completion(
            message=prompt,
            model_id=judge.model_id,
            api_key=judge.api_key,
            base_url=judge.base_url,
            max_tokens=2048,
            temperature=0.0,
            output_format=ChannelExtractionResult,
            **extra,
        )
    except Exception as exc:
        raise CoverageJudgeError(f"The coverage judge {judge.label} failed: {exc}") from exc

    if not isinstance(result, ChannelExtractionResult):
        raise CoverageJudgeError(
            f"The coverage judge {judge.label} returned no structured verdict."
        )
    covered = [expected[i] for i in result.covered_expected_indices if 0 <= i < len(expected)]
    return covered, result.extra_recommended


# ---------------------------------------------------------------------------
# Combined two-stage pipeline
# ---------------------------------------------------------------------------


def evaluate_response(
    response_text: str,
    expected: list[str],
    *,
    judge: JudgeRoute | None = None,
) -> tuple[list[str], dict]:
    """Evaluate a channel finder response.

    Stage 1 (always): Programmatic recall check — which expected channels
    appear literally in the response text.

    Stage 2 (opt-in): LLM coverage judge — resolves shorthand to coverage
    and detects over-recommendation. Runs whenever a ``judge`` is given
    and there is something to evaluate (``expected`` non-empty). Replaces
    Stage 1's literal-only signal with the judge's semantic coverage
    decision.

    Args:
        response_text: Full agent response text (plain string).
        expected: List of expected channel names.
        judge: The route of the Stage 2 LLM judge. Omitted — pure
            programmatic evaluation, no upstream LLM call.

    Returns:
        Tuple of (predicted_channels, metadata_dict). ``predicted_channels``
        is what should be fed to :func:`compute_f1`. With the judge it is
        ``covered_expected + extra_recommended``, so precision and recall
        both reflect the judge's decision.

    Raises:
        CoverageJudgeError: When the judge cannot score the response; the
            query is not rescored another way.
    """
    found, missing = programmatic_recall_check(response_text, expected)
    meta: dict[str, Any] = {"stage": 1, "found": found, "missing": missing}

    if judge is None:
        meta["evaluation"] = (
            "programmatic_recall_only" if not missing else "programmatic_recall_fail"
        )
        return found, meta

    if not expected:
        # Nothing to judge — skip the LLM call entirely.
        meta["evaluation"] = "programmatic_recall_only"
        return found, meta

    # Stage 2: judge runs whether or not Stage 1 found everything, so
    # shorthand-only answers ("all 96 BPMs") can still earn coverage.
    meta["stage"] = 2
    meta["judge"] = judge.label
    covered, extras = llm_judge_coverage(response_text, expected, judge=judge)
    meta["evaluation"] = "llm_judge"
    meta["llm_covered"] = covered
    meta["llm_extras"] = extras
    return covered + extras, meta


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_f1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from predicted and expected channel lists.

    Args:
        predicted: Channels the agent recommended.
        expected: Ground-truth channels.

    Returns:
        Tuple of (precision, recall, f1).  When both lists are empty the
        result is (1.0, 1.0, 1.0).  When only one is empty the result is
        (0.0, 0.0, 0.0).
    """
    pred_set = {p.upper() for p in predicted}
    exp_set = {e.upper() for e in expected}

    if not pred_set and not exp_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not exp_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall = tp / len(exp_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
