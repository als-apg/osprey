"""LLM-based judge for evaluating end-to-end workflow results.

The judge receives workflow execution results and plain-text expectations,
then uses an LLM to evaluate whether the workflow succeeded.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from osprey.models import get_chat_completion

#: How many times the judge is asked for a verdict before an unreadable reply
#: is the answer. Two retries absorb a model that formats one reply in three
#: badly; a judge that cannot be read three times running is reported.
JUDGE_ATTEMPTS = 3

#: How the provider adapter begins the error it raises when a structured reply
#: is not the JSON the output model describes. That message, and only that
#: message, means the judge should be asked again.
UNPARSED_VERDICT = "Failed to parse structured output"


def _default_provider_config(provider: str) -> dict[str, str] | None:
    """Build a self-contained provider_config from env vars for known providers.

    Lets the judge run without a config.yml in pytest's cwd. ``get_chat_completion``
    falls back to ``get_provider_config(provider)`` (which loads config.yml) when
    ``provider_config`` is None — that's what triggers the FileNotFoundError on
    bare repo cwd.
    """
    if provider == "als-apg":
        api_key = os.environ.get("ALS_APG_API_KEY")
        # The gateway has no built-in endpoint — it is a deployment's own host,
        # named by ALS_APG_BASE_URL. Without both halves there is nothing to
        # call, so the judge has no self-contained config and falls back to
        # whatever config.yml the run supplies.
        base_url = os.environ.get("ALS_APG_BASE_URL")
        if not api_key or not base_url:
            return None
        return {"api_key": api_key, "base_url": base_url}
    return None


class JudgeEvaluation(BaseModel):
    """Structured evaluation result from the LLM judge."""

    passed: bool = Field(description="Whether the workflow passed all expectations")
    reasoning: str = Field(description="Detailed explanation of the evaluation decision")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    warnings: list[str] = Field(
        default_factory=list, description="Non-critical issues or concerns found"
    )


@dataclass
class WorkflowResult:
    """Complete result package from a workflow execution."""

    query: str
    response: str
    execution_trace: str
    artifacts: list[Path]
    error: str | None = None
    execution_time: float | None = None


class LLMJudge:
    """LLM-based evaluator for end-to-end workflow testing.

    The judge evaluates workflow results against plain-text expectations
    using an LLM to make flexible, context-aware judgments.

    Example:
        >>> judge = LLMJudge(provider="als-apg", model="claude-haiku-4-5-20251001")
        >>> evaluation = await judge.evaluate(
        ...     result=workflow_result,
        ...     expectations="Should generate two plots and complete without errors"
        ... )
        >>> assert evaluation.passed, evaluation.reasoning
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str | None = None,
        verbose: bool = False,
        provider_config: dict[str, str] | None = None,
    ):
        """Initialize the LLM judge.

        Args:
            provider: AI provider to use for evaluation (keyword-only, required)
            model: Model name for the judge
            verbose: If True, prints detailed evaluation information
            provider_config: Explicit ``{api_key, base_url}`` to skip the
                config.yml lookup. Defaults to env-var-derived config for the
                known ``als-apg`` provider.
        """
        self.provider = provider
        # Model overridable via env so a redirected provider (e.g. cborg) can use
        # a valid model id. Explicit arg wins; unset env -> unchanged default.
        self.model = model or os.environ.get("OSPREY_E2E_JUDGE_MODEL", "claude-haiku-4-5-20251001")
        self.verbose = verbose
        self.provider_config = provider_config or _default_provider_config(provider)

    def _verdict(self, full_prompt: str) -> JudgeEvaluation:
        """Ask the judge for its structured verdict, again if it cannot be read.

        A verdict the adapter cannot parse says nothing about the run being
        judged: the model sampled a reply that is not the JSON it was asked
        for, which it does on some fraction of calls whatever the run did. The
        run itself is never repeated for that -- an agent run is the expensive
        thing under test, and its result already exists -- so the question is
        put to the judge again, the same prompt, a bounded number of times.
        The same prompt on purpose: a re-phrased question would make which
        attempt parsed a part of the verdict. The judge runs at the default
        temperature of zero, so a second ask leans on the serving side's own
        nondeterminism rather than on sampling. Any other failure of the call
        is raised as it comes: a gateway that refuses or a model that is
        missing is not a sampling accident.
        """
        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            try:
                return get_chat_completion(
                    message=full_prompt,
                    provider=self.provider,
                    model_id=self.model,
                    provider_config=self.provider_config,
                    output_model=JudgeEvaluation,
                    max_tokens=8096,
                )
            except ValueError as error:
                if not str(error).startswith(UNPARSED_VERDICT) or attempt == JUDGE_ATTEMPTS:
                    raise
                if self.verbose:
                    print(f"judge verdict did not parse (attempt {attempt}), asking again")
        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    async def evaluate(self, result: WorkflowResult, expectations: str) -> JudgeEvaluation:
        """Evaluate a workflow result against expectations.

        Args:
            result: Complete workflow execution result
            expectations: Plain text description of what should happen

        Returns:
            Structured evaluation with pass/fail and reasoning
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(result, expectations)

        if self.verbose:
            print("\n" + "=" * 80)
            print("LLM JUDGE EVALUATION")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")

        # Get LLM evaluation using structured output
        full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"

        evaluation = self._verdict(full_prompt)

        if self.verbose:
            print("\n" + "=" * 80)
            print("JUDGE DECISION")
            print("=" * 80)
            print(f"Passed: {evaluation.passed}")
            print(f"Confidence: {evaluation.confidence}")
            print(f"\nReasoning:\n{evaluation.reasoning}")
            if evaluation.warnings:
                print("\nWarnings:")
                for warning in evaluation.warnings:
                    print(f"  - {warning}")
            print("=" * 80 + "\n")

        return evaluation

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the judge."""
        return """You are an expert evaluator for AI agent workflows in a scientific control system environment.

Your role is to assess whether an AI agent successfully completed a given task by examining:
1. The user's query
2. The agent's execution trace (what steps it took)
3. The agent's response to the user
4. Artifacts produced (plots, notebooks, data files)
5. Any errors encountered

Evaluate against the stated expectations with clear pass/fail criteria:
- Did the workflow complete without critical errors?
- Were appropriate capabilities invoked?
- Were the expected outputs produced?
- Is the response coherent and helpful?
- Are there any concerning patterns or anomalies?

Be thorough but fair in your evaluation. Minor imperfections are acceptable if the core
expectations are met. Critical failures (crashes, wrong outputs, no response) should fail.

Provide a clear pass/fail decision with detailed reasoning. Be specific about what worked well and what didn't."""

    async def evaluate_text(
        self,
        result_text: str,
        expectations: str,
        query: str,
    ) -> JudgeEvaluation:
        """Evaluate arbitrary text against expectations.

        Useful for search result evaluation, RAG output validation, etc.

        Args:
            result_text: The text to evaluate (search results, generated answer, etc.)
            expectations: Plain text description of what should be present/true
            query: The original query that produced this result

        Returns:
            Structured evaluation with pass/fail and reasoning
        """
        prompt = f"""Evaluate the following search/retrieval result:

QUERY:
{query}

RESULT:
{result_text}

EXPECTATIONS:
{expectations}

Evaluate whether the result meets the expectations. Consider:
1. Are the expected items/concepts present?
2. Is the result relevant to the query?
3. Are there any factual errors or hallucinations?
4. Is important information missing?

Provide a clear PASS or FAIL decision with detailed reasoning."""

        if self.verbose:
            print("\n" + "=" * 80)
            print("LLM JUDGE TEXT EVALUATION")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")

        # Get LLM evaluation using structured output
        full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"

        evaluation = self._verdict(full_prompt)

        if self.verbose:
            print("\n" + "=" * 80)
            print("JUDGE DECISION")
            print("=" * 80)
            print(f"Passed: {evaluation.passed}")
            print(f"Confidence: {evaluation.confidence}")
            print(f"\nReasoning:\n{evaluation.reasoning}")
            if evaluation.warnings:
                print("\nWarnings:")
                for warning in evaluation.warnings:
                    print(f"  - {warning}")
            print("=" * 80 + "\n")

        return evaluation

    def _build_evaluation_prompt(self, result: WorkflowResult, expectations: str) -> str:
        """Build the evaluation prompt from result and expectations."""
        # Format artifacts list
        artifacts_str = (
            "\n".join(
                f"  - {artifact.name} ({artifact.stat().st_size if artifact.exists() else 'MISSING'} bytes)"
                for artifact in result.artifacts
            )
            if result.artifacts
            else "  (none)"
        )

        # Format error info
        error_str = f"\n\nERROR ENCOUNTERED:\n{result.error}" if result.error else ""

        # Build complete prompt
        prompt = f"""Evaluate the following workflow execution:

USER QUERY:
{result.query}

EXPECTATIONS:
{expectations}

EXECUTION TRACE:
{result._format_trace_excerpt()}

AGENT RESPONSE:
{result.response}

ARTIFACTS PRODUCED:
{artifacts_str}
{error_str}

Based on the expectations and the execution results, determine whether this workflow succeeded.

Provide:
1. A clear PASS or FAIL decision
2. Detailed reasoning explaining your decision
3. A confidence score (0.0 to 1.0)
4. Any warnings or concerns (even if passing)

Consider both critical failures (workflow didn't complete, wrong outputs, errors) and
quality issues (unclear response, missing context, suboptimal execution path)."""

        return prompt


# Add helper method to WorkflowResult
def _format_trace_excerpt(self: WorkflowResult, max_lines: int = 100) -> str:
    """Format execution trace with reasonable truncation."""
    lines = self.execution_trace.split("\n")
    if len(lines) <= max_lines:
        return self.execution_trace

    # Show first and last portions
    half = max_lines // 2
    truncated = (
        "\n".join(lines[:half])
        + f"\n\n... ({len(lines) - max_lines} lines truncated) ...\n\n"
        + "\n".join(lines[-half:])
    )
    return truncated


# Monkey-patch the method onto WorkflowResult
WorkflowResult._format_trace_excerpt = _format_trace_excerpt
