"""LLM-based judge for evaluating end-to-end workflow results.

The judge receives workflow execution results and plain-text expectations,
then uses an LLM to evaluate whether the workflow succeeded.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

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

_Reply = TypeVar("_Reply", bound=BaseModel)


def _default_provider_config(provider: str) -> dict[str, str] | None:
    """Build a self-contained provider_config from env vars for known providers.

    Lets the judge run without a config.yml in pytest's cwd. ``get_chat_completion``
    falls back to ``get_provider_config(provider)`` (which loads config.yml) when
    ``provider_config`` is None — that's what triggers the FileNotFoundError on
    bare repo cwd.
    """
    if provider == "als-apg":
        api_key = os.environ.get("ALS_APG_API_KEY")
        # The judge addresses the gateway itself rather than through the
        # provider catalog, so ALS_APG_BASE_URL is what makes it a route here.
        # Without both halves there is nothing to call, and the judge has no
        # self-contained config: it falls back to whatever config.yml the run
        # supplies.
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


@dataclass(frozen=True)
class Reference:
    """A reference value and the tolerance a reported value must hold to.

    Attributes:
        value: The correct value.
        tolerance: The largest error that still passes.
        relative: Measure the error relative to ``value`` instead of absolutely.
        modulo: Compare on a circle of this period (a tune's fractional part is
            ``modulo=1.0``), so 0.999 and 0.001 are 0.002 apart.
    """

    value: float
    tolerance: float
    relative: bool = False
    modulo: float | None = None


def check_reported(
    reported: dict[str, float | None], references: dict[str, Reference]
) -> list[str]:
    """Compare reported values to their references in code; one line per miss.

    Args:
        reported: Values read from an answer, ``None`` where it stated none.
        references: The value and tolerance each reported name must hold to.

    Returns:
        A line naming each reference that is missing or out of tolerance, with
        the reported value, the reference and the error; empty when all hold.
    """
    failures: list[str] = []
    for name, reference in references.items():
        got = reported.get(name)
        if got is None:
            failures.append(f"{name}: not reported (reference {reference.value!r})")
            continue
        difference = got - reference.value
        if reference.modulo is not None:
            half = reference.modulo / 2
            difference = (difference + half) % reference.modulo - half
        error = abs(difference) / abs(reference.value) if reference.relative else abs(difference)
        if not error <= reference.tolerance:
            kind = "relative" if reference.relative else "absolute"
            failures.append(
                f"{name}: reported {got!r}, reference {reference.value!r}, "
                f"{kind} error {error:.2e} exceeds {reference.tolerance:.0e}"
            )
    return failures


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
        """Ask the judge for its structured verdict, again if it cannot be read."""
        return self._structured(full_prompt, JudgeEvaluation)

    def _structured(self, full_prompt: str, output_model: type[_Reply]) -> _Reply:
        """Ask the judge model for a structured reply, again if it cannot be read.

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
                    output_model=output_model,
                    max_tokens=8096,
                )
            except ValueError as error:
                if not str(error).startswith(UNPARSED_VERDICT) or attempt == JUDGE_ATTEMPTS:
                    raise
                if self.verbose:
                    print(f"judge verdict did not parse (attempt {attempt}), asking again")
        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    async def extract(self, text: str, output_model: type[_Reply], instructions: str) -> _Reply:
        """Read values out of *text* into *output_model*, judging nothing.

        Extraction is the half of grading a model does reliably: finding a
        quantity under whatever name or layout an answer gave it. Comparing it
        to a reference is not -- a judge that does its own arithmetic can
        misjudge an order of magnitude -- so callers that hold reference values
        and tolerances extract here and compare in code. The references are
        never put in front of the extractor, so it cannot nudge a value toward
        them.

        Args:
            text: The answer to read.
            output_model: The fields to fill; a field the text does not state
                should be optional so the extractor can leave it unset.
            instructions: What each field means and how to read it from text.

        Returns:
            The filled-in *output_model*.
        """
        prompt = (
            "Extract values from the TEXT below into the requested structure. "
            "Copy each value exactly as the text states it -- do not compute, "
            "round, convert, or correct anything -- and leave a field null when "
            "the text does not state it.\n\n"
            f"FIELDS:\n{instructions}\n\nTEXT:\n{text}"
        )
        return self._structured(prompt, output_model)

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
