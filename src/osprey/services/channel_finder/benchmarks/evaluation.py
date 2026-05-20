"""Channel finder benchmark evaluation.

Stage 1 (always-on) — Programmatic recall check: case-insensitive substring
matching to determine which expected PVs appear anywhere in the agent's
response text. Returns ``(found, missing)``. Cheap, deterministic, no API
call.

Stage 2 (opt-in via ``use_llm_judge=True``) — LLM coverage judge: uses a
small LLM (Haiku via LiteLLM) with structured output to decide which
expected channels the agent's FINAL answer covers — counting both literal
mentions AND unambiguous shorthand (e.g. "all 96 BPMs", "BPM:01 through
BPM:96"). Also returns any channels the agent recommended outside the
expected set, so precision can be measured. Runs whenever the caller opts
in, regardless of whether Stage 1 found everything.

The opt-in default keeps single-paradigm benchmark runs free of upstream
LLM-judge cost; cross-paradigm research that wants shorthand-tolerant
scoring opts in explicitly.

Public API:
    programmatic_recall_check  — stage 1 only
    llm_judge_coverage         — stage 2 only
    evaluate_response          — pipeline (stage 1 default, stage 2 opt-in)
    compute_f1                 — precision / recall / F1 from predicted vs expected
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel

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


def llm_judge_coverage(
    response_text: str, expected: list[str]
) -> tuple[list[str], list[str]]:
    """Judge which expected channels the agent's final answer covers.

    Calls Haiku via OSPREY's LiteLLM adapter with structured output. The
    judge decides coverage based on the agent's FINAL answer only — both
    literal enumeration and unambiguous shorthand ("all 96 BPMs",
    "BPM:01 through BPM:96") count as coverage. It also returns any
    channels the agent recommended outside the expected set, so precision
    can be measured.

    Args:
        response_text: Full agent response text.
        expected: Expected channel names — the canonical naming the judge
            scores coverage against.

    Returns:
        Tuple of (covered_expected, extra_recommended). ``covered_expected``
        is a subset of ``expected``. ``extra_recommended`` is anything the
        agent named in its final answer that's not in ``expected``. Both
        empty on structured-output parse failure.
    """
    from osprey.models.providers.litellm_adapter import (
        execute_litellm_completion,
    )

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

    # Resolve provider. Preference order: direct Anthropic, then ALS-APG
    # (works off-VPN), then CBORG (LBLnet/VPN-only — last resort because
    # off-VPN traffic gets IP-blocked).
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    provider = "anthropic"
    model_id = "claude-haiku-4-5-20251001"
    base_url = None

    if not api_key:
        als_apg_key = os.environ.get("ALS_APG_API_KEY")
        if als_apg_key:
            provider = "als-apg"
            api_key = als_apg_key
            model_id = "claude-haiku-4-5-20251001"
            base_url = "https://llm.gianlucamartino.com"
        else:
            cborg_key = os.environ.get("CBORG_API_KEY")
            auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
            if cborg_key or auth_token:
                provider = "cborg"
                api_key = cborg_key or auth_token
                model_id = "anthropic/claude-haiku"
                base_url = os.environ.get(
                    "ANTHROPIC_BASE_URL", "https://api.cborg.lbl.gov/v1"
                )

    result = execute_litellm_completion(
        provider=provider,
        message=prompt,
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        max_tokens=2048,
        temperature=0.0,
        output_format=ChannelExtractionResult,
    )

    if isinstance(result, ChannelExtractionResult):
        covered = [
            expected[i]
            for i in result.covered_expected_indices
            if 0 <= i < len(expected)
        ]
        return covered, result.extra_recommended
    # Fallback: if structured output didn't parse, return empty lists.
    return [], []


# ---------------------------------------------------------------------------
# Combined two-stage pipeline
# ---------------------------------------------------------------------------


def evaluate_response(
    response_text: str,
    expected: list[str],
    *,
    use_llm_judge: bool = False,
) -> tuple[list[str], dict]:
    """Evaluate a channel finder response.

    Stage 1 (always): Programmatic recall check — which expected channels
    appear literally in the response text.

    Stage 2 (opt-in): LLM coverage judge — resolves shorthand to coverage
    and detects over-recommendation. Runs whenever ``use_llm_judge=True``
    and there is something to evaluate (``expected`` non-empty). Replaces
    Stage 1's literal-only signal with the judge's semantic coverage
    decision.

    Args:
        response_text: Full agent response text (plain string).
        expected: List of expected channel names.
        use_llm_judge: When True, run the Stage 2 LLM judge. Default False —
            pure programmatic evaluation, no upstream LLM call.

    Returns:
        Tuple of (predicted_channels, metadata_dict). ``predicted_channels``
        is what should be fed to :func:`compute_f1`. With the judge it is
        ``covered_expected + extra_recommended``, so precision and recall
        both reflect the judge's decision.
    """
    found, missing = programmatic_recall_check(response_text, expected)
    meta: dict[str, Any] = {"stage": 1, "found": found, "missing": missing}

    if not use_llm_judge:
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
    try:
        covered, extras = llm_judge_coverage(response_text, expected)
        predicted = covered + extras
        meta["evaluation"] = "llm_judge"
        meta["llm_covered"] = covered
        meta["llm_extras"] = extras
    except Exception as exc:
        # If the judge fails, fall back to Stage 1's literal found list.
        meta["evaluation"] = "llm_judge_error"
        meta["llm_error"] = str(exc)
        predicted = found

    return predicted, meta


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
