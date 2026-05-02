"""Channel finder benchmark evaluation.

Stage 1 (always-on) — Programmatic recall check: case-insensitive substring
matching to determine which expected PVs appear anywhere in the agent's
response text. Returns ``(found, meta)`` where ``found`` is the subset of
expected channels detected in the text. Cheap, deterministic, no API call.

Stage 2 (opt-in via ``use_llm_judge=True``) — LLM precision judge: uses a
small LLM (Haiku via LiteLLM) with structured output to extract the agent's
*final* recommended channels, distinguishing them from channels merely
mentioned during exploration. Requires an Anthropic-compatible API key
(``ANTHROPIC_API_KEY``, ``CBORG_API_KEY``, or ``ANTHROPIC_AUTH_TOKEN``).

The opt-in default keeps single-paradigm benchmark runs free of upstream
LLM-judge cost; cross-paradigm research that wants tighter precision
scoring opts in explicitly.

Public API:
    programmatic_recall_check  — stage 1 only
    llm_extract_channels       — stage 2 only
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
# Stage 2 — LLM precision judge
# ---------------------------------------------------------------------------


class ChannelExtractionResult(BaseModel):
    """Structured output for LLM channel extraction."""

    recommended_channels: list[str]
    reasoning: str


def llm_extract_channels(response_text: str, expected: list[str]) -> list[str]:
    """Use an LLM judge to extract the agent's final recommended channels.

    Calls Haiku via OSPREY's LiteLLM adapter with structured output to
    distinguish channels in the final answer from those mentioned during
    exploration.

    Args:
        response_text: Full agent response text.
        expected: Expected channel names (used as calibration context for
            the LLM so it knows what channel names look like).

    Returns:
        List of channel names the LLM identified as the agent's final
        recommendation. Empty list if structured output parsing fails.
    """
    from osprey.models.providers.litellm_adapter import (
        execute_litellm_completion,
    )

    expected_json = json.dumps(expected, indent=2)
    prompt = (
        "You are evaluating a control system channel finder agent's response.\n"
        "Extract the list of channels that the agent presents as its FINAL\n"
        "recommendation/answer.\n\n"
        "Important distinctions:\n"
        "- Channels mentioned only during exploration or reasoning do NOT count\n"
        "- If the agent lists a broad set then narrows down, only the final "
        "narrowed set counts\n"
        "- Channel names look like PV addresses "
        "(e.g., SR:MAG:DIPOLE:B05:CURRENT:SP)\n"
        "  or descriptive names (e.g., StorageRing_BeamCurrent_ReadBack)\n\n"
        f"Expected channels (for reference — use these to calibrate what "
        f"channel names look like):\n{expected_json}\n\n"
        f"Agent's full response:\n{response_text}"
    )

    # Resolve provider from available API keys: CBORG, Anthropic, or
    # whatever ANTHROPIC_AUTH_TOKEN was injected by the provider resolver.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    provider = "anthropic"
    model_id = "claude-haiku-4-5-20251001"
    base_url = None

    if not api_key:
        # Check for CBORG or proxy auth (injected by sdk_env/inject_provider_env)
        cborg_key = os.environ.get("CBORG_API_KEY")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if cborg_key or auth_token:
            provider = "cborg"
            api_key = cborg_key or auth_token
            model_id = "anthropic/claude-haiku"
            base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.cborg.lbl.gov/v1")

    result = execute_litellm_completion(
        provider=provider,
        message=prompt,
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        max_tokens=1024,
        temperature=0.0,
        output_format=ChannelExtractionResult,
    )

    if isinstance(result, ChannelExtractionResult):
        return result.recommended_channels
    # Fallback: if structured output didn't parse, return empty
    return []


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

    Stage 1 (always): Programmatic recall check — are all expected channels
    present in the response text? If any are missing, return immediately
    with ``predicted = found`` (precision contract: found ⊆ expected, so
    precision = 1.0 when measured against expected).

    Stage 2 (opt-in): LLM precision judge — extract the agent's FINAL
    recommended channel list from the response for tighter precision
    scoring. Only runs when ``use_llm_judge=True`` AND Stage 1 found all
    expected channels.

    Args:
        response_text: Full agent response text (plain string).
        expected: List of expected channel names.
        use_llm_judge: When True, run the Stage 2 LLM precision judge after
            Stage 1 succeeds. Default False — pure programmatic evaluation,
            no upstream LLM call.

    Returns:
        Tuple of (predicted_channels, metadata_dict).
    """
    found, missing = programmatic_recall_check(response_text, expected)

    meta: dict[str, Any] = {"stage": 1, "found": found, "missing": missing}

    if missing:
        # Stage 1 failure: some expected channels not in text at all.
        # predicted = found channels (precision=1.0 against expected by construction).
        meta["evaluation"] = "programmatic_recall_fail"
        return found, meta

    if not use_llm_judge:
        # Stage 1 succeeded; caller didn't opt in to Stage 2.
        meta["evaluation"] = "programmatic_recall_only"
        return found, meta

    # Stage 2: all expected channels appear in text and caller opted in —
    # use the LLM judge to extract the agent's final recommended list for
    # precision scoring.
    meta["stage"] = 2
    try:
        predicted = llm_extract_channels(response_text, expected)
        meta["evaluation"] = "llm_judge"
        meta["llm_extracted"] = predicted
    except Exception as exc:
        # If LLM judge fails, fall back to found channels.
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
