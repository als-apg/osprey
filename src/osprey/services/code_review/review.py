"""One code review: run the reviewer, decide what its result means, say so.

:func:`review_code` runs one tool-less reviewer turn and turns whatever came of it into a
:class:`ReviewOutcome`. It never reports a failure by raising, because the level decides what a
failure does. The rest is pure wording and fields derived from an outcome. This core writes no
records and sends no notices; each surface's adapter does, from what these functions return.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from osprey.agent_record.contract import (
    RECORD_OFF_ENV,
    RECORD_OFF_VALUE,
    SPOOL_DIR_ENV,
    TELEMETRY_SWITCH_ENV,
    reaches_a_record,
)
from osprey.agent_runner.errors import AgentCliUnavailableError
from osprey.agent_runner.primitives import SDKWorkflowResult
from osprey.agent_runner.runner import run_query
from osprey.agent_runner.sdk_context import make_tool_allowlist
from osprey.agent_runner.tool_names import READ_ONLY_DENIED_BUILTINS
from osprey.agent_runner.verdict import _BUDGET_SUBTYPES, EXIT_PASS, evaluate_verdict
from osprey.services.code_review.contract import (
    ReviewFailure,
    ReviewRequest,
    ReviewVerdict,
    parse_verdict,
    render_request,
)
from osprey.services.code_review.settings import (
    REVIEW_STATE_DIR,
    ReviewConfigError,
    ReviewSettings,
    resolve_prompt,
)

#: The code goes ahead.
ACTION_PROCEED = "proceed"
#: The code is stopped.
ACTION_REFUSE = "refuse"

#: The ``reason`` of every review's audit record.
AUDIT_REASON = "code_review"

#: The most one review may spend, in USD.
_REVIEW_BUDGET_USD = 1.0

#: Set over the reviewer's environment. A reviewer is tooling, not the agent's work, so it must not
#: land in the session's record or telemetry. :func:`_reviewer_env` adds whatever else in the
#: environment reaches a record, under these.
REVIEWER_ENV_OVERRIDES: Mapping[str, str] = MappingProxyType(
    {
        "CLAUDE_CODE_SESSION_ID": "",
        "OSPREY_SESSION_ID": "",
        "OSPREY_TELEMETRY_SESSION_ID": "",
        "OSPREY_TELEMETRY_SESSION_START": "",
        "OSPREY_POSTURE_SESSION": "",
        "OSPREY_TERMINAL_SESSION": "",
        "OTEL_RESOURCE_ATTRIBUTES": "",
        SPOOL_DIR_ENV: "",
        TELEMETRY_SWITCH_ENV: "0",
        RECORD_OFF_ENV: RECORD_OFF_VALUE,
    }
)


@dataclass(frozen=True)
class ReviewOutcome:
    """What a review came to.

    Attributes:
        action: ``ACTION_PROCEED`` or ``ACTION_REFUSE``.
        level: The level it was decided under; ``None`` when the level could not be read.
        result: The reviewer's verdict, or the failure that stands in for one.
    """

    action: str
    level: str | None
    result: ReviewVerdict | ReviewFailure


class AuditFields(NamedTuple):
    """The value-free fields of one review's audit record.

    Attributes:
        decision: ``"refused"`` or ``"allowed"``.
        reason: Always :data:`AUDIT_REASON`.
        subject: The request's audit subject.
        detail: The result in ``key=value`` form; never reasons or code.
    """

    decision: str
    reason: str
    subject: str
    detail: str


def decide(result: ReviewVerdict | ReviewFailure, level: str | None) -> ReviewOutcome:
    """Apply the level to a review's result.

    Only ``flag`` lets a block verdict or a failure go ahead, so a failure whose level could not
    be read (``None``) refuses.

    Args:
        result: The verdict, or the failure that stands in for one.
        level: ``"flag"``, ``"block"``, or ``None`` when the level could not be read.

    Returns:
        The outcome carrying *result* and *level*.
    """
    stopped = level != "flag" and (isinstance(result, ReviewFailure) or result.verdict == "block")
    return ReviewOutcome(ACTION_REFUSE if stopped else ACTION_PROCEED, level, result)


def _verdict_message(outcome: ReviewOutcome, verdict: ReviewVerdict) -> str | None:
    """Word a verdict for the agent; ``None`` when it has nothing to say."""
    reasons = [reason for reason in verdict.reasons if reason.strip()]
    revision = verdict.suggested_revision
    has_revision = revision is not None and bool(revision.strip())
    if verdict.verdict == "pass" and not reasons and not has_revision:
        return None

    if outcome.action == ACTION_REFUSE:
        opening = "The code review blocked this code: revise and resubmit."
    elif verdict.verdict == "block":
        opening = (
            "The code review found a blocking problem. This deployment only flags findings, "
            "so the code proceeded."
        )
    elif verdict.verdict == "concern":
        opening = "The code review raised a concern. The code proceeded."
    else:
        opening = "The code review passed this code, with a note."

    sections = [opening]
    if reasons:
        sections.append("Reasons:\n" + "\n".join(f"- {reason}" for reason in reasons))
    if has_revision:
        sections.append(f"Suggested revision:\n{revision}")
    return "\n\n".join(sections)


def agent_message(outcome: ReviewOutcome) -> str | None:
    """Word an outcome for the agent, in the one form both surfaces use.

    Args:
        outcome: The review's outcome.

    Returns:
        The message, or ``None`` for a pass with nothing to say.
    """
    result = outcome.result
    if isinstance(result, ReviewVerdict):
        return _verdict_message(outcome, result)
    if outcome.action == ACTION_REFUSE:
        return (
            f"The code review failed (kind {result.kind}), so the code was refused: "
            f"{result.message}\n\nChanging the code will not help: report it and do not retry."
        )
    return (
        f"The code review did not run (kind {result.kind}): {result.message}\n\n"
        "The code proceeded unreviewed."
    )


def _result_name(outcome: ReviewOutcome) -> str:
    """The word for a review's result: the verdict, or ``"failure"``."""
    return outcome.result.verdict if isinstance(outcome.result, ReviewVerdict) else "failure"


def notice_text(outcome: ReviewOutcome, request: ReviewRequest) -> str | None:
    """Word an outcome for the operator's activity feed.

    Reasons and code stay out: this names what happened and where, not what was found.

    Args:
        outcome: The review's outcome.
        request: The request it answers.

    Returns:
        The notice, or ``None`` for a pass, which sends none.
    """
    result = _result_name(outcome)
    if result == "pass":
        return None
    if isinstance(outcome.result, ReviewFailure):
        result = f"{result} ({outcome.result.kind})"
    taken = "refused" if outcome.action == ACTION_REFUSE else "proceeded"
    return f"Code review on {request.surface}: {result}, {taken}: {request.subject}"


def audit_fields(outcome: ReviewOutcome, request: ReviewRequest) -> AuditFields:
    """The audit record's fields for an outcome.

    Args:
        outcome: The review's outcome.
        request: The request it answers.

    Returns:
        The fields, holding no reasons, no code and no failure message.
    """
    parts = [f"review={_result_name(outcome)}"]
    if outcome.level is not None:
        parts.append(f"level={outcome.level}")
    if isinstance(outcome.result, ReviewFailure):
        parts.append(f"kind={outcome.result.kind}")
    if request.mode is not None:
        parts.append(f"mode={request.mode}")
    return AuditFields(
        decision="refused" if outcome.action == ACTION_REFUSE else "allowed",
        reason=AUDIT_REASON,
        subject=request.subject,
        detail=" ".join(parts),
    )


def _reviewer_env() -> dict[str, str]:
    """The reviewer's environment overrides, from the environment as it is now."""
    blanked = {key: "" for key in os.environ if reaches_a_record(key)}
    return blanked | dict(REVIEWER_ENV_OVERRIDES)


def _read_run(run: SDKWorkflowResult) -> ReviewVerdict | ReviewFailure:
    """Read a finished run: a capped or failed run has no answer to parse.

    The reply is the assistant's text as written. ``combined_text`` would lower-case it, and mix
    in tool results, which would corrupt a suggested revision.
    """
    message = run.result
    if message is None:
        return ReviewFailure("runner", "the reviewer's run ended without a result")
    if message.subtype in _BUDGET_SUBTYPES:
        return ReviewFailure("budget", f"the reviewer's run hit a cap: {message.subtype}")
    if evaluate_verdict(run, set()) != EXIT_PASS:
        return ReviewFailure("runner", f"the reviewer's run failed: {message.subtype}")
    return parse_verdict("".join(run.text_blocks))


async def _run_review(
    request: ReviewRequest,
    settings: ReviewSettings,
    repo_root: Path,
    render_dir: Path,
) -> ReviewVerdict | ReviewFailure:
    """Run the reviewer once, in a directory of its own, and read what it said."""
    work_dir: str | None = None
    try:
        system_prompt = resolve_prompt(settings, repo_root)
        state_dir = Path(repo_root) / REVIEW_STATE_DIR
        state_dir.mkdir(parents=True, exist_ok=True)
        work_dir = tempfile.mkdtemp(dir=state_dir)
        # One tool-less, capped, unrecorded turn. ``env_overrides``, never ``env``: ``env`` would
        # skip the provider routing, and the deployment's main model through its provider is the
        # reviewer's model. ``bypassPermissions`` would bypass the callback, so there is no
        # permission mode.
        run = await asyncio.wait_for(
            run_query(
                render_dir,
                render_request(request),
                system_prompt=system_prompt,
                tools=[],
                permission_mode=None,
                can_use_tool=make_tool_allowlist(()),
                disallowed_tools=READ_ONLY_DENIED_BUILTINS,
                setting_sources=[],
                mcp_servers={},
                await_mcp_servers=(),
                max_turns=1,
                max_budget_usd=_REVIEW_BUDGET_USD,
                model=None,
                cwd=Path(work_dir),
                env_overrides=_reviewer_env(),
            ),
            settings.time_limit,
        )
        return _read_run(run)
    except ReviewConfigError as exc:
        return ReviewFailure("prompt", str(exc))
    except TimeoutError:
        return ReviewFailure("timeout", f"no reply within {settings.time_limit:g} seconds")
    except (ImportError, AgentCliUnavailableError) as exc:
        return ReviewFailure("unavailable", str(exc))
    except Exception as exc:
        return ReviewFailure("runner", f"{type(exc).__name__}: {exc}")
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


async def review_code(
    request: ReviewRequest,
    settings: ReviewSettings,
    *,
    repo_root: Path,
    render_dir: Path,
) -> ReviewOutcome:
    """Review one piece of code under *settings*.

    Each review runs in a fresh temporary directory, so nothing planted in the tree or left by an
    earlier review reaches it. Only a ``BaseException`` propagates.

    Args:
        request: The code to review.
        settings: The validated settings, whose level decides what the result does.
        repo_root: Anchor of the prompt path and of the state directory.
        render_dir: The rendered project the reviewer's provider and launcher resolve from.

    Returns:
        The outcome, with every failure as its kind.
    """
    result = await _run_review(request, settings, repo_root, render_dir)
    return decide(result, settings.level)
