"""What a code review is asked and what it answers.

``render_request`` builds the reviewer's user message and ``parse_verdict`` reads its reply. The
reply shape is stated in one place, :data:`_REPLY_SHAPE`, so a facility's review prompt never has
to restate it. The prompt text itself belongs to the facility and is not part of this message.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

#: The ``surface`` of a review of code about to execute.
SURFACE_EXECUTE = "execute"
#: The ``surface`` of a review of a notebook cell edit.
SURFACE_NOTEBOOK = "notebook"

#: Every ``ReviewFailure.kind``.
FAILURE_KINDS = ("config", "prompt", "timeout", "budget", "unparseable", "unavailable", "runner")

#: The verdicts a reviewer may reach, in rising severity.
_VERDICTS = ("pass", "concern", "block")

#: The verdicts that mean nothing without a stated reason.
_VERDICTS_NEEDING_REASONS = ("concern", "block")

#: The closing section of every reviewer message.
_REPLY_SHAPE = """\
## Required reply

End your reply with one JSON object of exactly this shape:

{"verdict": "pass", "reasons": ["why"], "suggested_revision": "the corrected code"}

- `verdict` is "pass", "concern" or "block".
- `reasons` is a list of strings. "concern" and "block" need at least one non-empty reason; "pass" \
may give an empty list.
- `suggested_revision` is optional: a string, or left out when you have none.

Only the last JSON object in your reply is read, so write no other JSON object after it."""

_BACKTICK_RUN = re.compile(r"`+")


@dataclass(frozen=True)
class ReviewRequest:
    """One piece of code to review, on either surface.

    Attributes:
        surface: ``SURFACE_EXECUTE`` or ``SURFACE_NOTEBOOK``.
        code: For execute, the gated (operator-readable) code; for notebook, the new cell source.
        subject: Audit subject: the tool subject, or the notebook path under agent data.
        description: Execute only: the description given with the code.
        mode: Execute only: ``"readonly"`` or ``"readwrite"``.
        script_args: ``execute_file`` only: the script's arguments.
        notebook: Notebook only: code cells on disk, cell-delimited, outputs dropped.
        cell: Notebook only: the target cell (id, index, edit mode).
    """

    surface: str
    code: str
    subject: str
    description: str | None = None
    mode: str | None = None
    script_args: tuple[str, ...] = ()
    notebook: str | None = None
    cell: str | None = None


@dataclass(frozen=True)
class ReviewVerdict:
    """A reply that met the reply shape.

    Attributes:
        verdict: ``"pass"``, ``"concern"`` or ``"block"``.
        reasons: The reviewer's reasons; empty only on a pass.
        suggested_revision: Replacement code the reviewer offers, if any.
    """

    verdict: str
    reasons: tuple[str, ...]
    suggested_revision: str | None = None


@dataclass(frozen=True)
class ReviewFailure:
    """A review that produced no verdict.

    Attributes:
        kind: One of :data:`FAILURE_KINDS`.
        message: What went wrong.
    """

    kind: str
    message: str


def _unparseable(defect: str) -> ReviewFailure:
    return ReviewFailure("unparseable", defect)


def _last_top_level_object(text: str) -> dict | ReviewFailure:
    """Return the last JSON object in *text* that no other JSON value encloses.

    Only a value that decodes is stepped over, so a stray brace in prose hides nothing. A brace or
    bracket whose value runs off the end of the text means the reply was cut off: whatever came
    before it, or nested in it, is not the reviewer's answer, so that is a failure rather than a
    fall back to an earlier object.

    Args:
        text: The reply text, fenced or bare.

    Returns:
        The object, or the failure naming why there is none.
    """
    decoder = json.JSONDecoder()
    last: dict | None = None
    index = 0
    while (index := next((i for i in range(index, len(text)) if text[i] in "{["), -1)) != -1:
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            # An unterminated string reports where it starts, not where the text ran out.
            if exc.pos >= len(text) or exc.msg.startswith("Unterminated string"):
                return _unparseable("the reply ends inside an unterminated JSON value")
            index += 1
        except RecursionError:
            return _unparseable("the reply nests JSON too deeply to read")
        else:
            if isinstance(value, dict):
                last = value
    if last is None:
        return _unparseable("the reply holds no JSON object")
    return last


def parse_verdict(text: str) -> ReviewVerdict | ReviewFailure:
    """Read the reviewer's reply against the reply shape.

    The last top-level JSON object in *text* is the answer, whether fenced or bare, and it is the
    only one judged: a defective last object does not fall back to an earlier one.

    Args:
        text: The reviewer's reply.

    Returns:
        The verdict, or ``ReviewFailure("unparseable", ...)`` naming the defect.
    """
    if not text.strip():
        return _unparseable("the reply is empty")

    found = _last_top_level_object(text)
    if isinstance(found, ReviewFailure):
        return found

    verdict = found.get("verdict")
    if not isinstance(verdict, str) or verdict not in _VERDICTS:
        return _unparseable(f"verdict must be one of {_VERDICTS}, got {verdict!r}")

    reasons = found.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        return _unparseable(f"reasons must be a list of strings, got {reasons!r}")
    if verdict in _VERDICTS_NEEDING_REASONS and not any(reason.strip() for reason in reasons):
        return _unparseable(f"a {verdict} verdict needs at least one non-empty reason")

    revision = found.get("suggested_revision")
    if revision is not None and not isinstance(revision, str):
        return _unparseable(f"suggested_revision must be a string or absent, got {revision!r}")

    return ReviewVerdict(verdict, tuple(reasons), revision)


def fenced(content: str) -> str:
    """Fence *content* with a fence longer than any backtick run in it.

    The content is the code under review, so it must not be able to close its own block and go on
    as if it were the rest of the message.
    """
    longest = max((len(run) for run in _BACKTICK_RUN.findall(content)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{content}\n{fence}"


def render_request(request: ReviewRequest) -> str:
    """Build the reviewer's user message for *request*.

    Lays out the surface, the code, and whichever of description, mode, script arguments, notebook
    and target cell are present, then states the required reply shape. ``subject`` is for the
    audit record and stays out of the message.

    Args:
        request: The code to review.

    Returns:
        The message text, ending with the reply shape.
    """
    sections = [f"## Surface\n\n{request.surface}"]
    if request.description:
        sections.append(f"## Description\n\n{request.description}")
    if request.mode:
        sections.append(f"## Mode\n\n{request.mode}")
    if request.script_args:
        sections.append(f"## Script arguments\n\n{json.dumps(list(request.script_args))}")
    if request.notebook:
        sections.append(f"## Notebook\n\n{fenced(request.notebook)}")
    if request.cell:
        sections.append(f"## Target cell\n\n{request.cell}")
    sections.append(f"## Code\n\n{fenced(request.code)}")
    sections.append(_REPLY_SHAPE)
    return "\n\n".join(sections)
