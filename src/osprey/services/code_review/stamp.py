"""What an approval prompt's review leaves for the executor.

The approval hook reviews a run's code before it puts the call to the operator, and writes what
it found into the ask stamp the executor reads before a readwrite run anyway. The executor then
leaves its own review out when the stamp names the very code it is about to launch, and hands
the agent what the reviewer said. A stamp naming other code, or none, leaves the executor's
review in place: it is the backstop for a run no prompt saw.

Only a verdict is stamped. A review that produced no verdict is not a review of the code, so the
executor runs its own.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from osprey.services.code_review.contract import ReviewVerdict
from osprey.services.code_review.review import ReviewOutcome, agent_message

#: The ask-stamp field holding the digest of the code the prompt's review read.
REVIEWED_DIGEST_KEY = "code_review_sha256"
#: The ask-stamp field naming the verdict, for a reader of the stamp.
REVIEWED_VERDICT_KEY = "code_review_verdict"
#: The ask-stamp field holding what the agent is told; absent for a pass with nothing to say.
REVIEWED_MESSAGE_KEY = "code_review_message"


def code_digest(code: str) -> str:
    """The digest that binds a stamped review to the code it read."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def reviewed_fields(code: str, outcome: ReviewOutcome) -> dict[str, Any]:
    """The stamp fields for a review of *code* that reached *outcome*.

    Args:
        code: The code the reviewer read, as the executor will read it.
        outcome: The review's outcome.

    Returns:
        The fields, or none for an outcome without a verdict.
    """
    if not isinstance(outcome.result, ReviewVerdict):
        return {}
    fields: dict[str, Any] = {
        REVIEWED_DIGEST_KEY: code_digest(code),
        REVIEWED_VERDICT_KEY: outcome.result.verdict,
    }
    message = agent_message(outcome)
    if message is not None:
        fields[REVIEWED_MESSAGE_KEY] = message
    return fields


def stamped_review(stamp: Mapping[str, Any], code: str) -> tuple[bool, str | None]:
    """Whether *stamp* records a review of *code*, and what the agent is told about it.

    Args:
        stamp: The ask stamp the approval hook left for the call.
        code: The code the executor is about to launch, as the gates read it.

    Returns:
        ``(True, message)`` when the stamp's digest is this code's; ``(False, None)`` otherwise.
    """
    if stamp.get(REVIEWED_DIGEST_KEY) != code_digest(code):
        return False, None
    message = stamp.get(REVIEWED_MESSAGE_KEY)
    return True, message if isinstance(message, str) and message else None
