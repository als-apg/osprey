"""What the approval prompt's review leaves in the ask stamp, and how the executor reads it back."""

from __future__ import annotations

import hashlib

import pytest

from osprey.services.code_review.contract import ReviewFailure, ReviewVerdict
from osprey.services.code_review.review import agent_message, decide
from osprey.services.code_review.stamp import (
    REVIEWED_DIGEST_KEY,
    REVIEWED_MESSAGE_KEY,
    REVIEWED_VERDICT_KEY,
    code_digest,
    reviewed_fields,
    stamped_review,
)

CODE = "print('reviewed')\n"
CONCERN = ReviewVerdict("concern", ("the stub worries",))
BLOCK = ReviewVerdict("block", ("the stub objects",), "print('revised')\n")


def test_the_digest_is_sha256_of_the_utf8_text():
    assert code_digest(CODE) == hashlib.sha256(CODE.encode("utf-8")).hexdigest()
    assert code_digest("print('é')") != code_digest("print('e')")


@pytest.mark.parametrize(
    ("verdict", "level"),
    [(CONCERN, "block"), (CONCERN, "flag"), (BLOCK, "flag")],
    ids=["concern-block", "concern-flag", "block-flag"],
)
def test_a_verdict_with_something_to_say_is_stamped_whole(verdict, level):
    outcome = decide(verdict, level)

    fields = reviewed_fields(CODE, outcome)

    assert fields == {
        REVIEWED_DIGEST_KEY: code_digest(CODE),
        REVIEWED_VERDICT_KEY: verdict.verdict,
        REVIEWED_MESSAGE_KEY: agent_message(outcome),
    }
    assert stamped_review(fields, CODE) == (True, agent_message(outcome))


def test_a_silent_pass_is_stamped_without_a_message():
    fields = reviewed_fields(CODE, decide(ReviewVerdict("pass", ()), "block"))

    assert fields == {REVIEWED_DIGEST_KEY: code_digest(CODE), REVIEWED_VERDICT_KEY: "pass"}
    assert stamped_review(fields, CODE) == (True, None)


@pytest.mark.parametrize("level", ["block", "flag", None])
def test_a_failure_is_no_review_of_the_code(level):
    assert reviewed_fields(CODE, decide(ReviewFailure("timeout", "no reply"), level)) == {}


def test_other_code_is_not_the_reviewed_code():
    fields = reviewed_fields(CODE, decide(CONCERN, "block"))

    assert stamped_review(fields, CODE + "\n") == (False, None)
    assert stamped_review({}, CODE) == (False, None)
    assert stamped_review({REVIEWED_DIGEST_KEY: 7}, CODE) == (False, None)


def test_a_message_that_is_not_text_is_dropped():
    fields = {REVIEWED_DIGEST_KEY: code_digest(CODE), REVIEWED_MESSAGE_KEY: ["not", "text"]}
    assert stamped_review(fields, CODE) == (True, None)
    assert stamped_review(
        {REVIEWED_DIGEST_KEY: code_digest(CODE), REVIEWED_MESSAGE_KEY: ""}, CODE
    ) == (
        True,
        None,
    )
