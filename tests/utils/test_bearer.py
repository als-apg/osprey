"""The encode every bearer gate compares through.

The gates themselves are tested where they answer (tests/dispatch, and the
dispatch worker's API tests); what is pinned here is the one property they all
depend on and none of them can state alone: this encode is defined for every
``str``, so no credential a caller can present turns a refusal into a 500.
"""

from __future__ import annotations

import hmac

import pytest

from osprey.utils.bearer import credential_bytes


class TestTheEncodeIsTotal:
    """No text refuses to become bytes."""

    # One per way a str leaves ASCII, including the two surrogate halves
    # `os.environ` and a JSON decoder respectively produce.
    TEXTS = [
        "",
        "plain-ascii-token",
        "tökén",
        "secret-\udcff",
        "\ud800abc",
        "\U0001f511",
        "line\nbreak",
    ]

    @pytest.mark.parametrize("text", TEXTS)
    def test_every_text_encodes(self, text):
        assert isinstance(credential_bytes(text), bytes)

    @pytest.mark.parametrize("text", TEXTS)
    def test_the_comparison_it_feeds_never_raises(self, text):
        """The point of the encode: ``compare_digest`` refuses, never raises.

        Compared against a plain ASCII secret, which is what a deployment
        configures, so this is the shape every gate actually evaluates.
        """
        assert hmac.compare_digest(credential_bytes(text), credential_bytes("configured")) == (
            text == "configured"
        )


class TestTheSameTextGivesTheSameBytes:
    """A comparison is only as good as the encode being a function."""

    @pytest.mark.parametrize("text", ["tökén", "secret-\udcff", "\ud800abc"])
    def test_a_text_matches_itself(self, text):
        assert hmac.compare_digest(credential_bytes(text), credential_bytes(text))

    def test_texts_that_differ_do_not_match(self):
        assert not hmac.compare_digest(credential_bytes("töken"), credential_bytes("token"))
