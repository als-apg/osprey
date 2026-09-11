"""``osprey build`` naming a half-redirected feedback destination.

Feedback has ONE destination — whoever owns the deployment — reached by two
independent settings. Moving one and leaving the other pointed upstream sends
half a facility's reports to the OSPREY maintainers, who cannot answer them.
The symptom is invisible from the deployment itself, so the build says it.

Advisory only: a deployment that genuinely wants mail locally and issues
upstream is buildable, it is just asked to mean it.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile_reach import feedback_owner_advisories
from osprey.interfaces.web_terminal.feedback_destination import (
    DEFAULT_FEEDBACK_EMAIL,
    DEFAULT_FEEDBACK_GITHUB_REPO,
)

FACILITY_EMAIL = "controls@example.org"
FACILITY_REPO = "facility/ops"


def _config(**feedback: object) -> dict:
    """A rendered config carrying only the `web.feedback` block under test."""
    return {"web": {"feedback": feedback}}


class TestHalfMovedDestination:
    def test_a_moved_email_with_an_upstream_repo_is_named(self):
        advisories = feedback_owner_advisories(
            _config(email=FACILITY_EMAIL, github_repo=DEFAULT_FEEDBACK_GITHUB_REPO)
        )
        assert len(advisories) == 1
        assert "web.feedback.email" in advisories[0]
        assert "web.feedback.github_repo" in advisories[0]
        assert "web.feedback.owner" in advisories[0]

    def test_a_moved_email_with_an_unspelled_repo_is_named(self):
        """Absent is not silence: the tracker default still aims upstream."""
        advisories = feedback_owner_advisories(_config(email=FACILITY_EMAIL))
        assert len(advisories) == 1
        assert DEFAULT_FEEDBACK_GITHUB_REPO in advisories[0]

    def test_a_moved_repo_beside_the_shipped_email_is_silent(self):
        """The shipped mail default is blank, so it is a retired channel.

        There is no "upstream email" half to leak: a deployment that names its
        own tracker and says nothing about mail offers no mail channel, which
        is a coherent posture rather than a half-finished edit.
        """
        assert (
            feedback_owner_advisories(
                _config(email=DEFAULT_FEEDBACK_EMAIL, github_repo=FACILITY_REPO)
            )
            == []
        )

    def test_a_moved_repo_beside_an_unspelled_email_is_silent(self):
        """Absent reads as the shipped default, and the shipped default is blank."""
        assert feedback_owner_advisories(_config(github_repo=FACILITY_REPO)) == []

    def test_moving_both_is_silent(self):
        assert (
            feedback_owner_advisories(_config(email=FACILITY_EMAIL, github_repo=FACILITY_REPO))
            == []
        )

    def test_moving_neither_is_silent(self):
        """The shipped deployment is owned by the OSPREY project, coherently."""
        assert (
            feedback_owner_advisories(
                _config(email=DEFAULT_FEEDBACK_EMAIL, github_repo=DEFAULT_FEEDBACK_GITHUB_REPO)
            )
            == []
        )

    def test_a_config_with_no_feedback_block_is_silent(self):
        assert feedback_owner_advisories({}) == []
        assert feedback_owner_advisories({"web": {}}) == []


class TestRetiredChannelsAreNotHalfMoves:
    """Blank is a deliberate posture, not a half-finished edit."""

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_retired_email_beside_an_upstream_repo_is_silent(self, blank):
        assert (
            feedback_owner_advisories(
                _config(email=blank, github_repo=DEFAULT_FEEDBACK_GITHUB_REPO)
            )
            == []
        )

    def test_a_retired_repo_beside_a_moved_email_is_silent(self):
        """A retired channel delivers nothing, so it cannot be the leaking half."""
        assert feedback_owner_advisories(_config(email=FACILITY_EMAIL, github_repo="")) == []

    def test_a_retired_repo_beside_the_shipped_email_is_silent(self):
        """Neither channel reaches anyone; that is the air-gapped posture."""
        assert (
            feedback_owner_advisories(_config(email=DEFAULT_FEEDBACK_EMAIL, github_repo="")) == []
        )

    def test_both_retired_is_silent(self):
        """An air-gapped deployment offers no outbound channel at all."""
        assert feedback_owner_advisories(_config(email="", github_repo="")) == []


class TestOwnerBlock:
    """Naming an owner is the coherent way to redirect; it is never a half-move."""

    def test_an_owner_email_silences_the_advisory(self):
        assert feedback_owner_advisories(_config(owner={"email": FACILITY_EMAIL})) == []

    def test_an_owner_tracker_silences_the_advisory(self):
        config = _config(
            owner={"tracker": {"kind": "gitlab", "target": "https://git.example.org/x"}}
        )
        assert feedback_owner_advisories(config) == []

    def test_an_owner_naming_only_itself_does_not_silence_a_half_move(self):
        """A `name:` with no destination redirects nothing."""
        advisories = feedback_owner_advisories(
            _config(owner={"name": "ALS Controls"}, email=FACILITY_EMAIL)
        )
        assert len(advisories) == 1

    @pytest.mark.parametrize("bad", ["nonsense", 3, []])
    def test_an_unusable_owner_block_does_not_silence_a_half_move(self, bad):
        advisories = feedback_owner_advisories(_config(owner=bad, email=FACILITY_EMAIL))
        assert len(advisories) == 1
