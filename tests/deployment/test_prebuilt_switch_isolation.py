"""The prebuilt-images switch is absent from the environment of every deploy test.

``conftest.py`` clears ``OSPREY_PREBUILT_IMAGES`` for every test in this
directory. The tests here are what makes that clear observable rather than
merely present: they export the switch at module scope — which pytest sets up
before the function-scoped autouse fixtures of a conftest — so each assertion
below is about what a test actually sees on a shell that exported it.
"""

from __future__ import annotations

import os

import pytest

from osprey.deployment import container_lifecycle


@pytest.fixture(scope="module", autouse=True)
def the_switch_is_exported_for_this_module():
    """Put ``OSPREY_PREBUILT_IMAGES`` in the environment the way an operator does.

    Written straight to ``os.environ`` at module scope rather than through
    ``monkeypatch``: the switch has to be there before the suite-wide clear
    runs, and a clear with nothing to remove proves nothing. Whatever was
    there before is restored when the module is done, so no other module sees
    the export.
    """
    before = os.environ.get("OSPREY_PREBUILT_IMAGES")
    os.environ["OSPREY_PREBUILT_IMAGES"] = "1"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("OSPREY_PREBUILT_IMAGES", None)
        else:
            os.environ["OSPREY_PREBUILT_IMAGES"] = before


def test_a_deployment_test_sees_no_prebuilt_switch_in_its_environment():
    """What the shell exported is not what a deploy test asserts against.

    A truthy switch takes the standalone build out of every dev-mode start, so
    a suite that inherited it would report the missing build as a failure of
    the code rather than of the environment it ran in.
    """
    assert "OSPREY_PREBUILT_IMAGES" not in os.environ
    assert container_lifecycle._resolve_prebuilt_images({}) is False


def test_a_test_that_sets_the_switch_itself_still_gets_the_answer_it_asked_for(monkeypatch):
    """The suite-wide clear is a starting point, not a ceiling.

    The tests that pin what the switch does set it in their own bodies, which
    run after every autouse fixture and are undone after the test.
    """
    monkeypatch.setenv("OSPREY_PREBUILT_IMAGES", "1")

    assert container_lifecycle._resolve_prebuilt_images({}) is True
