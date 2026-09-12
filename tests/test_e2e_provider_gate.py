"""The provider an end-to-end run builds with, and the credential gate behind it.

Two environment variables and one registry table decide three things: which
provider a run builds its deployment repos with, whether a run that named none
is refused, and whether the lanes that build one can reach that provider at
all. The resolution lives in ``tests/e2e/provider.py`` and the gate in
``tests/conftest.py``; both are exercised here, in the fast lane, because
neither needs a credential to be wrong.
"""

from __future__ import annotations

import pytest

from tests.e2e.provider import E2E_PROVIDER_ENV, FORCE_PROVIDER_ENV, build_provider, e2e_provider


@pytest.fixture(autouse=True)
def _no_ambient_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every case from a shell that names no provider.

    A developer running the fast lane with either variable exported would
    otherwise see cases pass or fail on their environment rather than on what
    each case sets.
    """
    monkeypatch.delenv(E2E_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(FORCE_PROVIDER_ENV, raising=False)


# ---------------------------------------------------------------------------
# build_provider: what a call site pinned, and what overrides it
# ---------------------------------------------------------------------------


def test_build_provider_keeps_what_the_call_site_pinned() -> None:
    assert build_provider("cborg") == "cborg"


def test_build_provider_takes_the_suite_wide_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The benchmark matrix points the whole suite at one provider per cell
    without editing a fixture; the pinned value is what it overrides."""
    monkeypatch.setenv(FORCE_PROVIDER_ENV, "anthropic")
    assert build_provider("cborg") == "anthropic"


def test_an_empty_override_is_unset_for_both_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shell that exports the override empty has named no provider with it.
    Both functions of this module read the same variable, so they have to read
    it the same way — otherwise one call site builds with "" and the other with
    what it pinned."""
    monkeypatch.setenv(FORCE_PROVIDER_ENV, "   ")
    monkeypatch.setenv(E2E_PROVIDER_ENV, "cborg")
    assert build_provider("ds4") == "ds4"
    assert e2e_provider() == "cborg"


# ---------------------------------------------------------------------------
# e2e_provider: which provider the build-and-run lanes were told to use
# ---------------------------------------------------------------------------


def test_e2e_provider_reads_the_selection_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(E2E_PROVIDER_ENV, "cborg")
    assert e2e_provider() == "cborg"
