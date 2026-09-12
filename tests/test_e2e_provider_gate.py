"""The provider an end-to-end run builds with, and the credential gate behind it.

Two environment variables and one registry table decide three things: which
provider a run builds its deployment repos with, whether a run that named none
is refused, and whether the lanes that build one can reach that provider at
all. The resolution lives in ``tests/e2e/provider.py`` and the gate in
``tests/conftest.py``; both are exercised here, in the fast lane, because
neither needs a credential to be wrong.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.e2e import conftest as e2e_conftest
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


def test_e2e_provider_accepts_the_override_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The benchmark runner requires the override and sets nothing else, so the
    override has to satisfy the selection on its own."""
    monkeypatch.setenv(FORCE_PROVIDER_ENV, "ds4")
    assert e2e_provider() == "ds4"


def test_the_override_wins_over_the_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(E2E_PROVIDER_ENV, "cborg")
    monkeypatch.setenv(FORCE_PROVIDER_ENV, "ds4")
    assert e2e_provider() == "ds4"


def test_a_run_that_names_no_provider_is_refused() -> None:
    """No constant stands behind the variables: a run that named nothing is
    told what to set rather than sent to whichever gateway was compiled in."""
    from osprey.models.provider_registry import PROVIDER_API_KEYS

    with pytest.raises(RuntimeError) as excinfo:
        e2e_provider()
    message = str(excinfo.value)
    assert E2E_PROVIDER_ENV in message
    assert FORCE_PROVIDER_ENV in message
    assert any(name in message for name in PROVIDER_API_KEYS), (
        f"the refusal must list the providers it would accept: {message}"
    )


class _StubConfig:
    """Just enough pytest config for ``pytest_configure`` to register markers."""

    def __init__(self, *, collect_only: bool = False) -> None:
        self.markers: list[str] = []
        self.option = SimpleNamespace(markexpr="")
        self._collect_only = collect_only

    def getoption(self, name: str, default: object = None) -> object:
        return self._collect_only if name == "collectonly" else default

    def addinivalue_line(self, name: str, line: str) -> None:
        self.markers.append(line)


def test_the_e2e_session_is_refused_before_its_workers_spawn() -> None:
    """The refusal has to reach the operator when the lanes run distributed.
    Collection then happens inside an xdist worker, where a ``UsageError`` is
    reported as an internal error with a traceback; raised from
    ``pytest_configure`` of an initial-argument conftest it is raised once, in
    the controlling process, before a worker exists."""
    with pytest.raises(pytest.UsageError) as excinfo:
        e2e_conftest.pytest_configure(_StubConfig())
    assert E2E_PROVIDER_ENV in str(excinfo.value)


def test_enumerating_the_suite_needs_no_provider() -> None:
    """``--collect-only`` builds nothing and reaches no gateway, so it is not
    refused: the benchmark lane gate reads every test's lane marker out of a
    real collection of ``tests/e2e/`` and would have no manifest otherwise."""
    config = _StubConfig(collect_only=True)
    e2e_conftest.pytest_configure(config)
    e2e_conftest.pytest_collection_modifyitems(config, [])
    assert any(line.startswith("e2e:") for line in config.markers)


def test_a_named_provider_lets_configure_register_its_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is the only thing added to that hook: a run that named a
    provider still gets the markers the e2e lanes select on."""
    monkeypatch.setenv(E2E_PROVIDER_ENV, "cborg")
    config = _StubConfig()
    e2e_conftest.pytest_configure(config)
    assert any(line.startswith("e2e:") for line in config.markers)
