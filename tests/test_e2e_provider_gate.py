"""The provider an end-to-end run builds with, and the credential gate behind it.

Two environment variables and one registry table decide three things: which
provider a run builds its deployment repos with, whether a run that named none
is refused, and whether the lanes that build one can reach that provider at
all. The resolution lives in ``tests/e2e/provider.py`` and the gate in
``tests/conftest.py``; both are exercised here, in the fast lane, because
neither needs a credential to be wrong.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.models.provider_registry import PROVIDER_API_KEYS
from osprey.profiles.providers import load_provider_catalog
from tests.conftest import _e2e_provider_availability
from tests.e2e import conftest as e2e_conftest
from tests.e2e.provider import (
    E2E_PROVIDER_ENV,
    FORCE_PROVIDER_ENV,
    build_provider,
    e2e_provider,
    gateway_base_url,
)

#: The lanes that build a deployment repo and run an agent against it. They gate
#: on the provider the run named; every other e2e module pins a provider in its
#: own render and keeps the gateway-specific marker.
BUILD_AND_RUN_MODULES = (
    "e2e/test_claude_code_build_integration.py",
    "e2e/test_dispatch_tutorial.py",
    "e2e/test_dispatch_allowlist_parity.py",
    "e2e/test_dispatch_overlay_visibility.py",
)

#: A module that pins its own provider, so the gateway-specific marker is the
#: truth for it and the swap below must not have reached it.
PINNED_PROVIDER_MODULE = "e2e/test_preset_agentic.py"

PROVIDER_MARKER = "requires_e2e_provider"
GATEWAY_MARKER = "requires_als_apg"

_TESTS_ROOT = Path(__file__).resolve().parent


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


# ---------------------------------------------------------------------------
# gateway_base_url: the endpoint a lane calls, and what overrides it
# ---------------------------------------------------------------------------


def test_a_catalog_provider_takes_the_packaged_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The catalog is where a gateway's address is written, so a lane that
    exported no override gets whatever that entry carries — asserted against
    the catalog rather than a spelled host, which would pin an address no
    deployment renders."""
    monkeypatch.delenv("CBORG_BASE_URL", raising=False)
    assert (
        gateway_base_url("cborg", "CBORG_BASE_URL")
        == load_provider_catalog(None).entries["cborg"]["base_url"]
    )


def test_an_exported_override_beats_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner reaching the gateway somewhere else names that host, and the
    reader offers it instead of the catalog's own entry."""
    monkeypatch.setenv("CBORG_BASE_URL", "https://mirror.example.org/v1")
    assert gateway_base_url("cborg", "CBORG_BASE_URL") == "https://mirror.example.org/v1"


def test_a_blank_override_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace is not an address. The variable is read the way
    :func:`e2e_provider` reads its own, so the two agree on what a shell that
    exported an empty string said."""
    monkeypatch.setenv("CBORG_BASE_URL", "   ")
    assert (
        gateway_base_url("cborg", "CBORG_BASE_URL")
        == load_provider_catalog(None).entries["cborg"]["base_url"]
    )


def test_a_catalog_entry_that_defers_to_a_shell_names_no_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry whose ``base_url`` is an unexpanded reference states where the
    address comes from, not what it is. ``argo`` stands for that case: with
    nothing exported the lane has no route to offer, and with the variable set
    the same call returns what it holds."""
    monkeypatch.delenv("ARGO_BASE_URL", raising=False)
    assert gateway_base_url("argo", "ARGO_BASE_URL") is None
    monkeypatch.setenv("ARGO_BASE_URL", "https://gateway.example.org/v1")
    assert gateway_base_url("argo", "ARGO_BASE_URL") == "https://gateway.example.org/v1"


def test_an_override_replaces_the_address_an_entry_does_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``als-apg`` stands for the other case: its entry carries a host, and an
    exported variable is the runner saying it reaches that gateway elsewhere."""
    monkeypatch.delenv("ALS_APG_BASE_URL", raising=False)
    assert (
        gateway_base_url("als-apg", "ALS_APG_BASE_URL")
        == load_provider_catalog(None).entries["als-apg"]["base_url"]
    )
    monkeypatch.setenv("ALS_APG_BASE_URL", "https://gateway.example.org/v1")
    assert gateway_base_url("als-apg", "ALS_APG_BASE_URL") == "https://gateway.example.org/v1"


def test_a_provider_the_catalog_does_not_carry_has_no_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name no entry uses reads as endpointless rather than raising: the
    caller's answer is that there is no route, not that the catalog is wrong."""
    monkeypatch.delenv("NO_SUCH_GATEWAY_BASE_URL", raising=False)
    assert gateway_base_url("no-such-provider", "NO_SUCH_GATEWAY_BASE_URL") is None


# ---------------------------------------------------------------------------
# The credential gate: can the lanes reach the provider they were told to use?
# ---------------------------------------------------------------------------


def _a_provider_needing(key: bool) -> tuple[str, str | None]:
    """A provider from the registry's table that does (or does not) need a key."""
    for name, key_var in sorted(PROVIDER_API_KEYS.items()):
        if (key_var is not None) == key:
            return name, key_var
    raise AssertionError("PROVIDER_API_KEYS has no provider of that shape")


def test_a_provider_whose_key_is_exported_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, key_var = _a_provider_needing(key=True)
    monkeypatch.setenv(E2E_PROVIDER_ENV, provider)
    monkeypatch.setenv(key_var, "test-key")
    assert _e2e_provider_availability() == (True, "")


def test_a_provider_that_needs_no_key_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local server authenticates differently or not at all; the table says
    so with ``None``, and a gate that demanded a variable would skip a lane
    that would have run."""
    provider, _ = _a_provider_needing(key=False)
    monkeypatch.setenv(E2E_PROVIDER_ENV, provider)
    assert _e2e_provider_availability() == (True, "")


def test_a_missing_key_names_the_provider_and_its_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, key_var = _a_provider_needing(key=True)
    monkeypatch.setenv(E2E_PROVIDER_ENV, provider)
    monkeypatch.delenv(key_var, raising=False)
    available, reason = _e2e_provider_availability()
    assert not available
    assert provider in reason
    assert key_var in reason


def test_an_unknown_provider_is_a_reason_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(E2E_PROVIDER_ENV, "not-a-provider")
    available, reason = _e2e_provider_availability()
    assert not available
    assert "not-a-provider" in reason


def test_a_run_naming_no_provider_reads_as_unavailable() -> None:
    """The collection hook owns the refusal. The gate must not raise it a
    second time from inside a collection hook of its own — whichever hook runs
    first would then decide what the operator reads."""
    available, reason = _e2e_provider_availability()
    assert not available
    assert E2E_PROVIDER_ENV in reason


# ---------------------------------------------------------------------------
# Which modules carry which marker
# ---------------------------------------------------------------------------


def test_the_build_and_run_lanes_gate_on_the_named_provider() -> None:
    """The four lanes that build with whatever the run named must gate on that
    provider's credential. Gating them on one gateway's key is how a run that
    named another provider, and held its key, skipped anyway."""
    for relative in BUILD_AND_RUN_MODULES:
        source = (_TESTS_ROOT / relative).read_text(encoding="utf-8")
        assert PROVIDER_MARKER in source, f"tests/{relative} must gate on {PROVIDER_MARKER}"
        assert GATEWAY_MARKER not in source, (
            f"tests/{relative} builds with the provider the run named, so "
            f"{GATEWAY_MARKER} would gate it on a gateway it may never contact"
        )


def test_a_module_pinning_its_own_provider_keeps_the_gateway_marker() -> None:
    """The swap is scoped, not a retirement: for a module that passes one
    provider to every ``init_project`` call, the gateway-specific marker names
    exactly the credential it needs, and a neutral marker would make the gate
    lie."""
    source = (_TESTS_ROOT / PINNED_PROVIDER_MODULE).read_text(encoding="utf-8")
    assert GATEWAY_MARKER in source
