"""The provider an end-to-end run builds with, and the credential gate behind it.

Two environment variables and one registry table decide three things: which
provider a run builds its deployment repos with, whether a run that named none
is refused, and whether the lanes that build one can reach that provider at
all. A run that names none is refused only when it selects an e2e test that
does not carry the ``model_free`` marker, and the refusal names those tests.
The resolution lives in ``tests/e2e/provider.py``, the refusal in
``tests/e2e/conftest.py`` and the credential gate in ``tests/conftest.py``; all
three are exercised here, in the fast lane, because none needs a credential to
be wrong. The subprocess cases collect ``tests/e2e/`` modules under
``--setup-plan``, which runs the refusal and no fixture or test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from osprey.models.provider_registry import PROVIDER_API_KEYS
from osprey.profiles.providers import load_provider_catalog
from tests.conftest import _e2e_provider_availability
from tests.e2e import conftest as e2e_conftest
from tests.e2e import sdk_helpers
from tests.e2e.provider import (
    E2E_MODEL,
    E2E_PROVIDER_ENV,
    FORCE_PROVIDER_ENV,
    MODEL_FREE_MARKER,
    REFUSAL_LISTED_TESTS,
    build_model,
    build_provider,
    e2e_provider,
    gateway_base_url,
    provider_refusal,
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
_REPO_ROOT = _TESTS_ROOT.parent


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
# build_model: the model a lane builds with
# ---------------------------------------------------------------------------


def test_build_model_keeps_what_the_call_site_pinned() -> None:
    assert build_model("claude-opus-5") == "claude-opus-5"


def test_a_call_site_that_names_no_model_builds_with_haiku() -> None:
    assert build_model(None) == "claude-haiku-4-5-20251001"


def test_the_provider_ci_names_serves_the_lane_model() -> None:
    """A catalog or workflow change that strands the lanes on a model their
    provider does not serve fails here rather than at the gateway."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    provider = workflow["env"]["OSPREY_E2E_PROVIDER"]
    assert E2E_MODEL in load_provider_catalog(None).entries[provider]["models"]


class _InitReached(Exception):
    """Raised by the stand-in for ``osprey init`` so the build never runs."""


@pytest.mark.parametrize(
    ("pinned", "expected"),
    [(None, E2E_MODEL), ("claude-opus-5", "claude-opus-5")],
)
def test_init_project_writes_the_lane_model_into_the_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pinned: str | None, expected: str
) -> None:
    calls: list[list[str]] = []

    def _record(verb: str, args: list[str], *, timeout: int) -> None:  # noqa: ARG001 - stands in for _run_osprey, whose callers name timeout
        calls.append([verb, *args])
        raise _InitReached

    monkeypatch.setattr(sdk_helpers, "_run_osprey", _record)
    with pytest.raises(_InitReached):
        sdk_helpers.init_project(tmp_path, "proj", provider="als-apg", model=pinned)

    argv = calls[0]
    model_sets = [
        argv[i + 1]
        for i in range(len(argv) - 1)
        if argv[i] == "--set" and argv[i + 1].startswith("model=")
    ]
    assert model_sets == [f"model={expected}"]


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


class _StubHook:
    """Records what the refusal reports as deselected."""

    def __init__(self) -> None:
        self.deselected: list[object] = []

    def pytest_deselected(self, *, items: list[object]) -> None:
        self.deselected.extend(items)


class _StubConfig:
    """Just enough pytest config for the e2e conftest's configure and collection hooks.

    ``workerinput``/``workeroutput`` exist only when given, the way xdist sets
    them only on a worker's config.
    """

    def __init__(
        self,
        *,
        collect_only: bool = False,
        workerinput: dict | None = None,
    ) -> None:
        self.markers: list[str] = []
        self.option = SimpleNamespace(markexpr="")
        self.hook = _StubHook()
        self._collect_only = collect_only
        if workerinput is not None:
            self.workerinput = workerinput
            self.workeroutput: dict = {}

    def getoption(self, name: str, default: object = None) -> object:
        return self._collect_only if name == "collectonly" else default

    def addinivalue_line(self, _name: str, line: str) -> None:
        self.markers.append(line)


class _StubItem:
    """A collected test: its node id, its file, and whether it declared ``model_free``."""

    def __init__(self, nodeid: str, path: Path, *, marked: bool) -> None:
        self.nodeid = nodeid
        self.path = path
        self._marked = marked

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self._marked and name == MODEL_FREE_MARKER else None


_E2E_DIR = _TESTS_ROOT / "e2e"


def _e2e_item(name: str, *, marked: bool) -> _StubItem:
    return _StubItem(f"tests/e2e/test_x.py::{name}", _E2E_DIR / "test_x.py", marked=marked)


def test_configure_registers_the_markers_without_a_provider() -> None:
    """Configuring the session decides nothing about the provider: the
    refusal needs the final selection, which only collection has. A run that
    names none still gets every marker the lanes select on, ``model_free``
    among them."""
    config = _StubConfig()
    e2e_conftest.pytest_configure(config)
    assert any(line.startswith("e2e:") for line in config.markers)
    assert any(line.startswith(f"{MODEL_FREE_MARKER}:") for line in config.markers)


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
    """A run that named a provider still gets the markers the e2e lanes select on."""
    monkeypatch.setenv(E2E_PROVIDER_ENV, "cborg")
    config = _StubConfig()
    e2e_conftest.pytest_configure(config)
    assert any(line.startswith("e2e:") for line in config.markers)


# ---------------------------------------------------------------------------
# The refusal: a run naming no provider may select only model-free tests
# ---------------------------------------------------------------------------


def test_a_selection_of_model_free_tests_needs_no_provider() -> None:
    items = [_e2e_item("test_a", marked=True), _e2e_item("test_b", marked=True)]
    before = list(items)
    e2e_conftest.pytest_collection_modifyitems(_StubConfig(), items)
    assert items == before


def test_an_unmarked_test_is_refused_by_name() -> None:
    item = _e2e_item("test_live", marked=False)
    with pytest.raises(pytest.UsageError) as excinfo:
        e2e_conftest.pytest_collection_modifyitems(_StubConfig(), [item])
    message = str(excinfo.value)
    assert item.nodeid in message
    assert E2E_PROVIDER_ENV in message
    assert FORCE_PROVIDER_ENV in message
    assert f"-m {MODEL_FREE_MARKER}" in message


def test_only_unmarked_tests_are_named() -> None:
    free = _e2e_item("test_free", marked=True)
    live = _e2e_item("test_live", marked=False)
    with pytest.raises(pytest.UsageError) as excinfo:
        e2e_conftest.pytest_collection_modifyitems(_StubConfig(), [free, live])
    message = str(excinfo.value)
    assert live.nodeid in message
    assert free.nodeid not in message


def test_items_outside_the_e2e_directory_are_not_refused() -> None:
    item = _StubItem("tests/cli/test_x.py::test_a", _TESTS_ROOT / "cli" / "test_x.py", marked=False)
    items = [item]
    e2e_conftest.pytest_collection_modifyitems(_StubConfig(), items)
    assert items == [item]


def test_a_named_provider_lifts_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(E2E_PROVIDER_ENV, "cborg")
    items = [_e2e_item("test_live", marked=False)]
    e2e_conftest.pytest_collection_modifyitems(_StubConfig(), items)
    assert len(items) == 1


def test_the_refusal_names_twenty_tests_and_counts_the_rest() -> None:
    """A whole-directory run leaves hundreds of tests unmarked; listing every
    one would bury the remedy, so the refusal names the first few and counts
    the rest."""
    assert REFUSAL_LISTED_TESTS == 20
    items = [_e2e_item(f"test_{i:02d}", marked=False) for i in range(25)]
    with pytest.raises(pytest.UsageError) as excinfo:
        e2e_conftest.pytest_collection_modifyitems(_StubConfig(), items)
    message = str(excinfo.value)
    assert all(item.nodeid in message for item in items[:20])
    assert not any(item.nodeid in message for item in items[20:])
    assert "and 5 more" in message


def test_provider_refusal_without_tests_is_unchanged() -> None:
    """``e2e_provider()`` raises the bare text; naming no tests must not change it."""
    from osprey.models.provider_registry import PROVIDER_API_KEYS as keys

    known = ", ".join(sorted(keys))
    assert provider_refusal() == (
        f"This end-to-end run names no provider. Set {E2E_PROVIDER_ENV} to the provider "
        f"whose credential this environment holds, or {FORCE_PROVIDER_ENV} to point the "
        f"whole suite at one provider. Known providers: {known}."
    )


def test_a_worker_hands_the_refusal_to_the_controller() -> None:
    """Under xdist a ``UsageError`` raised in a worker reaches the operator as
    an internal error. A worker whose controller relays the refusal selects
    nothing and hands the message over instead."""
    config = _StubConfig(workerinput={e2e_conftest._RELAY_KEY: True})
    items = [_e2e_item("test_live", marked=False), _e2e_item("test_free", marked=True)]
    selected = list(items)
    e2e_conftest.pytest_collection_modifyitems(config, items)
    assert items == []
    assert config.hook.deselected == selected
    assert "test_x.py::test_live" in config.workeroutput[e2e_conftest._REFUSAL_KEY]


def test_a_worker_without_a_relay_refuses_itself() -> None:
    """A controller that never loaded this conftest cannot print the refusal;
    deselecting silently would pass a run that ran nothing, so the worker raises."""
    config = _StubConfig(workerinput={})
    with pytest.raises(pytest.UsageError):
        e2e_conftest.pytest_collection_modifyitems(config, [_e2e_item("test_live", marked=False)])


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    """Run pytest from the repo root in a shell that names no provider."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in (E2E_PROVIDER_ENV, FORCE_PROVIDER_ENV)
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


_TELEMETRY = "tests/e2e/test_openobserve_telemetry.py"
_LIVE_TELEMETRY = f"{_TELEMETRY}::test_live_agent_metric_lands"
_SYNTHETIC_TELEMETRY = (
    "test_synthetic_otlp_roundtrip_via_computed_header",
    "test_bad_credentials_are_rejected",
    "test_the_deploy_provisions_a_distinct_ingest_identity",
    "test_synthetic_otlp_roundtrip_via_the_ingest_identity",
    "test_synthetic_trace_roundtrip_via_the_ingest_identity",
    "test_a_wrong_token_for_the_ingest_account_is_rejected",
    "test_the_rendered_config_names_the_ingest_identity",
)
_DISTRIBUTION = pytest.mark.parametrize("dist", [(), ("-n", "2")], ids=["single", "xdist"])


@_DISTRIBUTION
def test_a_model_free_module_is_not_refused(dist: tuple[str, ...]) -> None:
    result = _run_pytest("tests/e2e/test_sdk_helpers.py", "--setup-plan", *dist)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "names no provider" not in output


@_DISTRIBUTION
def test_a_selected_model_driven_test_is_refused_by_name(dist: tuple[str, ...]) -> None:
    result = _run_pytest(_TELEMETRY, "--setup-plan", *dist)
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert _LIVE_TELEMETRY in output
    assert f"-m {MODEL_FREE_MARKER}" in output
    assert "INTERNALERROR" not in output
    assert not any(name in output for name in _SYNTHETIC_TELEMETRY), output


def test_deselecting_the_model_driven_test_lifts_the_refusal() -> None:
    result = _run_pytest(_TELEMETRY, "--deselect", _LIVE_TELEMETRY, "--setup-plan", "-n", "2")
    assert result.returncode == 0, result.stdout + result.stderr


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


def test_the_build_and_run_lanes_name_the_lane_model() -> None:
    """The four lanes build with whatever provider the run named, so each one
    names its model too; otherwise it runs the provider's catalog default, which
    the suite's budgets are not sized for."""
    for relative in BUILD_AND_RUN_MODULES:
        source = (_TESTS_ROOT / relative).read_text(encoding="utf-8")
        assert "model={E2E_MODEL}" in source or "model={build_model(model)}" in source, (
            f"tests/{relative} must build with the lane model"
        )


def test_a_module_pinning_its_own_provider_keeps_the_gateway_marker() -> None:
    """The swap is scoped, not a retirement: for a module that passes one
    provider to every ``init_project`` call, the gateway-specific marker names
    exactly the credential it needs, and a neutral marker would make the gate
    lie."""
    source = (_TESTS_ROOT / PINNED_PROVIDER_MODULE).read_text(encoding="utf-8")
    assert GATEWAY_MARKER in source
