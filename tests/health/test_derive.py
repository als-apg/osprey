"""Unit tests for :mod:`osprey.health.derive`.

Pin the frozen contract of :func:`derive_mcp_servers`: the enable gate, the
container-aware ``url_key`` resolution (with an explicit override always
winning), the ``expect_tools`` union, defensive skipping of malformed server
entries, and the always-returned (possibly empty) ``mcp_servers`` category.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.health.config import (
    DEFAULT_PROBE_TIMEOUTS,
    DEFAULT_SUITE_TIMEOUT_S,
    AutoMcpSettings,
    CategoryRecord,
    Cost,
    HealthSettings,
)
from osprey.health.derive import derive_mcp_servers
from osprey.health.models import Status

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _host_environment(monkeypatch):
    """Pin the default environment to "host" so container detection is
    deterministic regardless of where the suite runs. Container-focused tests
    re-monkeypatch on top of this."""
    monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _settings(
    *,
    enabled: bool = True,
    url_key: str = "host_url",
    url_key_explicit: bool = False,
    suite_timeout_s: float = DEFAULT_SUITE_TIMEOUT_S,
) -> HealthSettings:
    return HealthSettings(
        suite_timeout_s=suite_timeout_s,
        interval_s=max(60.0, 2 * suite_timeout_s),
        on_demand_timeout_s=None,
        auto=AutoMcpSettings(enabled=enabled, url_key=url_key, url_key_explicit=url_key_explicit),
    )


def _expanded(servers: dict[str, Any]) -> dict[str, Any]:
    return {"claude_code": {"servers": servers}}


def _server(
    *,
    url: str = "http://internal/mcp",
    port: int = 9000,
    host_url: str | None = "http://localhost:9000/mcp",
    docker_url: str | None = "http://matlab:9000/mcp",
    permissions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    network: dict[str, Any] = {"port": port}
    if host_url is not None:
        network["host_url"] = host_url
    if docker_url is not None:
        network["docker_url"] = docker_url
    entry: dict[str, Any] = {"transport": "http", "url": url, "network": network}
    if permissions is not None:
        entry["permissions"] = permissions
    return entry


def _only_check(record: CategoryRecord | None):
    assert record is not None
    assert record.checks is not None
    assert len(record.checks) == 1
    return record.checks[0]


# --------------------------------------------------------------------------- #
# Enable gate
# --------------------------------------------------------------------------- #


def test_disabled_returns_none():
    settings = _settings(enabled=False)
    assert derive_mcp_servers(settings, _expanded({"matlab": _server()})) is None


def test_enabled_returns_record():
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server()}))
    assert isinstance(record, CategoryRecord)
    assert record.name == "mcp_servers"
    assert record.cost is Cost.POLL


# --------------------------------------------------------------------------- #
# Category shape and timeout budget
# --------------------------------------------------------------------------- #


def test_category_budget_matches_suite_timeout():
    # Poll-cost category budget resolves to the configured suite timeout, exactly
    # as core_record does for a poll callable (resolve_callable_timeout_s).
    record = derive_mcp_servers(_settings(suite_timeout_s=42.0), _expanded({"matlab": _server()}))
    assert record is not None
    assert record.timeout_s == 42.0


def test_check_is_well_formed_mcp_spec():
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert check.name == "matlab"
    assert check.type == "mcp"
    assert check.timeout_s == DEFAULT_PROBE_TIMEOUTS["mcp"] == 10.0
    assert check.timeout_status is Status.ERROR
    assert check.requires == ()
    assert check.params["url"] == "http://localhost:9000/mcp"


# --------------------------------------------------------------------------- #
# url_key resolution — one test per network position
# --------------------------------------------------------------------------- #


def test_host_url_chosen_on_host(monkeypatch):
    monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: False)
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert check.params["url"] == "http://localhost:9000/mcp"


def test_docker_url_chosen_via_dockerenv(monkeypatch):
    monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: p == "/.dockerenv")
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert check.params["url"] == "http://matlab:9000/mcp"


def test_docker_url_chosen_via_env_flag(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    monkeypatch.setenv("OSPREY_IN_CONTAINER", "1")
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert check.params["url"] == "http://matlab:9000/mcp"


def test_explicit_url_key_overrides_host(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    settings = _settings(url_key="docker_url", url_key_explicit=True)
    record = derive_mcp_servers(settings, _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert check.params["url"] == "http://matlab:9000/mcp"


def test_explicit_host_url_wins_even_when_containerized(monkeypatch):
    # Explicit operator override is honored as written, container or not.
    monkeypatch.setenv("OSPREY_IN_CONTAINER", "1")
    settings = _settings(url_key="host_url", url_key_explicit=True)
    record = derive_mcp_servers(settings, _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert check.params["url"] == "http://localhost:9000/mcp"


# --------------------------------------------------------------------------- #
# expect_tools
# --------------------------------------------------------------------------- #


def test_expect_tools_union_of_allow_and_ask():
    perms = {"allow": ["read_pv", "list_pvs"], "ask": ["write_pv"]}
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server(permissions=perms)}))
    check = _only_check(record)
    assert check.params["expect_tools"] == ["read_pv", "list_pvs", "write_pv"]


def test_expect_tools_deduplicates_preserving_order():
    perms = {"allow": ["a", "b"], "ask": ["b", "c"]}
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server(permissions=perms)}))
    check = _only_check(record)
    assert check.params["expect_tools"] == ["a", "b", "c"]


def test_permissions_absent_no_expect_tools():
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server()}))
    check = _only_check(record)
    assert "expect_tools" not in check.params


def test_empty_allow_and_ask_no_expect_tools():
    perms = {"allow": [], "ask": []}
    record = derive_mcp_servers(_settings(), _expanded({"matlab": _server(permissions=perms)}))
    check = _only_check(record)
    assert "expect_tools" not in check.params


def test_permissions_non_mapping_treated_as_absent_but_check_emitted():
    entry = _server()
    entry["permissions"] = ["not", "a", "mapping"]
    record = derive_mcp_servers(_settings(), _expanded({"matlab": entry}))
    check = _only_check(record)
    assert "expect_tools" not in check.params
    assert check.params["url"] == "http://localhost:9000/mcp"


# --------------------------------------------------------------------------- #
# Defensive skipping of malformed entries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "servers",
    [
        pytest.param({"bad": "not-a-mapping"}, id="non_mapping_entry"),
        pytest.param({"bad": _server(url="")}, id="empty_url"),
        pytest.param(
            {"bad": {"transport": "http", "network": {"host_url": "http://x/mcp"}}},
            id="missing_url",
        ),
        pytest.param({"bad": {"transport": "http", "url": "http://x/mcp"}}, id="missing_network"),
        pytest.param(
            {"bad": {"transport": "http", "url": "http://x/mcp", "network": "nope"}},
            id="non_mapping_network",
        ),
        pytest.param({"bad": _server(host_url=None)}, id="missing_network_url_key"),
        pytest.param({"bad": _server(host_url=12345)}, id="non_string_network_url_key"),
    ],
)
def test_malformed_entries_skipped_silently(monkeypatch, servers):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    record = derive_mcp_servers(_settings(), _expanded(servers))
    assert record is not None
    assert record.checks == []


def test_malformed_entry_does_not_drop_good_neighbor(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    servers = {"broken": "not-a-mapping", "good": _server()}
    record = derive_mcp_servers(_settings(), _expanded(servers))
    check = _only_check(record)
    assert check.name == "good"


# --------------------------------------------------------------------------- #
# Absent / empty config → zero-check category (not None)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expanded",
    [
        pytest.param(None, id="expanded_none"),
        pytest.param({}, id="empty_mapping"),
        pytest.param({"claude_code": {}}, id="no_servers"),
        pytest.param({"claude_code": {"servers": {}}}, id="empty_servers"),
        pytest.param({"claude_code": "nope"}, id="non_mapping_claude_code"),
        pytest.param({"claude_code": {"servers": "nope"}}, id="non_mapping_servers"),
    ],
)
def test_absent_or_empty_config_yields_zero_checks(expanded):
    record = derive_mcp_servers(_settings(), expanded)
    assert record is not None
    assert record.name == "mcp_servers"
    assert record.cost is Cost.POLL
    assert record.checks == []


# --------------------------------------------------------------------------- #
# Multiple servers
# --------------------------------------------------------------------------- #


def test_multiple_qualifying_entries_one_check_each(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    servers = {
        "matlab": _server(host_url="http://localhost:9000/mcp"),
        "epics": _server(host_url="http://localhost:9100/mcp"),
    }
    record = derive_mcp_servers(_settings(), _expanded(servers))
    assert record is not None
    assert record.checks is not None
    names = [c.name for c in record.checks]
    assert names == ["matlab", "epics"]
    urls = [c.params["url"] for c in record.checks]
    assert urls == ["http://localhost:9000/mcp", "http://localhost:9100/mcp"]
