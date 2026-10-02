"""A web verb's flags win; only a flag left out is filled in from the config."""

import pytest

import osprey.registry.web as web
from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    WebServerConfigDepthError,
    resolve_web_server_address,
    resolve_web_server_bind,
)

CONFIG = {"ariel": {"web": {"host": "192.0.2.10", "port": 18300}}}


@pytest.fixture(autouse=True)
def _no_port_env(monkeypatch):
    for definition in FRAMEWORK_WEB_SERVERS.values():
        monkeypatch.delenv(definition.port_env_var, raising=False)


@pytest.fixture
def resolver_calls(monkeypatch):
    calls: list = []
    original = web.resolve_web_server_address

    def _recording(key, config=None):
        calls.append((key, config))
        return original(key, config)

    monkeypatch.setattr(web, "resolve_web_server_address", _recording)
    return calls


def test_both_flags_are_returned_without_resolving(resolver_calls):
    assert resolve_web_server_bind("ariel", CONFIG, host="127.0.0.1", port=18999) == (
        "127.0.0.1",
        18999,
    )
    assert resolver_calls == []


def test_missing_host_is_filled_from_the_config(resolver_calls):
    assert resolve_web_server_bind("ariel", CONFIG, host=None, port=18999) == (
        "192.0.2.10",
        18999,
    )
    assert len(resolver_calls) == 1
    assert resolver_calls[0][1] is CONFIG


def test_missing_port_is_filled_from_the_config(resolver_calls):
    assert resolve_web_server_bind("ariel", CONFIG, host="127.0.0.1", port=None) == (
        "127.0.0.1",
        18300,
    )
    assert len(resolver_calls) == 1


def test_neither_flag_is_the_resolved_address(resolver_calls):
    expected = resolve_web_server_address("ariel", CONFIG)
    assert resolve_web_server_bind("ariel", CONFIG, host=None, port=None) == expected
    assert len(resolver_calls) == 1


@pytest.mark.parametrize(
    ("host", "port", "expected", "resolves"),
    [
        ("", 18999, ("", 18999), False),
        ("127.0.0.1", 0, ("127.0.0.1", 0), False),
        ("", 0, ("", 0), False),
        ("", None, ("", 18300), True),
        (None, 0, ("192.0.2.10", 0), True),
    ],
)
def test_zero_and_empty_count_as_given(resolver_calls, host, port, expected, resolves):
    assert resolve_web_server_bind("ariel", CONFIG, host=host, port=port) == expected
    assert len(resolver_calls) == (1 if resolves else 0)


def test_both_flags_do_not_read_a_config_the_resolver_refuses():
    refused = {"ariel": {"port": 1}}
    assert resolve_web_server_bind("ariel", refused, host="127.0.0.1", port=18999) == (
        "127.0.0.1",
        18999,
    )
    with pytest.raises(WebServerConfigDepthError):
        resolve_web_server_bind("ariel", refused, host="127.0.0.1", port=None)


def test_unknown_key_raises_with_both_flags():
    with pytest.raises(KeyError):
        resolve_web_server_bind("no-such-server", {}, host="127.0.0.1", port=18999)


def test_none_config_loads_on_demand(monkeypatch):
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda *a, **k: {"artifact_server": {"host": "192.0.2.5", "port": 18500}},
    )
    assert resolve_web_server_bind("artifact", None, host=None, port=None) == (
        "192.0.2.5",
        18500,
    )
