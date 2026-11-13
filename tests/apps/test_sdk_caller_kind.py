"""``osprey.apps.sdk.caller_kind``: who the terminal proxy says is calling."""

from __future__ import annotations

import pytest

from osprey.apps.manifest import launch_token_env as derived_token_env
from osprey.apps.sdk import (
    CALLER_HEADER,
    LAUNCH_TOKEN_HEADER,
    caller_from_header,
    caller_kind,
    launch_token_env,
)

APP = "caller-test"
TOKEN_ENV = derived_token_env(APP)
TOKEN = "minted-by-the-deployment"


@pytest.fixture(autouse=True)
def _launch_token(monkeypatch):
    monkeypatch.setenv("OSPREY_APP_NAME", APP)
    monkeypatch.setenv(TOKEN_ENV, TOKEN)


def _proxied(**extra: str) -> dict[str, str]:
    """The headers a request forwarded by the proxy carries: the launch token and *extra*."""
    return {LAUNCH_TOKEN_HEADER: TOKEN, **extra}


@pytest.mark.parametrize("kind", ["agent", "panel"])
def test_the_proxy_stamped_kind_is_read(kind):
    assert caller_kind(_proxied(**{CALLER_HEADER: kind})) == kind


def test_the_header_names_are_matched_case_insensitively():
    headers = {"x-launch-token": TOKEN, "x-osprey-caller": "agent"}
    assert caller_kind(headers) == "agent"


def test_an_absent_header_raises_naming_it():
    with pytest.raises(ValueError, match=CALLER_HEADER):
        caller_kind(_proxied())


@pytest.mark.parametrize("value", ["operator", "Agent", "", "agent "])
def test_an_unknown_value_raises_naming_the_header(value):
    with pytest.raises(ValueError, match=CALLER_HEADER):
        caller_kind(_proxied(**{CALLER_HEADER: value}))


def test_a_forged_header_on_a_direct_call_without_the_credential_is_refused():
    with pytest.raises(ValueError, match=LAUNCH_TOKEN_HEADER):
        caller_kind({CALLER_HEADER: "agent"})


def test_a_forged_header_with_a_wrong_credential_is_refused():
    with pytest.raises(ValueError, match=LAUNCH_TOKEN_HEADER):
        caller_kind({LAUNCH_TOKEN_HEADER: "guessed", CALLER_HEADER: "agent"})


def test_a_backend_with_no_launch_token_configured_trusts_no_caller(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV)
    with pytest.raises(ValueError, match=TOKEN_ENV):
        caller_kind(_proxied(**{CALLER_HEADER: "panel"}))


def test_a_backend_started_without_its_application_name_trusts_no_caller(monkeypatch):
    monkeypatch.delenv("OSPREY_APP_NAME")
    with pytest.raises(ValueError, match="OSPREY_APP_NAME"):
        caller_kind(_proxied(**{CALLER_HEADER: "panel"}))


def test_launch_token_env_is_the_variable_the_build_derives_for_the_application():
    assert launch_token_env() == TOKEN_ENV == "CALLER_TEST_LAUNCH_TOKEN"


def test_launch_token_env_without_an_application_name_raises(monkeypatch):
    monkeypatch.delenv("OSPREY_APP_NAME")
    with pytest.raises(ValueError, match="OSPREY_APP_NAME"):
        launch_token_env()


@pytest.mark.parametrize(("value", "kind"), [("agent", "agent"), ("panel", "panel")])
def test_caller_from_header_reads_the_two_values(value, kind):
    assert caller_from_header(value) == kind


@pytest.mark.parametrize("value", [None, "", "browser", "AGENT"])
def test_caller_from_header_refuses_anything_else(value):
    assert caller_from_header(value) is None
