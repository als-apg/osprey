"""Unit tests for the Teams relay's Azure Functions shim.

``function_app.py`` is the public half of the relay: it reads the bearer token,
asks :mod:`validation` whether the activity is genuine, and turns the answer
into one of three status codes Teams treats differently. That mapping, and the
promise that the body reaches the queue exactly as Microsoft sent it, is what
these tests pin -- the judging itself belongs to ``test_relay_validation.py``.

Two things about the shim shape the fixtures here. It ships as a service
template rather than an importable package, so it is loaded by path the way its
sibling suite loads ``validation.py``. And it imports ``azure.functions``, which
is not installed anywhere in this repository: the Functions host provides it in
production, so a stand-in is placed in :data:`sys.modules` for exactly as long
as the load takes and removed again. Nothing of the stand-in outlives the load
-- the shim keeps its own reference to the module object -- which matters
because the sibling adapter suite proves things about ``azure`` imports failing,
and a stub left in the cache would quietly answer imports it is not for.

Every test drives the endpoint through a substitute ``validate_activity``. The
real one is unreachable from here by design: the relay is meant to hand a
verdict straight to a status code without adding any judgement of its own, and a
test that had to mint a JWT to reach the 200 path would be testing the validator
again.
"""

from __future__ import annotations

import enum
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_RELAY_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "templates"
    / "services"
    / "teams_bridge"
    / "relay"
)
_FUNCTION_APP_PATH = _RELAY_DIR / "function_app.py"

MODULE_NAME = "teams_relay_function_app"
"""The name the shim is executed under.

Deliberately not ``function_app``, and deliberately not the
``teams_relay_validation`` its sibling suite uses: both suites can be collected
into one session, and two modules sharing a cache key would hand whichever ran
second the other's globals.
"""

APP_ID = "11111111-2222-3333-4444-555555555555"

RAW_BODY = (
    b'{\n  "serviceUrl" : "https://smba.trafficmanager.net/amer/",\n'
    b'    "type":"message",   "id"  :  "activity-1",\n'
    b'  "text": "how is the beam?"\n}\n'
)
"""An activity whose whitespace and key order no JSON serialiser would produce.

The relay promises the bridge receives the bytes Microsoft signed rather than a
re-serialisation of them, and a canonical body cannot tell the two apart.
"""


# --- the Azure Functions stand-in -------------------------------------------


class AuthLevel(enum.Enum):
    """The trigger's authentication level, compared by identity as the real one is."""

    ANONYMOUS = "anonymous"
    FUNCTION = "function"
    ADMIN = "admin"


class Out:
    """An output binding, recording every value written through it."""

    def __init__(self) -> None:
        self.values: list[Any] = []

    def set(self, value: Any) -> None:
        self.values.append(value)

    def get(self) -> Any:
        return self.values[-1] if self.values else None

    def __class_getitem__(cls, item: Any) -> type[Out]:
        return cls


class _Headers:
    """Request headers, matched case-insensitively as HTTP headers are."""

    def __init__(self, headers: Mapping[str, str] | None) -> None:
        self._items = {name.lower(): value for name, value in (headers or {}).items()}

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._items.get(name.lower(), default)

    def __getitem__(self, name: str) -> str:
        return self._items[name.lower()]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items


class HttpRequest:
    """The little of an ``azure.functions.HttpRequest`` the shim reads."""

    def __init__(
        self,
        *,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
        method: str = "POST",
        url: str = "https://relay.example/api/messages",
    ) -> None:
        self._body = body
        self.headers = _Headers(headers)
        self.method = method
        self.url = url

    def get_body(self) -> bytes:
        return self._body


class HttpResponse:
    """The little of an ``azure.functions.HttpResponse`` the assertions read."""

    def __init__(self, body: Any = None, *, status_code: int = 200, **_kwargs: Any) -> None:
        self.body = body
        self.status_code = status_code


class FunctionApp:
    """A registrar that records what each decorator was given and changes nothing.

    Returning the undecorated function is what lets the tests call ``messages``
    as a plain function; recording the keyword arguments is what lets them
    assert on bindings that, in production, only the host ever reads.
    """

    def __init__(self) -> None:
        self.registrations: list[tuple[str, dict[str, Any]]] = []

    def _record(self, kind: str, kwargs: dict[str, Any]) -> Callable[[Any], Any]:
        def decorate(function: Any) -> Any:
            self.registrations.append((kind, kwargs))
            return function

        return decorate

    def function_name(self, **kwargs: Any) -> Callable[[Any], Any]:
        return self._record("function_name", kwargs)

    def route(self, **kwargs: Any) -> Callable[[Any], Any]:
        return self._record("route", kwargs)

    def service_bus_queue_output(self, **kwargs: Any) -> Callable[[Any], Any]:
        return self._record("service_bus_queue_output", kwargs)

    def kwargs_for(self, kind: str) -> dict[str, Any]:
        """The keyword arguments of the one registration of ``kind``."""
        matches = [kwargs for name, kwargs in self.registrations if name == kind]
        assert len(matches) == 1, f"expected exactly one {kind} registration, got {len(matches)}"
        return matches[0]


def _azure_functions_stub() -> tuple[ModuleType, ModuleType]:
    """The ``azure`` package and ``azure.functions`` module the shim imports."""
    functions = ModuleType("azure.functions")
    functions.__dict__.update(
        {
            "AuthLevel": AuthLevel,
            "FunctionApp": FunctionApp,
            "HttpRequest": HttpRequest,
            "HttpResponse": HttpResponse,
            "Out": Out,
        }
    )
    package = ModuleType("azure")
    package.__dict__["functions"] = functions
    return package, functions


def _load_relay() -> ModuleType:
    """Execute ``function_app.py`` against the stand-in and leave no trace of it.

    The shim puts its own directory on :data:`sys.path` so that its ``from
    validation import ...`` resolves the sibling file. Both that path entry and
    the ``validation`` module it produces are undone here: they are the shim's
    arrangements for a Functions host, not this session's, and a ``validation``
    left in the cache is a name common enough to collide with somebody else's.
    """
    package, functions = _azure_functions_stub()
    borrowed = ("azure", "azure.functions", "validation", MODULE_NAME)
    saved = {name: sys.modules.get(name) for name in borrowed}
    saved_path = list(sys.path)
    try:
        sys.modules["azure"] = package
        sys.modules["azure.functions"] = functions
        sys.modules.pop("validation", None)

        spec = importlib.util.spec_from_file_location(MODULE_NAME, _FUNCTION_APP_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        # Held open so the classes the shim imported from it keep a live module
        # behind them once the cache entry goes.
        module.__dict__["_test_validation_module"] = sys.modules.get("validation")
        return module
    finally:
        sys.path[:] = saved_path
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def relay() -> ModuleType:
    """The loaded shim, shared by the module: loading it has no per-test state."""
    return _load_relay()


class _Validator:
    """A stand-in for ``validate_activity`` that records its call and obeys a script."""

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []
        self.activity: dict[str, Any] = {"type": "message", "id": "activity-1"}
        self.error: Exception | None = None

    def __call__(self, body: Any, token: Any, *, cloud: str, app_id: str) -> dict[str, Any]:
        self.calls.append(SimpleNamespace(body=body, token=token, cloud=cloud, app_id=app_id))
        if self.error is not None:
            raise self.error
        return self.activity

    @property
    def call(self) -> SimpleNamespace:
        assert len(self.calls) == 1, f"expected exactly one call, got {len(self.calls)}"
        return self.calls[0]


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the two settings the shim reads, whatever the ambient environment holds."""
    monkeypatch.setenv("TEAMS_APP_ID", APP_ID)
    monkeypatch.delenv("TEAMS_CLOUD", raising=False)


@pytest.fixture
def validator(relay: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Validator:
    """Replace the shim's bound ``validate_activity`` for the length of one test."""
    stub = _Validator()
    monkeypatch.setattr(relay, "validate_activity", stub)
    return stub


def _post(
    relay: ModuleType,
    *,
    authorization: str | None = "Bearer a-signed-token",
    body: bytes = RAW_BODY,
) -> tuple[Any, Out]:
    """POST one activity and return the response beside the queue binding."""
    headers = {} if authorization is None else {"Authorization": authorization}
    queue = Out()
    response = relay.messages(HttpRequest(body=body, headers=headers), queue)
    return response, queue


# --- what never reaches the validator ---------------------------------------


def test_a_request_with_no_authorization_header_is_refused_without_a_key_lookup(
    relay: ModuleType, validator: _Validator
) -> None:
    response, queue = _post(relay, authorization=None)

    assert response.status_code == 401
    assert validator.calls == []
    assert queue.values == []


def test_a_request_authenticating_with_another_scheme_is_refused_without_a_key_lookup(
    relay: ModuleType, validator: _Validator
) -> None:
    response, queue = _post(relay, authorization="Basic dXNlcjpwYXNzd29yZA==")

    assert response.status_code == 401
    assert validator.calls == []
    assert queue.values == []


@pytest.mark.parametrize("header", ["Bearer", "Bearer ", "Bearer    ", "Bearer \t "])
def test_a_bearer_header_carrying_no_token_is_refused_without_a_key_lookup(
    relay: ModuleType, validator: _Validator, header: str
) -> None:
    response, queue = _post(relay, authorization=header)

    assert response.status_code == 401
    assert validator.calls == []
    assert queue.values == []


def test_a_body_that_is_not_utf_8_is_refused_without_a_key_lookup(
    relay: ModuleType, validator: _Validator
) -> None:
    """Undecodable bytes are not an activity, and asking the key source about them is waste."""
    response, queue = _post(relay, body=b'{"type": "message", "text": "\xff\xfe"}')

    assert response.status_code == 401
    assert validator.calls == []
    assert queue.values == []


# --- how the bearer value is read -------------------------------------------


def test_a_lower_case_bearer_scheme_is_accepted_as_the_http_standard_requires(
    relay: ModuleType, validator: _Validator
) -> None:
    response, queue = _post(relay, authorization="bearer a-signed-token")

    assert response.status_code == 200
    assert validator.call.token == "a-signed-token"
    assert len(queue.values) == 1


def test_the_bearer_value_reaches_the_validator_without_its_scheme_or_padding(
    relay: ModuleType, validator: _Validator
) -> None:
    _post(relay, authorization="Bearer   a-signed-token  ")

    assert validator.call.token == "a-signed-token"


# --- how a verdict becomes a status code ------------------------------------


def test_an_activity_the_validator_rejects_is_refused_for_good(
    relay: ModuleType, validator: _Validator
) -> None:
    validator.error = relay.ValidationError("the token was rejected")

    response, queue = _post(relay)

    assert response.status_code == 401
    assert queue.values == []


def test_an_unreachable_key_source_asks_teams_to_deliver_the_activity_again(
    relay: ModuleType, validator: _Validator
) -> None:
    validator.error = relay.KeySourceUnavailable("the signing keys are unreachable")

    response, queue = _post(relay)

    assert response.status_code == 503
    assert queue.values == []


def test_a_mis_configured_cloud_propagates_rather_than_becoming_a_refusal(
    relay: ModuleType, validator: _Validator
) -> None:
    """An unknown ``TEAMS_CLOUD`` is a statement about the relay, so it must reach the host."""
    validator.error = ValueError("unknown Teams cloud 'gccmid'")

    with pytest.raises(ValueError, match="unknown Teams cloud"):
        _post(relay)


def test_an_unexpected_validator_failure_is_not_mapped_to_a_status_code(
    relay: ModuleType, validator: _Validator
) -> None:
    """The two mapped exceptions are named individually, never caught as a group."""
    validator.error = RuntimeError("a bug in the relay")

    with pytest.raises(RuntimeError, match="a bug in the relay"):
        _post(relay)


# --- what reaches the queue -------------------------------------------------


@pytest.mark.usefixtures("validator")
def test_an_accepted_activity_is_enqueued_exactly_as_teams_sent_it(relay: ModuleType) -> None:
    response, queue = _post(relay)

    assert response.status_code == 200
    assert queue.values == [RAW_BODY.decode("utf-8")]
    assert queue.get().encode("utf-8") == RAW_BODY


@pytest.mark.usefixtures("validator")
def test_the_enqueued_body_is_the_original_text_and_not_a_re_serialisation(
    relay: ModuleType,
) -> None:
    """Guards the guard: a canonical fixture would let a round-trip pass unnoticed."""
    assert json.dumps(json.loads(RAW_BODY)).encode("utf-8") != RAW_BODY

    _, queue = _post(relay)

    assert queue.get() != json.dumps(json.loads(RAW_BODY))
    assert queue.get().encode("utf-8") == RAW_BODY


def test_the_validator_judges_the_same_text_that_is_enqueued(
    relay: ModuleType, validator: _Validator
) -> None:
    _, queue = _post(relay)

    assert validator.call.body == queue.get()


# --- what the settings decide -----------------------------------------------


@pytest.mark.parametrize("setting", [None, ""], ids=["unset", "empty"])
def test_an_unset_teams_cloud_validates_against_the_commercial_cloud(
    relay: ModuleType,
    validator: _Validator,
    monkeypatch: pytest.MonkeyPatch,
    setting: str | None,
) -> None:
    if setting is None:
        monkeypatch.delenv("TEAMS_CLOUD", raising=False)
    else:
        monkeypatch.setenv("TEAMS_CLOUD", setting)

    _post(relay)

    assert validator.call.cloud == "commercial"


def test_a_named_teams_cloud_is_passed_through_untouched(
    relay: ModuleType, validator: _Validator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEAMS_CLOUD", "gcchigh")

    _post(relay)

    assert validator.call.cloud == "gcchigh"


def test_the_app_id_the_token_must_name_comes_from_teams_app_id(
    relay: ModuleType, validator: _Validator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEAMS_APP_ID", "99999999-0000-0000-0000-000000000000")

    _post(relay)

    assert validator.call.app_id == "99999999-0000-0000-0000-000000000000"


def test_an_unset_app_id_still_reaches_the_validator_as_an_empty_audience(
    relay: ModuleType, validator: _Validator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audience check is the validator's to fail -- the relay does not pre-empt it."""
    monkeypatch.delenv("TEAMS_APP_ID", raising=False)

    _post(relay)

    assert validator.call.app_id == ""


# --- what the host is told at deployment ------------------------------------


def test_the_endpoint_is_the_anonymous_messages_route_teams_posts_to(relay: ModuleType) -> None:
    route = relay.app.kwargs_for("route")

    assert route["route"] == "messages"
    assert route["methods"] == ["POST"]
    assert route["auth_level"] is AuthLevel.ANONYMOUS


def test_the_output_binding_names_the_queue_setting_and_the_connection_setting(
    relay: ModuleType,
) -> None:
    """Both are names the host resolves, so a typo here is invisible until deployment."""
    output = relay.app.kwargs_for("service_bus_queue_output")

    assert output["arg_name"] == "queue"
    assert output["queue_name"] == "%TEAMS_SERVICEBUS_QUEUE%"
    assert output["connection"] == "SERVICEBUS_CONNECTION"


def test_the_function_is_registered_under_a_stable_name(relay: ModuleType) -> None:
    assert relay.app.kwargs_for("function_name") == {"name": "teams_messages"}
