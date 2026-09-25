"""Unit tests for the Teams relay's JWT validation boundary.

The relay ships as a service template rather than as an importable package, so
``validation.py`` is loaded by path the way ``tests/benchmark/test_matrix.py``
loads the matrix launcher. ``jwt`` is imported plainly: it is a dev-extra
dependency, so a missing install has to fail the suite rather than skip it.

Every token here is signed locally with a generated RSA key and every fetch is
stubbed -- the module's two network seams are its own ``urlopen`` (the OpenID
metadata document) and ``urllib.request.urlopen`` (the JWKS, fetched by
PyJWT). The autouse ``network`` fixture makes an unstubbed fetch a test
failure rather than a real request.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

_VALIDATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "templates"
    / "services"
    / "teams_bridge"
    / "relay"
    / "validation.py"
)


def _load_validation(module_name: str):
    """Execute ``validation.py`` as ``module_name`` and return the module."""
    spec = importlib.util.spec_from_file_location(module_name, _VALIDATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the relay ships as a service template, not
    # a package: validation.py is loaded by path and registered before exec so
    # its module-level dataclass resolves ``__module__``, and the module-level
    # load below feeds a parametrize decorator, which is evaluated at import.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


validation = _load_validation("teams_relay_validation")

APP_ID = "11111111-2222-3333-4444-555555555555"
SERVICE_URL = "https://smba.trafficmanager.net/amer/"
JWKS_URI = "https://sentinel.example/keys"
KID = "relay-test-key"

_SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_IMPOSTOR_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks(key: rsa.RSAPrivateKey = _SIGNING_KEY, kid: str = KID) -> dict:
    """A one-key JWKS document publishing ``key``'s public half under ``kid``."""
    entry = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    entry.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [entry]}


def _claims(**overrides) -> dict:
    now = int(time.time())
    claims = {
        "iss": validation.CLOUDS["commercial"].issuer,
        "aud": APP_ID,
        "serviceurl": SERVICE_URL,
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + 300,
    }
    claims.update(overrides)
    return claims


def _token(key=_SIGNING_KEY, *, kid: str | None = KID, algorithm: str = "RS256", **overrides):
    headers = {} if kid is None else {"kid": kid}
    return jwt.encode(_claims(**overrides), key, algorithm=algorithm, headers=headers)


def _body(service_url: str = SERVICE_URL) -> bytes:
    activity = {
        "type": "message",
        "id": "activity-1",
        "serviceUrl": service_url,
        "text": "how is the beam?",
    }
    return json.dumps(activity).encode()


class _Response:
    """The little of an ``http.client.HTTPResponse`` that ``json.load`` uses."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, *_args) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _metadata_source(record, *, payload: bytes | None = None, error: Exception | None = None):
    """A stand-in for the module's own ``urlopen``: the metadata GET.

    Given neither ``payload`` nor ``error``, reaching it fails the test.
    """

    def _open(url, timeout=None):
        record.metadata.append((url, timeout))
        if error is not None:
            raise error
        if payload is None:
            raise AssertionError(f"unexpected metadata fetch: {url}")
        return _Response(payload)

    return _open


def _jwks_source(record, *, payload: bytes | None = None, error: Exception | None = None):
    """A stand-in for ``urllib.request.urlopen``: PyJWT's JWKS GET.

    Given neither ``payload`` nor ``error``, reaching it fails the test.
    """

    def _open(request, timeout=None, context=None):  # noqa: ARG001 - stands in for urlopen, whose caller names timeout
        record.jwks.append((request.full_url, timeout))
        if error is not None:
            raise error
        if payload is None:
            raise AssertionError(f"unexpected JWKS fetch: {request.full_url}")
        return _Response(payload)

    return _open


@pytest.fixture(autouse=True)
def network(monkeypatch):
    """Stub both fetch seams; by default reaching either one fails the test."""
    record = SimpleNamespace(metadata=[], jwks=[])
    monkeypatch.setattr(validation, "urlopen", _metadata_source(record))
    monkeypatch.setattr(urllib.request, "urlopen", _jwks_source(record))
    return record


@pytest.fixture(autouse=True)
def _cold_key_sources(monkeypatch):
    """Each test starts with nothing cached, as a fresh worker would."""
    monkeypatch.setattr(validation, "_key_sources", {})


@pytest.fixture
def served(monkeypatch, network):
    """Serve a metadata document naming ``JWKS_URI`` and a JWKS holding the key."""
    monkeypatch.setattr(
        validation,
        "urlopen",
        _metadata_source(network, payload=json.dumps({"jwks_uri": JWKS_URI}).encode()),
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _jwks_source(network, payload=json.dumps(_jwks()).encode()),
    )
    return network


# --- what a genuine request looks like -------------------------------------


def test_a_valid_token_yields_the_parsed_activity(served):
    activity = validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)

    assert activity == json.loads(_body())
    assert served.metadata == [
        (validation.CLOUDS["commercial"].metadata_url, validation.KEY_FETCH_TIMEOUT_SEC)
    ]


@pytest.mark.usefixtures("served")
def test_a_token_not_yet_valid_within_the_skew_allowance_is_accepted():
    """A relay clock a minute behind the issuer must not reject live traffic."""
    token = _token(nbf=int(time.time()) + 60)

    assert validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


# --- what a bad token looks like -------------------------------------------


@pytest.mark.usefixtures("served")
def test_a_token_for_another_bot_is_rejected():
    token = _token(aud="99999999-0000-0000-0000-000000000000")

    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


@pytest.mark.usefixtures("served")
def test_a_token_from_another_issuer_is_rejected():
    token = _token(iss="https://login.example.test")

    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


@pytest.mark.usefixtures("served")
def test_a_token_that_expired_past_the_skew_allowance_is_rejected():
    now = int(time.time())
    token = _token(iat=now - 400, nbf=now - 400, exp=now - 360)

    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


@pytest.mark.usefixtures("served")
def test_a_token_naming_a_different_serviceurl_than_the_activity_is_rejected():
    token = _token(serviceurl="https://smba.trafficmanager.net/emea/")

    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


@pytest.mark.usefixtures("served")
def test_a_token_signed_with_another_algorithm_is_rejected():
    """The allow-list is pinned in code, never read from the token or metadata."""
    token = _token("a shared secret is not a signature", algorithm="HS256")

    # Matched on the reason: without the pin the token would still be refused,
    # but for a failed signature check rather than for the algorithm it names.
    with pytest.raises(validation.ValidationError, match="alg value is not allowed"):
        validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


@pytest.mark.usefixtures("served")
def test_a_bad_signature_under_a_known_kid_stays_a_validation_error():
    """The key source answered; the token simply is not what it claims to be."""
    token = _token(_IMPOSTOR_KEY)

    with pytest.raises(validation.ValidationError, match="Signature verification failed"):
        validation.validate_activity(_body(), token, cloud="commercial", app_id=APP_ID)


def test_a_bearer_value_that_is_not_a_jwt_costs_no_fetch(network):
    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), "not-a-jwt-at-all", cloud="commercial", app_id=APP_ID)

    assert network.metadata == []
    assert network.jwks == []


def test_a_token_without_a_kid_header_costs_no_fetch(network):
    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), _token(kid=None), cloud="commercial", app_id=APP_ID)

    assert network.metadata == []
    assert network.jwks == []


# --- what an unreachable key source looks like ------------------------------

_METADATA_FAILURES = {
    "connection times out": {"error": TimeoutError("timed out")},
    "connection is reset": {
        "error": http.client.RemoteDisconnected("remote end closed without a response")
    },
    "document names no jwks_uri": {"payload": json.dumps({"issuer": "x"}).encode()},
    "document is not an object": {"payload": b'["not", "a", "document"]'},
    "jwks_uri is not http(s)": {"payload": json.dumps({"jwks_uri": "file:///keys"}).encode()},
}


@pytest.mark.parametrize("case", sorted(_METADATA_FAILURES), ids=lambda case: case)
def test_a_metadata_failure_is_reported_as_an_outage(monkeypatch, network, case):
    monkeypatch.setattr(
        validation, "urlopen", _metadata_source(network, **_METADATA_FAILURES[case])
    )

    with pytest.raises(validation.KeySourceUnavailable):
        validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)


_JWKS_FAILURES = {
    "connection is reset": {
        "error": http.client.RemoteDisconnected("remote end closed without a response")
    },
    "body is not json": {"payload": b"<html>gateway error</html>"},
    "body is not utf-8": {"payload": b"\x80\x81 not decodable"},
    "body is a json array": {"payload": b'["not", "a", "key", "set"]'},
    "key set is empty": {"payload": b'{"keys": []}'},
    "kid is absent after the refresh": {
        "payload": json.dumps(_jwks(kid="some-other-key")).encode()
    },
}


@pytest.mark.parametrize("case", sorted(_JWKS_FAILURES), ids=lambda case: case)
def test_a_jwks_failure_is_reported_as_an_outage(monkeypatch, network, case):
    monkeypatch.setattr(
        validation,
        "urlopen",
        _metadata_source(network, payload=json.dumps({"jwks_uri": JWKS_URI}).encode()),
    )
    monkeypatch.setattr(urllib.request, "urlopen", _jwks_source(network, **_JWKS_FAILURES[case]))

    with pytest.raises(validation.KeySourceUnavailable):
        validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)


def test_an_unexpected_key_source_failure_is_not_masked_as_an_outage(monkeypatch):
    """Stage B's boundary is the stated exception tuple, not a bare ``except``."""

    def _explode(_name, _cloud):
        raise RuntimeError("a bug in the relay, not an outage")

    monkeypatch.setattr(validation, "_key_source", _explode)

    with pytest.raises(RuntimeError):
        validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)


# --- how the key source is built and kept -----------------------------------


def test_importing_the_module_fetches_nothing(monkeypatch):
    """A cold start must not depend on the metadata endpoint being up."""

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the module fetched something at import time")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    name = "teams_relay_validation_import_probe"
    try:
        probe = _load_validation(name)
        assert probe._key_sources == {}
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("cloud", sorted(validation.CLOUDS))
def test_the_key_client_is_built_on_the_jwks_uri_not_the_metadata_url(monkeypatch, network, cloud):
    built = []

    class _RecordingClient:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def get_signing_key(self, kid):
            assert kid == KID
            return SimpleNamespace(key=_SIGNING_KEY.public_key())

    monkeypatch.setattr(jwt, "PyJWKClient", _RecordingClient)
    monkeypatch.setattr(
        validation,
        "urlopen",
        _metadata_source(network, payload=json.dumps({"jwks_uri": JWKS_URI}).encode()),
    )
    token = _token(iss=validation.CLOUDS[cloud].issuer)

    assert validation.validate_activity(_body(), token, cloud=cloud, app_id=APP_ID)

    assert built == [
        {
            "uri": JWKS_URI,
            "lifespan": validation.KEY_SOURCE_LIFESPAN_SEC,
            "timeout": validation.KEY_FETCH_TIMEOUT_SEC,
        }
    ]
    assert network.metadata == [
        (validation.CLOUDS[cloud].metadata_url, validation.KEY_FETCH_TIMEOUT_SEC)
    ]


def test_gcchigh_uses_its_own_metadata_url_and_issuer(served):
    token = _token(iss=validation.CLOUDS["gcchigh"].issuer)

    assert validation.validate_activity(_body(), token, cloud="gcchigh", app_id=APP_ID)
    assert served.metadata[0][0] == validation.CLOUDS["gcchigh"].metadata_url

    with pytest.raises(validation.ValidationError):
        validation.validate_activity(_body(), _token(), cloud="gcchigh", app_id=APP_ID)


def test_the_key_source_is_reused_within_its_lifespan(served):
    validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)
    validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)

    assert len(served.metadata) == 1


def test_the_key_source_is_rebuilt_once_its_lifespan_runs_out(monkeypatch, served):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(validation, "monotonic", lambda: clock.now)

    validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)
    clock.now = validation.KEY_SOURCE_LIFESPAN_SEC + 1
    validation.validate_activity(_body(), _token(), cloud="commercial", app_id=APP_ID)

    assert len(served.metadata) == 2
