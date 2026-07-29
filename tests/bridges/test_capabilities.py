"""pair_supports: the per-dispatch capability probe. BOTH the dispatcher and the
worker /health bodies must advertise the named capability in their "capabilities"
list; any failure on either side is fail-closed to False.

Table-driven over the two bodies via httpx.MockTransport: capable / one-side-
missing / both-missing / non-200 / timeout / garbage-json / missing key, plus
request-shape, short-circuit, and direct-connection checks."""

from types import SimpleNamespace

import httpx
import pytest

from osprey.bridges.core import capabilities
from osprey.bridges.core.capabilities import pair_supports

CAP = "input_files"


def _cfg():
    # Structural stub of the CoreConfig fields pair_supports reads; guarded
    # against drift by test_pair_supports_reads_real_coreconfig_fields below.
    return SimpleNamespace(
        dispatcher_url="http://disp:8010",
        worker_url="http://work:9190",
        trust_env=False,
    )


def _health(caps, code=200):
    """A /health responder. ``caps`` as a list becomes the ``capabilities`` field;
    ``None`` omits the key (older endpoint); a dict is returned as the whole body
    (malformed-shape cases)."""

    def f(_req):
        if isinstance(caps, dict):
            return httpx.Response(code, json=caps)
        body = {"status": "ok"}
        if caps is not None:
            body["capabilities"] = caps
        return httpx.Response(code, json=body)

    return f


def _raise(exc):
    def f(_req):
        raise exc

    return f


def _client(disp=None, work=None, record=None):
    disp = disp if disp is not None else _health([CAP])
    work = work if work is not None else _health([CAP])

    def handler(request):
        if record is not None:
            record.append(request)
        host = request.url.host
        if host == "disp":
            return disp(request)
        if host == "work":
            return work(request)
        raise AssertionError(f"unexpected host {host}")

    # trust_env=False so ambient HTTP(S)_PROXY (dev shells, CI) can't mount an
    # env-proxy transport that bypasses the MockTransport and hits the network.
    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


# --- truth table over the two /health bodies -------------------------------


@pytest.mark.parametrize(
    "disp_caps, work_caps, expected",
    [
        ([CAP], [CAP], True),  # both advertise -> supported
        ([CAP, "x"], ["y", CAP], True),  # present among others on both sides
        ([CAP], [], False),  # worker missing
        ([], [CAP], False),  # dispatcher missing
        ([], [], False),  # both missing
        (["other"], ["other"], False),  # neither names it
    ],
)
def test_pair_supports_truth_table(disp_caps, work_caps, expected):
    client = _client(_health(disp_caps), _health(work_caps))
    assert pair_supports(_cfg(), CAP, client) is expected


# --- fail-closed on any probe failure --------------------------------------


def test_dispatcher_non_200_is_false():
    assert pair_supports(_cfg(), CAP, _client(disp=_health([CAP], code=503))) is False


def test_worker_non_200_is_false():
    assert pair_supports(_cfg(), CAP, _client(work=_health([CAP], code=500))) is False


def test_dispatcher_timeout_is_false():
    assert pair_supports(_cfg(), CAP, _client(disp=_raise(httpx.TimeoutException("t")))) is False


def test_worker_transport_error_is_false():
    assert pair_supports(_cfg(), CAP, _client(work=_raise(httpx.ConnectError("boom")))) is False


def test_capabilities_not_a_list_is_false():
    # A malformed body whose capabilities field is not a list -> fail-closed.
    bad = _health({"status": "ok", "capabilities": {"input_files": True}})
    assert pair_supports(_cfg(), CAP, _client(disp=bad)) is False


def test_capabilities_key_missing_is_false():
    # An older /health with no capabilities field at all.
    assert pair_supports(_cfg(), CAP, _client(disp=_health(None))) is False


def test_garbage_json_body_is_false():
    def not_json(_req):
        return httpx.Response(200, text="<html>not json</html>")

    assert pair_supports(_cfg(), CAP, _client(disp=not_json)) is False


def test_non_string_entries_ignored():
    # Junk non-string entries don't crash the frozenset build; the real cap counts.
    client = _client(_health([CAP, 5, None]), _health([CAP]))
    assert pair_supports(_cfg(), CAP, client) is True


# --- request shape / short-circuit / direct connection ---------------------


def test_probes_both_health_endpoints():
    record = []
    pair_supports(_cfg(), CAP, _client(record=record))
    seen = [(r.url.host, r.url.path) for r in record]
    assert ("disp", "/health") in seen
    assert ("work", "/health") in seen
    assert all(r.method == "GET" for r in record)


def test_dispatcher_failure_short_circuits_worker():
    # If the dispatcher probe already fails, the worker must not be queried.
    record = []
    assert pair_supports(_cfg(), CAP, _client(disp=_health([], code=503), record=record)) is False
    assert [r.url.host for r in record] == ["disp"]


def test_dispatcher_lacking_capability_short_circuits_worker():
    # A healthy dispatcher that simply doesn't advertise the cap also short-circuits.
    record = []
    assert pair_supports(_cfg(), CAP, _client(disp=_health(["other"]), record=record)) is False
    assert [r.url.host for r in record] == ["disp"]


def test_default_client_is_direct(monkeypatch):
    # With no injected client, pair_supports must build a proxy-ignoring
    # (trust_env=False) client. Scrub ambient proxy env so test order can't leak.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    captured = {}
    mock = _client(disp=_health([], code=503))  # any answer; we only inspect kwargs

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return mock

    monkeypatch.setattr(capabilities.httpx, "Client", fake_client)
    pair_supports(_cfg(), CAP)
    assert captured.get("trust_env") is False


def test_default_client_honors_configured_trust_env(monkeypatch):
    # trust_env is config, not a module constant: a site behind an egress proxy
    # sets it once on CoreConfig and the probe's own client picks it up.
    captured = {}
    mock = _client(disp=_health([], code=503))

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return mock

    monkeypatch.setattr(capabilities.httpx, "Client", fake_client)
    cfg = _cfg()
    cfg.trust_env = True
    pair_supports(cfg, CAP)
    assert captured.get("trust_env") is True


def test_pair_supports_reads_real_coreconfig_fields():
    """Drive pair_supports with a REAL CoreConfig, not the stub: an attribute-name
    drift between this module and CoreConfig would raise inside the fail-closed
    except and silently answer False forever — only a real instance catches it."""
    from osprey.bridges.core.config import CoreConfig

    cfg = CoreConfig(dispatcher_url="http://disp:8010", worker_url="http://work:9190")
    assert pair_supports(cfg, CAP, _client()) is True
