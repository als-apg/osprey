"""gate_open: fail-closed checks (dispatcher health, worker health, and — only
where an alert host is configured — no open osprey-alert issue), with the GitLab
*API failure* case unit-tested separately from the *open alert* case (both close
the gate, distinct causes).

The alert-issue check is opt-in: it runs only when both ``gitlab_url`` and
``gitlab_issues_token`` are set. The final section pins that behavior from both
sides — unconfigured the gate ignores GitLab entirely, configured it keeps every
fail-closed guarantee.
"""

from types import SimpleNamespace

import httpx

from osprey.bridges.core import health_gate
from osprey.bridges.core.health_gate import gate_open


def _cfg(**overrides):
    # A structural stub of the CoreConfig fields gate_open reads. Field names
    # MUST track CoreConfig's — test_gate_open_reads_real_coreconfig_fields
    # below guards against the stub and the real config drifting apart.
    fields = {
        "dispatcher_url": "http://disp:10010",
        "worker_url": "http://work:9190",
        "gitlab_url": "https://git.als.lbl.gov",
        "gitlab_project": "physics/production/als-profiles",
        "gitlab_issues_token": "gltok",
        "trust_env": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _health(status="ok", code=200):
    def f(_req):
        return httpx.Response(code, json={"status": status})

    return f


def _gitlab(code=200, body=None):
    def f(_req):
        return httpx.Response(code, json=[] if body is None else body)

    return f


def _raise(exc):
    def f(_req):
        raise exc

    return f


def _client(disp=None, work=None, gitlab=None, record=None):
    disp = disp or _health()
    work = work or _health()
    gitlab = gitlab or _gitlab()

    def handler(request):
        host = request.url.host
        if host == "git.als.lbl.gov":
            if record is not None:
                record.append(request)
            return gitlab(request)
        if host == "disp":
            return disp(request)
        if host == "work":
            return work(request)
        raise AssertionError(f"unexpected host {host}")

    # trust_env=False: with ambient HTTP(S)_PROXY set (dev shells, CI), httpx
    # would otherwise mount env-proxy transports that bypass the MockTransport
    # and hit the network — the suite must be immune to environment/test-order.
    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


# --- happy path ------------------------------------------------------------


def test_all_healthy_no_alert_opens_gate():
    assert gate_open(_cfg(), _client()) is True


def test_gitlab_empty_list_opens_gate():
    # Isolated positive case for check (c): a clean 200 + empty list.
    assert gate_open(_cfg(), _client(gitlab=_gitlab(200, []))) is True


# --- dispatcher / worker health --------------------------------------------


def test_dispatcher_status_not_ok_closes_gate():
    assert gate_open(_cfg(), _client(disp=_health(status="degraded"))) is False


def test_dispatcher_non_200_closes_gate():
    assert gate_open(_cfg(), _client(disp=_health(code=503))) is False


def test_worker_status_not_ok_closes_gate():
    assert gate_open(_cfg(), _client(work=_health(status="degraded"))) is False


def test_worker_non_200_closes_gate():
    assert gate_open(_cfg(), _client(work=_health(code=500))) is False


def test_health_transport_error_closes_gate():
    assert gate_open(_cfg(), _client(disp=_raise(httpx.ConnectError("boom")))) is False


# --- GitLab: open alert vs API failure are DISTINCT causes -----------------


def test_open_alert_closes_gate():
    # (c) fails because an alert IS open — a positive observation of breakage.
    # Also the phase's second success criterion: with gitlab_url and the issues
    # token both set, an open alert issue closes the gate.
    body = [{"iid": 7, "labels": ["osprey-alert"], "state": "opened"}]
    assert gate_open(_cfg(), _client(gitlab=_gitlab(200, body))) is False


def test_gitlab_api_failure_non_200_closes_gate():
    # (c) fails because we could not ASK — a 500 from GitLab, NOT an open alert.
    assert gate_open(_cfg(), _client(gitlab=_gitlab(500, {"message": "boom"}))) is False


def test_gitlab_api_failure_exception_closes_gate():
    # (c) fails on a transport error (unreachable/timeout) — again an API
    # failure distinct from observing an open alert.
    assert gate_open(_cfg(), _client(gitlab=_raise(httpx.TimeoutException("t")))) is False


def test_gitlab_non_list_body_closes_gate():
    assert gate_open(_cfg(), _client(gitlab=_gitlab(200, {"unexpected": "shape"}))) is False


# --- request shape / short-circuit / direct connection ---------------------


def test_gitlab_request_shape():
    record = []
    gate_open(_cfg(), _client(record=record))
    assert len(record) == 1
    req = record[0]
    assert req.headers["PRIVATE-TOKEN"] == "gltok"
    # project path is URL-encoded into the API path (no bare slashes).
    assert "physics%2Fproduction%2Fals-profiles" in str(req.url)
    assert req.url.path.endswith("/issues")
    assert req.url.params.get("labels") == "osprey-alert"
    assert req.url.params.get("state") == "opened"


def test_dispatcher_failure_short_circuits_gitlab():
    # A closed earlier check must not even query GitLab.
    record = []
    assert gate_open(_cfg(), _client(disp=_health(code=503), record=record)) is False
    assert record == []


def test_default_client_is_direct(monkeypatch):
    # With no injected client, gate_open must build a client honoring the
    # config's trust_env, which defaults False so the alert host is reached
    # directly. Ambient proxy env must not leak in regardless of test order, so
    # scrub it first.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    captured = {}
    mock = _client(disp=_health(status="down"))  # built before patching; any close

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return mock

    monkeypatch.setattr(health_gate.httpx, "Client", fake_client)
    gate_open(_cfg())
    assert captured.get("trust_env") is False


def test_gate_open_reads_real_coreconfig_fields():
    """Drive gate_open with a REAL CoreConfig, not the stub: an attribute-name
    drift between this module and CoreConfig (e.g. gitlab_token vs
    gitlab_issues_token) raises inside the fail-closed except and silently
    closes the gate forever — only a real instance can catch that."""
    from osprey.bridges.core.config import CoreConfig

    cfg = CoreConfig(
        dispatcher_url="http://disp:10010",
        worker_url="http://work:9190",
        gitlab_url="https://git.als.lbl.gov",
        gitlab_project="physics/production/als-profiles",
        gitlab_issues_token="real-cfg-token",
    )
    record = []
    assert gate_open(cfg, _client(record=record)) is True
    assert len(record) == 1
    assert record[0].headers["PRIVATE-TOKEN"] == "real-cfg-token"


# --- the alert-issue check is opt-in ---------------------------------------


def test_unconfigured_alert_check_opens_gate_on_healthy_pair():
    """The phase's first success criterion: with no alert host configured, two
    healthy /health responses are enough to open the gate. Were this check still
    mandatory, an unconfigured site would fail it forever, the drain would never
    run, and parked messages would age out into give-up notices."""
    record = []
    cfg = _cfg(gitlab_url="", gitlab_project="", gitlab_issues_token="")
    assert gate_open(cfg, _client(record=record)) is True
    # ...and nothing was asked of GitLab at all.
    assert record == []


def test_unconfigured_alert_check_skipped_with_real_coreconfig_defaults():
    """CoreConfig's shipped defaults leave gitlab_url/project/token empty, so an
    out-of-the-box config must take the skip path — not merely the stub."""
    from osprey.bridges.core.config import CoreConfig

    cfg = CoreConfig(dispatcher_url="http://disp:10010", worker_url="http://work:9190")
    assert cfg.gitlab_url == ""
    assert cfg.gitlab_issues_token == ""
    record = []
    assert gate_open(cfg, _client(record=record)) is True
    assert record == []


def test_url_without_token_skips_alert_check():
    # A host with no token cannot query the issues API; half-configured is
    # unconfigured, not fail-closed-forever.
    record = []
    cfg = _cfg(gitlab_issues_token="")
    assert gate_open(cfg, _client(record=record)) is True
    assert record == []


def test_token_without_url_skips_alert_check():
    record = []
    cfg = _cfg(gitlab_url="")
    assert gate_open(cfg, _client(record=record)) is True
    assert record == []


def test_unconfigured_alert_check_still_requires_dispatcher_health():
    # Skipping (c) must not weaken (a): the gate is skipping one check, not
    # opening unconditionally.
    cfg = _cfg(gitlab_url="", gitlab_issues_token="")
    assert gate_open(cfg, _client(disp=_health(code=503))) is False


def test_unconfigured_alert_check_still_requires_worker_health():
    # ...nor (b).
    cfg = _cfg(gitlab_url="", gitlab_issues_token="")
    assert gate_open(cfg, _client(work=_health(status="degraded"))) is False


def test_alert_check_configured_predicate():
    assert health_gate.alert_check_configured(_cfg()) is True
    assert health_gate.alert_check_configured(_cfg(gitlab_url="")) is False
    assert health_gate.alert_check_configured(_cfg(gitlab_issues_token="")) is False
    assert health_gate.alert_check_configured(_cfg(gitlab_url="", gitlab_issues_token="")) is False


# --- trust_env comes from config, not a module constant --------------------


def test_default_client_honors_configured_trust_env(monkeypatch):
    # A site behind an egress proxy sets trust_env once, in config, and the
    # gate's own client picks it up.
    captured = {}
    mock = _client(disp=_health(status="down"))

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return mock

    monkeypatch.setattr(health_gate.httpx, "Client", fake_client)
    gate_open(_cfg(trust_env=True))
    assert captured.get("trust_env") is True
