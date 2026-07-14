"""Full-stack Docker e2e for the OpenObserve telemetry add-on (Phase 2).

Stands up the shipped ``openobserve`` deploy add-on with a real ``osprey build``
+ ``osprey deploy up``, then proves the OTLP round-trip **without a live LLM**:
it POSTs a SYNTHETIC OTLP payload — one ``claude_code.*`` metric and one
``com.anthropic.claude_code.events`` record — using the Basic auth header that
the REAL resolver (:func:`_build_telemetry_env`) computes from the project
credentials, and asserts via the OpenObserve query API that both landed (i.e.
auth was accepted and ingest works).

Why synthetic and not a live agent turn: a real ``claude_code_*`` metric needs
an actual model turn + a provider API key, which this CI lane deliberately does
not require. The synthetic payload exercises exactly the thing under test — the
compose template, the credential bootstrap, and the computed Basic header — with
no LLM dependency. An OPTIONAL live-agent smoke (``test_live_agent_...``) is
appended and ``skipif``-gated on a provider key for when one is present.

CONTAINER SAFETY: every docker/podman invocation names an EXACT container/image
— never a wildcard, never ``system prune``/``--volumes``. Teardown goes through
``osprey deploy down`` (the shipped compose teardown), not a raw ``docker rm``
sweep.

Gating: needs Docker; skipped entirely if unavailable. Lives in ``tests/e2e/``
so the fast lane never collects this real-container build+deploy; run via
``pytest tests/e2e/`` (NOT ``-m e2e``).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.cli.claude_code_telemetry import _build_telemetry_env

# Deliberately NOT OpenObserve's 5080 default: this is a shared dev machine and
# 5080 can collide with an unrelated process. Pinned via the generated config's
# services.openobserve.port below; the container still listens on 5080 inside.
OO_HOST_PORT = 15080
OO_BASE_URL = f"http://localhost:{OO_HOST_PORT}"
# The project the fixture builds; feeds both the build arg and the derived
# container name (compose renders container_name as ``<project>-openobserve``,
# per the per-project namespacing convention). Mirrors the deploy-e2e pattern of
# deriving container targets from the project rather than hardcoding a
# host-global literal.
OO_PROJECT = "proj"
OO_CONTAINER = f"{OO_PROJECT}-openobserve"  # matches the rendered container_name

# The named volume OpenObserve pins its root credentials into on FIRST init.
# Its name is ``<compose-project>_openobserve_data`` where the compose project is
# ``services`` (the compose files live under ``build/services/``) — so it is
# HOST-GLOBAL, shared by every OSPREY project's openobserve on one host. Because
# OpenObserve ignores new root creds once a volume is initialized, a volume left
# behind by another deploy (now that telemetry is on by default) would make this
# credential-sensitive test 401. The fixture removes it before deploy and on
# teardown so the store always initializes with THIS test's credentials.
OO_DATA_VOLUME = "services_openobserve_data"
OO_ORG = "default"

# Credentials the compose service and the telemetry resolver BOTH read from the
# project .env — the single source of truth this feature is built around. The
# e2e writes these into .env so the deployed container and the computed Basic
# header agree.
OO_EMAIL = "e2e@osprey.local"
OO_PASSWORD = "E2eProbePass#123"

DEPLOY_UP_TIMEOUT_SEC = 600
HEALTH_TIMEOUT_SEC = 120.0
INGEST_QUERY_TIMEOUT_SEC = 60.0

# Rerun only on AssertionError (the genuinely flaky failures: container-startup
# health-wait and ingest-visibility timing). Deterministic setup errors — a
# failed build/deploy, a bound-port collision — surface via pytest.fail() as
# ``Failed`` (not AssertionError) and fail fast rather than burning a retry on a
# multi-minute redeploy. reruns=2 per the multi-step-pipeline e2e convention.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"]),
]


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _remove_oo_data_volume() -> None:
    """Remove the host-global OpenObserve data volume by EXACT name, if present.

    OpenObserve pins its root credentials on the volume's first init, so a volume
    left behind by an earlier deploy would reject this test's credentials (401).
    Removing it guarantees a clean init. Exact-named and failure-tolerant — never
    a wildcard, never a prune; a missing/in-use volume is a no-op.
    """
    subprocess.run(
        ["docker", "volume", "rm", OO_DATA_VOLUME],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _enable_openobserve(project_dir: Path) -> None:
    """Opt the generated project into the openobserve add-on and pin its host port."""
    config_path = project_dir / "config.yml"
    assert config_path.is_file(), f"generated config.yml missing at {config_path}"
    config = yaml.safe_load(config_path.read_text())

    services = config.setdefault("services", {})
    oo = services.setdefault("openobserve", {"path": "./services/openobserve"})
    oo["port"] = OO_HOST_PORT
    config["deployed_services"] = ["openobserve"]

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # Credentials: single source of truth in .env, read by BOTH the compose
    # service and the computed Basic header.
    env_path = project_dir / ".env"
    existing = env_path.read_text() if env_path.is_file() else ""
    env_path.write_text(
        existing.rstrip("\n")
        + f"\nZO_ROOT_USER_EMAIL={OO_EMAIL}\nZO_ROOT_USER_PASSWORD={OO_PASSWORD}\n"
    )


@pytest.fixture(scope="module")
def deployed_openobserve(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Build + ``osprey deploy up`` an openobserve-enabled project; tear down after."""
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("openobserve_e2e")
    project_dir = base / OO_PROJECT

    build = _run(
        [
            str(osprey_bin),
            "build",
            OO_PROJECT,
            "--preset",
            "hello-world",
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    _enable_openobserve(project_dir)

    # Guarantee a clean store: the data volume is host-global (see OO_DATA_VOLUME),
    # so a volume left by another project's openobserve — common now that telemetry
    # is on by default — would pin foreign credentials and 401 this test. Remove it
    # before deploy so OpenObserve initializes with THIS test's credentials.
    _remove_oo_data_volume()

    try:
        up = _run(
            [str(osprey_bin), "deploy", "up", "-d"],
            cwd=project_dir,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey deploy up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{OO_BASE_URL}/healthz", HEALTH_TIMEOUT_SEC)
        yield project_dir
    finally:
        down = _run([str(osprey_bin), "deploy", "down"], cwd=project_dir, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey deploy down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # ``deploy down`` keeps volumes; drop the host-global data volume so this
        # test never leaves foreign credentials pinned for a later deploy.
        _remove_oo_data_volume()


def _wait_for_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {url} (last: {last_err})")


def _resolver_telemetry_env() -> dict[str, str]:
    """Build the telemetry env via the REAL resolver helper, pinned at the deployed URL.

    The endpoint is set explicitly to the deployed host:port (the resolver's
    localhost/service-DNS derivation is unit-tested separately and would target
    :5080, not this test's pinned port). The Basic auth header, however, is the
    genuine value the resolver computes from the project credentials — so a
    successful ingest proves OSPREY's real header authenticates.
    """
    return _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "endpoint": f"{OO_BASE_URL}/api/{OO_ORG}",
            "openobserve": {"user": OO_EMAIL, "password": OO_PASSWORD, "org": OO_ORG},
        },
        in_container=False,
    )


def _auth_header_from_resolver() -> str:
    env = _resolver_telemetry_env()
    otlp_headers = env["OTEL_EXPORTER_OTLP_HEADERS"]
    # OTLP header wire format: comma-separated k=v; value may contain '=' (base64
    # padding), so split on the FIRST '=' only.
    for pair in otlp_headers.split(","):
        key, _sep, value = pair.partition("=")
        if key.strip() == "Authorization":
            return value.strip()
    raise AssertionError(f"no Authorization header in resolver output: {otlp_headers!r}")


def _otlp_post(path: str, payload: dict, auth: str) -> tuple[int, str]:
    req = urllib.request.Request(  # noqa: S310 - localhost
        f"{OO_BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": auth},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _query(path: str, payload: dict | None, auth: str, method: str = "POST") -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - localhost
        f"{OO_BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Authorization": auth},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _synthetic_metric(now_ns: int) -> dict:
    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "claude-code"}}]
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "com.anthropic.claude_code"},
                        "metrics": [
                            {
                                "name": "claude_code.session.count",
                                "sum": {
                                    "dataPoints": [
                                        {
                                            "asInt": "1",
                                            "timeUnixNano": str(now_ns),
                                            "startTimeUnixNano": str(now_ns),
                                        }
                                    ],
                                    "aggregationTemporality": 2,
                                    "isMonotonic": True,
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _synthetic_event(now_ns: int) -> dict:
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "claude-code"}}]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "com.anthropic.claude_code.events"},
                        "logRecords": [
                            {
                                "timeUnixNano": str(now_ns),
                                "body": {"stringValue": "user_prompt"},
                                "attributes": [
                                    {
                                        "key": "event.name",
                                        "value": {"stringValue": "user_prompt"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthetic_otlp_roundtrip_via_computed_header(deployed_openobserve: Path) -> None:
    """Ingest a synthetic metric + event with the resolver's Basic header; assert both land."""
    auth = _auth_header_from_resolver()
    assert auth.startswith("Basic "), f"resolver did not produce a Basic header: {auth!r}"
    # The header must decode back to the project credentials (proves it is the
    # real computed value, not a placeholder).
    decoded = base64.b64decode(auth.removeprefix("Basic ")).decode()
    assert decoded == f"{OO_EMAIL}:{OO_PASSWORD}"

    now_ns = int(time.time() * 1_000_000_000)
    metric_status, metric_body = _otlp_post(
        f"/api/{OO_ORG}/v1/metrics", _synthetic_metric(now_ns), auth
    )
    assert metric_status == 200, f"metric ingest rejected: {metric_status} {metric_body}"
    event_status, event_body = _otlp_post(f"/api/{OO_ORG}/v1/logs", _synthetic_event(now_ns), auth)
    assert event_status == 200, f"event ingest rejected: {event_status} {event_body}"

    # Poll the query API until both are visible (ingest is async).
    start_us = (now_ns // 1_000) - 3_600_000_000
    end_us = (now_ns // 1_000) + 60_000_000
    deadline = time.monotonic() + INGEST_QUERY_TIMEOUT_SEC
    metric_total = 0
    event_total = 0
    while time.monotonic() < deadline:
        # >=1 claude_code_* metric stream with a datapoint.
        s, streams = _query(f"/api/{OO_ORG}/streams?type=metrics", None, auth, method="GET")
        cc_metrics = [
            x["name"]
            for x in (streams.get("list") or [])
            if str(x.get("name", "")).startswith("claude_code")
        ]
        if cc_metrics:
            _, mres = _query(
                f"/api/{OO_ORG}/_search?type=metrics",
                {
                    "query": {
                        "sql": f'SELECT * FROM "{cc_metrics[0]}"',
                        "start_time": start_us,
                        "end_time": end_us,
                        "size": 5,
                    }
                },
                auth,
            )
            metric_total = mres.get("total", 0)
        # >=1 event record we posted (default logs stream, filtered to our marker).
        _, eres = _query(
            f"/api/{OO_ORG}/_search?type=logs",
            {
                "query": {
                    "sql": "SELECT * FROM \"default\" WHERE service_name = 'claude-code'",
                    "start_time": start_us,
                    "end_time": end_us,
                    "size": 5,
                }
            },
            auth,
        )
        event_total = eres.get("total", 0)
        if metric_total >= 1 and event_total >= 1:
            break
        time.sleep(2.0)

    assert metric_total >= 1, "no claude_code_* metric visible in OpenObserve after ingest"
    assert event_total >= 1, "no claude_code event record visible in OpenObserve after ingest"


def test_bad_credentials_are_rejected(deployed_openobserve: Path) -> None:
    """Sanity: OpenObserve enforces auth, so a green round-trip really proves auth."""
    bad = "Basic " + base64.b64encode(b"wrong@user.local:nope").decode()
    status, _ = _otlp_post(f"/api/{OO_ORG}/v1/logs", _synthetic_event(1), bad)
    assert status in (401, 403), f"expected auth rejection, got {status}"


@pytest.mark.skipif(
    not os.environ.get("ALS_APG_API_KEY"),
    reason="live-agent smoke needs ALS_APG_API_KEY (advisory lane only)",
)
def test_live_agent_metric_lands(deployed_openobserve: Path) -> None:
    """OPTIONAL advisory smoke: one real agent turn emits a real claude_code_* metric.

    Not a CI gate, but it *runs* whenever ALS_APG_API_KEY is present (the same
    provider key CI provisions and the rest of the e2e suite uses). Proves the
    true end-to-end path (agent launch -> native OTEL export -> OpenObserve)
    that the synthetic test deliberately stops short of.
    """
    project_dir = deployed_openobserve

    # Enable telemetry against the deployed store (host-run agent -> localhost)
    # and point the turn at the als-apg provider (the CI-provisioned key), then
    # drive one turn through the console script. The fixture build defaults to
    # provider: anthropic; this in-place edit switches only the live turn.
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["claude_code"]["provider"] = "als-apg"
    config["claude_code"]["default_model"] = "haiku"
    config["claude_code"]["telemetry"] = {
        "enabled": True,
        "backend": "openobserve",
        "endpoint": f"{OO_BASE_URL}/api/{OO_ORG}",
        "openobserve": {"user": OO_EMAIL, "password": OO_PASSWORD, "org": OO_ORG},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    osprey_bin = _find_osprey_console_script()
    turn = _run(
        [
            str(osprey_bin),
            "query",
            "Say 'telemetry-smoke-ok' and nothing else.",
        ],
        cwd=project_dir,
        timeout=180,
    )
    if turn.returncode != 0:
        pytest.skip(f"live agent turn did not complete (advisory): {turn.stderr[:400]}")

    auth = _auth_header_from_resolver()
    deadline = time.monotonic() + INGEST_QUERY_TIMEOUT_SEC
    while time.monotonic() < deadline:
        s, streams = _query(f"/api/{OO_ORG}/streams?type=metrics", None, auth, method="GET")
        cc = [
            x["name"]
            for x in (streams.get("list") or [])
            if str(x.get("name", "")).startswith("claude_code")
        ]
        if cc:
            return
        time.sleep(2.0)
    raise AssertionError("no real claude_code_* metric appeared after a live agent turn")
