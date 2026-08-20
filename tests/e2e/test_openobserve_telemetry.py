"""Full-stack Docker e2e for the OpenObserve telemetry add-on (Phase 2).

Stands up the shipped ``openobserve`` deploy add-on with a real ``osprey init`` +
``osprey build`` + ``osprey up``, then proves the OTLP round-trip **without a live LLM**:
it POSTs a SYNTHETIC OTLP payload — one ``claude_code.*`` metric and one
``com.anthropic.claude_code.events`` record — using the Basic auth header that
the REAL resolver (:func:`_build_telemetry_env`) computes from the project
credentials, and asserts via the OpenObserve query API that both landed (i.e.
auth was accepted and ingest works).

TWO identities are covered, and the distinction is the point. The root
credential (``ZO_ROOT_USER_*``) initializes the store and is what this fixture
brings it up with. The INGEST identity is what every shipped telemetry block
authenticates as: ``osprey up`` creates a service account in the store and
writes the token the store issues for it into the repo's ``.env`` as
``ZO_INGEST_SA_TOKEN``. The ingest round-trip reads those values back from
``.env`` rather than restating them, so it fails if the provisioning step
silently did not run — and a wrong-credential rejection is pinned for both, or
a store that stopped enforcing auth would pass every positive assertion.

Why synthetic and not a live agent turn: a real ``claude_code_*`` metric needs
an actual model turn + a provider API key, which this CI lane deliberately does
not require. The synthetic payload exercises exactly the thing under test — the
compose template, the credential bootstrap, and the computed Basic header — with
no LLM dependency. An OPTIONAL live-agent smoke (``test_live_agent_...``) is
appended and ``skipif``-gated on a provider key for when one is present.

CONTAINER SAFETY: every docker/podman invocation names an EXACT container/image
— never a wildcard, never ``system prune``/``--volumes``. Teardown goes through
``osprey down`` (the shipped compose teardown), not a raw ``docker rm`` sweep,
followed by exact-named removal of this project's own volumes
(``tests/e2e/_volumes.py``): ``down`` keeps them by design, and a rerun must
not inherit their state.

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

from osprey.build.claude_code_telemetry import _build_telemetry_env
from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS
from osprey.deployment.openobserve_provision import INGEST_TOKEN_VAR
from osprey.utils.dotenv import parse_dotenv_file
from tests.e2e._volumes import remove_project_volumes

#: The ingest account NAME, read off the registry that pairs it with the token
#: rather than spelled again here — the two must never disagree about which
#: identity the harvested token belongs to.
INGEST_EMAIL_VAR = _STORE_ISSUED_VARS[INGEST_TOKEN_VAR].identity_var

# Deliberately NOT OpenObserve's 5080 default: this is a shared dev machine and
# 5080 can collide with an unrelated process. Pinned through the SOURCE zone at
# init (see _write_port_override), because the compose file is rendered by the
# build; the container still listens on 5080 inside.
OO_HOST_PORT = 15080
OO_BASE_URL = f"http://localhost:{OO_HOST_PORT}"
# The deployment repo the fixture creates; its DIRECTORY NAME is the deployment's
# name, which is what the derived container name follows (compose renders
# container_name as ``<project>-openobserve``, per the per-project namespacing
# convention). Mirrors the deploy-e2e pattern of deriving container targets from
# the deployment rather than hardcoding a host-global literal.
OO_PROJECT = "proj"
OO_CONTAINER = f"{OO_PROJECT}-openobserve"  # matches the rendered container_name

# The named volume OpenObserve pins its root credentials into on FIRST init.
# Its name is ``<COMPOSE_PROJECT_NAME>_openobserve_data``; ``osprey up``
# pins ``COMPOSE_PROJECT_NAME`` to the project name (see
# ``runtime_helper.runtime_env``), so this test's volume is
# ``proj_openobserve_data``. Because OpenObserve ignores new root creds once a
# volume is initialized, a surviving volume from an earlier attempt (the
# fixture's own reruns included — ``osprey down`` keeps volumes) would pin
# whatever creds that attempt initialized with. The fixture removes it before
# deploy and on teardown so the store always initializes with THIS test's
# credentials. The project-named volume is removed by the shared label-scoped
# sweep (``tests/e2e/_volumes.py``); the pre-project-pinning deploys derived
# the compose project from the compose files' directory (``services``), so
# that legacy host-global name is removed by exact name too — a dev machine
# can still carry one, and it carries no project label the sweep could find.
OO_LEGACY_DATA_VOLUME = "services_openobserve_data"
OO_ORG = "default"

# Credentials the compose service and the telemetry resolver BOTH read from the
# repo's .env — the single source of truth this feature is built around. The
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
    # ZO_ROOT_USER_* are pinned to this test's constants so compose interpolates
    # the SAME credentials the computed Basic header uses, no matter what the
    # inherited process env carries. An earlier test in the same pytest process
    # can legitimately leave a foreign ZO_ROOT_USER_PASSWORD in os.environ
    # (``inject_provider_env`` copies every .env key into os.environ by
    # contract), and compose versions differ on whether the process env or the
    # ``--env-file`` wins ${VAR} interpolation — pinning here makes first-init
    # deterministic under either precedence.
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "CLAUDECODE": "",
            "ZO_ROOT_USER_EMAIL": OO_EMAIL,
            "ZO_ROOT_USER_PASSWORD": OO_PASSWORD,
        },
    )


def _remove_oo_data_volumes() -> None:
    """Remove this project's volumes, the OpenObserve data volume included.

    OpenObserve pins its root credentials on the volume's first init, so a volume
    left behind by an earlier deploy (including this fixture's own prior rerun
    attempt) would reject this test's credentials (401). Removing it guarantees a
    clean init. Exact-named and failure-tolerant — never a wildcard, never a
    prune; a missing/in-use volume is a no-op. The legacy directory-derived name
    is removed by exact name for dev machines that predate project-pinned compose
    namespacing — it carries no project label, so the shared sweep cannot see it.
    """
    remove_project_volumes(OO_PROJECT)
    subprocess.run(
        ["docker", "volume", "rm", OO_LEGACY_DATA_VOLUME],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _write_port_override(base: Path) -> Path:
    """Pin the openobserve host port in the SOURCE zone, for ``osprey init``.

    The service's compose file is rendered by ``osprey build`` and ``osprey up``
    deploys it as built, so the port has to be set on the profile the build
    reads — an edit to ``build/config.yml`` after the build never reaches the
    rendered compose. A dotted LEAF key on purpose: it sets only the port and
    leaves the preset's sibling ``services.openobserve`` settings (path,
    retention_days) intact, whereas a nested mapping would replace the subtree.
    """
    override_path = base / "override.yml"
    override_path.write_text(
        f"config:\n  services.openobserve.port: {OO_HOST_PORT}\n", encoding="utf-8"
    )
    return override_path


def _assert_render_deploys_openobserve(repo: Path) -> None:
    """Assert the build rendered the add-on this test probes, on the pinned port.

    The preset ships openobserve deployed by default, so the fixture opts into
    nothing — but a preset that stopped shipping it would otherwise deploy an
    empty stack and fail as an opaque health-wait timeout.

    ``pytest.fail``, not ``assert``: a render that does not carry the add-on is
    deterministic, and the rerun filter must not burn two more deploys on it.
    """
    config_path = repo / "build" / "config.yml"
    if not config_path.is_file():
        pytest.fail(f"rendered config.yml missing at {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if "openobserve" not in (config.get("deployed_services") or []):
        pytest.fail(f"render does not deploy openobserve: {config.get('deployed_services')!r}")

    compose_path = repo / "build" / "services" / "openobserve" / "docker-compose.yml"
    if not compose_path.is_file():
        pytest.fail(f"rendered openobserve compose missing at {compose_path}")
    ports = yaml.safe_load(compose_path.read_text())["services"]["openobserve"]["ports"]
    if not any(f":{OO_HOST_PORT}:" in str(port) for port in ports):
        pytest.fail(
            f"rendered compose does not publish the pinned host port {OO_HOST_PORT}: {ports!r}"
        )


def _write_credentials(repo: Path) -> None:
    """Write the OpenObserve root credentials to the repo's ``.env``."""
    # Credentials: single source of truth in the repo root's .env, read by BOTH
    # the compose service and the computed Basic header. Writing it here also
    # satisfies `osprey up`, which refuses to start a repo that has none.
    env_path = repo / ".env"
    existing = env_path.read_text() if env_path.is_file() else ""
    env_path.write_text(
        existing.rstrip("\n")
        + f"\nZO_ROOT_USER_EMAIL={OO_EMAIL}\nZO_ROOT_USER_PASSWORD={OO_PASSWORD}\n"
    )


@pytest.fixture(scope="module")
def deployed_openobserve(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Init + build + ``osprey up`` an openobserve-enabled repo; tear down after."""
    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("openobserve_e2e")
    repo = base / OO_PROJECT

    init = _run(
        [
            str(osprey_bin),
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--override",
            str(_write_port_override(base)),
        ],
        cwd=base,
        timeout=300,
    )
    if init.returncode != 0:
        pytest.fail(
            f"osprey init failed (rc={init.returncode}):\n"
            f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
        )

    build = _run(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        cwd=base,
        timeout=300,
    )
    if build.returncode != 0:
        pytest.fail(
            f"osprey build failed (rc={build.returncode}):\n"
            f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
        )

    _assert_render_deploys_openobserve(repo)
    # The .env is the repo's own zone (the build never touches it) and `osprey
    # up` refuses to start a repo that has none, so it only has to be in place
    # before the deploy — compose interpolates it at up time.
    _write_credentials(repo)

    # Guarantee a clean store: a volume left by an earlier deploy under this
    # project name (or the legacy ``services`` namespace) would pin foreign
    # credentials and 401 this test. Remove them before deploy so OpenObserve
    # initializes with THIS test's credentials (see _remove_oo_data_volumes).
    _remove_oo_data_volumes()

    try:
        up = _run(
            [str(osprey_bin), "up", "-d"],
            cwd=repo,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{OO_BASE_URL}/healthz", HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # ``osprey down`` keeps volumes; drop this project's own (legacy name
        # included) so this test never leaves foreign credentials pinned for a
        # later deploy (see _remove_oo_data_volumes).
        _remove_oo_data_volumes()


def _container_cred_diagnosis() -> str:
    """Report WHERE the deployed container's credentials came from, for 401
    triage: this test's constants, the compose-template default, or a foreign
    value (a leaked env var winning ``${VAR}`` interpolation over the
    ``--env-file``). Booleans only — never the secret itself.
    """
    probe = subprocess.run(
        [
            "docker",
            "exec",
            OO_CONTAINER,
            "printenv",
            "ZO_ROOT_USER_EMAIL",
            "ZO_ROOT_USER_PASSWORD",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if probe.returncode != 0:
        return f"cred diagnosis unavailable (docker exec rc={probe.returncode}: {probe.stderr!r})"
    lines = probe.stdout.splitlines()
    email = lines[0] if lines else ""
    password = lines[1] if len(lines) > 1 else ""
    return (
        f"container creds: email==test-constant {email == OO_EMAIL}, "
        f"password==test-constant {password == OO_PASSWORD}, "
        f"password==template-default {password == 'Complexpass#123'}"
    )


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


def _resolver_telemetry_env(user: str = OO_EMAIL, password: str = OO_PASSWORD) -> dict[str, str]:
    """Build the telemetry env via the REAL resolver helper, pinned at the deployed URL.

    The endpoint is set explicitly to the deployed host:port (the resolver's
    localhost/service-DNS derivation is unit-tested separately and would target
    :5080, not this test's pinned port). The Basic auth header, however, is the
    genuine value the resolver computes from the deployment's credentials — so a
    successful ingest proves OSPREY's real header authenticates.

    The credentials are arguments rather than the module constants because the
    same resolver serves two identities here: the root credential this fixture
    initializes the store with, and the ingest service account ``osprey up``
    provisions — which is what every shipped telemetry block names.
    """
    return _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "endpoint": f"{OO_BASE_URL}/api/{OO_ORG}",
            "openobserve": {"user": user, "password": password, "org": OO_ORG},
        },
        in_container=False,
    )


def _auth_header_from_resolver(user: str = OO_EMAIL, password: str = OO_PASSWORD) -> str:
    env = _resolver_telemetry_env(user, password)
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
        # Error bodies are not reliably JSON (e.g. a bare "Unauthorized Access"
        # string on 401) — surface them as data instead of crashing the test
        # with a JSONDecodeError that hides the real status code.
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body or "{}")
        except json.JSONDecodeError:
            return exc.code, {"_raw_error_body": body}


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
    # The header must decode back to the deployment's credentials (proves it is
    # the real computed value, not a placeholder).
    decoded = base64.b64decode(auth.removeprefix("Basic ")).decode()
    assert decoded == f"{OO_EMAIL}:{OO_PASSWORD}"

    now_ns = int(time.time() * 1_000_000_000)
    metric_status, metric_body = _otlp_post(
        f"/api/{OO_ORG}/v1/metrics", _synthetic_metric(now_ns), auth
    )
    assert metric_status == 200, (
        f"metric ingest rejected: {metric_status} {metric_body} — {_container_cred_diagnosis()}"
    )
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


# ---------------------------------------------------------------------------
# The ingest identity — what the shipped configs actually authenticate as
# ---------------------------------------------------------------------------


def _ingest_credentials(repo: Path) -> tuple[str, str]:
    """The ingest account name + token ``osprey up`` left in the repo's ``.env``.

    ``pytest.fail``, not ``assert``: an absent token means the provisioning step
    never ran (or failed), which is deterministic — rerunning would burn two
    more multi-minute deploys on the same outcome.
    """
    env = parse_dotenv_file(repo / ".env")
    email = env.get(INGEST_EMAIL_VAR, "")
    token = env.get(INGEST_TOKEN_VAR, "")
    if not token:
        pytest.fail(
            f"{INGEST_TOKEN_VAR} absent from {repo / '.env'} after `osprey up` — "
            f"the ingest identity was never provisioned. Keys present: {sorted(env)!r}"
        )
    if not email:
        pytest.fail(f"{INGEST_EMAIL_VAR} absent from {repo / '.env'} after `osprey up`")
    return email, token


def test_the_deploy_provisions_a_distinct_ingest_identity(deployed_openobserve: Path) -> None:
    """The token is the STORE's, not something OSPREY minted for itself.

    Pinned as "not the root password" rather than by format alone: the whole
    point of the service-account route is that the credential the agent carries
    is a different secret from the one that administers the store, so a
    provisioner that quietly reused the root password would satisfy every
    round-trip below and defeat the change entirely.
    """
    _email, token = _ingest_credentials(deployed_openobserve)

    assert token != OO_PASSWORD
    # Server-generated, and never handed to the root password's validator (8-128
    # chars, four character classes) — it would be refused.
    assert token.isalnum(), f"unexpected token shape: {len(token)} chars"


def test_synthetic_otlp_roundtrip_via_the_ingest_identity(deployed_openobserve: Path) -> None:
    """The round-trip the shipped telemetry blocks actually perform.

    Same synthetic payload and same REAL resolver as the root-credential
    round-trip above, with the one thing under test swapped: the credentials
    are the ones `osprey up` provisioned and wrote to ``.env``, read back from
    there rather than restated here.
    """
    email, token = _ingest_credentials(deployed_openobserve)
    auth = _auth_header_from_resolver(user=email, password=token)
    decoded = base64.b64decode(auth.removeprefix("Basic ")).decode()
    assert decoded == f"{email}:{token}"

    now_ns = int(time.time() * 1_000_000_000)
    metric_status, metric_body = _otlp_post(
        f"/api/{OO_ORG}/v1/metrics", _synthetic_metric(now_ns), auth
    )
    assert metric_status == 200, (
        f"ingest-identity metric rejected: {metric_status} {metric_body} — "
        f"{_container_cred_diagnosis()}"
    )
    event_status, event_body = _otlp_post(f"/api/{OO_ORG}/v1/logs", _synthetic_event(now_ns), auth)
    assert event_status == 200, f"ingest-identity event rejected: {event_status} {event_body}"

    start_us = (now_ns // 1_000) - 3_600_000_000
    end_us = (now_ns // 1_000) + 60_000_000
    deadline = time.monotonic() + INGEST_QUERY_TIMEOUT_SEC
    event_total = 0
    while time.monotonic() < deadline:
        # Queried with the SAME identity that wrote it: the account reads back
        # what it ingests, which is a documented property of this route (there
        # is no ingest-only role in any OpenObserve edition) and is what makes
        # this assertion possible without falling back to the root credential.
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
        if event_total >= 1:
            break
        time.sleep(2.0)

    assert event_total >= 1, "no record visible after ingest via the ingest identity"


def test_a_wrong_token_for_the_ingest_account_is_rejected(deployed_openobserve: Path) -> None:
    """The negative half, aimed at the identity that now carries the traffic.

    Without it, a store that had somehow stopped enforcing auth on the ingest
    account would pass every assertion above.
    """
    email, token = _ingest_credentials(deployed_openobserve)
    wrong = ("x" if not token.startswith("x") else "y") + token[1:]
    assert wrong != token

    bad = "Basic " + base64.b64encode(f"{email}:{wrong}".encode()).decode()
    status, _ = _otlp_post(f"/api/{OO_ORG}/v1/logs", _synthetic_event(1), bad)
    assert status in (401, 403), f"expected auth rejection, got {status}"


def test_the_rendered_config_names_the_ingest_identity(deployed_openobserve: Path) -> None:
    """The render is what the deployed agent reads, so pin it there and not only
    in the template: user carries its own fallback, the token carries none.

    A ``:-default`` on the token would be a literal credential shipped in every
    render — which is exactly what this deployment path exists to avoid.
    """
    config = yaml.safe_load((deployed_openobserve / "build" / "config.yml").read_text())
    openobserve = config["claude_code"]["telemetry"]["openobserve"]

    assert openobserve["user"] == f"${{{INGEST_EMAIL_VAR}:-ingest@example.com}}"
    assert openobserve["password"] == f"${{{INGEST_TOKEN_VAR}}}"


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
    repo = deployed_openobserve

    # Enable telemetry against the deployed store (host-run agent -> localhost)
    # and point the turn at the als-apg provider (the CI-provisioned key), then
    # drive one turn through the console script. The fixture build defaults to
    # provider: anthropic; this in-place edit to the render switches only the
    # live turn (`osprey query` reads the render, not the profile).
    config_path = repo / "build" / "config.yml"
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
        cwd=repo,
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
