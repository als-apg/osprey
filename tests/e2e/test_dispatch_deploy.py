"""Full-stack Docker e2e for event dispatch (L2) — highest fidelity.

Unlike the subprocess sweep (``test_dispatch_tutorial.py``), this exercises the
REAL shipped artifacts: the compose templates, the bundled Dockerfile (which
installs Node + the Claude Code CLI the worker needs), the worker ``env_file``
wiring that carries provider auth, and the in-network ``dispatch-worker-1:9190``
routing baked into the shipped ``tutorial_triggers.yml`` — none of which the
subprocess path touches.

It inits + builds a control-assistant deployment repo, deploys the stack with
``osprey up -d --dev``, fires all four tutorial webhooks at the dispatcher
(host-published on :8020), and asserts:

  * hello-dispatch / triage-event / save-report -> a run completes
  * denied-tool-demo -> rejected by the worker denylist (no completed run)

The same deploy also stands up the preset's multi-user web tier (local image
mode): the build renders one persona project per ``personas/`` delta into
``build/<repo>-<persona>/``, ``up`` builds one ``:local`` image per persona and
brings up nginx + one web-terminal container per roster user. Four more things
are asserted on top:

  T1 topology:      nginx + both per-user containers come up healthy and the
                    landing page lists both roster users.
  T2 readonly tier:  the rendered READ-ONLY persona project pins
                     ``control_system.writes_enabled: false`` and its rendered
                     ``settings.json`` denies the channel-write tool.
  T3 readwrite tier: the rendered READ-WRITE persona project arms writes and
                     keeps the tool on the ask (human-approval) path — the
                     positive control that T2 is a real posture difference.
  T4 same surface:   both persona projects declare the IDENTICAL ``.mcp.json``
                     server set — the tier boundary is enforcement, never a
                     quietly different tool surface.

COEXISTENCE: every host-published web port and the web-container name prefix
(``facility.prefix``) are remapped to e2e-unique values in the build override,
so this deploy can run beside a real control-assistant stack on the same host
without colliding on ports or container names. The persona ``:local`` image
tags are namespaced by the repo's own name (``osprey init`` writes each catalog
entry's ``project`` as ``<repo>-<persona>``), which keeps them off a real
deployment's tags for free. Teardown names exact resources only — never a
prune or wildcard.

The worker receives its provider key via ``env_file: ./.env`` (see the
dispatch_worker compose template) — the compose CLI reads the repo's ``.env``
on the HOST (as its owner, even when it's 0600) and injects the vars directly
into the container environment, so the non-root worker never has to open the
file itself. Without that wiring the agent run cannot authenticate, so this
test also guards it.

Gating: needs Docker and ``ALS_APG_API_KEY``. We do NOT gate on a host
``claude`` binary — the CLI lives inside the built image, not on the runner.
"""

from __future__ import annotations

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

from osprey.deployment.compose_generator import resolve_project_name, resolve_user_volume_names

DISPATCHER_URL = "http://localhost:8020"
TOKEN = "dev-token"  # matches the .env tokens written below

# Container image builds (Node + Claude CLI install; project + dispatch + two
# persona images) are slow on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 1800
HEALTH_TIMEOUT_SEC = 180.0
RUN_TIMEOUT_SEC = 300.0
CONTAINER_HEALTH_TIMEOUT_SEC = 180.0

# The deployment repo's directory name IS the deployment's name; compose renders
# each container_name as ``<project>-<service>`` (services/*/docker-compose.yml.j2),
# so derive the docker targets below rather than hardcode host-global names that
# break the moment the templates are namespaced per-project.
PROJECT_NAME = "proj"
DISPATCHER_CONTAINER = f"{PROJECT_NAME}-event-dispatcher"
WORKER_CONTAINER = f"{PROJECT_NAME}-dispatch-worker-1"
# Where the worker's agent-data volume mounts inside the project image: the
# repo's durable state zone, named by ``agent_data.base_dir`` in the config the
# compose generator renders the mount from.
WORKER_AGENT_DATA = f"/app/{PROJECT_NAME}/var/agent_data"

# ---------------------------------------------------------------------------
# Multi-user web tier (T1-T4). Prefix and ports are remapped off the preset
# defaults to e2e-unique values (see COEXISTENCE in the module docstring); the
# roster itself (alice→readonly, bob→readwrite) is the preset's own and is
# asserted, not configured, here. The persona project names are the repo's, not
# this test's: `osprey init` writes every catalog entry as
# ``<repo>-<persona>`` at ``build/<repo>-<persona>``, and `project` must equal
# `project_path`'s basename for the render to land where the deploy mounts it.
# ---------------------------------------------------------------------------
WEB_PREFIX = "dde"  # facility.prefix override: container names dde-nginx, dde-web-<user>
READONLY_USER = "alice"
READWRITE_USER = "bob"
READONLY_PROJECT = f"{PROJECT_NAME}-readonly"
READWRITE_PROJECT = f"{PROJECT_NAME}-readwrite"
READONLY_IMAGE = f"{READONLY_PROJECT}-readonly:local"
READWRITE_IMAGE = f"{READWRITE_PROJECT}-readwrite:local"

# The write tool the readonly tier's rendered settings.json must deny and the
# readwrite tier's must not (write tools land in permissions.deny when
# writes_enabled is false — see osprey.cli.templates.claude_code).
CHANNEL_WRITE_TOOL = "mcp__controls__channel_write"

# EVERY published web port is remapped off the preset defaults (nginx 9080,
# web 9091+) so this deploy coexists with a live control-assistant stack on
# the same host. Disjoint from every other deploy e2e's range too
# (test_deploy_lifecycle.py tops out ~20600).
WEB_PORTS = {
    "nginx": 20780,
    "web": 20800,
    "artifact": 20900,
    "ariel": 21000,
    "lattice": 21100,
    "channel_finder": 21200,
    "okf": 21300,
    "system_health": 21400,
}

# Probed from INSIDE the nginx container (docker exec + curl, which the
# container's own healthcheck already relies on), never from the host: with
# `network_mode: host` on Docker Desktop (macOS/Windows) the web stack binds
# inside the Docker Linux VM, so a host-side probe fails on any machine
# without the opt-in host-networking setting — while the in-container probe
# exercises the same nginx routing everywhere.
LANDING_URL = f"http://127.0.0.1:{WEB_PORTS['nginx']}/"


def _web_container(user: str) -> str:
    return f"{WEB_PREFIX}-web-{user}"


NGINX_CONTAINER = f"{WEB_PREFIX}-nginx"

# hello-dispatch / triage-event / save-report should complete; denied-tool-demo
# must be rejected by the server-side denylist.
_COMPLETING_TRIGGERS = ("hello-dispatch", "triage-event", "save-report")
_DENIED_TRIGGER = "denied-tool-demo"

_DEMO_PAYLOAD = {
    "signal": "demo:vacuum:pressure",
    "value": 4.2,
    "threshold": 3.0,
    "severity": "warning",
}

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_als_apg,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
    pytest.mark.flaky(reruns=1),
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


@pytest.fixture(scope="module")
def deployed_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Init + build + ``osprey up`` a control-assistant stack; tear down after."""
    if not os.environ.get("ALS_APG_API_KEY"):
        pytest.skip("ALS_APG_API_KEY not set")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("dispatch_deploy_build")
    repo = base / PROJECT_NAME

    # The preset's multi-user web tier deploys as shipped; only its
    # host-global identifiers are remapped to e2e-unique values (see
    # COEXISTENCE in the module docstring). Dotted LEAF keys on purpose --
    # each override sets only its own leaf and leaves the preset's
    # `modules.web_terminals` siblings (roster, catalog build_profiles,
    # image_source) intact, whereas a nested `modules:` mapping would
    # wholesale-replace the subtree (same convention as
    # tests/e2e/_orm_stack.py). The persona catalog's own `project` /
    # `project_path` are deliberately NOT overridden: `osprey init` writes them
    # from the repo's name, and the build renders each persona exactly there.
    override_path = base / "override.yml"
    override_lines = [
        "config:",
        f"  facility.prefix: {WEB_PREFIX}",
        f"  modules.web_terminals.nginx_port: {WEB_PORTS['nginx']}",
        f"  modules.web_terminals.web_base_port: {WEB_PORTS['web']}",
        f"  modules.web_terminals.artifact_base_port: {WEB_PORTS['artifact']}",
        f"  modules.web_terminals.ariel_base_port: {WEB_PORTS['ariel']}",
        f"  modules.web_terminals.lattice_base_port: {WEB_PORTS['lattice']}",
        f"  modules.web_terminals.channel_finder_base_port: {WEB_PORTS['channel_finder']}",
        f"  modules.web_terminals.okf_base_port: {WEB_PORTS['okf']}",
        f"  modules.web_terminals.system_health_base_port: {WEB_PORTS['system_health']}",
        "",
    ]
    override_path.write_text("\n".join(override_lines), encoding="utf-8")

    # Two steps, because the surface has two: `init` writes the repo's source
    # zone from the preset, `build` renders build/ from it — including one
    # persona project per `personas/` delta.
    init = _run(
        [
            str(osprey_bin),
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--override",
            str(override_path),
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
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

    # The repo root's .env is the deployment's whole secret store — the file the
    # worker's env_file (compose template) delivers to the container so
    # inject_provider_env can resolve the provider key, and the file `osprey up`
    # refuses to start without. The compose templates have no token default
    # (they fail closed), and `up` would otherwise auto-generate a random token;
    # we write fixed tokens here so the bearer below is predictable, and pass the
    # provider secret through. The endpoint override rides the same .env: without
    # it the container-side config expansion falls back to the provider's
    # built-in default gateway.
    base_url_line = (
        f"ALS_APG_BASE_URL={os.environ['ALS_APG_BASE_URL']}\n"
        if os.environ.get("ALS_APG_BASE_URL")
        else ""
    )
    (repo / ".env").write_text(
        "EVENT_DISPATCHER_TOKEN=dev-token\n"
        "DISPATCH_WORKER_TOKEN=dev-token\n"
        f"ALS_APG_API_KEY={os.environ['ALS_APG_API_KEY']}\n" + base_url_line,
        encoding="utf-8",
    )

    # Force a fresh image build so the deployed services run CURRENT source. The
    # dispatcher runs <project>-dispatch:local; the worker now runs the unified
    # project image <project>:local (built by `up --dev` from the repo root).
    # Both tags are project-prefixed, derived via resolve_project_name exactly as
    # the compose templates / project build do.
    # `osprey up` does not pass --build to compose, so it would otherwise reuse
    # existing images and silently test stale code. The freshly-built dev wheel
    # invalidates the relevant build-cache layers on rebuild.
    project = resolve_project_name({"project_name": PROJECT_NAME})
    for image in (f"{project}-dispatch:local", f"{project}:local"):
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)

    # The web-container names are host-global fixed identifiers, so a crashed
    # prior run could have left the exact-named containers behind. Remove them
    # by exact name before deploy so this run never adopts a stale container
    # (guardrail: exact-named only, never a prune or wildcard).
    for user in (READONLY_USER, READWRITE_USER):
        subprocess.run(["docker", "rm", "-f", _web_container(user)], capture_output=True, text=True)
    subprocess.run(["docker", "rm", "-f", NGINX_CONTAINER], capture_output=True, text=True)

    try:
        up = _run(
            [str(osprey_bin), "up", "-d", "--dev"],
            cwd=repo,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(
                f"osprey up failed (rc={up.returncode}):\n"
                f"--- stdout ---\n{up.stdout}\n--- stderr ---\n{up.stderr}"
            )
        _wait_for_health(f"{DISPATCHER_URL}/health", HEALTH_TIMEOUT_SEC)
        # The dispatcher being healthy does not mean the worker is: the worker runs
        # the heavier project image and boots later. Gate on its proxied feed so
        # tests never race a still-warming worker (and a worker that never comes up
        # fails here with container logs, not as a bare 502 mid-test).
        _wait_for_worker_feed(HEALTH_TIMEOUT_SEC)
        yield repo
    finally:
        # `osprey down` tears down the services stack AND the web stack
        # (deploy_down_web_terminals); the exact-named sweeps after it are
        # belt-and-suspenders for a down that failed midway. Volumes are
        # deliberately kept by `down`, so the per-user volumes this run created
        # are removed by exact name; the persona image tags are namespaced by
        # this repo's name (see COEXISTENCE), so removing them cannot untag a
        # real deployment's images.
        down = _run([str(osprey_bin), "down"], cwd=repo, timeout=300)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in CI logs
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        for user in (READONLY_USER, READWRITE_USER):
            subprocess.run(
                ["docker", "rm", "-f", _web_container(user)], capture_output=True, text=True
            )
            for volume in resolve_user_volume_names({"project_name": PROJECT_NAME}, user):
                subprocess.run(["docker", "volume", "rm", volume], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", NGINX_CONTAINER], capture_output=True, text=True)
        for image in (READONLY_IMAGE, READWRITE_IMAGE):
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True, text=True)


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


def _dump_stack_diagnostics(context: str) -> str:
    """Return container state + recent worker/dispatcher logs for failure output.

    A full-Docker e2e that fails mid-run is nearly undiagnosable from the pytest
    traceback alone (the interesting state is inside the containers). Surfacing
    ``docker ps`` plus the tail of both service logs turns an opaque ``502`` into
    an actionable report (e.g. the worker crash-looping vs. merely slow to boot).
    """
    lines = [f"=== stack diagnostics ({context}) ==="]
    ps = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name={PROJECT_NAME}-",
            "--format",
            "{{.Names}}\t{{.Status}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines.append("containers:\n" + (ps.stdout.strip() or ps.stderr.strip() or "(none)"))
    for name in (WORKER_CONTAINER, DISPATCHER_CONTAINER):
        logs = subprocess.run(
            ["docker", "logs", "--tail", "40", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        tail = ((logs.stdout or "") + (logs.stderr or "")).strip() or "(no logs)"
        lines.append(f"--- {name} (last 40 log lines) ---\n{tail}")
    return "\n".join(lines)


def _wait_for_worker_feed(timeout: float) -> None:
    """Wait until the dispatcher can proxy the worker run feed (HTTP 200).

    The fixture's ``/health`` gate proves only the DISPATCHER is up, but every run
    assertion reads ``/dashboard/runs``, which the dispatcher proxies to the
    worker. The worker runs the full project image and can take appreciably longer
    to become ready than the dispatcher — until it does, the proxy returns ``502``.
    Gate on a ``200`` here so the test never races a still-warming worker, and a
    genuine worker startup failure surfaces here (with container logs) instead of
    as a bare ``502`` mid-test.
    """
    deadline = time.monotonic() + timeout
    last = "(no response yet)"
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
                last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
        time.sleep(2.0)
    raise AssertionError(
        f"worker feed at {DISPATCHER_URL}/dashboard/runs not ready after "
        f"{timeout:.0f}s (last: {last})\n{_dump_stack_diagnostics('worker feed wait timeout')}"
    )


def _fire(trigger: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - localhost only
        f"{DISPATCHER_URL}/webhook/{trigger}",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
        assert resp.status == 202, f"{trigger}: expected 202 from webhook, got {resp.status}"
        fired = json.loads(resp.read().decode("utf-8"))
    assert fired.get("dispatched") is True, f"{trigger}: {fired}"


def _worker_artifact_files() -> list[str]:
    """List artifact files the worker persisted to its workspace volume.

    Read directly from the worker container because the dispatcher run feed
    reports only status/tool counts, not whether a real artifact landed. This is
    what distinguishes a genuine ``save-report`` (which must persist via the
    ``mcp__osprey_workspace__`` artifact tool) from a hollow "completed" run that
    only claimed success — see the assertion in ``test_full_stack_dispatch``.
    """
    proc = subprocess.run(
        ["docker", "exec", WORKER_CONTAINER, "ls", f"{WORKER_AGENT_DATA}/artifacts"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.split() if line.endswith(".md")]


def _runs_by_trigger() -> dict[str, dict]:
    """Snapshot the dispatcher run feed keyed by trigger_name (latest wins).

    The dispatcher's /dashboard/runs proxies the worker feed and enriches each
    run with the trigger_name that produced it. It is a bearer-gated read endpoint,
    so the snapshot must send the same EVENT_DISPATCHER_TOKEN written to .env above.
    """
    req = urllib.request.Request(  # noqa: S310
        f"{DISPATCHER_URL}/dashboard/runs",
        method="GET",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            runs = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # A transient gateway error means the dispatcher momentarily could not
        # reach the worker upstream (e.g. mid-restart). Treat it as "no feed yet"
        # so the caller's poll loop retries instead of aborting the whole run.
        if exc.code in (502, 503, 504):
            return {}
        raise
    except (urllib.error.URLError, ConnectionError):
        return {}
    by_trigger: dict[str, dict] = {}
    for run in runs:
        name = run.get("trigger_name")
        if name:
            by_trigger[name] = run
    return by_trigger


def test_full_stack_dispatch(deployed_stack: Path) -> None:
    """All four shipped triggers behave correctly through the real Docker stack."""
    # Fire the three completing triggers + the denied one.
    _fire("hello-dispatch", {})
    _fire("triage-event", _DEMO_PAYLOAD)
    _fire("save-report", _DEMO_PAYLOAD)
    _fire(_DENIED_TRIGGER, {})

    # Poll until each completing trigger has a terminal run.
    deadline = time.monotonic() + RUN_TIMEOUT_SEC
    by_trigger: dict[str, dict] = {}
    while time.monotonic() < deadline:
        by_trigger = _runs_by_trigger()
        if all(
            by_trigger.get(t, {}).get("status") in ("completed", "error")
            for t in _COMPLETING_TRIGGERS
        ):
            break
        time.sleep(3.0)

    # If the feed never converged, attach container state so a CI failure reads as
    # "worker crashed / never produced the run" rather than an opaque status miss.
    converged = all(
        by_trigger.get(t, {}).get("status") in ("completed", "error") for t in _COMPLETING_TRIGGERS
    )
    diag = "" if converged else "\n" + _dump_stack_diagnostics("run feed did not converge")

    for trigger in _COMPLETING_TRIGGERS:
        run = by_trigger.get(trigger)
        assert run is not None, (
            f"{trigger}: no run appeared in the dispatcher feed: {by_trigger}{diag}"
        )
        assert run.get("status") == "completed", (
            f"{trigger}: expected completed, got status={run.get('status')!r} "
            f"error={run.get('error')!r}{diag}"
        )

    # save-report must persist via the workspace artifact tool, not merely report
    # "completed". Without the worker's startup artifact provisioning, .mcp.json is
    # absent, mcp__osprey_workspace__artifact_save does not exist, the agent's
    # Write fallback is denied by the allowlist, and the run hollow-completes with
    # no artifact on disk. Asserting a real .md artifact landed guards that path.
    artifacts = _worker_artifact_files()
    assert artifacts, (
        "save-report completed but no .md artifact was persisted to the worker "
        "workspace — the osprey_workspace MCP server is likely not provisioned "
        "in-container (missing .mcp.json), so the agent could not actually save"
    )

    # The denylisted trigger is rejected at the worker /dispatch endpoint BEFORE
    # any run record is created, so it must never surface as a completed run.
    denied = by_trigger.get(_DENIED_TRIGGER)
    assert denied is None or denied.get("status") != "completed", (
        f"denied-tool-demo should be rejected by the denylist, not completed: {denied!r}"
    )


# ---------------------------------------------------------------------------
# Multi-user web tier (T1-T4) — the same deploy stands up nginx + one terminal
# container per roster user from the preset's persona catalog. T1 exercises the
# running topology; T2-T4 are render-layer assertions on the persona projects
# `osprey build` rendered into the repo's build/ zone.
# ---------------------------------------------------------------------------


def _inspect_container(name: str, fmt: str) -> str | None:
    result = subprocess.run(
        ["docker", "inspect", "--type", "container", "-f", fmt, name],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``.State.Health.Status`` until ``healthy`` or timeout.

    A container can be created and serving before Docker flips its healthcheck
    STATUS off ``starting`` (the healthcheck runs only on its interval, after
    ``start_period``), so an instant equality assert is racy.
    """
    deadline = time.monotonic() + timeout
    last = "(no status yet)"
    while time.monotonic() < deadline:
        if _inspect_container(container, "{{.Id}}") is None:
            last = "(container not present)"
        else:
            last = _inspect_container(container, "{{.State.Health.Status}}") or "(no health field)"
            if last == "healthy":
                return
        time.sleep(2.0)
    raise AssertionError(
        f"{container} did not reach 'healthy' within {timeout:.0f}s (last status: {last!r})"
    )


def _fetch_in_container(container: str, url: str, timeout: float) -> str:
    """Poll ``url`` from INSIDE ``container`` (docker exec + curl) until HTTP
    200 (or timeout); return the response body.

    curl is guaranteed present — the container's own compose healthcheck uses
    it. Probing in-container keeps the check independent of Docker Desktop's
    host-networking setting (see the LANDING_URL comment).
    """
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", container, "curl", "-fsS", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout
        last_err = (result.stderr or result.stdout).strip() or f"rc={result.returncode}"
        time.sleep(1.0)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} in {container} (last: {last_err})"
    )


def _persona_dir(repo: Path, persona_project: str) -> Path:
    """The rendered persona project directory.

    The catalog pins each ``project_path`` to ``build/<repo>-<persona>``, which
    is where the build renders it: a persona project is build output like every
    other render, so it lives in the repo's OUTPUT zone.
    """
    persona_dir = repo / "build" / persona_project
    assert persona_dir.is_dir(), (
        f"persona project {persona_project!r} was not rendered at {persona_dir}"
    )
    return persona_dir


def _writes_enabled(render: Path) -> bool:
    config = yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))
    return bool((config.get("control_system") or {}).get("writes_enabled", False))


def _permissions(render: Path) -> dict:
    settings_path = render / ".claude" / "settings.json"
    assert settings_path.is_file(), f"no settings.json rendered at {settings_path}"
    return json.loads(settings_path.read_text(encoding="utf-8")).get("permissions", {})


def test_web_tier_topology(deployed_stack: Path) -> None:
    """T1: nginx + both per-user containers come up healthy off the overridden
    prefix, both persona images were built locally, and the landing page lists
    both roster users."""
    config = yaml.safe_load((deployed_stack / "build" / "config.yml").read_text(encoding="utf-8"))
    prefix = ((config.get("facility") or {}).get("prefix") or "").strip()
    assert prefix == WEB_PREFIX, (
        f"facility.prefix override did not land in the rendered config: {prefix!r}"
    )

    expected = [NGINX_CONTAINER, _web_container(READONLY_USER), _web_container(READWRITE_USER)]
    missing = [name for name in expected if _inspect_container(name, "{{.Id}}") is None]
    assert not missing, f"container(s) not created by 'osprey up': {missing}"
    for name in expected:
        _wait_for_container_health(name, CONTAINER_HEALTH_TIMEOUT_SEC)

    # Both per-user images were built locally as `<persona-project>-<persona>:local`
    # (build_persona_images) — the readonly and readwrite tiers are genuinely two
    # distinct images, the core promise of the local-mode persona build.
    for image in (READONLY_IMAGE, READWRITE_IMAGE):
        inspect = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=15
        )
        assert inspect.returncode == 0, f"persona image {image} was not built by 'osprey up'"

    landing = _fetch_in_container(NGINX_CONTAINER, LANDING_URL, HEALTH_TIMEOUT_SEC)
    for user in (READONLY_USER, READWRITE_USER):
        assert user in landing, f"landing page does not list roster user {user!r}"


def test_readonly_persona_denies_writes(deployed_stack: Path) -> None:
    """T2: the rendered read-only persona project pins writes off and its rendered
    settings.json carries the channel-write tool in permissions.deny — the
    render-time layer that actually enforces the read-only posture. The
    deployment's provider must also have reached the render (the up-time
    credential preflight already depends on it)."""
    readonly_dir = _persona_dir(deployed_stack, READONLY_PROJECT)
    config = yaml.safe_load((readonly_dir / "config.yml").read_text(encoding="utf-8"))
    assert (config.get("claude_code") or {}).get("provider") == "als-apg", (
        "the deployment's `--set provider=als-apg` did not reach the rendered persona project"
    )
    assert _writes_enabled(readonly_dir) is False, (
        "readonly tier must render control_system.writes_enabled: false"
    )
    perms = _permissions(readonly_dir)
    assert CHANNEL_WRITE_TOOL in perms.get("deny", []), (
        f"readonly tier must deny {CHANNEL_WRITE_TOOL!r} in rendered settings.json, "
        f"but deny list is: {perms.get('deny', [])}"
    )


def test_readwrite_persona_arms_writes(deployed_stack: Path) -> None:
    """T3: positive control for T2 — the same render pipeline does NOT deny the
    write tool for the readwrite tier (so T2's deny is a real posture
    difference, not a render that denied the tool everywhere), and the write
    path stays supervised: the tool remains on the ask (human-approval) path."""
    readwrite_dir = _persona_dir(deployed_stack, READWRITE_PROJECT)
    assert _writes_enabled(readwrite_dir) is True, (
        "readwrite tier must render control_system.writes_enabled: true"
    )
    perms = _permissions(readwrite_dir)
    assert CHANNEL_WRITE_TOOL not in perms.get("deny", []), (
        f"readwrite tier must not deny {CHANNEL_WRITE_TOOL!r}, but deny list is: "
        f"{perms.get('deny', [])}"
    )
    assert CHANNEL_WRITE_TOOL in perms.get("ask", []), (
        f"readwrite tier must keep {CHANNEL_WRITE_TOOL!r} on the ask path, but ask "
        f"list is: {perms.get('ask', [])}"
    )


def test_tiers_share_identical_mcp_surface(deployed_stack: Path) -> None:
    """T4: both persona projects declare the IDENTICAL ``.mcp.json`` server set
    — the tier boundary is enforcement (``writes_enabled``), never a quietly
    different tool surface."""

    def mcp_server_keys(render: Path) -> set[str]:
        mcp_path = render / ".mcp.json"
        assert mcp_path.is_file(), f"no .mcp.json rendered at {mcp_path}"
        return set(json.loads(mcp_path.read_text(encoding="utf-8")).get("mcpServers", {}))

    readonly_keys = mcp_server_keys(_persona_dir(deployed_stack, READONLY_PROJECT))
    readwrite_keys = mcp_server_keys(_persona_dir(deployed_stack, READWRITE_PROJECT))
    assert readonly_keys == readwrite_keys, (
        "the two tiers must declare the identical MCP server set (the boundary "
        f"is writes_enabled, not tool absence): readonly={sorted(readonly_keys)} "
        f"readwrite={sorted(readwrite_keys)}"
    )
