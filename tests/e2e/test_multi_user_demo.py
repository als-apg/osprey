"""Acceptance proof: the ``multi-user-demo`` topology stands itself up from the
preset with a single ``osprey build`` + ``osprey deploy up``, and the two
personas it ships are genuinely different write postures over one deployment.

The ``multi-user-demo`` preset ships a ``modules.web_terminals`` block (a
read-only tier + a read-write tier, roster ``alice``/``bob``, an nginx reverse
proxy) and is deliberately scan-free — no Bluesky bridge, no Virtual
Accelerator, no panels sidecar. ``osprey deploy up`` in local image mode is
supposed to, from nothing but the built project: auto-render each referenced
persona project (``osprey build ... --skip-deps``), build both persona
``:local`` images, and bring up nginx + one web-terminal container per user.
This test drives that whole promise against a real container runtime and
asserts four things end to end:

  T1 topology:      nginx + both per-user containers come up healthy,
                     exact-named off the deploy's OWN ``facility.prefix`` (read
                     from the rendered config, never hardcoded) and project
                     name, and the landing page lists both roster users.
  T2 readonly tier:  the auto-rendered READ-ONLY project pins
                     ``control_system.writes_enabled: false`` and its rendered
                     ``settings.json`` carries the channel-write tool in
                     ``permissions.deny`` — the render-time layer that actually
                     enforces the read-only posture.
  T3 readwrite tier: the auto-rendered READ-WRITE project arms
                     ``writes_enabled: true`` and its deny list does NOT carry
                     the channel-write tool (it stays on the ``ask`` /
                     human-approval path) — the positive control that T2 is a
                     real posture difference, not a render that denied the tool
                     everywhere.
  T4 same surface:   both persona projects declare the IDENTICAL ``.mcp.json``
                     server set — the tier boundary is enforcement
                     (``writes_enabled``), never a quietly different tool
                     surface.

CONTAINER-OPS SAFETY: every runtime-mutating call names an EXACT resource this
test created — the ``<prefix>-nginx`` / ``<prefix>-web-<user>`` web containers
(prefix read from config), the ``<project>_<user>-*`` volumes, and the
``:local`` persona images — or is the project-scoped ``osprey deploy nuke`` /
``compose -p <project> down`` teardown the lifecycle code itself uses
(label-scoped, containers/networks only). Nothing here ever runs a prune, an
``-a``/``--all`` sweep, a ``volume prune``, or a wildcard container match.
Teardown runs from a fixture ``finally`` so a failed assertion mid-sequence
still leaves nothing stranded; the web-container names (whose prefix is a
shared preset default, not unique per run) are exact-name pre-cleaned before
deploy so a crashed prior run can never be adopted.

Coexistence with other osprey deploys on the same host is safe by design:
every ``osprey deploy`` compose invocation pins ``COMPOSE_PROJECT_NAME`` to its
own resolved project name, so this deploy's backend services live under the
``mudemo-e2e`` compose project and can never adopt or recreate another
deployment's containers or volumes. Together with the remapped ports above,
this test runs alongside live foreign osprey stacks; only the web containers
collide (shared preset ``facility.prefix``) and are exact-name pre-cleaned.

Gating: needs a container runtime whose daemon is actually responding
(``docker`` by default; ``OSPREY_E2E_RUNTIME=podman`` opt-in, matching the
sibling deploy e2e). Skipped cleanly otherwise. Slow: ``deploy up`` builds two
full persona images (minutes on a cold cache). Lives in ``tests/e2e/`` (never
collected by the fast lane).

Markers: ``e2e`` / ``slow`` / ``dockerbuild``, exactly like the sibling
``tests/e2e/test_deploy_lifecycle.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_user_volume_names

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.dockerbuild,
]

# ---------------------------------------------------------------------------
# Container runtime selection — ``docker`` by default, ``podman`` opt-in via
# OSPREY_E2E_RUNTIME (matching tests/e2e/test_deploy_lifecycle.py). Any other
# value fails at collection time rather than silently falling back to docker.
# ---------------------------------------------------------------------------
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# This interpreter's bin dir — prepended to PATH for every `osprey` subprocess so
# nested bare-`osprey` calls (persona auto-render) resolve to the venv's osprey.
_OSPREY_BIN_DIR = str(Path(sys.executable).parent)

# ---------------------------------------------------------------------------
# This deploy's identity. The project name (compose project / volume namespace)
# is unique to this file so it never collides with another deploy e2e. The
# web-container prefix is NOT set here — it is the preset's own
# `facility.prefix`, read from the rendered config at runtime (see
# _facility_prefix) so a future prefix change can't silently drift this test's
# expected names.
# ---------------------------------------------------------------------------
PROJECT_NAME = "mudemo-e2e"
DEPLOY_FQDN = "localhost"

# Roster shipped by the preset: alice -> readonly (default_persona), bob ->
# readwrite. Asserted, not configured, here — the preset owns the roster.
READONLY_USER = "alice"
READWRITE_USER = "bob"

# Persona catalog entries the preset ships (project name -> persona key). The
# auto-rendered project directory basename equals the catalog `project`, and the
# built image tag is `<project>-<persona>:local` (build_persona_images).
READONLY_PROJECT = "multi-user-demo-readonly"
READWRITE_PROJECT = "multi-user-demo-readwrite"
READONLY_IMAGE = f"{READONLY_PROJECT}-readonly:local"
READWRITE_IMAGE = f"{READWRITE_PROJECT}-readwrite:local"

# The write tool the readonly tier's rendered settings.json must deny and the
# readwrite tier's must not (see osprey.cli.templates.claude_code — write tools
# land in permissions.deny when writes_enabled is false).
CHANNEL_WRITE_TOOL = "mcp__controls__channel_write"

# EVERY published host port is remapped off the multi-user-demo preset's
# defaults to a unique range, so this deploy coexists with an already-running
# demo on the same host (which occupies the DEFAULT ports — nginx 9080, web
# 9091+, OpenObserve 5080, Postgres 5432). Without this, `deploy up` fails at
# compose-up with "port is already allocated". Web ports are disjoint from
# every other deploy e2e's range too (test_deploy_lifecycle.py tops out ~20600).
WEB_PORTS = {
    "nginx": 20780,
    "web": 20800,
    "artifact": 20900,
    "ariel": 21000,
    "lattice": 21100,
    "channel_finder": 21200,
}
OPENOBSERVE_PORT = 25080
POSTGRES_PORT = 25432

# Probed from INSIDE the nginx container (docker exec + curl, which the
# container's own healthcheck already relies on), never from the host: with
# `network_mode: host` on Docker Desktop (macOS/Windows) the web stack binds
# inside the Docker Linux VM, so a host-side probe fails on any machine
# without the opt-in host-networking setting — while the in-container probe
# exercises the same nginx routing everywhere.
LANDING_URL = f"http://127.0.0.1:{WEB_PORTS['nginx']}/"

BUILD_TIMEOUT_SEC = 300
# deploy up builds TWO full persona images — minutes each on a cold cache.
DEPLOY_UP_TIMEOUT_SEC = 1800
DEPLOY_DOWN_TIMEOUT_SEC = 300
HEALTH_TIMEOUT_SEC = 300.0
CONTAINER_HEALTH_TIMEOUT_SEC = 180.0


# ---------------------------------------------------------------------------
# Low-level helpers (shape mirrors tests/e2e/test_deploy_lifecycle.py)
# ---------------------------------------------------------------------------


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
        # CLAUDECODE="" so `osprey` never mistakes the pytest process for an
        # in-agent invocation; CONTAINER_RUNTIME pins the deploy-side runtime to
        # the same one _runtime_cli asserts against. PATH is prepended with this
        # interpreter's own bin dir so that `deploy up`'s persona auto-render —
        # which spawns a BARE `osprey build ...` subprocess
        # (web_terminals.provision._auto_render_missing_personas) inheriting this
        # env — resolves the SAME osprey the test drives (the venv's, which
        # knows the persona presets), not a stale `osprey` earlier on the
        # ambient PATH. Equivalent to running under an activated venv.
        env={
            **os.environ,
            "CLAUDECODE": "",
            "CONTAINER_RUNTIME": RUNTIME,
            "PATH": _OSPREY_BIN_DIR + os.pathsep + os.environ.get("PATH", ""),
        },
    )


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _runtime_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([RUNTIME, *args], capture_output=True, text=True, timeout=timeout)


def _container_id(name: str) -> str | None:
    result = _runtime_cli("inspect", "--type", "container", "-f", "{{.Id}}", name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _inspect(name: str, fmt: str) -> str | None:
    result = _runtime_cli("inspect", "--type", "container", "-f", fmt, name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _web_container(prefix: str, user: str) -> str:
    return f"{prefix}-web-{user}"


def _nginx_container(prefix: str) -> str:
    return f"{prefix}-nginx"


def _facility_prefix(project_dir: Path) -> str:
    """Read ``facility.prefix`` from the DEPLOYED project's rendered config.yml.

    Derived at runtime (never hardcoded) so a future preset prefix change can't
    silently drift the expected web-container names. A non-empty prefix is an
    invariant the preset guarantees (an empty one renders invalid ``-nginx``
    container names), so assert it here rather than paper over a regression.
    """
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    prefix = ((config.get("facility") or {}).get("prefix") or "").strip()
    assert prefix, (
        "facility.prefix is empty in the rendered config.yml — the preset must "
        "ship a non-empty prefix or every web container name is invalid"
    )
    return prefix


def _wait_for_container_health(container: str, timeout: float) -> None:
    """Poll ``.State.Health.Status`` until ``healthy`` or timeout.

    A container can be created and serving before Docker flips its healthcheck
    STATUS off ``starting`` (the healthcheck runs only on its interval, after
    ``start_period``), so an instant equality assert is racy.
    """
    deadline = time.monotonic() + timeout
    last = "(no status yet)"
    while time.monotonic() < deadline:
        if _container_id(container) is None:
            last = "(container not present)"
        else:
            last = _inspect(container, "{{.State.Health.Status}}") or "(no health field)"
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
        result = _runtime_cli("exec", container, "curl", "-fsS", url, timeout=15)
        if result.returncode == 0:
            return result.stdout
        last_err = (result.stderr or result.stdout).strip() or f"rc={result.returncode}"
        time.sleep(1.0)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} in {container} (last: {last_err})"
    )


# ---------------------------------------------------------------------------
# Deployed-stack fixture
# ---------------------------------------------------------------------------


class DemoStack:
    """Everything the T1-T4 tests need about the one co-deployed demo project."""

    def __init__(
        self,
        project_dir: Path,
        readonly_dir: Path,
        readwrite_dir: Path,
        prefix: str,
    ):
        self.project_dir = project_dir
        self.readonly_dir = readonly_dir
        self.readwrite_dir = readwrite_dir
        self.prefix = prefix


def _teardown(osprey_bin: Path, project_dir: Path, prefix: str | None) -> None:
    """Exact-named, best-effort teardown of everything this deploy created.

    ``osprey deploy nuke`` is the project's own label-scoped teardown (containers
    + volumes + label-verified persona images for THIS project only); the
    remaining calls are an exact-named belt-and-suspenders sweep — one exact
    container, one exact volume, one exact image tag, or the project-scoped
    ``compose -p <project> down`` the guardrail explicitly allows. Never a prune,
    ``-a``/``--all``, ``volume prune``, or wildcard. Failures (already gone,
    never created) are swallowed: this is a safety net, not an assertion.

    A hard SIGKILL of the pytest process (rare) bypasses this fixture finalizer
    entirely; the exact-named pre-clean at the top of ``demo_stack`` is the
    backstop that reclaims any web container a killed prior run left behind.
    """
    if project_dir.exists():
        nuke = _run(
            [str(osprey_bin), "deploy", "nuke", "--yes"], project_dir, DEPLOY_DOWN_TIMEOUT_SEC
        )
        if nuke.returncode != 0:
            print(_fmt("deploy nuke (teardown)", nuke))  # noqa: T201 - surface in CI logs

    if prefix:
        for user in (READONLY_USER, READWRITE_USER):
            _runtime_cli("rm", "-f", _web_container(prefix, user))
        _runtime_cli("rm", "-f", _nginx_container(prefix))
    _runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=DEPLOY_DOWN_TIMEOUT_SEC)
    for user in (READONLY_USER, READWRITE_USER):
        for volume in resolve_user_volume_names({"project_name": PROJECT_NAME}, user):
            _runtime_cli("volume", "rm", volume)
    for image in (READONLY_IMAGE, READWRITE_IMAGE):
        _runtime_cli("rmi", "-f", image)


@pytest.fixture(scope="module")
def demo_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DemoStack]:
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("mu_demo_build")
    project_dir = base / PROJECT_NAME
    # The preset pins each persona's project_path to `../<persona-project>`, so
    # auto-render lands them as siblings of the main project dir under `base`.
    readonly_dir = base / READONLY_PROJECT
    readwrite_dir = base / READWRITE_PROJECT

    # Supply the deploy-time landing origin the web-terminal render requires
    # (deploy.fqdn) plus this deploy's unique ports. Written as flat dotted
    # `config:` keys (the preset's own convention) so each sets only its leaf —
    # a nested `config:` mapping would deep-merge and clobber the preset's
    # rendered subtrees. facility.prefix is deliberately NOT overridden — the
    # preset ships it, and the test reads it back from the rendered config
    # (see _facility_prefix).
    override_path = base / "override.yml"
    override_lines = [
        "config:",
        f"  deploy.fqdn: {DEPLOY_FQDN}",
        f"  services.openobserve.port: {OPENOBSERVE_PORT}",
        f"  services.postgresql.port_host: {POSTGRES_PORT}",
        f"  modules.web_terminals.nginx_port: {WEB_PORTS['nginx']}",
        f"  modules.web_terminals.web_base_port: {WEB_PORTS['web']}",
        f"  modules.web_terminals.artifact_base_port: {WEB_PORTS['artifact']}",
        f"  modules.web_terminals.ariel_base_port: {WEB_PORTS['ariel']}",
        f"  modules.web_terminals.lattice_base_port: {WEB_PORTS['lattice']}",
        f"  modules.web_terminals.channel_finder_base_port: {WEB_PORTS['channel_finder']}",
        "",
    ]
    override_path.write_text("\n".join(override_lines), encoding="utf-8")

    build = _run(
        [
            str(osprey_bin),
            "build",
            PROJECT_NAME,
            "--preset",
            "multi-user-demo",
            "--override",
            str(override_path),
            "--skip-deps",
            "--skip-lifecycle",
            "--output-dir",
            str(base),
            "--force",
        ],
        cwd=base,
        timeout=BUILD_TIMEOUT_SEC,
    )
    if build.returncode != 0:
        pytest.fail(_fmt("osprey build", build))

    # The deploy-up credential preflight requires each persona's LLM key in .env
    # before it will generate .env.production; this test never sends an
    # authenticated prompt through the web terminals (it checks topology and
    # rendered posture only), so a placeholder satisfies the gate.
    env_path = project_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    env_path.write_text(existing + "ANTHROPIC_API_KEY=fake-llm-key-value\n", encoding="utf-8")

    prefix = _facility_prefix(project_dir)

    # The web-container names use the preset's shared `facility.prefix` (not this
    # test's unique project name), so a crashed prior run could have left the
    # exact-named containers behind. Remove them by exact name before deploy so
    # this run never adopts a stale container (guardrail: exact-named only).
    for user in (READONLY_USER, READWRITE_USER):
        _runtime_cli("rm", "-f", _web_container(prefix, user))
    _runtime_cli("rm", "-f", _nginx_container(prefix))

    try:
        up = _run(
            [str(osprey_bin), "deploy", "up", "-d", "--dev"],
            cwd=project_dir,
            timeout=DEPLOY_UP_TIMEOUT_SEC,
        )
        if up.returncode != 0:
            pytest.fail(_fmt("osprey deploy up -d --dev", up))
        yield DemoStack(project_dir, readonly_dir, readwrite_dir, prefix)
    finally:
        _teardown(osprey_bin, project_dir, prefix)


# ---------------------------------------------------------------------------
# T1: topology — nginx + both per-user containers come up healthy, and the
# landing page lists both roster users.
# ---------------------------------------------------------------------------


def test_t1_demo_topology_containers_up(demo_stack: DemoStack) -> None:
    prefix = demo_stack.prefix
    expected = [
        _nginx_container(prefix),
        _web_container(prefix, READONLY_USER),
        _web_container(prefix, READWRITE_USER),
    ]
    missing = [name for name in expected if _container_id(name) is None]
    assert not missing, f"container(s) not created by 'deploy up': {missing}"

    # Each container serves before Docker flips its healthcheck off "starting",
    # so poll for "healthy" rather than assert it the instant the fixture yields.
    for name in expected:
        _wait_for_container_health(name, CONTAINER_HEALTH_TIMEOUT_SEC)

    # Both per-user images were built locally as `<persona-project>-<persona>:local`
    # (build_persona_images) — the readonly and readwrite tiers are genuinely two
    # distinct images, the core promise of the local-mode persona build.
    for image in (READONLY_IMAGE, READWRITE_IMAGE):
        assert _runtime_cli("image", "inspect", image, timeout=15).returncode == 0, (
            f"persona image {image} was not built by 'deploy up'"
        )

    # The nginx landing page serves and lists both roster users' cards.
    landing = _fetch_in_container(_nginx_container(prefix), LANDING_URL, HEALTH_TIMEOUT_SEC)
    for user in (READONLY_USER, READWRITE_USER):
        assert user in landing, f"landing page does not list roster user {user!r}"


# ---------------------------------------------------------------------------
# T2/T3: write posture — the readonly tier pins writes off and denies the
# channel-write tool at render time; the readwrite tier arms writes and keeps
# the tool on the human-approval (ask) path.
# ---------------------------------------------------------------------------


def _writes_enabled(project_dir: Path) -> bool:
    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    return bool((config.get("control_system") or {}).get("writes_enabled", False))


def _permissions(project_dir: Path) -> dict:
    settings_path = project_dir / ".claude" / "settings.json"
    assert settings_path.is_file(), f"no settings.json rendered at {settings_path}"
    return json.loads(settings_path.read_text(encoding="utf-8")).get("permissions", {})


def test_t2_readonly_project_denies_writes(demo_stack: DemoStack) -> None:
    assert _writes_enabled(demo_stack.readonly_dir) is False, (
        "readonly tier must render control_system.writes_enabled: false"
    )
    perms = _permissions(demo_stack.readonly_dir)
    assert CHANNEL_WRITE_TOOL in perms.get("deny", []), (
        f"readonly tier must deny {CHANNEL_WRITE_TOOL!r} in rendered settings.json, "
        f"but deny list is: {perms.get('deny', [])}"
    )


def test_t3_readwrite_project_arms_writes(demo_stack: DemoStack) -> None:
    # Positive control for T2: the same render pipeline does NOT deny the write
    # tool for the readwrite tier — T2's deny is a real posture difference, not
    # a render that denied the tool everywhere.
    assert _writes_enabled(demo_stack.readwrite_dir) is True, (
        "readwrite tier must render control_system.writes_enabled: true"
    )
    perms = _permissions(demo_stack.readwrite_dir)
    assert CHANNEL_WRITE_TOOL not in perms.get("deny", []), (
        f"readwrite tier must not deny {CHANNEL_WRITE_TOOL!r}, but deny list is: "
        f"{perms.get('deny', [])}"
    )
    # The write path is supervised, not open: the tool stays on the ask
    # (human-approval) path.
    assert CHANNEL_WRITE_TOOL in perms.get("ask", []), (
        f"readwrite tier must keep {CHANNEL_WRITE_TOOL!r} on the ask path, but ask "
        f"list is: {perms.get('ask', [])}"
    )


# ---------------------------------------------------------------------------
# T4: same tool surface — the tier boundary is enforcement, never a quietly
# different tool surface.
# ---------------------------------------------------------------------------


def _mcp_server_keys(project_dir: Path) -> set[str]:
    mcp_path = project_dir / ".mcp.json"
    assert mcp_path.is_file(), f"no .mcp.json rendered at {mcp_path}"
    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    return set(data.get("mcpServers", {}))


def test_t4_tiers_share_identical_mcp_surface(demo_stack: DemoStack) -> None:
    readonly_keys = _mcp_server_keys(demo_stack.readonly_dir)
    readwrite_keys = _mcp_server_keys(demo_stack.readwrite_dir)
    assert readonly_keys == readwrite_keys, (
        "the two tiers must declare the identical MCP server set (the boundary "
        f"is writes_enabled, not tool absence): readonly={sorted(readonly_keys)} "
        f"readwrite={sorted(readwrite_keys)}"
    )
