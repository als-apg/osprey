"""Phase-6 acceptance proof: the ``control-assistant`` demo multi-user topology
stands itself up from the preset with a single ``osprey build`` + ``osprey
deploy up``, and the two personas it ships are genuinely different capability
tiers over a real containerized Virtual Accelerator driven through the Bluesky
scan bridge.

The ``control-assistant`` preset ships a ``modules.web_terminals`` block (an
operator tier + a physicist tier, roster ``alice``/``bob``, an nginx reverse
proxy) plus the Bluesky scan bridge, and — once ``control_system.type`` is
flipped to ``virtual_accelerator`` — a containerized PyAT Virtual Accelerator.
``osprey deploy up`` in local image mode is supposed to, from nothing but the
built project: auto-render each referenced persona project (``osprey build ...
--skip-deps``), build both persona ``:local`` images, and bring up nginx + one
web-terminal container per user alongside the bridge and the VA. This test drives
that whole promise against a real container runtime and asserts four things end
to end:

  T1 topology:      nginx + both per-user containers (+ bridge + VA) come up
                     healthy, exact-named off the deploy's OWN ``facility.prefix``
                     (read from the rendered config, never hardcoded) and project
                     name.
  T2 operator tier:  the auto-rendered OPERATOR project's ``.mcp.json`` has NO
                     ``bluesky`` MCP server (the operator tier is deliberately
                     kept out of scan tooling).
  T3 physicist tier: the auto-rendered PHYSICIST project's ``.mcp.json`` DOES
                     carry the ``bluesky`` server — the positive control that T2
                     is a real capability difference, not a render that dropped
                     the server everywhere.
  T4 VA drivable:    a connector-mediated ``grid_scan`` plan through the Bluesky
                     bridge against the containerized PyAT VA completes, and its
                     readback (``:RB``) detector tracks the swept setpoint
                     (``:SP``) at every step — proving the physicist tier's scan
                     target is a live, drivable soft-IOC (the ``VA = run``
                     capability the scan tier exists for), not a mock that never
                     moves.

No preset channel names are hardcoded: T4's ``:SP``/``:RB`` pair is derived from
the DEPLOYED project's own ``data/channel_limits.json`` restricted to the sp-echo
partition (``classify_partition``) — a pure software echo where ``:RB`` tracks
``:SP`` exactly, with none of the ring-wide physics side effects a pyat-coupled
corrector write has (wrong for a deterministic tracking probe). The scan is run
through the bridge exactly as ``tests/e2e/test_va_substrate_equivalence.py``
drives it (the ``BLUESKY_EPICS_*`` substrate scanner + a self-supplied launch
token, since this preset's writes_enabled+local-exec config deliberately gates
the bridge's auto-minting off).

CONTAINER-OPS SAFETY: every runtime-mutating call names an EXACT resource this
test created — the ``<prefix>-nginx`` / ``<prefix>-web-<user>`` web containers
(prefix read from config), the ``<project>-virtual-accelerator`` /
``<project>-bluesky-bridge`` service containers, the ``<project>_<user>-*``
volumes, and the ``:local`` persona/VA/bridge images — or is the project-scoped
``osprey deploy nuke`` / ``compose -p <project> down`` teardown the lifecycle
code itself uses (label-scoped, containers/networks only). Nothing here ever runs
a prune, an ``-a``/``--all`` sweep, a ``volume prune``, or a wildcard container
match. Teardown runs from a fixture ``finally`` so a failed assertion mid-sequence
still leaves nothing stranded; the web-container names (whose prefix is a shared
preset default, not unique per run) are exact-name pre-cleaned before deploy so a
crashed prior run can never be adopted.

Coexistence with other osprey deploys on the same host is safe by design:
every ``osprey deploy`` compose invocation pins ``COMPOSE_PROJECT_NAME`` to its
own resolved project name, so this deploy's backend services live under the
``castdemo-e2e`` compose project and can never adopt or recreate another
deployment's containers or volumes. Together with the remapped ports above,
this test runs alongside live foreign osprey stacks; only the web containers
collide (shared preset ``facility.prefix``) and are exact-name pre-cleaned.

Gating: needs a container runtime whose daemon is actually responding (``docker``
by default; ``OSPREY_E2E_RUNTIME=podman`` opt-in, matching the sibling deploy
e2e). Skipped cleanly otherwise. Slow: ``deploy up`` builds two full persona
images plus the bridge and the native PyAT VA image (source-compiled on Apple
Silicon — minutes on a cold cache). Lives in ``tests/e2e/`` (never collected by
the fast lane).

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
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_project_name, resolve_user_volume_names

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
# This deploy's identity. The project name (compose project / VA + bridge
# container names / volume namespace) is unique to this file so it never
# collides with another deploy e2e. The web-container prefix is NOT set here —
# it is the preset's own `facility.prefix`, read from the rendered config at
# runtime (see _facility_prefix) so a future prefix change can't silently drift
# this test's expected names.
# ---------------------------------------------------------------------------
PROJECT_NAME = "castdemo-e2e"
DEPLOY_FQDN = "localhost"

# Roster shipped by the preset: alice -> operator (default_persona), bob ->
# physicist. Asserted, not configured, here — the preset owns the roster.
OPERATOR_USER = "alice"
PHYSICIST_USER = "bob"

# Persona catalog entries the preset ships (project name -> persona key). The
# auto-rendered project directory basename equals the catalog `project`, and the
# built image tag is `<project>-<persona>:local` (build_persona_images).
OPERATOR_PROJECT = "control-assistant-operator"
PHYSICIST_PROJECT = "control-assistant-physicist"
OPERATOR_IMAGE = f"{OPERATOR_PROJECT}-operator:local"
PHYSICIST_IMAGE = f"{PHYSICIST_PROJECT}-physicist:local"

# The MCP server key the physicist opts in and the operator denies
# (registry/mcp.py's ServerDefinition name="bluesky"). Present as a key under
# `.mcp.json`'s `mcpServers` iff the tier enabled it.
SCAN_SERVER_KEY = "bluesky"

# EVERY published host port is remapped off the control-assistant preset's
# defaults to a unique range, so this deploy coexists with an already-running
# control-assistant demo on the same host (which occupies the DEFAULT ports —
# VA 5064, OpenObserve 5080, Postgres 5432, bridge 8090, tiled 8091). Without
# this, `deploy up` fails at compose-up with "port is already allocated". Web
# ports are disjoint from every other deploy e2e's range too (test_deploy_lifecycle.py
# tops out ~20600).
WEB_PORTS = {
    "nginx": 20780,
    "web": 20800,
    "artifact": 20900,
    "ariel": 21000,
    "lattice": 21100,
}
OPENOBSERVE_PORT = 25080
POSTGRES_PORT = 25432

# Channel Access port the VA serves on — remapped off the preset default 5064.
# `services.virtual_accelerator.port` drives the VA's published port, its served
# port (EPICS_CA_SERVER_PORT), its healthcheck, AND the bridge's
# EPICS_CA_NAME_SERVERS target, so setting it once keeps every reference
# consistent (see services/virtual_accelerator + services/bluesky compose
# templates). The connector gateway port (a separate config key) is overridden to
# match as belt-and-suspenders for any host-side connector.
VA_CA_PORT = 25064
# Locally-built VA image, derived the way the compose template does
# (``resolve_project_name`` -> ``<project>-va:local``) rather than a host-global
# tag that no longer matches the rendered image.
VA_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-va:local"

# Bluesky bridge + its Tiled store — non-default ports so the already-running
# demo's 8090/8091 never collide. The scan is driven through this bridge.
BRIDGE_PORT = 18096
TILED_PORT = 25091
BRIDGE_URL = f"http://localhost:{BRIDGE_PORT}"
BRIDGE_IMAGE = f"{resolve_project_name({'project_name': PROJECT_NAME})}-bluesky-bridge:local"

# The bluesky-panels sidecar the preset ships alongside the bridge — same
# rationale: off the preset default 8095 so an already-running demo's panels
# sidecar never collides.
PANELS_PORT = 18095

# The bridge's launch route (POST /runs/{id}/launch) fails closed on an unset
# BLUESKY_LAUNCH_TOKEN. This preset deploys with writes_enabled:true +
# execution_method:local, which deliberately gates auto-minting off
# (container_lifecycle._local_exec_arming_unsafe); this controlled test supplies
# its own token — the supported operator-provides-a-token path (mirrors
# test_va_substrate_equivalence.py).
LAUNCH_TOKEN = "e2e-control-assistant-demo-launch-token"

# Scan device names wired into the bridge via BLUESKY_EPICS_MOTORS/_DETECTORS —
# arbitrary handles resolved against explicit PV addresses, never a preset
# naming convention.
SCAN_MOTOR = "scan_motor"
SCAN_DETECTOR = "scan_det"
SCAN_NUM_POINTS = 4

BUILD_TIMEOUT_SEC = 300
# deploy up here builds TWO persona images plus the bridge and the native PyAT VA
# image (source build on Apple Silicon, no prebuilt aarch64 wheels) — heavier
# than the sibling single-image deploy e2e, so a wide ceiling.
DEPLOY_UP_TIMEOUT_SEC = 2400
DEPLOY_DOWN_TIMEOUT_SEC = 300
HEALTH_TIMEOUT_SEC = 300.0
CONTAINER_HEALTH_TIMEOUT_SEC = 180.0
SCAN_TIMEOUT_SEC = 90.0


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


def _va_container() -> str:
    # The compose templates render each service's container_name as
    # `<project>-<service>`; the VA/bridge compose services are
    # `virtual-accelerator` / `bluesky-bridge` — same derivation
    # test_va_substrate_equivalence.py uses.
    return f"{PROJECT_NAME}-virtual-accelerator"


def _bridge_container() -> str:
    return f"{PROJECT_NAME}-bluesky-bridge"


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


def _get(path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{BRIDGE_URL}{path}", method="GET")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(path: str, body: dict, headers: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        f"{BRIDGE_URL}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _find_column(columns: list[str], device_name: str) -> int:
    """The event-data column for a device — ophyd-async names a hinted child
    ``"<device>-<child>"``; match the device-name prefix rather than the exact
    key so this doesn't hardcode ophyd-async's internal child-attribute naming.
    """
    for i, col in enumerate(columns):
        if col == device_name or col.startswith(f"{device_name}-"):
            return i
    raise AssertionError(f"no column for device {device_name!r} in {columns!r}")


# ---------------------------------------------------------------------------
# Channel selection + scan-env wiring (mirrors test_va_substrate_equivalence.py)
# ---------------------------------------------------------------------------


def _channel_limits(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def _select_sp_echo_pair(channel_limits: dict[str, Any]) -> tuple[str, str]:
    """Derive one sp-echo (``:SP``, ``:RB``) pair from the deployed project's own
    channel_limits.json — no hardcoded preset channels.

    Restricted to the sp-echo partition (``classify_partition``): its ``:RB``
    tracks its ``:SP`` exactly with no ring-wide physics side effects, exactly
    what a deterministic RB-tracks-SP scan probe needs (a pyat-coupled ``:SP``
    write moves other devices via the lattice model — wrong for this probe).
    """
    from osprey.services.virtual_accelerator.manifest import PARTITION_SP_ECHO, classify_partition

    keys = {k for k in channel_limits if not k.startswith("_") and k != "defaults"}
    for sp in sorted(k for k in keys if k.endswith(":SP")):
        parts = sp.split(":")
        if len(parts) != 6:
            continue
        ring, system, family, device, field, subfield = parts
        path = {
            "ring": ring,
            "system": system,
            "family": family,
            "device": device,
            "field": field,
            "subfield": subfield,
        }
        if classify_partition(path) != PARTITION_SP_ECHO:
            continue
        rb = sp[:-3] + ":RB"
        if rb in keys:
            return sp, rb
    raise AssertionError("deployed project's channel_limits.json yields no sp-echo :SP/:RB pair")


def _write_scan_env(project_dir: Path, sp: str, rb: str) -> None:
    """Append the bridge's contract env vars to the project ``.env`` — BEFORE
    ``osprey deploy up`` (the bridge compose template passes these through from
    the project ``.env``).

    Wires the connector-backed EPICS substrate scanner (``BLUESKY_EPICS_SUBSTRATE``)
    with one motor (the sp-echo ``:SP``, readback ``:RB``) and one detector (its
    ``:RB``), plus the launch token this test supplies itself.
    """
    values = {
        "BLUESKY_LAUNCH_TOKEN": LAUNCH_TOKEN,
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": f"{SCAN_MOTOR}={sp}|{rb}",
        "BLUESKY_EPICS_DETECTORS": f"{SCAN_DETECTOR}={rb}",
    }
    env_path = project_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    env_path.write_text(
        existing + "".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8"
    )


def _run_scan(plan_name: str, plan_args: dict, timeout: float = SCAN_TIMEOUT_SEC) -> dict:
    """POST /runs -> launch -> poll to a terminal status. Returns the final status body."""
    status, body = _post("/runs", {"plan_name": plan_name, "plan_args": plan_args})
    assert status == 200, f"POST /runs failed: {status} {body}"
    run_id = body["id"]

    status, body = _post(f"/runs/{run_id}/launch", {}, headers={"X-Launch-Token": LAUNCH_TOKEN})
    assert status == 200, f"launch failed: {status} {body}"

    deadline = time.monotonic() + timeout
    last_status_body: dict = {}
    while time.monotonic() < deadline:
        _, last_status_body = _get(f"/runs/{run_id}")
        if last_status_body.get("status") in ("completed", "error", "stopped"):
            break
        time.sleep(0.2)
    last_status_body["id"] = run_id
    return last_status_body


# ---------------------------------------------------------------------------
# Deployed-stack fixture
# ---------------------------------------------------------------------------


class DemoStack:
    """Everything the T1-T4 tests need about the one co-deployed demo project."""

    def __init__(
        self,
        project_dir: Path,
        operator_dir: Path,
        physicist_dir: Path,
        prefix: str,
        sp: str,
        rb: str,
        limits: dict[str, Any],
    ):
        self.project_dir = project_dir
        self.operator_dir = operator_dir
        self.physicist_dir = physicist_dir
        self.prefix = prefix
        self.sp = sp
        self.rb = rb
        self.limits = limits

    def bounds(self, address: str) -> tuple[float, float]:
        entry = self.limits[address]
        return float(entry["min_value"]), float(entry["max_value"])


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
        for user in (OPERATOR_USER, PHYSICIST_USER):
            _runtime_cli("rm", "-f", _web_container(prefix, user))
        _runtime_cli("rm", "-f", _nginx_container(prefix))
    _runtime_cli("rm", "-f", _va_container())
    _runtime_cli("rm", "-f", _bridge_container())
    _runtime_cli("compose", "-p", PROJECT_NAME, "down", timeout=DEPLOY_DOWN_TIMEOUT_SEC)
    for user in (OPERATOR_USER, PHYSICIST_USER):
        for volume in resolve_user_volume_names({"project_name": PROJECT_NAME}, user):
            _runtime_cli("volume", "rm", volume)
    for image in (OPERATOR_IMAGE, PHYSICIST_IMAGE, VA_IMAGE, BRIDGE_IMAGE):
        _runtime_cli("rmi", "-f", image)


@pytest.fixture(scope="module")
def demo_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DemoStack]:
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    if _runtime_cli("ps", timeout=10).returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding")

    osprey_bin = _find_osprey_console_script()
    base = tmp_path_factory.mktemp("cast_demo_build")
    project_dir = base / PROJECT_NAME
    # The preset pins each persona's project_path to `../<persona-project>`, so
    # auto-render lands them as siblings of the main project dir under `base`.
    operator_dir = base / OPERATOR_PROJECT
    physicist_dir = base / PHYSICIST_PROJECT

    # Flip the control system to the VA and supply the deploy-time landing origin
    # the web-terminal render requires (deploy.fqdn) plus this deploy's unique web
    # ports. facility.prefix is deliberately NOT overridden — the preset ships it,
    # and the test reads it back from the rendered config (see _facility_prefix).
    # Written as flat dotted `config:` keys (the preset's own convention) so each
    # sets only its leaf — a nested `config:` mapping would deep-merge and clobber
    # the preset's rendered subtrees. `dispatch: null` drops the event-dispatcher
    # stack (Node + Claude CLI image), irrelevant here and far slower to build
    # than the rest of the stack already is.
    # All ports remapped off the preset defaults so this deploy coexists with an
    # already-running control-assistant demo (see the port constants above). The
    # VA gateway ports match the VA's remapped served port. facility.prefix is
    # deliberately NOT overridden — the preset ships it and the test reads it back
    # from the rendered config (see _facility_prefix).
    override_path = base / "override.yml"
    override_lines = [
        "config:",
        "  control_system.type: virtual_accelerator",
        f"  deploy.fqdn: {DEPLOY_FQDN}",
        f"  services.openobserve.port: {OPENOBSERVE_PORT}",
        f"  services.postgresql.port_host: {POSTGRES_PORT}",
        f"  control_system.connector.virtual_accelerator.gateways.read_only.port: {VA_CA_PORT}",
        f"  control_system.connector.virtual_accelerator.gateways.write_access.port: {VA_CA_PORT}",
        f"  modules.web_terminals.nginx_port: {WEB_PORTS['nginx']}",
        f"  modules.web_terminals.web_base_port: {WEB_PORTS['web']}",
        f"  modules.web_terminals.artifact_base_port: {WEB_PORTS['artifact']}",
        f"  modules.web_terminals.ariel_base_port: {WEB_PORTS['ariel']}",
        f"  modules.web_terminals.lattice_base_port: {WEB_PORTS['lattice']}",
        "dispatch: null",
        "",
    ]
    override_path.write_text("\n".join(override_lines), encoding="utf-8")

    build = _run(
        [
            str(osprey_bin),
            "build",
            PROJECT_NAME,
            "--preset",
            "control-assistant",
            "--override",
            str(override_path),
            # Turn the VA service on and pin its published CA port (the connector
            # gateway is hardcoded to VA_CA_PORT in config.yml.j2).
            "--set",
            f"virtual_accelerator.port={VA_CA_PORT}",
            # Point the shipped bridge + its Tiled store at this deploy's unique
            # ports and use the connector-backed EPICS substrate scanner (wired via
            # .env below), not the built-in demo scanner.
            "--set",
            f"bluesky.port={BRIDGE_PORT}",
            "--set",
            f"bluesky.tiled_port={TILED_PORT}",
            "--set",
            f"bluesky_panels.port={PANELS_PORT}",
            "--set",
            "bluesky.demo_scanner=false",
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

    # Select the scan channels from the DEPLOYED project's own limits and wire the
    # bridge's substrate scanner + launch token into .env, all BEFORE deploy up.
    limits = _channel_limits(project_dir)
    sp, rb = _select_sp_echo_pair(limits)
    _write_scan_env(project_dir, sp, rb)

    prefix = _facility_prefix(project_dir)

    # The web-container names use the preset's shared `facility.prefix` (not this
    # test's unique project name), so a crashed prior run could have left the
    # exact-named containers behind. Remove them by exact name before deploy so
    # this run never adopts a stale container (guardrail: exact-named only).
    for user in (OPERATOR_USER, PHYSICIST_USER):
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
        _wait_for_health(f"{BRIDGE_URL}/health", HEALTH_TIMEOUT_SEC)
        yield DemoStack(project_dir, operator_dir, physicist_dir, prefix, sp, rb, limits)
    finally:
        _teardown(osprey_bin, project_dir, prefix)


# ---------------------------------------------------------------------------
# T1: topology — nginx + both per-user containers + bridge + VA come up healthy
# ---------------------------------------------------------------------------


def test_t1_demo_topology_containers_up(demo_stack: DemoStack) -> None:
    prefix = demo_stack.prefix
    expected = [
        _nginx_container(prefix),
        _web_container(prefix, OPERATOR_USER),
        _web_container(prefix, PHYSICIST_USER),
        _bridge_container(),
        _va_container(),
    ]
    missing = [name for name in expected if _container_id(name) is None]
    assert not missing, f"container(s) not created by 'deploy up': {missing}"

    # Each container serves before Docker flips its healthcheck off "starting",
    # so poll for "healthy" rather than assert it the instant the fixture yields.
    for name in expected:
        _wait_for_container_health(name, CONTAINER_HEALTH_TIMEOUT_SEC)

    # Both per-user images were built locally as `<persona-project>-<persona>:local`
    # (build_persona_images) — the operator and physicist tiers are genuinely two
    # distinct images, the core promise of the local-mode persona build.
    for image in (OPERATOR_IMAGE, PHYSICIST_IMAGE):
        assert _runtime_cli("image", "inspect", image, timeout=15).returncode == 0, (
            f"persona image {image} was not built by 'deploy up'"
        )


# ---------------------------------------------------------------------------
# T2/T3: capability tiers — the bluesky MCP server is absent for the operator,
# present for the physicist (the same auto-rendered projects `deploy up` built).
# ---------------------------------------------------------------------------


def _mcp_server_keys(project_dir: Path) -> set[str]:
    mcp_path = project_dir / ".mcp.json"
    assert mcp_path.is_file(), f"no .mcp.json rendered at {mcp_path}"
    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    return set(data.get("mcpServers", {}))


def test_t2_operator_project_has_no_scan_server(demo_stack: DemoStack) -> None:
    keys = _mcp_server_keys(demo_stack.operator_dir)
    assert SCAN_SERVER_KEY not in keys, (
        f"operator tier must not expose the {SCAN_SERVER_KEY!r} MCP server, but "
        f".mcp.json declares: {sorted(keys)}"
    )


def test_t3_physicist_project_has_scan_server(demo_stack: DemoStack) -> None:
    # Positive control for T2: the same render pipeline DOES emit the scan server
    # for the physicist tier, so T2's absence is a real capability difference,
    # not a render that dropped the server everywhere.
    keys = _mcp_server_keys(demo_stack.physicist_dir)
    assert SCAN_SERVER_KEY in keys, (
        f"physicist tier must expose the {SCAN_SERVER_KEY!r} MCP server, but "
        f".mcp.json declares: {sorted(keys)}"
    )


# ---------------------------------------------------------------------------
# T4: the physicist tier's scan target is a live, drivable VA — a bridge scan
# completes and its :RB detector tracks the swept :SP setpoint at every step.
# ---------------------------------------------------------------------------


def test_t4_physicist_bridge_scan_drives_va(demo_stack: DemoStack) -> None:
    sp, rb = demo_stack.sp, demo_stack.rb
    lo, hi = demo_stack.bounds(sp)
    start = lo + 0.25 * (hi - lo)
    stop = lo + 0.75 * (hi - lo)
    num = SCAN_NUM_POINTS

    status_body = _run_scan(
        "grid_scan",
        {
            "detectors": [SCAN_DETECTOR],
            "axes": [{"setpoint": SCAN_MOTOR, "start": start, "stop": stop, "num_points": num}],
        },
    )
    assert status_body.get("status") == "completed", f"bridge scan did not complete: {status_body}"

    status, data = _get(f"/runs/{status_body['id']}/data")
    assert status == 200, f"GET /runs/{status_body['id']}/data failed: {status} {data}"
    assert data["row_count"] == num, f"expected {num} rows: {data}"

    col = _find_column(data["columns"], SCAN_DETECTOR)
    readbacks = [row[col] for row in data["rows"]]
    assert all(v is not None for v in readbacks), f"incomplete {SCAN_DETECTOR} column: {readbacks}"

    # sp-echo :RB tracks :SP exactly and settles per step, so each scanned
    # readback must equal the setpoint the bridge commanded at that step. A mock
    # connector's :RB never tracks (see the project memory note "Scan stack: mock
    # = browse, VA = run"), so a settled, tracking sweep is proof the deployed
    # target is the live PyAT soft-IOC.
    setpoints = [start + i * (stop - start) / (num - 1) for i in range(num)]
    for want, got in zip(sorted(setpoints), sorted(readbacks), strict=True):
        assert abs(got - want) <= 1e-6, (
            f"{rb} did not track {sp}: scanned readbacks {sorted(readbacks)} do not "
            f"match commanded setpoints {sorted(setpoints)}"
        )
