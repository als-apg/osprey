"""Real-container e2e: the dispatch pair booted in BOTH network shapes.

The network axis gives a deployment two topologies for the same pair of
services. On the compose **bridge** (the default) the dispatcher and its workers
get their own network namespace, address each other by compose DNS name, and
reach the host only through a published port. On the **host** namespace
(``dispatch.network: host``) there is no envelope at all: both processes bind
the machine's own interfaces, address each other as ``localhost``, and nothing
is published because there is no port map left to publish.

Every other test of this axis reads the RENDER — the compose YAML, the patched
triggers file, the goldens. None of them starts anything, so none of them can
say whether a stack rendered in either shape actually *works*. This module is
the instrument for that. It builds one dispatch-enabled deployment repo, boots
it twice — once per shape — and in each shape asserts the three things the axis
is supposed to guarantee:

  1. **The pair completes a dispatch round-trip.** A webhook fired at the
     dispatcher reaches the worker, and the worker's run feed comes back
     through the dispatcher's proxy. Both crossings travel whichever addresses
     the build emitted for the shape under test, so a topology that renders
     cleanly but cannot connect fails here.
  2. **The env chain resolves local-over-shared inside the container.** A key
     set in BOTH chain files arrives in the worker's environment carrying the
     ``.env`` value, and a key set only in ``.env.shared`` arrives at all —
     the second half is the positive control that makes the first meaningful
     (a chain that silently dropped ``.env.shared`` would satisfy "local wins"
     vacuously).
  3. **Under ``host``, the services bind loopback and nothing else.** Read off
     the real listening sockets (``/proc/net/tcp``), not off the config that
     asked for them. The bridge shape asserts the counterpart — every
     interface inside its own namespace, published to the host's loopback only
     — so this is a two-sided comparison rather than a one-sided claim.

ONE REPO, TWO SHAPES
--------------------
Both shapes are driven through the SAME deployment repo, re-rendered between
boots by flipping exactly one word in ``profile.yml``. That is deliberate on
two counts: the project identity, image tags, ports and roster are then
provably identical across the two boots, so the network axis is the only
variable; and the worker runs the full project image, which is expensive to
build once and free to reuse.

FIXTURE SHAPE (the routed constraint from the build's network checks)
--------------------------------------------------------------------
A render may not mix modes: a host-mode service and a bridge-resident consumer
in one build fails by design. This fixture is same-mode by construction — the
dispatcher and its worker are the only deployed services, and one profile knob
sets both. Telemetry is turned off and the store is left out of
``deployed_services`` for the same reason it would otherwise complicate this
fixture: an enabled telemetry block whose store is republished off port 5080
needs an explicit ``claude_code.telemetry.endpoint``, which is a separate
mechanism from the two this module is about.

CONTAINER-OPS SAFETY (every runtime-mutating call in this file honors this)
---------------------------------------------------------------------------
Every resource this module creates is exact-named off ``project_name:
osprey-e2e-two-shape``:

  * containers: ``osprey-e2e-two-shape-event-dispatcher``,
    ``osprey-e2e-two-shape-dispatch-worker-1``
  * volume: ``osprey-e2e-two-shape_dispatch_workspace_1``
  * images: ``osprey-e2e-two-shape:local``,
    ``osprey-e2e-two-shape-dispatch:local``
  * network: ``osprey-e2e-two-shape_osprey-network``

Nothing here ever runs a prune (system/container/image/volume/network), an
``-a``/``--all`` removal, or a wildcard/glob name match. Teardown is either
``osprey down`` from the repo (the verb under test) or an exact-named removal,
and it runs from a ``finally`` so a failed assertion mid-shape still leaves
nothing stranded — including the host ports the next shape needs free.

Gating: needs a container runtime whose daemon actually responds. The runtime
is ``docker`` by default; ``OSPREY_E2E_RUNTIME=podman`` switches this file's
real-runtime calls to podman (the CI two-shape lane sets it, pinning
podman-compose 1.0.6). Any other value fails at collection time. A missing CLI
skips the module; a daemon that is present but unresponsive FAILS it, because
"the daemon hung" and "no runtime here" are different facts and only one of
them is a reason to declare the axis untested.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_project_name

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.dockerbuild]

# ---------------------------------------------------------------------------
# Container runtime selection — ``docker`` by default, ``podman`` opt-in via
# OSPREY_E2E_RUNTIME (the CI two-shape lane sets it). Any other value fails
# clearly at collection time rather than silently falling back to docker.
# ---------------------------------------------------------------------------
_SUPPORTED_RUNTIMES = ("docker", "podman")
RUNTIME = os.environ.get("OSPREY_E2E_RUNTIME", "docker")
if RUNTIME not in _SUPPORTED_RUNTIMES:
    raise RuntimeError(
        f"OSPREY_E2E_RUNTIME={RUNTIME!r} is not supported; expected one of {_SUPPORTED_RUNTIMES}"
    )

# ---------------------------------------------------------------------------
# Fixture identity. The deployment repo's directory name IS the deployment's
# name, and the name every container/volume/image below is derived from — so
# each one is exact-named and obviously throwaway.
# ---------------------------------------------------------------------------
PROJECT_NAME = "osprey-e2e-two-shape"
COMPOSE_PROJECT = resolve_project_name({"project_name": PROJECT_NAME})

DISPATCHER_CONTAINER = f"{PROJECT_NAME}-event-dispatcher"
WORKER_CONTAINER = f"{PROJECT_NAME}-dispatch-worker-1"
DISPATCH_IMAGE = f"{PROJECT_NAME}-dispatch:local"
PROJECT_IMAGE = f"{PROJECT_NAME}:local"
WORKSPACE_VOLUME = f"{COMPOSE_PROJECT}_dispatch_workspace_1"
# Removed by name at teardown too. ``osprey down`` normally takes the network with
# it, but the exact-named removals exist for the case where ``down`` failed midway,
# and without this one the network is the single resource they would not cover.
COMPOSE_NETWORK = f"{COMPOSE_PROJECT}_osprey-network"

# BOTH ports are remapped off the shipped defaults (8020 and 9190) so this
# deploy coexists with any other OSPREY dispatch stack on the same host. Under
# `host` these are binds on the machine's own namespace rather than publishes,
# so a collision does not merely fail to publish — it refuses at the host-port
# preflight, and the shipped defaults are precisely the values another stack
# would already be holding.
#
# Remapping the worker obliges the repo's AUTHORED dispatch_target to name the
# same port, which the repo fixture writes (the shipped triggers file names the
# default). That is not a workaround around the axis: under `bridge` the build
# leaves a facility-authored target alone, so a non-default value demonstrates
# that leave-it-alone property in a way the default cannot — the default would
# survive a hypothetical hardcoded fallback just as convincingly.
DISPATCHER_PORT = 21781
WORKER_PORT = 21791

TOKEN = "two-shape-e2e-token"  # matches the .env written by the repo fixture
TRIGGER = "hello-dispatch"  # the bundled tutorial trigger, fired for its round-trip

# The env-chain probe. Both variables are read out of the WORKER's environment,
# where the compose ``env_file`` chain delivered them; neither is an OSPREY key,
# so nothing but the chain itself can put them there.
CONFLICT_VAR = "OSPREY_E2E_CHAIN_CONFLICT"
CONFLICT_SHARED_VALUE = "value-from-env-shared"
CONFLICT_LOCAL_VALUE = "value-from-env-local"
SHARED_ONLY_VAR = "OSPREY_E2E_CHAIN_SHARED_ONLY"
SHARED_ONLY_VALUE = "value-only-in-env-shared"

INIT_TIMEOUT_SEC = 300
BUILD_TIMEOUT_SEC = 300
# The worker runs the full project image (Node + the Claude Code CLI), which is
# slow on a cold layer cache. The second shape reuses it.
UP_TIMEOUT_SEC = 1800
DOWN_TIMEOUT_SEC = 300
HEALTH_TIMEOUT_SEC = 300.0
RUN_FEED_TIMEOUT_SEC = 120.0

BRIDGE, HOST = "bridge", "host"


# ---------------------------------------------------------------------------
# Low-level runtime / CLI helpers
# ---------------------------------------------------------------------------


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _runtime_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run one runtime CLI call. Deliberately does NOT swallow a timeout: a
    daemon that stops answering is a fact worth surfacing, not a false negative.
    """
    return subprocess.run([RUNTIME, *args], capture_output=True, text=True, timeout=timeout)


def _runtime_cli_safe(*args: str, timeout: int = 60) -> None:
    """Teardown-only: best-effort, never raises.

    Teardown runs from a ``finally``, where an exception would replace the real
    failure with a cleanup failure. Every caller names an exact resource, so a
    swallowed error means "already gone / never created".
    """
    try:
        _runtime_cli(*args, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        pass


def _run_osprey(
    osprey_bin: Path, args: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess:
    """Run an ``osprey`` verb from the deployment repo's own directory.

    Always from ``cwd=repo``: several deploy-side paths resolve the repo root
    with the cwd as their last resort, so a verb run from elsewhere can write
    machine artifacts outside the deployment.

    ``CONTAINER_RUNTIME`` outranks the rendered ``container_runtime:``, so
    forcing it to ``RUNTIME`` couples the deploy-side runtime to the
    assert-side ``_runtime_cli`` runtime regardless of what the ambient
    environment sets.
    """
    return subprocess.run(
        [str(osprey_bin), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": "", "CONTAINER_RUNTIME": RUNTIME},
    )


def _fmt(label: str, result: subprocess.CompletedProcess) -> str:
    return (
        f"{label} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _inspect(name: str, fmt: str) -> str | None:
    result = _runtime_cli("inspect", "--type", "container", "-f", fmt, name, timeout=15)
    return result.stdout.strip() if result.returncode == 0 else None


def _container_id(name: str) -> str | None:
    return _inspect(name, "{{.Id}}")


def _volume_exists(name: str) -> bool:
    return _runtime_cli("volume", "inspect", name, timeout=15).returncode == 0


#: Docker spells the healthcheck block ``.State.Health``; podman has historically used
#: ``.State.Healthcheck`` and added ``Health`` only for docker compatibility. This module
#: advertises both runtimes, so it accepts either spelling.
_HEALTH_STATE_FIELDS = ("Health", "Healthcheck")


def _container_health_status(container: str) -> str | None:
    """The container's healthcheck status, whichever field this runtime spells it in.

    Reads ``.State`` whole and looks for the key, rather than asking for
    ``.State.Health.Status`` and treating a template error as "absent". Those are
    different facts that the error-driven spelling cannot separate: a missing field
    and a container that is momentarily not inspectable (mid-create, mid-remove)
    both surface as a non-zero rc, so a transient would be indistinguishable from
    an unsupported runtime.

    ADVISORY ONLY. Readiness is decided by :func:`_wait_for_service_ready`, which
    asks the service itself — podman's healthchecks are systemd-timer driven and
    never run at all on a rootless runner without a user session. This value is
    reported alongside a failure so a reader can see what the runtime believed,
    never to decide whether the stack came up.

    Returns ``None`` when the container is not inspectable right now — a transient
    the caller polls through. A runtime that reports NEITHER field raises, because
    that is a schema mismatch rather than a transient and is worth naming; every
    caller guards the raise, since a value this advisory must not be able to
    replace the failure it was being printed to explain.
    """
    raw = _inspect(container, "{{json .State}}")
    if raw is None:
        return None
    state = json.loads(raw)
    for field in _HEALTH_STATE_FIELDS:
        block = state.get(field)
        if isinstance(block, dict) and block.get("Status"):
            return str(block["Status"])
    raise AssertionError(
        f"{container}: {RUNTIME} reports neither .State.Health.Status nor "
        f".State.Healthcheck.Status, so this module cannot tell whether the service "
        f"came up. Failing now rather than polling to the {HEALTH_TIMEOUT_SEC:.0f}s "
        f"timeout. Keys present in .State: {sorted(state)}"
    )


def _dump_diagnostics(context: str) -> str:
    """Container state + recent logs, for a failure that is otherwise opaque.

    A boot e2e that fails mid-run is nearly undiagnosable from the pytest
    traceback alone — the interesting state is inside the containers. This turns
    "no run appeared" into "the worker is crash-looping, here is why".
    """
    lines = [f"=== two-shape stack diagnostics ({context}) ==="]
    for name in (DISPATCHER_CONTAINER, WORKER_CONTAINER):
        # Two reads rather than one compound template: as a single template a missing
        # health field fails the WHOLE call, and the fallback would then report a
        # running container as "(absent)" — precisely backwards in the one situation
        # this helper exists to explain.
        status = _inspect(name, "{{.State.Status}}") or "(absent)"
        try:
            health = _container_health_status(name) or "(not inspectable)"
        except AssertionError:
            # A diagnostics dump must never raise. It only ever runs when something
            # has already gone wrong, and raising here would replace the real failure
            # with this one. The readiness wait guards the same call for the same
            # reason — health is advisory on both paths.
            health = f"(no health field in {RUNTIME}'s inspect output)"
        lines.append(f"--- {name}: {status} health={health}")
        logs = _runtime_cli("logs", "--tail", "40", name, timeout=30)
        tail = ((logs.stdout or "") + (logs.stderr or "")).strip() or "(no logs)"
        lines.append(tail)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# In-container probes. Both run ``python``, which every image here is
# guaranteed to have — it is the entrypoint of both services and what their
# own compose healthchecks shell out to. Nothing assumes curl, ss or netstat.
# ---------------------------------------------------------------------------

_HTTP_PROBE = """
import sys, urllib.error, urllib.request

url, method, token, body = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
data = body.encode("utf-8") if body else None
req = urllib.request.Request(
    url,
    data=data,
    method=method,
    headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(resp.status)
        sys.stdout.write(resp.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    print(exc.code)
    sys.stdout.write(exc.read().decode("utf-8"))
except Exception as exc:
    print(0)
    sys.stdout.write(repr(exc))
"""

_LISTEN_PROBE = """
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path) as handle:
            rows = handle.read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        fields = row.split()
        # st == 0A is TCP_LISTEN; anything else is an established/closing socket.
        if len(fields) < 4 or fields[3] != "0A":
            continue
        local_address, _, hex_port = fields[1].rpartition(":")
        print(local_address, int(hex_port, 16))
"""

_GETENV_PROBE = """
import os, sys

sys.stdout.write(os.environ.get(sys.argv[1], ""))
"""


def _http_in_container(
    container: str, url: str, *, method: str = "GET", body: str = ""
) -> tuple[int, str]:
    """Issue an HTTP request from INSIDE *container*; return ``(status, body)``.

    Probing in-container rather than from the host is what makes one assertion
    work in both shapes. Under ``host`` on a Docker Desktop machine the stack
    binds inside the runtime's Linux VM and is invisible to the host outright,
    while a container in that same namespace reaches it exactly as the
    dispatcher's own healthcheck does.
    """
    result = _runtime_cli(
        "exec", container, "python", "-c", _HTTP_PROBE, url, method, TOKEN, body, timeout=60
    )
    assert result.returncode == 0, _fmt(f"http probe in {container} ({method} {url})", result)
    status_line, _, payload = result.stdout.partition("\n")
    return int(status_line.strip()), payload


def _container_env_var(container: str, name: str) -> str:
    """The value *name* actually holds in the container's environment (``""``
    when unset). Read through python rather than ``printenv`` so the probe does
    not depend on which coreutils the base image ships.
    """
    result = _runtime_cli("exec", container, "python", "-c", _GETENV_PROBE, name, timeout=30)
    assert result.returncode == 0, _fmt(f"env probe {name} in {container}", result)
    return result.stdout.strip()


def _decode_proc_address(hex_address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Decode one ``/proc/net/tcp{,6}`` local address.

    The kernel writes each 32-bit word host-endian, which on every platform this
    runs on is little-endian — so an IPv4 row's ``0100007F`` is 127.0.0.1 and
    ``00000000`` is 0.0.0.0, and an IPv6 row is four such words back to back.
    """
    raw = bytes.fromhex(hex_address)
    if len(raw) == 4:
        return ipaddress.ip_address(raw[::-1])
    return ipaddress.ip_address(b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4)))


def _listen_addresses(container: str, port: int) -> set[str]:
    """Addresses *port* is actually LISTENing on, in *container*'s namespace.

    The real evidence for the exposure half of the axis: a config that asked
    for a loopback bind and a socket that took one are different facts, and
    only the second is what an operator is exposed to. IPv4-mapped IPv6 rows
    are reported as their IPv4 form so both socket families compare equal.

    Under ``host`` both services share ONE network namespace, so *container* here
    selects a vantage point onto that shared table, not the socket's owner. The
    claim this supports is "port *port* is loopback-only in the namespace", which
    is the operator-facing fact; it is not "this process bound it".
    """
    result = _runtime_cli("exec", container, "python", "-c", _LISTEN_PROBE, timeout=30)
    assert result.returncode == 0, _fmt(f"listening-socket probe in {container}", result)

    addresses: set[str] = set()
    for row in result.stdout.splitlines():
        hex_address, _, listening_port = row.partition(" ")
        if not listening_port.strip() or int(listening_port) != port:
            continue
        address = _decode_proc_address(hex_address)
        mapped = getattr(address, "ipv4_mapped", None)
        addresses.add(str(mapped or address))
    return addresses


def _wait_for_service_ready(container: str, url: str, timeout: float) -> None:
    """Poll *url* from inside *container* until it answers ``200``.

    Readiness is asked of the SERVICE, not of the runtime's opinion of it,
    because the compose ``healthcheck:`` block cannot be the instrument here.
    Podman drives container healthchecks from systemd transient timers, so on a
    rootless runner with no user systemd session the probe never runs even once:
    the container sits at ``Health.Status: starting`` with ``FailingStreak: 0``
    and an empty log until the deadline, whatever the service is actually doing.
    Docker runs the identical probe inside its daemon and goes healthy, so a
    wait written against the health field passes on one runtime and can never
    pass on the other — while the service is up on both.

    The same in-container form as :func:`_http_in_container`, and for the same
    reason: under ``host`` the stack may bind inside the runtime's VM where the
    host cannot reach it, but a container in that namespace always can.

    The runtime's health verdict is still read on failure, as colour for the
    report rather than as the verdict itself.
    """
    deadline = time.monotonic() + timeout
    last = "(never answered)"
    while time.monotonic() < deadline:
        if _container_id(container) is None:
            last = "(container not present)"
        else:
            result = _runtime_cli(
                "exec", container, "python", "-c", _HTTP_PROBE, url, "GET", TOKEN, "", timeout=60
            )
            if result.returncode == 0:
                status_line, _, payload = result.stdout.partition("\n")
                code = status_line.strip()
                if code == "200":
                    return
                # The probe prints 0 and the exception repr when it cannot
                # connect at all, which is the ordinary shape of "not up yet".
                last = f"HTTP {code}: {payload.strip()[:160]}"
            else:
                last = f"exec failed (rc={result.returncode}): {result.stderr.strip()[:160]}"
        time.sleep(2.0)

    # Guarded exactly like the diagnostics dump: this call can raise on a runtime
    # whose `.State` spells neither health field, and letting that through would
    # replace the readiness failure being reported with an unrelated one.
    try:
        health = _container_health_status(container) or "(not inspectable)"
    except AssertionError:
        health = f"(no health field in {RUNTIME}'s inspect output)"
    raise AssertionError(
        f"{container} did not answer {url} within {timeout:.0f}s (last: {last}).\n"
        f"The runtime's own healthcheck says {health!r}, which on rootless podman "
        f"means only that its systemd timer never fired — the failure above is the "
        f"service not answering.\n"
        f"{_dump_diagnostics(f'{container} readiness timeout')}"
    )


# ---------------------------------------------------------------------------
# The deployment repo
# ---------------------------------------------------------------------------

#: Layered onto ``hello-world`` at init. ``dispatch:`` is what makes the build
#: stage the bundled dispatcher/worker templates and register both in
#: ``deployed_services``; ``network:`` is the one word the shape fixture flips.
#:
#: The two config overrides drop the preset's telemetry store: it is not part
#: of what this module tests, and deploying it would add a third container
#: whose published port collides with any other OSPREY stack on the host. The
#: dispatch injection appends its own two services to the emptied
#: ``deployed_services``, which is what leaves the render same-mode by
#: construction — exactly the pair, nothing else.
_OVERRIDE_YML = f"""\
dispatch:
  triggers: tutorial_triggers.yml
  worker_count: 1
  workspace_mode: isolated
  dispatcher_port: {DISPATCHER_PORT}
  worker_port_base: {WORKER_PORT}
  network: {BRIDGE}
config:
  claude_code.telemetry.enabled: false
  deployed_services: []
"""

#: The repo's own ``.env.shared`` (committed defaults) and ``.env`` (local
#: secrets) — the two chain members, in ascending precedence. Both tokens are
#: pinned in the LOCAL file so the bearer below is predictable and ``up`` does
#: not auto-generate random ones; the conflict key is set in BOTH files, which
#: is the whole point of the chain assertion.
_ENV_SHARED_ADDITIONS = f"""
# Appended by tests/e2e/test_two_shape_boot.py — the chain probe.
{CONFLICT_VAR}={CONFLICT_SHARED_VALUE}
{SHARED_ONLY_VAR}={SHARED_ONLY_VALUE}
"""

_ENV_LOCAL = f"""\
EVENT_DISPATCHER_TOKEN={TOKEN}
DISPATCH_WORKER_TOKEN={TOKEN}
{CONFLICT_VAR}={CONFLICT_LOCAL_VALUE}
"""


def _require_responsive_runtime() -> None:
    """Skip when the runtime is absent; FAIL when it is present but hung.

    These are different facts. A CLI that is not installed means this host was
    never going to run the module. A daemon that accepts the call and then
    stops answering means the axis is untested for a reason someone has to look
    at — reporting it as a skip would file it under "not applicable".
    """
    if shutil.which(RUNTIME) is None:
        pytest.skip(f"{RUNTIME} not available")
    try:
        info = _runtime_cli("info", timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"`{RUNTIME} info` did not return within 60s. The daemon is present but "
            f"unresponsive — this is a hang, not a stopped runtime, and the two-shape "
            f"boot cannot be declared skipped on it."
        ) from exc
    if info.returncode != 0:
        pytest.skip(f"{RUNTIME} daemon not responding (rc={info.returncode})")


def _assert_ports_free(*ports: int) -> None:
    """Fail naming the PORT, before ``osprey up`` fails naming nothing in particular.

    A collision is already caught downstream — the host-port preflight refuses under
    ``host`` and the publish fails under ``bridge`` — so this is legibility, not
    correctness. But what the reader gets from those is ``osprey up failed (rc=1)``
    followed by a wall of compose output; what they need is "21781 was already
    taken". A stray stack sitting on the shipped default ports is exactly what forced
    these two off theirs, so the next collision should introduce itself.

    Both ports are checked in both shapes even though ``bridge`` never publishes the
    worker's: the host boot that comes after it does bind that port for real, and
    finding out before the first boot beats finding out after the bridge half passed.
    """
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # SO_REUSEADDR so a TIME_WAIT socket left by the previous shape's
            # teardown does not read as a live collision. A genuine LISTENer still
            # refuses the bind.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise AssertionError(
                    f"127.0.0.1:{port} is already in use, so this deployment cannot "
                    f"boot. Something else on this host holds it — find it with "
                    f"`lsof -nP -iTCP:{port} -sTCP:LISTEN`. Nothing was started or "
                    f"torn down."
                ) from exc


def _set_dispatch_network(profile_path: Path, mode: str) -> None:
    """Flip ``dispatch.network`` in the repo's profile, in place.

    Round-tripped rather than re-dumped so the rest of the profile — including
    the comments ``osprey init`` wrote — survives, and so the only difference
    between the two boots is this one value.
    """
    from ruamel.yaml import YAML

    round_trip = YAML()
    round_trip.preserve_quotes = True
    with open(profile_path) as handle:
        profile = round_trip.load(handle)
    profile["dispatch"]["network"] = mode
    with open(profile_path, "w") as handle:
        round_trip.dump(profile, handle)


def _set_authored_dispatch_target(triggers_path: Path, port: int) -> None:
    """Point the repo's authored triggers file at the worker's remapped port.

    ``osprey init`` copies the preset's triggers file to the repo root and
    rewrites ``dispatch.triggers`` to name that copy, so this is the file the
    build reads — and it ships naming the DEFAULT worker port, which this
    fixture deliberately does not use. Round-tripped so the file's comments
    survive; the address is the only thing that changes.
    """
    from ruamel.yaml import YAML

    round_trip = YAML()
    round_trip.preserve_quotes = True
    with open(triggers_path) as handle:
        triggers = round_trip.load(handle)
    triggers["dispatcher"]["dispatch_target"] = f"http://dispatch-worker-1:{port}"
    with open(triggers_path, "w") as handle:
        round_trip.dump(triggers, handle)


def _rendered_compose(repo: Path, service: str) -> str:
    path = repo / "build" / "services" / service / "docker-compose.yml"
    assert path.is_file(), f"{service} compose file was not rendered at {path}"
    return path.read_text(encoding="utf-8")


def _rendered_dispatch_target(repo: Path) -> str:
    """The address the DEPLOYED dispatcher routes to.

    Read from the triggers file the build staged into the dispatcher's service
    directory — the copy compose bind-mounts — not the repo-root source, which
    the build never patches.
    """
    path = repo / "build" / "services" / "event_dispatcher" / "triggers.yml"
    assert path.is_file(), f"staged triggers file not rendered at {path}"
    triggers = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str((triggers.get("dispatcher") or {}).get("dispatch_target", ""))


def _assert_render_is_shape(repo: Path, shape: str) -> None:
    """Guard: the render really is the shape this boot claims to be testing.

    Cheap, and it turns the worst failure mode of a parametrized boot — both
    parameters quietly booting the same topology, and both passing — into a
    setup error that names the mismatch.
    """
    for service in ("event_dispatcher", "dispatch_worker"):
        rendered = _rendered_compose(repo, service)
        on_host = "network_mode: host" in rendered
        assert on_host == (shape == HOST), (
            f"{service} rendered network_mode: host = {on_host}, expected shape {shape!r}"
        )


def _teardown_shape(osprey_bin: Path, repo: Path) -> None:
    """Tear the booted shape down, exact-named.

    ``osprey down`` is the verb that owns this; the exact-named removals after
    it are belt-and-suspenders for a ``down`` that failed midway. Both matter
    more here than in a single-boot test: under ``host`` these containers hold
    real host ports, and the next shape cannot start until they are released.
    """
    try:
        _run_osprey(osprey_bin, ["down"], repo, timeout=DOWN_TIMEOUT_SEC)
    except (subprocess.SubprocessError, OSError):
        pass
    for container in (WORKER_CONTAINER, DISPATCHER_CONTAINER):
        _runtime_cli_safe("rm", "-f", container)


@pytest.fixture(scope="module")
def dispatch_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One dispatch-enabled deployment repo, shared by both shapes.

    Init only — no build, no boot. The shape fixture owns those, because they
    are what differs between the two runs.
    """
    _require_responsive_runtime()
    osprey_bin = _find_osprey_console_script()

    base = tmp_path_factory.mktemp("two_shape_boot")
    repo = base / PROJECT_NAME
    override_path = base / "override.yml"
    override_path.write_text(_OVERRIDE_YML, encoding="utf-8")

    init = _run_osprey(
        osprey_bin,
        [
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--override",
            str(override_path),
        ],
        base,
        timeout=INIT_TIMEOUT_SEC,
    )
    assert init.returncode == 0, _fmt("osprey init", init)

    shared_path = repo / ".env.shared"
    assert shared_path.is_file(), "osprey init did not author the repo's .env.shared"
    with open(shared_path, "a", encoding="utf-8") as handle:
        handle.write(_ENV_SHARED_ADDITIONS)
    (repo / ".env").write_text(_ENV_LOCAL, encoding="utf-8")

    triggers_path = repo / "triggers.yml"
    assert triggers_path.is_file(), (
        "osprey init did not copy the preset's triggers file to the repo root"
    )
    _set_authored_dispatch_target(triggers_path, WORKER_PORT)

    # A crashed prior run could have left the exact-named containers behind;
    # adopting one would test the wrong stack. Exact names only.
    for container in (WORKER_CONTAINER, DISPATCHER_CONTAINER):
        _runtime_cli_safe("rm", "-f", container)

    try:
        yield repo
    finally:
        _teardown_shape(osprey_bin, repo)
        _runtime_cli_safe("volume", "rm", WORKSPACE_VOLUME)
        # After the containers, or the network still has endpoints attached.
        _runtime_cli_safe("network", "rm", COMPOSE_NETWORK)
        for image in (PROJECT_IMAGE, DISPATCH_IMAGE):
            _runtime_cli_safe("rmi", "-f", image)


@pytest.fixture(scope="module", params=(BRIDGE, HOST))
def booted_shape(request: pytest.FixtureRequest, dispatch_repo: Path) -> Iterator[tuple[Path, str]]:
    """Re-render the repo in *param*'s shape, boot it, and hand it to the tests.

    Module-scoped and parametrized, so the whole test set runs against the
    bridge boot, that stack comes down, and only then does the host boot start.
    Sequential is not merely cheaper: under ``host`` the two stacks would
    compete for the same host ports.
    """
    shape = request.param
    osprey_bin = _find_osprey_console_script()

    _set_dispatch_network(dispatch_repo / "profile.yml", shape)
    build = _run_osprey(
        osprey_bin,
        ["build", "--skip-deps", "--skip-lifecycle", "--dev"],
        dispatch_repo,
        timeout=BUILD_TIMEOUT_SEC,
    )
    assert build.returncode == 0, _fmt(f"osprey build ({shape})", build)
    _assert_render_is_shape(dispatch_repo, shape)

    _assert_ports_free(DISPATCHER_PORT, WORKER_PORT)

    try:
        up = _run_osprey(osprey_bin, ["up", "-d", "--dev"], dispatch_repo, timeout=UP_TIMEOUT_SEC)
        assert up.returncode == 0, _fmt(f"osprey up ({shape})", up)

        for container, health_url in (
            (DISPATCHER_CONTAINER, f"http://localhost:{DISPATCHER_PORT}/health"),
            (WORKER_CONTAINER, f"http://localhost:{WORKER_PORT}/health"),
        ):
            assert _container_id(container) is not None, (
                f"{container} was not created by 'osprey up' in {shape} mode:\n"
                f"{_dump_diagnostics(f'{shape} up')}"
            )
            _wait_for_service_ready(container, health_url, HEALTH_TIMEOUT_SEC)

        yield dispatch_repo, shape
    finally:
        _teardown_shape(osprey_bin, dispatch_repo)


# ---------------------------------------------------------------------------
# The tests — each runs once per shape
# ---------------------------------------------------------------------------


def test_dispatch_round_trip(booted_shape: tuple[Path, str]) -> None:
    """The dispatcher and the worker reach each other over the shape's own
    addresses: a fired webhook lands as a run in the worker's feed, read back
    through the dispatcher's proxy.

    Three crossings, all over the topology under test — the webhook into the
    dispatcher, the dispatcher's POST to the worker, and the dispatcher's proxy
    of the worker's run feed. The run's terminal STATUS is deliberately not
    asserted: completing it needs a live model provider, while its existence
    needs only that the two services found each other, which is the axis this
    module is about.
    """
    repo, shape = booted_shape

    # Bind the round-trip below to the topology: under `host` the build rewrites
    # the target to the worker's loopback address, under `bridge` it leaves the
    # facility-authored compose DNS name alone.
    target = _rendered_dispatch_target(repo)
    expected = (
        f"http://localhost:{WORKER_PORT}"
        if shape == HOST
        else f"http://dispatch-worker-1:{WORKER_PORT}"
    )
    assert target == expected, f"{shape}: dispatcher routes to {target!r}, expected {expected!r}"

    status, body = _http_in_container(
        DISPATCHER_CONTAINER,
        f"http://localhost:{DISPATCHER_PORT}/webhook/{TRIGGER}",
        method="POST",
        body="{}",
    )
    assert status == 202, f"{shape}: webhook returned {status}: {body}"
    assert json.loads(body).get("dispatched") is True, f"{shape}: {body}"

    # The 202 is the dispatcher accepting the event onto its pool, not the
    # worker acknowledging it — the POST to the worker happens on a pool task
    # afterwards. The run appearing in the feed is what proves it arrived.
    deadline = time.monotonic() + RUN_FEED_TIMEOUT_SEC
    seen: list[dict] = []
    last = "(no feed response yet)"
    while time.monotonic() < deadline:
        status, body = _http_in_container(
            DISPATCHER_CONTAINER, f"http://localhost:{DISPATCHER_PORT}/dashboard/runs"
        )
        if status == 200:
            seen = json.loads(body)
            if any(run.get("trigger_name") == TRIGGER for run in seen):
                return
            last = f"HTTP 200, runs={seen}"
        else:
            # A gateway error means the dispatcher momentarily could not reach
            # the worker upstream; keep polling rather than aborting.
            last = f"HTTP {status}: {body[:200]}"
        time.sleep(3.0)

    raise AssertionError(
        f"{shape}: no {TRIGGER!r} run reached the worker within "
        f"{RUN_FEED_TIMEOUT_SEC:.0f}s (last: {last})\n"
        f"{_dump_diagnostics(f'{shape} run feed')}"
    )


def test_env_chain_local_file_wins_inside_the_worker(booted_shape: tuple[Path, str]) -> None:
    """A key set in BOTH chain files arrives in the worker carrying the ``.env``
    value, and a key set only in ``.env.shared`` arrives at all.

    The second assertion is the positive control for the first: "local wins"
    is satisfied vacuously by a chain that never delivered ``.env.shared``, and
    only the shared-only key can tell those apart. Both are read from the
    container's real environment, which is the only place the question is
    finally settled — the ordering is decided by the compose implementation
    reading the ``env_file`` list, and that implementation differs between the
    runtimes this module runs under.
    """
    _, shape = booted_shape

    assert _container_env_var(WORKER_CONTAINER, SHARED_ONLY_VAR) == SHARED_ONLY_VALUE, (
        f"{shape}: .env.shared did not reach the worker at all — the local-wins "
        f"assertion below would be vacuous"
    )
    assert _container_env_var(WORKER_CONTAINER, CONFLICT_VAR) == CONFLICT_LOCAL_VALUE, (
        f"{shape}: {CONFLICT_VAR} is set in both chain files; the worker must see "
        f"the .env value ({CONFLICT_LOCAL_VALUE!r}), not the .env.shared one "
        f"({CONFLICT_SHARED_VALUE!r})"
    )


def test_bind_addresses_and_exposure_follow_the_shape(booted_shape: tuple[Path, str]) -> None:
    """Under ``host`` both services bind loopback and publish nothing; under
    ``bridge`` both bind every interface inside their own namespace and only the
    dispatcher is published, to the host's loopback.

    The listening addresses are read off ``/proc/net/tcp`` rather than off the
    environment that asked for them, because "the deployment set a bind
    variable" and "the socket took that address" are different facts and only
    the second is the exposure. The bridge half is the positive control: it
    shows the same probe reporting ``0.0.0.0`` when that is genuinely what the
    service bound, so the host half's ``127.0.0.1`` is a real narrowing.
    """
    _, shape = booted_shape

    dispatcher_bind = _container_env_var(DISPATCHER_CONTAINER, "FASTMCP_HOST")
    worker_bind = _container_env_var(WORKER_CONTAINER, "DISPATCH_WORKER_BIND")
    dispatcher_sockets = _listen_addresses(DISPATCHER_CONTAINER, DISPATCHER_PORT)
    worker_sockets = _listen_addresses(WORKER_CONTAINER, WORKER_PORT)
    network_modes = {
        name: _inspect(name, "{{.HostConfig.NetworkMode}}")
        for name in (DISPATCHER_CONTAINER, WORKER_CONTAINER)
    }
    # No `or "{}"` fallback here. A failed inspect returns None, and defaulting it to
    # "{}" would be exactly the value the host branch below asserts — so the claim
    # "under host nothing is published" would pass when the probe never ran, on the
    # half where that claim IS the security property. Demand the read succeeded.
    published = _inspect(DISPATCHER_CONTAINER, "{{json .HostConfig.PortBindings}}")
    assert published is not None, (
        f"could not read PortBindings on {DISPATCHER_CONTAINER} — the exposure "
        f"assertions below cannot be evaluated without it"
    )

    if shape == HOST:
        assert set(network_modes.values()) == {"host"}, (
            f"host shape must put both services in the host namespace: {network_modes}"
        )
        assert dispatcher_bind == "127.0.0.1", f"dispatcher FASTMCP_HOST={dispatcher_bind!r}"
        assert worker_bind == "127.0.0.1", f"worker DISPATCH_WORKER_BIND={worker_bind!r}"
        assert dispatcher_sockets == {"127.0.0.1"}, (
            f"dispatcher listens on {sorted(dispatcher_sockets)} — on the host namespace "
            f"it must bind loopback and nothing else"
        )
        assert worker_sockets == {"127.0.0.1"}, (
            f"worker listens on {sorted(worker_sockets)} — on the host namespace it must "
            f"bind loopback and nothing else"
        )
        # Nothing is published under `host`: there is no port map left to
        # publish, and an emitted one would be a second, un-narrowed exposure.
        assert json.loads(published) in ({}, None), (
            f"host shape published ports: {published} — the render must emit none"
        )
    else:
        assert "host" not in set(network_modes.values()), (
            f"bridge shape must not put a service in the host namespace: {network_modes}"
        )
        assert dispatcher_bind == "0.0.0.0", f"dispatcher FASTMCP_HOST={dispatcher_bind!r}"
        assert worker_bind == "", (
            f"bridge shape must not set DISPATCH_WORKER_BIND (got {worker_bind!r}) — the "
            f"compose network is the boundary there, and the default bind is what makes "
            f"the worker reachable by its service name at all"
        )
        assert dispatcher_sockets == {"0.0.0.0"}, (
            f"dispatcher listens on {sorted(dispatcher_sockets)}"
        )
        assert worker_sockets == {"0.0.0.0"}, f"worker listens on {sorted(worker_sockets)}"
        # The only host-visible surface in this shape, and it is loopback-bound.
        bindings = json.loads(published)
        assert list(bindings) == [f"{DISPATCHER_PORT}/tcp"], (
            f"bridge shape must publish exactly the dispatcher's port: {bindings}"
        )
        assert [entry.get("HostIp") for entry in bindings[f"{DISPATCHER_PORT}/tcp"]] == [
            "127.0.0.1"
        ], f"dispatcher must publish on the host's loopback only: {bindings}"
