"""Reusable turn-key plan-stack deploy configuration (task 4.3 / PROPOSAL FR11).

Builds the shipped deploy config that brings up the Virtual Accelerator +
Bluesky bridge + co-deployed Tiled catalog with
``control_system.type=virtual_accelerator`` and the ``bluesky`` MCP server
enabled (``default_enabled=False`` in the framework registry; opted in here
via ``claude_code.servers.bluesky.enabled``). ``BLUESKY_LAUNCH_TOKEN`` is
minted unconditionally by ``osprey up``, so no execution-method
override is needed to get the agent armed. Corrector setpoints and BPM
readbacks reach the queueserver worker as a DEVICE FILE -- authored at
``<repo>/data/bluesky_devices.yml`` before ``osprey build``, staged by the
build into ``build/services/bluesky/bluesky_devices.yml`` and bind-mounted
into the worker -- derived from the deployment repo's own
``channel_limits.json``, never a hardcoded preset channel (mirrors
``tests/e2e/test_va_substrate_equivalence.py``'s ``_select_sp_echo_pairs``,
restricted here to correctors/BPMs specifically since the ORM plan sweeps
correctors and reads BPMs, not arbitrary writable setpoints).

Authoring that file is why the builders take a ``pre_build`` hook: the device
file has to exist in the repo's source zone by the time ``osprey build`` runs,
and ``osprey init`` must have created that zone first. Anything a caller wants
to put into the repo between the two verbs goes through that hook.

Not a test module itself (no ``test_`` functions) -- the single source of
this config for:
  * ``tests/deployment/test_compose_generator.py``'s ``orm_stack`` render
    gate (this task, Docker-free, via ``build_via_cli_runner``),
  * the real-container round-trip e2e (task 5.2, ``test_orm_roundtrip.py``),
  * the agentic-discovery e2e (tasks 5.3/5.4),
via ``build_project_subprocess`` + ``select_correctors``/``select_bpms``/
``write_devices_file``.

Building this config never touches Docker by itself -- only a subsequent
``osprey up`` does (left to each caller, since only the real e2e/agentic
tests need a live stack).

``select_correctors``/``select_bpms``/``write_devices_file`` delegate to the
canonical derivation in
``osprey.services.bluesky_bridge.substrate_devices`` -- the single source of
this logic, shared with the build path
(``compose_generator._stage_bluesky_devices``, which derives the very same
document for a VA-backed stack that authored no file of its own). Delegating
rather than re-deriving is what keeps a harness-authored device set and a
turn-key derived one from ever disagreeing about what the worker is handed.
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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from click.testing import CliRunner, Result

# Channel Access port the Virtual Accelerator serves on, and the value every
# caller of this module gets unless it passes its own.
#
# This IS freely overridable via `--set virtual_accelerator.port=...`: the
# Control Assistant preset deliberately leaves
# `control_system.connector.virtual_accelerator.gateways.*.port` UNSET, so the
# connector follows `services.virtual_accelerator.port` and moving the deployed
# soft-IOC's port is a one-place edit that carries the connector with it.
# Callers that must coexist with another VA deploy on the same host -- 5064 is
# the tutorial's default and routinely held -- should pass an explicit
# `va_port=` rather than assume this default is free.
VA_CA_PORT = 5064

# Bluesky bridge HTTP port. Distinct from the other e2e modules' pinned
# ports (test_bluesky_deploy.py's 18090, test_va_substrate_equivalence.py's
# 18099, test_tiled_roundtrip.py's 18101) so all four can run concurrently on
# a shared dev machine without a port collision.
BRIDGE_PORT = 18102

# Locally-built service image tags are intentionally NOT module constants here:
# each service compose template defaults its image to
# ``{{ osprey_labels.project_name }}-<service>:local`` (rendered from
# ``resolve_project_name``), and every caller of this module builds under a
# DIFFERENT project name -- so the tag depends on the caller's project_name.
# Derive it at the call site via the helpers below rather than hardcode a
# host-global name that is wrong for any non-default project. Container names
# follow the same ``<project>-<service>`` rule -- derive those at the call site
# too (e.g. ``f"{project_name}-bluesky-bridge"``).


def project_prefix(project_name: str) -> str:
    """The ``<project>`` prefix compose gives every container and locally-built
    image of a deploy, resolved exactly as the templates resolve it.

    Container names (``<project>-bluesky-bridge``) and image tags
    (``<project>-va:local``) are both built from it, so anything that must name
    a deployed container -- a health probe, a log dump -- derives it here
    rather than hardcoding a host-global name that is wrong for any other
    project.
    """
    from osprey.deployment.compose_generator import resolve_project_name

    return str(resolve_project_name({"project_name": project_name}))


def _service_image(project_name: str, service: str) -> str:
    """Derive a locally-built ``<project>-<service>:local`` image tag the way
    the service compose templates do.

    The templates default their image to
    ``{{ osprey_labels.project_name }}-<service>:local`` -- rendered from
    :func:`osprey.deployment.compose_generator.resolve_project_name` -- so a
    caller that force-rebuilds via ``docker rmi -f`` must target that SAME
    project-prefixed tag, never a host-global name.
    """
    return f"{project_prefix(project_name)}-{service}:local"


def bridge_image(project_name: str) -> str:
    """``<project>-bluesky-bridge:local`` for ``project_name``."""
    return _service_image(project_name, "bluesky-bridge")


def va_image(project_name: str) -> str:
    """``<project>-va:local`` for ``project_name``."""
    return _service_image(project_name, "va")


def panels_image(project_name: str) -> str:
    """``<project>-bluesky-web:local`` for ``project_name``."""
    return _service_image(project_name, "bluesky-web")


def force_image_rebuild(*images: str) -> None:
    """Remove locally-built images so a later ``osprey up --dev``
    rebuilds them from CURRENT source (``osprey up`` does not pass ``--build``
    to compose, so it would otherwise reuse a stale cached image). Exact-named
    images only — never a wildcard, never a prune, never a volume operation.

    No-op when ``E2E_REUSE_IMAGES`` is set, for fast local iteration on a warm
    cache; never set it in CI, where a source change must always rebuild.

    Bounded and non-fatal: a removal that fails or hangs only means a stale
    image survives, which ``osprey up`` will rebuild over anyway — never worth
    wedging a fixture before a container is even started.
    """
    if os.environ.get("E2E_REUSE_IMAGES"):
        return
    for image in images:
        try:
            subprocess.run(
                ["docker", "rmi", "-f", image], capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            continue


BUILD_TIMEOUT_SEC = 300

# A small, concurrency-friendly corrector/BPM count for the render + the
# real-container round-trip gate. The agentic e2e scenarios (5.3/5.4) name
# their own errant device/location and don't depend on this count.
DEFAULT_CORRECTOR_COUNT = 4
DEFAULT_BPM_COUNT = 4


# The archive every VA lane deploys, shrunk to what a lane actually reads.
#
# The control-assistant preset declares a `va_archiver:` block sized for a
# tutorial deployment -- a month of history behind two dense days -- and
# `osprey up` writes every sample of it into the store before the stack answers.
# No lane here reads that history; they need the store to exist, the recorder to
# be recording, and the two-tier boundary to be somewhere a contract can find it.
# Two days of retention behind a two-hour dense head is about a sixteenth of the
# samples: seconds of seeding instead of a minute, and a store sized to match.
#
# One snippet shared by all four VA lanes (`override_yaml` below, plus the three
# that write their own override) rather than four hand-copied blocks that drift.
# Deep-merged onto the preset's block, so `host:` and both cadences keep the
# values the preset ships -- only the two span knobs move.
VA_ARCHIVER_CI_KNOBS = "va_archiver:\n  retention_days: 2\n  hot_span_hours: 2\n"


def override_yaml() -> str:
    """FR11's ``--override`` YAML content: VA control system + the bluesky MCP
    server.

    ``dispatch: null`` drops control-assistant's default event-dispatcher
    stack (Node + Claude CLI image) -- irrelevant to the plan stack and far
    slower to build than the VA/bridge images already are (mirrors
    test_va_substrate_equivalence.py / test_tiled_roundtrip.py).

    ``modules.web_terminals.enabled: false`` drops the preset's per-persona
    web-terminal stack (two persona images + nginx, all built locally) for
    the same reason: nothing in the plan stack touches persona routing, and
    that coverage lives in the dedicated web-terminals lanes
    (control-assistant-demo-e2e, multi-user-deploy-lifecycle-e2e,
    tests/e2e/web_terminals/). One dotted LEAF key on purpose -- the preset
    sets the whole ``modules.web_terminals`` subtree as a single dotted key,
    and overriding just ``.enabled`` leaves its siblings intact, whereas a
    nested ``modules:`` mapping would wholesale-replace the subtree (see the
    preset's own comment above its ``modules.web_terminals`` block).

    ``VA_ARCHIVER_CI_KNOBS`` shrinks the archive the preset's ``va_archiver:``
    block declares to a CI-sized one -- see the constant for why. It trails
    ``dispatch: null`` so it stays outside the ``config:`` block (both are
    top-level profile keys, and ``test_bluesky_web_deploy`` splices its port
    moves in ahead of that line).

    Written as flat dotted-string keys under ``config:`` (matching the
    preset's own convention), not a `--set config.control_system.type=...`
    CLI override -- `--set` builds a NESTED dict for every dotted segment,
    which would replace the entire `control_system:`/`execution:` block
    instead of overriding just one field.
    """
    return (
        "config:\n"
        "  control_system.type: virtual_accelerator\n"
        "  claude_code.servers.bluesky.enabled: true\n"
        "  modules.web_terminals.enabled: false\n"
        "dispatch: null\n" + VA_ARCHIVER_CI_KNOBS
    )


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base``, returning a new dict.

    Nested mappings merge key-by-key; every other value (scalar, list, ``None``)
    replaces whatever ``base`` held. Neither input is mutated.
    """
    merged = dict(base)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def merged_override_yaml(extra_config: dict[str, Any]) -> str:
    """``override_yaml()`` with ``extra_config`` deep-merged into it.

    Reaches the keys ``build_args`` has no ``--set`` hook for -- e.g. the
    postgres/openobserve/tiled/panels HOST ports a module must move to run
    concurrently with another deployed stack::

        merged_override_yaml({"config": {"services.postgresql.port_host": 15433}})

    Round-trips through YAML (``safe_load`` -> merge -> ``safe_dump``) rather
    than returning ``override_yaml()``'s hand-written text, so nothing here
    guarantees byte-identity with it: an empty ``extra_config`` happens to
    re-emit the same bytes today, but that is incidental to PyYAML's current
    formatting, not a promise. Callers that need the exact hand-written bytes
    should use ``override_yaml()`` directly -- only callers that actually need
    a merge take this path (see ``init_args``).
    """
    base = yaml.safe_load(override_yaml()) or {}
    return yaml.safe_dump(_deep_merge(base, extra_config), sort_keys=False)


def init_args(
    project_name: str,
    *,
    override_path: Path,
    output_dir: Path,
    bridge_port: int = BRIDGE_PORT,
    va_port: int = VA_CA_PORT,
    provider: str | None = None,
    model: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> list[str]:
    """``osprey init`` CLI args (sans the leading ``init`` token) for FR11's
    turn-key plan-stack deployment.

    The stack is materialized in two steps, because the surface has two:
    ``osprey init`` writes the deployment repo's source zone from the preset
    plus these overrides, and a later ``osprey build`` renders ``build/`` from
    it. This function covers the first step only; both builders below run the
    second. ``--no-git`` because every caller works in a throwaway directory
    and none of them reads the history.

    Works both as ``CliRunner().invoke(init, init_args(...))`` (in-process, no
    Docker -- see ``build_via_cli_runner``) and as
    ``[osprey_bin, "init", *init_args(...)]`` (subprocess, for a real
    ``osprey up`` afterward -- see ``build_project_subprocess``).

    ``provider``/``model``, when given, append ``--set provider=<provider>``
    and/or ``--set model=<model>`` overrides -- e.g. an agentic-discovery
    caller that must pin an explicit provider rather than let the
    control-assistant preset's own default apply silently (this project's
    "no default provider" convention). Left ``None`` by default: nothing is
    appended and the preset's own provider/model apply unchanged, so the
    default deploy shape is unaffected by these params.

    ``extra_config``, when given, is deep-merged into ``override_yaml()`` and
    the result REWRITES ``override_path`` -- the way to reach config keys that
    have no ``--set`` hook here (postgres/openobserve/tiled/panels host ports).
    Writing rather than appending another CLI flag keeps a single ``--override``
    file, which is what ``osprey build`` wants. That rewrite OVERWRITES whatever
    the caller previously wrote to ``override_path``, so a caller that hand-rolls
    its own override text (as the bluesky-web e2e does) must pass its
    additions here rather than pre-writing them. Empty or ``None`` is a no-op:
    ``override_path`` is left exactly as the caller wrote it, byte for byte.
    """
    if extra_config:
        override_path.write_text(merged_override_yaml(extra_config), encoding="utf-8")

    args = [
        str(output_dir / project_name),
        "--preset",
        "control-assistant",
        "--no-git",
        "--override",
        str(override_path),
        "--set",
        f"virtual_accelerator.port={va_port}",
        "--set",
        f"bluesky.port={bridge_port}",
        "--set",
        "bluesky.tiled_enabled=true",
    ]
    if provider is not None:
        args += ["--set", f"provider={provider}"]
    if model is not None:
        args += ["--set", f"model={model}"]
    return args


def build_via_cli_runner(
    runner: CliRunner,
    tmp_path: Path,
    *,
    project_name: str = "orm-stack",
    bridge_port: int = BRIDGE_PORT,
    va_port: int = VA_CA_PORT,
    pre_build: Callable[[Path], None] | None = None,
) -> Path:
    """In-process ``osprey init`` + ``osprey build`` (``CliRunner``, no
    subprocess/Docker) for fast render-only gates -- see
    ``tests/cli/test_va_default_config.py`` for the same in-process pattern.
    Renders config.yml, the service compose templates, and the Claude Code
    artifacts (``.mcp.json`` included); never starts a container.

    Returns the RENDER -- ``<repo>/build`` -- because that is the directory
    holding config.yml and the compose files a caller goes on to read. The repo
    root above it is ``result.parent``.

    ``pre_build``, when given, is called with the deployment REPO after
    ``osprey init`` has written it and before ``osprey build`` renders it --
    the only window in which a caller can put a file into the repo's source
    zone and still have the build stage it (``<repo>/data/bluesky_devices.yml``
    is what every plan-stack caller writes there; see
    :func:`write_devices_file`). It must not run BEFORE ``init``: init copies
    the preset's ``data/`` into place without ``dirs_exist_ok``, so a
    pre-created ``data/`` makes the copy fail outright.
    """
    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    override_path = tmp_path / "override.yml"
    override_path.write_text(override_yaml(), encoding="utf-8")

    repo = tmp_path / project_name
    result: Result = runner.invoke(
        init,
        init_args(
            project_name,
            override_path=override_path,
            output_dir=tmp_path,
            bridge_port=bridge_port,
            va_port=va_port,
        ),
    )
    if result.exit_code != 0:
        raise AssertionError(f"osprey init failed (exit={result.exit_code}):\n{result.output}")

    if pre_build is not None:
        pre_build(repo)

    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    if result.exit_code != 0:
        raise AssertionError(f"osprey build failed (exit={result.exit_code}):\n{result.output}")
    return repo / "build"


def find_osprey_console_script() -> Path:
    """Locate the ``osprey`` console script for subprocess invocations.

    Centralized here since every real-container e2e that builds this stack
    (task 5.2, and the agentic e2e in 5.3/5.4) needs it, mirroring the
    identical helper duplicated in test_va_substrate_equivalence.py /
    test_tiled_roundtrip.py / test_bluesky_deploy.py.
    """
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def build_project_subprocess(
    project_name: str,
    *,
    output_dir: Path,
    bridge_port: int = BRIDGE_PORT,
    va_port: int = VA_CA_PORT,
    timeout: int = BUILD_TIMEOUT_SEC,
    provider: str | None = None,
    model: str | None = None,
    extra_config: dict[str, Any] | None = None,
    pre_build: Callable[[Path], None] | None = None,
) -> Path:
    """Real ``osprey init`` + ``osprey build`` subprocesses for a deployment a
    caller will later ``osprey up`` (that step needs Docker; these don't -- they
    only render config.yml/compose templates/.mcp.json, same as
    ``build_via_cli_runner``, but out-of-process so ``--dev``/``osprey up``
    against the resulting repo behave exactly as they would for an operator
    running the real CLI).

    Returns the deployment REPO, not its render: the start verbs are repo-scoped
    and this is what a caller hands to ``osprey up --repo``.

    ``provider``/``model``/``extra_config`` thread straight through to
    ``init_args`` (see its docstring). All ``None`` by default, which preserves
    the exact default deploy shape -- including a byte-identical override file
    (an empty ``extra_config`` is likewise a no-op).

    ``pre_build``, when given, is called with the deployment REPO after
    ``osprey init`` has written it and before ``osprey build`` renders it --
    the only window in which a caller can put a file into the repo's source
    zone and still have the build stage it (``<repo>/data/bluesky_devices.yml``
    is what every plan-stack caller writes there; see
    :func:`write_devices_file`). It must not run BEFORE ``init``: init copies
    the preset's ``data/`` into place without ``dirs_exist_ok``, so a
    pre-created ``data/`` makes the copy fail outright.
    """
    osprey_bin = find_osprey_console_script()
    override_path = output_dir / "override.yml"
    override_path.write_text(override_yaml(), encoding="utf-8")

    cmd = [
        str(osprey_bin),
        "init",
        *init_args(
            project_name,
            override_path=override_path,
            output_dir=output_dir,
            bridge_port=bridge_port,
            va_port=va_port,
            provider=provider,
            model=model,
            extra_config=extra_config,
        ),
    ]
    repo = output_dir / project_name

    def run_step(label: str, argv: list[str]) -> None:
        result = subprocess.run(
            argv,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CLAUDECODE": ""},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} failed (rc={result.returncode}):\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )

    run_step("osprey init", cmd)

    # STRICTLY between the two verbs -- see the ``pre_build`` paragraph above.
    if pre_build is not None:
        pre_build(repo)

    run_step(
        "osprey build",
        [
            str(osprey_bin),
            "build",
            "--repo",
            str(repo),
            "--skip-deps",
            "--skip-lifecycle",
            "--dev",
        ],
    )
    return repo


def channel_limits(project_dir: Path) -> dict[str, Any]:
    """The project's own ``data/channel_limits.json`` — the source of every
    device name the plan-stack e2es use, so no preset channel is ever
    hardcoded.

    Callers pass the deployment REPO. ``osprey build`` copies ``<repo>/data``
    into the build zone verbatim, so ``<repo>/data/channel_limits.json`` and
    the render's ``build/data/channel_limits.json`` are the same bytes and name
    the same channels to the deployed containers — but only the repo copy
    exists before the build, which is when a ``pre_build`` hook has to choose
    the plan devices.
    """
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


def minted_launch_token(project_dir: Path) -> str:
    """The ``BLUESKY_LAUNCH_TOKEN`` ``osprey up`` minted into the
    project ``.env``.

    Callers supply no token of their own: the deploy path mints one for every
    deployed service that declares it, and the arming action on the queue is
    gated by exactly that value.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = project_dir / ".env"
    assert env_path.is_file(), f"no .env written at {env_path} — token was not minted"
    token = parse_dotenv_file(env_path).get("BLUESKY_LAUNCH_TOKEN")
    assert token, (
        "BLUESKY_LAUNCH_TOKEN missing/empty in the project .env — `osprey up` "
        "mints it for every deployed service that declares it"
    )
    return token


def wait_for_health(url: str, timeout: float) -> None:
    """Poll ``url`` until it answers HTTP 200, or fail after ``timeout``
    seconds with the last error seen."""
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


#: The bridge's compose SERVICE name -- the key under ``services:`` in
#: ``services/bluesky/docker-compose.yml.j2``, and what a compose subcommand
#: takes. Distinct from the CONTAINER name the same template pins with
#: ``container_name:`` (``<project>-bluesky-bridge``, see
#: :func:`bridge_container`), which is what ``docker inspect``/``docker logs``
#: take.
BRIDGE_SERVICE = "bluesky-bridge"

#: Budget for the ``compose restart`` call itself. A restart stops and starts
#: one already-built container -- no image build, no dependency resolution --
#: so this is headroom over a healthy stop/start, not a build allowance.
RESTART_TIMEOUT_SEC = 120

#: Budget for the bridge to answer ``/health`` again after a restart. Far
#: shorter than a cold deploy's health wait: the image exists and the container
#: exists, so this covers one process start plus its manager reconnect.
RESTART_HEALTH_TIMEOUT_SEC = 180.0


def bridge_container(project_name: str) -> str:
    """``<project>-bluesky-bridge`` for ``project_name`` -- the CONTAINER name
    the bridge compose template pins with ``container_name:``.

    Derived from :func:`project_prefix` rather than hardcoded, for the same
    reason the image helpers are: the name is project-scoped, and a host-global
    literal would be wrong for every other deployed project.
    """
    return f"{project_prefix(project_name)}-{BRIDGE_SERVICE}"


def _container_started_at(container: str) -> str:
    """``.State.StartedAt`` of ``container``, as docker reports it.

    Compared across a restart to prove the process was actually replaced -- the
    whole point of :func:`restart_bridge`, and the one thing a zero exit code
    from compose does not establish on its own.
    """
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.StartedAt}}", container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"could not inspect {container} (rc={result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def restart_bridge(
    project_name: str,
    *,
    bridge_url: str,
    health_timeout: float = RESTART_HEALTH_TIMEOUT_SEC,
) -> None:
    """Restart ONLY the bluesky-bridge container of a deployed stack, then wait
    for it to answer ``/health`` again.

    Why an e2e needs this: the bridge holds a run's rows in an in-process ring
    buffer (``live_rows``), and nothing an HTTP client can call empties it --
    ``live_rows._clear()`` is in-process, and eviction needs 50 further runs.
    Restarting the process is the only way a test can drop that buffer and so
    force the read paths (``/runs/{id}/data``, ``/runs/{id}/figure``) onto their
    Tiled branch. Everything else keeps running: the queueserver, its Redis, the
    Tiled catalog and the Virtual Accelerator are untouched, so the completed
    run's documents are still in the catalog and the plan stack is still
    deployed.

    Restarting the bridge cannot lose documents, because the bridge is not what
    writes them: the ``TiledWriter`` subscription lives in the QUEUESERVER
    WORKER (``qserver_startup``), on the other side of this restart. What dies
    with the bridge process is its READ-side live buffer, which is exactly the
    state a test wants gone.

    WHAT THIS PROVES, AND WHAT IT DOES NOT. The health wait proves the bridge
    process is UP and serving. It says nothing about TiledWriter having flushed
    the run to the catalog -- that is a separate, asynchronous fact. A caller
    that needs the Tiled branch to actually answer must poll for it (a bounded
    poll on the route it cares about until ``source == "tiled"``), never infer
    it from this function returning. That poll belongs to the caller, whose
    route knows what "answered from Tiled" looks like -- see
    ``test_orm_roundtrip.py``, which waits on ``/runs/{id}/figure`` reporting
    both ``source == "tiled"`` and ``partial == false``.

    Nor does it restore enqueue readiness: the RE worker environment is opened
    by the bridge off its readiness path, so a caller that goes on to enqueue
    after a restart must re-wait via ``_queue_drive.wait_for_worker_environment``.

    Container safety: the compose invocation is pinned to THIS deployment's
    project (``-p <project>``, resolved exactly as the deploy pins
    ``COMPOSE_PROJECT_NAME``) and names one exact service, so it can never reach
    another session's containers. No ``-f`` is passed: compose resolves the
    project from the running containers' own labels, which is what makes this a
    one-service operation on a live stack rather than a re-read of a render.

    :param project_name: The name the stack was built/deployed under (the raw
        name, as passed to ``build_project_subprocess``) -- the compose project
        and container names are derived from it here.
    :param bridge_url: The bridge's base URL, e.g. ``http://localhost:18102``.
        ``/health`` is appended to it.
    :param health_timeout: Seconds to wait for ``/health`` after the restart.
    :raises AssertionError: if the restart fails, if the container was not
        actually replaced, or if ``/health`` does not come back.
    """
    container = bridge_container(project_name)
    started_before = _container_started_at(container)

    # --no-deps is what makes "only the bridge" structural rather than
    # incidental: compose's default service selection follows depends_on edges
    # whose `restart: true` flag is set, and the bluesky-web template
    # declares `depends_on: bluesky-bridge` in this same project. No template
    # sets that flag today, so the selection happens to be one service -- an
    # accident this flag stops depending on.
    restart = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            project_prefix(project_name),
            "restart",
            "--no-deps",
            BRIDGE_SERVICE,
        ],
        capture_output=True,
        text=True,
        timeout=RESTART_TIMEOUT_SEC,
    )
    if restart.returncode != 0:
        raise AssertionError(
            f"compose restart {BRIDGE_SERVICE} failed (rc={restart.returncode}):\n"
            f"--- stdout ---\n{restart.stdout}\n--- stderr ---\n{restart.stderr}"
        )

    started_after = _container_started_at(container)
    if started_after == started_before:
        raise AssertionError(
            f"compose reported success but {container} was not restarted "
            f"(StartedAt still {started_before}) -- the in-process live buffer "
            "is therefore still populated and any Tiled-branch assertion after "
            "this would be vacuous"
        )

    wait_for_health(f"{bridge_url}/health", health_timeout)


def select_correctors(
    limits: dict[str, Any], count: int | None = DEFAULT_CORRECTOR_COUNT
) -> dict[str, tuple[str, str]]:
    """Derive ``count`` SR corrector (HCM/VCM) ``:SP``/``:RB`` pairs from the
    deployed project's own ``channel_limits.json`` -- never a hardcoded
    preset channel.

    Restricted to the pyat-coupled corrector partition (a write actually
    steers the beam via the AT lattice model) rather than any writable
    ``:SP``: the ORM plan sweeps correctors specifically, so a generic
    sp-echo pair (physics-free) would be the wrong device class here.

    If ``count`` is ``None``, returns the FULL available pyat-coupled
    corrector set instead of a fixed-size slice -- no assertion is raised in
    that case, regardless of how many pairs are found.

    Returns a dict of ``sp_address -> (sp_address, rb_address)`` -- the setpoint's
    device name is its own ``:SP`` address -- ready to hand to
    :func:`write_devices_file` as the device file's ``settables``.

    Thin wrapper: delegates to the canonical
    ``osprey.services.bluesky_bridge.substrate_devices.select_correctors``
    (same logic; this module keeps the ``DEFAULT_CORRECTOR_COUNT`` default
    the e2e suite has always used, versus the product module's ``None``/
    full-set default).
    """
    from osprey.services.bluesky_bridge.substrate_devices import (
        select_correctors as _select_correctors,
    )

    return _select_correctors(limits, count)


def select_bpms(limits: dict[str, Any], count: int | None = DEFAULT_BPM_COUNT) -> dict[str, str]:
    """Derive ``count`` SR BPM ``:POSITION:X``/``:POSITION:Y`` readbacks from
    the deployed project's own ``channel_limits.json`` -- same generic,
    no-hardcoded-channel convention as ``select_correctors``.

    If ``count`` is ``None``, returns the FULL available pyat-coupled BPM set
    instead of a fixed-size slice -- no assertion is raised in that case.

    Returns a dict of ``read_address -> read_address`` -- the readback's
    device name is its own read address -- ready to hand to
    :func:`write_devices_file` as the device file's ``readables``.

    Thin wrapper: delegates to the canonical
    ``osprey.services.bluesky_bridge.substrate_devices.select_bpms`` (same
    logic; this module keeps the ``DEFAULT_BPM_COUNT`` default the e2e suite
    has always used, versus the product module's ``None``/full-set default).
    """
    from osprey.services.bluesky_bridge.substrate_devices import select_bpms as _select_bpms

    return _select_bpms(limits, count)


def write_devices_file(
    repo: Path,
    *,
    correctors: dict[str, tuple[str, str]],
    bpms: dict[str, str],
    launch_token: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Author the queueserver worker's plan devices at
    ``<repo>/data/bluesky_devices.yml`` -- BEFORE ``osprey build``, from a
    ``pre_build`` hook -- and return the document written.

    ``repo`` is the repo ROOT, not its render. That is deliberate and is the
    whole reason the hook exists: the build copies ``<repo>/data`` into the
    build zone verbatim, and only then resolves ``bluesky.devices_file``
    against the rendered config -- so a file authored here is the AUTHORED
    device set the render stages into
    ``build/services/bluesky/bluesky_devices.yml`` and the worker mounts. Write
    it after the build and the render has already chosen (deriving its own set
    from the project's channel limits) and nothing picks this file up.

    ``correctors``/``bpms`` carry the same shapes
    :func:`select_correctors`/:func:`select_bpms` return, and select WHICH
    channels reach the worker: only these devices are registered, and each one
    is a Channel Access connection the RE worker opens at startup.

    ``launch_token``, if given, is written to the repo's ``.env``. The deploy
    path normally mints one on ``osprey up``; callers that need a deterministic
    value for a scripted launch call supply their own.

    The document itself is produced and written by the canonical
    ``osprey.services.bluesky_bridge.substrate_devices`` -- the same producer
    the build path uses -- rather than assembled here, so a harness-authored
    device set and a turn-key derived one are byte-identical for the same
    channels. That producer derives from a channel-limits mapping, so the
    requested devices reach it as the SLICE of channel keys naming exactly them
    (it reads keys, never values), and the document it returns is checked to
    name exactly what was asked for.
    """
    from osprey.cli.build_profile_schema import BlueskyConfig
    from osprey.services.bluesky_bridge.devices._specs_from_file import (
        READABLES_KEY,
        SETTABLES_KEY,
    )
    from osprey.services.bluesky_bridge.substrate_devices import (
        write_devices_file as _write_devices_file,
    )

    limits_slice: dict[str, Any] = {address: {} for pair in correctors.values() for address in pair}
    limits_slice.update({address: {} for address in bpms.values()})

    # ``BlueskyConfig.devices_file`` is authored relative to the rendered
    # config's directory; joining that same relative path onto the REPO root
    # lands it inside ``<repo>/data``, which the build copies into the build
    # zone -- so the render finds it exactly where it looks.
    devices_path = repo / BlueskyConfig.devices_file
    devices_path.parent.mkdir(parents=True, exist_ok=True)
    document = _write_devices_file(devices_path, limits_slice)

    written_settables = {entry["name"] for entry in document[SETTABLES_KEY]}
    written_readables = {entry["name"] for entry in document[READABLES_KEY]}
    if written_settables != set(correctors) or written_readables != set(bpms):
        raise AssertionError(
            "the authored device file does not name the requested devices "
            f"(settables {sorted(written_settables)} vs {sorted(correctors)}, "
            f"readables {sorted(written_readables)} vs {sorted(bpms)}) -- a "
            "requested channel is one the canonical derivation does not "
            "classify as a corrector/BPM"
        )

    if launch_token:
        env_path = repo / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        env_path.write_text(f"{existing}BLUESKY_LAUNCH_TOKEN={launch_token}\n", encoding="utf-8")

    return document


def assert_devices_authored(correctors: dict[str, tuple[str, str]], bpms: dict[str, str]) -> None:
    """Fail HERE if the ``pre_build`` device-authoring hook never ran.

    Every plan-lane fixture seeds its corrector/BPM dicts empty and fills them
    from inside its ``pre_build`` hook, so a hook that was dropped -- from the
    builder call, or by a builder that stopped invoking it -- leaves both empty
    and the fixture goes on to deploy a browse-only worker. Without this the
    first symptom is a plan a hundred lines later naming no devices, which reads
    as a queueserver fault rather than as a fixture that skipped a step.

    The counterpart of ``test_bump_roundtrip``'s ``assert stack is not None``,
    for the fixtures whose hook fills dicts rather than building an object.
    """
    assert correctors and bpms, (
        "the pre-build hook did not run -- no plan devices were authored, so "
        "`osprey build` staged no device file and the queueserver worker comes "
        "up browse-only"
    )


def seed_repo_env(repo: Path) -> None:
    """Give the deployment repo the ``.env`` ``osprey up`` refuses to start
    without.

    The repo root's ``.env`` is the deployment's whole secret store and the
    file every compose invocation is pointed at, so ``up`` aborts when it is
    absent. ``osprey init`` writes one only when the shell exports a key for
    the profile's provider, which the plan-stack lanes do not need — this is
    the ``cp .env.example .env`` the CLI itself recommends, done for the
    operator. Everything ``up`` mints (the launch token, the service secrets)
    is appended to whatever is here.

    Its own step because nothing else creates that file any more: the plan
    devices moved out of ``.env`` and into the mounted device file, so the
    deployment's secret store and its device set are now two separate concerns
    and neither may quietly depend on the other having run.
    """
    env_path = repo / ".env"
    if not env_path.exists():
        shutil.copy(repo / ".env.example", env_path)


def staged_devices_file(repo: Path) -> Path:
    """Where ``osprey build`` stages the worker's device file inside the render.

    The compose template mounts this literal path, so the name is part of the
    contract and is imported from the generator rather than re-spelled here.
    """
    from osprey.deployment.compose_generator import BLUESKY_DEVICES_FILENAME

    return repo / "build" / "services" / "bluesky" / BLUESKY_DEVICES_FILENAME


def staged_devices(repo: Path) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Read the device file the BUILD staged, as ``(correctors, bpms)`` -- the
    same shapes the selectors return.

    For the turn-key lanes, which author no device file of their own and so
    prove that the build derives one from the deployment's own channel limits.
    Read back from the staged file rather than re-derived here, so the devices
    a test names are exactly the ones the deployed worker registered and a
    change in that derivation surfaces as a real failure instead of a silently
    diverging second copy of the logic.
    """
    from osprey.services.bluesky_bridge.devices._specs_from_file import (
        READABLES_KEY,
        SETTABLES_KEY,
    )

    path = staged_devices_file(repo)
    if not path.is_file():
        raise AssertionError(
            f"the build staged no plan device file at {path} -- the worker came "
            "up browse-only, so no plan this test composes can name a device"
        )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a device document mapping: {document!r}")

    correctors = {
        entry["name"]: (entry["setpoint"], entry.get("readback", entry["setpoint"]))
        for entry in document.get(SETTABLES_KEY) or []
    }
    bpms = {entry["name"]: entry["pv"] for entry in document.get(READABLES_KEY) or []}
    return correctors, bpms
