"""Reusable turn-key plan-stack deploy configuration (task 4.3 / PROPOSAL FR11).

Builds the shipped deploy config that brings up the Virtual Accelerator +
Bluesky bridge + co-deployed Tiled catalog with
``control_system.type=virtual_accelerator`` and the ``bluesky`` MCP server
enabled (``default_enabled=False`` in the framework registry; opted in here
via ``claude_code.servers.bluesky.enabled``). ``BLUESKY_LAUNCH_TOKEN`` is
minted unconditionally by ``osprey up``, so no execution-method
override is needed to get the agent armed. The queueserver worker's device
file is the build's Bluesky view of the deployment's facility file, staged
unchanged into ``build/services/bluesky/bluesky_devices.yml``
(``compose_generator._stage_bluesky_devices``) and bind-mounted into the
worker: every setpoint channel a settable, every readback channel a readable.
A lane chooses the correctors and BPMs its plans drive from that staged view
(:func:`select_correctors`/:func:`select_bpms`), never from a hardcoded preset
channel, restricted to the channels the repo's bindings document binds as a
kick or as a monitor, since the ORM plan sweeps correctors and reads monitors
rather than arbitrary writable setpoints.

The builders take a ``pre_build`` hook for a lane that edits the repo's source
zone -- its facility tree, its bindings -- after ``osprey init`` has created
that zone and before ``osprey build`` renders it.

Not a test module itself (no ``test_`` functions) -- the single source of
this config for:
  * ``tests/deployment/test_compose_generator.py``'s ``orm_stack`` render
    gate (this task, Docker-free, via ``build_via_cli_runner``),
  * the real-container round-trip e2e (task 5.2, ``test_orm_roundtrip.py``),
  * the agentic-discovery e2e (tasks 5.3/5.4),
via ``build_project_subprocess`` + ``select_correctors``/``select_bpms``.

Building this config never touches Docker by itself -- only a subsequent
``osprey up`` does (left to each caller, since only the real e2e/agentic
tests need a live stack).

Where the work is split. The build's view says which channels the worker
holds; ``select_correctors``/``select_bpms`` are the HARNESS's own, because
which of them a corrector-sweeping plan can do physics with is a question a
lane asks of the repo's own bindings document (:func:`repo_bindings`) and has
no place in a framework that must stay facility-agnostic. The limits table is
the view of ``data/facility/limits.yaml``, the same table the render enforces,
readable before the build through :func:`channel_limits`.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml

from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths
from tests.e2e.profile_edits import set_pairs

if TYPE_CHECKING:
    from click.testing import CliRunner, Result

    from osprey.services.virtual_accelerator.bindings import BindingsDocument

#: What :func:`_keyed_by_address` keys -- a corrector ``(sp, rb)`` pair or a
#: BPM address, both of which name their device by an address the selector
#: reads off the item itself.
_T = TypeVar("_T")

# Channel Access port the Virtual Accelerator serves on, and the value every
# caller of this module gets unless it passes its own.
#
# This IS freely overridable via `--set virtual_accelerator.port=...`: the
# Control Assistant preset deliberately leaves
# `control_system.connector.virtual_accelerator.gateways.*.port` UNSET, so the
# connector follows `services.virtual_accelerator.port` and moving the deployed
# soft-IOC's port is a one-place edit that carries the connector with it.
# The default is an ephemeral free port rather than 5064: the tutorial default
# is routinely held by a real deployment on a dev host (`port_layout.
# CA_DEFAULT_PORT` keeps VA instance 1 there on purpose), and every caller of
# this module already gets the one value plumbed through both the service port
# and the connector. Pass an explicit `va_port=` to pin one.
#
# The pvAccess port the VA publishes its model surface on moves the same way,
# through `virtual_accelerator.pva_port`. Its default, 5075, is the one every
# other VA on the host publishes too, so every caller gets a free one reserved
# here. Pass an explicit `va_pva_port=` to pin one.


def _reserve_free_ports(count: int) -> tuple[int, ...]:
    """Reserve ``count`` distinct free loopback ports.

    Every socket stays bound until all are, so the ports are distinct by
    construction rather than by the kernel's choice.
    """
    with contextlib.ExitStack() as stack:
        socks = [
            stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            for _ in range(count)
        ]
        for sock in socks:
            sock.bind(("127.0.0.1", 0))
        return tuple(sock.getsockname()[1] for sock in socks)


# import-time required because VA_CA_PORT and VA_PVA_PORT bind into `va_port=`
# and `va_pva_port=` default arguments across the importing e2e modules, which
# evaluate at import.
VA_CA_PORT, VA_PVA_PORT = _reserve_free_ports(2)

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
# One mapping shared by all four VA lanes (`profile_edits` below, plus the
# three that state their own edits) rather than four hand-copied blocks that
# drift. Stated as top-level profile keys, so the two span knobs replace the
# preset's values for those two keys while `host:` and both cadences keep what
# the preset ships.
VA_ARCHIVER_CI_KNOBS: dict[str, Any] = {"va_archiver": {"retention_days": 2, "hot_span_hours": 2}}


def profile_edits() -> dict[str, Any]:
    """The lane's profile pins: VA control system + the bluesky MCP server.

    ``dispatch: None`` drops control-assistant's default event-dispatcher
    stack (Node + Claude CLI image) -- irrelevant to the plan stack and far
    slower to build than the VA/bridge images already are (mirrors
    test_va_substrate_equivalence.py / test_tiled_roundtrip.py). The preset's
    ``services:`` block holds the record archive, which runs the same project
    image, so it goes with the dispatch stack: ``services: {}`` is the spelling
    because a single service cannot be nulled.

    ``modules.web_terminals.enabled: False`` drops the preset's per-persona
    web-terminal stack (two persona images + nginx, all built locally) for
    the same reason: nothing in the plan stack touches persona routing, and
    that coverage lives in the dedicated web-terminals lanes
    (control-assistant-demo-e2e, multi-user-deploy-lifecycle-e2e,
    tests/e2e/web_terminals/). One dotted LEAF key on purpose -- the preset
    sets the whole ``modules.web_terminals`` subtree as a single dotted key,
    and stating just ``.enabled`` leaves its siblings intact, whereas stating
    the ``modules.web_terminals`` key itself would replace the subtree (see
    the preset's own comment above its ``modules.web_terminals`` block).

    ``VA_ARCHIVER_CI_KNOBS`` shrinks the archive the preset's ``va_archiver:``
    block declares to a CI-sized one -- see the constant for why.

    ``config:`` keys are flat dotted strings (matching the preset's own
    convention): each names one leaf of the rendered config, so pinning
    ``control_system.type`` leaves the rest of the ``control_system:`` block
    as the preset wrote it.
    """
    return {
        "config": {
            "control_system.type": "virtual_accelerator",
            "claude_code.servers.bluesky.enabled": True,
            "modules.web_terminals.enabled": False,
        },
        "dispatch": None,
        "services": {},
        **VA_ARCHIVER_CI_KNOBS,
    }


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


def merged_profile_edits(extra_config: dict[str, Any]) -> dict[str, Any]:
    """``profile_edits()`` with ``extra_config`` deep-merged into it.

    Reaches the keys ``init_args`` has no named parameter for -- e.g. the
    postgres/openobserve/tiled/panels HOST ports a module must move to run
    concurrently with another deployed stack::

        merged_profile_edits({"config": {"services.postgresql.port_host": 15433}})

    Shaped like the profile, exactly as ``profile_edits()`` is, so a caller
    reads its own additions beside the shared pins they land on.
    """
    return _deep_merge(profile_edits(), extra_config)


def init_args(
    project_name: str,
    *,
    output_dir: Path,
    bridge_port: int = BRIDGE_PORT,
    va_port: int = VA_CA_PORT,
    va_pva_port: int = VA_PVA_PORT,
    port_base: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> list[str]:
    """``osprey init`` CLI args (sans the leading ``init`` token) for FR11's
    turn-key plan-stack deployment.

    The stack is materialized in two steps, because the surface has two:
    ``osprey init`` writes the deployment repo's source zone from the preset
    plus these edits, and a later ``osprey build`` renders ``build/`` from
    it. This function covers the first step only; both builders below run the
    second. ``--no-git`` because every caller works in a throwaway directory
    and none of them reads the history.

    Works both as ``CliRunner().invoke(init, init_args(...))`` (in-process, no
    Docker -- see ``build_via_cli_runner``) and as
    ``[osprey_bin, "init", *init_args(...)]`` (subprocess, for a real
    ``osprey up`` afterward -- see ``build_project_subprocess``).

    ``provider``/``model``, when given, append ``--set provider=<provider>``
    and/or ``--set model=<model>`` edits -- e.g. an agentic-discovery
    caller that must pin an explicit provider rather than let the
    control-assistant preset's own default apply silently (this project's
    "no default provider" convention). Left ``None`` by default: nothing is
    appended and the preset's own provider/model apply unchanged, so the
    default deploy shape is unaffected by these params.

    ``extra_config``, when given, is deep-merged into ``profile_edits()`` --
    the way to reach config keys that have no named parameter here
    (postgres/openobserve/tiled/panels host ports). Empty or ``None`` is a
    no-op: the shared pins go out on their own.

    ``va_port``/``va_pva_port`` pin the VA's Channel Access and pvAccess host
    ports, the two a deployment's port block does not move.
    """
    args = [
        str(output_dir / project_name),
        "--preset",
        "control-assistant",
        "--no-git",
        *set_pairs(merged_profile_edits(extra_config or {})),
        "--set",
        f"virtual_accelerator.port={va_port}",
        "--set",
        f"virtual_accelerator.pva_port={va_pva_port}",
        "--set",
        f"bluesky.port={bridge_port}",
        "--set",
        "bluesky.tiled_enabled=true",
        # The preset defaults to graph mode; the plan stack pins the
        # hierarchical database so the lane's channel finder serves the
        # bundle's hierarchical index.
        "--set",
        "channel_finder_mode=hierarchical",
    ]
    if port_base is not None:
        # Every framework port the caller does NOT pin explicitly follows this
        # block (`deployment.port_base`); a caller that actually starts the
        # stack passes its own thousand-port block so the deploy cannot land
        # on a real deployment's default 10000 block (openobserve, tiled,
        # live-standin, the stores). Build-only callers may leave it unset —
        # a render binds nothing.
        args += ["--set", f"port_base={port_base}"]
    if provider is not None:
        args += ["--set", f"provider={provider}"]
    if model is not None:
        args += ["--set", f"model={model}"]
    return args


def _calling_module() -> str:
    """The module that called into this one — for a guard's failure message.

    A shared helper's assertion is read by whoever owns the LANE, not by
    whoever owns the helper, so the message has to say which lane it is
    talking about. The stack is the only thing that knows: the first frame
    outside this file is the caller's.
    """
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code.co_filename != __file__:
                return str(frame.f_globals.get("__name__") or frame.f_code.co_filename)
            frame = frame.f_back
        return __name__
    finally:
        # Frames hold references to this one; dropping the local keeps the
        # cycle off the collector's desk.
        del frame


def assert_off_default_block(repo: Path, project_name: str) -> None:
    """Refuse a rendered deployment that would bind the framework's DEFAULT
    thousand-port block.

    ``deployment.port_base`` names the first port of the block a deployment
    claims, and :data:`~osprey.port_layout.DEFAULT_PORT_BASE` (10000) is where
    a deployment lands when nobody moves it — including the real deployment a
    developer is already running on the host, and every e2e lane that forgot to
    pick a band. Two stacks in one block do not fail cleanly: they fail as a
    connection error or a wrong-service answer, minutes into a container build,
    in a lane that looks broken rather than colliding.

    So the check happens HERE, on the render, before anything binds. It reads
    the base back out of ``<repo>/build/config.yml`` rather than trusting the
    ``port_base=`` argument that was passed in: an overlay key that failed to
    land looks identical at the call site and only differs on disk, and that is
    precisely the failure this exists to catch.

    Deliberately a ``!=`` default check and not an ``==`` some-expected-value
    one: this is the shared seam, and it cannot know which band a given lane
    booked. A lane that knows its own number should assert that number too (see
    ``tests/e2e/test_full_chain_auth.py``'s ``_make_repo``) — the two checks
    are complementary, and this one is the floor.

    :param repo: The deployment REPO (the directory holding ``build/``), as
        :func:`build_project_subprocess` returns.
    :param project_name: The name the stack was built under, for the message.
    :raises AssertionError: If the render produced no config, or resolved the
        default base — whether by naming it or by never setting one.
    """
    from osprey.port_layout import (
        BLOCK_SIZE,
        DEFAULT_PORT_BASE,
        PORT_BASE_CONFIG_KEY,
        resolve_port_base,
    )

    caller = _calling_module()
    config_path = repo / "build" / "config.yml"
    if not config_path.is_file():
        raise AssertionError(
            f"{project_name} (built by {caller}) rendered no config at {config_path}, so "
            f"nothing can say which port block this deployment would claim"
        )

    rendered = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(rendered, dict):
        raise AssertionError(f"{config_path} is not a config mapping: {rendered!r}")

    declared = (rendered.get("deployment") or {}).get("port_base")
    if resolve_port_base(rendered) != DEFAULT_PORT_BASE:
        return

    block_top = DEFAULT_PORT_BASE + BLOCK_SIZE - 1
    raise AssertionError(
        f"{project_name} (built by {caller}) resolved "
        f"{PORT_BASE_CONFIG_KEY}={declared!r} — the framework DEFAULT block "
        f"{DEFAULT_PORT_BASE}-{block_top}, which a real deployment on this host "
        f"already claims and which every lane that books no band of its own lands "
        f"in together. Starting this stack would collide with them, and the first "
        f"symptom would be a connection error minutes into a container build.\n"
        f"Fix in {caller}: give this lane its own thousand-port band — pass "
        f"`port_base=<band>` to this module's builder (which emits "
        f"`--set port_base=<band>`, the profile shorthand for "
        f"`{PORT_BASE_CONFIG_KEY}=<band>`), or set `{PORT_BASE_CONFIG_KEY}=<band>` "
        f"in the lane's --override config block. Bands in use are listed at the "
        f"top of each e2e lane that pins one."
    )


def build_via_cli_runner(
    runner: CliRunner,
    tmp_path: Path,
    *,
    project_name: str = "orm-stack",
    bridge_port: int = BRIDGE_PORT,
    va_port: int = VA_CA_PORT,
    va_pva_port: int = VA_PVA_PORT,
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
    the only window in which a caller can edit the repo's source zone and
    still have the build render the edit. It must not run BEFORE ``init``:
    init copies the preset's ``data/`` into place without ``dirs_exist_ok``,
    so a pre-created ``data/`` makes the copy fail outright.
    """
    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    repo = tmp_path / project_name
    result: Result = runner.invoke(
        init,
        init_args(
            project_name,
            output_dir=tmp_path,
            bridge_port=bridge_port,
            va_port=va_port,
            va_pva_port=va_pva_port,
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
    va_pva_port: int = VA_PVA_PORT,
    port_base: int | None = None,
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
    the exact default deploy shape (an empty ``extra_config`` is likewise a
    no-op).

    ``pre_build``, when given, is called with the deployment REPO after
    ``osprey init`` has written it and before ``osprey build`` renders it --
    the only window in which a caller can edit the repo's source zone and
    still have the build render the edit. It must not run BEFORE ``init``:
    init copies the preset's ``data/`` into place without ``dirs_exist_ok``,
    so a pre-created ``data/`` makes the copy fail outright.
    """
    osprey_bin = find_osprey_console_script()

    cmd = [
        str(osprey_bin),
        "init",
        *init_args(
            project_name,
            output_dir=output_dir,
            bridge_port=bridge_port,
            va_port=va_port,
            va_pva_port=va_pva_port,
            port_base=port_base,
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

    # The render is done and nothing is bound yet — the one moment a port-block
    # mistake is still cheap. See :func:`assert_off_default_block`. Checked here
    # rather than on the ``port_base=`` argument because only the render knows
    # whether the override actually landed, and checked in THIS builder rather
    # than in ``init_args`` because ``build_via_cli_runner`` renders without
    # binding anything and may legitimately leave the base alone.
    assert_off_default_block(repo, project_name)
    return repo


def channel_limits(project_dir: Path) -> dict[str, Any]:
    """The limits view of the repo's ``data/facility/limits.yaml``, parsed --
    the same table the render enforces on the channels it can write.

    Callers pass the deployment REPO. The view is rendered from the repo's own
    facility tree into a scratch directory beside the repo, never inside it,
    so it is readable before the build and nothing of it reaches the build
    zone. Not what the plan-stack lanes choose their devices from: it holds
    the records the tree authors and enumerates no other channel, so a lane
    reads a channel's limit VALUES here and its devices from the staged view.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs

    render = project_dir.parent / f"{project_dir.name}.limits"
    if render.exists():
        shutil.rmtree(render)
    render.mkdir(parents=True)
    facility_dir = project_dir / "data" / "facility"
    document = build_facility(facility_dir, project_name=project_dir.name)
    render_facility_outputs(render, document, {}, facility_dir)
    return json.loads((render / "data" / "channel_limits.json").read_text(encoding="utf-8"))


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
            with urllib.request.urlopen(url, timeout=3.0) as resp:  # localhost
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


#: The bound-element attribute component a horizontal kick is written to, in
#: pyAT's ``KickAngle`` order -- ``(horizontal, vertical)``. Framework
#: vocabulary rather than a facility's: a lane wanting one plane of corrector
#: asks the bindings for the component, never the address text for a family
#: name.
KICK_HORIZONTAL = 0
KICK_VERTICAL = 1

#: The transverse axis a monitor binding reads, in the spelling
#: ``bindings.ATTRIBUTES_BY_KIND`` reserves for a ``monitor``.
MONITOR_X = "x"
MONITOR_Y = "y"


@cache
def _bindings_of_resolved(data_root: Path) -> BindingsDocument:
    """The memoized read behind :func:`_bindings_at`, keyed on a resolved tree."""
    from osprey.services.virtual_accelerator.bindings import load_bindings

    return load_bindings(ManifestPaths(data_root).va_bindings)


def _bindings_at(data_root: Path) -> BindingsDocument:
    """The bindings document of one facility data tree, read once per tree.

    Memoized because a lane calls the selectors several times and the demo
    document is some hundreds of bindings.

    The memo is keyed on the RESOLVED tree, so the relative and absolute
    spellings of one tree -- and a symlinked temporary directory against the
    path it points at -- are one entry rather than two. The document is read
    once per tree per process: a lane that rewrites
    ``simulation/va_bindings.json`` under a tree already read here gets the
    first read back, and must stage the new document under a fresh path to be
    served the new one.
    """
    return _bindings_of_resolved(data_root.resolve())


def repo_bindings(repo: Path) -> BindingsDocument:
    """The bindings document of the deployment repo's own ``data`` tree.

    The one authority on which channels the accelerator model drives and what
    each of them does to it: a binding names the element it writes, the
    attribute it writes there, and the calibration between the facility's
    hardware unit and the lattice's physics unit. A lane choosing devices with
    a plane, a kind or a calibration in mind reads them from here.

    Lives HERE rather than in the product because choosing a physics-appropriate
    subset of a facility's channels is a harness concern: the staged view
    holds every channel and says which way each points, and which of them a
    given plan can do physics with is the lane's own question.
    """
    return _bindings_at(repo / "data")


def _addresses_of_kind(document: BindingsDocument, kind: str) -> frozenset[str]:
    """Every address the document binds as ``kind``, by its own ``kind`` field.

    Grouping by what a binding DOES -- a kick, a monitor -- rather than by the
    family name its address spells is what keeps a lane's device choice the
    same question on every facility.
    """
    return frozenset(
        binding.setpoint_address for binding in document.bindings if binding.kind == kind
    )


def claimed_addresses(document: BindingsDocument) -> frozenset[str]:
    """Every address ``document`` claims -- each binding's setpoint, plus the
    readback it serves where it serves one.

    The single spelling of what "the accelerator model drives this channel"
    means, so a lane asking whether a channel is coupled and a lane asking for
    the channels that are not both read the rule from here. A binding kind that
    one day claims a second readback widens both at once.
    """
    claimed = {binding.setpoint_address for binding in document.bindings}
    claimed |= {
        binding.readback_address
        for binding in document.bindings
        if binding.readback_address is not None
    }
    return frozenset(claimed)


def pyat_coupled(address: str, *, data_root: Path | None = None) -> bool:
    """Whether ``address`` is a channel the accelerator model actually drives.

    A channel is coupled because a binding claims it -- as the address it
    writes or reads, or as the readback that binding serves -- and for no other
    reason. Everything else the deployment serves is either a physics-free
    software echo or a plausible noisy constant: a plan sweeping one finishes
    as fast as the network round-trips allow and proves nothing about rows
    arriving while physics runs. Lanes that read the BUILD's staged device file
    directly narrow it with this so the device they drive is a modelled one
    whichever order the build wrote the file in.

    ``data_root`` is the facility tree to ask; it defaults to the bundled demo
    tree, which is the tree a deployment built from the shipped preset serves.
    A lane deploying a facility's own tree passes that tree's ``data/``.
    """
    document = _bindings_at(PACKAGE_PATHS.data_root if data_root is None else data_root)
    return address in claimed_addresses(document)


def _keyed_by_address(
    items: list[_T], address_of: Callable[[_T], str], count: int | None, unit_label: str
) -> dict[str, _T]:
    """Key ``items`` by ``address_of(item)`` -- the device name IS the channel
    address, the convention the product's own derivation follows.

    ``count=None`` takes all; an int raises ``AssertionError`` when fewer than
    ``count`` are available, else slices to exactly ``count``.

    The "exactly ``count``" promise holds only while the addresses are distinct,
    which the view guarantees by naming each channel once. A colliding
    address would silently return a shorter dict, so the invariant is asserted
    rather than assumed.
    """
    if count is not None and len(items) < count:
        raise AssertionError(
            f"the build's staged device view only yields {len(items)} {unit_label}, need {count}"
        )
    take = len(items) if count is None else count
    keyed = {address_of(items[i]): items[i] for i in range(take)}
    if len(keyed) != take:
        raise AssertionError(
            f"duplicate addresses among the selected {unit_label}: "
            f"{take} selected, {len(keyed)} distinct names"
        )
    return keyed


def select_correctors(
    repo: Path, count: int | None = DEFAULT_CORRECTOR_COUNT
) -> dict[str, tuple[str, str]]:
    """Pick ``count`` corrector ``:SP``/``:RB`` pairs out of the device view the
    build staged for ``repo`` (:func:`staged_devices`), never a hardcoded
    preset channel.

    A corrector is an address the repo's bindings document binds as a ``kick``
    (:func:`repo_bindings`): a write to it steers the beam by changing an
    element's kick angle through the lattice model. The ORM plan sweeps
    correctors specifically, so a writable channel no binding claims -- a
    physics-free software echo -- is the wrong device class for it, and the
    kind is the same question on any facility where the address text is not.
    A settable the view names no readback for is skipped: these plans read a
    corrector back after setting it, and a device whose readback is its own
    setpoint would echo the demand rather than report the magnet.

    If ``count`` is ``None``, returns the FULL available corrector set instead
    of a fixed-size slice -- no assertion is raised in that case, regardless of
    how many pairs are found. Pairs are taken in address order.

    Returns a dict of ``sp_address -> (sp_address, rb_address)`` -- the
    setpoint's device name is its own ``:SP`` address, as it is in the view.
    """
    kicks = _addresses_of_kind(repo_bindings(repo), "kick")
    settables, _ = staged_devices(repo)
    pairs = [
        (setpoint, readback)
        for setpoint, readback in sorted(settables.values())
        if readback != setpoint and setpoint in kicks
    ]
    return _keyed_by_address(pairs, lambda pair: pair[0], count, "corrector pairs")


def select_bpms(repo: Path, count: int | None = DEFAULT_BPM_COUNT) -> dict[str, str]:
    """Pick ``count`` beam-position readbacks out of the device view the build
    staged for ``repo`` -- same view, same no-hardcoded-channel convention as
    :func:`select_correctors`.

    A beam-position readback is an address the repo's bindings document binds
    as a ``monitor``: a reading the lattice model solves for, which moves when
    a corrector is swept. A readable no binding claims is static noise and
    would sit still through any sweep.

    If ``count`` is ``None``, returns the FULL available monitor set instead of
    a fixed-size slice -- no assertion is raised in that case. Readbacks are
    taken in address order.

    Returns a dict of ``read_address -> read_address`` -- the readback's device
    name is its own read address, as it is in the view.
    """
    monitors = _addresses_of_kind(repo_bindings(repo), "monitor")
    _, readables = staged_devices(repo)
    addresses = sorted(address for address in readables.values() if address in monitors)
    return _keyed_by_address(addresses, lambda address: address, count, "monitor readbacks")


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

    The worker's device file is the build's Bluesky view of the facility
    file, staged unchanged, and every lane chooses its devices from it. Read
    back from the staged file rather than re-derived here, so the devices a
    test names are exactly the ones the deployed worker registered and a
    change in that view surfaces as a real failure instead of a silently
    diverging second copy of the logic. A settable the view names no readback
    for reads its setpoint back.
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
