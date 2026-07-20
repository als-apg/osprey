"""Reusable turn-key scan-stack deploy configuration (task 4.3 / PROPOSAL FR11).

Builds the shipped deploy config that brings up the Virtual Accelerator +
Bluesky bridge + co-deployed Tiled catalog with
``control_system.type=virtual_accelerator``, ``execution.execution_method=
container`` (so ``BLUESKY_LAUNCH_TOKEN`` mints safely and the agent can arm
-- see ``container_lifecycle.py``'s ``_local_exec_arming_unsafe``), and the
``scan`` MCP server enabled (``default_enabled=False`` in the framework
registry; opted in here via ``claude_code.servers.bluesky.enabled``). Corrector
setpoints and BPM readbacks are wired into ``BLUESKY_EPICS_MOTORS``/
``_DETECTORS`` from the *built* project's own ``channel_limits.json`` --
never a hardcoded preset channel (mirrors
``tests/e2e/test_va_substrate_equivalence.py``'s ``_select_sp_echo_pairs``,
restricted here to correctors/BPMs specifically since the ORM plan sweeps
correctors and reads BPMs, not arbitrary writable setpoints).

Not a test module itself (no ``test_`` functions) -- the single source of
this config for:
  * ``tests/deployment/test_compose_generator.py``'s ``orm_stack`` render
    gate (this task, Docker-free, via ``build_via_cli_runner``),
  * the real-container round-trip e2e (task 5.2, ``test_orm_roundtrip.py``),
  * the agentic-discovery e2e (tasks 5.3/5.4),
via ``build_project_subprocess`` + ``select_correctors``/``select_bpms``/
``write_scan_env``.

Building this config never touches Docker by itself -- only a subsequent
``osprey deploy up`` does (left to each caller, since only the real
e2e/agentic tests need a live stack).

``select_correctors``/``select_bpms``/``write_scan_env`` delegate to the
canonical derivation in
``osprey.services.bluesky_bridge.substrate_devices`` (the single source of
this logic, also used by ``osprey deploy up`` to auto-configure a VA-backed
scan stack's ``.env`` -- see ``container_lifecycle._ensure_bluesky_substrate_env``).
This module keeps its own public API/signatures/defaults unchanged so every
existing e2e importer is unaffected.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from click.testing import CliRunner, Result

# Channel Access port the Virtual Accelerator serves on. NOT freely
# overridable: the Control Assistant preset's config.yml.j2 hardcodes
# `control_system.connector.virtual_accelerator.gateways.*.port: 5064` (it is
# not templated from `services.virtual_accelerator.port`) -- so the
# published container port must stay at this value, or the connector and the
# container silently drift apart (see test_va_substrate_equivalence.py's
# identical note).
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


def _service_image(project_name: str, service: str) -> str:
    """Derive a locally-built ``<project>-<service>:local`` image tag the way
    the service compose templates do.

    The templates default their image to
    ``{{ osprey_labels.project_name }}-<service>:local`` -- rendered from
    :func:`osprey.deployment.compose_generator.resolve_project_name` -- so a
    caller that force-rebuilds via ``docker rmi -f`` must target that SAME
    project-prefixed tag, never a host-global name.
    """
    from osprey.deployment.compose_generator import resolve_project_name

    return f"{resolve_project_name({'project_name': project_name})}-{service}:local"


def bridge_image(project_name: str) -> str:
    """``<project>-bluesky-bridge:local`` for ``project_name``."""
    return _service_image(project_name, "bluesky-bridge")


def va_image(project_name: str) -> str:
    """``<project>-va:local`` for ``project_name``."""
    return _service_image(project_name, "va")


def panels_image(project_name: str) -> str:
    """``<project>-bluesky-panels:local`` for ``project_name``."""
    return _service_image(project_name, "bluesky-panels")


BUILD_TIMEOUT_SEC = 300

# A small, concurrency-friendly corrector/BPM count for the render + the
# real-container round-trip gate. The agentic e2e scenarios (5.3/5.4) name
# their own errant device/location and don't depend on this count.
DEFAULT_CORRECTOR_COUNT = 4
DEFAULT_BPM_COUNT = 4


def override_yaml() -> str:
    """FR11's ``--override`` YAML content: VA control system + container
    exec (arming-safe) + the scan MCP server.

    ``dispatch: null`` drops control-assistant's default event-dispatcher
    stack (Node + Claude CLI image) -- irrelevant to the scan stack and far
    slower to build than the VA/bridge images already are (mirrors
    test_va_substrate_equivalence.py / test_tiled_roundtrip.py).

    ``modules.web_terminals.enabled: false`` drops the preset's per-persona
    web-terminal stack (two persona images + nginx, all built locally) for
    the same reason: nothing in the scan stack touches persona routing, and
    that coverage lives in the dedicated web-terminals lanes
    (control-assistant-demo-e2e, multi-user-deploy-lifecycle-e2e,
    tests/e2e/web_terminals/). One dotted LEAF key on purpose -- the preset
    sets the whole ``modules.web_terminals`` subtree as a single dotted key,
    and overriding just ``.enabled`` leaves its siblings intact, whereas a
    nested ``modules:`` mapping would wholesale-replace the subtree (see the
    preset's own comment above its ``modules.web_terminals`` block).

    Written as flat dotted-string keys under ``config:`` (matching the
    preset's own convention), not a `--set config.control_system.type=...`
    CLI override -- `--set` builds a NESTED dict for every dotted segment,
    which would replace the entire `control_system:`/`execution:` block
    instead of overriding just one field.
    """
    return (
        "config:\n"
        "  control_system.type: virtual_accelerator\n"
        "  execution.execution_method: container\n"
        "  claude_code.servers.bluesky.enabled: true\n"
        "  modules.web_terminals.enabled: false\n"
        "dispatch: null\n"
    )


def build_args(
    project_name: str,
    *,
    override_path: Path,
    output_dir: Path,
    bridge_port: int = BRIDGE_PORT,
    va_port: int = VA_CA_PORT,
    provider: str | None = None,
    model: str | None = None,
) -> list[str]:
    """``osprey build`` CLI args (sans the leading ``build`` subcommand
    token) for FR11's turn-key scan-stack deploy config.

    Works both as ``CliRunner().invoke(build, build_args(...))`` (in-process,
    no Docker -- see ``build_via_cli_runner``) and as
    ``[osprey_bin, "build", *build_args(...)]`` (subprocess, for a real
    ``deploy up`` afterward -- see ``build_project_subprocess``).

    ``provider``/``model``, when given, append ``--set provider=<provider>``
    and/or ``--set model=<model>`` overrides -- e.g. an agentic-discovery
    caller that must pin an explicit provider rather than let the
    control-assistant preset's own default apply silently (this project's
    "no default provider" convention). Left ``None`` by default: nothing is
    appended and the preset's own provider/model apply unchanged, so the
    default deploy shape is byte-identical to before these params existed.
    """
    args = [
        project_name,
        "--preset",
        "control-assistant",
        "--override",
        str(override_path),
        "--set",
        f"virtual_accelerator.port={va_port}",
        "--set",
        f"bluesky.port={bridge_port}",
        "--set",
        "bluesky.tiled_enabled=true",
        "--skip-deps",
        "--skip-lifecycle",
        "--output-dir",
        str(output_dir),
        "--force",
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
) -> Path:
    """In-process ``osprey build`` (``CliRunner``, no subprocess/Docker) for
    fast render-only gates -- see ``tests/cli/test_va_default_config.py`` for
    the same in-process pattern. Renders config.yml, the service compose
    templates, and the Claude Code artifacts (``.mcp.json`` included); never
    starts a container.

    Returns the built project directory.
    """
    from osprey.cli.build_cmd import build

    override_path = tmp_path / "override.yml"
    override_path.write_text(override_yaml(), encoding="utf-8")

    result: Result = runner.invoke(
        build,
        build_args(
            project_name,
            override_path=override_path,
            output_dir=tmp_path,
            bridge_port=bridge_port,
            va_port=va_port,
        ),
    )
    if result.exit_code != 0:
        raise AssertionError(f"osprey build failed (exit={result.exit_code}):\n{result.output}")
    return tmp_path / project_name


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
) -> Path:
    """Real ``osprey build`` subprocess for a project a caller will later
    ``osprey deploy up`` (that step needs Docker; this one doesn't -- it only
    renders config.yml/compose templates/.mcp.json, same as
    ``build_via_cli_runner``, but out-of-process so ``--dev``/``deploy up``
    against the resulting project directory behave exactly as they would for
    an operator running the real CLI).

    ``provider``/``model`` thread straight through to ``build_args`` (see its
    docstring for the override they append). Left ``None`` by default, which
    preserves the exact default deploy shape.
    """
    osprey_bin = find_osprey_console_script()
    override_path = output_dir / "override.yml"
    override_path.write_text(override_yaml(), encoding="utf-8")

    cmd = [
        str(osprey_bin),
        "build",
        *build_args(
            project_name,
            override_path=override_path,
            output_dir=output_dir,
            bridge_port=bridge_port,
            va_port=va_port,
            provider=provider,
            model=model,
        ),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"osprey build failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return output_dir / project_name


def _channel_limits(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "data" / "channel_limits.json").read_text(encoding="utf-8"))


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

    Returns a dict of synthetic motor name -> ``(sp_address, rb_address)``,
    ready for ``write_scan_env``'s ``BLUESKY_EPICS_MOTORS`` wiring.

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

    Returns a dict of synthetic detector name -> readback address, ready for
    ``write_scan_env``'s ``BLUESKY_EPICS_DETECTORS`` wiring.

    Thin wrapper: delegates to the canonical
    ``osprey.services.bluesky_bridge.substrate_devices.select_bpms`` (same
    logic; this module keeps the ``DEFAULT_BPM_COUNT`` default the e2e suite
    has always used, versus the product module's ``None``/full-set default).
    """
    from osprey.services.bluesky_bridge.substrate_devices import select_bpms as _select_bpms

    return _select_bpms(limits, count)


def write_scan_env(
    project_dir: Path,
    *,
    correctors: dict[str, tuple[str, str]],
    bpms: dict[str, str],
    launch_token: str | None = None,
) -> None:
    """Wire correctors + BPMs into ``BLUESKY_EPICS_MOTORS``/``_DETECTORS``
    and set ``BLUESKY_EPICS_SUBSTRATE=1``, appended to the project ``.env``
    BEFORE ``osprey deploy up`` (the bridge compose template passes these
    through from the project ``.env``, same mechanism as
    ``BLUESKY_LAUNCH_TOKEN``).

    ``launch_token``, if given, is also written. The container-exec path
    normally auto-mints one on ``deploy up``; callers that need a
    deterministic value for a scripted launch call supply their own (the
    same operator-provides-a-token path used elsewhere in this e2e suite).

    Formatting delegates to the canonical
    ``osprey.services.bluesky_bridge.substrate_devices`` formatters (same
    ``name=SP|RB`` / ``name=RB`` syntax the product deploy-time producer
    uses -- one source of the wire format).
    """
    from osprey.services.bluesky_bridge.substrate_devices import (
        format_detectors_env,
        format_motors_env,
    )

    values = {
        "BLUESKY_EPICS_SUBSTRATE": "1",
        "BLUESKY_EPICS_MOTORS": format_motors_env(correctors),
        "BLUESKY_EPICS_DETECTORS": format_detectors_env(bpms),
    }
    if launch_token:
        values["BLUESKY_LAUNCH_TOKEN"] = launch_token

    env_path = project_dir / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    new_lines = "".join(f"{k}={v}\n" for k, v in values.items())
    env_path.write_text(existing + new_lines, encoding="utf-8")
