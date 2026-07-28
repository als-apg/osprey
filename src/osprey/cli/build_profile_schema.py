"""Build-profile config dataclasses — the schema a profile YAML deserializes into.

The declarative half of the build profile: the nested config blocks a
``profile.yml`` may declare (``mcp_servers``, ``lifecycle``, ``env``,
``services``, ``dispatch``, ``bluesky``, ``virtual_accelerator``,
``bluesky_panels``) plus the environment-variable name pattern their
validators share. Parsing, inheritance merging, and validation live in
:mod:`osprey.cli.build_profile_load`, :mod:`osprey.cli.build_profile_merge`,
and :mod:`osprey.cli.build_profile_model`, respectively; this module is a
leaf holding only the shapes, so the service injectors can type against them
without importing the loader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class McpServerDef:
    """Definition of an MCP server to inject into a built project."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    permissions: dict[str, list[str]] = field(default_factory=dict)
    # permissions: {"allow": ["tool1"], "ask": ["tool2"]}
    url: str | None = None  # HTTP/SSE transport URL (mutually exclusive with command)
    # Single port the HTTP MCP service binds AND publishes. Compose maps
    # host:port → container:port 1:1, so consumers can derive every URL
    # variant from this single value. Mutually exclusive with command;
    # compatible with url (a port hint for non-Claude consumers).
    port: int | None = None


@dataclass
class LifecycleStep:
    """A single command to run during a lifecycle phase."""

    name: str
    run: str
    cwd: str | None = None
    timeout: int = 120  # seconds; override per-step in YAML
    stream: bool = False  # stream stdout in real-time for this step


@dataclass
class LifecycleConfig:
    """Lifecycle commands run before/after build and for validation."""

    pre_build: list[LifecycleStep] = field(default_factory=list)
    post_build: list[LifecycleStep] = field(default_factory=list)
    validate: list[LifecycleStep] = field(default_factory=list)


@dataclass
class EnvConfig:
    """Environment variable template configuration."""

    required: list[str] = field(default_factory=list)
    defaults: dict[str, str] = field(default_factory=dict)
    file: str | None = None  # Profile-relative path to copy as .env


@dataclass
class ServiceDef:
    """Definition of a container service for ``osprey deploy``."""

    template: str  # Path to template dir (relative to profile dir)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchConfig:
    """Event-dispatch configuration for a build profile (opt-in via the ``dispatch:`` key).

    Consumed by the build pipeline's dispatch-injection step to deploy the
    event_dispatcher + dispatch_worker services. All ports/counts are validated
    by :meth:`BuildProfile.validate`.
    """

    # Bundled trigger-file name (e.g. "tutorial_triggers.yml") or profile-relative path.
    triggers: str
    worker_count: int = 1
    workspace_mode: Literal["isolated", "shared"] = "isolated"
    max_concurrent_runs: int = 2
    max_queue_depth: int = 50
    dispatcher_port: int = 8020
    worker_port_base: int = 9190
    timeout_sec: int = 300
    inactivity_sec: int = 120
    facility_name: str = ""
    pv_strip_prefix: str = ""


@dataclass
class BlueskyConfig:
    """Bluesky scan-bridge configuration for a build profile (opt-in via the ``bluesky:`` key).

    Consumed by the build pipeline's bluesky-injection step to deploy the
    single ``bluesky_bridge`` service (see NAMING-ADDENDUM.md: deploy key
    ``bluesky``, env var ``BLUESKY_LAUNCH_TOKEN``, MCP server name ``scan``).
    Ports are validated by :meth:`BuildProfile.validate`.
    """

    port: int = 8090
    tiled_enabled: bool = False
    tiled_port: int = 8091
    demo_runner: bool = False
    """Opt-in only for the deploy-smoke-demo / tutorial case: wires the
    container's bridge process to a real bluesky RunEngine against mock
    ophyd-async devices (``devices/mock.py``) via app.py's guarded startup
    hook (task 2.14a), instead of the Phase 1 no-op ``FakePlanRunner`` default.
    MUST stay False for any facility wiring real EPICS hardware — turning
    this on would silently override real device/plan wiring with an
    in-memory mock runner.
    """
    plan_dir: str | None = None
    """Optional host directory of facility plan files (Task 1.4),
    bind-mounted read-only into the bridge container and surfaced to the
    plan loader as a ``BLUESKY_PLAN_DIRS`` (facility-tier) layer — see
    ``plan_loader.py``. ``None`` (default) deploys the bridge with no
    facility plan directory, matching every prior bluesky-only build.
    """
    excluded_plans: list[str] = field(default_factory=list)
    """Named plans to hide from the agent while the bluesky server stays
    enabled (dev/local convenience). Production uses the
    ``BLUESKY_EXCLUDED_PLANS`` env var instead.
    """


@dataclass
class VAConfig:
    """Virtual Accelerator soft-IOC configuration for a build profile (opt-in
    via the ``virtual_accelerator:`` key).

    Consumed by the build pipeline's VA-injection step to deploy the single
    ``virtual_accelerator`` service (compose service ``virtual-accelerator``,
    container ``<project>-virtual-accelerator``). Port is validated by
    :meth:`BuildProfile.validate`.
    """

    port: int = 5064
    """Channel Access TCP port the soft-IOC serves PVs on (see
    src/osprey/services/virtual_accelerator/entrypoint.py's run contract)."""


@dataclass
class BlueskyPanelsConfig:
    """Scan-panels sidecar configuration for a build profile (opt-in via the
    ``bluesky_panels:`` key).

    Consumed by the build pipeline's bluesky-panels-injection step
    (``_inject_bluesky_panels`` in ``build_cmd.py``) to deploy the single
    ``bluesky_panels`` FastAPI sidecar (compose service ``bluesky-panels``) that
    serves the three operator web panels (``plan``, ``results``,
    ``health``) and read-proxies the bluesky bridge. Port is validated
    by :meth:`BuildProfile.validate`.
    """

    port: int = 8095
    """Host/container port the sidecar's uvicorn process binds and publishes
    (see ``templates/services/bluesky_panels/docker-compose.yml.j2``)."""


_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
