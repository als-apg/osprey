"""Build-profile config dataclasses — the schema a profile YAML deserializes into.

The declarative half of the build profile: the nested config blocks a
``profile.yml`` may declare (``mcp_servers``, ``lifecycle``, ``env``,
``services``, ``dispatch``, ``bluesky``, ``virtual_accelerator``,
``bluesky_panels``, ``nextcloud_bridge``) plus the environment-variable name
pattern their validators share. Parsing, inheritance merging, and validation live in
:mod:`osprey.cli.build_profile_load`, :mod:`osprey.cli.build_profile_merge`,
and :mod:`osprey.cli.build_profile_model`, respectively; this module is a
leaf holding only the shapes, so the service injectors can type against them
without importing the loader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
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
class EnvironmentConfig:
    """The Python environment agent-authored code executes in (``environment:`` block).

    Distinct from :class:`EnvConfig` (the ``env:`` block), which templates
    environment *variables*. This block describes the *interpreter and packages*
    the project's environment is built from.

    There is deliberately no mode flag. :attr:`python` may name either a bare
    interpreter or the interpreter of an already-initialised venv — a venv's
    python *is* an interpreter, so both bases take the identical path when the
    project environment is created. What distinguishes a venv base is that its
    installed distributions can additionally be frozen into the project's
    dependency record, with :attr:`inherit_exclude` naming distributions to
    leave out of that freeze.

    Basing an environment on a venv's interpreter does **not** inherit that
    venv's packages (``uv venv --python <venv>/bin/python`` yields an empty
    environment); the freeze is what reproduces them.
    """

    python: str | None = None
    """Absolute path (``~`` is expanded) to the interpreter the project
    environment is based on. ``None`` — the default — means the build uses its
    own interpreter, i.e. no custom base."""

    packages: list[str] = field(default_factory=list)
    """Additional requirement specifiers (PEP 508) to install into the project
    environment. Independent of, and additive to, ``BuildProfile.dependencies``."""

    inherit_exclude: list[str] = field(default_factory=list)
    """Distribution names to omit when freezing a venv base's installed
    packages. Only meaningful when :attr:`python` is a venv interpreter —
    validation rejects it otherwise, since a bare interpreter has no installed
    set to exclude from."""

    def resolved_python(self) -> Path | None:
        """Return :attr:`python` as a ``~``-expanded path, or ``None`` when unset.

        Returns:
            The interpreter path, or ``None`` if no custom base is declared.
            Validation guarantees the path is absolute, existing, and
            executable whenever it is not ``None``.
        """
        if not self.python:
            return None
        return Path(self.python).expanduser()

    def venv_base(self) -> Path | None:
        """Return the venv root when :attr:`python` is a venv's interpreter.

        Venv-ness is *detected*, not declared: a ``pyvenv.cfg`` beside the
        interpreter's directory (``<venv>/bin/python`` → ``<venv>/pyvenv.cfg``)
        marks the base as a venv.

        Returns:
            The venv root directory, or ``None`` when :attr:`python` is unset or
            names a bare interpreter. ``venv_base() is not None`` is therefore
            the venv-base predicate.
        """
        python = self.resolved_python()
        if python is None:
            return None
        # `<venv>/bin/python` first (the posix layout); fall back to an
        # interpreter sitting directly in the venv root.
        for candidate in (python.parent.parent, python.parent):
            if (candidate / "pyvenv.cfg").is_file():
                return candidate
        return None


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


@dataclass
class NextcloudBridgeProfileConfig:
    """Nextcloud Talk bridge configuration for a build profile (opt-in via the
    ``nextcloud_bridge:`` key).

    Consumed by the build pipeline's nextcloud-bridge-injection step
    (``_inject_nextcloud_bridge`` in ``build_cmd.py``) to deploy the single
    ``nextcloud_bridge`` service — an outbound-only poller that ingests Talk
    mentions and dispatches them through the event-dispatch pair, so the block
    is only meaningful alongside a ``dispatch:`` block.

    Talk room tokens and bot credentials are deliberately *not* profile fields:
    ``NEXTCLOUD_ROOMS``, ``NEXTCLOUD_BOT_ACCOUNT`` and
    ``NEXTCLOUD_APP_PASSWORD`` are user-supplied runtime env (declared via
    ``env.required``), never baked into a build. Validated by
    :meth:`BuildProfile.validate`.
    """

    trigger: str = "nextcloud-question"
    """Dispatcher trigger the bridge fires (``POST /webhook/{trigger}``),
    rendered as ``DISPATCH_TRIGGER`` in the service's compose template.

    This default is the ONLY place the ``nextcloud-question`` name is defaulted:
    the runtime config's ``from_env`` applies no trigger default, so a
    hand-rolled (non-build) deployment still fails loudly on a missing trigger
    rather than silently firing a name nobody declared. The value must name a
    trigger declared in the ``dispatch.triggers`` file.
    """


_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
