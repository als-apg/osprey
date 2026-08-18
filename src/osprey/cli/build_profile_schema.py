"""Build-profile config dataclasses — the schema a profile YAML deserializes into.

The declarative half of the build profile: the nested config blocks a
``profile.yml`` may declare (``mcp_servers``, ``lifecycle``, ``env``,
``services``, ``dispatch``, ``bluesky``, ``virtual_accelerator``,
``bluesky_panels``, ``nextcloud_bridge``, ``gchat_bridge``) plus the environment-variable name
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

NetworkMode = Literal["bridge", "host"]
"""How a deployed service attaches to the network."""

VALID_NETWORK_MODES: tuple[NetworkMode, ...] = ("bridge", "host")
"""The closed network vocabulary, in declaration order. ``bridge`` is the
compose-managed project network every service has always used; ``host`` shares
the host's network namespace, which is what a site needs when a service must
see broadcast traffic (control-system protocols, discovery) or reach ports
published on the host. Consumers branch on exactly these two, so a third mode
is a deliberate addition here rather than a string a profile can invent."""

DEFAULT_NETWORK_MODE: NetworkMode = "bridge"
"""Attachment a service gets when it declares none — the behaviour that
existed before the axis, so an unset ``network:`` renders as it always did."""


def network_mode_errors(value: Any, key: str) -> list[str]:
    """Return the problems with one ``network:`` declaration (empty when valid).

    Accumulates rather than raising, the way :meth:`BuildProfile.validate`
    does, so a profile author sees every axis problem at once.

    Args:
        value: The declared mode, exactly as it came out of the YAML.
        key: Dotted path of the declaration (e.g. ``"dispatch.network"``),
            used verbatim in the message so the author can find it.

    Returns:
        Human-readable error messages; empty when ``value`` names a mode in
        :data:`VALID_NETWORK_MODES`.
    """
    if isinstance(value, str) and value in VALID_NETWORK_MODES:
        return []
    message = f"{key} must be one of {', '.join(VALID_NETWORK_MODES)} (got {value!r})"
    if isinstance(value, bool):
        # A bare `network: on` is read on the YAML 1.1 resolver as the bool
        # True, so the author sees their own spelling reported back as a
        # boolean they never wrote. Name the resolver rather than let them
        # re-read the same line looking for a typo that is not there.
        message += ". A bare yes/no/on/off in YAML parses as a boolean — quote the value."
    return [message]


@dataclass
class ProfileProvenance:
    """What a materialized profile was emitted from (``provenance:``).

    Written by ``osprey init`` and never by hand. It is the
    MACHINE-READABLE record of the profile's source — the emitted header says
    the same thing in prose, for people — and it is what a later build compares
    against the installed preset to notice that the preset has moved on since
    the profile was materialized (FR-6). That comparison is advisory: a profile
    is the source of truth once it exists, so drift is reported, never enforced.
    """

    preset: str
    """Bundled preset name the profile was materialized from."""
    preset_hash: str
    """Content hash of that preset as resolved at materialization time."""


@dataclass
class McpServerDef:
    """Definition of an MCP server to inject into a built project."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    permissions: dict[str, list[str]] = field(default_factory=dict)
    # permissions: {"allow": ["tool1"], "ask": ["tool2"]}
    url: str | None = None  # Remote transport URL (mutually exclusive with command)
    # Wire transport for URL servers: "http" (streamable-HTTP, the default) or
    # "sse" (legacy Server-Sent Events). Stdio servers (command) must not set
    # it — load_profile rejects that.
    transport: str = "http"
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
    """Definition of a container service for ``osprey up``.

    :attr:`config` is a free-form pass-through: whatever a profile declares
    under ``services.<name>.config`` is written to the rendered ``config.yml``
    and is visible to the service's compose template. One key in it is
    understood by the build itself — ``network:``, the service's attachment,
    one of :data:`VALID_NETWORK_MODES` and defaulting to
    :data:`DEFAULT_NETWORK_MODE`. It is validated by
    :meth:`BuildProfile.validate` and read through :meth:`network_mode`.
    """

    template: str  # Path to template dir (relative to profile dir)
    config: dict[str, Any] = field(default_factory=dict)

    def network_mode(self) -> str:
        """Return the service's declared network attachment.

        The single place the axis default is applied for a declared service,
        so a consumer never has to spell ``"bridge"`` itself.

        Returns:
            The declared mode, or :data:`DEFAULT_NETWORK_MODE` when the service
            declares none. Guaranteed to name a mode in
            :data:`VALID_NETWORK_MODES` for any profile that passed
            :meth:`BuildProfile.validate`.
        """
        if not isinstance(self.config, dict):
            return DEFAULT_NETWORK_MODE
        mode = self.config.get("network", DEFAULT_NETWORK_MODE)
        # A non-string here means the profile never passed validation (a bare
        # `network: on` reaches this as a bool); fall back rather than hand a
        # consumer a value it would compare against the vocabulary and miss.
        return mode if isinstance(mode, str) else DEFAULT_NETWORK_MODE


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

    network: NetworkMode = DEFAULT_NETWORK_MODE
    """Network attachment for the dispatcher and its workers, one of
    :data:`VALID_NETWORK_MODES`.

    ONE knob covers the pair: the dispatcher and the workers talk to each other
    over addresses the build emits, so a half on the compose network and a half
    on the host's could not reach each other. A ``network:`` authored directly
    on ``services.event_dispatcher`` or ``services.dispatch_worker`` is
    therefore rejected by :meth:`BuildProfile.validate` — the build writes this
    single value into both halves instead.
    """


@dataclass
class BlueskyConfig:
    """Bluesky bridge configuration for a build profile (opt-in via the ``bluesky:`` key).

    Consumed by the build pipeline's bluesky-injection step to deploy the
    single ``bluesky_bridge`` service (see NAMING-ADDENDUM.md: deploy key
    ``bluesky``, env var ``BLUESKY_LAUNCH_TOKEN``, MCP server name ``bluesky``).
    Ports are validated by :meth:`BuildProfile.validate`.
    """

    port: int = 8090
    tiled_enabled: bool = False
    tiled_port: int = 8091
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


@dataclass
class GChatBridgeProfileConfig:
    """Google Chat bridge configuration for a build profile (opt-in via the
    ``gchat_bridge:`` key).

    Consumed by the build pipeline's gchat-bridge-injection step
    (``_inject_gchat_bridge`` in ``build_cmd.py``) to deploy the single
    ``gchat_bridge`` service — a Pub/Sub subscriber that ingests Google Chat
    messages and dispatches them through the event-dispatch pair, so the block
    is only meaningful alongside a ``dispatch:`` block.

    The Google credentials and destinations are deliberately *not* profile
    fields: ``GCHAT_SA_KEY``, ``GCHAT_SUBSCRIPTION``, ``GCHAT_APP_ID`` and the
    optional artifact-publishing pair ``GCS_BUCKET``/``GCS_PROJECT`` are
    user-supplied runtime env (declared via ``env.required``), never baked into
    a build. Validated by :meth:`BuildProfile.validate`.
    """

    trigger: str = "gchat-question"
    """Dispatcher trigger the bridge fires (``POST /webhook/{trigger}``),
    rendered as ``DISPATCH_TRIGGER`` in the service's compose template.

    This default is the ONLY place the ``gchat-question`` name is defaulted:
    the runtime config's ``from_env`` applies no trigger default, so a
    hand-rolled (non-build) deployment still fails loudly on a missing trigger
    rather than silently firing a name nobody declared. The value must name a
    trigger declared in the ``dispatch.triggers`` file.
    """


_ENV_VAR_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
