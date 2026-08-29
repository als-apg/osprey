"""Single source of truth for resolving the Bluesky bridge connection.

Two independent OSPREY components talk to the same facility-side Bluesky bridge
and must agree, exactly, on *which* bridge instance and *which* launch token
they are using:

- the Bluesky MCP server (``osprey.mcp_server.bluesky.server_context``), the
  agent's path to the bridge, and
- the operator bluesky-web sidecar (``osprey.interfaces.bluesky_web``), the
  browser's path to the bridge.

If those two ever drift in how they resolve the bridge URL or the launch token
-- a different env-var name, a different config key, a different fallback order
-- the agent and the panel silently arm (or fail to arm) *different* bridge
instances. That is a safety-relevant bug class: the human could approve a
launch in a panel that targets one bridge while the agent's writes-arming
token belongs to another. Keeping the resolution logic here, imported by both,
makes that drift impossible by construction.

Every resolver here takes an optional PLAN LANE. A lane is a whole bridge
stack bound at render time to the control-system target it serves, and a
project renders a second one only when its build profile opted in. Lane 1 is
the default on every entry point, so a caller that knows nothing about lanes
resolves exactly what it always resolved -- and a caller that names a lane
this deployment does not render is REFUSED rather than quietly handed lane 1,
because that fallback is the wrong-machine bug this whole axis exists to
prevent.

This is a top-level leaf module (the precedent is
``osprey.bluesky_tool_names``): it imports **nothing** from
``osprey.mcp_server`` or ``osprey.services`` so both may import it without a
cycle. ``osprey.utils.workspace`` is imported lazily, inside the resolver
functions, only when a config fallback is actually needed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from osprey.port_layout import default_port, resolve_port_base

#: The port the bridge publishes when nothing configures one, **at the layout's
#: default base** — the schema default of the build profile's ``bluesky.port``.
#: Right only for a caller with no config to resolve a base from;
#: :func:`bridge_url_from_config` derives the same slot at its own config's base.
DEFAULT_BRIDGE_PORT = default_port("bluesky")

DEFAULT_BRIDGE_URL = f"http://127.0.0.1:{DEFAULT_BRIDGE_PORT}"

#: The variable that names lane 1's bridge outright — set per bridge instance
#: by a compose service that reaches it by its in-network name, or by an
#: operator. Every lane has one, spelled ``<PREFIX>_BRIDGE_URL`` for
#: :func:`lane_env_prefix` of that lane; this is that spelling for
#: :data:`LANE_ONE`, whose prefix is the pre-lane name unchanged.
BRIDGE_URL_ENV_VAR = "BLUESKY_BRIDGE_URL"


def bridge_url_from_config(config: Mapping[str, Any] | None) -> str:
    """The bridge base URL a loaded ``config.yml`` implies, trailing slash stripped.

    Resolution order, the config half of :func:`resolve_bridge_url` for
    :data:`LANE_ONE`. A second lane has no ``bluesky.bridge_url`` of its own
    and no default to fall back to, so it resolves its port directly there.

    1. ``bluesky.bridge_url`` — a full URL, for a bridge that is not the one
       this deployment publishes on loopback (a dev convenience).
    2. ``services.bluesky.port`` — the port the deployment publishes its own
       bridge on. Written by the build for a deploying project and projected
       into every attached render (a web-terminal persona), so a bridge moved
       on the hosting profile moves every client with it. Dialed on loopback:
       a persona container shares the host network namespace, and the host's
       own processes reach the published port there too.
    3. The ``bluesky`` layout slot at this config's own ``deployment.port_base``
       — the port the build would have published had it written the key. Not
       :data:`DEFAULT_BRIDGE_URL`, whose base belongs to no particular
       deployment: on a host running two, arming the wrong one is exactly the
       drift this module exists to prevent.

    Duplicated, deliberately and knowingly, by the approval hook's
    ``_resolve_bridge_url`` (``templates/claude_code/claude/hooks/osprey_approval.py``),
    which runs standalone in a different venv and cannot import this module;
    a change here is a change there.
    """
    bluesky = (config or {}).get("bluesky")
    url = bluesky.get("bridge_url") if isinstance(bluesky, Mapping) else None
    if url:
        return str(url).rstrip("/")
    services = (config or {}).get("services")
    block = services.get("bluesky") if isinstance(services, Mapping) else None
    port = block.get("port") if isinstance(block, Mapping) else None
    if not port:
        port = default_port("bluesky", base=resolve_port_base(config))
    return f"http://127.0.0.1:{port}"


#: Service key of the plan lane every deployment has had since the bridge
#: shipped. It stays the default everywhere here, so a caller that knows
#: nothing about lanes resolves exactly what it always resolved.
LANE_ONE = "bluesky"

#: A plan lane's service key, by the control-system target it serves -- and the
#: ONE place any of those keys is spelled. A lane is named for its target,
#: never for its index: a lane's identity is fixed at render time, and "lane 2"
#: does not say which machine it drives.
#:
#: This module is the registry every other holder imports from -- the build
#: injector, the compose generator, the container lifecycle, the Reach
#: Contract. A second spelling elsewhere is how one holder ends up provisioning
#: a lane another holder cannot see, so the literals live here and nowhere else
#: under ``src/osprey`` (``tests/cli/test_bluesky_lane_config.py`` pins that).
#: The one deliberate exception is the standalone approval hook, deployed into
#: a project's own venv, which can import nothing from OSPREY at all.
#:
#: The keys are :data:`osprey_connectors.types.CONTROL_TARGETS` -- the whole
#: vocabulary a lane can serve, ``standin`` included, because the live stand-in
#: is a third machine with a connector block of its own rather than a mode of
#: ``live``. Restated as literals rather than imported so this stays a leaf
#: module; the equality is pinned by test rather than left to hope.
SECOND_LANE_KEYS = {
    "va": "bluesky_va",
    "live": "bluesky_live",
    "standin": "bluesky_standin",
}

#: Every service key that can name a lane, in render order: lane 1's historical
#: key followed by each target-named sibling. Derived from
#: :data:`SECOND_LANE_KEYS` so the registry has exactly one entry point.
LANE_KEYS = (LANE_ONE, *SECOND_LANE_KEYS.values())


class UnknownBlueskyLaneError(ValueError):
    """A lane was named that this deployment does not render.

    Raised rather than falling back to lane 1, and that is the whole point of
    the exception existing: silently resolving an unknown lane to the default
    one would send a plan to a bridge bound to a DIFFERENT machine than the
    caller asked for. A refusal is recoverable; a wrong-machine write is not.
    """


def lane_env_prefix(lane: str) -> str:
    """The environment-variable prefix a lane's own settings are spelled under.

    ``bluesky`` -> ``BLUESKY``, which is every pre-lane variable name exactly
    as it always was; ``bluesky_va`` -> ``BLUESKY_VA``. Matched, by
    construction, to the prefix ``osprey up`` mints under
    (``deployment/container_lifecycle.py``) and the one the compose template
    interpolates -- the three are one contract, and this is its spelling.
    """
    return lane.upper()


def _lane_or_default(lane: str | None) -> str:
    """Validate a caller-supplied lane, defaulting to lane 1."""
    if lane is None:
        return LANE_ONE
    if lane not in LANE_KEYS:
        raise UnknownBlueskyLaneError(
            f"{lane!r} is not a bluesky plan lane. The lanes are "
            f"{', '.join(repr(key) for key in LANE_KEYS)}; a single-lane deployment "
            f"renders only {LANE_ONE!r}."
        )
    return lane


def resolve_bridge_url(lane: str | None = None) -> str:
    """Resolve the Bluesky bridge base URL, for one lane.

    Resolution order (``<PREFIX>`` is :func:`lane_env_prefix` of the lane, so
    lane 1's names are the pre-lane ones unchanged):

    1. ``<PREFIX>_BRIDGE_URL`` env var (full URL) -- set by the framework server
       definition per bridge instance; wins outright.
    2. For lane 1, the loaded ``config.yml`` through
       :func:`bridge_url_from_config` -- ``bluesky.bridge_url``, then the
       loopback URL of the port the deployment publishes its own bridge on
       (``services.bluesky.port``). For a second lane, the loopback URL of its
       own published port (``services.<lane>.port``), which is where
       ``_inject_bluesky`` put it -- there is no second ``bluesky.bridge_url``
       to read, and inventing one would be a key an operator has to keep in
       step with a port the build derives.
    3. :data:`DEFAULT_BRIDGE_URL` (lane 1 only, and only when its config names
       no port either).

    The returned URL has any trailing slash stripped so callers can append a
    path verbatim.

    :param lane: A lane's ``services.<lane>`` key, or ``None`` for lane 1.
    :raises UnknownBlueskyLaneError: ``lane`` is not a lane key, or names a
        second lane this deployment does not render. Never falls back to lane
        1 -- see :class:`UnknownBlueskyLaneError`.
    """
    lane_key = _lane_or_default(lane)
    full = os.environ.get(f"{lane_env_prefix(lane_key)}_BRIDGE_URL")
    if full:
        return full.rstrip("/")

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    if lane_key == LANE_ONE:
        return bridge_url_from_config(config)

    port = (config.get("services", {}).get(lane_key) or {}).get("port")
    if not port:
        raise UnknownBlueskyLaneError(
            f"This deployment renders no {lane_key!r} lane: its config.yml has no "
            f"services.{lane_key}.port. A second lane is opt-in "
            "(`bluesky.second_lane` in the build profile) and has to be built and "
            "deployed before anything can be queued on it."
        )
    return f"http://127.0.0.1:{port}"


def resolve_launch_token(lane: str | None = None) -> str | None:
    """Resolve the Bluesky bridge launch token, for one lane.

    Resolution order:

    1. ``<PREFIX>_LAUNCH_TOKEN`` env var -- minted fail-closed per lane by
       ``osprey up``; wins outright.
    2. ``bluesky.launch_token`` in config.yml for lane 1, or
       ``bluesky.lane_launch_tokens.<lane>`` for any lane (local/dev
       convenience only). Lane 1 keeps its own historical spelling rather than
       being moved under the map, so no existing project's key changes name.
    3. ``None`` -- launch is refused client-side, before contacting the bridge,
       when no token is resolved.

    Per lane because the token is what ARMS a launch: one token shared across
    two lanes would let a launch a human approved against one machine be
    replayed against the other.

    :param lane: A lane's ``services.<lane>`` key, or ``None`` for lane 1.
    :raises UnknownBlueskyLaneError: ``lane`` is not a lane key.
    """
    lane_key = _lane_or_default(lane)
    token = os.environ.get(f"{lane_env_prefix(lane_key)}_LAUNCH_TOKEN")
    if token:
        return token

    from osprey.utils.workspace import load_osprey_config

    bluesky = load_osprey_config().get("bluesky", {})
    per_lane = bluesky.get("lane_launch_tokens") or {}
    token = per_lane.get(lane_key)
    if not token and lane_key == LANE_ONE:
        token = bluesky.get("launch_token")
    return str(token) if token else None


def _loaded_config(config: dict | None) -> dict:
    """*config* when given, else the rendered project config, else ``{}``.

    Never raises: lane discovery runs on every queue call and inside a sidecar's
    startup, and neither may fail over a config that is missing (a dev checkout,
    most unit tests) or unreadable. An empty mapping reads as "one lane", which
    is the deployment shape every project had before the lane axis existed.
    """
    if config is not None:
        return config
    try:
        from osprey.utils.workspace import load_osprey_config

        loaded = load_osprey_config()
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def discover_lane_keys(config: dict | None = None) -> tuple[str, ...]:
    """The plan lanes this deployment renders, in render order.

    A lane exists in the rendered config as its own ``services.<lane>`` block
    (``_inject_bluesky`` writes one per lane), so the blocks are the deployment's
    own statement of how many lanes it has. Lane 1 is always reported, whether or
    not the config names it: every deployment has had one since the bridge
    shipped, and a config that could not be read must not be able to make the
    default lane disappear.

    :param config: A pre-loaded config mapping, or ``None`` to load the rendered
        one.
    :returns: Lane service keys, lane 1 first. A one-element tuple IS the
        single-lane deployment.
    """
    services = _loaded_config(config).get("services")
    if not isinstance(services, dict):
        services = {}
    keys = [key for key in LANE_KEYS if isinstance(services.get(key), dict)]
    if LANE_ONE not in keys:
        keys.insert(0, LANE_ONE)
    return tuple(keys)


def lane_declared_target(lane: str, config: dict | None = None) -> str | None:
    """The control target a lane's own config block declares, or ``None``.

    ``None`` is the single-lane deployment: its block has never carried a
    ``target:`` key, because it serves the only target the deployment has. The
    caller supplies the deployment baseline in that case — the same fallback
    ``queue_backend.resolve_lane_identity`` applies bridge-side.

    :param lane: A lane's ``services.<lane>`` key.
    :param config: A pre-loaded config mapping, or ``None`` to load the
        rendered one.
    """
    services = _loaded_config(config).get("services")
    block = services.get(lane) if isinstance(services, dict) else None
    target = block.get("target") if isinstance(block, dict) else None
    return str(target) if isinstance(target, str) and target else None


def resolve_lane_bridge_urls(config: dict | None = None) -> dict[str, str]:
    """``{lane: base URL}`` for a multi-lane deployment; ``{}`` for a single-lane one.

    The empty mapping is not a failure — it is the statement that this
    deployment has nothing to address per lane, and it is what keeps a
    single-lane consumer on the one ``bridge_url`` it has always used. A lane
    whose URL cannot be resolved at all (a block with no port) is dropped rather
    than guessed at, so a consumer asking for it gets "no such lane" instead of
    another lane's bridge.
    """
    urls: dict[str, str] = {}
    for key in discover_lane_keys(config):
        try:
            urls[key] = resolve_bridge_url(key)
        except UnknownBlueskyLaneError:
            continue
    return urls if len(urls) > 1 else {}


def bridge_error_message(body: object, status: int) -> str:
    """Extract the bridge's FastAPI ``detail`` message, falling back to the status."""
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"Bluesky bridge returned HTTP {status}."


def unwrap_bridge_conflict_detail(body: object) -> dict | None:
    """Unwrap the bridge's nested 409 ``detail`` dict, or ``None`` when absent.

    A structured 409 from ``POST /draft/run`` nests a ``{"code", "detail",
    "revision"}`` payload under FastAPI's top-level ``detail`` key
    (``{"detail": {"code": ..., ...}}``). Return that nested dict so callers can
    read the discriminator, or ``None`` for a 409 whose ``detail`` is a plain
    string (e.g. a validation-gate rejection) -- in which case the caller falls
    back to its own default rendering. This captures only the shared
    unwrap-or-fallback decision; the divergent downstream rendering (the MCP
    server's ``run_launch_conflict`` error envelope vs the panel sidecar's raw
    409 JSON) stays at each call site.
    """
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return detail
    return None
