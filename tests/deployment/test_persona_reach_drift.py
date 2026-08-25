"""Persona clients must follow the ports their hosting deployment actually publishes.

The invariant under test: **an attached render dials the ports its hosting
deployment actually publishes.** Web-terminal persona projects (the
``control-assistant-{readonly,readwrite,ariel}`` presets) render as *attached*
projects whose ``services:`` carries no ``bluesky`` or ``virtual_accelerator``
block, so several in-container clients resolve their endpoint through a
fallback chain that bottoms out in a compiled-in default. At the shipped
defaults that default happens to equal the port the host publishes — so every
existing test passes while proving nothing about *following*. This module
breaks the coincidence: it builds one control-assistant stack whose hosting
profile MOVES the ports, confirms the host render moved (the premise), then
resolves each endpoint exactly the way the in-container client does and
asserts the persona followed.

Closest existing tests, and why they were blind:

* ``tests/cli/test_profile_va_archiver_block.py`` proves the invariant for the
  one client that does it right — ``va_archiver_config_overrides`` derives
  ``archiver.mongodb_archiver.*`` for attached renders from the hosting
  profile, and refuses an attached profile without ``va_archiver.host``. It
  never looks at the bluesky, VA-gateway, or telemetry clients.
* ``tests/services/ariel_search/test_dsn_derivation.py`` proves
  ``resolve_ariel_dsn`` follows ``services.postgresql`` when the block is
  present — but it hands the resolver hand-built dicts and never renders an
  attached persona, so it cannot say what block a persona actually has.
* ``tests/deployment/test_persona_reach_endpoints.py`` catches the telemetry
  endpoint deriving a wrong HOST under ``network_mode: host`` — at the shipped
  default ports. It never moves a port, so port drift stays invisible there
  for every client.
* ``tests/cli/test_persona_presets.py`` builds this same stack — at default
  ports, where "follows the config" and "hardcodes the default" produce
  identical renders.

One candidate client is deliberately NOT parametrized here: the ARIEL
Postgres DSN. Its only operator-facing port lever is the profile's
``config: services.postgresql.port_host`` dotted key, and a persona profile is
the persona delta merged over the hosting ``profile.yml`` — ``config:``
overlay included — so the moved key lands in every persona render's own
``services.postgresql`` block and ``resolve_ariel_dsn`` follows it (verified:
with ``services.postgresql.port_host: 15432`` on the hosting profile, every
persona render carries the same block and the derived DSN dials 15432). That
client already satisfies the invariant; asserting drift for it would fail to
fail.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

import pytest
import yaml
from click.testing import CliRunner

#: The fixture drives a real ``osprey init`` + ``osprey build`` — seconds, not
#: milliseconds, exactly like tests/deployment/test_persona_reach_endpoints.py.
pytestmark = pytest.mark.slow

# Moved host ports — each distinct from the shipped default it replaces
# (bluesky 8090, virtual_accelerator 5064, openobserve 5080), so a client that
# keeps its compiled-in default is distinguishable from one that follows.
MOVED_BLUESKY_PORT = 18090
MOVED_VA_PORT = 15064
MOVED_OPENOBSERVE_PORT = 15080


@pytest.fixture(scope="module")
def moved_port_stack(tmp_path_factory) -> Path:
    """A control-assistant stack whose hosting profile moved its service ports.

    Same build pattern as ``tests/cli/test_persona_presets.py::
    _build_persona_stack``, with one twist: between ``init`` and ``build`` the
    hosting ``profile.yml`` is edited to move every port under test — the
    top-level ``bluesky.port`` and ``virtual_accelerator.port`` blocks, and the
    ``config:`` dotted key ``services.openobserve.port``. The build renders the
    host project at ``build/`` and one attached project per catalog persona
    beside it at ``build/<repo>-<persona>``.
    """
    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    repo = tmp_path_factory.mktemp("persona-drift") / "my-facility"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output

    profile_path = repo / "profile.yml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile.setdefault("bluesky", {})["port"] = MOVED_BLUESKY_PORT
    profile.setdefault("virtual_accelerator", {})["port"] = MOVED_VA_PORT
    profile.setdefault("config", {})["services.openobserve.port"] = MOVED_OPENOBSERVE_PORT
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    built = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return repo


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dial_port_bluesky(persona_dir: Path, persona_cfg: dict, monkeypatch) -> int:
    """The port the persona's Bluesky clients dial — via the real resolver.

    ``resolve_bridge_url`` is the single source of truth both the bluesky MCP
    server and the bluesky-web sidecar import; its config fallback reads the
    ``OSPREY_CONFIG`` file, which here is the persona's own rendered
    ``config.yml`` — exactly the file the in-container process sees. The
    env-var override is cleared: it is minted per bridge instance at runtime
    and an attached persona render carries none.
    """
    from osprey.bluesky_bridge_connection import resolve_bridge_url
    from osprey.utils.workspace import reset_config_cache

    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(persona_dir / "config.yml"))
    reset_config_cache()
    try:
        url = resolve_bridge_url()
        return urlsplit(url).port or 80
    finally:
        reset_config_cache()


def _dial_port_va(persona_dir: Path, persona_cfg: dict, monkeypatch) -> int:
    """The CA port the persona's VA connector dials — via the real fill.

    ``VirtualAcceleratorConnector`` default-fills every gateway ``port`` from
    ``services.virtual_accelerator.port`` through ``fill_gateway_ports`` before
    any socket is opened; that pure resolution path is called here with the
    persona's own rendered connector block and the persona's own rendered
    ``config.yml`` as the ambient config (``CONFIG_FILE``).
    """
    from osprey_connectors.control_system.va_connector import fill_gateway_ports
    from osprey_connectors.workspace import reset_config_cache

    monkeypatch.setenv("CONFIG_FILE", str(persona_dir / "config.yml"))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    reset_config_cache()
    try:
        va_block = persona_cfg["control_system"]["connector"]["virtual_accelerator"]
        filled = fill_gateway_ports(va_block)
        ports = {gateway["port"] for gateway in filled["gateways"].values()}
        assert len(ports) == 1, f"gateways disagree on the VA port: {sorted(ports)}"
        return int(ports.pop())
    finally:
        reset_config_cache()


def _dial_port_telemetry(persona_dir: Path, persona_cfg: dict, monkeypatch) -> int:
    """The OTLP port the persona's agent launch exports to — via the real resolver.

    The port is resolved the way the launch path resolves it
    (``resolve_openobserve_port``): the deploy environment's declaration, else
    the port the persona's own config publishes the store on. The environment
    is cleared because the per-user compose declares a host and no port.
    """
    from osprey.build.claude_code_telemetry import (
        OPENOBSERVE_PORT_ENV_VAR,
        _resolve_telemetry_endpoint,
        resolve_openobserve_port,
    )

    monkeypatch.delenv(OPENOBSERVE_PORT_ENV_VAR, raising=False)
    endpoint = _resolve_telemetry_endpoint(
        persona_cfg["claude_code"]["telemetry"],
        in_container=True,
        openobserve_host=None,
        openobserve_port=resolve_openobserve_port(persona_cfg),
    )
    return urlsplit(endpoint).port or 80


class _Client(NamedTuple):
    """One in-container persona client and the host port it must follow."""

    client: str
    persona: str
    host_key: str  # dotted key under `services` in the HOST build/config.yml
    moved_port: int
    dial_port: Callable[[Path, dict, pytest.MonkeyPatch], int]


CLIENTS = [
    _Client(
        client="bluesky bridge URL",
        persona="readwrite",  # the only persona that runs the bluesky MCP server
        host_key="bluesky.port",
        moved_port=MOVED_BLUESKY_PORT,
        dial_port=_dial_port_bluesky,
    ),
    _Client(
        client="virtual-accelerator CA gateway port",
        persona="readonly",
        host_key="virtual_accelerator.port",
        moved_port=MOVED_VA_PORT,
        dial_port=_dial_port_va,
    ),
    _Client(
        client="telemetry OTLP endpoint",
        persona="readonly",
        host_key="openobserve.port",
        moved_port=MOVED_OPENOBSERVE_PORT,
        dial_port=_dial_port_telemetry,
    ),
]


@pytest.mark.parametrize("spec", CLIENTS, ids=[c.client.replace(" ", "-") for c in CLIENTS])
def test_persona_clients_follow_the_hosts_moved_ports(
    moved_port_stack: Path, spec: _Client, monkeypatch
) -> None:
    """An attached render dials the ports its hosting deployment actually
    publishes — even after the operator moves them on the hosting profile.

    Premise first: the HOST render must reflect the moved port (it does — the
    host stack genuinely moves). Then the persona's endpoint is resolved
    exactly the way the in-container client resolves it, and the port it would
    dial must equal the port the host publishes.
    """
    host_cfg = _load_config(moved_port_stack / "build" / "config.yml")
    service, key = spec.host_key.split(".")
    host_port = ((host_cfg.get("services") or {}).get(service) or {}).get(key)
    assert host_port == spec.moved_port, (
        f"premise broken: host render services.{spec.host_key} is {host_port}, "
        f"expected the moved port {spec.moved_port}"
    )

    persona_dir = moved_port_stack / "build" / f"{moved_port_stack.name}-{spec.persona}"
    persona_cfg = _load_config(persona_dir / "config.yml")

    dialed = spec.dial_port(persona_dir, persona_cfg, monkeypatch)
    assert dialed == host_port, (
        f"{spec.client}: the '{spec.persona}' persona would dial port {dialed}, but the "
        f"hosting deployment publishes this service on port {host_port} (host "
        f"build/config.yml services.{spec.host_key}) — the attached render silently kept "
        f"its compiled-in default after the operator moved the port on the hosting profile."
    )
