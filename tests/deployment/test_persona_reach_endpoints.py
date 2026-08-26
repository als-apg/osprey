"""Persona containers must reach host-network endpoints where they actually live.

Web-terminal persona projects (the ``control-assistant-{readonly,readwrite,
ariel}`` presets) render as *attached* projects and run as per-user containers
with ``network_mode: host``. Inside such a container ``localhost`` IS the
deployment host, and compose service DNS names (``openobserve``) resolve to
nothing. Every URL a persona container derives therefore has to target the
loopback address and the HOST deployment's ports — and every host-side probe of
a persona sidecar has to knock on the port the sidecar actually binds.

Both tests here join two subsystems that each have thorough unit coverage in
isolation — the telemetry endpoint resolver and the compose render; the ariel
health category and the web-server address registry — because the defects live
exactly in the seam neither suite crosses.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from click.testing import CliRunner

from osprey.build.claude_code_telemetry import (
    OPENOBSERVE_PORT_ENV_VAR,
    _resolve_telemetry_endpoint,
    openobserve_published_port,
)
from osprey.deployment.web_terminals.personas import resolve_personas
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.health.core.ariel import ariel
from osprey.registry.web import resolve_web_server_address

#: The T4 fixture drives a real ``osprey init`` + ``osprey build`` — seconds,
#: not milliseconds, exactly like tests/deployment/test_persona_render_e2e.py.
pytestmark = pytest.mark.slow

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1"}

#: Minimal compose ``${VAR}`` / ``${VAR:-default}`` interpolation against an
#: empty deploy env — what compose does for a variable unset at create time.
_COMPOSE_INTERP = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}$")


def _interpolate(value: str) -> str:
    match = _COMPOSE_INTERP.match(value)
    if match is None:
        return value
    return match.group("default") or ""


def _service_env(service: dict) -> dict[str, str | None]:
    """The env mapping a compose service hands its container, both YAML forms.

    The list form (``- KEY=value``) is what the web overlay template emits; the
    mapping form is accepted too so the test keeps working if the template ever
    switches. A ``- KEY`` entry with no ``=`` maps to ``None`` (pass-through).
    """
    raw = service.get("environment") or []
    env: dict[str, str | None] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            env[str(key)] = None if value is None else _interpolate(str(value))
    else:
        for line in raw:
            key, sep, value = str(line).partition("=")
            env[key.strip()] = _interpolate(value) if sep else None
    return env


@pytest.fixture(scope="module")
def built_persona_stack(tmp_path_factory) -> Path:
    """The control-assistant hosting preset built once, personas included.

    Same pattern as ``tests/cli/test_persona_presets.py::_build_persona_stack``:
    one real ``osprey init --preset control-assistant`` + ``osprey build``
    renders the host project at ``build/`` and one attached project per catalog
    persona beside it at ``build/<repo>-<persona>``.
    """
    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    repo = tmp_path_factory.mktemp("persona-reach") / "my-facility"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    built = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return repo


def test_host_networked_persona_telemetry_targets_loopback(built_persona_stack: Path) -> None:
    """A host-networked persona container's derived OTLP endpoint must target
    loopback at the HOST deployment's ``services.openobserve.port`` — inside
    ``network_mode: host`` the compose DNS name ``openobserve`` resolves to
    nothing, so an endpoint carrying it drops every metric and log silently.

    Closest existing tests, and why they could not see this:

    * ``tests/cli/test_telemetry_env.py::test_openobserve_default_endpoint_container_host``
      pins ``in_container=True`` → ``http://openobserve:5080/...`` as the
      CORRECT behaviour — true for a bridge-networked service, but the test
      never joins the resolver to a compose service that declares
      ``network_mode: host`` and exports no ``OSPREY_OTEL_OPENOBSERVE_HOST``.
    * ``tests/deployment/web_terminals/test_render.py`` /
      ``test_golden_render.py`` parse the rendered compose (and see
      ``network_mode: host``) but never feed a per-user service's environment
      into the telemetry resolver, so the endpoint the agent would actually
      derive inside that container is asserted nowhere.
    """
    repo = built_persona_stack
    host_config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    host_openobserve_port = int(host_config["services"]["openobserve"]["port"])

    compose = yaml.safe_load(render_web_terminals(host_config)["docker-compose.web.yml"])
    web_terminals = host_config["modules"]["web_terminals"]
    catalog = web_terminals["personas"]
    roster = resolve_personas(
        web_terminals,
        host_config.get("registry") or {},
        (host_config.get("facility") or {}).get("prefix") or "",
        strict=True,
    )

    checked = 0
    wrong: list[str] = []
    for entry in roster:
        persona = entry.get("persona")
        if not persona or persona not in catalog:
            continue
        persona_config = yaml.safe_load(
            (repo / catalog[persona]["project_path"] / "config.yml").read_text(encoding="utf-8")
        )
        telemetry = (persona_config.get("claude_code") or {}).get("telemetry") or {}
        if not telemetry.get("enabled") or telemetry.get("backend") != "openobserve":
            continue

        service = compose["services"][f"web-{entry['name']}"]
        if service.get("network_mode") != "host":
            continue
        env = _service_env(service)

        # Exactly what the agent launcher resolves inside this container:
        # /.dockerenv exists in every persona container (in_container=True), the
        # only host override is whatever this compose service exports, and the
        # port is the service's declaration or else the port this persona's own
        # config says the store publishes (resolve_openobserve_port's rule).
        declared_port = env.get(OPENOBSERVE_PORT_ENV_VAR)
        endpoint = _resolve_telemetry_endpoint(
            telemetry,
            in_container=True,
            openobserve_host=env.get("OSPREY_OTEL_OPENOBSERVE_HOST"),
            openobserve_port=(
                int(declared_port) if declared_port else openobserve_published_port(persona_config)
            ),
        )
        parsed = urlsplit(endpoint)
        checked += 1
        if parsed.hostname not in _LOOPBACK_HOSTS or parsed.port != host_openobserve_port:
            wrong.append(f"web-{entry['name']} (persona {persona}): {endpoint}")

    assert checked, (
        "Precondition failed: the built control-assistant stack exposed no "
        "host-networked persona with an enabled openobserve telemetry backend — "
        "the defect surface this test exists for was never reached."
    )
    assert not wrong, (
        "Host-networked persona containers derive an OTLP endpoint that is not "
        f"loopback:{host_openobserve_port} (the host deployment's "
        "services.openobserve.port): "
        + "; ".join(wrong)
        + ". The web-terminal compose overlay sets no OSPREY_OTEL_OPENOBSERVE_HOST "
        "for per-user services (only the dispatch-worker compose does), so "
        "in_container=True derives the compose DNS name 'openobserve' — which "
        "resolves to nothing under network_mode: host, and every web-terminal "
        "agent's telemetry is dropped silently."
    )


async def test_ariel_health_probes_the_port_the_panel_listens_on(monkeypatch) -> None:
    """The ``ariel`` health category must probe the address the ARIEL panel
    actually binds — ``resolve_web_server_address("ariel", cfg)``, which honours
    the ``OSPREY_ARIEL_PORT`` env override multi-user compose renders export —
    not the ``ariel.web.port``/default-8085 config value alone.

    Closest existing tests, and why they could not see this:

    * ``tests/health/core/test_ariel.py::test_status_url_defaults`` pins the
      probe URL to ``http://127.0.0.1:8085/api/status`` with the env var unset —
      it asserts the config-only derivation as CORRECT and never sets
      ``OSPREY_ARIEL_PORT``.
    * ``tests/health/core/test_web_panels.py::TestBuiltinAddressResolution::
      test_port_env_override_is_honoured`` proves exactly this env override —
      with OSPREY_ARIEL_PORT=9391, even — but for the ``web_panels`` category,
      which was converted to the canonical resolver; the ``ariel`` category
      never was, and no test joins the two.
    """
    monkeypatch.setenv("OSPREY_ARIEL_PORT", "9391")

    config = {"ariel": {"database": {"uri": "postgresql://ariel@localhost/ariel"}}}
    # The expectation is the resolver's answer, not a literal: this is the same
    # derivation the panel launcher is partial-bound to, so the test encodes
    # "the address the panel binds" rather than restating 9391.
    expected_host, expected_port = resolve_web_server_address("ariel", config)
    assert expected_port == 9391  # sanity: the override reached the resolver

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"healthy": True})

    await ariel(config, transport=httpx.MockTransport(handler))()

    assert len(seen) == 1
    probed = urlsplit(seen[0])
    assert probed.port == expected_port, (
        f"osprey health probed ARIEL at port {probed.port}, but the panel in this "
        f"environment binds {expected_host}:{expected_port} (OSPREY_ARIEL_PORT, "
        "per registry.web.resolve_web_server_address). The ariel category reads "
        "ariel.web.port / the 8085 default only and ignores the multi-user env "
        "override — so every multi-user terminal reports ARIEL unreachable."
    )
