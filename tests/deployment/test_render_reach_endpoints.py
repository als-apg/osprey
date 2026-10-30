"""Render containers must reach host-network endpoints where they actually live.

Web-terminal render projects (the ``control-assistant-{readonly,readwrite,
ariel}`` presets) render as *attached* projects and run as per-user containers
with ``network_mode: host``. Inside such a container ``localhost`` IS the
deployment host, and compose service DNS names (``agent_record``) resolve to
nothing. Every URL a render container derives therefore has to target the
loopback address and the HOST deployment's ports — and every host-side probe of
a render sidecar has to knock on the port the sidecar actually binds.

Each test here joins two subsystems that have thorough unit coverage in
isolation — the OTLP block and the compose render; the ariel
health category and the web-server address registry; the reach projection and
the connectors' own endpoint resolvers — because the defects live exactly in
the seam neither suite crosses.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record.config import parse_agent_record_block
from osprey.build.claude_code_telemetry import record_telemetry_env
from osprey.deployment.reach import project_attached_overrides, reach_dials, reach_errors
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.deployment.web_terminals.roster import resolve_cards
from osprey.health.core.ariel import ariel
from osprey.port_layout import resolve_port_base
from osprey.registry.web import resolve_web_server_address

#: The T4 fixture drives a real ``osprey init`` + ``osprey build`` — seconds,
#: not milliseconds, exactly like tests/deployment/test_render_e2e.py.
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
def built_render_stack(tmp_path_factory) -> Path:
    """The control-assistant hosting preset built once, renders included.

    Same pattern as ``tests/cli/test_tier_app_presets.py::_build_render_stack``:
    one real ``osprey init --preset control-assistant`` + ``osprey build``
    renders the host project at ``build/`` and one attached project per catalog
    render beside it at ``build/<repo>-<render>``.
    """
    from osprey.cli.build_cmd import build
    from osprey.cli.init_cmd import init

    repo = tmp_path_factory.mktemp("render-reach") / "my-facility"
    runner = CliRunner()
    created = runner.invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    built = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code == 0, built.output
    return repo


def test_host_networked_render_telemetry_targets_loopback(built_render_stack: Path) -> None:
    """A host-networked render container's OTLP endpoint must target loopback
    at the HOST deployment's agent-record port — inside ``network_mode: host``
    a compose DNS name resolves to nothing, so an endpoint carrying it drops
    every log event silently.

    Joins the rendered per-user service's environment to the OTLP block
    (``claude_code_telemetry.record_telemetry_env``), exactly as the agent
    launcher inside that container builds it once its record login is in
    place; neither the render tests nor the block's unit tests cross that seam.
    """
    repo = built_render_stack
    host_config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    host_record_port = parse_agent_record_block(
        host_config, base=resolve_port_base(host_config)
    ).port

    compose = yaml.safe_load(render_web_terminals(host_config)["docker-compose.web.yml"])
    web_terminals = host_config["modules"]["web_terminals"]
    catalog = web_terminals["renders"]
    roster = resolve_cards(
        web_terminals,
        host_config.get("registry") or {},
        strict=True,
    )

    checked = 0
    wrong: list[str] = []
    for entry in roster:
        tier = entry.get("render")
        if not tier or tier not in catalog:
            continue
        service = compose["services"][f"web-{entry['name']}"]
        if service.get("network_mode") != "host":
            continue
        env = {k: _interpolate(v) for k, v in _service_env(service).items() if v is not None}
        # The login arrives by the service's own env_file; any usable one will do.
        env["OSPREY_AGENT_RECORD_LOGIN"] = "probe:token"

        block = record_telemetry_env(env)
        endpoints = (
            block.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
            block.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", ""),
        )
        checked += 1
        for endpoint in endpoints:
            parsed = urlsplit(endpoint)
            if parsed.hostname not in _LOOPBACK_HOSTS or parsed.port != host_record_port:
                wrong.append(f"web-{entry['name']} (render {tier}): {endpoint!r}")

    assert checked, (
        "Precondition failed: the built control-assistant stack exposed no "
        "host-networked render — the defect surface this test exists for was "
        "never reached."
    )
    assert not wrong, (
        "Host-networked render containers derive an OTLP endpoint that is not "
        f"loopback:{host_record_port} (the host deployment's agent-record "
        "port): " + "; ".join(wrong)
    )


async def test_ariel_health_probes_the_port_the_panel_listens_on(monkeypatch) -> None:
    """The ``ariel`` health category must probe the address the ARIEL panel
    actually binds — ``resolve_web_server_address("ariel", cfg)``, which honours
    the ``OSPREY_ARIEL_PORT`` env override multi-user compose renders export —
    not the ``ariel.web.port``/layout-default config value alone.

    Closest existing tests, and why they could not see this:

    * ``tests/health/core/test_ariel.py::test_status_url_defaults`` pins the
      probe URL to the ARIEL slot of the default block with the env var unset —
      it asserts the config-only derivation as CORRECT and never sets
      ``OSPREY_ARIEL_PORT``.
    * ``tests/health/core/test_web_panels.py::TestBuiltinAddressResolution::
      test_port_env_override_is_honoured`` proves exactly this env override —
      with OSPREY_ARIEL_PORT set, even — but for the ``web_panels`` category,
      which was converted to the canonical resolver; the ``ariel`` category
      never was, and no test joins the two.
    """
    monkeypatch.setenv("OSPREY_ARIEL_PORT", "10301")

    config = {"ariel": {"database": {"uri": "postgresql://ariel@localhost/ariel"}}}
    # The expectation is the resolver's answer, not a literal: this is the same
    # derivation the panel launcher is partial-bound to, so the test encodes
    # "the address the panel binds" rather than restating 9391.
    expected_host, expected_port = resolve_web_server_address("ariel", config)
    assert expected_port == 10301  # sanity: the override reached the resolver

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
        "ariel.web.port / the layout default only and ignores the multi-user env "
        "override — so every multi-user terminal reports ARIEL unreachable."
    )


# ---------------------------------------------------------------------------
# Every control target a render can switch to, on the host's own ports
# ---------------------------------------------------------------------------

_HOST_STANDIN_PORT = 5074
_HOST_VA_PORT = 5064


def _standin_baseline_host() -> dict:
    """A hosting deployment whose baseline is the live stand-in.

    Three targets at once: the facility's ``epics`` block naming the real
    machine ``live`` still means, the sandbox virtual accelerator, and the
    stand-in soft IOC this deployment stood up for itself — each with its own
    connector block, each published on its own port.
    """
    return {
        "control_system": {
            "type": "live_standin",
            "connector": {
                "epics": {
                    "gateways": {"read_only": {"address": "gateway.facility.org", "port": 5066}}
                },
                "virtual_accelerator": {"gateways": {"read_only": {"address": "localhost"}}},
                "live_standin": {
                    "gateways": {
                        "read_only": {
                            "address": "localhost",
                            "port": _HOST_STANDIN_PORT,
                            "use_name_server": True,
                        }
                    }
                },
            },
        },
        "services": {
            "virtual_accelerator": {"port": _HOST_VA_PORT},
            "live_standin": {"port": _HOST_STANDIN_PORT},
        },
        "deployed_services": ["virtual_accelerator", "live_standin"],
    }


def _with_overrides(render: dict, overrides: dict) -> dict:
    """*render* with each dotted override written in, as the build writes them."""
    for dotted, value in overrides.items():
        node = render
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return render


def test_a_standin_baseline_render_dials_the_hosts_ports_on_loopback() -> None:
    """A render of a stand-in-baseline deployment must reach BOTH self-standing
    machines where the host actually publishes them.

    An attached render container shares the host's network namespace, renders
    ``services: {}``, and is told each port by the projection — so what its
    connectors dial is decided entirely by what the reach registry projected
    into it. Gated on ``control_system.type``, the virtual accelerator's port
    was withheld from exactly these renders (the baseline is the stand-in, not
    the VA), and a session switched to ``va`` inside the container fell back to
    the connector's compiled-in default port — a dial at a port this deployment
    does not publish. SC-9: the VA port a render carries is the host's.
    """
    host = _standin_baseline_host()
    tier = {
        "control_system": host["control_system"],
        "services": {},
        "deployed_services": [],
    }

    rendered = _with_overrides(tier, project_attached_overrides(host, tier))

    dials = {contract.service: dial for contract, _consumer, dial in reach_dials(rendered)}
    assert dials["virtual_accelerator"] == ("localhost", _HOST_VA_PORT)
    assert dials["live_standin"] == ("localhost", _HOST_STANDIN_PORT)
    for service, dial in dials.items():
        if service in {"virtual_accelerator", "live_standin"}:
            assert dial is not None and dial[0] in _LOOPBACK_HOSTS, (service, dial)
    assert reach_errors(rendered) == []
