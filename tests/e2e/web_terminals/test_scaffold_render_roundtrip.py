"""End-to-end proof of the multi-user web-terminal generator
(``osprey.deployment.web_terminals``).

Two parts:

1. **Scaffold-render consistency** — render a sample facility-config (adapted
   from the shipped ``templates/facility-config.example.yml``
   ``modules.web_terminals`` stanza) through the REAL
   :func:`render_web_terminals` + :func:`lint_web_terminals` and assert
   internal consistency across every generated artifact: one compose service +
   one nginx route + one landing card + one volume pair per user, all four
   port families allocated and non-colliding, ``OSPREY_TERMINAL_USER=<user>``
   per service, and a clean lint (zero findings). Also exercises the
   ``osprey scaffold web-terminals render`` CLI verb via subprocess for a true
   operator-path check.

2. **als-profiles topology round-trip** — the external ``als-profiles``
   reference repo (NOT part of this worktree; the production ALS deploy) has
   no ``facility-config.yml`` of its own, but its ``docker-compose.host.yml``
   encodes the REAL per-user, four-port-family topology
   (``OSPREY_WEB_PORT``/``OSPREY_ARTIFACT_SERVER_PORT``/``OSPREY_ARIEL_PORT``/
   ``OSPREY_LATTICE_DASHBOARD_PORT`` per user). This test parses that topology
   LIVE (never hardcodes the users/ports), constructs a facility-config that
   reproduces it, renders it through the same generator, and asserts the
   output matches that topology's SHAPE — same users, four families per user,
   matching per-user container/volume names — with the Phase-1
   reconciliations applied (internal port 8087, fixed
   ``OSPREY_TERMINAL_USER=<user>`` replacing the pre-Phase-1 9087 /
   ``ALS_TERMINAL_USER`` convention still live in als-profiles today). This is
   an EXPECTED-SHAPE match, not byte-equality.

   Guarded by a skip when als-profiles isn't checked out locally (e.g. CI) so
   part 1 always runs regardless of this environment's layout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.deployment.web_terminals.render import render_web_terminals

pytestmark = pytest.mark.e2e

# The generator's full family set — web plus one family per registry companion
# server, derived exactly the way the render derives it (a newly registered
# panel shows up here without touching this file).
_FAMILY_ENV_VARS = {"web": "OSPREY_WEB_PORT", **PANEL_ENV_VARS}
_PORT_FAMILIES = tuple(_FAMILY_ENV_VARS)

# The classic four families the REAL als-profiles compose carries — that
# external artifact predates the registry-derived families, so Part 2 parses
# it with this historical set only (never the full derived set above).
_ALS_CLASSIC_ENV_VARS = {
    "web": "OSPREY_WEB_PORT",
    "artifact": "OSPREY_ARTIFACT_SERVER_PORT",
    "ariel": "OSPREY_ARIEL_PORT",
    "lattice": "OSPREY_LATTICE_DASHBOARD_PORT",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALS_PROFILES_ROOT = Path("/Users/thellert/LBL/ML/als-profiles")
_ALS_HOST_COMPOSE = _ALS_PROFILES_ROOT / "docker-compose.host.yml"


def _env_map(env_list: list) -> dict[str, str]:
    return {item.split("=", 1)[0]: item.split("=", 1)[1] for item in env_list if "=" in item}


# ---------------------------------------------------------------------------
# Part 1: scaffold-render consistency
# ---------------------------------------------------------------------------


def _sample_config() -> dict:
    """A facility-config exercising the web_terminals stanza, adapted from the
    shipped ``templates/facility-config.example.yml`` (same base
    ports/users/landing-groups shape as that reference file's
    ``modules.web_terminals`` stanza), plus the deploy/facility/registry
    sections ``render_web_terminals()`` reads. The roster uses the explicit
    ``{name, index}`` form — the lint-clean identity form the
    ``bare_list_port_drift_risk`` warning steers legacy bare-string lists
    toward."""
    return {
        "facility": {
            "name": "Demo Light Source",
            "prefix": "dls",
            "timezone": "America/Los_Angeles",
        },
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 9080,
                "web_base_port": 9091,
                "artifact_base_port": 9291,
                "ariel_base_port": 9391,
                "lattice_base_port": 9491,
                "users": [
                    {"name": "alice", "index": 0},
                    {"name": "bob", "index": 1},
                    {"name": "carol", "index": 2},
                ],
                "landing": {
                    "groups": [
                        {"type": "users"},
                        {
                            "type": "links",
                            "label": "Facility Tools",
                            "links": [
                                {"label": "Elog", "url": "https://elog.dls.example.org"},
                                {
                                    "label": "Status Page",
                                    "url": "https://status.dls.example.org",
                                },
                            ],
                        },
                    ]
                },
            }
        },
    }


def test_scaffold_render_consistency_across_all_generated_artifacts() -> None:
    """The full generator (render + lint) produces internally-consistent,
    per-family artifacts for a sample facility-config, with a clean lint
    (zero findings, not just zero errors)."""
    # Arrange
    config = _sample_config()
    web_terminals = config["modules"]["web_terminals"]
    roster = web_terminals["users"]
    users = [entry["name"] for entry in roster]
    # Same effective base set the render allocates from: config values plus
    # registry defaults for families the config doesn't pin.
    base_ports = base_ports_from_config(web_terminals)

    # Act
    findings = lint_web_terminals(config)
    artifacts = render_web_terminals(config)

    # Assert: clean lint.
    assert findings == [], f"lint reported findings on a well-formed config: {findings}"

    # Assert: exactly the three generated artifacts.
    assert set(artifacts) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }

    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])
    nginx_conf = artifacts["nginx/nginx.conf"]
    landing_html = artifacts["nginx/landing.html"]

    # One compose service + one volume pair per user (+ nginx).
    assert set(compose["services"]) == {"nginx", *(f"web-{u}" for u in users)}
    assert set(compose["volumes"]) == {
        vol for u in users for vol in (f"{u}-claude-config", f"{u}-agent-data")
    }

    # One reverse-proxy route + one trailing-slash-redirect bookmark per user —
    # no more, no less (no drift between the number of compose services and the
    # number of routes). bind-nginx-reverse-proxy (task 1.2) replaced the
    # Phase-1 `location = /<user>` redirect-to-distinct-origin menu with a
    # single-origin `/u/<user>/` reverse proxy.
    assert nginx_conf.count("location /u/") == len(users)
    assert nginx_conf.count("location = /u/") == len(users)
    # One landing card per user PLUS the two extra "Facility Tools" links
    # (this sample config's landing.groups; the als-profiles round-trip below
    # has no extra groups, so its check compares 1:1 against users).
    extra_links = len(web_terminals["landing"]["groups"][1]["links"])
    assert landing_html.count('class="landing-card-label"') == len(users) + extra_links

    seen_ports: set[int] = set()
    for entry in roster:
        index, user = entry["index"], entry["name"]
        service = compose["services"][f"web-{user}"]
        env = _env_map(service["environment"])

        # Every port family allocated, matching allocate_ports() exactly,
        # and never colliding across users or families.
        expected = allocate_ports(base_ports, index)
        actual = {family: int(env[var]) for family, var in _FAMILY_ENV_VARS.items()}
        assert actual == expected, f"user {user!r} ports drifted from allocate_ports()"
        for port in actual.values():
            assert port not in seen_ports, f"port {port} collides across users/families"
            seen_ports.add(port)

        # nginx reverse-proxies /u/<user>/ to that user's own loopback `web`
        # upstream, and 301-redirects the no-trailing-slash bookmark into it.
        assert f"location /u/{user}/ {{" in nginx_conf
        assert f"proxy_pass http://127.0.0.1:{expected['web']}/;" in nginx_conf
        assert f"location = /u/{user} {{" in nginx_conf
        assert f"return 301 /u/{user}/;" in nginx_conf
        assert f">{user}<" in landing_html
        assert f'href="/u/{user}/"' in landing_html

        # Fixed per-service env var (Phase-1 contract, replaces the old
        # `${prefix|upper}_TERMINAL_USER` convention for every facility).
        assert env["OSPREY_TERMINAL_USER"] == user

    assert len(seen_ports) == len(users) * len(_PORT_FAMILIES)


def test_scaffold_render_cli_verb_matches_library_call(tmp_path: Path) -> None:
    """``osprey scaffold web-terminals render`` (the true operator path) writes
    the exact same three artifacts ``render_web_terminals()`` returns in-process
    — proving the CLI verb is a thin, non-drifting wrapper around the generator."""
    # Arrange
    config = _sample_config()
    config_path = tmp_path / "facility-config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "deploy"

    # Act
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "osprey.cli.main",
            "scaffold",
            "web-terminals",
            "render",
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Assert
    assert result.returncode == 0, (
        f"CLI render verb failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    expected = render_web_terminals(config)
    for relative_path, content in expected.items():
        written = output_dir / relative_path
        assert written.exists(), f"CLI did not write {relative_path}"
        assert written.read_text(encoding="utf-8") == content, (
            f"CLI-rendered {relative_path} diverged from the direct render_web_terminals() call"
        )


# ---------------------------------------------------------------------------
# Part 2: als-profiles topology round-trip
# ---------------------------------------------------------------------------


def _als_profiles_topology(host_compose_path: Path) -> tuple[list[str], dict[str, int]]:
    """Parse als-profiles' ``docker-compose.host.yml`` for its per-user,
    four-port-family services.

    A ``web-<user>`` service qualifies only if it carries all four
    ``OSPREY_*_PORT`` env vars — this deliberately EXCLUDES ``web-ariel`` (the
    shared ARIEL logbook terminal), which has no ``OSPREY_LATTICE_DASHBOARD_PORT``
    and is not part of the generic per-user model this generator targets.

    Returns ``(users, base_ports)`` with users ordered by ascending web port
    (index 0 = lowest) and base_ports derived from the data (the minimum port
    per family) — nothing here is hardcoded from a prior manual read of the file.
    """
    doc = yaml.safe_load(host_compose_path.read_text(encoding="utf-8")) or {}
    services = doc.get("services") or {}

    entries: list[tuple[str, dict[str, int]]] = []
    for name, svc in services.items():
        if not isinstance(name, str) or not name.startswith("web-"):
            continue
        env_map = _env_map((svc or {}).get("environment") or [])
        if not all(var in env_map for var in _ALS_CLASSIC_ENV_VARS.values()):
            continue
        user = name[len("web-") :]
        ports = {family: int(env_map[var]) for family, var in _ALS_CLASSIC_ENV_VARS.items()}
        entries.append((user, ports))

    assert entries, (
        f"no four-port-family web-<user> services found in {host_compose_path} — "
        "als-profiles' topology shape may have changed"
    )
    entries.sort(key=lambda entry: entry[1]["web"])
    users = [user for user, _ in entries]
    base_ports = {
        family: min(ports[family] for _, ports in entries) for family in _ALS_CLASSIC_ENV_VARS
    }
    return users, base_ports


@pytest.mark.skipif(
    not _ALS_HOST_COMPOSE.exists(),
    reason=(
        f"external als-profiles reference repo not found at {_ALS_PROFILES_ROOT} "
        "(it lives outside this worktree and is absent in CI/most environments); "
        "the scaffold-render consistency check above still runs unconditionally"
    ),
)
def test_generator_reproduces_als_profiles_topology_shape() -> None:
    """A facility-config built from als-profiles' REAL per-user port topology
    (parsed live from ``docker-compose.host.yml``, never hardcoded) renders
    through the same generator and reproduces that topology's SHAPE: same
    users, the classic four families at als-profiles' real ports (newer
    registry families allocate from their defaults), matching per-user
    container/volume names — with the Phase-1 reconciliations applied
    (8087 internal port, fixed ``OSPREY_TERMINAL_USER``)."""
    # Arrange
    users, base_ports = _als_profiles_topology(_ALS_HOST_COMPOSE)
    config = {
        "facility": {
            "name": "Advanced Light Source",
            "prefix": "als",
            "timezone": "America/Los_Angeles",
        },
        "registry": {"url": "git-local.als.lbl.gov:5050/physics/production/als-profiles"},
        "deploy": {"host": "als-deploy", "fqdn": "als-deploy.lbl.gov"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                # als-profiles' docker-compose.web.yml publishes nginx at 9080
                # (`ports: ["0.0.0.0:9080:9080"]`); docker-compose.host.yml
                # itself carries no nginx entry (it's the CA-broadcast
                # host-networking override, not the base compose file).
                "nginx_port": 9080,
                "web_base_port": base_ports["web"],
                "artifact_base_port": base_ports["artifact"],
                "ariel_base_port": base_ports["ariel"],
                "lattice_base_port": base_ports["lattice"],
                # Explicit {name, index} entries (the lint-clean identity form);
                # _als_profiles_topology() orders users by ascending web port,
                # so list position IS the port offset the live topology encodes.
                "users": [{"name": user, "index": i} for i, user in enumerate(users)],
            }
        },
    }

    # The effective base set the render allocates from: als-profiles' real
    # classic-four bases plus registry defaults for the newer families.
    full_base_ports = base_ports_from_config(config["modules"]["web_terminals"])

    # Act
    findings = lint_web_terminals(config)
    artifacts = render_web_terminals(config)

    # Assert: clean lint on the reproduced topology.
    assert findings == [], f"reproduced als-profiles topology failed lint: {findings}"

    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])
    nginx_conf = artifacts["nginx/nginx.conf"]
    landing_html = artifacts["nginx/landing.html"]
    compose_text = artifacts["docker-compose.web.yml"]

    # Same set of users as the real topology, one service + matching volumes each.
    assert set(compose["services"]) == {"nginx", *(f"web-{u}" for u in users)}
    assert set(compose["volumes"]) == {
        vol for u in users for vol in (f"{u}-claude-config", f"{u}-agent-data")
    }
    # bind-nginx-reverse-proxy (task 1.2): one reverse-proxy route + one
    # trailing-slash-redirect bookmark per user under the single-origin
    # `/u/<user>/` scheme, replacing the Phase-1 `location = /<user>` redirect.
    assert nginx_conf.count("location /u/") == len(users)
    assert nginx_conf.count("location = /u/") == len(users)
    assert landing_html.count('class="landing-card-label"') == len(users)

    seen_ports: set[int] = set()
    for index, user in enumerate(users):
        service = compose["services"][f"web-{user}"]
        env = _env_map(service["environment"])

        # als-profiles' ACTUAL container names are `als-web-<user>` — this is a
        # literal, not just shape-level, match against the production topology.
        assert service["container_name"] == f"als-web-{user}"
        assert f">{user}<" in landing_html

        # Phase-1 reconciliation: fixed OSPREY_TERMINAL_USER=<user>, replacing
        # the pre-Phase-1 `ALS_TERMINAL_USER` convention als-profiles uses today.
        assert env["OSPREY_TERMINAL_USER"] == user
        assert "ALS_TERMINAL_USER" not in env

        # The classic four families reproduce the exact als-profiles host ports
        # (index-aligned); the newer registry families allocate from defaults.
        expected = allocate_ports(full_base_ports, index)
        actual = {family: int(env[var]) for family, var in _FAMILY_ENV_VARS.items()}
        assert actual == expected, f"user {user!r} ports diverged from als-profiles topology"
        for port in actual.values():
            assert port not in seen_ports
            seen_ports.add(port)

        # nginx reverse-proxies /u/<user>/ to that user's own loopback `web`
        # upstream (als-profiles' real host port), and 301-redirects the
        # no-trailing-slash bookmark into it.
        assert f"location /u/{user}/ {{" in nginx_conf
        assert f"proxy_pass http://127.0.0.1:{expected['web']}/;" in nginx_conf
        assert f"location = /u/{user} {{" in nginx_conf
        assert f"return 301 /u/{user}/;" in nginx_conf

        # Phase-1 reconciliation: internal container port is 8087 (never the
        # pre-Phase-1 9087 seen in als-profiles' real `0.0.0.0:<port>:9087` map).
        assert f"http://127.0.0.1:{expected['web']}/health" in service["healthcheck"]["test"][1]

    assert len(seen_ports) == len(users) * len(_PORT_FAMILIES)
    assert "8087" in compose_text
    assert "9087" not in compose_text
