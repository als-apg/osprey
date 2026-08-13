"""End-to-end proof of the multi-user web-terminal generator
(``osprey.deployment.web_terminals``).

**Scaffold-render consistency** — render a sample ``modules.web_terminals``
stanza through the REAL :func:`render_web_terminals` +
:func:`lint_web_terminals` and assert internal consistency across every
generated artifact: one compose service + one nginx route + one landing card +
one volume pair per user, all four port families allocated and non-colliding,
``OSPREY_TERMINAL_USER=<user>`` per service, and a clean lint (zero findings).

Also exercises the ``osprey scaffold web-terminals render`` CLI verb via
subprocess for a true operator-path check. That verb reads the stanza from a
deployment repo's BUILT config, so the sample is written where a build would
have put it.

(A second part — a round-trip against als-profiles' hand-rolled
``docker-compose.host.yml`` topology — was removed when als-profiles migrated
to rendering its web stack from the profile: the external reference topology
this generator was validated against no longer exists, because the generator
replaced it.)
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


def _env_map(env_list: list) -> dict[str, str]:
    return {item.split("=", 1)[0]: item.split("=", 1)[1] for item in env_list if "=" in item}


# ---------------------------------------------------------------------------
# Scaffold-render consistency
# ---------------------------------------------------------------------------


def _sample_config() -> dict:
    """A self-contained config exercising the web_terminals stanza: the base
    ports/users/landing-groups shape of a ``modules.web_terminals`` block, plus
    the deploy/facility/registry sections ``render_web_terminals()`` reads. The roster uses the explicit
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
    # (this sample config's landing.groups).
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
    — proving the CLI verb is a thin, non-drifting wrapper around the generator.

    The verb is repo-scoped and reads the stanza from the repo's BUILT config,
    so the sample is written to ``<repo>/build/config.yml`` — where a build puts
    it — and the repo is marked the way every repo-scoped verb finds one, by a
    ``profile.yml`` at its root. Nothing is built here: the render is a pure
    function of the stanza, and staging the config directly is what keeps this
    a test of the wrapper rather than of the build.

    ``project_root`` is stamped into the sample because a real render always
    carries it, and both sides of this comparison read it: the repo-id label the
    compose overlay carries is derived from that path, so a sample without it
    would have the two renders disagree on the label for a reason that has
    nothing to do with the wrapper.
    """
    # Arrange
    repo = tmp_path / "demo-project"
    build_dir = repo / "build"
    build_dir.mkdir(parents=True)
    config = _sample_config()
    config["project_root"] = str(repo)
    (build_dir / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (repo / "profile.yml").write_text("name: demo-project\n", encoding="utf-8")
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
            "--repo",
            str(repo),
            "--output",
            str(output_dir),
        ],
        cwd=str(tmp_path),
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
