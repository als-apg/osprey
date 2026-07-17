"""Golden-fixture byte-equality baseline for `render_web_terminals()` (PLAN Task 1.1).

This is the FIRST guard committed before any persona-render change lands (P4
personas/portability). Its job is narrow and mechanical: pin today's rendered
output for a representative, no-personas facility config so any *unintended*
drift in Task 2's render threading (`render.py`, `docker-compose.web.yml.j2`,
`seeding.py`) shows up as a byte-diff here, immediately, rather than as a
runtime surprise in a downstream lifecycle/e2e test.

**Update discipline** — read this before touching the golden files:
  - A failure here means the rendered compose/nginx/landing output changed.
    That is NOT automatically a bug: personas threading (Task 2.1) is
    *expected* to eventually touch `docker-compose.web.yml.j2` (at minimum,
    the new `OSPREY_TERMINAL_WEB_PORT` declaration line — see PROPOSAL.md's
    Scope section, "the one known case").
  - Byte-equality exists to guard against DRIFT, not to freeze the templates
    forever. When a render change is deliberate and reviewed, re-generate the
    three files under `golden/` from the new `render_web_terminals()` output
    in the SAME reviewed change that made the template edit (never as a
    separate, unreviewed "make the test pass again" commit) — so the diff a
    reviewer sees is exactly: template change + the resulting golden delta,
    side by side.
  - To regenerate: call `render_web_terminals(EXAMPLE_CONFIG)` (defined
    below) and overwrite `golden/docker-compose.web.yml`,
    `golden/nginx.conf`, and `golden/landing.html` with the three returned
    values (`docker-compose.web.yml`, `nginx/nginx.conf`, and
    `nginx/landing.html` respectively). Do not hand-edit the golden files.

``EXAMPLE_CONFIG`` mirrors the shipped
`osprey-build-deploy` skill's `facility-config.example.yml`
`modules.web_terminals` stanza (two bare-string users, the `users` +
`links` landing groups, no `personas:` block) — the reference "no-personas"
config every real facility profile's web-terminals section is patterned on.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.render import render_web_terminals

_GOLDEN_DIR = Path(__file__).parent / "golden"

# Kept in sync with the `modules.web_terminals` stanza in
# src/osprey/templates/skills/osprey-build-deploy/templates/facility-config.example.yml
# (facility/registry/deploy values match that file's top-level sections too),
# minus the commented-out (inert) auth/tls block, which the example file itself
# leaves disabled by default.
EXAMPLE_CONFIG: dict = {
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
            "users": ["alice", "bob"],
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


def _read_golden(name: str) -> str:
    return (_GOLDEN_DIR / name).read_text()


def test_golden_fixtures_exist() -> None:
    """Sanity check the baseline itself is present before comparing against it —
    a missing golden file must fail loudly here, not be misread as an empty-string
    byte-match by the tests below."""
    for name in ("docker-compose.web.yml", "nginx.conf", "landing.html"):
        assert (_GOLDEN_DIR / name).is_file(), f"missing golden fixture: {name}"


def test_render_matches_golden_compose_byte_for_byte() -> None:
    """`docker-compose.web.yml` output is byte-identical to the committed baseline."""
    artifacts = render_web_terminals(EXAMPLE_CONFIG)
    assert artifacts["docker-compose.web.yml"] == _read_golden("docker-compose.web.yml")


def test_render_matches_golden_nginx_conf_byte_for_byte() -> None:
    """`nginx/nginx.conf` output is byte-identical to the committed baseline."""
    artifacts = render_web_terminals(EXAMPLE_CONFIG)
    assert artifacts["nginx/nginx.conf"] == _read_golden("nginx.conf")


def test_render_matches_golden_landing_html_byte_for_byte() -> None:
    """`nginx/landing.html` output is byte-identical to the committed baseline."""
    artifacts = render_web_terminals(EXAMPLE_CONFIG)
    assert artifacts["nginx/landing.html"] == _read_golden("landing.html")


def test_golden_compose_is_valid_yaml_with_expected_services() -> None:
    """The committed baseline itself must stay sane YAML with the two example
    users' services — guards against a corrupt/truncated golden file passing
    the byte-equality checks above by accident (e.g. both sides empty)."""
    compose = yaml.safe_load(_read_golden("docker-compose.web.yml"))
    assert set(compose["services"].keys()) == {"nginx", "web-alice", "web-bob"}


def _persona_config() -> dict:
    """EXAMPLE_CONFIG reshaped into the demo's two-persona roster: alice=operator,
    bob=physicist. Each user carries an explicit ``persona`` reference resolved
    against a matching ``personas`` catalog, so resolve_personas() returns a
    non-``None`` persona for both — the case that produces sublabel badges."""
    config = copy.deepcopy(EXAMPLE_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["personas"] = {
        "operator": {
            "project": "dls-operator",
            "project_path": "../dls-operator",
            "build_profile": "profiles/operator.yml",
        },
        "physicist": {
            "project": "dls-physicist",
            "project_path": "../dls-physicist",
            "build_profile": "profiles/physicist.yml",
        },
    }
    web_terminals["users"] = [
        {"name": "alice", "index": 0, "persona": "operator"},
        {"name": "bob", "index": 1, "persona": "physicist"},
    ]
    return config


def test_persona_users_render_persona_sublabel_badge() -> None:
    """A roster whose users resolve to personas renders each user card with its
    persona name as a sublabel badge (the demo's alice=operator / bob=physicist
    shape). The badge span is distinct from the `.landing-card-sublabel` CSS rule,
    so counting `class="landing-card-sublabel"` counts only rendered badges."""
    artifacts = render_web_terminals(_persona_config())
    landing = artifacts["nginx/landing.html"]

    assert landing.count('class="landing-card-sublabel"') == 2
    assert '<span class="landing-card-sublabel">operator</span>' in landing
    assert '<span class="landing-card-sublabel">physicist</span>' in landing


def test_bare_string_users_render_no_persona_sublabel() -> None:
    """The no-personas EXAMPLE_CONFIG (bare-string roster) renders user cards with
    NO persona sublabel badge — resolve_personas() returns ``persona=None``, the
    caller omits the ``sublabel`` key, and the template's guard skips the span so
    the card stays a plain {label, url} card, unchanged from pre-persona output."""
    artifacts = render_web_terminals(EXAMPLE_CONFIG)
    landing = artifacts["nginx/landing.html"]

    assert 'class="landing-card-sublabel"' not in landing
