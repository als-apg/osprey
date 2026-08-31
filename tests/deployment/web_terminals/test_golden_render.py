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
    `nginx/landing.html` respectively). The `golden/tls_custom_port/`
    variant regenerates the same way from `_tls_custom_port_config()`, and
    holds `nginx.conf` alone. Do not hand-edit the golden files.

`golden/tls_custom_port/` is the ONE variant this module keeps: the same
facility with TLS terminated on a non-default port. It exists because the
listener port is the one value that reaches three separate places in the
rendered nginx.conf at once — both `listen ... ssl` lines and the cleartext
server's `301` target — and a default-port baseline pins none of them (at 443
the redirect deliberately names no port at all, so the redirect's port arm is
invisible to the default golden). Only `nginx.conf` is committed for the
variant: `tls.port` changes nothing in the landing output, and the two places
it does reach in the compose output — each user's `OSPREY_TERMINAL_LANDING_URL`
and `OSPREY_TERMINAL_EXTERNAL_ORIGIN` — are already pinned by `test_render.py`'s
non-default-TLS-port origin tests, so a second compose golden would only be
another file to regenerate.

``EXAMPLE_CONFIG`` is the reference "no-personas" shape a facility profile's
web-terminals section is patterned on: two bare-string users, the `users` +
`links` landing groups, and no `personas:` block.

It also carries **no port keys at all** — no ``nginx_port``, no
``*_base_port``. That is deliberate and load-bearing: every port in the three
golden files is one the port layout supplied from ``deployment.port_base``, so
this baseline pins the defaults a facility gets for free. Adding a port key
back would render that facility's override instead and stop covering them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.render import render_web_terminals

_GOLDEN_DIR = Path(__file__).parent / "golden"

# The auth/tls block is deliberately absent: it is off by default, and the
# goldens cover the shape a facility gets without opting into it.
#: Stands in for the one rendered value that cannot be committed: the
#: `com.osprey.repo-id` label is a hash of the deployment repo's RESOLVED path,
#: so it differs per checkout, and a literal in the golden would fail on every
#: machine but the one that generated it. Substituted at comparison time, the
#: same trick the exemplar fixture uses for the osprey version — byte-equality
#: still covers every other byte, including the label's presence and placement.
_REPO_ID_SENTINEL = "@REPO_ID@"


def _rendered_repo_id() -> str:
    """The identity `render_web_terminals` will bake for ``EXAMPLE_CONFIG``."""
    from osprey.deployment.compose_generator import repo_identity, resolve_repo_root

    return repo_identity(resolve_repo_root(EXAMPLE_CONFIG))


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
    """The committed baseline, with its one per-checkout sentinel resolved.

    `name` is relative to `golden/`, so a variant subdirectory's file is read
    through the SAME sentinel substitution as the default baseline
    (``_read_golden("tls_custom_port/nginx.conf")``) rather than through a
    second reader that could drift from this one.
    """
    return (_GOLDEN_DIR / name).read_text().replace(_REPO_ID_SENTINEL, _rendered_repo_id())


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


#: The variant's TLS listener. 8443 is the conventional unprivileged HTTPS
#: alternate — the port a rootless deployment actually terminates on — and it is
#: deliberately outside every band OSPREY owns: it is not in the port-block
#: layout (a default deployment's block is `port_base`..`port_base + 999`, and
#: `tls.port` is not a layout slot at all), and it is not one of the retired
#: framework literals `tests/test_port_literals.py` guards (the 8085-8097 and
#: 9070-9100 bands, the 9x00 family anchors, the loose bluesky/qmd/VA numbers).
#: So at the default base the number is outside every band OSPREY owns, and
#: `tls.port` is not a layout slot at any base.
_TLS_CUSTOM_PORT = 8443

#: Where the variant's single rendered file lives, relative to `golden/`.
_TLS_CUSTOM_PORT_GOLDEN = "tls_custom_port/nginx.conf"


def _tls_custom_port_config() -> dict:
    """EXAMPLE_CONFIG with TLS terminated on a NON-default port.

    Everything else is the baseline facility, so a diff between this variant's
    golden and the default one is exactly what `tls.port` moves and nothing
    else. The cert/key paths are the in-container mount points the TLS seam
    tests use; they are rendered as literal strings and no file is read.
    """
    config = copy.deepcopy(EXAMPLE_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "port": _TLS_CUSTOM_PORT,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }
    return config


def test_golden_tls_custom_port_fixture_exists() -> None:
    """The variant baseline is present before the comparison below runs — a
    missing file must fail loudly here rather than be misread as a byte-match
    against an empty string."""
    assert (_GOLDEN_DIR / _TLS_CUSTOM_PORT_GOLDEN).is_file(), (
        f"missing golden fixture: {_TLS_CUSTOM_PORT_GOLDEN}"
    )


def test_render_matches_golden_tls_custom_port_nginx_conf_byte_for_byte() -> None:
    """`nginx/nginx.conf` for the non-default-TLS-port facility is byte-identical
    to its committed variant baseline."""
    artifacts = render_web_terminals(_tls_custom_port_config())
    assert artifacts["nginx/nginx.conf"] == _read_golden(_TLS_CUSTOM_PORT_GOLDEN)


def test_golden_tls_custom_port_binds_and_redirects_to_the_custom_port() -> None:
    """The committed variant itself must still carry the three places the custom
    port lands — both `listen ... ssl` lines and the cleartext server's 301
    target. Guards against a truncated or stale variant passing the byte-equality
    check above by matching an equally wrong render."""
    conf = _read_golden(_TLS_CUSTOM_PORT_GOLDEN)

    assert f"listen {_TLS_CUSTOM_PORT} ssl;" in conf
    assert f"listen [::]:{_TLS_CUSTOM_PORT} ssl;" in conf
    assert f"return 301 https://$host:{_TLS_CUSTOM_PORT}$request_uri;" in conf


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
