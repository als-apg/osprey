"""The identity headers the nginx perimeter forwards — and refuses to relay.

The auth sidecar answers an authorized ``auth_request`` with two headers naming
who the request belongs to (:data:`~osprey.services.auth_sidecar.identity_headers.SUBJECT_HEADER`)
and what privilege they hold (:data:`~osprey.services.auth_sidecar.identity_headers.ROLE_HEADER`).
nginx is what carries those two values from the subrequest's response into the
proxied request, and it is also the only thing standing between a client that
simply *types* them and a terminal that would believe them. Both halves are
rendered here, so both are pinned here.

Three nginx facts shape every assertion below, and getting any of them wrong is
silent rather than loud:

1. **A location answers a header name from its own ``proxy_set_header`` table,
   or from the client.** nginx builds that table per location and copies the
   request's remaining headers around it, so naming a header — as a clear, or
   as a forward whose value happens to be empty — is what drops the client's
   own. A location naming neither identity header is a hole; which of the two
   directives claims the name is a separate question from whether it is
   claimed.
2. **A ``proxy_set_header`` at location level replaces the entire inherited
   set.** A single server-level clear would therefore be discarded by every
   location that sets any header of its own — which is all of them. The claim
   must be written into each proxying location, and there must be none at
   server level pretending to cover them.
3. **An auth subrequest inherits the parent request's headers.** So
   ``/_osprey_auth/<user>`` needs its own clear too: without one, a client's
   forged subject reaches the sidecar as though the perimeter had put it there.

The structural test (:func:`test_every_proxying_location_claims_both_identity_headers_with_auth_on`)
is the one that has to keep biting: it finds every location with a
``proxy_pass`` and demands of each that it write both names ITSELF — the gated
forward, or the clear, never nothing and never both — so a proxying location
added later cannot quietly become a hole. The auth-off case is not an exception
— a deployment with no login flow still must not let a client name itself.

"Never both" is not pedantry: a name entered twice in one location's table
makes real nginx log ``could not build optimal proxy_headers_hash`` on every
start and every reload, from the security perimeter itself.
"""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from osprey.deployment.web_terminals.render import (
    NGINX_TEMPLATES_OUTPUT_DIR,
    render_web_terminals,
    terminal_secret_env_var,
)
from osprey.services.auth_sidecar.identity_headers import ROLE_HEADER, SUBJECT_HEADER

_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}


def _config(users: list) -> dict:
    """Minimal-but-complete facility config that exercises render_web_terminals()."""
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
                "web_base_port": _BASE_PORTS["web"],
                "artifact_base_port": _BASE_PORTS["artifact"],
                "ariel_base_port": _BASE_PORTS["ariel"],
                "lattice_base_port": _BASE_PORTS["lattice"],
                "users": users,
            }
        },
    }


def _auth_config(users: list) -> dict:
    """`_config` with password auth on over cleartext.

    Without TLS the render refuses `auth.method` unless the facility accepts the
    risk explicitly; that gate is a different seam's business, so it is opted
    out of here to get at the auth surface itself.
    """
    config = copy.deepcopy(_config(users))
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }
    return config


def _render_nginx(config: dict) -> str:
    return render_web_terminals(config)["nginx/nginx.conf"]


def _directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    This template explains the very constructs these tests count and rule out,
    at length, in prose directly above them — so an absence assertion made
    against the raw text would be answered by a comment rather than by a
    directive.
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


#: One rendered `location <header> { … }` block: the header line, then everything
#: up to the closing brace at the location's own indent. Anchored on the four-space
#: indent every location in this template is emitted at, so a match can only ever
#: be one location's body and never a run that swallows its siblings.
_LOCATION_RE = re.compile(r"^    location ([^\n{]*)\{\n(.*?)\n    \}$", re.DOTALL | re.MULTILINE)


def _locations(conf: str) -> dict[str, str]:
    """Every location block in the rendered fragment, keyed by its header text.

    Parsed from the COMMENT-STRIPPED text, because this template explains the
    very directives these tests look for — verbatim, at length — in prose
    directly above them. Read raw, a body would let a commented-out directive
    answer an assertion about a real one, and a comment mentioning
    ``proxy_pass`` would enrol a non-proxying location into the structural
    guard's set.

    A repeated location header is a failure rather than a silent overwrite:
    two bodies collapsing onto one key would drop a location out of the
    guard's coverage, which is a false green on the very property the guard
    exists for.
    """
    pairs = [
        (match.group(1).strip(), match.group(2))
        for match in _LOCATION_RE.finditer(_directives(conf))
    ]
    assert pairs, "no location blocks in the rendered fragment"
    found = dict(pairs)
    assert len(found) == len(pairs), (
        f"duplicate location header(s) in the rendered fragment: {sorted(h for h, _ in pairs)}"
    )
    return found


def _location_body(conf: str, header: str) -> str:
    locations = _locations(conf)
    assert header in locations, (
        f"no {header!r} location in the rendered fragment: {sorted(locations)}"
    )
    return locations[header]


def _server_level(conf: str) -> str:
    """The rendered fragment with every location body removed.

    What is left is what a `server`-level directive would live in — the scope a
    `proxy_set_header` must NOT be written at, because a location that sets any
    header of its own inherits none of them.
    """
    return _LOCATION_RE.sub("", conf)


def _clear(header: str) -> str:
    return f'proxy_set_header {header} "";'


#: nginx's own spelling of the auth response's header, as a variable: lowercased,
#: dashes to underscores. Derived from the sidecar's constant rather than typed
#: out, so a renamed header cannot leave the template reading a variable that is
#: forever empty — which would fail OPEN, forwarding no identity at all.
def _upstream_var(header: str) -> str:
    return "$upstream_http_" + header.lower().replace("-", "_")


_SUBJECT_VAR = "$osprey_auth_subject"
_ROLE_VAR = "$osprey_auth_role"

#: The two identity headers, each paired with the variable a gated location
#: forwards it from.
_IDENTITY = ((SUBJECT_HEADER, _SUBJECT_VAR), (ROLE_HEADER, _ROLE_VAR))


def _forward(header: str, variable: str) -> str:
    return f"proxy_set_header {header} {variable};"


def _identity_writes(body: str) -> dict[str, list[str]]:
    """Every directive in *body* that claims an identity header name, per name.

    A list rather than a flag, so a name written TWICE is visible instead of
    collapsing into "yes, it is written". Both entries are what nginx would
    hash, and the second one is what makes it warn.
    """
    return {
        header: ["clear"] * body.count(_clear(header))
        + ["forward"] * body.count(_forward(header, variable))
        for header, variable in _IDENTITY
    }


def _assert_claims_both_identity_headers(location: str, body: str) -> None:
    """*location* writes both identity names itself, the same way, exactly once.

    The safety property is that the name is claimed at all — nginx answers a
    named header from the location's own table and never from the request, so
    a clear and a forward are equally good at dropping a forged value, and
    neither is optional. What is ruled out here is a location that writes one
    name and not the other, one that writes neither (the client's own header
    reaches the container), and one that writes a name twice (no extra safety,
    a `proxy_headers_hash` warning on every reload).
    """
    writes = _identity_writes(body)
    assert writes[SUBJECT_HEADER] == writes[ROLE_HEADER], (
        f"{location} claims the two identity headers differently: {writes}"
    )
    assert len(writes[SUBJECT_HEADER]) == 1, (
        f"{location} must claim each identity header exactly once: {writes}"
    )
    if writes[SUBJECT_HEADER] == ["forward"]:
        assert "auth_request /" in body, (
            f"{location} forwards an identity, but holds no auth_request that could establish one"
        )


_ONE_USER = ["alice"]
_TWO_USERS = ["alice", "bob"]
#: A roster whose second entry opts out of the login flow (`login: false`): it is
#: proxied with no gate at all, which makes it the sharpest test of the clears —
#: there is no `auth_request` here whose answer could overwrite a forged header.
_EXEMPT_ROSTER = [
    {"name": "alice", "index": 0},
    {"name": "ariel", "index": 1, "login": False},
]


# --------------------------------------------------------------------------
# The spelling itself
# --------------------------------------------------------------------------


def test_rendered_header_names_are_the_sidecar_s_own_spelling() -> None:
    """The template writes the header names as literals — nginx conf has no way
    to import a Python constant — so the two spellings are held together by this
    assertion instead. A rename on the sidecar side that stopped here would
    leave nginx forwarding a header nothing reads and clearing one nothing sets.
    """
    # Arrange / Act
    nginx_conf = _render_nginx(_auth_config(_ONE_USER))

    # Assert
    assert SUBJECT_HEADER == "X-Osprey-Auth-Subject"
    assert ROLE_HEADER == "X-Osprey-Auth-Role"
    assert _clear(SUBJECT_HEADER) in nginx_conf
    assert _clear(ROLE_HEADER) in nginx_conf


# --------------------------------------------------------------------------
# The forward — gated branch only
# --------------------------------------------------------------------------


def test_gated_user_location_forwards_both_identity_headers_from_the_auth_answer() -> None:
    """A gated `/u/<user>/` reads both values off its own `auth_request`'s
    response and sets them on the proxied request. `auth_request_set` is the
    only construct that can do this: the subrequest's response headers are
    otherwise not visible to the parent request at all."""
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_TWO_USERS)), "/u/alice/")

    # Assert — the values are lifted out of alice's own subrequest…
    assert f"auth_request_set {_SUBJECT_VAR} {_upstream_var(SUBJECT_HEADER)};" in body
    assert f"auth_request_set {_ROLE_VAR} {_upstream_var(ROLE_HEADER)};" in body

    # Assert — …and put on the request that reaches alice's container.
    assert f"proxy_set_header {SUBJECT_HEADER} {_SUBJECT_VAR};" in body
    assert f"proxy_set_header {ROLE_HEADER} {_ROLE_VAR};" in body


def test_the_forward_is_emitted_once_per_gated_user_and_nowhere_else() -> None:
    """Two gated users, two forwards each — and nothing on the exempt entry, the
    internal verify target, `/auth/`, or the landing page. The count is what
    makes this bite: a forward hoisted somewhere shared would still satisfy a
    plain substring check while handing one user's identity to another."""
    # Arrange / Act
    directives = _directives(_render_nginx(_auth_config(_TWO_USERS)))

    # Assert
    assert directives.count(f"proxy_set_header {SUBJECT_HEADER} {_SUBJECT_VAR};") == 2
    assert directives.count(f"proxy_set_header {ROLE_HEADER} {_ROLE_VAR};") == 2
    assert directives.count(f"auth_request_set {_SUBJECT_VAR} ") == 2
    assert directives.count(f"auth_request_set {_ROLE_VAR} ") == 2


def test_login_exempt_location_forwards_no_identity_at_all() -> None:
    """`login: false` means no gate, and no gate means no authorized identity to
    forward. Emitting the header here from an unset variable would announce an
    empty subject as though the perimeter had checked one."""
    # Arrange / Act
    nginx_conf = _render_nginx(_auth_config(_EXEMPT_ROSTER))
    body = _directives(_location_body(nginx_conf, "/u/ariel/"))

    # Assert
    assert "auth_request_set" not in body
    assert _SUBJECT_VAR not in body
    assert _ROLE_VAR not in body


def test_auth_off_render_forwards_no_identity_and_has_no_auth_request_set() -> None:
    """With no login flow there is no subrequest to read an identity from, so
    the whole forwarding half is absent — while the clears below are not."""
    # Arrange / Act
    directives = _directives(_render_nginx(_config(_TWO_USERS)))

    # Assert
    assert "auth_request_set" not in directives
    assert _SUBJECT_VAR not in directives
    assert _ROLE_VAR not in directives


# --------------------------------------------------------------------------
# The claim — every proxying location, every topology
# --------------------------------------------------------------------------


def test_every_proxying_location_claims_both_identity_headers_with_auth_on() -> None:
    """The structural guard, auth on: every location that opens an upstream
    connection writes both identity names itself — the gated one by forwarding
    what the sidecar answered, the rest by clearing. Written against
    `proxy_pass` rather than against a list of known paths so a location added
    later is covered the day it is added, not the day someone remembers this
    file."""
    # Arrange / Act
    nginx_conf = _render_nginx(_auth_config(_EXEMPT_ROSTER))

    # Assert — the set is non-trivial: gated user, exempt user, both verify
    # targets' worth of surface, and the public login prefix.
    proxying = {
        header: body for header, body in _locations(nginx_conf).items() if "proxy_pass " in body
    }
    assert set(proxying) == {"/u/alice/", "/u/ariel/", "= /_osprey_auth/alice", "/auth/"}

    # Assert — and not one of them leaves either name for the client to supply.
    for location, body in proxying.items():
        _assert_claims_both_identity_headers(location, body)

    # Assert — exactly one of them is entitled to forward, and it is the gated
    # user's own location. Everything else clears.
    forwarding = {
        location
        for location, body in proxying.items()
        if _identity_writes(body)[SUBJECT_HEADER] == ["forward"]
    }
    assert forwarding == {"/u/alice/"}


def test_every_proxying_location_claims_both_identity_headers_with_auth_off() -> None:
    """The same guard with the login flow off — the deliberate part. An
    auth-off deployment has no sidecar to contradict a forged header, so a
    terminal that read one would be taking the client's word for who it is.
    With no subrequest anywhere in the render there is nothing to forward, so
    every claim here has to be a clear."""
    # Arrange / Act
    nginx_conf = _render_nginx(_config(_TWO_USERS))

    # Assert
    proxying = {
        header: body for header, body in _locations(nginx_conf).items() if "proxy_pass " in body
    }
    assert set(proxying) == {"/u/alice/", "/u/bob/"}
    for location, body in proxying.items():
        _assert_claims_both_identity_headers(location, body)
        assert _identity_writes(body)[SUBJECT_HEADER] == ["clear"], (
            f"{location} forwards an identity in a render that authorizes nobody"
        )


def test_the_internal_verify_target_clears_both_identity_headers() -> None:
    """Called out on its own because the reason is not the same as everywhere
    else: an auth subrequest INHERITS the parent request's headers, so without
    a clear here a client's forged subject arrives at the sidecar looking like
    something the perimeter established."""
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_ONE_USER)), "= /_osprey_auth/alice")

    # Assert
    assert _clear(SUBJECT_HEADER) in body
    assert _clear(ROLE_HEADER) in body


def test_the_public_auth_prefix_clears_both_identity_headers() -> None:
    """`/auth/` is the one location deliberately outside `auth_request` — it is
    where a session comes from. That makes it reachable by anyone, so it is the
    one an unauthenticated client would forge into."""
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_ONE_USER)), "/auth/")

    # Assert
    assert _clear(SUBJECT_HEADER) in body
    assert _clear(ROLE_HEADER) in body


def test_the_gated_location_forwards_instead_of_also_clearing() -> None:
    """The forward IS the claim, so the gated location does not clear as well.

    A location's `proxy_set_header` table is what answers a named header, and
    the client's own value is never consulted for a name in it — including when
    the directive evaluates empty, which sends nothing rather than falling back
    to what arrived. So a clear written beside the forward drops no additional
    forgery; it only enters the same name in the table twice, which is what
    makes nginx log `could not build optimal proxy_headers_hash` on every start
    and reload of every auth-on deployment.
    """
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_ONE_USER)), "/u/alice/")

    # Assert — claimed, by the forward…
    assert _forward(SUBJECT_HEADER, _SUBJECT_VAR) in body
    assert _forward(ROLE_HEADER, _ROLE_VAR) in body

    # Assert — …and by nothing else.
    assert _clear(SUBJECT_HEADER) not in body
    assert _clear(ROLE_HEADER) not in body


def test_no_location_writes_the_same_proxy_set_header_name_twice() -> None:
    """Across every topology, and for EVERY header name — not just the two
    identity ones.

    nginx hashes each location's `proxy_set_header` entries by name, and a
    duplicate name is what pushes that hash past its default bucket and makes
    the daemon warn on start and on reload. The warning is harmless in itself
    (nginx retries with a bigger bucket) and permanent, which is the problem: a
    recurring warning from the security perimeter is the one an operator learns
    to scroll past, right up until it is a real one.
    """
    # Arrange
    configs = (
        _config(_TWO_USERS),
        _auth_config(_ONE_USER),
        _auth_config(_TWO_USERS),
        _auth_config(_EXEMPT_ROSTER),
    )
    name_of = re.compile(r"^\s*proxy_set_header\s+(\S+)", re.MULTILINE)

    for config in configs:
        for location, body in _locations(_render_nginx(config)).items():
            # Act
            names = name_of.findall(body)

            # Assert
            duplicated = {name for name in names if names.count(name) > 1}
            assert not duplicated, f"{location} sets {sorted(duplicated)} more than once"


# --------------------------------------------------------------------------
# Never at server level
# --------------------------------------------------------------------------


def test_no_identity_header_directive_is_written_at_server_level() -> None:
    """A `proxy_set_header` at server level is not a safety net for these
    locations — it is a trap. Location-level `proxy_set_header` replaces the
    whole inherited set rather than adding to it, so every location here (all of
    them set `Host` at minimum) would discard a server-level clear entirely,
    while the config would read as though the perimeter were covered."""
    # Arrange / Act
    for config in (_config(_TWO_USERS), _auth_config(_EXEMPT_ROSTER)):
        outside_locations = _directives(_server_level(_render_nginx(config)))

        # Assert
        assert "proxy_set_header X-Osprey-Auth-" not in outside_locations
        assert "auth_request_set" not in outside_locations


# --------------------------------------------------------------------------
# What real nginx says about it
# --------------------------------------------------------------------------

#: The base image the deployed stack runs, matching `test_nginx_validate.py`.
_NGINX_IMAGE = "nginx:1.27-alpine"
#: Where the compose overlay's entrypoint writes the per-user secret snippets
#: each gated location `include`s. `nginx -t` reads the include, so the file has
#: to exist — but nothing here cares about the envsubst chain that normally
#: produces it (`test_nginx_validate.py` owns that), so a substituted copy is
#: mounted directly.
_SECRET_INCLUDE_DIR = "/etc/nginx/osprey"


def _nginx_t(conf: str, secret_snippets: dict[str, str]) -> subprocess.CompletedProcess:
    """Run `nginx -t` on *conf* inside the real base image."""
    with tempfile.TemporaryDirectory() as tmp:
        conf_dir = Path(tmp) / "conf"
        include_dir = Path(tmp) / "osprey"
        conf_dir.mkdir()
        include_dir.mkdir()
        (conf_dir / "default.conf").write_text(conf)
        for name, snippet in secret_snippets.items():
            (include_dir / name).write_text(snippet)
        return subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{conf_dir}/default.conf:/etc/nginx/conf.d/default.conf:ro",
                "-v",
                f"{include_dir}:{_SECRET_INCLUDE_DIR}:ro",
                _NGINX_IMAGE,
                "nginx",
                "-t",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.dockerbuild
@pytest.mark.skipif(not _docker_available(), reason="docker not available")
def test_real_nginx_reads_the_auth_on_render_without_a_single_warning() -> None:
    """The auth-on render starts CLEAN — not merely "exit 0".

    `test_nginx_validate.py` asserts the return code, which a warning does not
    change: nginx logs `could not build optimal proxy_headers_hash`, recovers,
    and exits 0. That is exactly the failure this test exists to see, because
    the noise would be emitted on every start and every reload of every
    authenticated deployment, by the process enforcing the perimeter.
    """
    # Arrange — a topology with all three claim shapes in it at once: a gated
    # user forwarding, an exempt user clearing, and the shared `/auth/` prefix.
    # Every roster user needs a secret for the render to succeed, even the
    # exempt one whose location never includes it.
    secrets = {"alice": "terminal-secret-for-alice", "ariel": "terminal-secret-for-ariel"}
    artifacts = render_web_terminals(_auth_config(_EXEMPT_ROSTER), terminal_secrets=secrets)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Arrange — guard against a vacuous green: a render that had stopped
    # emitting the forwarding half entirely would also warn about nothing.
    assert _forward(SUBJECT_HEADER, _SUBJECT_VAR) in nginx_conf
    assert _forward(ROLE_HEADER, _ROLE_VAR) in nginx_conf
    assert _clear(SUBJECT_HEADER) in nginx_conf

    # Only the gated users get a snippet — an exempt location includes none —
    # so the set is taken from the render rather than from the roster.
    prefix = f"{NGINX_TEMPLATES_OUTPUT_DIR}/secret-"
    snippets = {
        path[len(f"{NGINX_TEMPLATES_OUTPUT_DIR}/") :].removesuffix(".template"): content.replace(
            f"${{{terminal_secret_env_var(path[len(prefix) : -len('.conf.template')])}}}",
            "substituted-at-container-start",
        )
        for path, content in artifacts.items()
        if path.startswith(prefix)
    }
    assert snippets, "the gated location's include has no rendered snippet behind it"

    # Act
    result = _nginx_t(nginx_conf, snippets)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "[warn]" not in result.stderr, result.stderr
