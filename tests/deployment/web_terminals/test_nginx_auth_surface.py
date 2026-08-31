"""The identity headers the nginx perimeter forwards — and refuses to relay.

The auth sidecar answers an authorized ``auth_request`` with four headers naming
which roster card the request is on
(:data:`~osprey.services.auth_sidecar.identity_headers.ACCOUNT_HEADER`),
who proved the login (:data:`~osprey.services.auth_sidecar.identity_headers.SUBJECT_HEADER`),
what privilege they hold (:data:`~osprey.services.auth_sidecar.identity_headers.ROLE_HEADER`)
and where that privilege came from
(:data:`~osprey.services.auth_sidecar.identity_headers.ROLE_SOURCE_HEADER`).
nginx is what carries those values from the subrequest's response into the
proxied request, and it is also the only thing standing between a client that
simply *types* them and a terminal that would believe them. Both halves are
rendered here, so both are pinned here.

Three nginx facts shape every assertion below, and getting any of them wrong is
silent rather than loud:

1. **A location answers a header name from its own ``proxy_set_header`` table,
   or from the client.** nginx builds that table per location and copies the
   request's remaining headers around it, so naming a header — as a clear, or
   as a forward whose value happens to be empty — is what drops the client's
   own. A location naming none of the identity headers is a hole; which of the
   two directives claims a name is a separate question from whether it is
   claimed.
2. **A ``proxy_set_header`` at location level replaces the entire inherited
   set.** A single server-level clear would therefore be discarded by every
   location that sets any header of its own — which is all of them. The claim
   must be written into each proxying location, and there must be none at
   server level pretending to cover them.
3. **An auth subrequest inherits the parent request's headers.** So
   ``/_osprey_auth/<user>`` needs its own clear too: without one, a client's
   forged subject reaches the sidecar as though the perimeter had put it there.

The structural test (:func:`test_every_proxying_location_claims_every_identity_header_with_auth_on`)
is the one that has to keep biting: it finds every location with a
``proxy_pass`` and demands of each that it write every name ITSELF — the gated
forward, or the clear, never nothing and never both — so a proxying location
added later cannot quietly become a hole. The two wall-less postures are not
exceptions — ``token`` (the default: each terminal's own ``?token=`` exchange
is the gate) and ``none`` (open: reaching this nginx IS the authorization)
render no sidecar whose answer could contradict a forged header, so a client
that named itself there would simply be believed.

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
    TERMINAL_SECRET_HEADER,
    render_web_terminals,
    terminal_secret_env_var,
)
from osprey.port_layout import default_port
from osprey.services.auth_sidecar.identity_headers import (
    ACCOUNT_HEADER,
    ROLE_HEADER,
    ROLE_SOURCE_HEADER,
    SUBJECT_HEADER,
)

#: The per-user family bases these renders run on. Nothing in the configs below
#: moves them, so they are the layout's own — derived here so a cookie name or a
#: proxy_pass target asserted further down follows a slot that moves.
_BASE_PORTS = {slot: default_port(slot) for slot in ("web", "artifact", "ariel", "lattice")}


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


def _open_config(users: list) -> dict:
    """`_config` with the open posture on (`auth.method: none`).

    The one render shape where a non-exempt location carries the operator
    secret with NO subrequest in front of it: reaching this nginx is the
    authorization, so the perimeter vouches for the request itself. No
    `allow_insecure_http` is needed — the cleartext refusal guards the sidecar
    methods' session cookies, and this posture mints no session.
    """
    config = copy.deepcopy(_config(users))
    config["modules"]["web_terminals"]["auth"] = {"method": "none"}
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


_ACCOUNT_VAR = "$osprey_auth_account"
_SUBJECT_VAR = "$osprey_auth_subject"
_ROLE_VAR = "$osprey_auth_role"
_ROLE_SOURCE_VAR = "$osprey_auth_role_source"

#: The four identity headers, each paired with the variable a gated location
#: forwards it from.
_IDENTITY = (
    (ACCOUNT_HEADER, _ACCOUNT_VAR),
    (SUBJECT_HEADER, _SUBJECT_VAR),
    (ROLE_HEADER, _ROLE_VAR),
    (ROLE_SOURCE_HEADER, _ROLE_SOURCE_VAR),
)


def _forward(header: str, variable: str) -> str:
    return f"proxy_set_header {header} {variable};"


def _secret_include(user: str) -> str:
    """The per-user snippet an injecting location pulls the secret header in from.

    The header is SET by that file rather than by a directive in this fragment,
    which is why the open render's assertions look for the include and for the
    ABSENCE of a clear beside it.
    """
    return f"include /etc/nginx/osprey/secret-{user}.conf;"


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


def _assert_claims_every_identity_header(location: str, body: str) -> None:
    """*location* writes every identity name itself, the same way, exactly once.

    The safety property is that the name is claimed at all — nginx answers a
    named header from the location's own table and never from the request, so
    a clear and a forward are equally good at dropping a forged value, and not
    one of them is optional. What is ruled out here is a location that writes
    some of the names and not the rest, one that writes none of them (the
    client's own header reaches the container), and one that writes a name
    twice (no extra safety, a `proxy_headers_hash` warning on every reload).

    Written over whatever `_IDENTITY` holds rather than name by name, so a
    header added to that tuple is *checked* the day it is added rather than
    merely computed: a name nothing asserts on is no guard at all.
    """
    writes = _identity_writes(body)
    claimed = {tuple(claims) for claims in writes.values()}
    assert len(claimed) == 1, f"{location} claims the identity headers differently: {writes}"
    (claim,) = claimed
    assert len(claim) == 1, f"{location} must claim each identity header exactly once: {writes}"
    if claim == ("forward",):
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
    to import a Python constant — so the four spellings are held together by
    this assertion instead. A rename on the sidecar side that stopped here would
    leave nginx forwarding a header nothing reads and clearing one nothing sets.
    """
    # Arrange / Act
    nginx_conf = _render_nginx(_auth_config(_ONE_USER))

    # Assert
    assert ACCOUNT_HEADER == "X-Osprey-Auth-Account"
    assert SUBJECT_HEADER == "X-Osprey-Auth-Subject"
    assert ROLE_HEADER == "X-Osprey-Auth-Role"
    assert ROLE_SOURCE_HEADER == "X-Osprey-Auth-Role-Source"
    assert _clear(ACCOUNT_HEADER) in nginx_conf
    assert _clear(SUBJECT_HEADER) in nginx_conf
    assert _clear(ROLE_HEADER) in nginx_conf
    assert _clear(ROLE_SOURCE_HEADER) in nginx_conf


# --------------------------------------------------------------------------
# The forward — gated branch only
# --------------------------------------------------------------------------


def test_gated_user_location_forwards_every_identity_header_from_the_auth_answer() -> None:
    """A gated `/u/<user>/` reads every value off its own `auth_request`'s
    response and sets them on the proxied request. `auth_request_set` is the
    only construct that can do this: the subrequest's response headers are
    otherwise not visible to the parent request at all.

    Written over `_IDENTITY` rather than name by name: a capture that named the
    wrong upstream variable would leave that header forwarding empty, and for
    the account that fails OPEN — the container falls back to comparing the
    subject and marks the request a mismatch."""
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_TWO_USERS)), "/u/alice/")

    # Assert
    for header, variable in _IDENTITY:
        # …the value is lifted out of alice's own subrequest…
        assert f"auth_request_set {variable} {_upstream_var(header)};" in body, (
            f"/u/alice/ does not capture {header} off its own auth_request"
        )
        # …and put on the request that reaches alice's container.
        assert _forward(header, variable) in body, (
            f"/u/alice/ captures {header} but does not forward it"
        )


def test_the_forward_is_emitted_once_per_gated_user_and_nowhere_else() -> None:
    """Two gated users, one forward per identity header each — and nothing on
    the exempt entry, the internal verify target, `/auth/`, or the landing page.
    The count is what makes this bite: a forward hoisted somewhere shared would
    still satisfy a plain substring check while handing one user's identity to
    another."""
    # Arrange / Act
    directives = _directives(_render_nginx(_auth_config(_TWO_USERS)))

    # Assert
    for header, variable in _IDENTITY:
        assert directives.count(_forward(header, variable)) == 2, (
            f"{header} is not forwarded exactly once per gated user"
        )
        assert directives.count(f"auth_request_set {variable} ") == 2, (
            f"{header} is not captured exactly once per gated user"
        )


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
    assert _ROLE_SOURCE_VAR not in body


@pytest.mark.parametrize(
    "config", [_config(_TWO_USERS), _open_config(_TWO_USERS)], ids=["token", "open"]
)
def test_a_wall_less_render_forwards_no_identity_and_has_no_auth_request_set(
    config: dict,
) -> None:
    """With no login flow there is no subrequest to read an identity from, so
    the whole forwarding half is absent — while the clears below are not.

    True of `none` as much as of `token`: the open posture injects an operator
    secret, which is a credential the perimeter vouches for, but it establishes
    no identity and so has none to name.
    """
    # Arrange / Act
    directives = _directives(_render_nginx(config))

    # Assert
    assert "auth_request_set" not in directives
    assert _SUBJECT_VAR not in directives
    assert _ROLE_VAR not in directives
    assert _ROLE_SOURCE_VAR not in directives


# --------------------------------------------------------------------------
# The claim — every proxying location, every topology
# --------------------------------------------------------------------------


def test_every_proxying_location_claims_every_identity_header_with_auth_on() -> None:
    """The structural guard, auth on: every location that opens an upstream
    connection writes every identity name itself — the gated one by forwarding
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

    # Assert — and not one of them leaves any of the names for the client to
    # supply.
    for location, body in proxying.items():
        _assert_claims_every_identity_header(location, body)

    # Assert — exactly one of them is entitled to forward, and it is the gated
    # user's own location. Everything else clears.
    forwarding = {
        location
        for location, body in proxying.items()
        if _identity_writes(body)[SUBJECT_HEADER] == ["forward"]
    }
    assert forwarding == {"/u/alice/"}


@pytest.mark.parametrize(
    "config,proxying_locations",
    [
        (_config(_TWO_USERS), {"/u/alice/", "/u/bob/"}),
        (_config(_EXEMPT_ROSTER), {"/u/alice/", "/u/ariel/"}),
        (_open_config(_TWO_USERS), {"/u/alice/", "/u/bob/"}),
        (_open_config(_EXEMPT_ROSTER), {"/u/alice/", "/u/ariel/"}),
    ],
    ids=["token", "token-exempt", "open", "open-exempt"],
)
def test_every_proxying_location_claims_every_identity_header_without_a_login_wall(
    config: dict, proxying_locations: set[str]
) -> None:
    """The same guard with no login wall — the deliberate part, and the reason
    both wall-less postures are run through it rather than just the default one.

    Neither `token` nor `none` renders a sidecar, so neither has an answer that
    could contradict a forged header: a terminal reading one would be taking
    the client's word for who it is. With no subrequest anywhere in either
    render there is nothing to forward, so every claim here has to be a clear —
    including in the `none` render, whose non-exempt locations DO carry an
    injected credential and could otherwise look authorized enough to forward
    an identity nobody checked.

    The exempt roster is run through both because `login: false` is the entry
    that changes shape under `none` (it is the one opted out of the injection),
    and a claim must not be what the opt-out drops.
    """
    # Arrange / Act
    nginx_conf = _render_nginx(config)

    # Assert
    proxying = {
        header: body for header, body in _locations(nginx_conf).items() if "proxy_pass " in body
    }
    assert set(proxying) == proxying_locations
    for location, body in proxying.items():
        _assert_claims_every_identity_header(location, body)
        assert _identity_writes(body)[SUBJECT_HEADER] == ["clear"], (
            f"{location} forwards an identity in a render that authorizes nobody"
        )


def test_the_internal_verify_target_clears_every_identity_header() -> None:
    """Called out on its own because the reason is not the same as everywhere
    else: an auth subrequest INHERITS the parent request's headers, so without
    a clear here a client's forged subject arrives at the sidecar looking like
    something the perimeter established."""
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_ONE_USER)), "= /_osprey_auth/alice")

    # Assert
    assert _clear(SUBJECT_HEADER) in body
    assert _clear(ROLE_HEADER) in body
    assert _clear(ROLE_SOURCE_HEADER) in body


def test_the_public_auth_prefix_clears_every_identity_header() -> None:
    """`/auth/` is the one location deliberately outside `auth_request` — it is
    where a session comes from. That makes it reachable by anyone, so it is the
    one an unauthenticated client would forge into."""
    # Arrange / Act
    body = _location_body(_render_nginx(_auth_config(_ONE_USER)), "/auth/")

    # Assert
    assert _clear(SUBJECT_HEADER) in body
    assert _clear(ROLE_HEADER) in body
    assert _clear(ROLE_SOURCE_HEADER) in body


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

    # Assert
    for header, variable in _IDENTITY:
        # …claimed, by the forward…
        assert _forward(header, variable) in body, f"/u/alice/ does not forward {header}"
        # …and by nothing else.
        assert _clear(header) not in body, f"/u/alice/ clears {header} beside its forward"


def test_no_location_writes_the_same_proxy_set_header_name_twice() -> None:
    """Across every topology, and for EVERY header name — not just the three
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
        _config(_EXEMPT_ROSTER),
        _auth_config(_ONE_USER),
        _auth_config(_TWO_USERS),
        _auth_config(_EXEMPT_ROSTER),
        _open_config(_TWO_USERS),
        _open_config(_EXEMPT_ROSTER),
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
    for config in (
        _config(_TWO_USERS),
        _auth_config(_EXEMPT_ROSTER),
        _open_config(_EXEMPT_ROSTER),
    ):
        outside_locations = _directives(_server_level(_render_nginx(config)))

        # Assert
        assert "proxy_set_header X-Osprey-Auth-" not in outside_locations
        assert "auth_request_set" not in outside_locations


# --------------------------------------------------------------------------
# The open posture: injection with no wall in front of it
# --------------------------------------------------------------------------


def test_open_render_injects_each_non_exempt_users_own_secret_with_no_gate_in_front() -> None:
    """`none` is the one posture where a location carries the operator secret
    with nothing standing in front of it.

    Under `password`/`oidc` the `include` sits behind an `auth_request`, so a
    request that failed the gate never reaches the `proxy_pass`. Under `none`
    the include IS the whole mechanism — reaching this port is the
    authorization — so the three halves that make it safe are asserted
    together: the location injects that user's own snippet, it does so without
    pretending a subrequest authorized it, and it pairs that injection with the
    NARROWED cookie rather than with the sidecar methods' outright strip.

    That last pairing is this posture's own, and it is asserted here because
    nowhere else does: `test_render.py`'s shared-predicate test holds the
    sidecar methods, where the header and the blanket `Cookie ""` strip move
    together. They must not move together here. With no sidecar there is no
    origin-wide `osprey_auth_session` for a strip to protect, and cutting the
    jar would take the app's own session cookie with it — the cookie that
    carries this user's per-terminal state. What must still be cut is the rest
    of the jar: one jar serves this whole origin, so a bare `$http_cookie`
    forward would hand this container every OTHER terminal this browser has
    unlocked, inside a container that runs agent-generated code.
    """
    # Arrange / Act
    nginx_conf = _render_nginx(_open_config(_TWO_USERS))

    # Assert — each non-exempt location pulls in its OWN user's snippet…
    for index, (user, other) in enumerate((("alice", "bob"), ("bob", "alice"))):
        body = _directives(_location_body(nginx_conf, f"/u/{user}/"))
        assert _secret_include(user) in body
        assert _secret_include(other) not in body

        # …holds no gate that could be mistaken for one…
        assert "auth_request" not in body
        assert "error_page 401" not in body

        # …and forwards exactly this user's own app session cookie: not the
        # strip (which would leave the app with no session at all), and not the
        # whole jar.
        cookie = f"osprey_terminal_session_{_BASE_PORTS['web'] + index}"
        assert f'proxy_set_header Cookie "{cookie}=$cookie_{cookie}";' in body
        assert 'proxy_set_header Cookie "";' not in body
        assert "$http_cookie" not in body


@pytest.mark.parametrize(
    "config", [_open_config(_ONE_USER), _open_config(_EXEMPT_ROSTER)], ids=["plain", "exempt"]
)
def test_open_injecting_location_leaves_the_secret_name_to_the_include_alone(
    config: dict,
) -> None:
    """No clear beside the include, and the absence is load-bearing.

    The included snippet SETS `X-Osprey-Terminal-Secret`, which already claims
    the name in this location's header table — nginx answers a named header
    from that table and never from the request, so an arriving value is
    overwritten exactly as a clear would overwrite it. A clear written as well
    would drop no additional forgery and would enter one name twice, which is
    what makes nginx log `could not build optimal proxy_headers_hash` on every
    start and reload.

    Invisible to the duplicate-name guard above, which reads this fragment
    only: the second directive lives in a file `nginx -T` resolves at start.

    Run on the exempt roster as well: that render is the one where the OTHER
    arm of the split branch is live next door, and an edit that hoisted the
    clear out of it would land it here.
    """
    # Arrange / Act
    body = _directives(_location_body(_render_nginx(config), "/u/alice/"))

    # Assert — claimed, by the include…
    assert _secret_include("alice") in body

    # Assert — …and by no directive in this fragment at all.
    assert TERMINAL_SECRET_HEADER not in body


def test_open_login_exempt_location_is_injected_nothing_and_clears_the_secret_header() -> None:
    """`login: false` opts an entry out of the open posture's injection exactly
    as it opts one out of a login wall — and the clear is what makes that safe.

    Nothing is vouched for here, so nothing sets the header; a value arriving
    in it therefore came from the client, and forwarding it would let anyone
    who can reach this port present the credential the app trusts absolutely.
    This is the one arm of the split branch that must write the clear.
    """
    # Arrange / Act
    nginx_conf = _render_nginx(_open_config(_EXEMPT_ROSTER))
    exempt = _directives(_location_body(nginx_conf, "/u/ariel/"))

    # Assert
    assert _secret_include("ariel") not in exempt
    assert _clear(TERMINAL_SECRET_HEADER) in exempt

    # Assert — and the roster's other entry is still injected, so the opt-out
    # is the entry's own and not the render's.
    assert _secret_include("alice") in _directives(_location_body(nginx_conf, "/u/alice/"))


def test_the_exempt_entry_is_served_the_same_directives_under_open_as_under_token() -> None:
    """A `login: false` entry gets the ungated treatment whatever the posture
    around it is.

    Compared as the directives alone, in the same order, rather than as text:
    the open render explains the opt-out in prose the token render has no
    reason to carry, so the two bodies differ by design in every way except the
    one that matters. What has to match is what nginx executes — same clears,
    same single forwarded cookie, no include on either side — and order is kept
    in the comparison because a location's header table is built in the order
    it is written.
    """
    # Arrange / Act
    served = {
        posture: [
            line.strip()
            for line in _directives(_location_body(_render_nginx(config), "/u/ariel/")).splitlines()
            if line.strip()
        ]
        for posture, config in (
            ("open", _open_config(_EXEMPT_ROSTER)),
            ("token", _config(_EXEMPT_ROSTER)),
        )
    }

    # Assert
    assert served["open"] == served["token"]
    # Guard against a vacuous green: an empty body would compare equal to an
    # empty body, and this location is one of the two the render exists for.
    assert "proxy_pass " in " ".join(served["open"])


def test_the_open_render_holds_no_login_surface_anywhere() -> None:
    """Open means open: no sidecar, so nothing that would talk to one.

    Each of these is a separate way for the wall to half-exist. A `/auth/`
    prefix with no sidecar behind it is a public 502; an `internal`
    `/_osprey_auth/<user>` target is the endpoint an `auth_request` would name;
    the named 401 handler is where a denied request would be sent. A render
    that emitted any of them would be describing a gate this deployment does
    not have.
    """
    # Arrange / Act
    directives = _directives(_render_nginx(_open_config(_EXEMPT_ROSTER)))

    # Assert
    assert "auth_request" not in directives
    assert "/_osprey_auth/" not in directives
    # The bare prefix, not `location /auth/`: a proxy_pass, a rewrite or a
    # return naming it would be just as public as a location would.
    assert "/auth/" not in directives
    assert "osprey_auth_401" not in directives
    # And no variable the sidecar's half of the template feeds — the maps and
    # the `auth_request_set` targets alike. One left behind would be forever
    # empty, which is how an identity forward fails OPEN.
    assert "$osprey_auth_" not in directives

    # Assert — and every location the render DOES hold is one of the roster's.
    assert set(_locations(directives)) == {
        "/u/alice/",
        "/u/ariel/",
        "= /",
        "= /u/alice",
        "= /u/ariel",
    }


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
    assert _forward(ROLE_SOURCE_HEADER, _ROLE_SOURCE_VAR) in nginx_conf
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
