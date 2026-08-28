"""The navigation-only perimeter stamp on every per-user web-terminal service.

Under ``auth.method: none`` the perimeter is nginx alone: it injects each
user's operator secret on every proxied location, so a request that reaches a
web port from the deploy host arrives already authenticated as whoever owns
that port — including a request made by agent-generated code inside one of
these containers, which share the host network namespace with each other and
with nginx.

The render's half of the answer is these two lines on each per-user service:
``OSPREY_WEB_PERIMETER=open`` (the posture) and
``OSPREY_WEB_PERIMETER_DENY_PORTS`` (nginx's published port, the TLS listener
when ``tls.enabled`` puts the real content server behind it, and every roster
user's terminal port). The MCP server process inside the container reads them
and hands the parsed list to the execution sandbox it spawns — which is what
makes the deny-list something the sandbox is TOLD rather than something it
derives, and why the list has to be computed here, where the whole roster is
known.

The three credentialed methods render neither line: a caller there still has to
present something the sandbox does not hold, so there is nothing to deny.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from osprey.deployment.web_terminals.render import render_web_terminals

_NGINX_PORT = 9080
_WEB_BASE_PORT = 9091

#: The marker and list names, spelled once. The executor reads these back from
#: the container's environment; a rename on either side is a stamp nobody reads.
_MARKER_VAR = "OSPREY_WEB_PERIMETER"
_DENY_PORTS_VAR = "OSPREY_WEB_PERIMETER_DENY_PORTS"


def _config(users: list[str], **auth: object) -> dict:
    """A minimal renderable config, optionally with an ``auth:`` stanza.

    ``allow_insecure_http`` is set whenever a sidecar method is asked for: the
    render refuses a login wall over cleartext, and that gate has its own
    coverage elsewhere.
    """
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": _NGINX_PORT,
        "web_base_port": _WEB_BASE_PORT,
        "artifact_base_port": 9291,
        "ariel_base_port": 9391,
        "lattice_base_port": 9491,
        "users": users,
    }
    if auth:
        stanza = dict(auth)
        if stanza.get("method") in ("password", "oidc"):
            stanza.setdefault("allow_insecure_http", True)
        web_terminals["auth"] = stanza
    return {
        "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


def _user_services(config: dict) -> dict[str, dict]:
    """The rendered per-user services, keyed by service name (``nginx`` dropped)."""
    compose = yaml.safe_load(render_web_terminals(copy.deepcopy(config))["docker-compose.web.yml"])
    return {name: svc for name, svc in compose["services"].items() if name != "nginx"}


def _env(service: dict) -> dict[str, str]:
    """A compose service's ``environment:`` list as a mapping."""
    return dict(entry.split("=", 1) for entry in service["environment"])


# ---------------------------------------------------------------------------
# The open posture stamps every per-user service
# ---------------------------------------------------------------------------


def test_open_posture_marks_every_per_user_service() -> None:
    """Every per-user container is told the perimeter is open — not just the first.

    Per service rather than deployment-wide because that is the only place a
    container reads: each one is a separate process tree with its own
    environment, and a marker on one of three terminals leaves the other two
    running an unguarded sandbox.
    """
    # Arrange
    users = ["alice", "bob", "carol"]
    config = _config(users, method="none")

    # Act
    services = _user_services(config)

    # Assert
    assert sorted(services) == [f"web-{user}" for user in users]
    for name, service in services.items():
        assert _env(service)[_MARKER_VAR] == "open", name


def test_open_posture_denies_nginx_and_every_roster_web_port() -> None:
    """The list is exactly the front door plus every user's terminal port.

    Every user's port, not just this container's own: the containers share the
    host network namespace, so alice's sandbox can reach bob's terminal on
    loopback, and nginx's injected secret makes that request bob's.
    """
    # Arrange
    users = ["alice", "bob", "carol"]
    config = _config(users, method="none")
    expected = f"{_NGINX_PORT},{_WEB_BASE_PORT},{_WEB_BASE_PORT + 1},{_WEB_BASE_PORT + 2}"

    # Act
    services = _user_services(config)

    # Assert
    for name, service in services.items():
        assert _env(service)[_DENY_PORTS_VAR] == expected, name


def test_deny_list_excludes_companion_panel_ports() -> None:
    """Companion families (artifact, ariel, lattice) are NOT denied.

    They are not what the injected secret opens; the in-container panel proxy
    addresses them legitimately, and none of them fronts a terminal, an agent
    session, or an approval prompt — which is what this deny-list guards.
    Denying them would break every panel tab to guard a surface nginx does not
    front.
    """
    # Arrange
    config = _config(["alice"], method="none")

    # Act
    denied = _env(_user_services(config)["web-alice"])[_DENY_PORTS_VAR]

    # Assert
    assert denied == f"{_NGINX_PORT},{_WEB_BASE_PORT}"
    for companion_base in (9291, 9391, 9491):
        assert str(companion_base) not in denied.split(",")


def test_deny_list_is_sorted_and_deduplicated() -> None:
    """Ascending, no repeats — the rendered artifact is diffed between deploys.

    A list whose order followed roster order (or dict iteration) would churn
    this file whenever a user was added in the middle, hiding real changes in
    the noise.

    The terminals are deliberately put BELOW nginx here (web_base_port 9000
    against nginx_port 9080), and the exact string is asserted rather than
    "is it sorted": with the default layout the front door happens to come
    first anyway, so a build that simply prepended nginx and appended the
    roster would pass a sortedness check while being sorted by luck.
    """
    # Arrange
    config = _config(["zoe", "alice", "mike"], method="none")
    config["modules"]["web_terminals"]["web_base_port"] = 9000

    # Act
    denied = _env(_user_services(config)["web-zoe"])[_DENY_PORTS_VAR]

    # Assert
    assert denied == f"9000,9001,9002,{_NGINX_PORT}"


def test_tls_deployment_denies_the_tls_listener() -> None:
    """With TLS on, 443 is in the list — it is the port that actually serves.

    nginx.conf.j2 renders `listen 443 ssl` as the SOLE content server under
    `tls.enabled` and demotes the plain `nginx_port` to a redirect. A deny-list
    that named only `nginx_port` would name the door that redirects and leave
    the one that injects the operator secret and serves the terminal reachable
    from inside the sandbox. `nginx_port` stays in the list too: the redirect
    listener still accepts the connection.
    """
    # Arrange
    config = _config(["alice"], method="none")
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/tls.crt",
        "key": "/etc/nginx/certs/tls.key",
    }

    # Act
    denied = _env(_user_services(config)["web-alice"])[_DENY_PORTS_VAR]

    # Assert
    assert denied == f"443,{_NGINX_PORT},{_WEB_BASE_PORT}"


def test_plain_http_deployment_does_not_deny_443() -> None:
    """Without TLS nothing listens on 443, and the list does not invent it.

    Denying a port this deployment never publishes would be a guess about the
    host rather than a fact about the render — and the whole point of deriving
    the list here is that it names ports this render knows exist.
    """
    # Arrange
    config = _config(["alice"], method="none")

    # Act
    denied = _env(_user_services(config)["web-alice"])[_DENY_PORTS_VAR]

    # Assert
    assert "443" not in denied.split(",")
    assert denied == f"{_NGINX_PORT},{_WEB_BASE_PORT}"


def test_single_user_deployment_still_denies_the_nginx_port() -> None:
    """A one-user roster carries the front door in its list.

    nginx is the port that injects the secret; a deny-list of terminals alone
    would leave the one route that authenticates on the caller's behalf open.
    """
    # Arrange
    config = _config(["alice"], method="none")

    # Act
    denied = _env(_user_services(config)["web-alice"])[_DENY_PORTS_VAR]

    # Assert
    assert denied.split(",")[0] == str(_NGINX_PORT)


def test_stamp_is_deterministic_across_renders() -> None:
    """Same config in, byte-identical stamp out."""
    # Arrange
    config = _config(["alice", "bob"], method="none")

    # Act
    first = render_web_terminals(copy.deepcopy(config))["docker-compose.web.yml"]
    second = render_web_terminals(copy.deepcopy(config))["docker-compose.web.yml"]

    # Assert
    assert first == second


# ---------------------------------------------------------------------------
# Every other posture renders neither line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "auth",
    [
        pytest.param({}, id="absent-auth-block"),
        pytest.param({"method": "token"}, id="token"),
        pytest.param({"method": "password"}, id="password"),
        pytest.param({"method": "oidc", "oidc": {"issuer": "https://idp.example.org"}}, id="oidc"),
    ],
)
def test_credentialed_postures_render_no_perimeter_stamp(auth: dict) -> None:
    """`token`, `password`, `oidc` and an absent block carry neither variable.

    Absence, not an inert value: a marker set to something falsy would still be
    a variable to interpret, and the postures that render it are the postures
    that mean it.
    """
    # Arrange
    config = _config(["alice", "bob"], **auth)

    # Act
    services = _user_services(config)

    # Assert
    for name, service in services.items():
        env = _env(service)
        assert _MARKER_VAR not in env, name
        assert _DENY_PORTS_VAR not in env, name


def test_no_perimeter_text_at_all_outside_the_open_posture() -> None:
    """Not even a comment mentions the perimeter under `token`.

    The gate wraps the template comment as well as the two lines, so a
    credentialed deployment's compose file is byte-identical to what it
    rendered before this feature — which is what the auth-off baseline pin
    checks line by line.
    """
    # Arrange
    config = _config(["alice"], method="token")

    # Act
    compose = render_web_terminals(copy.deepcopy(config))["docker-compose.web.yml"]

    # Assert
    assert "PERIMETER" not in compose


def test_open_posture_keeps_the_stamp_out_of_the_secret_channel() -> None:
    """The stamp is a literal in `environment:`, never a `.env.users` reference.

    A port list this container's own compose file already publishes is not a
    credential; routing it through the secret channel would imply it is one and
    would put it in a file `osprey users env` rewrites whole.
    """
    # Arrange
    config = _config(["alice"], method="none")

    # Act
    env = _env(_user_services(config)["web-alice"])

    # Assert
    assert "$" not in env[_DENY_PORTS_VAR]
    assert "$" not in env[_MARKER_VAR]
