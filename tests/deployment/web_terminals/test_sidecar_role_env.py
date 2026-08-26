"""How a role reaches the auth sidecar — two inputs, and what each is for.

There are exactly two, and each has its own end of the compose overlay to cross:

* the deployment's GROUP BINDING, under `oidc` only. `_authorization_context()`
  parses `modules.web_terminals.authorization.claims` into
  `authorization_claim` / `authorization_claim_map`, and the sidecar's
  `RoleBinding.from_env()` reads it back out of `OSPREY_AUTH_ROLE_CLAIM` /
  `OSPREY_AUTH_ROLE_MAP`. It says which role a validated ID token resolves to.
* the ROSTER's own `role:`, under BOTH login methods, carried per user onto
  render.py's service dicts and read back by `RosterRoles.from_env()` out of
  `OSPREY_AUTH_ROSTER_ROLE_<SUFFIX>`. It is what the render resolved this user's
  persona from — so it is the session role under `password`, and under `oidc`
  it is the CROSS-CHECK TARGET the claim's role must agree with.

The compose overlay is the only thing joining either pair of ends, and until it
emitted these lines the feature was dark on both sides: a deployment with a
claim map, or with a roster full of roles, resolved no role at all. Deny-safe,
and invisible.

Only one of them is gated on the method, and the asymmetry is the point: a claim
binding under `password` would be read out of an ID token nobody presented, so
it is absent there — while the roster role is emitted under both, because a
container is rendered from it either way and the federated posture needs it to
have something to cross-check against. Both gates are asserted here.

These tests assert on the two ends together — the exact rendered text, and the
table the sidecar's own parser builds from it — because either end alone can be
self-consistently wrong. Every variable name is imported from the sidecar rather
than spelled out here, so a rename on that side fails here instead of silently
rendering a variable nothing reads.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import yaml

from osprey.deployment.web_terminals import render as render_module
from osprey.deployment.web_terminals.personas import env_var_suffix
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.app import ENV_USERS
from osprey.services.auth_sidecar.routes.oidc import ENV_ROLE_CLAIM, ENV_ROLE_MAP, RoleBinding
from osprey.services.auth_sidecar.routes.recheck import ENV_ROSTER_ROLE_PREFIX, RosterRoles

#: A group name shaped the way an LDAP-backed IdP emits one: commas and equals
#: signs from the DN itself, and — inside the `cn` — the `": "` an operator
#: writes in a human-readable group name. That last sequence is the one that
#: makes a YAML plain scalar parse as a mapping instead of a string, which is
#: what the quoting in the template is for.
_DN_GROUP = "cn=ALS Operators: Day Shift, ou=Groups, dc=dls, dc=example, dc=org"


def _config(
    *,
    authorization: dict | None = None,
    method: str = "oidc",
    users: list | None = None,
    personas: dict | None = None,
    default_persona: str | None = None,
) -> dict:
    """A rendering config with the sidecar on and, optionally, an authorization block.

    ``users`` takes bare names or full roster mappings; ``personas`` /
    ``default_persona`` are what a roster carrying `role:` needs, since the role
    resolves to a persona the catalog has to declare before the strict render
    will produce anything at all.
    """
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": 9091,
        "artifact_base_port": 9291,
        "ariel_base_port": 9391,
        "lattice_base_port": 9491,
        "users": users if users is not None else ["alice", "bob"],
        # `allow_insecure_http` because the render refuses any auth method over
        # cleartext otherwise — a gate with its own coverage, not this file's subject.
        "auth": {"method": method, "allow_insecure_http": True},
    }
    if method == "oidc":
        web_terminals["auth"]["oidc"] = {"issuer": "https://sso.dls.example.org"}
    if authorization is not None:
        web_terminals["authorization"] = authorization
    if personas is not None:
        web_terminals["personas"] = personas
    if default_persona is not None:
        web_terminals["default_persona"] = default_persona
    return {
        "facility": {
            "name": "Demo Light Source",
            "prefix": "dls",
            "timezone": "America/Los_Angeles",
        },
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


def _authorization(claim_map: dict[str, str] | None) -> dict:
    """Two declared roles, and (when `claim_map` is given) the claim binding them."""
    block: dict = {
        "roles": {
            "operator": {"persona": "operator"},
            "observer": {"persona": "observer"},
        }
    }
    if claim_map is not None:
        block["claims"] = {"claim": "groups", "map": claim_map}
    return block


def _sidecar_env(config: dict) -> list[str]:
    """The `KEY=value` lines the rendered overlay gives the sidecar, YAML-parsed.

    Parsed rather than grepped: the whole point of quoting these lines is that a
    group value containing `": "` still arrives as one scalar, and only the YAML
    loader can tell the difference between that and a line that split.
    """
    overlay = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    env_lines = overlay["services"]["auth"]["environment"]
    # A value that split mid-scalar comes back as a nested mapping, which would
    # otherwise surface as an AttributeError deep in a later helper rather than
    # as the quoting failure it is.
    assert all(isinstance(line, str) for line in env_lines), (
        f"an environment entry did not parse as one scalar: {env_lines!r}"
    )
    return env_lines


def _binding(env_lines: list[str]) -> RoleBinding:
    """The binding the sidecar builds from exactly what the render handed it."""
    return RoleBinding.from_env(dict(line.split("=", 1) for line in env_lines))


def _role_lines(env_lines: list[str]) -> list[str]:
    return [line for line in env_lines if line.startswith((ENV_ROLE_CLAIM, ENV_ROLE_MAP))]


def test_a_configured_binding_reaches_the_sidecar_as_the_two_role_variables() -> None:
    """The claim name and the compact-JSON map, verbatim.

    The map is pinned as an exact string, not as a re-parse: the separator
    spelling is the contract (`json.dumps(..., separators=(",", ":"))`), and a
    `tojson`-rendered `{"a": "b"}` would satisfy any assertion that only checked
    what it parses back to.
    """
    # Arrange
    claim_map = {"als-operators": "operator", "als-observers": "observer"}

    # Act
    env_lines = _sidecar_env(_config(authorization=_authorization(claim_map)))

    # Assert
    assert f"{ENV_ROLE_CLAIM}=groups" in env_lines
    expected = json.dumps(claim_map, separators=(",", ":"))
    assert f"{ENV_ROLE_MAP}={expected}" in env_lines
    # Spelled out once, so the compact separator is asserted and not merely implied.
    assert f'{ENV_ROLE_MAP}={{"als-operators":"operator","als-observers":"observer"}}' in env_lines


def test_the_rendered_binding_round_trips_through_the_sidecars_own_parser() -> None:
    """What the render emits is what `RoleBinding.from_env()` reads back.

    The assertion that matters end to end: not that two env lines exist, but that
    the sidecar's parser turns them into the table the facility wrote.
    """
    # Arrange
    claim_map = {"als-operators": "operator", "als-observers": "observer"}

    # Act
    binding = _binding(_sidecar_env(_config(authorization=_authorization(claim_map))))

    # Assert
    assert binding.configured
    assert binding.claim == "groups"
    assert binding.claim_map == claim_map


@pytest.mark.parametrize(
    "claim_value",
    [
        pytest.param('CN=ops&more,OU="x",DC=lab', id="ampersand-quotes-commas"),
        pytest.param("<ops> 'shift' \\ back", id="angles-apostrophe-backslash"),
        pytest.param("Bedienung-Jörg", id="non-ascii"),
    ],
)
def test_a_hostile_claim_value_round_trips_through_the_sidecars_parser(claim_value: str) -> None:
    """Group values belong to the IdP, and `tojson` is HTML-safe: `&`, `<`, `>`
    and `'` leave the render as `\\uXXXX` escapes rather than literally. The
    contract is the round trip, not the spelling — so this asserts only what the
    sidecar reads back, never the rendered bytes."""
    # Arrange
    claim_map = {claim_value: "operator"}

    # Act
    binding = _binding(_sidecar_env(_config(authorization=_authorization(claim_map))))

    # Assert
    assert binding.claim_map == claim_map


def test_a_deployment_with_no_authorization_block_binds_no_roles() -> None:
    """No stanza renders no lines, and the sidecar reads that as an unconfigured
    binding — the roleless login, not the broken one. An empty-valued pair would
    read the same way here, but would put a dead variable in the committed
    artifact for every deployment that never asked for roles."""
    # Act
    env_lines = _sidecar_env(_config())

    # Assert
    assert _role_lines(env_lines) == []
    assert not _binding(env_lines).configured


def test_roles_without_a_claims_stanza_emit_neither_half() -> None:
    """Roles declared for the roster to name, with no IdP binding, is a complete
    configuration — not half of one.

    An empty `OSPREY_AUTH_ROLE_CLAIM=` rendered beside a populated map (or the
    reverse) would make `RoleBinding.configured` true with nothing to resolve,
    and every OIDC login would then be refused on a deployment whose
    authorization block is exactly as its operator wrote it.
    """
    # Act
    env_lines = _sidecar_env(_config(authorization=_authorization(None)))

    # Assert
    assert _role_lines(env_lines) == []
    assert not _binding(env_lines).configured


def test_a_dn_style_group_value_survives_as_one_yaml_scalar() -> None:
    """Group values are the provider's, and an LDAP-backed IdP's are DNs.

    A DN carries `,`, `=` and `": "`; unquoted, the last of those turns this list
    item into a YAML mapping and the value the sidecar reads is a truncated group
    name that matches nobody. Asserted through the YAML loader and the sidecar's
    parser, which is where a quoting mistake actually shows up.
    """
    # Arrange
    claim_map = {_DN_GROUP: "operator"}

    # Act
    env_lines = _sidecar_env(_config(authorization=_authorization(claim_map)))
    binding = _binding(env_lines)

    # Assert — one scalar, still the whole DN, and still one key in the table.
    assert f"{ENV_ROLE_MAP}={json.dumps(claim_map, separators=(',', ':'))}" in env_lines
    assert binding.claim_map == {_DN_GROUP: "operator"}
    assert list(binding.claim_map) == [_DN_GROUP]


def test_a_half_binding_from_the_context_renders_no_line_at_all() -> None:
    """The pair-gate in the template, exercised directly.

    `_authorization_context()` already refuses half a `claims:` stanza, so no
    config can reach the template with one half — which is exactly why this is
    asserted against a substituted context instead. The template is the last
    thing between a context that lost a half and a sidecar that reads
    `configured` as true with an empty table, and a rule with no way to fail is
    a rule that quietly stops holding when the layer above it changes.
    """
    # Arrange — a context shaped like a future parser change: claim, no map.
    half = {
        "authorization_roles": {"operator": "operator"},
        "authorization_claim": "groups",
        "authorization_claim_map": {},
    }

    # Act
    with patch.object(render_module, "_authorization_context", return_value=half):
        env_lines = _sidecar_env(_config())

    # Assert — neither half, so the sidecar reads "no roles" rather than "broken".
    assert _role_lines(env_lines) == []
    assert not _binding(env_lines).configured


def test_the_role_variables_are_gated_on_the_oidc_method() -> None:
    """A password deployment carries no OIDC settings at all, and the role claim is
    one: it is read out of an ID token, and a password login presents none. That
    posture's role comes from the roster entry instead, so a binding rendered here
    would be a variable nothing reads — and, being half the OIDC story, one that
    reads as a broken deployment rather than an absent feature.
    """
    # Act
    env_lines = _sidecar_env(
        _config(
            method="password",
            authorization=_authorization({"als-operators": "operator"}),
        )
    )

    # Assert
    assert _role_lines(env_lines) == []
    assert not [line for line in env_lines if "OIDC" in line]


# ---------------------------------------------------------------------------
# The password posture: the roster's own `role:`
# ---------------------------------------------------------------------------

#: Two personas for the roles below to bind. A role that named no catalogued
#: persona would not render at all (`resolve_personas` is strict here), so the
#: catalog is a precondition of the fixture rather than part of its subject.
_ROSTER_PERSONAS = {
    "console": {"project": "dls-assistant", "project_path": "profiles/console"},
    "readonly": {"project": "dls-assistant", "project_path": "profiles/readonly"},
}

#: A role name written the way an operator writes a human-readable one, carrying
#: the `": "` that turns an unquoted YAML list item into a mapping. Lint has its
#: own charset rule for role names; the render does not enforce it, so the
#: quoting is what stands between a config like this and a sidecar reading a
#: truncated role.
_SEPARATOR_ROLE = "shift lead: day"


def _roster_authorization(extra_roles: dict[str, str] | None = None) -> dict:
    """The `{role: persona}` table the roster entries below bind through."""
    roles: dict = {
        "operator": {"persona": "console"},
        "observer": {"persona": "readonly"},
    }
    for role_name, persona in (extra_roles or {}).items():
        roles[role_name] = {"persona": persona}
    return {"roles": roles}


#: alice binds a role; `bob-j` binds one under a name whose env-var suffix is not
#: just an uppercasing (`-` maps to `_`); carol pins a persona directly and so
#: holds no role at all. One roster covering all three shapes, because they have
#: to coexist — the render walks them in one pass.
_ROLE_ROSTER = [
    {"name": "alice", "index": 0, "role": "operator"},
    {"name": "bob-j", "index": 1, "role": "observer"},
    {"name": "carol", "index": 2, "persona": "readonly"},
]


def _roster_config(*, method: str = "password", users: list | None = None, **kwargs) -> dict:
    """A rendering config whose roster binds personas through `role:`."""
    return _config(
        method=method,
        users=_ROLE_ROSTER if users is None else users,
        authorization=_roster_authorization(kwargs.pop("extra_roles", None)),
        personas=_ROSTER_PERSONAS,
        default_persona="console",
        **kwargs,
    )


def _roster_role_lines(env_lines: list[str]) -> list[str]:
    return [line for line in env_lines if line.startswith(ENV_ROSTER_ROLE_PREFIX)]


def _roster_roles(env_lines: list[str]) -> RosterRoles:
    """The role table the sidecar builds from exactly what the render handed it.

    The roster is taken from the rendered `OSPREY_AUTH_USERS` line rather than
    from the fixture, because that is where the running service takes it from:
    `RosterRoles.from_env` builds its keys from the roster it was told about, so
    a render emitting a role under a suffix that roster never derives would
    still satisfy a test that supplied the names by hand.
    """
    env = dict(line.split("=", 1) for line in env_lines)
    users = tuple(name for name in env[ENV_USERS].split(",") if name)
    return RosterRoles.from_env(users, env)


def _role_var(username: str) -> str:
    """The variable one user's role is expected under, derived the one way."""
    return f"{ENV_ROSTER_ROLE_PREFIX}{env_var_suffix(username)}"


class TestTheRosterRoleReachesTheSidecar:
    """One `OSPREY_AUTH_ROSTER_ROLE_<SUFFIX>` per role-carrying entry, both methods."""

    def test_each_role_is_emitted_under_this_users_own_variable(self) -> None:
        """The exact lines, with the variable names derived rather than spelled.

        `env_var_suffix` is the single definition of the username->env-var
        mapping — it is what keys this user's password hash — so the expectation
        is built from it here. `bob-j` is in the roster precisely because its
        suffix is not a plain uppercasing: a template that re-derived the
        mapping with `upper` alone would emit `OSPREY_AUTH_ROSTER_ROLE_BOB-J`,
        which is not a legal variable name and which nothing would ever read.
        """
        # Act
        env_lines = _sidecar_env(_roster_config())

        # Assert
        assert _roster_role_lines(env_lines) == [
            f"{_role_var('alice')}=operator",
            f"{_role_var('bob-j')}=observer",
        ]
        assert f"{ENV_ROSTER_ROLE_PREFIX}BOB_J=observer" in env_lines

    def test_the_rendered_roles_round_trip_through_the_sidecars_own_table(self) -> None:
        """What the render emits is what `RosterRoles.from_env()` reads back.

        The assertion that matters end to end: not that env lines exist, but
        that the table the sidecar builds answers `role_for` with the role the
        facility wrote against the user it wrote it for.
        """
        # Act
        roles = _roster_roles(_sidecar_env(_roster_config()))

        # Assert
        assert roles.role_for("alice") == "operator"
        assert roles.role_for("bob-j") == "observer"
        # And the roleless entry is absent from the table rather than keyed to
        # "": an empty role is "no privileges", not a role named "".
        assert roles.role_for("carol") == ""
        assert dict(roles.roles) == {"alice": "operator", "bob-j": "observer"}

    def test_a_persona_pinned_entry_gets_no_role_of_its_own(self) -> None:
        """carol pins `persona: readonly` and holds no role — and none is invented.

        The direction of the mapping is one-way: a role BINDS a persona
        (`authorization.roles.<role>.persona`), so a persona can be read off a
        role but never the other way. Deriving carol a role from her pin would
        have to search the role table backwards for whichever role happens to
        name `readonly` today — handing her a privilege she never asked for, and
        silently re-tiering her the next time another role is bound to the same
        persona.
        """
        # Act
        env_lines = _sidecar_env(_roster_config())
        role_lines = _roster_role_lines(env_lines)

        # Assert — no variable for carol, under any spelling...
        assert not [line for line in role_lines if line.startswith(_role_var("carol"))]
        assert not [line for line in role_lines if "CAROL" in line]
        # ...and her persona name appears as nobody's role, including her own.
        assert not [line for line in role_lines if line.endswith("=readonly")]
        assert "observer" == _roster_roles(env_lines).role_for("bob-j")

    def test_a_roster_that_declares_no_roles_at_all_emits_nothing(self) -> None:
        """Every deployment written before roles existed, unchanged.

        A bare-string roster carries no `role:`, so the sidecar reads an empty
        table and every login resolves "". An empty-valued variable per user
        would read the same way and would put a dead line in the committed
        artifact of every deployment that never asked for roles.
        """
        # Act
        env_lines = _sidecar_env(_config(method="password"))

        # Assert
        assert _roster_role_lines(env_lines) == []
        assert dict(_roster_roles(env_lines).roles) == {}

    def test_the_roster_roles_reach_the_sidecar_under_oidc_too(self) -> None:
        """NOT gated on the method, unlike the claim binding — and this is the
        line that carries SC4.

        A roster `role:` is what the render resolved this user's persona from,
        so it already decided which container their login lands in. Under `oidc`
        the sidecar needs it as the CROSS-CHECK TARGET: a validated ID token
        mapping someone into a different role describes a terminal they are not
        about to enter, and the sidecar can only refuse that login if the
        rendered role reached it. Without these lines the container had no
        per-user role input at all and the cross-check was unenforceable.
        """
        # Act
        env_lines = _sidecar_env(_roster_config(method="oidc"))

        # Assert — the same lines as under `password`, read back the same way.
        assert _roster_role_lines(env_lines) == [
            f"{_role_var('alice')}=operator",
            f"{_role_var('bob-j')}=observer",
        ]
        assert dict(_roster_roles(env_lines).roles) == {"alice": "operator", "bob-j": "observer"}

    def test_a_role_name_carrying_a_yaml_separator_survives_as_one_scalar(self) -> None:
        """Role names are config, and an operator writes readable ones.

        `": "` inside an unquoted list item makes YAML parse it as a mapping;
        what reaches the sidecar is then a truncated role that matches no
        declared privilege — or, worse, a shorter role name that matches a
        different one. Asserted through the YAML loader and the sidecar's own
        table, which is where a quoting mistake actually shows up.
        """
        # Arrange
        roster = [{"name": "alice", "index": 0, "role": _SEPARATOR_ROLE}]

        # Act
        env_lines = _sidecar_env(
            _roster_config(users=roster, extra_roles={_SEPARATOR_ROLE: "console"})
        )

        # Assert — one scalar, and still the whole role name.
        assert _roster_role_lines(env_lines) == [f"{_role_var('alice')}={_SEPARATOR_ROLE}"]
        assert _roster_roles(env_lines).role_for("alice") == _SEPARATOR_ROLE
