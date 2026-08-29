"""The ``modules.web_terminals.authorization`` parser: roles, claims, and the
lint rules that guard the two tables it produces.

``authorization:`` is the static binding between an identity and the privilege
set it runs with. It has two halves::

    authorization:
      roles:
        operator: {persona: operator}
        expert:   {persona: expert}
      claims:
        claim: groups
        map:
          als-operators: operator
          als-experts:   expert

``roles`` is the static half and applies in every auth posture (a roster
entry's ``role:`` names one of them); ``claims`` is the OIDC half, mapping an
ID-token claim's values onto the same role names.

Three properties are under test here, and each has a section:

* **One parse site.** ``render._authorization_context`` is the only place the
  stanza is read, mirroring ``_auth_tls_context`` for ``auth``/``tls``: it
  returns flat context keys, reads wrong-typed containers defensively, and
  raises only where a wrong reading would bind a privilege SILENTLY WRONG.
* **Nothing already deployed changes.** A config with no ``authorization:``
  block — every ``none``/``password``/``oidc``-without-claims deployment
  written so far — parses to inert defaults and renders exactly as before.
* **Lint is the schema gate.** Role-name charset, ``$``-bearing values, roster
  entries naming a role that was never declared, and the parse errors
  themselves all surface as findings rather than as an exception out of a
  scaffold command.

Downstream consumers (the shared persona helper, the sidecar's claim -> role
step, the sidecar's role payload) read the parsed keys, never the raw stanza,
so the contract those tasks build on is pinned here.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey.deployment.web_terminals.lint import Finding, lint_web_terminals
from osprey.deployment.web_terminals.personas import normalize_users
from osprey.deployment.web_terminals.render import (
    _authorization_context,
    render_web_terminals,
)

from .web_terminals.test_golden_render import EXAMPLE_CONFIG

#: A coherent stanza, used wherever a test needs the working shape rather than a
#: broken one. Two roles and a claim map over both, which is the smallest config
#: that can express the ambiguity the downstream resolver must fail closed on.
#: The personas it names are deliberately not declared in the lint config below:
#: whether a role's persona exists in the catalog is the unknown-persona rule's
#: question, asked once the shared helper resolves a role INTO a persona
#: (Task 4.2), and duplicating it here would report the same fact twice.
_AUTHORIZATION: dict[str, Any] = {
    "roles": {
        "operator": {"persona": "operator"},
        "expert": {"persona": "physicist"},
    },
    "claims": {
        "claim": "groups",
        "map": {"dls-operators": "operator", "dls-experts": "expert"},
    },
}


def _web(authorization: Any = None, **extra: Any) -> dict[str, Any]:
    """A bare ``modules.web_terminals`` dict, as the parser is handed one."""
    web_terminals: dict[str, Any] = {"enabled": True, **extra}
    if authorization is not None:
        web_terminals["authorization"] = authorization
    return web_terminals


# ---------------------------------------------------------------------------
# One parse site: the shape it produces
# ---------------------------------------------------------------------------


def test_no_authorization_stanza_parses_to_inert_defaults() -> None:
    """Every deployment written before roles existed must parse to "no roles"."""
    # Arrange
    web_terminals = _web()

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context == {
        "authorization_roles": {},
        "authorization_claim": None,
        "authorization_claim_map": {},
    }


def test_roles_parse_into_a_role_to_persona_table() -> None:
    """The static half: each role names exactly one catalog persona."""
    # Arrange
    web_terminals = _web(copy.deepcopy(_AUTHORIZATION))

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_roles"] == {"operator": "operator", "expert": "physicist"}


def test_claims_parse_into_a_claim_name_and_a_value_to_role_table() -> None:
    """The OIDC half: which claim carries membership, and what its values mean."""
    # Arrange
    web_terminals = _web(copy.deepcopy(_AUTHORIZATION))

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_claim"] == "groups"
    assert context["authorization_claim_map"] == {
        "dls-operators": "operator",
        "dls-experts": "expert",
    }


def test_roles_without_claims_is_a_supported_posture() -> None:
    """Roles are assigned from the roster in `none`/`password` deployments, where
    there is no IdP claim to map at all."""
    # Arrange
    web_terminals = _web({"roles": {"operator": {"persona": "operator"}}})

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_roles"] == {"operator": "operator"}
    assert context["authorization_claim"] is None
    assert context["authorization_claim_map"] == {}


def test_parsed_tables_are_new_objects_not_the_config_s_own() -> None:
    """A consumer must not be able to edit the deployment's config by editing
    the table it was handed."""
    # Arrange
    authorization = copy.deepcopy(_AUTHORIZATION)
    web_terminals = _web(authorization)

    # Act
    context = _authorization_context(web_terminals)
    context["authorization_roles"]["intruder"] = "admin"
    context["authorization_claim_map"]["dls-intruders"] = "operator"

    # Assert
    assert "intruder" not in authorization["roles"]
    assert "dls-intruders" not in authorization["claims"]["map"]


def test_role_declaration_order_is_preserved() -> None:
    """Resolution must never be order-dependent, but a report that lists the
    declared roles reads best in the order the operator wrote them."""
    # Arrange
    web_terminals = _web(
        {
            "roles": {
                "zebra": {"persona": "operator"},
                "alpha": {"persona": "physicist"},
            }
        }
    )

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert list(context["authorization_roles"]) == ["zebra", "alpha"]


# ---------------------------------------------------------------------------
# One parse site: defensive reads vs. the loud refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "authorization",
    ["roles", 7, ["operator"], True],
    ids=["string", "int", "list", "bool"],
)
def test_a_wrong_typed_authorization_stanza_reads_as_no_stanza(authorization: Any) -> None:
    """Wrong-typed containers fall back to their default, as everywhere else in
    the render — lint is the authoritative gate on well-formedness."""
    # Arrange
    web_terminals = _web(authorization)

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_roles"] == {}


@pytest.mark.parametrize("key", ["roles", "claims"])
def test_a_wrong_typed_inner_container_reads_as_absent(key: str) -> None:
    """Same defensive reading one level down."""
    # Arrange
    web_terminals = _web({key: "operator"})

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_roles"] == {}
    assert context["authorization_claim_map"] == {}


def test_a_non_string_role_name_is_dropped_rather_than_parsed() -> None:
    """A YAML key that is not a string can never be named by a roster entry's
    `role:` or by a claim value, so it is inert; lint reports it."""
    # Arrange
    web_terminals = _web({"roles": {7: {"persona": "operator"}}})

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_roles"] == {}


@pytest.mark.parametrize(
    "entry",
    [{}, {"persona": ""}, {"persona": None}, {"persona": 7}, "operator", None],
    ids=["empty", "empty-string", "null", "int", "shorthand-string", "null-entry"],
)
def test_a_role_that_names_no_persona_is_refused(entry: Any) -> None:
    """Not inert: every entry carrying such a role would fall back to the
    deployment's default persona — a different privilege set, silently."""
    # Arrange
    web_terminals = _web({"roles": {"operator": entry}})

    # Act / Assert
    with pytest.raises(ValueError, match="operator"):
        _authorization_context(web_terminals)


def test_a_claim_map_naming_an_undeclared_role_is_refused() -> None:
    """Dead config: a login the operator believes is granted that role is not."""
    # Arrange
    web_terminals = _web(
        {
            "roles": {"operator": {"persona": "operator"}},
            "claims": {"claim": "groups", "map": {"dls-experts": "expert"}},
        }
    )

    # Act / Assert
    with pytest.raises(ValueError, match="dls-experts"):
        _authorization_context(web_terminals)


@pytest.mark.parametrize(
    ("value", "role"),
    [(7, "operator"), ("dls-operators", 7), ("dls-operators", None)],
    ids=["non-string-value", "non-string-role", "null-role"],
)
def test_a_claim_map_entry_that_cannot_name_a_role_is_refused(value: Any, role: Any) -> None:
    """Neither half of a mapping may be a type the resolver can never match."""
    # Arrange
    web_terminals = _web(
        {
            "roles": {"operator": {"persona": "operator"}},
            "claims": {"claim": "groups", "map": {value: role}},
        }
    )

    # Act / Assert
    with pytest.raises(ValueError, match="claims.map"):
        _authorization_context(web_terminals)


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        ({"claim": "groups"}, "a claim but no map"),
        ({"claim": "groups", "map": {}}, "a claim but no map"),
        ({"map": {"dls-operators": "operator"}}, "a map but no claim"),
        ({"claim": "", "map": {"dls-operators": "operator"}}, "a map but no claim"),
        ({"unrelated": "key"}, "neither a claim nor a map"),
    ],
    ids=["no-map", "empty-map", "no-claim", "blank-claim", "neither"],
)
def test_half_a_claims_stanza_is_refused(claims: dict[str, Any], expected: str) -> None:
    """A claim with no map resolves nothing; a map with no claim is read from
    nowhere. Either half alone is a stanza that cannot mean what it looks like.

    The message is pinned, not just the raise: it names which half the operator
    actually wrote, and a message that names the other one sends them to edit
    the line that is already correct.
    """
    # Arrange
    web_terminals = _web({"roles": {"operator": {"persona": "operator"}}, "claims": claims})

    # Act / Assert
    with pytest.raises(ValueError, match=f"this stanza has {expected}"):
        _authorization_context(web_terminals)


@pytest.mark.parametrize("claims", [{}, None], ids=["empty", "null"])
def test_an_empty_claims_stanza_is_inert_rather_than_refused(claims: Any) -> None:
    """`claims:` written with nothing under it is absence, not a broken half."""
    # Arrange
    web_terminals = _web({"roles": {"operator": {"persona": "operator"}}, "claims": claims})

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context["authorization_claim"] is None
    assert context["authorization_claim_map"] == {}


# ---------------------------------------------------------------------------
# Nothing already deployed changes
# ---------------------------------------------------------------------------


def test_the_reference_config_parses_to_inert_defaults() -> None:
    """The no-personas reference facility config the goldens are rendered from
    declares no authorization, and must keep meaning exactly that."""
    # Arrange
    web_terminals = copy.deepcopy(EXAMPLE_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context == {
        "authorization_roles": {},
        "authorization_claim": None,
        "authorization_claim_map": {},
    }


def test_an_oidc_deployment_without_a_claims_map_parses_to_inert_defaults() -> None:
    """The shipped OIDC posture maps identities via `oidc_subject`, not claims;
    adding the parser must not conjure an authorization layer for it."""
    # Arrange
    web_terminals = _web(
        auth={
            "method": "oidc",
            "allow_insecure_http": True,
            "oidc": {"issuer": "https://idp.dls.example.org"},
        }
    )

    # Act
    context = _authorization_context(web_terminals)

    # Assert
    assert context == {
        "authorization_roles": {},
        "authorization_claim": None,
        "authorization_claim_map": {},
    }


def test_the_reference_config_still_renders_all_three_artifacts() -> None:
    """The parser runs on every render, so an authorization-less deployment must
    not gain a way to fail."""
    # Arrange
    config = copy.deepcopy(EXAMPLE_CONFIG)

    # Act
    rendered = render_web_terminals(config)

    # Assert
    assert set(rendered) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_a_role_only_roster_renders_byte_identically_to_a_persona_less_one() -> None:
    """Declaring roles binds privileges; on its own it publishes nothing. The
    parser is the whole change at this stage, so the artifacts must not move."""
    # Arrange
    baseline = copy.deepcopy(EXAMPLE_CONFIG)
    with_roles = copy.deepcopy(EXAMPLE_CONFIG)
    with_roles["modules"]["web_terminals"]["authorization"] = {
        "roles": {"operator": {"persona": "operator"}}
    }

    # Act
    rendered = render_web_terminals(with_roles)

    # Assert
    assert rendered == render_web_terminals(baseline)


def test_the_render_refuses_an_incoherent_authorization_block() -> None:
    """The parse site is on the render path, so an incoherent stanza stops the
    deployment rather than producing artifacts that bind the wrong privileges."""
    # Arrange
    config = copy.deepcopy(EXAMPLE_CONFIG)
    config["modules"]["web_terminals"]["authorization"] = {"roles": {"operator": {}}}

    # Act / Assert
    with pytest.raises(ValueError, match="operator"):
        render_web_terminals(config)


# ---------------------------------------------------------------------------
# `role` on a roster entry
# ---------------------------------------------------------------------------


def test_normalize_users_carries_an_entry_s_role() -> None:
    """`role:` is how a roster entry names its privilege binding, so it has to
    survive normalization the way `persona` and `oidc_subject` do."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "role": "operator"}]

    # Act
    users = normalize_users(users_raw)

    # Assert
    assert users == [{"name": "alice", "index": 0, "role": "operator"}]


@pytest.mark.parametrize("role", ["", 7, None, True], ids=["empty", "int", "null", "bool"])
def test_normalize_users_drops_an_unusable_role(role: Any) -> None:
    """An authorization mapping, not a cosmetic one: carrying `""` through would
    leave an entry claiming a role that no table can answer. Dropping it leaves
    the entry with no binding at all, which the consumers read as "default"."""
    # Arrange
    users_raw = [{"name": "alice", "index": 0, "role": role}]

    # Act
    users = normalize_users(users_raw)

    # Assert
    assert "role" not in users[0]


def test_normalize_users_does_not_mutate_the_roster_it_reads() -> None:
    """The roster is written back verbatim by `osprey users remove`, so the
    normalizer must keep its hands off the authored entries."""
    # Arrange
    entry = {"name": "alice", "index": 0, "role": "operator"}
    users_raw = [entry]

    # Act
    users = normalize_users(users_raw)

    # Assert
    assert users[0] is not entry
    assert entry == {"name": "alice", "index": 0, "role": "operator"}


def test_a_bare_string_roster_entry_carries_no_role() -> None:
    """The legacy roster shape has no place to write one."""
    # Arrange
    users_raw = ["alice"]

    # Act
    users = normalize_users(users_raw)

    # Assert
    assert users == [{"name": "alice", "index": 0}]


# ---------------------------------------------------------------------------
# Lint is the schema gate
# ---------------------------------------------------------------------------

_LINT_CONFIG: dict[str, Any] = {
    "facility": {"prefix": "dls"},
    "modules": {
        "web_terminals": {
            "enabled": True,
            "users": ["alice", "bob"],
        }
    },
}


def _lint(
    authorization: Any = None, *, registry: Any = None, **web_terminals: Any
) -> list[Finding]:
    """Lint a well-formed config carrying the authorization stanza under test.

    ``registry`` is a ROOT-level override, not a ``web_terminals`` one: a test
    that declares a ``personas`` catalog opts the config into the persona
    system, and registry mode then wants a ``registry.url`` (an unrelated rule,
    ``web_terminals.registry_mode_missing_url``, which would otherwise drown the
    finding under test).
    """
    config = copy.deepcopy(_LINT_CONFIG)
    if authorization is not None:
        config["modules"]["web_terminals"]["authorization"] = authorization
    if registry is not None:
        config["registry"] = copy.deepcopy(registry)
    config["modules"]["web_terminals"].update(copy.deepcopy(web_terminals))
    return lint_web_terminals(config, rendered_project=False)


def _codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings if f.severity == "error"}


def test_lint_accepts_a_coherent_authorization_block() -> None:
    """The working shape must produce no findings of its own."""
    # Arrange / Act
    findings = _lint(copy.deepcopy(_AUTHORIZATION))

    # Assert
    assert _codes(findings) == set()


@pytest.mark.parametrize(
    "role_name",
    ["Operator", "op erator", "-operator", "opérateur", "op/erator"],
    ids=["uppercase", "space", "leading-dash", "non-ascii", "slash"],
)
def test_lint_rejects_a_role_name_outside_the_username_charset(role_name: str) -> None:
    """A role name is carried on a roster entry, mapped from a claim value and
    forwarded in an HTTP header — the same charset a username is held to."""
    # Arrange / Act
    findings = _lint({"roles": {role_name: {"persona": "operator"}}})

    # Assert
    assert "web_terminals.invalid_role_charset" in _codes(findings)


def test_lint_rejects_a_non_string_role_name() -> None:
    """The parser drops it as unreferenceable; lint is where the operator is
    told why their role never applies."""
    # Arrange / Act
    findings = _lint({"roles": {7: {"persona": "operator"}}})

    # Assert
    assert "web_terminals.invalid_role_charset" in _codes(findings)


@pytest.mark.parametrize(
    "authorization",
    [
        {"roles": {"oper$ator": {"persona": "operator"}}},
        {"roles": {"operator": {"persona": "oper$ator"}}},
        {
            "roles": {"operator": {"persona": "operator"}},
            "claims": {"claim": "grou$ps", "map": {"dls-operators": "operator"}},
        },
        {
            "roles": {"operator": {"persona": "operator"}},
            "claims": {"claim": "groups", "map": {"dls-oper$ators": "operator"}},
        },
    ],
    ids=["role-name", "persona", "claim-name", "claim-value"],
)
def test_lint_refuses_a_dollar_sign_anywhere_in_the_authorization_stanza(
    authorization: dict[str, Any],
) -> None:
    """These strings are rendered into the compose document, where `$` sequences
    are interpolated: the sidecar would compare against a rewritten string and
    the role would silently never be granted."""
    # Arrange / Act
    findings = _lint(authorization)

    # Assert
    assert "web_terminals.authorization_unsafe_value" in _codes(findings)


def test_lint_reports_a_parse_error_as_a_finding_rather_than_raising() -> None:
    """A scaffold command must report every problem in one pass, not die on the
    first one — the same wrapper the auth checks use."""
    # Arrange / Act
    findings = _lint({"roles": {"operator": {}}})

    # Assert
    assert "web_terminals.invalid_authorization" in _codes(findings)


def test_lint_rejects_a_wrong_typed_authorization_stanza() -> None:
    """The render reads it as no stanza at all, so the deployment would come up
    with none of the privilege bindings the operator wrote."""
    # Arrange / Act
    findings = _lint("operator")

    # Assert
    assert "web_terminals.invalid_authorization_stanza" in _codes(findings)


@pytest.mark.parametrize("key", ["roles", "claims"])
def test_lint_rejects_a_wrong_typed_inner_container(key: str) -> None:
    """Same silent-disablement, one level down."""
    # Arrange / Act
    findings = _lint({key: ["operator"]})

    # Assert
    assert "web_terminals.invalid_authorization_stanza" in _codes(findings)


def test_lint_rejects_a_roster_entry_naming_an_undeclared_role() -> None:
    """The entry would render with the deployment's default persona instead of
    the one the operator meant to bind."""
    # Arrange / Act
    findings = _lint(
        copy.deepcopy(_AUTHORIZATION),
        users=[{"name": "alice", "index": 0, "role": "admin"}],
    )

    # Assert
    assert "web_terminals.unknown_role_reference" in _codes(findings)


def test_lint_rejects_a_wrong_typed_role_on_a_roster_entry() -> None:
    """`normalize_users` drops it defensively, so nothing downstream can see the
    typo — this is the only surface that reports it."""
    # Arrange / Act
    findings = _lint(
        copy.deepcopy(_AUTHORIZATION),
        users=[{"name": "alice", "index": 0, "role": 7}],
    )

    # Assert
    assert "web_terminals.invalid_user_role" in _codes(findings)


def test_lint_accepts_a_roster_entry_naming_a_declared_role() -> None:
    """The whole point of the feature must not itself be a finding.

    The catalog is declared here because Task 4.2 wired roles through
    ``effective_persona``: a role now resolves INTO a persona before
    ``_check_unknown_persona_reference`` runs, so ``operator`` must be a real
    catalog entry for this roster to be clean. That is the rule this file's
    ``_AUTHORIZATION`` docstring deferred to the shared helper rather than
    duplicating as a parser rule — pinned as its own case in
    ``tests/deployment/web_terminals/test_personas.py``.
    """
    # Arrange / Act
    findings = _lint(
        copy.deepcopy(_AUTHORIZATION),
        users=[{"name": "alice", "index": 0, "role": "operator"}],
        personas={
            "operator": {
                "project": "dls-operator",
                "build_profile": "personas/operator.yml",
            }
        },
        registry={"url": "registry.example.org"},
    )

    # Assert
    assert _codes(findings) == set()


# ---------------------------------------------------------------------------
# Reserved audit identities (routed from the compose-audit-mounts review)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["sidecar", "dispatch-worker-1", "dispatch-worker-12"], ids=lambda n: n
)
def test_lint_rejects_a_roster_name_that_is_a_service_s_audit_identity(name: str) -> None:
    """Each user's container binds `var/audit/<user>/` read-write, so such a user
    would read and rewrite the audit trail of the component that records them.
    The render already refuses it; lint says so earlier, and `--no-lint` is the
    only path that reaches the render's raise."""
    # Arrange / Act
    findings = _lint(users=[name])

    # Assert
    assert "web_terminals.reserved_audit_identity" in _codes(findings)


@pytest.mark.parametrize(
    "name",
    ["nginx", "dispatch-worker", "dispatch-worker-a", "sidecars"],
    ids=["service-key", "no-index", "non-numeric-index", "superstring"],
)
def test_lint_accepts_a_roster_name_that_merely_resembles_one(name: str) -> None:
    """The rule is about names that ARE a service's audit subdirectory, not names
    that look service-ish: a user's compose service key is `web-<user>`, so
    `nginx` collides with nothing."""
    # Arrange / Act
    findings = _lint(users=[name])

    # Assert
    assert "web_terminals.reserved_audit_identity" not in _codes(findings)
