"""The audit-and-roles feature must be INVISIBLE to a deployment that asked for
neither authentication nor roles — except for the audit trail itself.

This is the line-level pin behind that promise (PROPOSAL success criterion SC6).
A facility whose ``modules.web_terminals`` declares ``auth.method: none`` (or no
``auth:`` block at all) and no ``authorization:`` block must render
byte-for-byte what it rendered before this feature, with exactly two exceptions:

  1. **the audit emitters and mounts** — ``OSPREY_AUDIT_IDENTITY``,
     ``OSPREY_AUDIT_DIR`` and the per-identity ``./var/audit/<identity>`` bind,
     which are unconditional by design: an unauthenticated deployment still
     records what its agents did, and a posture that only writes a trail when
     someone opted into logins would have the trail missing exactly where it is
     least supervised.
  2. **the identity-header clears** — ``proxy_set_header X-Osprey-Auth-Subject
     ""``/``X-Osprey-Auth-Role ""`` in every proxying location. With
     authentication off there is no sidecar to answer for a subject, so nginx
     must claim both names anyway: a location that names neither would hand a
     client's own ``X-Osprey-Auth-Subject: root`` straight to a terminal
     container, which reads that header to learn who is on the other end.

Everything else — every port, volume, header, ``location`` block, comment and
blank line — must be untouched, with one REPLACEMENT the feature makes rather
than an addition: the interim per-user audit bind the baseline already carries
(``./var/audit/<user>`` mounted at the container's audit ROOT, shipped for the
executor's old refusal ledger) becomes the identity-addressed bind above. The
old line and its two comment lines are the only pre-feature lines that may
disappear, and they are allowlisted below by exact spelling.

**The baseline is frozen, not regenerated.** ``golden/pre_audit_roles/`` holds
the three artifacts as ``render_web_terminals(EXAMPLE_CONFIG)`` produced them
*before* the audit-and-roles work, copied out of the commit that preceded it.
Unlike ``golden/`` proper (see ``test_golden_render.py``, which tracks today's
output and is re-generated with every deliberate template change), this
directory is a historical record and must NEVER be refreshed from the current
renderer — doing so would delete the very thing being compared against and turn
this module into a test that passes unconditionally.
``test_the_frozen_baseline_really_predates_the_feature`` exists to make that
mistake fail loudly rather than silently.

**When a hunk here fails**, the question is not "how do I widen the allowlist".
It is: does the new line belong in an auth-off, roles-off render at all? If it
carries authorization, a role, a claim or a login, the answer is no and the
template is wrong. If it is a genuine third exception to SC6, add it to the
allowlist below *and* say so in the PROPOSAL — SC6 is a promise to operators
who never asked for any of this, and it is worth exactly as much as the list.
"""

from __future__ import annotations

import copy
import difflib
from collections import Counter
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.routes.recheck import ENV_ROSTER_ROLE_PREFIX

from .test_golden_render import EXAMPLE_CONFIG, _rendered_repo_id

_BASELINE_DIR = Path(__file__).parent / "golden" / "pre_audit_roles"

#: baseline filename -> the `render_web_terminals()` artifact key it froze.
_ARTIFACTS = {
    "docker-compose.web.yml": "docker-compose.web.yml",
    "nginx.conf": "nginx/nginx.conf",
    "landing.html": "nginx/landing.html",
}

#: Same per-checkout sentinel `test_golden_render.py` uses: the
#: `com.osprey.repo-id` label hashes the resolved deployment path, so it cannot
#: be committed literally.
_REPO_ID_SENTINEL = "@REPO_ID@"

# --- The allowlist -----------------------------------------------------------
# Exact rendered lines, indentation included, with the multiplicity they are
# expected at (EXAMPLE_CONFIG's roster is alice + bob, so most appear once per
# user). Literals rather than patterns on purpose: a regex here would quietly
# absorb a changed path, a renamed variable or a third user's worth of lines.

#: Task 3.1 (compose-audit-mounts). Two env emitters and one bind per identity.
_ALLOWED_COMPOSE_LINES = Counter(
    {
        "      - OSPREY_AUDIT_IDENTITY=alice": 1,
        "      - OSPREY_AUDIT_DIR=/app/dls-assistant/var/audit/alice": 1,
        "      - ./var/audit/alice:/app/dls-assistant/var/audit/alice": 1,
        "      - OSPREY_AUDIT_IDENTITY=bob": 1,
        "      - OSPREY_AUDIT_DIR=/app/dls-assistant/var/audit/bob": 1,
        "      - ./var/audit/bob:/app/dls-assistant/var/audit/bob": 1,
    }
)

#: The interim per-user audit bind the pre-feature render carried — mounted at
#: the container's audit ROOT, for the executor's old refusal ledger — which the
#: identity-addressed bind above replaces. These, and ONLY these, may vanish
#: from the auth-off render; anything else that disappears is still a failure.
_REPLACED_COMPOSE_LINES = frozenset(
    {
        "      # This user's refusal audit log (`var/audit/<user>/` on the host), bound",
        "      # so the record survives a recreate and is readable from the host.",
        "      - ./var/audit/alice:/app/dls-assistant/var/audit",
        "      - ./var/audit/bob:/app/dls-assistant/var/audit",
    }
)

#: Task 4.7 (nginx-identity-headers), ungated arm. Two clears per `/u/<user>/`
#: location, and NOTHING else: no `auth_request_set`, no forward, no `/auth/`
#: location — those render only with authentication on.
_ALLOWED_NGINX_LINES = Counter(
    {
        '        proxy_set_header X-Osprey-Auth-Subject "";': 2,
        '        proxy_set_header X-Osprey-Auth-Role "";': 2,
    }
)

#: Landing page: no exception at all. The audit trail and the identity headers
#: are both invisible to it, so it must match the baseline byte for byte.
_ALLOWED_LANDING_LINES: Counter[str] = Counter()

_ALLOWED = {
    "docker-compose.web.yml": _ALLOWED_COMPOSE_LINES,
    "nginx.conf": _ALLOWED_NGINX_LINES,
    "landing.html": _ALLOWED_LANDING_LINES,
}


def _baseline(name: str) -> str:
    """The frozen pre-feature artifact, with its one per-checkout sentinel resolved."""
    return (_BASELINE_DIR / name).read_text().replace(_REPO_ID_SENTINEL, _rendered_repo_id())


def _auth_off_render() -> dict[str, str]:
    """Today's render of the SC6 shape: no ``auth:``, no ``authorization:``."""
    return render_web_terminals(EXAMPLE_CONFIG)


def _is_comment(line: str) -> bool:
    """True for a comment or blank line in either dialect (both use ``#``)."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _opcodes(name: str) -> tuple[list[str], list[str], list[tuple]]:
    """Baseline lines, current lines, and the line-level edit script between them."""
    old = _baseline(name).splitlines()
    new = _auth_off_render()[_ARTIFACTS[name]].splitlines()
    return old, new, difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()


def test_the_frozen_baseline_really_predates_the_feature() -> None:
    """Anti-tamper: the baseline must not contain the feature it is the baseline
    FOR. Refreshing ``golden/pre_audit_roles/`` from the current renderer is the
    one way to make every test below pass while proving nothing, so the markers
    this module allowlists are asserted ABSENT from the frozen copy."""
    for name in _ARTIFACTS:
        assert (_BASELINE_DIR / name).is_file(), f"missing frozen baseline: {name}"
        assert (_BASELINE_DIR / name).read_text().strip(), f"empty frozen baseline: {name}"

    compose = (_BASELINE_DIR / "docker-compose.web.yml").read_text()
    assert "OSPREY_AUDIT_IDENTITY" not in compose
    assert "OSPREY_AUDIT_DIR" not in compose
    # The identity-addressed target is the feature's marker; the baseline may
    # carry the interim root-of-`var/audit` bind it replaces (see the module
    # docstring), so the bare `/var/audit/` substring is not the test.
    assert "/app/dls-assistant/var/audit/alice" not in compose
    assert "/app/dls-assistant/var/audit/bob" not in compose
    for line in _REPLACED_COMPOSE_LINES:
        assert line in compose, (
            f"the frozen baseline lacks the interim bind it is said to carry: {line!r}"
        )

    nginx = (_BASELINE_DIR / "nginx.conf").read_text()
    assert "X-Osprey-Auth-Subject" not in nginx
    assert "X-Osprey-Auth-Role" not in nginx


def test_an_explicit_auth_method_none_renders_what_an_absent_auth_block_renders() -> None:
    """SC6 names ``auth.method: none``; ``EXAMPLE_CONFIG`` omits ``auth:``
    entirely. `_auth_tls_context` defaults the missing block to ``"none"``, so
    the two are the same posture — pinned here rather than assumed, because if
    they ever diverge the allowlist below would be guarding only one of them."""
    explicit = copy.deepcopy(EXAMPLE_CONFIG)
    explicit["modules"]["web_terminals"]["auth"] = {"method": "none"}

    assert render_web_terminals(explicit) == _auth_off_render()


def test_no_pre_feature_line_is_removed_or_reworded() -> None:
    """Half of "byte-identical EXCEPT": the feature may only ADD. Every baseline
    line — directives, comments and blanks alike — must still be there, in
    order. A reworded comment, a re-indented directive or a dropped blank line
    fails here even though the allowlist below would never see it (it inspects
    insertions only)."""
    for name in _ARTIFACTS:
        old, new, opcodes = _opcodes(name)
        replaceable = _REPLACED_COMPOSE_LINES if name == "docker-compose.web.yml" else frozenset()
        lost = [
            (tag, old[i1:i2], new[j1:j2])
            for tag, i1, i2, j1, j2 in opcodes
            if tag in ("delete", "replace") and not set(old[i1:i2]) <= replaceable
        ]
        assert not lost, (
            f"{name}: the auth-off render no longer contains lines the pre-feature "
            f"render had. SC6 permits additions only: {lost}"
        )


def test_landing_page_is_byte_identical_to_the_pre_feature_render() -> None:
    """The strictest form of SC6, available for the one artifact with no
    exception: an operator's landing page must not have moved by one byte."""
    assert _auth_off_render()["nginx/landing.html"] == _baseline("landing.html")


def test_compose_adds_exactly_the_audit_emitters_and_mounts() -> None:
    """The compose overlay's whole SC6 exception: two env emitters and one bind
    per identity, from task 3.1. Set-with-multiplicity equality both ways, so a
    stray new line fails AND a silently dropped audit mount fails."""
    _assert_added_directives_are_exactly_allowed("docker-compose.web.yml")


def test_nginx_adds_exactly_the_two_identity_header_clears() -> None:
    """The nginx config's whole SC6 exception: the two clears, once each in each
    of the two ungated `/u/<user>/` locations, from task 4.7. Nothing from the
    gated arm may appear here — no `auth_request_set`, no forward — and the
    count catches a clear that reached only one of the two locations."""
    _assert_added_directives_are_exactly_allowed("nginx.conf")


def test_no_authorization_vocabulary_reaches_a_roles_off_render() -> None:
    """The intent behind the allowlist, stated directly. A deployment that
    declared no ``authorization:`` block must carry no role machinery in any
    artifact: not a role claim, not a role map, not a sidecar recheck. This
    catches a whole class of leak in one assertion, including in the comment
    text the line allowlist deliberately ignores."""
    artifacts = _auth_off_render()
    for key, text in artifacts.items():
        for marker in (
            "OSPREY_AUTH_ROLE_CLAIM",
            "OSPREY_AUTH_ROLE_MAP",
            # Imported rather than typed: a rename that reached only the sidecar
            # would leave this guard watching a spelling nothing emits any more.
            ENV_ROSTER_ROLE_PREFIX,
            "auth_request_set",
            "$osprey_auth_subject",
            "$osprey_auth_role",
        ):
            assert marker not in text, (
                f"{key}: {marker!r} reached a render with no authentication and no "
                f"authorization block"
            )


def test_the_auth_off_render_has_no_auth_sidecar_service_at_all() -> None:
    """The structural reason the allowlist above stays short, pinned so it cannot
    quietly stop being true.

    Everything the sidecar carries — its password hashes, its mapped subjects,
    its OIDC settings, its role claim/map, its per-user
    ``OSPREY_AUTH_ROSTER_ROLE_<suffix>`` bindings — lives inside a service that
    ``auth.method: none`` never renders. So work on the sidecar's environment
    block cannot move this render, and lines from it must never be added to the
    SC6 allowlist: the allowlist asserts equality in BOTH directions, so an entry
    that never renders here would fail as a missing exception rather than pass as
    a permitted one.

    If this test ever fails, an auth-off deployment has grown a login service it
    did not ask for, and that is the finding — not the allowlist's shortness."""
    compose = yaml.safe_load(_auth_off_render()["docker-compose.web.yml"])

    assert set(compose["services"]) == {"nginx", "web-alice", "web-bob"}


def _assert_added_directives_are_exactly_allowed(name: str) -> None:
    """Every inserted non-comment line is in the allowlist for ``name``, at the
    expected multiplicity, and every allowlisted line was actually inserted.

    Comments are excluded on purpose: pinning prose line by line would duplicate
    the golden fixture without adding a guarantee, and reworded prose is already
    caught by `test_no_pre_feature_line_is_removed_or_reworded` (which sees
    comments) plus `test_no_authorization_vocabulary_reaches_a_roles_off_render`
    (which reads the comment text for the vocabulary that must not appear).
    """
    _old, new, opcodes = _opcodes(name)
    inserted = Counter(
        line
        for tag, _i1, _i2, j1, j2 in opcodes
        if tag in ("insert", "replace")
        for line in new[j1:j2]
        if not _is_comment(line)
    )

    unexpected = inserted - _ALLOWED[name]
    missing = _ALLOWED[name] - inserted
    assert not unexpected, (
        f"{name}: line(s) added to the auth-off render that SC6 does not allow. "
        f"Add them to the allowlist only if they genuinely belong in a render "
        f"that asked for neither logins nor roles: {sorted(unexpected.elements())}"
    )
    assert not missing, (
        f"{name}: SC6's allowed exception(s) are no longer being rendered — the "
        f"audit trail or the identity-header clears went missing: "
        f"{sorted(missing.elements())}"
    )
