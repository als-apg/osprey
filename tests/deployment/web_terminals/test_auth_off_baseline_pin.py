"""The audit-and-roles feature must be INVISIBLE to a deployment that asked for
neither authentication nor roles — except for the audit trail itself.

This is the line-level pin behind that promise (PROPOSAL success criterion SC6),
and it is also what makes ``auth.method: token`` the default rather than a new
posture nobody has run. ``token`` carries what ``auth.method: none`` used to
mean — a per-user ``?token=`` magic link, no sidecar, no injected secret — and
it is what an absent ``auth:`` block renders. The frozen golden below is that
posture's output, copied out of a commit that predates the four-method scheme,
so "``token`` is today's shipped behaviour under a new name" is a claim these
tests check byte by byte rather than a claim the PROPOSAL merely makes. That is
the argument for defaulting to it: writing no ``auth:`` block gets a facility
the one posture whose bytes have been pinned all along.

``none`` now means *open* — navigation-only, nginx injects each user's terminal
secret — and renders something deliberately different. Nothing in this module
describes ``none``; its render is pinned by ``test_nginx_auth_surface.py``.

So: a facility whose ``modules.web_terminals`` declares ``auth.method: token``
(or no ``auth:`` block at all) and no ``authorization:`` block must render
byte-for-byte what it rendered before this feature, with exactly four
exceptions:

  1. **the audit emitters and mounts** — ``OSPREY_AUDIT_IDENTITY``,
     ``OSPREY_AUDIT_DIR`` and the per-identity ``./var/audit/<identity>`` bind,
     which are unconditional by design: a deployment with no login wall still
     records what its agents did, and a posture that only writes a trail when
     someone opted into logins would have the trail missing exactly where it is
     least supervised.
  2. **the identity-header clears** — ``proxy_set_header X-Osprey-Auth-Account
     ""``/``X-Osprey-Auth-Subject ""``/``X-Osprey-Auth-Role ""``/
     ``X-Osprey-Auth-Role-Source ""`` in every proxying location. Under
     ``token`` there is no sidecar to answer for an account or a subject, so
     nginx must claim all four names anyway: a location that names none of them
     would hand a client's own ``X-Osprey-Auth-Subject: root`` straight to a
     terminal container, which reads that header to learn who is on the other
     end.
  3. **the terminal session lifetime** — ``OSPREY_TERMINAL_SESSION_LIFETIME``
     on every per-user container. ``token`` mints a session cookie of its own
     (the ``?token=`` exchange trades the magic link for one), so the setting
     that governs how long it lives is not an authentication feature this
     posture opted out of; it is the posture's own cookie. It rides in the
     environment because the persona's ``config.yml`` is baked into the image
     at build time, and an operator who edits ``auth.session_lifetime`` and
     re-renders would otherwise see nothing change. The value emitted here is
     the default an absent ``auth:`` block already resolved to, so nothing
     about a baseline deployment's behaviour moves.
  4. **the landing page's token-login badge** — a chip and one hint line on
     every user card, naming ``osprey users login-url <name>``. This is the one
     exception that changes what an operator SEES, and it is here because under
     ``token`` the page was making a claim it could not keep: every card looked
     like a door you could walk through, when every one of them needs a URL that
     is minted by a verb nobody guesses and appears nowhere on the page. The
     posture did not move — nginx vouches for nobody under ``token`` before and
     after — only the page's honesty about it. The cards keep their hrefs (the
     ``?token=`` exchange leaves a session cookie, so the link is a real return
     path), so nothing about navigation changes either. Under ``password``, the
     posture whose cards this baseline does not cover, the marks appear on the
     ``login: false`` entries alone; ``test_landing_token_badge.py`` pins that
     split.

Everything else — every volume, header, ``location`` block, comment and blank
line, and every port *site* (see the mask below) — must be untouched, with one
REPLACEMENT the feature makes rather than an addition: the interim per-user audit
bind the baseline already carries
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
renderer — doing so would delete the very thing being compared against, turn
this module into a test that passes unconditionally, and destroy the evidence
that ``token`` reproduces the old ``none``.
``test_the_frozen_baseline_really_predates_the_feature`` exists to make that
mistake fail loudly rather than silently.

**Ports are masked, not pinned.** The baseline was frozen when every host port
was an authored literal — nginx on 9080, alice's terminal on 9100. Today they
come from the port layout, so this same posture renders the same artifacts on
different numbers: 10000 and 10100 at the default ``deployment.port_base``, and
different numbers again at every other base. Comparing raw bytes would report
the whole port table as a difference and drown the question SC6 actually asks.
So both sides go through :func:`_mask_ports` first, which replaces the number at
every *structural* port site with ``<port>``:

  * an nginx ``listen`` directive, in both its ``9080`` and ``[::]:9080`` forms;
  * the ``host:port`` authority of any URL — ``proxy_pass``, the healthcheck
    ``curl``, ``OSPREY_TERMINAL_LANDING_URL``, the external origin;
  * an ``OSPREY_…_PORT=`` emitter in the compose environment;
  * a compose publish pair (``- "9080:9080"``), which this host-network render
    does not emit today but a bridge-mode topology would;
  * the port inside the ``osprey_terminal_session_<port>`` cookie name and the
    ``$cookie_…`` variable that reads it back.

The mask keys on shape, never on a list of numbers, so it holds at any
``port_base`` and needs no edit when the layout moves. What survives it is what
SC6 is about: which directives, mounts, headers and ``location`` blocks a
``token`` render emits. The numbers themselves are pinned elsewhere — by
``test_golden_render.py`` for today's output and ``test_ports.py`` for the
layout — which is where a wrong port belongs as a failure.

A mask over the numbers would also mask a *refreshed* baseline's numbers, so
``test_the_frozen_baseline_really_predates_the_feature`` reads them back
unmasked: the frozen copy must still spell ports the layout cannot produce for
this config, which is exactly what a copy regenerated from today's renderer
could not do.

**When a hunk here fails**, the question is not "how do I widen the allowlist".
It is: does the new line belong in a ``token``, roles-off render at all? If it
carries authorization, a role, a claim or a login, the answer is no and the
template is wrong. If it is an injected terminal secret, the ``none`` work has
leaked into ``token`` and the fix is in the template's predicate, not here. If
it is a genuine further exception to SC6, add it to the allowlist below, extend
the enumeration above, *and* say so in the PROPOSAL — SC6 is a promise to
operators who never asked for any of this, and it is worth exactly as much as
the list.
"""

from __future__ import annotations

import copy
import difflib
import re
from collections import Counter
from pathlib import Path

import yaml

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.port_layout import default_port, resolve_port_base
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

# --- The port mask -----------------------------------------------------------
# See "Ports are masked, not pinned" in the module docstring. Every pattern
# carries exactly one group named `port`, and only that group's span is
# replaced, so a host or a path that happens to hold the same digits is left
# alone.

#: What a masked port reads as. Deliberately not a number, so a site the mask
#: reaches can never compare equal to a site it missed.
_PORT_MASK = "<port>"

#: The structural port sites, across the two dialects the baseline covers.
_PORT_SITES: tuple[re.Pattern[str], ...] = (
    # `listen 9080;` and `listen [::]:9080;`
    re.compile(r"listen\s+(?:\[::\]:)?(?P<port>\d+)"),
    # The authority of any URL: `proxy_pass http://127.0.0.1:9100/`, the
    # healthcheck `curl`, `OSPREY_TERMINAL_LANDING_URL`, the external origin.
    # Anchored on `://` so the registry reference `git.dls.example.org:5050/…`
    # — a port the layout does not own, and one this pin should keep watching
    # byte for byte — is not a site.
    re.compile(r'://[^\s/:"]+:(?P<port>\d+)'),
    # `- OSPREY_TERMINAL_WEB_PORT=9100`, and the panel emitters beside it.
    re.compile(r"_PORT=(?P<port>\d+)"),
    # A compose publish pair, `- "9080:9080"`. This render is host-network and
    # emits none; a bridge-mode topology would, and the host half is the one
    # `port_base` moves.
    re.compile(r'^\s*-\s*"?(?P<port>\d+):\d+"?\s*$', re.MULTILINE),
    # `osprey_terminal_session_9100` — the cookie name nginx sets and the
    # `$cookie_osprey_terminal_session_9100` variable that reads it back.
    re.compile(r"osprey_terminal_session_(?P<port>\d+)"),
)


def _mask_one(match: re.Match[str]) -> str:
    """Replace just the ``port`` group inside one matched site.

    Args:
        match: A match of one of :data:`_PORT_SITES`.

    Returns:
        The matched text with the ``port`` group's digits replaced by
        :data:`_PORT_MASK` and every other character kept as it was.
    """
    start, end = match.span("port")
    text = match.group(0)
    return text[: start - match.start()] + _PORT_MASK + text[end - match.start() :]


def _mask_ports(text: str) -> str:
    """Blank out the number at every structural port site in one artifact.

    Args:
        text: A rendered or frozen artifact.

    Returns:
        The same text with each port site's number replaced by
        :data:`_PORT_MASK`, so an artifact rendered at one ``port_base``
        compares equal to the same artifact rendered at another — or to the
        frozen baseline, whose ports predate the layout entirely.
    """
    for pattern in _PORT_SITES:
        text = pattern.sub(_mask_one, text)
    return text


def _ports_in(text: str) -> frozenset[int]:
    """Every port number one artifact spells at a structural site.

    The inverse of :func:`_mask_ports` over the same patterns, so the numbers
    read back here are exactly the ones the diffs below stop seeing.

    Args:
        text: A rendered or frozen artifact.

    Returns:
        The set of ports found — empty for an artifact that carries none, as
        ``landing.html`` does not: it addresses users by relative path.
    """
    return frozenset(
        int(match.group("port")) for pattern in _PORT_SITES for match in pattern.finditer(text)
    )


#: The registry port families `EXAMPLE_CONFIG`'s roster renders, one port per
#: user in each.
_RENDERED_FAMILIES = (
    "web",
    "artifact",
    "ariel",
    "lattice",
    "channel_finder",
    "okf",
    "system_health",
)

#: Every port the layout gives `EXAMPLE_CONFIG`, which declares no
#: `deployment.port_base` and so resolves to the default one. Derived rather
#: than typed out: the frozen baseline predates the layout, and "predates" is
#: asserted below as "spells none of these".
_LAYOUT_PORTS = frozenset(
    [default_port("nginx", base=resolve_port_base(EXAMPLE_CONFIG))]
    + [
        default_port(family, index, base=resolve_port_base(EXAMPLE_CONFIG))
        for family in _RENDERED_FAMILIES
        for index in range(len(EXAMPLE_CONFIG["modules"]["web_terminals"]["users"]))
    ]
)


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
        # The cookie-lifecycle feature — the deliberate THIRD exception to SC6,
        # stated in its PROPOSAL. The terminal's own session-cookie lifetime now
        # travels to every per-user container, under `token` as under every
        # other method, because every method mints that cookie and the persona's
        # `config.yml` is baked into the image at build time — so a deploy-time
        # edit of `auth.session_lifetime` is visible to a running container only
        # as environment. Count 2 = alice + bob. The value is the default an
        # absent `auth:` block already resolves to, so the line restates today's
        # behaviour rather than changing it.
        "      - OSPREY_TERMINAL_SESSION_LIFETIME=43200": 2,
    }
)

#: The interim per-user audit bind the pre-feature render carried — mounted at
#: the container's audit ROOT, for the executor's old refusal ledger — which the
#: identity-addressed bind above replaces. These, and ONLY these, may vanish
#: from the `token` render; anything else that disappears is still a failure.
_REPLACED_COMPOSE_LINES = frozenset(
    {
        "      # This user's refusal audit log (`var/audit/<user>/` on the host), bound",
        "      # so the record survives a recreate and is readable from the host.",
        "      - ./var/audit/alice:/app/dls-assistant/var/audit",
        "      - ./var/audit/bob:/app/dls-assistant/var/audit",
    }
)

#: The per-user healthcheck comment as the pre-feature render worded it, when
#: `osprey web`'s bare default was the fixed constant 8087. The port-block
#: layout rewords it (the default is now the `web` slot at the deployment's
#: base), so these exact comment lines — and ONLY these — may be replaced in
#: the `token` render. The frozen copy keeps the old wording: it is a
#: historical record and is never refreshed (see the module docstring).
_REWORDED_COMPOSE_LINES = frozenset(
    {
        "      # osprey web's own default listen port is the fixed constant 8087",
        "      # (cli/web_cmd.py) — only relevant to a bare `osprey web` with no",
        "      # OSPREY_WEB_PORT override. Every per-user container here sets",
        "      # OSPREY_WEB_PORT above, so the probe targets the ACTUAL bound port,",
        "      # never the bare 8087 default. Probed via bind_host itself (not a",
        '      # hardcoded "127.0.0.1" literal) since that\'s the same loopback',
        "      # address OSPREY_TERMINAL_BIND_HOST bakes into the app's own bind.",
    }
)

#: Task 4.7 (nginx-identity-headers), ungated arm. Four clears per `/u/<user>/`
#: location, and NOTHING else: no `auth_request_set`, no forward, no `/auth/`
#: location — those render only with authentication on.
_ALLOWED_NGINX_LINES = Counter(
    {
        '        proxy_set_header X-Osprey-Auth-Account "";': 2,
        '        proxy_set_header X-Osprey-Auth-Subject "";': 2,
        '        proxy_set_header X-Osprey-Auth-Role "";': 2,
        '        proxy_set_header X-Osprey-Auth-Role-Source "";': 2,
    }
)

#: The token-login badge — SC6 exception 4, and the only one an operator sees.
#: The audit trail and the identity headers are still invisible to this page;
#: what is added is the badge's three style rules and the two spans each user
#: card renders (EXAMPLE_CONFIG's roster is alice + bob, and under `token` every
#: card is a token-login card, which is the whole reason the badge exists).
#:
#: The declaration lines are pinned as difflib aligns them, which is why a few
#: read oddly: `max-width: 100%;` and `color: var(--text-secondary);` appear
#: twice and `}` three times, while `overflow`/`text-overflow`/`white-space`
#: appear once each — the rest matched identical declarations already in the
#: stylesheet and are not insertions at all. That is a property of the edit
#: script, not of the rules; the rules themselves are readable in
#: `landing.html.j2` and in `golden/landing.html`.
_ALLOWED_LANDING_LINES: Counter[str] = Counter(
    {
        "    .landing-card-token {": 1,
        "      align-self: flex-start;": 1,
        "      max-width: 100%;": 2,
        "      margin: 0.05rem 0 0.1rem;": 1,
        "      padding: 0.12rem 0.55rem;": 1,
        "      border: 1px solid var(--border-default);": 1,
        "      border-radius: 999px;": 1,
        "      font-size: 0.68rem;": 1,
        "      font-weight: 600;": 1,
        "      letter-spacing: 0.05em;": 1,
        "      text-transform: uppercase;": 1,
        "      color: var(--text-secondary);": 2,
        "      overflow: hidden;": 1,
        "      text-overflow: ellipsis;": 1,
        "      white-space: nowrap;": 1,
        "    }": 3,
        "    .landing-card-token-hint {": 1,
        "      font-size: 0.72rem;": 1,
        "      line-height: 1.35;": 1,
        "    .landing-card-token-hint code {": 1,
        '      font-family: ui-monospace, "JetBrains Mono", monospace;': 1,
        "      word-break: break-all;": 1,
        '            <span class="landing-card-token">login link</span>': 2,
        (
            '            <span class="landing-card-token-hint">entered via a login link'
            " — <code>osprey users login-url alice</code></span>"
        ): 1,
        (
            '            <span class="landing-card-token-hint">entered via a login link'
            " — <code>osprey users login-url bob</code></span>"
        ): 1,
    }
)

_ALLOWED = {
    "docker-compose.web.yml": _ALLOWED_COMPOSE_LINES,
    "nginx.conf": _ALLOWED_NGINX_LINES,
    "landing.html": _ALLOWED_LANDING_LINES,
}


def _baseline(name: str) -> str:
    """The frozen pre-feature artifact, with its one per-checkout sentinel resolved."""
    return (_BASELINE_DIR / name).read_text().replace(_REPO_ID_SENTINEL, _rendered_repo_id())


def _token_render() -> dict[str, str]:
    """Today's render of the SC6 shape: no ``auth:``, no ``authorization:``.

    The absent stanza is the ``token`` posture, because that is the default
    `_auth_tls_context` supplies. The explicit spelling is
    :func:`_explicit_token_render`, and the two are pinned equal below.
    """
    return render_web_terminals(EXAMPLE_CONFIG)


def _explicit_token_render() -> dict[str, str]:
    """The same posture written out: ``auth: {method: token}``, roles off."""
    explicit = copy.deepcopy(EXAMPLE_CONFIG)
    explicit["modules"]["web_terminals"]["auth"] = {"method": "token"}
    return render_web_terminals(explicit)


def _is_comment(line: str) -> bool:
    """True for a comment or blank line in the two ``#`` dialects (compose, nginx)."""
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _comment_mask(lines: list[str]) -> list[bool]:
    """Which of *lines* are comment or blank, for the artifact they came from.

    The third artifact is a stylesheet inside a page, and CSS comments SPAN
    lines: ``/*`` opens one and ``*/`` closes it, with prose in between that
    starts with an ordinary word. A per-line predicate cannot see that, so the
    block state is carried down the artifact here instead. Without it the
    interior of a rule's explanatory comment reads as inserted directives, and
    the allowlist below would have to pin prose line by line — the one thing
    :func:`_assert_added_directives_are_exactly_allowed` exists to avoid.

    Args:
        lines: One whole artifact's lines, in order — not a slice, since the
            state is only correct when the opener was seen.

    Returns:
        One boolean per input line, true where the line is blank, a ``#``
        comment, or any part of a ``/* … */`` block including its delimiters.
    """
    mask: list[bool] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if in_block:
            mask.append(True)
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("/*"):
            mask.append(True)
            in_block = "*/" not in stripped[2:]
            continue
        mask.append(_is_comment(line))
    return mask


def _opcodes(
    name: str, artifacts: dict[str, str] | None = None
) -> tuple[list[str], list[str], list[tuple]]:
    """Baseline lines, current lines, and the line-level edit script between them.

    Both sides are port-masked first (see the module docstring), so the edit
    script reports structure and never renumbering.

    ``artifacts`` defaults to the absent-stanza render; pass the explicit-token
    render to hold that spelling to the same frozen baseline.
    """
    rendered = _token_render() if artifacts is None else artifacts
    old = _mask_ports(_baseline(name)).splitlines()
    new = _mask_ports(rendered[_ARTIFACTS[name]]).splitlines()
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
    for line in _REWORDED_COMPOSE_LINES:
        assert line in compose, (
            f"the frozen baseline lacks the pre-layout comment it is said to carry: {line!r}"
        )

    nginx = (_BASELINE_DIR / "nginx.conf").read_text()
    assert "X-Osprey-Auth-Subject" not in nginx
    assert "X-Osprey-Auth-Role" not in nginx

    _assert_the_frozen_ports_predate_the_layout()


def _assert_the_frozen_ports_predate_the_layout() -> None:
    """The same anti-tamper argument, for the one axis the port mask hides.

    Every marker above is one the diffs still see, so a refreshed baseline is
    caught there. The ports are the exception: :func:`_mask_ports` blanks them
    on both sides, so a refreshed baseline's renumbered ports would slip
    through every comparison in this module unnoticed. They are therefore read
    back unmasked here, and the claim is the strongest one available without
    naming a single old number — the frozen copy spells ports the layout does
    not produce for this config, and today's render spells only ports it does.
    A copy regenerated from the current renderer fails both halves.
    """
    live = frozenset().union(*(_ports_in(text) for text in _token_render().values()))
    assert live, "today's render spells no port at all — the mask has stopped matching anything"
    assert live <= _LAYOUT_PORTS, (
        f"today's `token` render spells port(s) the layout does not give this config: "
        f"{sorted(live - _LAYOUT_PORTS)}. Either a port escaped the layout, or "
        f"`_RENDERED_FAMILIES` no longer names every family this config renders."
    )

    frozen = frozenset().union(
        *(_ports_in((_BASELINE_DIR / name).read_text()) for name in _ARTIFACTS)
    )
    assert frozen, "the frozen baseline spells no port at all — it is not the artifact it claims"
    assert frozen.isdisjoint(_LAYOUT_PORTS), (
        f"the frozen baseline spells port(s) only the CURRENT layout produces: "
        f"{sorted(frozen & _LAYOUT_PORTS)}. `golden/pre_audit_roles/` predates the "
        f"layout and must never be regenerated from the current renderer — see the "
        f"module docstring."
    )


def test_the_frozen_golden_holds_todays_token_bytes_under_both_spellings() -> None:
    """Why ``token`` is the default, stated as a test rather than as an argument.

    ``EXAMPLE_CONFIG`` omits ``auth:`` entirely, and `_auth_tls_context` defaults
    the missing block to ``"token"`` — so the first assertion is that the written
    and the defaulted spelling are one posture, not two. It is pinned rather than
    assumed because everything else in this module renders the DEFAULTED
    spelling: if the two ever forked, the whole allowlist would be guarding only
    one of them.

    The loop then holds the EXPLICIT spelling to the frozen pre-feature golden
    through the same allowlist, which proves one direction only — that this
    render ADDS nothing beyond the two SC6 exceptions.
    `test_no_pre_feature_line_is_removed_or_reworded` supplies the other
    direction (nothing is removed or reworded) for the defaulted spelling, and
    the asserted equality above is what extends that half to this one.

    Together they say the golden is unchanged. It was copied out of a commit
    where the only posture that could produce it was ``auth.method: none`` — so
    the bytes ``none`` shipped are the bytes ``token`` renders, and defaulting
    the absent stanza to ``token`` hands a facility a posture that has been
    running all along.

    When this fails, read WHICH half failed. A failed equality means the
    explicit and defaulted paths through `_auth_tls_context` have forked. A
    failed loop means the ``token`` render grew a line — most likely because
    ``none``'s open-mode secret injection reached a branch that is not gated on
    ``inject_secret``, which is a bug in the template, not in this pin.
    """
    explicit = _explicit_token_render()
    assert explicit == _token_render()

    for name in _ARTIFACTS:
        _assert_added_directives_are_exactly_allowed(name, explicit)


def test_no_pre_feature_line_is_removed_or_reworded() -> None:
    """Half of "byte-identical EXCEPT": the feature may only ADD. Every baseline
    line — directives, comments and blanks alike — must still be there, in
    order. A reworded comment, a re-indented directive or a dropped blank line
    fails here even though the allowlist below would never see it (it inspects
    insertions only)."""
    for name in _ARTIFACTS:
        old, new, opcodes = _opcodes(name)
        replaceable = (
            _REPLACED_COMPOSE_LINES | _REWORDED_COMPOSE_LINES
            if name == "docker-compose.web.yml"
            else frozenset()
        )
        lost = [
            (tag, old[i1:i2], new[j1:j2])
            for tag, i1, i2, j1, j2 in opcodes
            if tag in ("delete", "replace") and not set(old[i1:i2]) <= replaceable
        ]
        assert not lost, (
            f"{name}: the `token` render no longer contains lines the pre-feature "
            f"render had. SC6 permits additions only: {lost}"
        )


def test_landing_page_adds_exactly_the_token_login_badge() -> None:
    """The landing page's whole SC6 exception: the badge, and nothing else.

    This assertion used to be byte identity — the strictest form of SC6, back
    when this artifact had no exception at all. It has one now (exception 4 in
    the module docstring), and the change was deliberate: the page's silence
    about `token` was not fidelity to the pre-feature render, it was the page
    telling an operator that every terminal is one click away when not one of
    them is reachable without a URL `osprey users login-url` mints. The bytes it
    was pinned to were dishonest bytes.

    What the pin becomes is the same guarantee one step weaker and stated
    exactly: today's `token` landing page is the frozen one PLUS the badge's
    style rules and the two spans per user card, and nothing at all besides.
    Anything else that appears on this page — a role, a login form, a claim —
    still fails here, which is what SC6 was ever protecting.
    """
    _assert_added_directives_are_exactly_allowed("landing.html")


def test_compose_adds_exactly_the_audit_emitters_and_mounts() -> None:
    """The compose overlay's whole SC6 exception: two env emitters and one bind
    per identity, from task 3.1. Set-with-multiplicity equality both ways, so a
    stray new line fails AND a silently dropped audit mount fails."""
    _assert_added_directives_are_exactly_allowed("docker-compose.web.yml")


def test_nginx_adds_exactly_the_four_identity_header_clears() -> None:
    """The nginx config's whole SC6 exception: the four clears, once each in
    each of the two ungated `/u/<user>/` locations, from task 4.7. Nothing from
    the gated arm may appear here — no `auth_request_set`, no forward — and the
    count catches a clear that reached only one of the two locations."""
    _assert_added_directives_are_exactly_allowed("nginx.conf")


def test_no_authorization_vocabulary_reaches_a_roles_off_render() -> None:
    """The intent behind the allowlist, stated directly. A deployment that
    declared no ``authorization:`` block must carry no role machinery in any
    artifact: not a role claim, not a role map, not a sidecar recheck. This
    catches a whole class of leak in one assertion, including in the comment
    text the line allowlist deliberately ignores."""
    artifacts = _token_render()
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


def test_the_token_render_has_no_auth_sidecar_service_at_all() -> None:
    """The structural reason the allowlist above stays short, pinned so it cannot
    quietly stop being true.

    Everything the sidecar carries — its password hashes, its mapped subjects,
    its OIDC settings, its role claim/map, its per-user
    ``OSPREY_AUTH_ROSTER_ROLE_<suffix>`` bindings — lives inside a service that
    ``auth.method: token`` never renders (only ``password``/``oidc`` stand one
    up). So work on the sidecar's environment block cannot move this render, and
    lines from it must never be added to the SC6 allowlist: the allowlist asserts
    equality in BOTH directions, so an entry that never renders here would fail
    as a missing exception rather than pass as a permitted one.

    If this test ever fails, a deployment that asked for no login wall has grown
    a login service anyway, and that is the finding — not the allowlist's
    shortness."""
    compose = yaml.safe_load(_token_render()["docker-compose.web.yml"])

    assert set(compose["services"]) == {"nginx", "web-alice", "web-bob"}


def _assert_added_directives_are_exactly_allowed(
    name: str, artifacts: dict[str, str] | None = None
) -> None:
    """Every inserted non-comment line is in the allowlist for ``name``, at the
    expected multiplicity, and every allowlisted line was actually inserted.

    ``artifacts`` defaults to the absent-stanza render; the explicit-token
    spelling is held to the same allowlist by
    `test_the_frozen_golden_holds_todays_token_bytes_under_both_spellings`.

    Comments are excluded on purpose: pinning prose line by line would duplicate
    the golden fixture without adding a guarantee, and reworded prose is already
    caught by `test_no_pre_feature_line_is_removed_or_reworded` (which sees
    comments) plus `test_no_authorization_vocabulary_reaches_a_roles_off_render`
    (which reads the comment text for the vocabulary that must not appear).
    """
    _old, new, opcodes = _opcodes(name, artifacts)
    is_comment = _comment_mask(new)
    inserted = Counter(
        new[j]
        for tag, _i1, _i2, j1, j2 in opcodes
        if tag in ("insert", "replace")
        for j in range(j1, j2)
        if not is_comment[j]
    )

    unexpected = inserted - _ALLOWED[name]
    missing = _ALLOWED[name] - inserted
    assert not unexpected, (
        f"{name}: line(s) added to the `token` render that SC6 does not allow. "
        f"Add them to the allowlist only if they genuinely belong in a render "
        f"that asked for neither logins nor roles: {sorted(unexpected.elements())}"
    )
    assert not missing, (
        f"{name}: SC6's allowed exception(s) are no longer being rendered — the "
        f"audit trail or the identity-header clears went missing: "
        f"{sorted(missing.elements())}"
    )
