"""Contract tests for :mod:`osprey.interfaces.web_auth`.

The module under test is the only place the web surfaces decide who may talk to
them, so the properties pinned here are the ones whose quiet failure would
re-open the escalation path rather than break a test: that the operator secret
LEAVES ``os.environ`` when it is read, that the container shape refuses to mint
a secret nginx does not know, that an empty environment value is treated as no
credential at all, and that every app in one process ends up with the SAME
credentials rather than each minting its own.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette

from osprey.interfaces import web_auth
from osprey.interfaces.web_auth import (
    BIND_HOST_ENV,
    OPERATOR_SECRET_ENV,
    PANEL_TOKEN_ENV,
    WebCredentials,
    get_web_credentials,
    mint_secret,
    reset_web_credentials,
)

#: Length of ``secrets.token_urlsafe(32)``, the minting recipe every absent
#: credential falls back to.
_MINTED_LENGTH = 43


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch: pytest.MonkeyPatch):
    """Give each test an unpopulated holder and an environment with no carriers.

    Population is process-wide and one-shot by design, so without this the
    first test to run would decide the credentials for every test after it and
    the environment-driven paths below would never execute. ``delenv`` also
    hands teardown the job of restoring anything the module's ``pop`` removed.
    """
    for var in (OPERATOR_SECRET_ENV, PANEL_TOKEN_ENV, BIND_HOST_ENV):
        monkeypatch.delenv(var, raising=False)
    reset_web_credentials()
    yield
    reset_web_credentials()


def _fake_app() -> SimpleNamespace:
    """A stand-in carrying only the ``app.state`` attribute the module touches."""
    return SimpleNamespace(state=SimpleNamespace())


# ---------------------------------------------------------------------------
# Population: reading the environment
# ---------------------------------------------------------------------------


def test_operator_secret_is_popped_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The secret must LEAVE ``os.environ``, or the sandbox inherits it.

    This is the whole point of the module: the agent SDK overlays its own
    environment on top of ``os.environ`` when it spawns the sandboxed child, so
    a secret still sitting there at construction time is handed straight to the
    process the credential exists to keep out.
    """
    import os

    monkeypatch.setenv(OPERATOR_SECRET_ENV, "supplied-operator-secret")

    credentials = get_web_credentials()

    assert credentials.operator_secret == "supplied-operator-secret"
    assert OPERATOR_SECRET_ENV not in os.environ


def test_panel_token_is_popped_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The panel token leaves the environment for the same reason."""
    import os

    monkeypatch.setenv(PANEL_TOKEN_ENV, "supplied-panel-token")

    credentials = get_web_credentials()

    assert credentials.panel_token == "supplied-panel-token"
    assert PANEL_TOKEN_ENV not in os.environ


def test_supplied_values_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrounding whitespace in a ``.env`` value is not part of the credential."""
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "  padded-secret\n")
    monkeypatch.setenv(PANEL_TOKEN_ENV, "\tpadded-panel ")

    credentials = get_web_credentials()

    assert credentials.operator_secret == "padded-secret"
    assert credentials.panel_token == "padded-panel"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_operator_secret_counts_as_absent(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """An unset compose variable interpolates to empty; that must not authorise anyone.

    Were the empty string accepted as a credential, every caller who also sent
    nothing would compare equal to it.
    """
    monkeypatch.setenv(OPERATOR_SECRET_ENV, blank)

    credentials = get_web_credentials()

    assert credentials.operator_secret not in ("", blank)
    assert len(credentials.operator_secret) == _MINTED_LENGTH
    assert credentials.verify_operator("") is False
    assert credentials.verify_operator(blank) is False


def test_blank_panel_token_counts_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same rule for the panel token."""
    monkeypatch.setenv(PANEL_TOKEN_ENV, "   ")

    credentials = get_web_credentials()

    assert len(credentials.panel_token) == _MINTED_LENGTH
    assert credentials.verify_panel("   ") is False


def test_single_user_shape_mints_both_credentials() -> None:
    """With nothing in the environment and no declared bind host, mint."""
    credentials = get_web_credentials()

    assert len(credentials.operator_secret) == _MINTED_LENGTH
    assert len(credentials.panel_token) == _MINTED_LENGTH
    assert credentials.operator_secret != credentials.panel_token


# ---------------------------------------------------------------------------
# Population: the container shape refuses to mint
# ---------------------------------------------------------------------------


def test_container_shape_without_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared bind host with no supplied secret is fatal, not a mint.

    nginx forwards the value the deploy ``.env`` pinned. A locally minted
    secret would never match it, so the process would come up healthy and
    refuse every request that reached it.
    """
    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")

    with pytest.raises(RuntimeError) as excinfo:
        get_web_credentials()

    message = str(excinfo.value)
    assert OPERATOR_SECRET_ENV in message, "the error must name the missing compose variable"
    assert BIND_HOST_ENV in message, "the error must name the signal that made it fatal"


def test_container_shape_blank_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty compose interpolation is 'absent' here too, and so is fatal."""
    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "  ")

    with pytest.raises(RuntimeError):
        get_web_credentials()


def test_container_shape_failure_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second call must raise too, rather than mint on the now-empty environment.

    Population popped the (blank) variable on the first attempt. If the failure
    left the holder in a state where a retry took the minting path, a second
    request would quietly bring up a server nginx can never authenticate
    against — exactly the outcome the first call refused.
    """
    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "")

    with pytest.raises(RuntimeError):
        get_web_credentials()
    with pytest.raises(RuntimeError):
        get_web_credentials()


def test_container_shape_failure_still_consumes_the_panel_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fatal path must not be the one path that leaves a credential in the environment.

    A caller may catch this ``RuntimeError`` and keep serving; the process can
    still spawn the sandboxed child. A panel token left in ``os.environ`` at
    that point is inherited by exactly the process this module exists to keep
    out, so both carriers are popped before the check that raises.
    """
    import os

    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(PANEL_TOKEN_ENV, "supplied-panel-token")

    with pytest.raises(RuntimeError):
        get_web_credentials()

    assert PANEL_TOKEN_ENV not in os.environ
    assert OPERATOR_SECRET_ENV not in os.environ


def test_container_shape_with_supplied_secret_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment's value is used verbatim; only the panel token is minted."""
    monkeypatch.setenv(BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "from-deploy-env")

    credentials = get_web_credentials()

    assert credentials.operator_secret == "from-deploy-env"
    assert len(credentials.panel_token) == _MINTED_LENGTH


def test_bind_host_is_not_popped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``osprey web`` reads the bind host too — consuming it would unbind the server."""
    import os

    monkeypatch.setenv(BIND_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "from-deploy-env")

    get_web_credentials()

    assert os.environ[BIND_HOST_ENV] == "127.0.0.1"


# ---------------------------------------------------------------------------
# Idempotence and sharing
# ---------------------------------------------------------------------------


def test_population_is_idempotent() -> None:
    """Calling twice returns the same object, not a second set of secrets."""
    first = get_web_credentials()
    second = get_web_credentials()

    assert first is second


def test_reset_repopulates(monkeypatch: pytest.MonkeyPatch) -> None:
    """``reset_web_credentials`` is what makes the environment paths testable."""
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "first-secret")
    first = get_web_credentials()

    reset_web_credentials()
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "second-secret")
    second = get_web_credentials()

    assert first.operator_secret == "first-secret"
    assert second.operator_secret == "second-secret"
    assert first is not second


def test_credentials_are_cached_on_app_state() -> None:
    """The middleware reads ``app.state.web_credentials``; put them there."""
    app = _fake_app()

    credentials = get_web_credentials(app)

    assert app.state.web_credentials is credentials
    assert get_web_credentials(app) is credentials


def test_real_starlette_app_state_carries_credentials() -> None:
    """The same against a real ``Starlette`` app, not just the stand-in."""
    app = Starlette()

    credentials = get_web_credentials(app)

    assert app.state.web_credentials is credentials


def test_companion_apps_inherit_the_process_default() -> None:
    """Two apps in one process share credentials, or the cookie only works on one.

    An in-process companion app that minted its own secret would refuse the
    session cookie the terminal handed the browser.
    """
    terminal = _fake_app()
    companion = _fake_app()

    assert get_web_credentials(terminal) is get_web_credentials(companion)
    assert get_web_credentials() is terminal.state.web_credentials


def test_app_state_holding_a_non_credential_is_ignored() -> None:
    """A junk value on ``app.state`` must not be handed to the middleware as credentials."""
    app = _fake_app()
    app.state.web_credentials = "not-a-credential-holder"

    credentials = get_web_credentials(app)

    assert isinstance(credentials, WebCredentials)
    assert app.state.web_credentials is credentials


def test_app_without_state_still_resolves() -> None:
    """A caller passing something stateless gets the process default, not an error."""
    stateless = object()

    assert get_web_credentials(stateless) is get_web_credentials()


def test_concurrent_first_use_populates_once() -> None:
    """Threads racing on the first request must not end up with different secrets.

    Two workers minting under an unguarded read would each serve a different
    secret, and a browser holding one would be refused by the other.
    """
    results: list[WebCredentials] = []
    barrier = threading.Barrier(8)

    def _resolve() -> None:
        barrier.wait()
        results.append(get_web_credentials())

    threads = [threading.Thread(target=_resolve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(item is results[0] for item in results)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@pytest.fixture
def credentials() -> WebCredentials:
    """A holder with known, distinct secrets and no environment involvement."""
    return WebCredentials(operator_secret="operator-value", panel_token="panel-value")


def test_verify_operator_accepts_only_the_operator_secret(credentials: WebCredentials) -> None:
    """The panel token must not open an operator door."""
    assert credentials.verify_operator("operator-value") is True
    assert credentials.verify_operator("panel-value") is False
    assert credentials.verify_operator("operator-valu") is False
    assert credentials.verify_operator("operator-value ") is False


def test_verify_panel_accepts_only_the_panel_token(credentials: WebCredentials) -> None:
    """And the operator secret is not rejected here — it is simply a different tier's key."""
    assert credentials.verify_panel("panel-value") is True
    assert credentials.verify_panel("operator-value") is False


@pytest.mark.parametrize("missing", [None, ""])
def test_absent_candidates_are_refused(credentials: WebCredentials, missing: str | None) -> None:
    """A request with no credential is a refusal, not a crash."""
    assert credentials.verify_operator(missing) is False
    assert credentials.verify_panel(missing) is False
    assert credentials.verify_session(missing) is False


def test_non_ascii_candidate_is_refused_not_raised(credentials: WebCredentials) -> None:
    """``compare_digest`` raises on non-ASCII ``str``; the bytes encoding is why it does not here.

    Headers and cookies are attacker-controlled, so one accented character
    would otherwise be an unhandled 500 rather than a 401.
    """
    assert credentials.verify_operator("öperator-value") is False
    assert credentials.verify_panel("pänel-value") is False
    assert credentials.verify_session("sessiön") is False


# ---------------------------------------------------------------------------
# Browser sessions
# ---------------------------------------------------------------------------


def test_session_round_trip(credentials: WebCredentials) -> None:
    """A minted session verifies; an unrelated value of the same shape does not."""
    session_id = credentials.create_session()

    assert len(session_id) == _MINTED_LENGTH
    assert credentials.verify_session(session_id) is True
    assert credentials.verify_session(mint_secret()) is False


def test_sessions_are_distinct(credentials: WebCredentials) -> None:
    """Two browsers get two ids, and both stay valid."""
    first = credentials.create_session()
    second = credentials.create_session()

    assert first != second
    assert credentials.verify_session(first) is True
    assert credentials.verify_session(second) is True


def test_expired_session_is_refused(credentials: WebCredentials) -> None:
    """A deadline in the past is a refusal without waiting for a reaper to run."""
    session_id = credentials.create_session(ttl_seconds=0)

    assert credentials.verify_session(session_id) is False


def test_expired_sessions_are_purged(credentials: WebCredentials) -> None:
    """The map must not grow without bound as tabs come and go."""
    credentials.create_session(ttl_seconds=0)
    credentials.create_session(ttl_seconds=0)
    live = credentials.create_session()

    assert credentials.verify_session(live) is True
    assert list(credentials.sessions) == [live]


def test_purge_happens_on_create_too(credentials: WebCredentials) -> None:
    """Creating a session is the call whose frequency tracks the map's growth."""
    credentials.create_session(ttl_seconds=0)
    assert len(credentials.sessions) == 1

    credentials.create_session()

    assert len(credentials.sessions) == 1


def test_revoke_session_invalidates_the_cookie(credentials: WebCredentials) -> None:
    """Logout has to refuse a cookie value that has already left the process."""
    session_id = credentials.create_session()

    assert credentials.revoke_session(session_id) is True
    assert credentials.verify_session(session_id) is False
    assert credentials.revoke_session(session_id) is False


def test_the_cost_equalizing_decoy_is_not_a_credential(credentials: WebCredentials) -> None:
    """A candidate equal to the miss-path decoy must still be refused.

    The decoy exists only so a miss costs what a hit costs. If the answer were
    read off the comparison alone, sending the decoy's own value would compare
    equal to it and authenticate against an EMPTY session map — and a NUL run
    survives a ``%00`` query string, a JSON body and a websocket frame intact,
    so it is a value an attacker can actually deliver to the token exchange
    this map serves.
    """
    forged = "\x00" * 43

    assert credentials.verify_session(forged) is False
    assert credentials.sessions == {}

    # Still refused once the map is non-empty, so a live session cannot be the
    # thing that makes the forgery work either.
    credentials.create_session()
    assert credentials.verify_session(forged) is False


@pytest.mark.parametrize("length", [1, 42, 44, 200])
def test_wrong_length_forgeries_are_refused(credentials: WebCredentials, length: int) -> None:
    """A NUL run of any other length is refused too — the decoy's length is not a key."""
    credentials.create_session()

    assert credentials.verify_session("\x00" * length) is False


def test_a_secret_is_not_a_session(credentials: WebCredentials) -> None:
    """The three credentials are separate keyspaces; none substitutes for another."""
    session_id = credentials.create_session()

    assert credentials.verify_session("operator-value") is False
    assert credentials.verify_operator(session_id) is False
    assert credentials.verify_panel(session_id) is False


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_repr_does_not_leak_secrets(credentials: WebCredentials) -> None:
    """Tracebacks, debuggers and log lines must not print the credentials."""
    credentials.create_session()
    rendered = repr(credentials)

    assert "operator-value" not in rendered
    assert "panel-value" not in rendered
    assert "sessions=1" in rendered


def test_mint_secret_is_unique_and_url_safe() -> None:
    """The minted value travels in a query string and a ``.env`` without escaping."""
    minted = {mint_secret() for _ in range(50)}

    assert len(minted) == 50
    for value in minted:
        assert len(value) == _MINTED_LENGTH
        assert set(value) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_module_exports_the_documented_surface() -> None:
    """Later tasks import these names; ``__all__`` is the contract they build on."""
    for name in web_auth.__all__:
        assert hasattr(web_auth, name), f"__all__ names {name}, which the module does not define"


# ---------------------------------------------------------------------------
# Route tiers
# ---------------------------------------------------------------------------

#: Every ``(method, path)`` the table is expected to place in the panel tier
#: unconditionally. Spelled out here rather than imported from the module so a
#: widening of :data:`web_auth.PANEL_TIER_ROUTES` fails this file instead of
#: agreeing with itself.
_EXPECTED_PANEL_ROUTES = {
    ("GET", "/api/panels"),
    ("POST", "/api/panel-focus"),
    ("POST", "/api/panel-visibility"),
    ("POST", "/api/panel-close"),
    ("POST", "/api/panel-arrange"),
    ("POST", "/api/agent-activity"),
    ("POST", "/api/focus"),
}

#: Routes that must stay operator-only. Each is a real route in the tree (see
#: ``web_terminal/routes/`` and ``artifacts/app.py``) picked because a panel
#: token reaching it would be an escalation: config writes, scaffold creation,
#: memory writes, chat, feedback, the panel proxy, and both websockets.
_EXPECTED_OPERATOR_ROUTES = [
    ("GET", "/api/config"),
    ("PUT", "/api/config"),
    ("PATCH", "/api/config"),
    ("GET", "/api/claude-setup"),
    ("PUT", "/api/claude-setup"),
    ("POST", "/api/claude-setup"),
    ("GET", "/api/scaffold"),
    ("POST", "/api/scaffold/create"),
    ("POST", "/api/scaffold/untracked/register"),
    ("GET", "/api/claude-memory"),
    ("POST", "/api/claude-memory"),
    ("PUT", "/api/claude-memory/CLAUDE.md"),
    ("DELETE", "/api/claude-memory/CLAUDE.md"),
    ("POST", "/api/chat"),
    ("DELETE", "/api/chat/abc"),
    ("POST", "/api/feedback"),
    ("POST", "/api/feedback/bundle"),
    ("POST", "/api/terminal/restart"),
    ("POST", "/api/terminal/logout"),
    ("POST", "/api/panel-layout"),
    ("GET", "/api/files/tree"),
    ("GET", "/api/hooks/debug-log"),
    ("GET", "/api/session-log"),
    ("GET", "/api/mcp-servers"),
    ("GET", "/panel/events/"),
    ("POST", "/panel/events/api/thing"),
    ("GET", "/ws/terminal"),
    ("GET", "/ws/operator"),
    ("DELETE", "/api/artifacts/abc"),
    ("GET", "/api/artifacts"),
    ("POST", "/api/artifacts/abc/pin"),
]


def test_route_tiers() -> None:
    """The panel tier holds exactly the documented routes, and nothing else.

    This is the C5 proof in miniature: the panel token is a weaker credential
    handed to in-process companions, so every route it unlocks is a route the
    agent's own sandbox can drive. A route that lands in this tier by accident
    is a privilege escalation, which is why the expectation is an equality
    against a hand-written set rather than a membership check.
    """
    assert set(web_auth.PANEL_TIER_ROUTES) == _EXPECTED_PANEL_ROUTES

    for method, path in _EXPECTED_PANEL_ROUTES:
        assert web_auth.classify(method, path, False) is web_auth.Tier.PANEL, (
            f"{method} {path} should be panel-tier"
        )

    for method, path in _EXPECTED_OPERATOR_ROUTES:
        assert web_auth.classify(method, path, False) is web_auth.Tier.OPERATOR, (
            f"{method} {path} must be operator-only"
        )
        assert web_auth.classify(method, path, True) is web_auth.Tier.OPERATOR


def test_panel_register_is_panel_tier_only_without_a_url() -> None:
    """URL-backed registration points the proxy at an arbitrary host — operator-only.

    A registration carrying a ``url`` decides where the terminal's own proxy
    forwards operator traffic, so it is not something the weak credential may
    do; a registration without one cannot repoint anything.
    """
    assert web_auth.classify("POST", "/api/panels/register", False) is web_auth.Tier.PANEL
    assert web_auth.classify("POST", "/api/panels/register", True) is web_auth.Tier.OPERATOR
    assert web_auth.PANEL_REGISTER_ROUTE == ("POST", "/api/panels/register")


def test_panel_register_route_is_not_in_the_unconditional_set() -> None:
    """The conditional route must not also sit in the flat table.

    If it did, the ``body_has_url`` branch would be dead and every
    registration — URL-backed included — would clear the panel tier.
    """
    assert web_auth.PANEL_REGISTER_ROUTE not in web_auth.PANEL_TIER_ROUTES


@pytest.mark.parametrize(
    "method",
    ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
def test_panels_listing_is_panel_tier_only_for_get(method: str) -> None:
    """The table keys on the method too: only ``GET /api/panels`` is readable-tier."""
    assert web_auth.classify(method, "/api/panels", False) is web_auth.Tier.OPERATOR


def test_reading_panel_focus_is_operator_only() -> None:
    """``GET /api/panel-focus`` is a real route the plan did not grant.

    Pinned because the ``POST`` sibling *is* panel-tier, which makes this the
    likeliest place for a future edit to widen the tier by pattern-matching on
    the path alone.
    """
    assert web_auth.classify("GET", "/api/panel-focus", False) is web_auth.Tier.OPERATOR


@pytest.mark.parametrize(
    "path",
    [
        "/api/panels/",
        "/api/panels/register/",
        "/API/PANELS",
        "/api/panels?x=1",
        "/api/panels/foo",
        "/u/alice/api/panels",
        "//api/panels",
        "/api/panel-focus/../config",
        " /api/panels",
        "/api/panels ",
        "",
    ],
)
def test_near_miss_paths_fail_closed(path: str) -> None:
    """Anything that is not the exact path is operator-only.

    The match is exact on purpose. A prefix or regex match would let
    ``/api/panels/register`` — or a proxied ``/u/<user>`` form, or a
    query-string-bearing spelling — inherit the weak tier. Paths arrive bare
    and already normalised by the ASGI server, so a spelling that misses here
    is either not a real route or a route the operator credential covers.
    """
    assert web_auth.classify("GET", path, False) is web_auth.Tier.OPERATOR
    assert web_auth.classify("POST", path, False) is web_auth.Tier.OPERATOR


@pytest.mark.parametrize("method", ["get", "Get", "gEt"])
def test_method_case_is_normalised(method: str) -> None:
    """ASGI promises an uppercase method; normalising costs nothing and fails safe."""
    assert web_auth.classify(method, "/api/panels", False) is web_auth.Tier.PANEL


@pytest.mark.parametrize(
    ("method", "path"),
    [(None, "/api/panels"), ("GET", None), (None, None), ("", "/api/panels"), ("GET", "")],
)
def test_missing_method_or_path_is_operator_only(method: str | None, path: str | None) -> None:
    """A scope missing either field is refused rather than trusted.

    A websocket scope carries no ``method``; the middleware supplies one, and
    if it ever supplies nothing the answer must be the strong credential.
    """
    assert web_auth.classify(method, path, False) is web_auth.Tier.OPERATOR


def test_body_has_url_has_no_default() -> None:
    """The flag is required, so no caller can grant registration by forgetting it.

    A ``False`` default would make an omitted argument the permissive answer;
    a ``True`` default would be a parameter whose name contradicts its value.
    Pinned as a ``TypeError`` so a later edit cannot quietly add either.
    """
    with pytest.raises(TypeError):
        web_auth.classify("POST", "/api/panels/register")  # type: ignore[call-arg]


def test_panel_tier_table_is_immutable() -> None:
    """The table is a frozenset, so no importer can widen the tier at runtime."""
    assert isinstance(web_auth.PANEL_TIER_ROUTES, frozenset)
    with pytest.raises(AttributeError):
        web_auth.PANEL_TIER_ROUTES.add(("POST", "/api/config"))  # type: ignore[attr-defined]


def test_tier_members_are_distinct_and_named() -> None:
    """The middleware branches on these two members; nothing else exists."""
    assert {member.name for member in web_auth.Tier} == {"PANEL", "OPERATOR"}
    assert web_auth.Tier.PANEL is not web_auth.Tier.OPERATOR


# ---------------------------------------------------------------------------
# One-time URL announcement
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_announced_carrier():
    """Undo the environment variable ``mint_and_announce`` deliberately publishes.

    The module's own population *pops* the carrier, so ``monkeypatch`` has
    nothing recorded for it and would not clean up a value a test caused the
    product code to write. Without this teardown the first announcement in this
    file would leak a real operator secret into ``os.environ`` for every test
    that runs after it, in this file and any other in the same worker.
    """
    import os

    yield
    os.environ.pop(OPERATOR_SECRET_ENV, None)


def test_mint_and_announce() -> None:
    """The announced URL carries the process's own operator secret, once and stably.

    This is the whole single-user entry path in one assertion block: the
    operator's only way in is the URL printed here, the token in it must be the
    credential the serving process will check, and the value must reach a child
    process through the environment because under ``--reload`` and ``--detach``
    the process that prints is not the process that answers.
    """
    import os
    import urllib.parse

    url = web_auth.mint_and_announce("127.0.0.1", 8765)

    parsed = urllib.parse.urlsplit(url)
    assert parsed.scheme == "http"
    assert parsed.netloc == "127.0.0.1:8765"
    assert parsed.path == "/"

    # The token is the operator secret itself, and survives the round trip
    # through query-string quoting that a browser performs.
    token = urllib.parse.parse_qs(parsed.query)["token"][0]
    credentials = get_web_credentials()
    assert token == credentials.operator_secret
    assert credentials.verify_operator(token)

    # Idempotent per process: a second entry point, or a retry after a port
    # fallback, announces the same token rather than invalidating the first.
    assert web_auth.mint_and_announce("127.0.0.1", 8765) == url

    # An app built afterwards in this process serves the announced secret.
    app = _fake_app()
    assert get_web_credentials(app).operator_secret == token
    assert app.state.web_credentials.verify_operator(token)

    # The carrier is re-published for the child about to be spawned, and that
    # child — a fresh, unpopulated holder — pops the very same value.
    assert os.environ[OPERATOR_SECRET_ENV] == token
    reset_web_credentials()
    assert get_web_credentials().operator_secret == token
    assert OPERATOR_SECRET_ENV not in os.environ

    # The path is where the token is exchanged, and defaults to the app root.
    other = web_auth.mint_and_announce("127.0.0.1", 8765, path="/static/session.html")
    assert other.startswith("http://127.0.0.1:8765/static/session.html?token=")


def test_announcement_reuses_a_supplied_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """A secret handed in by the environment is announced, not replaced.

    Anything else would print a URL that the credential actually being served
    refuses — the failure mode is a login page that rejects the only link the
    operator was given.
    """
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "supplied-operator-secret")

    url = web_auth.mint_and_announce("127.0.0.1", 8765)

    assert url.endswith("?token=supplied-operator-secret")


def test_announcement_percent_encodes_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment-supplied secret may hold characters a query string reserves.

    A minted secret is URL-safe by construction, but a value out of a deploy
    ``.env`` is not: an unescaped ``&`` or ``#`` would truncate the token the
    browser sends back into something that authenticates nobody.
    """
    import urllib.parse

    monkeypatch.setenv(OPERATOR_SECRET_ENV, "a&b#c d/e")

    url = web_auth.mint_and_announce("127.0.0.1", 8765)

    assert "a&b#c d/e" not in url
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["token"] == ["a&b#c d/e"]


def test_announcement_does_not_break_the_container_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """In the container shape the announcement inherits the refusal to mint.

    ``mint_and_announce`` is a host-side call, but nothing stops a container
    entry point from reaching it. If it minted there it would print a URL
    carrying a secret nginx has never heard of, and every forwarded request
    would be refused with no indication why.
    """
    monkeypatch.setenv(BIND_HOST_ENV, "0.0.0.0")

    with pytest.raises(RuntimeError, match=OPERATOR_SECRET_ENV):
        web_auth.mint_and_announce("0.0.0.0", 8765)


def test_announcement_in_the_container_shape_carries_the_deployment_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the deployment's secret supplied, the container shape announces THAT value.

    The bind-host tell is read and never popped, so the announcement leaves the
    shape of the process exactly as it found it.
    """
    import os

    monkeypatch.setenv(BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "deployment-secret")

    url = web_auth.mint_and_announce("0.0.0.0", 8765)

    assert url == "http://0.0.0.0:8765/?token=deployment-secret"
    assert os.environ[BIND_HOST_ENV] == "0.0.0.0"


@pytest.mark.parametrize(
    ("host", "expected_netloc"),
    [
        ("127.0.0.1", "127.0.0.1:8765"),
        ("0.0.0.0", "0.0.0.0:8765"),
        ("localhost", "localhost:8765"),
        ("::1", "[::1]:8765"),
        ("[::1]", "[::1]:8765"),
    ],
)
def test_ipv6_hosts_are_bracketed(host: str, expected_netloc: str) -> None:
    """``--host`` is honoured verbatim, so an IPv6 literal can reach this function.

    ``http://::1:8765/`` has no unambiguous parse, and a browser handed it does
    not reach the terminal. Bracketing is the only spelling that survives; a
    host that is already bracketed must not be bracketed twice.
    """
    import urllib.parse

    url = web_auth.mint_and_announce(host, 8765)

    assert urllib.parse.urlsplit(url).netloc == expected_netloc


def test_a_relative_path_is_anchored_at_the_root() -> None:
    """A caller that forgets the leading slash still gets a reachable URL.

    Without this, ``path="static/session.html"`` yields
    ``http://host:8765static/session.html``, whose authority is a hostname that
    does not resolve — a broken link rather than a refused one.
    """
    url = web_auth.mint_and_announce("127.0.0.1", 8765, path="static/session.html")

    assert url.startswith("http://127.0.0.1:8765/static/session.html?token=")


def test_announcement_does_not_disturb_the_panel_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the operator carrier is re-published; the panel token stays popped.

    The panel token reaches a companion by an arrangement of its own. Putting
    it back into ``os.environ`` here would hand the weak credential to every
    child of the launcher, including the agent's sandbox.
    """
    import os

    monkeypatch.setenv(PANEL_TOKEN_ENV, "supplied-panel-token")

    web_auth.mint_and_announce("127.0.0.1", 8765)

    assert PANEL_TOKEN_ENV not in os.environ


# ---------------------------------------------------------------------------
# close_env_carriers: the pop that runs on EVERY launch, not just the first
# ---------------------------------------------------------------------------


def test_close_env_carriers_removes_both_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both carriers go, whatever put them back there.

    ``_populate`` pops them once per process; ``mint_and_announce`` publishes
    the operator secret again on every launch, deliberately, so a spawned
    worker or detached child can inherit it. On the direct-serve path there is
    no such child — the launcher becomes the server — and this function is what
    closes the window that re-publication opened.
    """
    import os

    get_web_credentials()  # settle the holder first
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "republished-by-a-launcher")
    monkeypatch.setenv(PANEL_TOKEN_ENV, "republished-panel-token")

    web_auth.close_env_carriers()

    assert OPERATOR_SECRET_ENV not in os.environ
    assert PANEL_TOKEN_ENV not in os.environ


def test_close_env_carriers_does_not_change_the_settled_credentials() -> None:
    """Closing the carriers is not re-population: the held values are untouched."""
    credentials = get_web_credentials()
    before = (credentials.operator_secret, credentials.panel_token)

    web_auth.close_env_carriers()

    after = get_web_credentials()
    assert after is credentials
    assert (after.operator_secret, after.panel_token) == before


def test_close_env_carriers_is_idempotent_and_safe_when_nothing_is_set() -> None:
    """Called with no carriers present it is a no-op, not a KeyError.

    It runs at every app construction, including the many that happen with the
    environment already clean, so raising on absence would turn the common case
    into the failure case.
    """
    web_auth.close_env_carriers()
    web_auth.close_env_carriers()


def test_configure_interface_app_closes_the_carriers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The real seam: constructing any interface app closes both carriers.

    Asserted through ``configure_interface_app`` rather than by calling
    ``close_env_carriers`` directly, because the wiring is the part that was
    missing — the helper existing but never being called would leave the
    default ``osprey web`` launch handing its operator secret to every agent it
    spawns.
    """
    import os

    from osprey.interfaces._app_setup import configure_interface_app

    monkeypatch.setenv(OPERATOR_SECRET_ENV, "supplied-operator-secret")
    monkeypatch.setenv(PANEL_TOKEN_ENV, "supplied-panel-token")

    app = Starlette()
    configure_interface_app(app, static_dir=tmp_path)

    assert OPERATOR_SECRET_ENV not in os.environ
    assert PANEL_TOKEN_ENV not in os.environ
    # The values were taken up by the holder, not merely discarded.
    credentials = get_web_credentials()
    assert credentials.operator_secret == "supplied-operator-secret"
    assert credentials.panel_token == "supplied-panel-token"
    assert app.state.web_credentials is credentials


def test_configure_interface_app_resolves_before_it_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Order matters: popping first would discard a deployment-supplied secret.

    In the container shape nginx forwards the value the deploy ``.env`` pinned.
    A construction that closed the carriers before resolving would leave the
    holder minting a local secret instead, and every proxied request would then
    be refused with nothing to explain it.
    """
    from osprey.interfaces._app_setup import configure_interface_app

    monkeypatch.setenv(BIND_HOST_ENV, "0.0.0.0")
    monkeypatch.setenv(OPERATOR_SECRET_ENV, "the-value-nginx-forwards")

    app = Starlette()
    configure_interface_app(app, static_dir=tmp_path)

    assert get_web_credentials().operator_secret == "the-value-nginx-forwards"
