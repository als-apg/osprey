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

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette

from osprey.interfaces import web_auth
from osprey.interfaces.common_middleware import WEB_PORT_ENV
from osprey.interfaces.web_auth import (
    _SESSION_DECOY,
    BIND_HOST_ENV,
    DEFAULT_SESSION_LIFETIME,
    OPERATOR_SECRET_ENV,
    PANEL_TOKEN_ENV,
    ROSTER_ACCEPT_ENV,
    ROSTER_SECRET_ENV_PREFIX,
    SESSION_LIFETIME_ENV,
    SESSION_STORE_DIR_ENV,
    SessionStore,
    WebCredentials,
    _digest,
    get_web_credentials,
    mint_secret,
    reset_web_credentials,
)

#: Length of ``secrets.token_urlsafe(32)``, the minting recipe every absent
#: credential falls back to. Still the length of the id ``create_session``
#: returns — the map key derived from it is a 64-character digest.
_MINTED_LENGTH = 43


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch: pytest.MonkeyPatch):
    """Give each test an unpopulated holder and an environment with no carriers.

    Population is process-wide and one-shot by design, so without this the
    first test to run would decide the credentials for every test after it and
    the environment-driven paths below would never execute. ``delenv`` also
    hands teardown the job of restoring anything the module's ``pop`` removed.
    """
    for var in (
        OPERATOR_SECRET_ENV,
        PANEL_TOKEN_ENV,
        BIND_HOST_ENV,
        SESSION_LIFETIME_ENV,
        SESSION_STORE_DIR_ENV,
        WEB_PORT_ENV,
    ):
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
# Population: the configured session lifetime
# ---------------------------------------------------------------------------


def test_session_lifetime_defaults_when_nothing_configures_it() -> None:
    """An unset carrier means nothing configured it, which is the default."""
    assert get_web_credentials().session_ttl_seconds == DEFAULT_SESSION_LIFETIME


def test_session_lifetime_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launcher resolves env > config > default and publishes it here."""
    monkeypatch.setenv(SESSION_LIFETIME_ENV, "3600")

    assert get_web_credentials().session_ttl_seconds == 3600


def test_session_lifetime_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value out of a ``.env`` may carry surrounding whitespace."""
    monkeypatch.setenv(SESSION_LIFETIME_ENV, " 60 ")

    assert get_web_credentials().session_ttl_seconds == 60


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_session_lifetime_takes_the_default(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """``${VAR:-}`` of an unset compose variable is 'nothing configured it'.

    This is the one non-numeric value that is not a typo, so it is the one that
    may fall back rather than refuse.
    """
    monkeypatch.setenv(SESSION_LIFETIME_ENV, blank)

    assert get_web_credentials().session_ttl_seconds == DEFAULT_SESSION_LIFETIME


@pytest.mark.parametrize("bad", ["0", "-1", "12h", "1.5", "abc", "1e3", " -0 "])
def test_unreadable_session_lifetime_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """A lifetime that is not a positive whole number of seconds is fatal.

    Substituting the default would hide a config typo on a shared console: the
    deployment would believe it had shortened the lifetime while every terminal
    went on handing out twelve-hour sessions. Zero and negatives are refused
    rather than clamped for the same reason — neither is a lifetime anyone
    means, and a session that expires the instant it is minted is a login page
    that never lets anybody in.
    """
    monkeypatch.setenv(SESSION_LIFETIME_ENV, bad)

    with pytest.raises(RuntimeError) as excinfo:
        get_web_credentials()

    message = str(excinfo.value)
    assert "modules.web_terminals.auth.session_lifetime" in message, (
        "the error must name the config key an operator would edit"
    )
    assert SESSION_LIFETIME_ENV in message, "and the environment variable carrying it"


def test_session_lifetime_carrier_is_not_popped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duration is not a credential, and every terminal in the deployment reads it.

    ``osprey web`` reads the same variable back after publishing it, and the
    multi-user compose hands it to each terminal container. Consuming it here
    would leave the next reader looking at an unset name and silently falling
    back to the default.
    """
    import os

    monkeypatch.setenv(SESSION_LIFETIME_ENV, "3600")

    get_web_credentials()

    assert os.environ[SESSION_LIFETIME_ENV] == "3600"


def test_a_refused_session_lifetime_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second call refuses too, rather than defaulting on a consumed carrier."""
    monkeypatch.setenv(SESSION_LIFETIME_ENV, "0")

    with pytest.raises(RuntimeError):
        get_web_credentials()
    with pytest.raises(RuntimeError):
        get_web_credentials()


def test_a_refused_lifetime_still_consumes_the_credential_carriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This refusal is fatal like the container-shape one, and must leak no less.

    A caller may catch the ``RuntimeError`` and keep serving, so the ordering
    property the container shape pins holds here too: both credentials are
    popped before anything can raise.
    """
    import os

    monkeypatch.setenv(OPERATOR_SECRET_ENV, "supplied-operator-secret")
    monkeypatch.setenv(PANEL_TOKEN_ENV, "supplied-panel-token")
    monkeypatch.setenv(SESSION_LIFETIME_ENV, "12h")

    with pytest.raises(RuntimeError):
        get_web_credentials()

    assert OPERATOR_SECRET_ENV not in os.environ
    assert PANEL_TOKEN_ENV not in os.environ


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


def test_roster_secrets_authorise_as_the_operator() -> None:
    """A shared sidecar handed the roster's per-user secrets lets each of them
    in as the operator — and still nothing else."""
    credentials = WebCredentials(
        operator_secret="operator-value",
        panel_token="panel-value",
        roster_secrets=("alice-value", "bob-value"),
    )
    assert credentials.verify_operator("operator-value") is True
    assert credentials.verify_operator("alice-value") is True
    assert credentials.verify_operator("bob-value") is True
    assert credentials.verify_operator("panel-value") is False
    assert credentials.verify_operator("alice-valu") is False
    assert credentials.verify_panel("alice-value") is False


def test_roster_secrets_are_popped_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the sidecar's accept flag set, every ``OSPREY_TERMINAL_SECRET_<USER>``
    leaves ``os.environ`` for the reason the operator secret does (the flag
    with them), blank ones (``${VAR:-}`` of an unset variable) count as
    absent, and the rest are accepted."""
    import os

    monkeypatch.setenv(OPERATOR_SECRET_ENV, "supplied-operator-secret")
    monkeypatch.setenv(ROSTER_ACCEPT_ENV, "1")
    monkeypatch.setenv(f"{ROSTER_SECRET_ENV_PREFIX}ALICE", "alice-value")
    monkeypatch.setenv(f"{ROSTER_SECRET_ENV_PREFIX}BOB", " bob-value ")
    monkeypatch.setenv(f"{ROSTER_SECRET_ENV_PREFIX}CAROL", "")

    credentials = get_web_credentials()

    assert credentials.roster_secrets == ("alice-value", "bob-value")
    assert credentials.verify_operator("alice-value") is True
    assert credentials.verify_operator("bob-value") is True
    assert credentials.verify_operator("") is False
    assert not [name for name in os.environ if name.startswith(ROSTER_SECRET_ENV_PREFIX)]
    assert ROSTER_ACCEPT_ENV not in os.environ


def test_roster_secrets_without_the_accept_flag_are_popped_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host's own ``osprey web`` loads the deploy ``.env``, which carries
    EVERY roster user's secret. Seeing them is not being told to accept them:
    without the flag only the bluesky-web sidecar's compose sets, they are
    popped (never inherited by a child) and no roster user is the operator."""
    import os

    monkeypatch.setenv(OPERATOR_SECRET_ENV, "supplied-operator-secret")
    monkeypatch.delenv(ROSTER_ACCEPT_ENV, raising=False)
    monkeypatch.setenv(f"{ROSTER_SECRET_ENV_PREFIX}ALICE", "alice-value")

    credentials = get_web_credentials()

    assert credentials.roster_secrets == ()
    assert credentials.verify_operator("alice-value") is False
    assert credentials.verify_operator("supplied-operator-secret") is True
    assert not [name for name in os.environ if name.startswith(ROSTER_SECRET_ENV_PREFIX)]


def test_single_user_shape_has_no_roster() -> None:
    assert get_web_credentials().roster_secrets == ()


def test_non_ascii_candidate_is_refused_not_raised(credentials: WebCredentials) -> None:
    """``compare_digest`` raises on non-ASCII ``str``; nothing here lets it.

    Headers and cookies are attacker-controlled, so one accented character
    would otherwise be an unhandled 500 rather than a 401. The two secret
    comparisons encode to UTF-8 bytes first; the session path digests the
    candidate first, which is ASCII hex whatever arrived.
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


def test_create_session_uses_the_configured_lifetime() -> None:
    """With no argument, a session lasts what the deployment configured.

    The serving callers all mint sessions with no argument, so this is the only
    thing that carries ``session_lifetime`` from the environment to a real
    session's deadline.
    """
    credentials = WebCredentials(
        operator_secret="operator-value",
        panel_token="panel-value",
        session_ttl_seconds=5,
    )

    before = time.time()
    session_id = credentials.create_session()
    after = time.time()

    deadline = credentials.sessions[_digest(session_id)]
    assert before + 5 <= deadline <= after + 5


def test_the_deadline_is_wall_clock() -> None:
    """Deadlines are epoch seconds, because the store persists them across a restart.

    A monotonic reading is meaningless to the process that loads it, so this is
    the property the on-disk store depends on rather than a free choice.
    """
    credentials = WebCredentials(operator_secret="operator-value", panel_token="panel-value")

    session_id = credentials.create_session(ttl_seconds=1000)
    deadline = credentials.sessions[_digest(session_id)]

    assert abs(deadline - (time.time() + 1000)) < 5


def test_an_explicit_ttl_overrides_the_configured_one(credentials: WebCredentials) -> None:
    """The argument is still there for tests that need a deadline they control."""
    session_id = credentials.create_session(ttl_seconds=1000)

    assert credentials.sessions[_digest(session_id)] <= time.time() + 1000


def test_a_populated_holder_carries_the_configured_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the environment value reaches the process's own holder."""
    monkeypatch.setenv(SESSION_LIFETIME_ENV, "3600")

    assert get_web_credentials().session_ttl_seconds == 3600


def test_the_holder_default_is_the_module_default(credentials: WebCredentials) -> None:
    """A directly-constructed holder is not a second opinion on the lifetime."""
    assert credentials.session_ttl_seconds == DEFAULT_SESSION_LIFETIME


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
    assert list(credentials.sessions) == [_digest(live)]


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


@pytest.mark.parametrize("forged", ["\x00" * 43, _SESSION_DECOY])
def test_the_cost_equalizing_decoy_is_not_a_credential(
    credentials: WebCredentials, forged: str
) -> None:
    """A candidate shaped like — or equal to — the miss-path decoy is still refused.

    The decoy exists only so a miss costs what a hit costs. If the answer were
    read off the comparison alone, sending the decoy's own value would compare
    equal to it and authenticate against an EMPTY session map. The cases here
    are the ones an attacker can actually deliver to the token exchange this
    map serves: a NUL run, which survives a ``%00`` query string, a JSON body
    and a websocket frame intact; and the decoy's own hex spelling, which is
    plain ASCII and survives anything.
    """
    assert credentials.verify_session(forged) is False
    assert credentials.sessions == {}

    # Still refused once the map is non-empty, so a live session cannot be the
    # thing that makes the forgery work either — and the attempt adds nothing
    # to the map.
    credentials.create_session()
    assert credentials.verify_session(forged) is False
    assert len(credentials.sessions) == 1


@pytest.mark.parametrize("length", [1, 42, 44, 63, 65, 200])
def test_wrong_length_forgeries_are_refused(credentials: WebCredentials, length: int) -> None:
    """A NUL run of any other length is refused too — no length is a key."""
    credentials.create_session()

    assert credentials.verify_session("\x00" * length) is False


def test_a_map_key_presented_as_the_cookie_is_refused(credentials: WebCredentials) -> None:
    """The map — and the file it is persisted to — authenticates nobody.

    This is the property that lets the store live on a volume the agent's own
    PTY can read: what is written there is a digest, and a digest handed back
    as a cookie is digested AGAIN before the lookup, so it misses.
    """
    credentials.create_session()
    key = next(iter(credentials.sessions))

    assert credentials.verify_session(key) is False


def test_the_map_holds_no_raw_session_id(credentials: WebCredentials) -> None:
    """What is keyed is the digest, never the value the browser sends back."""
    session_id = credentials.create_session()

    assert session_id not in credentials.sessions
    assert hashlib.sha256(session_id.encode("utf-8")).hexdigest() in credentials.sessions
    assert credentials.verify_session(session_id) is True


def test_a_secret_is_not_a_session(credentials: WebCredentials) -> None:
    """The three credentials are separate keyspaces; none substitutes for another."""
    session_id = credentials.create_session()

    assert credentials.verify_session("operator-value") is False
    assert credentials.verify_operator(session_id) is False
    assert credentials.verify_panel(session_id) is False


def test_a_non_persisting_session_is_marked_ephemeral(credentials: WebCredentials) -> None:
    """``persist=False`` records the digest the store's snapshot must skip.

    The session is otherwise ordinary — it verifies like any other; the flag
    only decides whether it may reach the disk.
    """
    session_id = credentials.create_session(persist=False)
    digest = _digest(session_id)

    assert digest in credentials.sessions
    assert digest in credentials._ephemeral
    assert credentials.verify_session(session_id) is True


def test_a_persisting_session_is_not_marked_ephemeral(credentials: WebCredentials) -> None:
    """The default is persistable: an ordinary login must survive a restart."""
    session_id = credentials.create_session()

    assert _digest(session_id) not in credentials._ephemeral
    assert credentials._ephemeral == set()


def test_revoking_clears_both_the_map_and_the_ephemeral_set(
    credentials: WebCredentials,
) -> None:
    """A logout must not leave the digest behind in the set that tracks the map."""
    session_id = credentials.create_session(persist=False)

    assert credentials.revoke_session(session_id) is True
    assert credentials.sessions == {}
    assert credentials._ephemeral == set()


def test_expiry_purges_the_ephemeral_set_too(credentials: WebCredentials) -> None:
    """Otherwise the set would accumulate digests of sessions that are long gone."""
    credentials.create_session(ttl_seconds=0, persist=False)
    assert credentials._ephemeral != set()

    credentials.create_session()

    assert len(credentials.sessions) == 1
    assert credentials._ephemeral == set()


# ---------------------------------------------------------------------------
# Browser sessions: the on-disk store
# ---------------------------------------------------------------------------


def _write_store(path, sessions: dict[str, float]) -> None:
    """Put a store file on disk in the shape :class:`SessionStore` reads."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"v": 1, "sessions": sessions}), encoding="utf-8")


def _read_store(path) -> dict[str, float]:
    """Read the persisted ``{digest: deadline}`` map straight off the disk."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))["sessions"]


def test_restored_deadlines_are_clamped_to_the_configured_lifetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A restart that SHORTENED the lifetime must shorten the sessions it inherits.

    The stored deadline was written under whatever lifetime was configured
    then. An operator who cuts the lifetime and restarts has said what the
    longest live session may now be, so a deadline from the old value is capped
    at ``now + ttl`` rather than honoured. A deadline already inside the new
    window is untouched — clamping is a ceiling, never an extension.
    """
    now = time.time()
    store_dir = tmp_path / "web_terminal"
    _write_store(store_dir / "sessions.json", {"far": now + 100_000, "near": now + 30})
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(store_dir))
    monkeypatch.setenv(SESSION_LIFETIME_ENV, "3600")

    sessions = get_web_credentials().sessions

    assert set(sessions) == {"far", "near"}
    # Two-sided: a clamp that landed short would log every operator out on the
    # restart the store exists to carry them through, which is the same failure
    # as not clamping at all, only in the other direction.
    assert sessions["far"] <= time.time() + 3600
    assert sessions["far"] >= now + 3599
    assert sessions["near"] == pytest.approx(now + 30)


def test_an_expired_store_restores_nothing_and_the_next_login_replaces_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Restoring is not resurrecting: a deadline in the past is gone for good.

    And the stale entries are not merely hidden in memory — the first login
    after the restart writes the map as it now is, so the dead digests leave
    the disk as well.
    """
    store_dir = tmp_path / "web_terminal"
    path = store_dir / "sessions.json"
    _write_store(path, {"stale": time.time() - 1})
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(store_dir))

    credentials = get_web_credentials()
    assert credentials.sessions == {}

    session_id = credentials.create_session()

    assert set(_read_store(path)) == {_digest(session_id)}


def test_an_ephemeral_session_never_reaches_the_store(tmp_path) -> None:
    """``persist=False`` is what keeps a process-scoped session off the disk.

    The persisting session written in the same call proves the write happened
    at all, so an empty file cannot pass this by accident.
    """
    store = SessionStore(tmp_path / "web_terminal", "")
    credentials = WebCredentials(
        operator_secret="operator-value", panel_token="panel-value", store=store
    )

    ephemeral = credentials.create_session(persist=False)
    persisting = credentials.create_session()

    assert set(_read_store(store.path)) == {_digest(persisting)}
    assert _digest(ephemeral) not in _read_store(store.path)
    # The excluded session is still an ordinary one in this process.
    assert credentials.verify_session(ephemeral) is True


def test_a_revoke_that_matched_nothing_does_not_rewrite_the_store(tmp_path) -> None:
    """A logout tries every cookie candidate; at most one of them is a session.

    ``revoke_session`` purges no expiries, so a call that matched nothing has
    left the map exactly as it found it and has nothing to persist. Writing
    anyway would turn one interactive logout into a full atomic rewrite per
    candidate.
    """
    saves: list[int] = []

    class CountingStore(SessionStore):
        def save(self, snapshot, seq):
            saves.append(seq)
            super().save(snapshot, seq)

    store = CountingStore(tmp_path / "web_terminal", "")
    credentials = WebCredentials(
        operator_secret="operator-value", panel_token="panel-value", store=store
    )
    live = credentials.create_session()
    saves.clear()

    assert credentials.revoke_session(mint_secret()) is False
    assert saves == []

    # The revoke that DOES match still writes — and the map it leaves is empty,
    # which is exactly the snapshot that has to reach the disk to clear it.
    assert credentials.revoke_session(live) is True
    assert len(saves) == 1
    assert _read_store(store.path) == {}


def test_a_store_that_is_not_utf8_costs_a_re_login_not_the_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Bytes that do not decode are just another unusable file.

    A decode failure is a ``ValueError``, not an ``OSError``, so it would slip
    past the read guard and out of population — and a failed population is
    deliberately not cached, which makes it permanent: every request for the
    life of the process would be refused over a file that holds nothing but
    deadlines. Re-resolving here is what pins that, since a raise on the second
    call is what a lockout actually looks like.
    """
    store_dir = tmp_path / "web_terminal"
    store_dir.mkdir(parents=True)
    (store_dir / "sessions.json").write_bytes(b"\xff\xfe not utf8")
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(store_dir))

    credentials = get_web_credentials()

    assert credentials.store is not None
    assert credentials.sessions == {}
    assert get_web_credentials() is credentials


def test_population_survives_a_store_read_that_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The second guard, at the call site that would turn a raise into a lockout.

    :meth:`SessionStore.load` promises never to raise and is tested on its own
    for that. This pins the promise being broken anyway: a process must serve
    with no restored sessions rather than not serve at all.
    """

    def explode(self):
        raise RuntimeError("the volume went away mid-read")

    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(tmp_path / "web_terminal"))
    monkeypatch.setattr(SessionStore, "load", explode)

    credentials = get_web_credentials()

    assert credentials.store is not None
    assert credentials.sessions == {}


def test_no_store_dir_means_no_store_and_no_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unconfigured process must not touch a disk at all.

    A default store path would put the deployment's session file wherever the
    process happened to start. ``os.replace`` is the last step of every write
    the store makes, so a call that never arrives there is a write that never
    happened. ``pathlib.Path.write_text`` is guarded the same way even though
    :meth:`SessionStore._write_atomic` does not use it — belt-and-suspenders
    against a future write path taking that shortcut instead of the atomic
    temp-file-and-replace one.

    This also doubles as the guard for the reason
    ``tests/conftest.py::reset_web_credentials_between_tests`` clears
    ``SESSION_STORE_DIR_ENV`` (and ``SESSION_LIFETIME_ENV``) before every
    test in the suite: a worker that inherited a real store directory from an
    earlier in-process ``osprey web`` launch would fail exactly this test.
    """
    import os
    from pathlib import Path

    replacements: list[tuple] = []

    def refuse(*args, **kwargs):
        replacements.append(args)
        raise AssertionError("a holder with no store wrote a file")

    def refuse_write_text(*args, **kwargs):
        raise AssertionError("a holder with no store wrote a file via Path.write_text")

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(Path, "write_text", refuse_write_text)

    credentials = get_web_credentials()
    assert credentials.store is None

    session_id = credentials.create_session()
    credentials.revoke_session(session_id)

    assert replacements == []


def test_population_never_writes_to_the_store(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Starting up must not replace the store it just read.

    A process that rewrote the store at population would empty it on the way
    through a crash loop — each start would persist the sessions it had not yet
    been handed. Nothing here is allowed to reach :meth:`SessionStore.save`, so
    a ``save`` that raises must not stop the process from populating.
    """
    live = time.time() + 600
    store_dir = tmp_path / "web_terminal"
    _write_store(store_dir / "sessions.json", {"live": live})
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(store_dir))

    def explode(self, snapshot, seq):
        raise AssertionError("population wrote to the store")

    monkeypatch.setattr(SessionStore, "save", explode)

    credentials = get_web_credentials()

    assert credentials.sessions == {"live": pytest.approx(live)}
    assert credentials.store is not None


def test_a_slow_save_does_not_block_verification(tmp_path) -> None:
    """The write happens outside the lock, so a stalled disk cannot stall a request.

    :meth:`verify_session` runs on every cookie-bearing request. If the store
    were written while the session map's lock was held, a full or hung
    filesystem would queue every page load and every websocket frame behind it
    — the outage the store exists to be cheaper than.
    """
    inside_save = threading.Event()
    release = threading.Event()

    class BlockingStore(SessionStore):
        def save(self, snapshot, seq):
            inside_save.set()
            release.wait(5)

    credentials = WebCredentials(operator_secret="operator-value", panel_token="panel-value")
    live = credentials.create_session()
    credentials.store = BlockingStore(tmp_path / "web_terminal", "")

    writer = threading.Thread(target=credentials.create_session, daemon=True)
    writer.start()
    try:
        assert inside_save.wait(5), "the store write never started"
        started = time.monotonic()
        verified = credentials.verify_session(live)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        writer.join(5)

    assert verified is True
    assert elapsed < 0.2, f"verify_session waited {elapsed:.2f}s on a blocked store write"


def test_a_slow_load_does_not_block_another_holders_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Population reads the disk; a holder already serving must not feel it.

    The two hold different locks — a companion app's credentials are only
    shared once population finishes — so a terminal starting up against a slow
    volume must not freeze the sessions of one already answering requests.
    """
    inside_load = threading.Event()
    release = threading.Event()

    def blocking_load(self):
        inside_load.set()
        release.wait(5)
        return {}

    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(tmp_path / "web_terminal"))
    monkeypatch.setattr(SessionStore, "load", blocking_load)

    serving = WebCredentials(operator_secret="operator-value", panel_token="panel-value")
    live = serving.create_session()

    populating = threading.Thread(target=get_web_credentials, daemon=True)
    populating.start()
    try:
        assert inside_load.wait(5), "the store read never started"
        started = time.monotonic()
        verified = serving.verify_session(live)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        populating.join(5)

    assert verified is True
    assert elapsed < 0.2, f"verify_session waited {elapsed:.2f}s on a blocked store read"


def test_the_store_file_is_named_for_the_served_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Two terminals on one host each keep their own sessions.

    The port is derived exactly as the session cookie's name is, so the browser
    holding a cookie named for one port finds its session in the store named
    for the same one.
    """
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(tmp_path / "web_terminal"))
    monkeypatch.setenv(WEB_PORT_ENV, "8080")

    store = get_web_credentials().store

    assert store is not None
    assert store.path == tmp_path / "web_terminal" / "sessions-8080.json"


def test_a_non_numeric_port_gives_the_bare_store_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A stray value names no port, and must not name a file either.

    The same fallback :func:`session_cookie_name` takes: an unset compose
    variable or a hostname left in the carrier would otherwise produce a store
    file the next start does not look for.
    """
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(tmp_path / "web_terminal"))
    monkeypatch.setenv(WEB_PORT_ENV, "not-a-port")

    store = get_web_credentials().store

    assert store is not None
    assert store.path == tmp_path / "web_terminal" / "sessions.json"


def test_the_store_carriers_are_read_never_popped(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Neither the store directory nor the port is a credential, and both outlive population.

    Two later readers depend on it: a re-population in the same process has to
    resolve the same directory and the same port-named file, and the CLI that
    clears a deployment's sessions resolves the store the same way. Popping
    either — which is one well-meaning line in
    :func:`close_env_carriers` — would leave the second reader looking at a
    different file, or at no store at all, with nothing to say why.
    """
    import os

    store_dir = str(tmp_path / "web_terminal")
    monkeypatch.setenv(SESSION_STORE_DIR_ENV, store_dir)
    monkeypatch.setenv(WEB_PORT_ENV, "8080")

    get_web_credentials()
    web_auth.close_env_carriers()

    assert os.environ[SESSION_STORE_DIR_ENV] == store_dir
    assert os.environ[WEB_PORT_ENV] == "8080"


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


def test_session_lifetime_default_is_defined_exactly_once() -> None:
    """Every surface that needs the default imports it rather than repeating 12 hours.

    A second literal would drift the moment one of the three is tuned, and the
    two shapes would then disagree about how long a session lasts. Identity —
    not equality — is what is pinned: the render path and the auth sidecar must
    hold *this* object, so re-defining either locally fails here.
    """
    from osprey.deployment.web_terminals import render
    from osprey.services.auth_sidecar import app

    assert render.DEFAULT_SESSION_LIFETIME is web_auth.DEFAULT_SESSION_LIFETIME
    assert app.DEFAULT_SESSION_LIFETIME is web_auth.DEFAULT_SESSION_LIFETIME
    assert web_auth.DEFAULT_SESSION_LIFETIME == 12 * 60 * 60


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
    ("POST", "/api/terminal/posture"),
    ("GET", "/api/terminal/posture"),
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
