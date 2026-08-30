"""Contract tests for the two HTTP-layer audit emitters.

Osprey's HTTP surface decides two things worth a ledger entry, at two different
depths, and this module pins both:

* :class:`~osprey.interfaces.common_middleware.WebAuthMiddleware` refuses an
  unauthenticated or cross-origin connection — a ``401``/``403`` that today
  exists only as a status code the caller sees and nobody else does.
* :class:`~osprey.interfaces.common_middleware.HttpAuditMiddleware`, installed
  innermost by :func:`~osprey.interfaces._app_setup.configure_interface_app`,
  records every *admitted* state-changing request, so the ledger says what got
  through the gate as well as what did not.

Like ``test_auth_middleware.py``, most of these drive raw ASGI scopes: a
middleware's input *is* a scope, and a raw one is immune to the credential a
shared test-client fixture would otherwise inject into a refusal test.

The emitters are asserted two ways on purpose. Most tests stub
``osprey.audit.writer.record`` and read the fields the emitter passed, because
that is the only way to see a field that the envelope would happily accept but
that nobody meant to send. One test per emitter goes through the *real* writer
onto a redirected audit root, because a stub accepts a misspelled keyword
forever and the ledger would be silently empty in production.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.concurrency import run_in_threadpool

from osprey.audit.envelope import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    MAX_DETAIL_CHARS,
    POSTURE_SOURCE_APP,
)
from osprey.interfaces import common_middleware
from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.common_middleware import (
    AUDIT_ACCOUNT_HEADER,
    AUDIT_ACCOUNT_KEY,
    AUDIT_EXPECTED_ACCOUNT_KEY,
    AUDIT_OIDC_SUBJECT_KEY,
    AUDIT_ROLE_HEADER,
    AUDIT_ROLE_SOURCE_HEADER,
    AUDIT_SUBJECT_HEADER,
    AUDITED_REFUSAL_STATUSES,
    EXTERNAL_ORIGIN_ENV,
    HTTP_MUTATION_POSTURE,
    HTTP_MUTATION_SURFACE,
    MAX_FORWARDED_VALUE_CHARS,
    OPERATOR_SECRET_HEADER,
    REASON_MUTATION,
    REASON_MUTATION_UNANSWERED,
    REASON_ROUTE_REFUSED,
    REFUSAL_REASONS,
    TERMINAL_USER_ENV,
    UNSAFE_FORWARDED_VALUE,
    WEB_AUTH_POSTURE,
    WEB_AUTH_SURFACE,
    HttpAuditMiddleware,
    WebAuthMiddleware,
    forwarded_identity,
)
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials
from osprey.utils.identity import AUDIT_IDENTITY_ENV

MIDDLEWARE_LOGGER = common_middleware.logger.name

OPERATOR_SECRET = "operator-secret-value"
PANEL_TOKEN = "panel-token-value"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


class RecordingApp:
    """A downstream ASGI app that answers ``status`` and records what it saw."""

    def __init__(self, status: int = 200, *, explode: bool = False) -> None:
        self.status = status
        self.explode = explode
        self.scopes: list[dict[str, Any]] = []
        self.bodies: list[bytes] = []

    async def __call__(self, scope, receive, send) -> None:
        self.scopes.append(scope)
        if self.explode:
            raise RuntimeError("route blew up")
        if scope["type"] == "http":
            body = b""
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    break
                body += message.get("body") or b""
                if not message.get("more_body", False):
                    break
            self.bodies.append(body)
            await send({"type": "http.response.start", "status": self.status, "headers": []})
            await send({"type": "http.response.body", "body": b"downstream"})
        elif scope["type"] == "websocket":
            await send({"type": "websocket.accept"})

    @property
    def called(self) -> bool:
        return bool(self.scopes)


def encode_headers(headers: dict[str, str] | None) -> list[tuple[bytes, bytes]]:
    return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()]


def http_scope(
    path: str = "/api/config",
    method: str = "GET",
    headers: dict[str, str] | None = None,
    *,
    app: Any = None,
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": "http",
        "headers": encode_headers(headers),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": app,
    }


def ws_scope(
    path: str = "/ws",
    headers: dict[str, str] | None = None,
    *,
    app: Any = None,
) -> dict[str, Any]:
    return {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "scheme": "ws",
        "headers": encode_headers(headers),
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8080),
        "app": app,
    }


def drive(middleware, scope, incoming: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Run one connection through ``middleware`` and return everything it sent."""
    pending = list(incoming if incoming is not None else [{"type": "http.request", "body": b""}])
    sent: list[dict[str, Any]] = []

    async def receive():
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def status_of(sent: list[dict[str, Any]]) -> int | None:
    for message in sent:
        if message["type"] in ("http.response.start", "websocket.http.response.start"):
            return message["status"]
    return None


@pytest.fixture
def credentials() -> WebCredentials:
    """A known credential holder, with the process holder emptied around it."""
    reset_web_credentials()
    yield WebCredentials(operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN)
    reset_web_credentials()


@pytest.fixture
def app_stub(credentials: WebCredentials) -> Any:
    """The minimum an ASGI ``scope["app"]`` needs to carry the credentials."""
    return SimpleNamespace(state=SimpleNamespace(web_credentials=credentials))


@pytest.fixture
def audit_writer():
    """The live ``osprey.audit.writer``, resolved when the test runs.

    Deliberately not a module-level ``import``. Sibling modules in this package
    use ``patch.dict(sys.modules, ...)``, whose exit restores the snapshot it
    took on entry — evicting any module first imported inside the block. A
    fixture holding the object bound at collection time would then keep
    patching a stale copy while the emitter's own lazy import pulled a fresh
    one, and every assertion here would quietly stop reaching the code it
    names. Resolving through :mod:`importlib` gets whatever ``sys.modules``
    holds now, which is by definition what the emitter will get too.
    """
    return importlib.import_module("osprey.audit.writer")


@pytest.fixture
def records(audit_writer, monkeypatch) -> list[dict[str, Any]]:
    """Capture what the emitters hand the writer, without writing anything."""
    captured: list[dict[str, Any]] = []

    def fake_record(**fields):
        captured.append(fields)
        return None

    monkeypatch.setattr(audit_writer, "record", fake_record)
    return captured


@pytest.fixture
def audit_root(audit_writer, tmp_path, monkeypatch):
    """Point the real writer at a scratch audit zone for the end-to-end tests."""
    target = tmp_path / "audit"
    monkeypatch.setattr(audit_writer, "audit_dir", lambda: target)
    return target


@pytest.fixture(autouse=True)
def _no_inherited_identity(monkeypatch):
    """No deployment marker leaks in from the developer's own environment."""
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.delenv(AUDIT_IDENTITY_ENV, raising=False)
    monkeypatch.delenv(EXTERNAL_ORIGIN_ENV, raising=False)


def only(records: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(records) == 1, records
    return records[0]


def detail_parts(record: dict[str, Any]) -> dict[str, str]:
    """The emitter's ``k=v`` detail string, back as a mapping."""
    detail = record.get("detail") or ""
    return dict(part.split("=", 1) for part in detail.split() if "=" in part)


# --------------------------------------------------------------------------- #
# (a) The gate's own refusals
# --------------------------------------------------------------------------- #


class TestRefusalsAreRecorded:
    def test_a_missing_credential_is_recorded(self, app_stub, records):
        sent = drive(WebAuthMiddleware(RecordingApp()), http_scope(app=app_stub))

        assert status_of(sent) == 401
        record = only(records)
        assert record["surface"] == WEB_AUTH_SURFACE
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == "missing_credential"
        assert record["subject"] == "GET /api/config"

    def test_an_invalid_credential_is_recorded_under_its_own_reason(self, app_stub, records):
        sent = drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={OPERATOR_SECRET_HEADER: "wrong"}),
        )

        assert status_of(sent) == 401
        assert only(records)["reason"] == "invalid_credential"

    def test_a_cross_origin_mutation_is_recorded_as_an_origin_mismatch(self, app_stub, records):
        sent = drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                method="POST",
                app=app_stub,
                headers={
                    OPERATOR_SECRET_HEADER: OPERATOR_SECRET,
                    "host": "127.0.0.1:8080",
                    "origin": "http://evil.test",
                },
            ),
        )

        assert status_of(sent) == 403
        record = only(records)
        assert record["reason"] == "origin_mismatch"
        assert record["decision"] == DECISION_REFUSED
        assert record["subject"] == "POST /api/config"

    def test_a_panel_token_on_a_websocket_is_recorded_as_a_tier_refusal(self, app_stub, records):
        sent = drive(
            WebAuthMiddleware(RecordingApp()),
            ws_scope(app=app_stub, headers={"authorization": f"Bearer {PANEL_TOKEN}"}),
        )

        assert any(message["type"] == "websocket.close" for message in sent)
        assert only(records)["reason"] == "tier_refused"

    def test_a_refused_websocket_names_the_websocket_method(self, app_stub, records):
        drive(WebAuthMiddleware(RecordingApp()), ws_scope(path="/ws/terminal", app=app_stub))

        assert only(records)["subject"] == "WEBSOCKET /ws/terminal"

    def test_an_admitted_request_produces_no_gate_record(self, app_stub, records):
        downstream = RecordingApp()
        drive(
            WebAuthMiddleware(downstream),
            http_scope(app=app_stub, headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET}),
        )

        assert downstream.called
        assert records == []

    def test_an_exempt_path_produces_no_gate_record(self, app_stub, records):
        drive(WebAuthMiddleware(RecordingApp()), http_scope(path="/health", app=app_stub))

        assert records == []

    def test_the_unavailable_gate_503_is_not_an_authorization_record(self, monkeypatch, records):
        """A 503 says the gate cannot run, not that this caller was refused.

        It is already an ``ERROR`` on the service log and it names no decision
        about the request, so filing it as a refusal would put a deployment
        fault in the column an operator reads for caller behaviour.
        """

        def unpopulatable(_app):
            raise RuntimeError("no operator secret was supplied")

        monkeypatch.setattr(common_middleware, "get_web_credentials", unpopulatable)
        sent = drive(WebAuthMiddleware(RecordingApp()), http_scope())

        assert status_of(sent) == 503
        assert records == []

    def test_only_401_and_403_are_audited_statuses(self):
        assert AUDITED_REFUSAL_STATUSES == frozenset({401, 403})

    def test_every_refusal_detail_the_gate_can_send_has_a_reason(self):
        """No refusal may fall through to the generic reason unnoticed.

        The four details below are the whole vocabulary of
        :meth:`WebAuthMiddleware._refuse` on an audited status; a fifth added
        without a mapping would file its records under a category that says
        nothing, which is how a ledger stops being able to answer *why*.
        """
        audited_details = {
            common_middleware._MISSING_DETAIL,
            common_middleware._INVALID_DETAIL,
            common_middleware._TIER_DETAIL,
            common_middleware._ORIGIN_DETAIL,
        }
        assert audited_details <= set(REFUSAL_REASONS)
        assert common_middleware._UNAVAILABLE_DETAIL not in REFUSAL_REASONS
        assert len(set(REFUSAL_REASONS.values())) == len(REFUSAL_REASONS)


class TestRefusalRecordProvenance:
    def test_the_gate_stamps_the_app_posture_source(self, app_stub, records):
        drive(WebAuthMiddleware(RecordingApp()), http_scope(app=app_stub))

        record = only(records)
        assert record["posture_source"] == POSTURE_SOURCE_APP
        assert record["posture"] == WEB_AUTH_POSTURE
        assert record["session"] is None

    def test_the_gate_posture_is_sandbox(self):
        """The gate governs entry and writes nothing; ``writes`` would overstate it."""
        assert WEB_AUTH_POSTURE == "sandbox"

    def test_the_actor_is_left_to_the_writer_not_taken_from_the_caller(self, app_stub, records):
        """The emitter names no ``actor``: the writer fills it from the same
        identity ladder the ledger directory is keyed on, so a forwarded subject
        can never become the actor by any route through this layer. (That the
        writer's answer is the container identity is pinned end to end in
        :class:`TestThroughTheRealWriter`.)"""
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={AUDIT_SUBJECT_HEADER: "alice"}),
        )

        assert "actor" not in only(records)


# --------------------------------------------------------------------------- #
# The forwarded identity
# --------------------------------------------------------------------------- #


class TestForwardedIdentity:
    def test_the_forwarded_account_and_role_ride_the_record(self, app_stub, records):
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_ROLE_HEADER: "operator"},
            ),
        )

        record = only(records)
        assert detail_parts(record)[AUDIT_ACCOUNT_KEY] == "alice"
        assert record["role"] == "operator"

    def test_no_forwarded_identity_leaves_the_keys_off(self, app_stub, records):
        drive(WebAuthMiddleware(RecordingApp()), http_scope(app=app_stub))

        record = only(records)
        assert AUDIT_ACCOUNT_KEY not in detail_parts(record)
        assert AUDIT_OIDC_SUBJECT_KEY not in detail_parts(record)
        assert record["role"] is None

    def test_the_decoder_names_each_of_the_four_headers_it_returns(self):
        """Four names, each decoded under the same bound, each reachable by name.

        Asserted on the decoder directly, not through an emitter, and by field
        rather than by tuple position: ``subject`` and ``account`` read alike
        and are not the same question, so a swap of the two would be invisible
        to a positional assertion while inverting what the ledger compares. The
        role source is here for the same reason it always was — the ledger
        reads it and records nothing from it, so a name that quietly stopped
        being decoded would leave every record in this module exactly as it is
        and only the terminal's chip would go blank.
        """
        base = {
            AUDIT_SUBJECT_HEADER.lower(): "idp|alice",
            AUDIT_ACCOUNT_HEADER.lower(): "alice",
            AUDIT_ROLE_HEADER.lower(): "operator",
        }

        decoded = forwarded_identity(base)
        assert decoded.subject == "idp|alice"
        assert decoded.account == "alice"
        assert decoded.role == "operator"
        assert decoded.role_source is None

        with_source = forwarded_identity({**base, AUDIT_ROLE_SOURCE_HEADER.lower(): "roster"})
        assert with_source.role_source == "roster"

        unsafe_source = forwarded_identity({**base, AUDIT_ROLE_SOURCE_HEADER.lower(): "ro ster"})
        assert unsafe_source.role_source == UNSAFE_FORWARDED_VALUE

    def test_an_absent_account_header_decodes_as_none_not_as_the_subject(self):
        """Absence is a state of its own, and the readers' fallback depends on it.

        A sidecar image built before :data:`AUDIT_ACCOUNT_HEADER` existed sends
        no account at all. Flattening that to the subject here would hide the
        one fact ``_audit_detail`` needs in order to know it is on the fallback
        path rather than looking at a password session.
        """
        decoded = forwarded_identity({AUDIT_SUBJECT_HEADER.lower(): "alice"})
        assert decoded.subject == "alice"
        assert decoded.account is None

    def test_an_unsafe_account_is_named_unsafe_under_the_same_bound(self):
        """The fourth header is not exempt from the guard the other three carry."""
        decoded = forwarded_identity({AUDIT_ACCOUNT_HEADER.lower(): "ali ce"})
        assert decoded.account == UNSAFE_FORWARDED_VALUE

    def test_an_account_matching_this_container_is_not_flagged(
        self, app_stub, records, monkeypatch
    ):
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={AUDIT_ACCOUNT_HEADER: "alice"}),
        )

        assert AUDIT_EXPECTED_ACCOUNT_KEY not in detail_parts(only(records))

    def test_an_oidc_subject_beside_a_matching_account_is_not_a_mismatch(
        self, app_stub, records, monkeypatch, caplog
    ):
        """The live defect this comparison exists to fix.

        Under ``auth.method: oidc`` the subject is the IdP's assertion about a
        person and ``OSPREY_TERMINAL_USER`` is the roster name of the card, so
        comparing those two disagreed by construction: every audited request of
        every card wrote a false mismatch marker and a WARNING, forever. The
        account is the only forwarded name this container can be the container
        *for*, so it is the only one compared. The subject is still recorded —
        under its own key, which is the point of having one.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            drive(
                WebAuthMiddleware(RecordingApp()),
                http_scope(
                    app=app_stub,
                    headers={
                        AUDIT_ACCOUNT_HEADER: "alice",
                        AUDIT_SUBJECT_HEADER: "idp|8f21c0",
                    },
                ),
            )

        parts = detail_parts(only(records))
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert AUDIT_EXPECTED_ACCOUNT_KEY not in parts
        assert parts[AUDIT_OIDC_SUBJECT_KEY] == "idp|8f21c0"
        assert caplog.messages == []

    def test_a_subject_equal_to_the_account_adds_no_subject_key(
        self, app_stub, records, monkeypatch
    ):
        """A password deployment's records do not change at all.

        There the roster username *is* the proof, so the two headers carry the
        same string and a second key naming it would be noise in every record
        of every password deployment.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_SUBJECT_HEADER: "alice"},
            ),
        )

        parts = detail_parts(only(records))
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert AUDIT_OIDC_SUBJECT_KEY not in parts

    def test_an_account_mismatching_this_container_is_audited(
        self, app_stub, records, monkeypatch, caplog
    ):
        """One user's authorization reaching another user's container.

        The emitters *record* this; they do not refuse it — enforcement is not
        an audit task, and a gate that started refusing on a header it has
        never enforced before would be a behaviour change smuggled in under an
        audit heading.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            drive(
                WebAuthMiddleware(RecordingApp()),
                http_scope(app=app_stub, headers={AUDIT_ACCOUNT_HEADER: "alice"}),
            )

        parts = detail_parts(only(records))
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert parts[AUDIT_EXPECTED_ACCOUNT_KEY] == "bob"
        assert any("alice" in message and "bob" in message for message in caplog.messages)

    def test_a_genuine_mismatch_under_oidc_still_records_both_names(
        self, app_stub, records, monkeypatch, caplog
    ):
        """Someone reaching another user's card keeps both the marker and the warning.

        The fix removes the false mismatch, not the real one — and the subject
        rides along, because "which login opened it" is the first question
        asked of a record that says a card was reached by the wrong account.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            drive(
                WebAuthMiddleware(RecordingApp()),
                http_scope(
                    app=app_stub,
                    headers={
                        AUDIT_ACCOUNT_HEADER: "alice",
                        AUDIT_SUBJECT_HEADER: "idp|8f21c0",
                    },
                ),
            )

        parts = detail_parts(only(records))
        assert parts[AUDIT_EXPECTED_ACCOUNT_KEY] == "bob"
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert parts[AUDIT_OIDC_SUBJECT_KEY] == "idp|8f21c0"
        assert len([m for m in caplog.messages if "alice" in m and "bob" in m]) == 1

    @pytest.mark.parametrize(
        "forged",
        [
            pytest.param("op\x01er", id="control-character"),
            pytest.param("jörg", id="non-ascii"),
            pytest.param("alice bob", id="interior-space"),
            pytest.param("a" * (MAX_FORWARDED_VALUE_CHARS + 1), id="over-long"),
        ],
    )
    def test_an_unrecordable_forwarded_value_is_named_unsafe(self, app_stub, records, forged):
        """A forged header never becomes the text of the record.

        The account arrives over the wire and nothing upstream of this gate is
        obliged to have produced it. The first two below are outside the
        sidecar's printable-ASCII contract and so cannot have come from it; the
        third would silently become a second ``key=value`` field in a
        space-separated detail; the fourth is longer than any identifier the
        sidecar can mint. All four are named as unusable rather than copied
        into a column an operator reads as an identifier.
        """
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: forged, AUDIT_ROLE_HEADER: forged},
            ),
        )

        record = only(records)
        assert detail_parts(record)[AUDIT_ACCOUNT_KEY] == UNSAFE_FORWARDED_VALUE
        assert record["role"] == UNSAFE_FORWARDED_VALUE

    def test_an_unrecordable_subject_is_named_unsafe_under_its_own_key(self, app_stub, records):
        """The new key is guarded like the one it sits beside.

        The subject is now written under :data:`AUDIT_OIDC_SUBJECT_KEY`, so it
        is a second caller-controlled value reaching ``detail`` — and would be
        the easy way back in if only the account were bounded.
        """
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_SUBJECT_HEADER: "alice bob"},
            ),
        )

        parts = detail_parts(only(records))
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert parts[AUDIT_OIDC_SUBJECT_KEY] == UNSAFE_FORWARDED_VALUE

    def test_the_unsafe_marker_is_itself_one_detail_token(self):
        """Otherwise the guard would break the format it exists to protect."""
        assert " " not in UNSAFE_FORWARDED_VALUE

    def test_an_unsafe_account_still_counts_as_a_mismatch(self, app_stub, records, monkeypatch):
        """A value the sidecar could not have produced is not this container's user.

        The guard replaces the text, not the comparison: ``bob\x01`` is not
        ``bob``, and a record that quietly dropped ``expected=`` here would let
        a forged header buy silence that the real ``bob`` does not need.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={AUDIT_ACCOUNT_HEADER: "bob\x01"}),
        )

        parts = detail_parts(only(records))
        assert parts[AUDIT_ACCOUNT_KEY] == UNSAFE_FORWARDED_VALUE
        assert parts[AUDIT_EXPECTED_ACCOUNT_KEY] == "bob"

    def test_a_forged_subject_beside_a_matching_account_buys_no_mismatch(
        self, app_stub, records, monkeypatch, caplog
    ):
        """And the reverse: the subject is recorded, never compared.

        A caller who can set headers can set the subject to anything at all.
        Now that only the account is compared, that must not be a way to
        manufacture a mismatch marker against a container whose real account
        is the one it was rendered for.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            drive(
                WebAuthMiddleware(RecordingApp()),
                http_scope(
                    app=app_stub,
                    headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_SUBJECT_HEADER: "mallory"},
                ),
            )

        parts = detail_parts(only(records))
        assert AUDIT_EXPECTED_ACCOUNT_KEY not in parts
        assert parts[AUDIT_OIDC_SUBJECT_KEY] == "mallory"
        assert caplog.messages == []

    def test_a_value_at_the_cap_is_still_recorded_verbatim(self, app_stub, records):
        """The bound is on forgery-scale input, not on anything real."""
        at_cap = "a" * MAX_FORWARDED_VALUE_CHARS
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={AUDIT_ACCOUNT_HEADER: at_cap}),
        )

        assert detail_parts(only(records))[AUDIT_ACCOUNT_KEY] == at_cap

    def test_the_cap_leaves_room_for_a_real_identifier(self):
        """Above any OIDC ``sub`` or role name, and well below the detail bound."""
        assert MAX_FORWARDED_VALUE_CHARS >= 64
        assert MAX_FORWARDED_VALUE_CHARS < MAX_DETAIL_CHARS

    def test_an_over_long_account_cannot_truncate_the_mismatch_away(
        self, app_stub, audit_root, monkeypatch
    ):
        """The one input a forger fully controls must not erase the signal.

        ``AuditEnvelope`` truncates ``detail`` silently at
        :data:`MAX_DETAIL_CHARS`. Unbounded, a forwarded account of ~1000
        printable characters fills ``detail`` and pushes the mismatch marker
        appended after it off the end, so the stored record reads as an
        ordinary admitted request carrying a caller-chosen wall of text.
        Driven through the *real* writer and envelope, because the truncation
        being defended against is the envelope's own.
        """
        # No AUDIT_IDENTITY_ENV: TERMINAL_USER_ENV is the ladder's first rung,
        # so setting the container user to make the mismatch also makes "bob"
        # the actor, and the ledger directory.
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: "a" * (MAX_DETAIL_CHARS + 200)},
            ),
        )

        ledger = audit_root / "bob" / f"{WEB_AUTH_SURFACE}.jsonl"
        record = json.loads(ledger.read_text().splitlines()[0])
        parts = detail_parts(record)
        assert parts[AUDIT_EXPECTED_ACCOUNT_KEY] == "bob"
        assert parts[AUDIT_ACCOUNT_KEY] == UNSAFE_FORWARDED_VALUE
        assert len(record["detail"]) < MAX_DETAIL_CHARS

    def test_the_account_key_does_not_collide_with_the_envelope_subject(self, app_stub, records):
        """``subject`` means two different things across the epic's surfaces.

        On these two HTTP surfaces the envelope's top-level ``subject`` is the
        **route**; on the auth sidecar's surface it is the **account**. Naming
        the forwarded account ``subject=`` inside ``detail`` would have put
        both meanings in one record, one level apart, and made "every record
        for account alice" a per-surface question.

        Driven with both identity headers, because the subject now has a key of
        its own inside ``detail`` and ``oidc_subject`` was chosen partly so that
        neither key is bare ``subject``.
        """
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_SUBJECT_HEADER: "idp|8f21c0"},
            ),
        )

        record = only(records)
        parts = detail_parts(record)
        assert record["subject"] == "GET /api/config"
        assert AUDIT_ACCOUNT_KEY != "subject"
        assert AUDIT_OIDC_SUBJECT_KEY != "subject"
        assert "subject" not in parts
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert parts[AUDIT_OIDC_SUBJECT_KEY] == "idp|8f21c0"

    def test_the_oidc_subject_key_is_not_the_writer_s_identity(self):
        """The audit writer owns ``identity`` for the ledger's directory component.

        A ``detail`` key of that name would put one word on the folder a record
        is filed under and on a person named inside it.
        """
        assert AUDIT_OIDC_SUBJECT_KEY == "oidc_subject"
        assert AUDIT_OIDC_SUBJECT_KEY not in {"identity", "subject", AUDIT_ACCOUNT_KEY}

    def test_the_mismatch_marker_is_written_before_the_account(
        self, app_stub, records, monkeypatch
    ):
        """Belt to the cap's braces: the signal survives any future truncation.

        The subject comes last for the same reason — it is a second value a
        caller controls, and appending it ahead of the account would put it
        between the marker and what the marker disagrees with.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(
                app=app_stub,
                headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_SUBJECT_HEADER: "idp|8f21c0"},
            ),
        )

        keys = [part.split("=", 1)[0] for part in only(records)["detail"].split()]
        assert keys.index(AUDIT_EXPECTED_ACCOUNT_KEY) < keys.index(AUDIT_ACCOUNT_KEY)
        assert keys.index(AUDIT_ACCOUNT_KEY) < keys.index(AUDIT_OIDC_SUBJECT_KEY)

    def test_the_header_names_are_the_sidecar_s_own(self):
        """The names are spelled locally; this is what keeps them in step.

        Importing them would put a service package in the interfaces' import
        closure for four string constants, so the drift check is a test — the
        same trade the rest of the audit work makes.
        """
        from osprey.services.auth_sidecar.identity_headers import (
            ACCOUNT_HEADER,
            ROLE_HEADER,
            ROLE_SOURCE_HEADER,
            SUBJECT_HEADER,
        )

        assert AUDIT_SUBJECT_HEADER == SUBJECT_HEADER
        assert AUDIT_ACCOUNT_HEADER == ACCOUNT_HEADER
        assert AUDIT_ROLE_HEADER == ROLE_HEADER
        assert AUDIT_ROLE_SOURCE_HEADER == ROLE_SOURCE_HEADER

    def test_the_two_identity_headers_are_not_the_same_name(self):
        """The comparison's whole premise: they name two different things."""
        assert AUDIT_ACCOUNT_HEADER != AUDIT_SUBJECT_HEADER


# --------------------------------------------------------------------------- #
# The mixed-version fallback
# --------------------------------------------------------------------------- #


class TestOlderSidecarFallback:
    """A deployment pinning ``auth.image`` to a build with no account header.

    The header is what the fix compares, so an image that does not send it must
    keep the behaviour it has today — the subject compared against
    ``OSPREY_TERMINAL_USER``, recorded under ``account=`` — rather than lose
    the mismatch check altogether. Such a deployment also keeps today's false
    mismatch under OIDC; that is the bounded cost of the pin, and it is named
    in the changelog.
    """

    def test_a_subject_only_match_is_recorded_as_the_account(
        self, app_stub, records, monkeypatch, caplog
    ):
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            drive(
                WebAuthMiddleware(RecordingApp()),
                http_scope(app=app_stub, headers={AUDIT_SUBJECT_HEADER: "alice"}),
            )

        parts = detail_parts(only(records))
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert AUDIT_EXPECTED_ACCOUNT_KEY not in parts
        assert caplog.messages == []

    def test_a_subject_only_mismatch_is_still_flagged_and_warned(
        self, app_stub, records, monkeypatch, caplog
    ):
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            drive(
                WebAuthMiddleware(RecordingApp()),
                http_scope(app=app_stub, headers={AUDIT_SUBJECT_HEADER: "alice"}),
            )

        parts = detail_parts(only(records))
        assert parts[AUDIT_EXPECTED_ACCOUNT_KEY] == "bob"
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert len([m for m in caplog.messages if "alice" in m and "bob" in m]) == 1

    def test_a_subject_only_record_gains_no_second_key(self, app_stub, records, monkeypatch):
        """The fallback records exactly the keys it recorded before the fix.

        ``oidc_subject=`` names the subject *when it differs from the account*;
        with no account forwarded the subject is what stands in for one, so
        there is nothing for a second key to add.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={AUDIT_SUBJECT_HEADER: "alice"}),
        )

        assert only(records)["detail"] == (
            f"{AUDIT_EXPECTED_ACCOUNT_KEY}=bob {AUDIT_ACCOUNT_KEY}=alice"
        )


# --------------------------------------------------------------------------- #
# (b) The inner layer: admitted mutations
# --------------------------------------------------------------------------- #


class TestAdmittedMutations:
    def test_a_safe_method_is_not_recorded(self, records):
        downstream = RecordingApp()
        drive(HttpAuditMiddleware(downstream), http_scope(method="GET"))

        assert downstream.called
        assert records == []

    @pytest.mark.parametrize("method", ["HEAD", "OPTIONS"])
    def test_the_other_safe_methods_are_not_recorded(self, records, method):
        drive(HttpAuditMiddleware(RecordingApp()), http_scope(method=method))

        assert records == []

    def test_an_admitted_mutation_is_recorded(self, records):
        drive(HttpAuditMiddleware(RecordingApp()), http_scope(method="POST"))

        record = only(records)
        assert record["surface"] == HTTP_MUTATION_SURFACE
        assert record["decision"] == DECISION_ALLOWED
        assert record["reason"] == REASON_MUTATION
        assert record["subject"] == "POST /api/config"
        assert record["posture"] == HTTP_MUTATION_POSTURE
        assert record["posture_source"] == POSTURE_SOURCE_APP
        assert record["session"] is None
        assert detail_parts(record)["status"] == "200"

    @pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
    def test_every_unsafe_method_is_recorded(self, records, method):
        drive(HttpAuditMiddleware(RecordingApp()), http_scope(method=method))

        assert only(records)["subject"].startswith(f"{method} ")

    def test_the_mutation_posture_is_writes(self):
        """This layer exists because the surface admits state-changing requests."""
        assert HTTP_MUTATION_POSTURE == "writes"

    def test_a_route_that_refuses_is_not_recorded_as_allowed(self, records):
        """A ``403`` from the route is a refusal, whatever the gate decided.

        This is the shape the dedup work (task 2.10) generalises: an outer
        layer that stamped every admitted request ``allowed`` would overwrite
        the inner decision with its own.
        """
        drive(HttpAuditMiddleware(RecordingApp(status=403)), http_scope(method="POST"))

        record = only(records)
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == REASON_ROUTE_REFUSED
        assert detail_parts(record)["status"] == "403"

    def test_a_route_that_crashes_is_still_recorded(self, records):
        with pytest.raises(RuntimeError):
            drive(HttpAuditMiddleware(RecordingApp(explode=True)), http_scope(method="POST"))

        record = only(records)
        assert record["subject"] == "POST /api/config"
        assert detail_parts(record)["status"] == "none"
        # Admitted and run, with no answer produced — not guessed into a
        # refusal the route never made, and not filed under the reason the
        # requests that actually succeeded carry.
        assert record["decision"] == DECISION_ALLOWED
        assert record["reason"] == REASON_MUTATION_UNANSWERED

    def test_a_crash_is_readable_without_parsing_the_detail(self, records):
        """``decision``/``reason`` is the pair an operator scans.

        ``decision=allowed`` is true of a crashed route — it *was* admitted —
        so on its own it cannot separate "admitted and succeeded" from
        "admitted and blew up". The reason column carries that, rather than
        leaving it only in ``detail``'s ``status=none``.
        """
        assert REASON_MUTATION_UNANSWERED != REASON_MUTATION
        drive(HttpAuditMiddleware(RecordingApp(status=200)), http_scope(method="POST"))
        with pytest.raises(RuntimeError):
            drive(HttpAuditMiddleware(RecordingApp(explode=True)), http_scope(method="POST"))

        answered, unanswered = records
        assert answered["decision"] == unanswered["decision"] == DECISION_ALLOWED
        assert {answered["reason"], unanswered["reason"]} == {
            REASON_MUTATION,
            REASON_MUTATION_UNANSWERED,
        }

    def test_a_websocket_is_not_a_mutation(self, records):
        downstream = RecordingApp()
        drive(HttpAuditMiddleware(downstream), ws_scope())

        assert downstream.called
        assert records == []

    def test_a_lifespan_scope_passes_through_untouched(self, records):
        downstream = RecordingApp()
        drive(HttpAuditMiddleware(downstream), {"type": "lifespan"})

        assert downstream.called
        assert records == []

    def test_the_request_body_still_reaches_the_route(self, records):
        downstream = RecordingApp()
        drive(
            HttpAuditMiddleware(downstream),
            http_scope(method="POST"),
            incoming=[{"type": "http.request", "body": b'{"key": "value"}'}],
        )

        assert downstream.bodies == [b'{"key": "value"}']

    def test_the_response_still_reaches_the_client(self, records):
        sent = drive(HttpAuditMiddleware(RecordingApp(status=201)), http_scope(method="POST"))

        assert status_of(sent) == 201
        assert any(message["type"] == "http.response.body" for message in sent)

    def test_the_forwarded_identity_rides_the_mutation_record(self, records, monkeypatch):
        """Both emitters read the same decoder, so both compare the account.

        The inner layer is where an *admitted* request is filed, which is the
        record the OIDC defect polluted on every card; a fix that reached only
        the gate would have left the noisier half of the ledger untouched.
        """
        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")
        drive(
            HttpAuditMiddleware(RecordingApp()),
            http_scope(
                method="POST",
                headers={
                    AUDIT_ACCOUNT_HEADER: "alice",
                    AUDIT_SUBJECT_HEADER: "idp|8f21c0",
                    AUDIT_ROLE_HEADER: "operator",
                },
            ),
        )

        record = only(records)
        parts = detail_parts(record)
        assert parts[AUDIT_ACCOUNT_KEY] == "alice"
        assert parts[AUDIT_EXPECTED_ACCOUNT_KEY] == "bob"
        assert parts[AUDIT_OIDC_SUBJECT_KEY] == "idp|8f21c0"
        assert record["role"] == "operator"

    def test_a_matching_account_leaves_the_mutation_record_unflagged(self, records, monkeypatch):
        """The defect's own shape on the inner layer: the rightful owner's card."""
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        drive(
            HttpAuditMiddleware(RecordingApp()),
            http_scope(
                method="POST",
                headers={AUDIT_ACCOUNT_HEADER: "alice", AUDIT_SUBJECT_HEADER: "idp|8f21c0"},
            ),
        )

        assert AUDIT_EXPECTED_ACCOUNT_KEY not in detail_parts(only(records))

    def test_an_audit_failure_never_costs_the_request(self, audit_writer, monkeypatch):
        """The ledger degrades; the mutation does not."""

        def explode(**_fields):
            raise OSError("audit zone is read-only")

        monkeypatch.setattr(audit_writer, "record", explode)
        sent = drive(HttpAuditMiddleware(RecordingApp()), http_scope(method="POST"))

        assert status_of(sent) == 200


class TestGateFailureIsolation:
    def test_a_refusal_is_still_answered_when_the_ledger_explodes(
        self, audit_writer, app_stub, monkeypatch
    ):
        def explode(**_fields):
            raise OSError("audit zone is read-only")

        monkeypatch.setattr(audit_writer, "record", explode)
        sent = drive(WebAuthMiddleware(RecordingApp()), http_scope(app=app_stub))

        assert status_of(sent) == 401
        assert json.loads(sent[1]["body"])["detail"]


class TestTheNeverRaisesBoundaryEnclosesTheWholeEmitter:
    """Not just the write — every step an emitter takes to reach it.

    The write is the obvious failure, and it was already fenced. The rest of
    the emitter — header decoding, detail assembly, the mismatch ``WARNING`` —
    ran outside that fence, where a raise costs the gate its ``401`` and the
    mutation layer an already-served ``200``. These patch ``_audit_detail``,
    which is squarely in that outside region, rather than the writer.
    """

    @staticmethod
    def _explode(*_args, **_kwargs):
        raise RuntimeError("detail assembly blew up")

    def test_the_gate_still_refuses(self, app_stub, records, monkeypatch, caplog):
        monkeypatch.setattr(common_middleware, "_audit_detail", self._explode)
        with caplog.at_level("WARNING", logger=MIDDLEWARE_LOGGER):
            sent = drive(WebAuthMiddleware(RecordingApp()), http_scope(app=app_stub))

        assert status_of(sent) == 401
        assert json.loads(sent[1]["body"])["detail"]
        assert records == []
        assert any("audit record" in message for message in caplog.messages)

    def test_the_mutation_is_still_answered(self, records, monkeypatch):
        """Worse here: ``_record`` runs from a ``finally`` over a sent response."""
        monkeypatch.setattr(common_middleware, "_audit_detail", self._explode)
        sent = drive(HttpAuditMiddleware(RecordingApp(status=201)), http_scope(method="POST"))

        assert status_of(sent) == 201
        assert records == []

    def test_the_routes_own_exception_is_not_masked(self, records, monkeypatch):
        """A raise from the ``finally`` would replace the route's own failure."""
        monkeypatch.setattr(common_middleware, "_audit_detail", self._explode)
        with pytest.raises(RuntimeError, match="route blew up"):
            drive(HttpAuditMiddleware(RecordingApp(explode=True)), http_scope(method="POST"))


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


class TestWiring:
    def test_the_audit_layer_is_installed_on_every_interface_app(self, tmp_path):
        from fastapi import FastAPI

        app = FastAPI()
        configure_interface_app(app, static_dir=tmp_path)

        assert HttpAuditMiddleware in {mw.cls for mw in app.user_middleware}

    def test_the_audit_layer_is_innermost_and_the_gate_stays_outermost(self, tmp_path):
        """Order is the contract, not an accident of the add sequence.

        Innermost means it sees only what the gate admitted, and sees it after
        every other layer has had its say — so the record describes the request
        the route actually answered. Outermost stays the gate: everything else
        in this file assumes an unauthenticated request never reaches a second
        layer at all.
        """
        from fastapi import FastAPI

        app = FastAPI()
        configure_interface_app(app, static_dir=tmp_path)

        assert app.user_middleware[0].cls is WebAuthMiddleware
        assert app.user_middleware[-1].cls is HttpAuditMiddleware


# --------------------------------------------------------------------------- #
# Through the real writer
# --------------------------------------------------------------------------- #


class TestThroughTheRealWriter:
    def test_the_two_surface_names_are_the_documented_literals(self):
        """Pinned as STRINGS, not as symbols imported from the module under test.

        ``web_auth.jsonl`` and ``http_mutation.jsonl`` are file names an
        operator is told to grep (``docs/source/how-to/protected-set.rst``), so
        renaming either constant is a documentation break, not a refactor. The
        separation test below compares against these literals for the same
        reason: with both sides imported from the module, two constants that
        came to hold the SAME string would collapse expected onto observed and
        the suite would stay green while the two surfaces merged into one file.
        """
        assert WEB_AUTH_SURFACE == "web_auth"
        assert HTTP_MUTATION_SURFACE == "http_mutation"
        assert WEB_AUTH_SURFACE != HTTP_MUTATION_SURFACE

    def test_a_gate_refusal_lands_as_one_ledger_line(self, app_stub, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")
        drive(
            WebAuthMiddleware(RecordingApp()),
            http_scope(app=app_stub, headers={AUDIT_SUBJECT_HEADER: "alice"}),
        )

        ledger = audit_root / "svc.terminal" / f"{WEB_AUTH_SURFACE}.jsonl"
        lines = ledger.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["surface"] == WEB_AUTH_SURFACE
        assert record["actor"] == "svc.terminal"
        assert record["decision"] == DECISION_REFUSED
        assert record["reason"] == "missing_credential"
        assert record["posture_source"] == POSTURE_SOURCE_APP
        assert record["session"] is None
        assert "alice" in record["detail"]

    def test_an_admitted_mutation_lands_as_one_ledger_line(self, audit_root, monkeypatch):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")
        drive(HttpAuditMiddleware(RecordingApp()), http_scope(method="POST", path="/api/panels"))

        ledger = audit_root / "svc.terminal" / f"{HTTP_MUTATION_SURFACE}.jsonl"
        lines = ledger.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["subject"] == "POST /api/panels"
        assert record["decision"] == DECISION_ALLOWED
        assert record["posture"] == HTTP_MUTATION_POSTURE

    def test_the_two_emitters_file_under_different_surfaces(self, app_stub, audit_root):
        """Two emitters, two files — asserted against literals, not symbols.

        Refused upgrades belong in ``web_auth`` and admitted mutations in
        ``http_mutation``; that separation is the whole reason
        ``common_middleware`` carries two constants.
        """
        drive(WebAuthMiddleware(RecordingApp()), http_scope(app=app_stub))
        drive(HttpAuditMiddleware(RecordingApp()), http_scope(method="POST"))

        stems = {path.stem for path in audit_root.rglob("*.jsonl")}
        assert stems == {"web_auth", "http_mutation"}


# --------------------------------------------------------------------------- #
# (c) The route can own the decision
# --------------------------------------------------------------------------- #


class TestTheRouteCanOwnTheDecision:
    """The dedup pairing, wired rather than described.

    No route writes to the unified ledger today, so on the running path the
    marker is always absent and these tests drive the seam with a route that
    records. They exist because the one plausible way to get the pairing wrong
    — reading the marker after :func:`~osprey.audit.dedup.decision_scope` has
    closed — silently produces two records, the second of them ``allowed`` on a
    refused request, and nothing else in the suite would say so.
    """

    @pytest.fixture
    def dedup(self):
        """The live ``osprey.audit.dedup``, resolved when the test runs.

        Same reason as the ``audit_writer`` fixture: sibling modules swap
        ``sys.modules``, and a module object bound at collection time can end
        up patching a copy nobody runs.
        """
        return importlib.import_module("osprey.audit.dedup")

    @staticmethod
    def ledger_lines(audit_root) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for path in sorted(audit_root.rglob("*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def recording_route(dedup, *, status: int = 200, raises: bool = False):
        """An ``async def``-equivalent route that refuses and records it itself."""

        async def route(scope, receive, send):
            dedup.record_and_mark(
                decision=DECISION_REFUSED,
                reason="protected_key",
                surface="web_terminal",
                posture=HTTP_MUTATION_POSTURE,
                posture_source=POSTURE_SOURCE_APP,
                session=None,
                subject="POST /api/config",
            )
            if raises:
                raise RuntimeError("route blew up after recording")
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send({"type": "http.response.body", "body": b"refused"})

        return route

    def test_a_route_that_recorded_a_refusal_is_not_recorded_again(
        self, dedup, audit_root, monkeypatch
    ):
        """The headline: one line, ``refused``, filed by the layer that decided.

        Move the ``recorded_decision()`` read one line outside the scope in
        ``HttpAuditMiddleware.__call__`` and this fails with two records, the
        second of them ``allowed`` on a request the route refused.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

        drive(
            HttpAuditMiddleware(self.recording_route(dedup)),
            http_scope(method="POST", path="/api/config"),
        )

        lines = self.ledger_lines(audit_root)
        assert len(lines) == 1
        assert lines[0]["decision"] == DECISION_REFUSED
        assert lines[0]["surface"] == "web_terminal"

    def test_a_route_that_recorded_nothing_is_recorded_here_as_before(
        self, dedup, audit_root, monkeypatch
    ):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

        drive(HttpAuditMiddleware(RecordingApp()), http_scope(method="POST", path="/api/config"))

        lines = self.ledger_lines(audit_root)
        assert len(lines) == 1
        assert lines[0]["surface"] == HTTP_MUTATION_SURFACE
        assert lines[0]["reason"] == REASON_MUTATION

    def test_the_marker_does_not_survive_into_the_next_request(
        self, dedup, audit_root, monkeypatch
    ):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")
        middleware = HttpAuditMiddleware(self.recording_route(dedup))

        drive(middleware, http_scope(method="POST", path="/api/config"))
        drive(
            HttpAuditMiddleware(RecordingApp()),
            http_scope(method="POST", path="/api/panels"),
        )

        decisions = [(line["surface"], line["decision"]) for line in self.ledger_lines(audit_root)]
        assert decisions.count((HTTP_MUTATION_SURFACE, DECISION_ALLOWED)) == 1
        assert len(decisions) == 2

    def test_a_recorder_on_a_worker_thread_is_not_seen(self, dedup, audit_root, monkeypatch):
        """The limitation, pinned: a ``def`` route cannot be an inner recorder.

        Starlette runs a synchronous route through ``run_in_threadpool``, which
        *copies* the context — so the mark dies with the worker thread and this
        layer records over the route's decision. Asserted in the direction it
        actually behaves, because the failure is silent.
        """
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

        def sync_route_body():
            dedup.record_and_mark(
                decision=DECISION_REFUSED,
                reason="protected_key",
                surface="web_terminal",
                posture=HTTP_MUTATION_POSTURE,
                posture_source=POSTURE_SOURCE_APP,
                session=None,
                subject="POST /api/config",
            )

        async def route(scope, receive, send):
            await run_in_threadpool(sync_route_body)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"refused"})

        drive(HttpAuditMiddleware(route), http_scope(method="POST", path="/api/config"))

        surfaces = [line["surface"] for line in self.ledger_lines(audit_root)]
        assert sorted(surfaces) == ["http_mutation", "web_terminal"]

    def test_a_route_that_recorded_and_then_crashed_is_not_recorded_again(
        self, dedup, audit_root, monkeypatch
    ):
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

        with pytest.raises(RuntimeError):
            drive(
                HttpAuditMiddleware(self.recording_route(dedup, raises=True)),
                http_scope(method="POST", path="/api/config"),
            )

        lines = self.ledger_lines(audit_root)
        assert len(lines) == 1
        assert lines[0]["surface"] == "web_terminal"

    def test_an_unstored_inner_record_on_a_refusal_is_filed_by_this_layer(
        self, dedup, audit_root, monkeypatch
    ):
        """A record that never landed plus a 4xx would otherwise be total silence."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

        async def route(scope, receive, send):
            dedup.mark_recorded(DECISION_REFUSED, "protected_key", stored=False)
            await send({"type": "http.response.start", "status": 403, "headers": []})
            await send({"type": "http.response.body", "body": b"refused"})

        drive(HttpAuditMiddleware(route), http_scope(method="POST", path="/api/config"))

        lines = self.ledger_lines(audit_root)
        assert len(lines) == 1
        assert (lines[0]["decision"], lines[0]["reason"]) == (
            DECISION_REFUSED,
            REASON_ROUTE_REFUSED,
        )

    def test_an_unstored_inner_record_on_a_success_is_still_deferred_to(
        self, dedup, audit_root, monkeypatch
    ):
        """Never ``allowed`` over a refusal, even one that reached no ledger."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "svc.terminal")

        async def route(scope, receive, send):
            dedup.mark_recorded(DECISION_REFUSED, "protected_key", stored=False)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        drive(HttpAuditMiddleware(route), http_scope(method="POST", path="/api/config"))

        assert self.ledger_lines(audit_root) == []
