"""A deploy that runs the telemetry store also gets an account that can write to it.

Every other credential ``osprey up`` handles is one it chooses: it generates a
value, writes it into ``.env``, and the container is started with it. The ingest
identity cannot work that way. It is an OpenObserve service account, and the
server issues its token, so the value does not exist until the store is running
and has been asked for it. Provisioning is a harvest, and it happens after the
store is live.

What the tests below hold the provisioner to:

* a token that still works costs one read and produces no output at all, because
  this runs on every start;
* an account that is missing is created, and one whose token the store no longer
  accepts is re-issued (delete plus create: this store has no in-place rotation);
* the store's replies are treated as the store actually replies. A rejection is
  HTTP 401 carrying the bare string ``Unauthorized Access``, a successful ingest
  carries an EMPTY body, and a duplicate create is a 400 that means "already
  done" rather than a failure;
* nothing here is fatal, and nothing here is silent: every failure names the one
  thing to do about it and the deploy carries on;
* the root password is used in this process and reaches no output, no log record
  and no file.

The store is modelled by :class:`FakeStore`, which holds real state (which
accounts exist, which tokens they carry) so that a sequence of calls is checked
against a store that changes rather than against a fixed script.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from osprey.deployment import openobserve_provision as provision
from osprey.deployment.reset import MINTED_ENV_BANNERS
from osprey.utils.dotenv import parse_dotenv_file

TOKEN_VAR = provision.INGEST_TOKEN_VAR
EMAIL_VAR = "ZO_INGEST_USER_EMAIL"
INGEST_EMAIL = "ingest@example.com"
ROOT_EMAIL = "root@example.com"
ROOT_PASSWORD = "Str0ng!rootpass"

CONFIG: dict[str, Any] = {
    "project_name": "demo",
    "deployed_services": ["openobserve"],
    "services": {"openobserve": {"port": 5080}},
    "deployment": {"bind_address": "127.0.0.1"},
}


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class FakeStore:
    """A stateful stand-in for OpenObserve, answering the way the real one does.

    Only the five endpoints the provisioner uses, but with the real store's
    quirks kept: bare-string 401 bodies, a 400 on a duplicate create, and a
    service-account token the SERVER picks rather than the caller.
    """

    def __init__(
        self,
        *,
        accounts: dict[str, str] | None = None,
        org: str = "default",
        healthz_failures: int = 0,
        connect_failures: int = 0,
        root: tuple[str, str] = (ROOT_EMAIL, ROOT_PASSWORD),
        search_status: int = 200,
        refuse: set[str] | None = None,
    ) -> None:
        self.accounts = dict(accounts or {})
        self.org = org
        self.healthz_failures = healthz_failures
        # Calls refused before any server sees them, which is what a host with
        # nothing bound on the store's port does. Distinct from
        # ``healthz_failures``: that one is a store that answered.
        self.connect_failures = connect_failures
        self.root = root
        self.search_status = search_status
        # Tokens the store still lists for an account but no longer accepts.
        # That combination is the one a re-issue exists for, and no consistent
        # store models it, so it is stated rather than derived.
        self.refuse = set(refuse or ())
        self.calls: list[tuple[str, str]] = []
        self.issued: list[str] = []
        self._counter = 0

    # -- helpers ---------------------------------------------------------
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _next_token(self) -> str:
        self._counter += 1
        token = f"tok{self._counter:013d}"  # 16 chars, like the real one
        self.issued.append(token)
        return token

    @staticmethod
    def _credentials(request: httpx.Request) -> tuple[str, str] | None:
        header = request.headers.get("Authorization")
        if not header or not header.startswith("Basic "):
            return None
        identity, _, secret = base64.b64decode(header[6:]).decode().partition(":")
        return identity, secret

    def _is_root(self, request: httpx.Request) -> bool:
        return self._credentials(request) == self.root

    @staticmethod
    def _unauthorized() -> httpx.Response:
        # The real store's rejection: a bare, non-JSON body.
        return httpx.Response(401, text="Unauthorized Access")

    # -- routing ---------------------------------------------------------
    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.connect_failures > 0:
            self.connect_failures -= 1
            # Not an answer of any kind: nothing is listening, so the call never
            # reaches a store and is not recorded as having done so.
            raise httpx.ConnectError("Connection refused", request=request)

        path = request.url.path
        self.calls.append((request.method, path))

        if path == "/healthz":
            if self.healthz_failures > 0:
                self.healthz_failures -= 1
                return httpx.Response(503, text="not ready")
            return httpx.Response(200, json={"status": "ok"})

        prefix = f"/api/{self.org}"
        if path == f"{prefix}/_search":
            credentials = self._credentials(request)
            if credentials is None:
                return self._unauthorized()
            identity, secret = credentials
            if secret in self.refuse:
                return self._unauthorized()
            if credentials == self.root or self.accounts.get(identity) == secret:
                if self.search_status != 200:
                    return httpx.Response(self.search_status, text="stream not found")
                return httpx.Response(200, json={"hits": [], "total": 0})
            return self._unauthorized()

        if path == f"{prefix}/service_accounts" and request.method == "POST":
            if not self._is_root(request):
                return self._unauthorized()
            email = json.loads(request.content)["email"]
            if email in self.accounts:
                return httpx.Response(400, json={"code": 400, "message": "User already exists"})
            self.accounts[email] = self._next_token()
            return httpx.Response(200, json={"code": 200, "message": "User saved successfully"})

        if path.startswith(f"{prefix}/service_accounts/"):
            if not self._is_root(request):
                return self._unauthorized()
            email = path.rsplit("/", 1)[1]
            if request.method == "DELETE":
                self.accounts.pop(email, None)
                return httpx.Response(
                    200, json={"code": 200, "message": "User removed from organization"}
                )
            if email not in self.accounts:
                return httpx.Response(404, json={"code": 404, "message": "User not found"})
            return httpx.Response(200, json={"token": self.accounts[email], "user": email})

        return httpx.Response(404, text="no route")

    # -- readable assertions --------------------------------------------
    @property
    def paths(self) -> list[str]:
        return [path for _method, path in self.calls]

    def touched(self, needle: str) -> bool:
        return any(needle in path for path in self.paths)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def empty_ledger():
    """The closing ledger is module state. A test's rows are its own."""
    from osprey.cli import output

    output._ledger.clear()
    yield
    output._ledger.clear()


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No ZO_* variable leaks in from the shell running the suite.

    ``_effective_value`` reads the process environment ahead of the ``.env``, so
    a stray export would decide half these tests.
    """
    for name in (TOKEN_VAR, EMAIL_VAR, "ZO_ROOT_USER_EMAIL", "ZO_ROOT_USER_PASSWORD"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def env_file(tmp_path) -> Path:
    """A project ``.env`` in the state the mint leaves it in before a harvest."""
    path = tmp_path / ".env"
    path.write_text(
        "# Auto-generated service auth tokens (osprey deploy up)\n"
        f"ZO_ROOT_USER_PASSWORD={ROOT_PASSWORD}\n"
        "\n"
        "# Service account names (osprey up)\n"
        f"ZO_ROOT_USER_EMAIL={ROOT_EMAIL}\n"
        f"{EMAIL_VAR}={INGEST_EMAIL}\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def run(store: FakeStore, env_path: Path, config: dict | None = None, **kwargs):
    """Provision once against ``store``."""
    return provision.provision_ingest_identity(
        config if config is not None else CONFIG,
        env_path,
        transport=store.transport(),
        **kwargs,
    )


def token_in(env_path: Path) -> str | None:
    return parse_dotenv_file(env_path).get(TOKEN_VAR)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_a_project_without_the_store_is_left_alone(env_file):
    """No store, no account: this deploy has no root credential for someone
    else's store and no business creating accounts in it."""
    store = FakeStore()

    outcome = run(store, env_file, {**CONFIG, "deployed_services": ["mongodb"]})

    assert outcome.action == "skipped"
    assert store.calls == []
    assert token_in(env_file) is None


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_provisioning_waits_for_the_store_to_be_ready(env_file, monkeypatch):
    """Running is not ready: the port answers before the API will."""
    monkeypatch.setattr(provision, "READY_POLL_S", 0.0)
    store = FakeStore(healthz_failures=2)

    outcome = run(store, env_file)

    assert outcome.action == "created"
    assert store.paths.count("/healthz") == 3


def test_a_store_that_never_answers_degrades_with_a_remedy(env_file, monkeypatch, capsys):
    """The deploy carries on, and the operator is told what to do."""
    monkeypatch.setattr(provision, "READY_POLL_S", 0.0)
    store = FakeStore(healthz_failures=10_000)

    outcome = run(store, env_file, ready_timeout_s=0.0)

    assert outcome.action == "failed"
    assert outcome.remedy == "Run `osprey up` again once the telemetry store is running."
    assert token_in(env_file) is None
    printed = " ".join(capsys.readouterr().err.split())
    assert "did not answer" in printed
    assert "osprey up" in printed


def test_a_port_nothing_accepts_a_connection_on_is_given_up_on_early(env_file, monkeypatch, capsys):
    """A store nobody started is not a store that is slow to start.

    The runtime binds a published port the moment the container is created, so a
    connection that keeps being refused says the container is not there — and no
    length of waiting produces one. The minute the readiness budget allows is for
    a store that answers; spending it here would make every start against a
    stopped runtime sit silent for a minute before saying so.
    """
    monkeypatch.setattr(provision, "READY_POLL_S", 0.0)
    monkeypatch.setattr(provision, "CONNECT_TIMEOUT_S", 0.05)
    store = FakeStore(connect_failures=10_000)

    started = time.monotonic()
    outcome = run(store, env_file, ready_timeout_s=30.0)
    elapsed = time.monotonic() - started

    assert outcome.action == "failed"
    assert elapsed < 5.0, (
        f"waited {elapsed:.1f}s, which is the readiness budget, not the connect one"
    )
    assert store.calls == []
    printed = " ".join(capsys.readouterr().err.split())
    assert "did not answer" in printed
    # The line reports the wait that happened, not the budget it did not use.
    assert "within 30s" not in printed


def test_a_store_that_answers_at_all_still_gets_the_whole_readiness_budget(env_file, monkeypatch):
    """503 is a store opening its database, and that one IS worth waiting out.

    The short connect budget must not leak into this case: the store is there,
    it is simply not finished, and giving up on it after a few seconds would
    trade a rare slow start for a deploy with no telemetry account.
    """
    monkeypatch.setattr(provision, "READY_POLL_S", 0.01)
    monkeypatch.setattr(provision, "CONNECT_TIMEOUT_S", 0.0)
    store = FakeStore(healthz_failures=10_000)

    started = time.monotonic()
    outcome = run(store, env_file, ready_timeout_s=0.3)
    elapsed = time.monotonic() - started

    assert outcome.action == "failed"
    assert elapsed >= 0.3, f"gave up after {elapsed:.2f}s, inside the readiness budget"
    assert store.paths.count("/healthz") > 1


def test_a_store_whose_port_binds_late_is_still_provisioned(env_file, monkeypatch):
    """The connect budget is a window, not a single attempt.

    A container the runtime has just been asked for takes a moment to exist, so
    the first call or two can be refused on a store that is coming up perfectly
    well.
    """
    monkeypatch.setattr(provision, "READY_POLL_S", 0.0)
    store = FakeStore(connect_failures=2)

    outcome = run(store, env_file)

    assert outcome.action == "created"
    assert store.connect_failures == 0
    assert token_in(env_file) == store.issued[-1]


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def test_a_working_token_is_verified_and_nothing_else_happens(env_file, capsys):
    """The common case, on every start of every deploy: one read, no writes,
    no output."""
    store = FakeStore(accounts={INGEST_EMAIL: "liveliveliveliv"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=liveliveliveliv\n")
    before = env_file.read_text(encoding="utf-8")

    outcome = run(store, env_file)

    assert outcome.action == "verified"
    assert env_file.read_text(encoding="utf-8") == before
    assert not store.touched("service_accounts")
    assert capsys.readouterr().out == ""


def test_verification_reads_rather_than_writes_telemetry(env_file):
    """Checking a credential must not put a row into the operator's data."""
    store = FakeStore(accounts={INGEST_EMAIL: "liveliveliveliv"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=liveliveliveliv\n")

    run(store, env_file)

    assert not store.touched("/v1/logs")
    assert not store.touched("/v1/metrics")
    assert store.touched("/_search")


def test_a_store_with_nothing_ingested_yet_is_not_a_dead_credential(env_file):
    """A search over a stream that does not exist is not a 401.

    Read as one, it would rotate a perfectly good token on every fresh deploy.
    """
    store = FakeStore(accounts={INGEST_EMAIL: "liveliveliveliv"}, search_status=404)
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=liveliveliveliv\n")

    outcome = run(store, env_file)

    assert outcome.action == "verified"
    assert not store.touched("service_accounts")


# ---------------------------------------------------------------------------
# Heal
# ---------------------------------------------------------------------------


def test_a_missing_account_is_created_and_its_token_harvested(env_file):
    """The first deploy: the store has no such account, and the token it issues
    is the one the .env has to end up holding."""
    store = FakeStore()

    outcome = run(store, env_file)

    assert outcome.action == "created"
    assert store.accounts[INGEST_EMAIL] == token_in(env_file)
    assert ("POST", "/api/default/service_accounts") in store.calls


def test_the_account_is_created_without_a_password_or_a_role(env_file):
    """Neither is the caller's to send: the server issues the token, and the
    role of a service account is not chosen here."""
    sent: list[dict] = []
    store = FakeStore()
    handler = store.handle

    def record(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/service_accounts"):
            sent.append(json.loads(request.content))
        return handler(request)

    provision.provision_ingest_identity(CONFIG, env_file, transport=httpx.MockTransport(record))

    assert sent and set(sent[0]) == {"email", "first_name", "last_name", "organization"}


def test_an_env_with_no_token_beside_a_store_that_has_the_account_just_harvests(env_file):
    """The .env was out of date, not wrong. Nothing is deleted and nothing is
    re-issued: the live token is read back."""
    store = FakeStore(accounts={INGEST_EMAIL: "aliveandwellxyz"})

    outcome = run(store, env_file)

    assert outcome.action == "harvested"
    assert token_in(env_file) == "aliveandwellxyz"
    assert ("DELETE", f"/api/default/service_accounts/{INGEST_EMAIL}") not in store.calls
    assert store.issued == []


def test_a_dead_token_is_reissued_by_delete_and_create(env_file):
    """The store's only rotation. A PUT answers 200 and re-issues nothing, so a
    provisioner that used it would report a rotation that never happened."""
    # The store still lists this token for the account and no longer accepts it.
    store = FakeStore(accounts={INGEST_EMAIL: "deaddeaddeaddea"}, refuse={"deaddeaddeaddea"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=deaddeaddeaddea\n")

    outcome = run(store, env_file)

    assert outcome.action == "rotated"
    assert ("DELETE", f"/api/default/service_accounts/{INGEST_EMAIL}") in store.calls
    assert token_in(env_file) == store.accounts[INGEST_EMAIL] != "deaddeaddeaddea"
    assert not any(method == "PUT" for method, _path in store.calls)


def test_a_stale_env_token_beside_a_live_account_is_replaced_without_rotating(env_file):
    """The account is fine; only this deployment's copy was stale. Deleting the
    account here would break anything else already ingesting with it."""
    store = FakeStore(accounts={INGEST_EMAIL: "aliveandwellxyz"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=stalestalestale\n")

    outcome = run(store, env_file)

    assert outcome.action == "harvested"
    assert token_in(env_file) == "aliveandwellxyz"
    assert store.issued == []


def test_a_duplicate_create_is_not_a_failure(env_file):
    """Two runs of an idempotent step must not fight. The store answers a
    duplicate create with 400 'User already exists'."""
    store = FakeStore()

    def racing(request: httpx.Request) -> httpx.Response:
        # The account appears between the read that missed it and the create.
        if request.method == "GET" and request.url.path.endswith(INGEST_EMAIL):
            if INGEST_EMAIL not in store.accounts and not store.issued:
                store.accounts[INGEST_EMAIL] = "racedintobeingx"
                store.calls.append((request.method, request.url.path))
                return httpx.Response(404, json={"code": 404, "message": "User not found"})
        return store.handle(request)

    outcome = provision.provision_ingest_identity(
        CONFIG, env_file, transport=httpx.MockTransport(racing)
    )

    assert outcome.action == "created"
    assert token_in(env_file) == "racedintobeingx"


def test_the_volume_was_replaced_under_a_surviving_env(env_file):
    """The inverse of the familiar stale-volume trap, and the case this whole
    provisioner exists for: the account lived in the volume, the token lived in
    the .env, and only one of them survived."""
    store = FakeStore()  # fresh volume: no accounts at all
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=fromtheoldvolum\n")

    outcome = run(store, env_file)

    assert outcome.action == "created"
    assert token_in(env_file) == store.accounts[INGEST_EMAIL] != "fromtheoldvolum"


# ---------------------------------------------------------------------------
# Robustness against what this store actually returns
# ---------------------------------------------------------------------------


def test_a_bare_string_401_body_is_not_parsed_as_json(env_file):
    """The rejection body is the bare string 'Unauthorized Access'. Anything
    that called .json() on it would raise instead of healing."""
    store = FakeStore(accounts={INGEST_EMAIL: "goodgoodgoodgoo"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=wrongwrongwrong\n")

    outcome = run(store, env_file)

    assert outcome.action == "harvested"
    assert token_in(env_file) == "goodgoodgoodgoo"


def test_an_empty_success_body_is_tolerated(env_file):
    """Some successful calls answer with no body at all."""
    store = FakeStore(accounts={INGEST_EMAIL: "liveliveliveliv"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=liveliveliveliv\n")

    def empty_bodied(request: httpx.Request) -> httpx.Response:
        response = store.handle(request)
        if request.url.path.endswith("/_search") and response.status_code == 200:
            return httpx.Response(200, content=b"")
        return response

    outcome = provision.provision_ingest_identity(
        CONFIG, env_file, transport=httpx.MockTransport(empty_bodied)
    )

    assert outcome.action == "verified"


def test_a_bare_text_error_from_the_create_degrades_rather_than_raising(env_file, capsys):
    """Role and field errors come back as a bare serde message, not JSON."""
    store = FakeStore()

    def refusing(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/service_accounts"):
            return httpx.Response(400, text="Json deserialize error: missing field `role`")
        return store.handle(request)

    outcome = provision.provision_ingest_identity(
        CONFIG, env_file, transport=httpx.MockTransport(refusing)
    )

    assert outcome.action == "failed"
    assert token_in(env_file) is None
    assert "refused to create" in " ".join(capsys.readouterr().err.split())


def test_a_root_credential_the_store_rejects_is_named_as_such(env_file, capsys):
    """The likeliest real failure here: a data volume kept from an earlier
    deploy still holds the root password it was created with."""
    store = FakeStore(root=("root@example.com", "whatthevolumeactuallyhas"))

    outcome = run(store, env_file)

    assert outcome.action == "failed"
    assert "refused this deployment's root credential" in outcome.problem
    assert "ZO_ROOT_USER_PASSWORD" in outcome.remedy
    assert token_in(env_file) is None


def test_a_store_that_stops_answering_mid_sequence_degrades(env_file, capsys):
    """A transport error is not an answer about the credential and must not be
    read as one."""
    store = FakeStore()

    def dropping(request: httpx.Request) -> httpx.Response:
        if "service_accounts" in request.url.path:
            raise httpx.ConnectError("connection reset")
        return store.handle(request)

    outcome = provision.provision_ingest_identity(
        CONFIG, env_file, transport=httpx.MockTransport(dropping)
    )

    assert outcome.action == "failed"
    assert outcome.remedy == "Run `osprey up` again once the telemetry store is running."
    assert token_in(env_file) is None


def test_a_reissued_token_the_store_then_refuses_is_reported(env_file):
    """Writing a token the store will not accept would leave a .env that looks
    healthy and telemetry that silently 401s."""
    store = FakeStore(accounts={INGEST_EMAIL: "deaddeaddeaddea"})
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{TOKEN_VAR}=deaddeaddeaddea\n")

    def never_accepts(request: httpx.Request) -> httpx.Response:
        response = store.handle(request)
        if request.url.path.endswith("/_search"):
            # Every ingest credential is refused, including the one just issued.
            return httpx.Response(401, text="Unauthorized Access")
        return response

    outcome = provision.provision_ingest_identity(
        CONFIG, env_file, transport=httpx.MockTransport(never_accepts)
    )

    assert outcome.action == "failed"
    assert token_in(env_file) == "deaddeaddeaddea"  # unchanged: nothing better to write


# ---------------------------------------------------------------------------
# What lands in the .env
# ---------------------------------------------------------------------------


def test_the_first_harvest_lands_under_the_banner_reset_knows(env_file):
    """A block `osprey reset` cannot see is a dead credential left beside a
    store that no longer has the account it belonged to."""
    store = FakeStore()

    run(store, env_file)

    text = env_file.read_text(encoding="utf-8")
    assert f"# {provision.HARVEST_BANNER}\n{TOKEN_VAR}=" in text
    assert provision.HARVEST_BANNER in MINTED_ENV_BANNERS
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_a_reissue_replaces_the_line_instead_of_stacking_a_second_one(env_file):
    """Appending would read correctly (the last assignment wins) and would leave
    one dead credential in the file per rotation the deployment ever does."""
    store = FakeStore()

    run(store, env_file)
    first = token_in(env_file)
    # The account's token changed under this deployment, so the next run heals.
    store.accounts[INGEST_EMAIL] = "rotatedelsewher"
    run(store, env_file)

    text = env_file.read_text(encoding="utf-8")
    assert text.count(f"{TOKEN_VAR}=") == 1
    assert token_in(env_file) == "rotatedelsewher" != first
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_the_operators_own_lines_survive_a_reissue(env_file):
    """A provisioner edits one line. It does not reformat somebody's file."""
    store = FakeStore()
    run(store, env_file)
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write("\n# my own note\nOLOG_PASSWORD=hunter2\n")
    store.accounts[INGEST_EMAIL] = "rotatedelsewher"

    run(store, env_file)

    text = env_file.read_text(encoding="utf-8")
    assert "# my own note\nOLOG_PASSWORD=hunter2\n" in text
    assert parse_dotenv_file(env_file)["ZO_ROOT_USER_EMAIL"] == ROOT_EMAIL


def test_a_second_run_changes_nothing(env_file):
    """Idempotent, because it runs on every start."""
    store = FakeStore()
    run(store, env_file)
    after_first = env_file.read_text(encoding="utf-8")
    store.calls.clear()

    outcome = run(store, env_file)

    assert outcome.action == "verified"
    assert env_file.read_text(encoding="utf-8") == after_first
    assert not store.touched("service_accounts")


def test_the_process_copy_is_kept_in_step_only_where_it_already_exists(env_file, monkeypatch):
    """A stale empty copy in the environment outranks the freshly written line
    for compose's interpolation; a name that was never there stays out."""
    monkeypatch.setenv(TOKEN_VAR, "")
    compose_env = {TOKEN_VAR: "", "OTHER": "x"}
    store = FakeStore()

    run(store, env_file, env=compose_env)

    harvested = token_in(env_file)
    assert os.environ[TOKEN_VAR] == harvested
    assert compose_env[TOKEN_VAR] == harvested


def test_an_environment_that_never_carried_the_name_does_not_gain_it(env_file):
    """It reaches the containers through --env-file, as it always has."""
    compose_env: dict[str, str] = {"OTHER": "x"}
    store = FakeStore()

    run(store, env_file, env=compose_env)

    assert TOKEN_VAR not in compose_env
    assert TOKEN_VAR not in os.environ


def test_a_token_a_compose_env_file_cannot_carry_is_refused(env_file, capsys):
    """Compose mangles a `$` on both routes out of this file, so a value
    carrying one reaches the container truncated."""
    store = FakeStore(accounts={INGEST_EMAIL: "has$dollar$sign"})

    outcome = run(store, env_file)

    assert outcome.action == "failed"
    assert token_in(env_file) is None


# ---------------------------------------------------------------------------
# Address, organization, and the credentials this reads
# ---------------------------------------------------------------------------


def test_the_organization_comes_from_the_setting_the_agent_uses(env_file):
    """The account is created in the organization the agent's own OTLP endpoint
    names. Two answers here would create it in one and use it against another."""
    config = {
        **CONFIG,
        "claude_code": {"telemetry": {"openobserve": {"org": "als"}}},
    }
    store = FakeStore(org="als")

    outcome = run(store, env_file, config)

    assert outcome.action == "created"
    assert all(path.startswith(("/healthz", "/api/als/")) for path in store.paths)


def test_the_organization_is_resolved_the_way_the_agent_resolves_it(env_file):
    """One config, one organization — including where the setting is odd.

    The two sides read the same key through different code, so the risk is not a
    typo but a disagreement about what a value MEANS. The resolver applies its
    default only when the key is absent; an explicit ``org: ""`` is an empty
    organization to it, and the agent exports to ``/api/``. A provisioner that
    read the empty string as "unset" would create the ingest account in
    ``default`` and leave the panel the agent writes to empty, with nothing
    failing anywhere. Asserted against the resolver itself rather than against a
    hand-copied rule, so a change to either side has to change both.
    """
    from osprey.build.claude_code_telemetry import _resolve_telemetry_endpoint

    def agent_org(telemetry: dict) -> str:
        endpoint = _resolve_telemetry_endpoint(
            {"backend": "openobserve", **telemetry}, in_container=False
        )
        return endpoint.rpartition("/api/")[2]

    for telemetry in (
        {},
        {"openobserve": {}},
        {"openobserve": {"org": "als"}},
        {"openobserve": {"org": ""}},
    ):
        config = {**CONFIG, "claude_code": {"telemetry": telemetry}}
        assert provision.store_org(config) == agent_org(telemetry), telemetry


def test_the_address_is_the_one_this_deploy_publishes(env_file):
    """Read the way the health category reads it, so the two cannot disagree."""
    config = {
        **CONFIG,
        "services": {"openobserve": {"port": 15080}},
        "deployment": {"bind_address": "0.0.0.0"},
    }

    assert provision.store_base_url(config) == "http://0.0.0.0:15080"
    assert provision.store_base_url(CONFIG) == "http://127.0.0.1:5080"
    assert provision.store_org(CONFIG) == "default"


def test_an_ingest_account_with_no_name_is_reported(tmp_path, capsys):
    """The token is meaningless without the identity it was issued for."""
    env_path = tmp_path / ".env"
    env_path.write_text(f"ZO_ROOT_USER_PASSWORD={ROOT_PASSWORD}\n", encoding="utf-8")
    store = FakeStore()

    outcome = run(store, env_path)

    assert outcome.action == "failed"
    assert EMAIL_VAR in outcome.problem
    assert store.calls == []


def test_a_deployment_with_no_root_credential_is_reported(tmp_path, capsys):
    """Healing is done AS root. Without one there is nothing to do but say so."""
    env_path = tmp_path / ".env"
    env_path.write_text(f"{EMAIL_VAR}={INGEST_EMAIL}\n", encoding="utf-8")
    store = FakeStore()

    outcome = run(store, env_path)

    assert outcome.action == "failed"
    assert "ZO_ROOT_USER_PASSWORD" in outcome.remedy
    assert not store.touched("service_accounts")


def test_the_identity_variable_is_read_from_the_registry(env_file):
    """Spelled once, in the registry that records where the credential comes
    from, rather than again here."""
    from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS

    assert _STORE_ISSUED_VARS[TOKEN_VAR].identity_var == EMAIL_VAR
    assert _STORE_ISSUED_VARS[TOKEN_VAR].service == provision.SERVICE


# ---------------------------------------------------------------------------
# What the operator is told
# ---------------------------------------------------------------------------


def test_the_scope_of_the_account_is_disclosed_when_it_is_granted(env_file, capsys):
    """OpenObserve has no ingest-only role in any edition: this account reads
    back every log and metric and lists the store's users. An operator learns
    that at the moment it is granted, not from a document nobody opens."""
    store = FakeStore()

    run(store, env_file)

    printed = " ".join(capsys.readouterr().out.split())
    assert "read all telemetry and the user roster" in printed
    assert "no ingest-only role in any edition" in printed


def test_the_root_password_reaches_no_output_and_no_log(env_file, capsys, caplog):
    """It is read host-side, used in this process, and goes nowhere else."""
    import logging

    caplog.set_level(logging.DEBUG)
    store = FakeStore()

    run(store, env_file)

    captured = capsys.readouterr()
    assert ROOT_PASSWORD not in captured.out + captured.err + caplog.text


def test_the_harvested_token_is_never_printed(env_file, capsys):
    """The ledger names the variable. It does not quote the secret."""
    from osprey.cli import output

    store = FakeStore()

    run(store, env_file)

    captured = capsys.readouterr()
    harvested = token_in(env_file)
    assert harvested and harvested not in captured.out + captured.err
    # The variable name reaches the operator through the closing ledger, which
    # the owning verb flushes after its summary card.
    rows = [value for label, value in output._ledger if label == ".env"]
    assert any(TOKEN_VAR in row for row in rows)
    assert not any(harvested in row for row in rows)


# ---------------------------------------------------------------------------
# The deploy-path call site
# ---------------------------------------------------------------------------


def test_the_store_is_staged_before_the_stack_and_then_provisioned(monkeypatch, tmp_path):
    """The credential can only be harvested from a store that is running, and an
    attached `osprey up` never returns to do it afterwards."""
    from osprey.deployment import container_lifecycle

    commands: list[list[str]] = []
    seen: list[Path] = []
    (tmp_path / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(container_lifecycle, "runtime_env", lambda config, env: dict(env))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle, "compose_base_cmd", lambda *a, **k: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle, "_env_file_args", lambda root=None, provider=None: [])
    monkeypatch.setattr(
        container_lifecycle, "run_captured", lambda cmd, **kw: commands.append(list(cmd))
    )
    monkeypatch.setattr(
        provision,
        "provision_ingest_identity",
        lambda config, env_path, **kw: (
            seen.append(Path(env_path)) or provision.ProvisionOutcome("created")
        ),
    )

    container_lifecycle._stage_openobserve_identity(CONFIG, ["compose.yml"], {}, tmp_path)

    assert commands == [["docker", "compose", "up", "-d", "openobserve"]]
    assert seen == [tmp_path / ".env"]


def test_a_project_without_the_store_stages_nothing(monkeypatch, tmp_path):
    """No compose call, no provisioning."""
    from osprey.deployment import container_lifecycle

    commands: list[list[str]] = []
    monkeypatch.setattr(
        container_lifecycle, "run_captured", lambda cmd, **kw: commands.append(list(cmd))
    )

    container_lifecycle._stage_openobserve_identity(
        {**CONFIG, "deployed_services": []}, ["compose.yml"], {}, tmp_path
    )

    assert commands == []
