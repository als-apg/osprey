"""Provision the OpenObserve ingest identity against a store that is already up.

Every other credential this deploy handles is one OSPREY chooses: the mint in
:func:`osprey.deployment.container_lifecycle._ensure_service_tokens` generates a
value, writes it into the project ``.env``, and compose hands it to the
container that will accept it. The ingest identity inverts that. It is an
OpenObserve **service account**, and a service account's token is issued by the
server: it does not exist until the store is running and has been asked to
create the account, so there is nothing to write before ``compose up`` and no
value for a generator to invent. Provisioning is therefore a *harvest*, and it
runs after the store is live.

The sequence is verify-then-heal, and it is idempotent by design because it runs
on every start:

1. wait for ``GET /healthz`` (running is not ready);
2. **verify** the token the ``.env`` already carries by making one read-only
   call as the ingest identity. A store that accepts it is a store that needs
   nothing, and the run says nothing at all;
3. **heal** as root, and only then: read the account back (``404`` means it was
   never created, or its data volume was replaced), create it when it is
   missing, and re-issue the token when the one the store holds does not
   authenticate. OpenObserve has no in-place rotation for a service account
   token, so re-issuing is delete plus create plus read back;
4. write the harvested token into the ``.env`` and say so.

Two properties of the store shape the code and are easy to get wrong:

* **its bodies are not reliably JSON.** A rejection is HTTP 401 with the bare
  string ``Unauthorized Access``; a successful ingest returns an EMPTY body; a
  malformed field is answered with a bare serde message. Nothing here calls
  ``.json()`` without tolerating a parse failure.
* **the identity lives in the data volume and the token lives in the ``.env``.**
  Recreating ``openobserve_data`` deletes the account while the ``.env`` keeps
  its now-dead secret, which is exactly the case step 3 repairs.

Nothing here is fatal. Telemetry is one panel of a control room, and a deploy
whose channels, plan runs and archive are all up should not be refused because
its ingest account could not be provisioned. Every failure warns with the one
thing to do about it, and the deploy carries on.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from osprey.cli import output
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

logger = get_logger("deployment.openobserve")

#: The ``deployed_services`` key whose store issues the credential.
SERVICE = "openobserve"

#: The ``.env`` name carrying the harvested token. Registered in
#: ``container_lifecycle._STORE_ISSUED_VARS``, which is where the identity var
#: is read from rather than spelled again.
INGEST_TOKEN_VAR = "ZO_INGEST_SA_TOKEN"

#: The root account this deploy authenticates AS while healing. Read host-side,
#: used in-process, and never written anywhere: not into a log record, not into
#: a printed line, not into the ``.env`` block below (it is already there, put
#: there by the mint).
ROOT_EMAIL_VAR = "ZO_ROOT_USER_EMAIL"
ROOT_PASSWORD_VAR = "ZO_ROOT_USER_PASSWORD"

#: Header for the harvested block. ``osprey reset`` strips only the blocks whose
#: banner it knows (``reset.MINTED_ENV_BANNERS``), so this string is half of a
#: contract: a token left behind by a reset is a dead credential sitting beside
#: a store that no longer has the account it belonged to.
HARVEST_BANNER = "Harvested OpenObserve ingest token (osprey up)"

# Store address defaults, mirroring the openobserve compose template and the
# health category that probes the same endpoint.
_DEFAULT_BIND = "127.0.0.1"
_DEFAULT_PORT = 5080
_DEFAULT_ORG = "default"

#: Readiness budget. The store answered ``/healthz`` about a second after start
#: when this was measured, so the budget is for a loaded host rather than for a
#: slow store, and a store that is still silent at the end of it is reported
#: rather than waited on forever.
READY_TIMEOUT_S = 60.0
READY_POLL_S = 1.0

#: How long the store gets to accept a TCP connection at all, which is a much
#: smaller budget than the one above and is spent before it. A published port is
#: accepted by the runtime's proxy the moment the container starts, so a
#: connection refused over and over is not a store that is still waking up: it is
#: no store at all, and waiting out ``READY_TIMEOUT_S`` for one only delays the
#: warning. See :func:`wait_for_store`.
CONNECT_TIMEOUT_S = 5.0

#: Per-request budget. Every call here is a local HTTP round trip.
REQUEST_TIMEOUT_S = 10.0

#: What an operator does about any failure below. One command, because the
#: whole sequence is idempotent: a later start re-runs it from wherever it got
#: to, and nothing here leaves half-written state behind.
_RETRY_REMEDY = "Run `osprey up` again once the telemetry store is running."


class ProvisionOutcome(NamedTuple):
    """What one provisioning run did, in a form a caller can act on.

    ``action`` is one of:

    * ``skipped`` — this project does not deploy the store;
    * ``verified`` — the token on file works. Nothing was written and nothing
      was printed;
    * ``harvested`` — the store already had the account, and its token was
      read back and written to the ``.env``;
    * ``created`` — the account did not exist, so it was created and its token
      harvested;
    * ``rotated`` — the account existed but its token no longer authenticated,
      so a new one was issued;
    * ``failed`` — nothing was written. ``problem`` and ``remedy`` carry what
      an operator is told.
    """

    action: str
    problem: str | None = None
    remedy: str | None = None

    @property
    def wrote_token(self) -> bool:
        """True when this run put a new value into the project ``.env``."""
        return self.action in {"harvested", "created", "rotated"}


class _HealFailed(Exception):
    """One healing step could not be completed. Carries what to tell the operator."""

    def __init__(self, problem: str, remedy: str = _RETRY_REMEDY) -> None:
        super().__init__(problem)
        self.problem = problem
        self.remedy = remedy


def _report_fact(message: str, /, *, wrote: tuple[str, object] | None = None) -> None:
    """Report ``message`` under this module's logger (see ``output.report_fact``)."""
    output.report_fact(logger, message, wrote=wrote)


def _warn_fact(summary: str, detail: str | None = None, remedy: str | None = None, /) -> None:
    """Warn under this module's logger (see ``output.warn_fact``)."""
    output.warn_fact(logger, summary, detail, remedy)


def store_deployed(config: Mapping[str, Any]) -> bool:
    """True when this project runs the telemetry store itself.

    Membership in ``deployed_services`` is the whole gate, and it is the same
    gate the ingest account NAME is written under: a project pointed at a store
    somebody else runs has no root credential for it and no business creating
    accounts in it.
    """
    return SERVICE in (config.get("deployed_services") or [])


def store_base_url(config: Mapping[str, Any]) -> str:
    """The store's host address, as this deploy publishes it.

    Read exactly the way the ``openobserve`` health category reads it, so the
    provisioner and the health row can never disagree about which store they are
    talking about.
    """
    services = config.get("services") or {}
    settings = services.get(SERVICE) or {}
    bind = (config.get("deployment") or {}).get("bind_address", _DEFAULT_BIND)
    port = settings.get("port", _DEFAULT_PORT)
    return f"http://{bind}:{port}"


def store_org(config: Mapping[str, Any]) -> str:
    """The OpenObserve organization every API path is scoped to.

    Taken from ``claude_code.telemetry.openobserve.org``, which is the setting
    the agent's own OTLP endpoint is derived from
    (:func:`osprey.build.claude_code_telemetry._resolve_telemetry_endpoint`).
    Both sides must name the same organization or the account is created in one
    and used against another, so the resolution is the resolver's, directive for
    directive: the default applies when the key is ABSENT, and a key that is
    present takes its value as written.

    That distinction is the whole point. An explicit ``org: ""`` reaches the
    resolver as an empty organization and the agent exports to ``/api/``; the
    reading that treated the empty string as "unset" would send the provisioner
    to ``default`` instead, and the ingest account would be created in an
    organization nothing writes to — a telemetry panel that stays empty with
    every credential valid and every request answered ``200``. Whether an empty
    org is a sensible thing to configure is a question for the resolver, and it
    gets one answer for both callers either way.
    """
    telemetry = (config.get("claude_code") or {}).get("telemetry") or {}
    openobserve = telemetry.get("openobserve") or {}
    return str(openobserve.get("org", _DEFAULT_ORG))


def _basic_auth(identity: str, secret: str) -> str:
    """The ``Authorization`` value for one identity.

    A service account presents its server-issued token exactly where a password
    would go, so both credentials are encoded the same way.
    """
    encoded = base64.b64encode(f"{identity}:{secret}".encode()).decode("ascii")
    return f"Basic {encoded}"


def _token_from(response: httpx.Response) -> str | None:
    """The ``token`` field of a service-account read, if the body carries one.

    Tolerant on purpose: this store answers some requests with a bare string and
    some with an empty body, so a parse failure is an absent token rather than a
    crash.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    token = body.get("token")
    return str(token) if token else None


def _says_already_exists(response: httpx.Response) -> bool:
    """True when a create was refused because the account is already there.

    The store answers a duplicate create with ``400 {"code":400,"message":"User
    already exists"}``. That is not a failure for a provisioner that runs on
    every start: it means another run, or another deploy of the same project,
    got there first. Read out of the message text when the body parses, and out
    of the raw text when it does not.
    """
    try:
        body = response.json()
        message = str(body.get("message", "")) if isinstance(body, dict) else ""
    except ValueError:
        message = response.text
    return "already exists" in message.lower()


def wait_for_store(
    client: httpx.Client, base_url: str, *, deadline: float, poll_s: float | None = None
) -> bool:
    """Block until ``GET /healthz`` answers 200, or give up at ``deadline``.

    Running is not ready: the container accepts a connection before it will
    accept an API call, and provisioning against a store that is still opening
    its database reads as an authentication failure. Returns False rather than
    raising, because a store that never came up is reported like every other
    problem here and does not stop the deploy.

    The wait is in two phases, because the two ways this can fail deserve very
    different patience. A published port is bound by the runtime's own proxy as
    soon as the container is created, so a connection that is *refused* is not a
    slow store — it is the absence of one, and no amount of waiting produces a
    container nobody started. The first connection therefore has only
    ``CONNECT_TIMEOUT_S`` to happen; a wait that never gets one gives up there.
    Once any connection has been accepted — an answer of any status, including
    the ``503`` of a store still opening its database — the full ``deadline``
    applies unchanged, which is the case the long budget was chosen for.

    ``poll_s`` resolves the module default at CALL time rather than binding it
    as a default argument, so a test can shorten the wait without reaching into
    the ``time`` module (patching that is process-wide, not call-scoped).
    ``CONNECT_TIMEOUT_S`` is read the same way, for the same reason.
    """
    poll_s = READY_POLL_S if poll_s is None else poll_s
    connect_deadline = time.monotonic() + CONNECT_TIMEOUT_S
    connected = False
    while True:
        try:
            response = client.get(f"{base_url}/healthz")
        except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as exc:
            # Nothing accepted the connection, so nothing is publishing the port.
            # ``ConnectTimeout`` is named beside ``ConnectError`` because httpx
            # descends the two from different parents, and ``OSError`` covers the
            # raw ``ConnectionRefusedError`` a client that is not httpx's raises.
            logger.debug(f"Telemetry store not accepting connections yet: {exc}")
        except httpx.HTTPError as exc:
            # A transport failure that is not a connect failure — a read that
            # timed out, a truncated response — happens on a connection that WAS
            # accepted, so it counts as one.
            connected = True
            logger.debug(f"Telemetry store not ready yet: {exc}")
        else:
            connected = True
            if response.status_code == 200:
                return True
            logger.debug(f"Telemetry store not ready yet: HTTP {response.status_code}")
        now = time.monotonic()
        if now >= deadline:
            return False
        if not connected and now >= connect_deadline:
            return False
        time.sleep(poll_s)


def _identity_accepted(
    client: httpx.Client, base_url: str, org: str, identity: str, token: str
) -> bool:
    """Does the store still accept this identity and token?

    One READ, authenticated as the ingest identity itself. Deliberately not an
    ingest: writing telemetry to test a credential puts a row into the operator's
    data every time a deploy starts, and the question being asked is about the
    credential rather than about the ingest path.

    The rule is narrow on purpose. Only HTTP 401 means the credential is dead;
    the store answers a search over a stream that does not exist yet with
    something else entirely, and treating that as a dead credential would rotate
    a perfectly good token on every fresh deploy. Anything that is an answer at
    all means the store authenticated the caller.

    :raises _HealFailed: when the store could not be reached, which is not an
        answer about the credential and must not be read as one.
    """
    now_us = int(time.time() * 1_000_000)
    # The stream named here need not exist. A store with nothing ingested yet
    # answers with something other than 401, which is all this asks about.
    payload = {
        "query": {
            "sql": 'SELECT * FROM "default"',
            "start_time": now_us - 60_000_000,
            "end_time": now_us,
            "from": 0,
            "size": 1,
        }
    }
    try:
        response = client.post(
            f"{base_url}/api/{org}/_search",
            params={"type": "logs"},
            json=payload,
            headers={"Authorization": _basic_auth(identity, token)},
        )
    except (httpx.HTTPError, OSError) as exc:
        raise _HealFailed(
            f"The telemetry store stopped answering while its ingest account was being "
            f"checked: {exc}"
        ) from exc
    return response.status_code != 401


def _read_service_account(
    client: httpx.Client, base_url: str, org: str, identity: str, root: str
) -> tuple[int, str | None]:
    """Read one service account as root, returning ``(status, token)``.

    ``404`` is the store saying the account does not exist, which is the normal
    state of a first deploy and of a deploy whose data volume was replaced.
    """
    try:
        response = client.get(
            f"{base_url}/api/{org}/service_accounts/{identity}",
            headers={"Authorization": root},
        )
    except (httpx.HTTPError, OSError) as exc:
        raise _HealFailed(
            f"The telemetry store could not be asked about its ingest account: {exc}"
        ) from exc
    if response.status_code == 200:
        return 200, _token_from(response)
    if response.status_code == 404:
        return 404, None
    if response.status_code == 401:
        # The likeliest real failure on this path, and the one worth naming: a
        # data volume kept from an earlier deploy still holds the root password
        # it was created with, so the value in this `.env` is not the store's.
        raise _HealFailed(
            "The telemetry store refused this deployment's root credential, so its ingest "
            "account could not be provisioned.",
            f"Check {ROOT_EMAIL_VAR} and {ROOT_PASSWORD_VAR} against the store. A data volume "
            "kept from an earlier deploy still holds the password it was created with.",
        )
    raise _HealFailed(
        f"The telemetry store answered HTTP {response.status_code} when asked about its "
        "ingest account, so no token could be read back."
    )


def _create_service_account(
    client: httpx.Client, base_url: str, org: str, identity: str, root: str
) -> None:
    """Create the ingest service account as root.

    No password and no role field: the server issues the token, and a service
    account's role is the one thing about it this deploy does not choose. A
    duplicate create is tolerated, because two runs of an idempotent step must
    not fight.
    """
    try:
        response = client.post(
            f"{base_url}/api/{org}/service_accounts",
            json={
                "email": identity,
                "first_name": "OSPREY",
                "last_name": "Ingest",
                "organization": org,
            },
            headers={"Authorization": root},
        )
    except (httpx.HTTPError, OSError) as exc:
        raise _HealFailed(
            f"The telemetry store could not be asked to create its ingest account: {exc}"
        ) from exc
    if response.status_code == 200:
        return
    if response.status_code == 400 and _says_already_exists(response):
        return
    raise _HealFailed(
        f"The telemetry store refused to create its ingest account with HTTP "
        f"{response.status_code}."
    )


def _delete_service_account(
    client: httpx.Client, base_url: str, org: str, identity: str, root: str
) -> None:
    """Remove the ingest service account as root, so a new token can be issued.

    Half of the only rotation this store has. ``PUT`` on a service account
    returns 200 and edits the name; it does NOT re-issue the token, so a
    provisioner that used it would report a rotation that never happened.
    """
    try:
        response = client.delete(
            f"{base_url}/api/{org}/service_accounts/{identity}",
            headers={"Authorization": root},
        )
    except (httpx.HTTPError, OSError) as exc:
        raise _HealFailed(
            f"The telemetry store could not be asked to remove its ingest account: {exc}"
        ) from exc
    if response.status_code not in (200, 404):
        raise _HealFailed(
            f"The telemetry store refused to remove its ingest account with HTTP "
            f"{response.status_code}, so its token could not be re-issued."
        )


def _issue_and_read(client: httpx.Client, base_url: str, org: str, identity: str, root: str) -> str:
    """Create the account and read its token back, or say why neither happened."""
    _create_service_account(client, base_url, org, identity, root)
    _status, token = _read_service_account(client, base_url, org, identity, root)
    if not token:
        raise _HealFailed(
            "The telemetry store created its ingest account but returned no token for it."
        )
    return token


def _heal(
    client: httpx.Client,
    base_url: str,
    org: str,
    identity: str,
    dead_token: str,
    root: str,
) -> tuple[str, str]:
    """Bring the store's ingest account into a state the ``.env`` can record.

    Returns ``(action, token)``. Three shapes, in the order they are cheapest to
    rule out:

    * the account is absent — create it and read its token back;
    * the account is present and its token is not the one that just failed —
      the ``.env`` was simply out of date, so read the live token back;
    * the account is present and its token IS the one that just failed — the
      only fix the store offers is a re-issue, which is delete plus create.
    """
    status, stored = _read_service_account(client, base_url, org, identity, root)
    if status == 404 or stored is None:
        return "created", _issue_and_read(client, base_url, org, identity, root)

    if stored != dead_token and _identity_accepted(client, base_url, org, identity, stored):
        return "harvested", stored

    _delete_service_account(client, base_url, org, identity, root)
    reissued = _issue_and_read(client, base_url, org, identity, root)
    if not _identity_accepted(client, base_url, org, identity, reissued):
        raise _HealFailed(
            "The telemetry store issued a new ingest token that it then refused, so the "
            "agent's telemetry would not be accepted."
        )
    return "rotated", reissued


def _write_token(env_path: Path, token: str) -> None:
    """Record the harvested token in the project ``.env``.

    Two shapes, because this runs again every time the deploy does. A first
    harvest appends a commented block through the writer every deploy-time
    provisioner shares (which owns the trailing newline and the ``0600`` mode),
    so ``osprey reset`` finds a banner it knows. A re-issue REPLACES the line
    already on file, rewritten whole through the atomic writer: appending a
    second assignment would work (the last one wins on read) but would leave a
    dead credential in the file for every rotation the deployment ever does.
    """
    from osprey.deployment.container_lifecycle import _append_env_block
    from osprey.utils.dotenv import atomic_write, dotenv_line_var, format_env_line

    line = format_env_line(INGEST_TOKEN_VAR, token)
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if any(dotenv_line_var(existing) == INGEST_TOKEN_VAR for existing in lines):
            rewritten: list[str] = []
            for existing in lines:
                if dotenv_line_var(existing) != INGEST_TOKEN_VAR:
                    rewritten.append(existing)
                    continue
                # Keep the line's own ending, so a file with no trailing newline
                # still has none and nothing below the replaced line moves.
                rewritten.append(line + existing[len(existing.rstrip("\r\n")) :])
            atomic_write(env_path, "".join(rewritten))
            return
    _append_env_block(env_path, HARVEST_BANNER, {INGEST_TOKEN_VAR: token})


def _mirror_into_environment(token: str, env: MutableMapping[str, str] | None) -> None:
    """Keep this process's copies of the token in step with the file.

    The deploy loads the project ``.env`` over its own environment before it gets
    here, so a name the file spelled empty left an empty copy behind in
    ``os.environ`` that would outrank the freshly written line for compose's
    interpolation. The same is true of the environment the caller has already
    built for its compose invocation. Names that were never there stay out of
    both: they reach the containers through ``--env-file`` as they always have.
    """
    if INGEST_TOKEN_VAR in os.environ:
        os.environ[INGEST_TOKEN_VAR] = token
    if env is not None and INGEST_TOKEN_VAR in env:
        env[INGEST_TOKEN_VAR] = token


def _announce(action: str) -> None:
    """Say what was provisioned, and say what the account can reach.

    The second line is the disclosure, and it is printed whenever this deploy
    creates or re-issues the account rather than buried in a document nobody
    opens. OpenObserve has no ingest-only role in any edition: the account this
    provisions can read back every log and metric in the store and list its
    users. That is the whole scope, and an operator gets to know it at the
    moment it is granted.
    """
    written = {
        "created": "created the telemetry ingest account and saved its token",
        "harvested": "read the telemetry ingest token back from the store",
        "rotated": "the telemetry ingest token was refused, so a new one was issued",
    }[action]
    _report_fact(
        f"{written} → .env",
        wrote=(
            ".env",
            f"{INGEST_TOKEN_VAR} (gitignored; keep it secret). The ingest account can also "
            "read every log and metric in the store and list its users.",
        ),
    )
    _report_fact(
        "the ingest account can read all telemetry and the user roster: OpenObserve has no "
        "ingest-only role in any edition. The root password stays on this host."
    )


def _degraded(problem: str, remedy: str) -> ProvisionOutcome:
    """Report a failure the deploy carries on from, and name the one fix."""
    _warn_fact(
        problem,
        "The agent's telemetry will not be accepted until this is provisioned. Every other "
        "service in this deployment is unaffected.",
        remedy,
    )
    return ProvisionOutcome("failed", problem, remedy)


def provision_ingest_identity(
    config: Mapping[str, Any],
    env_path: Path | str,
    *,
    env: MutableMapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    ready_timeout_s: float = READY_TIMEOUT_S,
) -> ProvisionOutcome:
    """Verify, and where needed heal, this deployment's ingest identity.

    Safe to call on every start of a deployment that runs the store: a working
    token is verified with one read and nothing else happens.

    :param config: The rendered deploy config. Read for ``deployed_services``,
        the store's published address, and the organization the agent's own
        telemetry endpoint names.
    :param env_path: The project ``.env``. Both credentials are read from it (or
        from this process's environment, which the deploy loaded it into), and
        the harvested token is written back to it.
    :param env: The environment the caller has already built for its compose
        invocation, when it has one. Updated in place if it already carries the
        token, for the reason ``_mirror_into_environment`` gives.
    :param transport: Optional httpx transport, for tests.
    :param ready_timeout_s: How long a store that is answering at all gets to
        answer ``/healthz`` with a 200. A port nothing accepts a connection on
        is given up on much sooner; see :func:`wait_for_store`.
    :return: What this run did. Never raises for a store-side problem; see the
        module docstring for why nothing here is fatal.
    """
    if not store_deployed(config):
        return ProvisionOutcome("skipped")

    from osprey.deployment.container_lifecycle import _STORE_ISSUED_VARS
    from osprey.deployment.service_tokens import _effective_value
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = Path(env_path)
    on_disk = parse_dotenv_file(env_path) if env_path.is_file() else {}
    identity_var = _STORE_ISSUED_VARS[INGEST_TOKEN_VAR].identity_var
    identity = _effective_value(identity_var, on_disk).strip()
    if not identity:
        return _degraded(
            f"The telemetry store's ingest account has no name: {identity_var} is empty.",
            f"Set {identity_var} in the project .env, or let `osprey up` write its default.",
        )

    token = _effective_value(INGEST_TOKEN_VAR, on_disk).strip()
    base_url = store_base_url(config)
    org = store_org(config)

    with httpx.Client(timeout=REQUEST_TIMEOUT_S, transport=transport) as client:
        started = time.monotonic()
        if not wait_for_store(client, base_url, deadline=started + ready_timeout_s):
            # The wait that happened, not the budget it was given: a port nothing
            # ever accepted a connection on is given up on well inside
            # ``ready_timeout_s``, and naming the budget would describe a wait
            # nobody sat through.
            waited = time.monotonic() - started
            return _degraded(
                f"The telemetry store did not answer at {base_url} within "
                f"{waited:.0f}s, so its ingest account was not provisioned.",
                _RETRY_REMEDY,
            )

        try:
            if token and _identity_accepted(client, base_url, org, identity, token):
                logger.debug("The telemetry store still accepts the ingest token on file.")
                return ProvisionOutcome("verified")

            # Root is read here rather than at the top: a deployment whose token
            # already works never needs it, and reading a credential that is not
            # about to be used is a habit worth not having.
            root_email = _effective_value(ROOT_EMAIL_VAR, on_disk).strip()
            root_password = _effective_value(ROOT_PASSWORD_VAR, on_disk)
            if not root_email or not root_password:
                return _degraded(
                    "The telemetry store's ingest account needs provisioning, but this "
                    f"deployment holds no {ROOT_EMAIL_VAR} / {ROOT_PASSWORD_VAR} to do it with.",
                    f"Set {ROOT_EMAIL_VAR} and {ROOT_PASSWORD_VAR} in the project .env, then "
                    "run `osprey up` again.",
                )

            action, harvested = _heal(
                client,
                base_url,
                org,
                identity,
                token,
                _basic_auth(root_email, root_password),
            )
        except _HealFailed as failure:
            return _degraded(failure.problem, failure.remedy)

    if "$" in harvested:
        # Compose mangles a `$` on both routes out of this file, so a value
        # carrying one reaches the container truncated. The store issues a
        # 16-character alphanumeric token, so this is a guard against a future
        # release changing the alphabet rather than a case seen in the wild.
        return _degraded(
            "The telemetry store issued an ingest token that a compose env file cannot carry "
            "intact, so it was not saved.",
            "Report this: the store's token format changed. Telemetry keeps working through "
            "any credential already on file.",
        )

    _write_token(env_path, harvested)
    _mirror_into_environment(harvested, env)
    _announce(action)
    return ProvisionOutcome(action)
