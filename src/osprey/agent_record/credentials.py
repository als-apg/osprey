"""The agent record's credential files, and the one function that owns them.

Every login agent-record accepts (:func:`~osprey.agent_record.identities.credential_table`)
has its secret in a file of its own at the repo root, mode 0600, written once
and never rewritten:

=========================================  ====================================
``.env.agent-record.login.<login>``        one writer's or front's login; that
                                           container's ``env_file``
``.env.agent-record.reader.<name>``        one named reader's key; no
                                           container's file
``.env.agent-record.clickhouse``           agent-record's ClickHouse account
                                           (``backend: clickhouse`` only)
``.env.agent-record.clickhouse-admin``     the ClickHouse store's bootstrap
                                           account, which staging uses
                                           (``backend: clickhouse`` only)
``.env.agent-record.service``              derived: the verifier and the
                                           ingest secret set, regenerated on
                                           every build and start
``.env.agent-record.history-gate``         the secret the history page is
                                           gated on; the history front's and
                                           the proxy's ``env_file`` (while the
                                           page is deployed)
=========================================  ====================================

Each login file holds :data:`LOGIN_ENV` ``=<login>:<token>``; a reader's key
file holds the same variable with the login ``reader.<name>``.

The service file is agent-record's alone. It carries :data:`VERIFIER_ENV`, the
base64 of one ``<login>:<sha256 hex of the token>`` line per login, and
:data:`SECRETS_ENV`, the base64 of the JSON list of every value ingest scrubs.
Base64 keeps a ``$``, a quote or a newline in a secret out of compose's
``env_file`` parsing.

The ``.env.agent-record.`` prefix is this module's alone.

Recreate triggers
-----------------
A container reads an ``env_file`` when it is created, so every owner of one of
these files carries a :data:`DIGEST_LABEL` label interpolated from the variable
:func:`digest_env` computes for it;
:func:`~osprey.deployment.runtime_helper.runtime_env` sets them on every compose
invocation. A terminal's and the sidecar's login are baked into the web overlay
at render time (:func:`login_digests`).

Host launches
-------------
A session started on the host (``osprey chat``, ``query``, ``web``) records
under ``deployment.host_identity``; :func:`export_built_host_identity` tells it
so, reading the config the build rendered.

Nothing here prints, and nothing here returns a secret:
:func:`ensure_agent_record_credentials` returns file names.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.agent_record import identities as ids
from osprey.agent_record.config import HISTORY_SERVICE, history_page_deployed, uses_clickhouse
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors.safety_file import atomic_replace

__all__ = [
    "CLICKHOUSE_ADMIN_FILE",
    "CLICKHOUSE_FILE",
    "CLICKHOUSE_SERVICE",
    "CREDENTIAL_PREFIX",
    "DIGEST_ENV_PREFIX",
    "DIGEST_LABEL",
    "HISTORY_GATE_ENV",
    "HISTORY_GATE_FILE",
    "LOGIN_ENV",
    "LOGIN_FILE_PREFIX",
    "READER_FILE_PREFIX",
    "PROXY_SERVICE",
    "RECORD_SERVICE",
    "SECRETS_ENV",
    "SERVICE_FILE",
    "VERIFIER_ENV",
    "AgentRecordCredentialReport",
    "credential_file_problems",
    "decode_secret_set",
    "decode_verifier",
    "digest_env",
    "digest_env_name",
    "ensure_agent_record_credentials",
    "existing_credential_files",
    "export_built_host_identity",
    "files_digest",
    "is_credential_file",
    "login_digests",
    "login_file",
    "owner_files",
    "read_env_value",
    "read_login",
    "reader_file",
    "render_service_env",
    "verifier_line",
]

#: Every file this module writes starts with this, and nothing else does.
CREDENTIAL_PREFIX = ".env.agent-record."
LOGIN_FILE_PREFIX = f"{CREDENTIAL_PREFIX}login."
READER_FILE_PREFIX = f"{CREDENTIAL_PREFIX}reader."
SERVICE_FILE = f"{CREDENTIAL_PREFIX}service"
CLICKHOUSE_FILE = f"{CREDENTIAL_PREFIX}clickhouse"
CLICKHOUSE_ADMIN_FILE = f"{CREDENTIAL_PREFIX}clickhouse-admin"
HISTORY_GATE_FILE = f"{CREDENTIAL_PREFIX}history-gate"

#: The variable a login or reader file carries: ``<login>:<token>``.
LOGIN_ENV = "OSPREY_AGENT_RECORD_LOGIN"
#: The service file's two variables.
VERIFIER_ENV = "OSPREY_AGENT_RECORD_VERIFIER"
SECRETS_ENV = "OSPREY_AGENT_RECORD_SECRETS"
#: The history gate file's variable. Spelled so the proxy's
#: ``^OSPREY_TERMINAL_SECRET_`` substitution filter passes it and the history
#: app's roster-secret seam accepts it.
HISTORY_GATE_ENV = "OSPREY_TERMINAL_SECRET_HISTORY"

#: agent-record's ClickHouse account.
CLICKHOUSE_USER = "agent_record"
CLICKHOUSE_USER_ENV = "OSPREY_AGENT_RECORD_CLICKHOUSE_USER"
CLICKHOUSE_PASSWORD_ENV = "OSPREY_AGENT_RECORD_CLICKHOUSE_PASSWORD"  # noqa: S105 -- names an environment variable, holds no secret
#: The ClickHouse store's bootstrap account; the store container's ``env_file``.
CLICKHOUSE_ADMIN_USER = "agent_record_admin"

#: The compose services that own the fixed files.
RECORD_SERVICE = "agent-record"
CLICKHOUSE_SERVICE = "agent_record_clickhouse"
#: The proxy, which stamps the history gate secret on the requests it forwards
#: to the history page.
PROXY_SERVICE = "nginx"

#: The label every owner of a credential file carries, and the prefix of the
#: variable it is interpolated from.
DIGEST_LABEL = "osprey.agent-record.cred.digest"
DIGEST_ENV_PREFIX = "OSPREY_AGENT_RECORD_DIGEST_"

#: The secret-set rule for rendered env values: a value at least this long
#: whose name holds one of these hints is scrubbed.
SECRET_ENV_NAME_HINTS: tuple[str, ...] = ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
MIN_SECRET_ENV_LEN = 12
#: The dispatch bearers, scrubbed whatever their length.
WORKER_BEARER_ENVS: tuple[str, ...] = ("DISPATCH_WORKER_TOKEN", "EVENT_DISPATCHER_TOKEN")

_FILE_MODE = 0o600
_WITHDRAWN_MARK = ".withdrawn-"

#: The logins no roster user, worker, reader or host identity may take.
_FIXED_LOGINS: tuple[str, ...] = (
    ids.OPERATOR_IDENTITY,
    ids.HISTORY_IDENTITY,
    ids.SIDECAR_IDENTITY,
    *ids.BRIDGE_IDENTITIES.values(),
)


@dataclass
class AgentRecordCredentialReport:
    """What one :func:`ensure_agent_record_credentials` call did, by file name.

    Attributes:
        minted: Files created by this call.
        purged: Files removed because their login left the config.
        reader_keys: Names of the readers whose key this call minted.
        withdrawn: ``(reader name, new file name)`` for every reader whose
            key file was set aside because the reader left the config.
        service_changed: Whether the service file was (re)written.
    """

    minted: list[str] = field(default_factory=list)
    purged: list[str] = field(default_factory=list)
    reader_keys: list[str] = field(default_factory=list)
    withdrawn: list[tuple[str, str]] = field(default_factory=list)
    service_changed: bool = False


# --- names ---------------------------------------------------------------------


def login_file(login: str) -> str:
    """The name of the file holding *login*'s token."""
    return f"{LOGIN_FILE_PREFIX}{login}"


def reader_file(name: str) -> str:
    """The name of the file holding reader *name*'s key."""
    return f"{READER_FILE_PREFIX}{name}"


def is_credential_file(name: str) -> bool:
    """Whether a file called *name* is one this module writes or sets aside."""
    return name.startswith(CREDENTIAL_PREFIX)


def _grant_file(grant: ids.CredentialGrant) -> str:
    if grant.identity.startswith(ids.READER_IDENTITY_PREFIX):
        return reader_file(grant.identity[len(ids.READER_IDENTITY_PREFIX) :])
    return login_file(grant.login)


# --- the login table -------------------------------------------------------------


def _check_fixed_logins(config: Mapping[str, Any] | None) -> None:
    """Refuse a roster user, worker or reader named like a fixed login, naming both.

    The fixed logins are ``operator``, ``history``, ``sidecar``, every bridge
    identity whether or not that bridge is deployed today, and the host
    identity. The credential table catches a clash only between logins it
    lists; a reader logs in as ``reader.<name>`` and an undeployed bridge has
    no login, so those clashes are caught here: a reader called ``operator``
    reads as the operator everywhere a name is shown, and deploying the bridge
    later would break a deployment that started fine.
    """
    fixed = {login: f"fixed login {login!r}" for login in _FIXED_LOGINS}
    host = ids.host_identity(config)
    if host is not None:
        fixed.setdefault(host, f"deployment.host_identity {host!r}")
    claimants = [("roster user", name) for name in ids.terminal_identities(config)]
    claimants += [("dispatch worker", name) for name in ids.dispatch_identities(config)]
    claimants += [("agent_record.readers name", e["name"]) for e in ids.reader_entries(config)]
    for what, name in claimants:
        if name in fixed:
            raise ValueError(
                f"login {name!r} is claimed twice: by the {what} {name!r} and by the {fixed[name]}"
            )


def _grants(config: Mapping[str, Any] | None) -> list[ids.CredentialGrant]:
    """Every login of *config*, refusing a name that takes a fixed login.

    Raises:
        ValueError: Naming both claimants of a login.
    """
    _check_fixed_logins(config)
    return list(ids.credential_table(config).values())


# --- file contents ---------------------------------------------------------------


def _token() -> str:
    """A fresh secret. URL-safe base64, so it holds no ``$`` for compose to expand."""
    return secrets.token_urlsafe(32)


def _env_lines(pairs: Iterable[tuple[str, str]]) -> str:
    return "".join(f"{key}={value}\n" for key, value in pairs)


def read_env_value(path: Path, key: str) -> str | None:
    """The value *key* carries in the env-style file at *path*, if any."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == key and value.strip():
            return value.strip()
    return None


def read_login(path: Path, login: str) -> str | None:
    """The token *path* holds for *login*, or ``None`` when it holds none.

    ``None`` too when the file names another login: a token is only ever used
    under the login its file is filed under.
    """
    value = read_env_value(path, LOGIN_ENV)
    if value is None or not value.startswith(f"{login}:") or len(value) <= len(login) + 1:
        return None
    return value[len(login) + 1 :]


def _create(path: Path, content: str) -> bool:
    """Write *path* with mode 0600 unless it exists. True when this call wrote it.

    Whole or absent: the content is written beside the destination and linked
    into place, so no reader ever opens a half-written credential and an
    existing file is never replaced.
    """
    if path.exists() or path.is_symlink():
        return False
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
    return True


def _remove(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _suffixes(root: Path, prefix: str) -> list[str]:
    """What follows *prefix* in the name of every regular file under *root* carrying it."""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    return [
        entry.name[len(prefix) :]
        for entry in entries
        if entry.name.startswith(prefix)
        and entry.is_file()
        and not entry.is_symlink()
        and not entry.name.endswith(".tmp")
    ]


def _withdraw(root: Path, name: str, today: datetime.date) -> str:
    """Set reader *name*'s key file aside under a dated name, and return that name.

    Linked to the new name, then unlinked from the old: the new name is never
    an existing file, and the key is never in neither place.
    """
    source = root / reader_file(name)
    base = f"{reader_file(name)}{_WITHDRAWN_MARK}{today.isoformat()}"
    attempt = 1
    while True:
        target = root / (base if attempt == 1 else f"{base}.{attempt}")
        try:
            os.link(source, target)
        except FileExistsError:
            attempt += 1
            continue
        break
    source.unlink()
    return target.name


def verifier_line(login: str, token: str) -> str:
    """``<login>:<sha256 hex of token>``, one line of the verifier."""
    return f"{login}:{hashlib.sha256(token.encode('utf-8')).hexdigest()}"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def render_service_env(verifier_lines: Iterable[str], secret_set: Iterable[str]) -> str:
    """The service file's text: the verifier and the secret set, each base64."""
    verifier = "".join(f"{line}\n" for line in verifier_lines)
    secrets_json = json.dumps(list(secret_set), ensure_ascii=False)
    return _env_lines([(VERIFIER_ENV, _b64(verifier)), (SECRETS_ENV, _b64(secrets_json))])


def decode_verifier(value: str) -> dict[str, str]:
    """``{login: sha256 hex}`` from a :data:`VERIFIER_ENV` value."""
    lines = base64.b64decode(value.encode("ascii")).decode("utf-8").splitlines()
    pairs = (line.rpartition(":") for line in lines if line)
    return {login: digest for login, _, digest in pairs}


def decode_secret_set(value: str) -> list[str]:
    """The secret set from a :data:`SECRETS_ENV` value."""
    decoded = json.loads(base64.b64decode(value.encode("ascii")).decode("utf-8"))
    return [item for item in decoded if isinstance(item, str)]


def _env_secrets(repo_root: Path) -> set[str]:
    """The env chain's values ingest scrubs: the dispatch bearers and the rule's matches."""
    from osprey_connectors.dotenv import merge_chain

    try:
        chain = merge_chain(repo_root)
    except (OSError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    for name, value in chain.items():
        if not value:
            continue
        if name in WORKER_BEARER_ENVS or (
            len(value) >= MIN_SECRET_ENV_LEN
            and any(hint in name.upper() for hint in SECRET_ENV_NAME_HINTS)
        ):
            found.add(value)
    return found


def _clickhouse_contents() -> dict[str, Any]:
    return {
        CLICKHOUSE_FILE: lambda: _env_lines(
            [(CLICKHOUSE_USER_ENV, CLICKHOUSE_USER), (CLICKHOUSE_PASSWORD_ENV, _token())]
        ),
        CLICKHOUSE_ADMIN_FILE: lambda: _env_lines(
            [
                ("CLICKHOUSE_USER", CLICKHOUSE_ADMIN_USER),
                ("CLICKHOUSE_PASSWORD", _token()),
                ("CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT", "1"),
            ]
        ),
    }


_CLICKHOUSE_PASSWORD_KEYS: Mapping[str, str] = {
    CLICKHOUSE_FILE: CLICKHOUSE_PASSWORD_ENV,
    CLICKHOUSE_ADMIN_FILE: "CLICKHOUSE_PASSWORD",
}


def _write_service_file(
    root: Path, grants: Iterable[ids.CredentialGrant], clickhouse: bool
) -> bool:
    """Regenerate the service file from the files on disk; True when it changed."""
    lines: list[str] = []
    found: set[str] = set()
    for grant in sorted(grants, key=lambda g: g.login):
        token = read_login(root / _grant_file(grant), grant.login)
        if token is None:
            continue
        lines.append(verifier_line(grant.login, token))
        found.add(token)
    if clickhouse:
        for name, key in _CLICKHOUSE_PASSWORD_KEYS.items():
            if (password := read_env_value(root / name, key)) is not None:
                found.add(password)
    found |= _env_secrets(root)
    # Longest first, so a value that holds another is scrubbed whole; then by
    # value, so the same set always renders the same bytes.
    ordered = sorted(found, key=lambda value: (-len(value), value))
    content = render_service_env(lines, ordered)
    path = root / SERVICE_FILE
    with contextlib.suppress(OSError, UnicodeDecodeError):
        if path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == content:
            return False
    atomic_replace(path, content.encode("utf-8"), mode=_FILE_MODE, sync="best_effort")
    return True


def ensure_agent_record_credentials(
    repo_root: Path | str,
    config: Mapping[str, Any] | None,
    *,
    reader_keys: bool,
    today: datetime.date | None = None,
) -> AgentRecordCredentialReport:
    """Bring the agent record's credential files at *repo_root* in step with *config*.

    The record is always deployed, so this always acts. It mints every login
    file that is missing, removes the login files of identities the config no
    longer has, and regenerates the service file from the files on disk. A
    login or key file that exists is never rewritten — rotating one is
    deleting its file and calling this.

    * One login file per entry of the credential table, except the readers.
    * With ``backend: clickhouse``: agent-record's ClickHouse account and the
      store's admin account. Kept when the backend changes, because a store
      volume that outlives its config entry is only reachable with them.
    * With ``reader_keys``: one key per configured reader, and the key file of
      a reader that left the config is set aside as ``.withdrawn-<date>``, so
      adding the name back mints a new key instead of reviving the old one.
    * The history page's gate secret while the page is deployed
      (:func:`~osprey.agent_record.config.history_page_deployed`); removed
      otherwise.
    * The service file, always: the verifier of every login and key on disk
      and the secret set.

    Args:
        repo_root: The deployment repo root, where the files live.
        config: The rendered config.
        reader_keys: Mint and withdraw reader keys. ``osprey up`` does; a
            build does not, so a build never produces a key nobody was told
            about.
        today: The date a withdrawn key is filed under. Defaults to today's.

    Returns:
        What was minted, removed and set aside, by file name. Never a value.

    Raises:
        ValueError: If two identities claim one login, naming both, or an
            identity cannot be a login.
    """
    report = AgentRecordCredentialReport()
    root = Path(repo_root)
    grants = _grants(config)
    clickhouse = uses_clickhouse(config)

    if clickhouse:
        for name, content in _clickhouse_contents().items():
            if _create(root / name, content()):
                report.minted.append(name)

    if history_page_deployed(config):
        if _create(root / HISTORY_GATE_FILE, _env_lines([(HISTORY_GATE_ENV, _token())])):
            report.minted.append(HISTORY_GATE_FILE)
    elif _remove(root / HISTORY_GATE_FILE):
        report.purged.append(HISTORY_GATE_FILE)

    logins = {g.login: g for g in grants if not g.identity.startswith(ids.READER_IDENTITY_PREFIX)}
    for login in logins:
        if _create(root / login_file(login), _env_lines([(LOGIN_ENV, f"{login}:{_token()}")])):
            report.minted.append(login_file(login))
    for login in _suffixes(root, LOGIN_FILE_PREFIX):
        if login not in logins and _remove(root / login_file(login)):
            report.purged.append(login_file(login))

    if reader_keys:
        configured = {
            g.identity[len(ids.READER_IDENTITY_PREFIX) :]: g
            for g in grants
            if g.identity.startswith(ids.READER_IDENTITY_PREFIX)
        }
        for suffix in _suffixes(root, READER_FILE_PREFIX):
            # A reader name holds no dot, so a suffix with one is a key already set aside.
            if "." not in suffix and suffix not in configured:
                report.withdrawn.append(
                    (suffix, _withdraw(root, suffix, today or datetime.date.today()))
                )
        for name, grant in configured.items():
            content = _env_lines([(LOGIN_ENV, f"{grant.login}:{_token()}")])
            if _create(root / reader_file(name), content):
                report.minted.append(reader_file(name))
                report.reader_keys.append(name)

    report.service_changed = _write_service_file(root, grants, clickhouse)
    return report


# --- digests -----------------------------------------------------------------------


def owner_files(config: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """Which credential files each compose service is handed, by service name.

    agent-record and its ClickHouse store, the history front (its login file
    and the gate file) and the proxy (the gate file), always: a label
    interpolated from an unset variable reads as a document change. Then the
    sidecar, and every dispatch worker and bridge the config renders, each
    with its login file. A terminal's is baked by the web render
    (:func:`login_digests`).
    """
    from osprey.deployment.compose_generator import dispatch_worker_audit_identities

    owners: dict[str, tuple[str, ...]] = {
        RECORD_SERVICE: (SERVICE_FILE, CLICKHOUSE_FILE)
        if uses_clickhouse(config)
        else (SERVICE_FILE,),
        CLICKHOUSE_SERVICE: (CLICKHOUSE_ADMIN_FILE,),
        HISTORY_SERVICE: (login_file(ids.HISTORY_IDENTITY), HISTORY_GATE_FILE),
        PROXY_SERVICE: (HISTORY_GATE_FILE,),
        ids.SIDECAR_IDENTITY: (login_file(ids.SIDECAR_IDENTITY),),
    }
    shared = ids.dispatch_shared(config)
    for worker in dispatch_worker_audit_identities(dict(config or {})):
        login = ids.SHARED_DISPATCH_IDENTITY if shared else worker
        owners[worker] = (login_file(login),)
    deployed = set(ids.bridge_identities(config))
    for service, identity in ids.BRIDGE_IDENTITIES.items():
        if identity in deployed:
            owners[service] = (login_file(identity),)
    return owners


def digest_env_name(service: str) -> str:
    """The variable a service's :data:`DIGEST_LABEL` is interpolated from."""
    return DIGEST_ENV_PREFIX + "".join(ch if ch.isalnum() else "_" for ch in service).upper()


def files_digest(repo_root: Path | str, files: Iterable[str]) -> str:
    """sha256 over the named files' bytes, concatenated in name order.

    A file that is missing or unreadable contributes nothing.
    """
    root = Path(repo_root)
    digest = hashlib.sha256()
    for name in sorted(files):
        with contextlib.suppress(OSError):
            digest.update((root / name).read_bytes())
    return digest.hexdigest()


def digest_env(repo_root: Path | str, config: Mapping[str, Any] | None) -> dict[str, str]:
    """Every ``OSPREY_AGENT_RECORD_DIGEST_*`` variable a rendered compose file reads.

    One per entry of :func:`owner_files`. Never raises: a config this module
    cannot read yields agent-record and its store alone.
    """
    try:
        owners = owner_files(config)
    except Exception:
        owners = {RECORD_SERVICE: (SERVICE_FILE,), CLICKHOUSE_SERVICE: (CLICKHOUSE_ADMIN_FILE,)}
    return {
        digest_env_name(service): files_digest(repo_root, files)
        for service, files in owners.items()
    }


def login_digests(repo_root: Path | str, config: Mapping[str, Any] | None) -> dict[str, str]:
    """The digest of each terminal's and the sidecar's login file, by identity.

    For the web render, which bakes them into the overlay. Empty when the
    logins cannot be resolved.
    """
    try:
        grants = _grants(config)
    except Exception:
        return {}
    from osprey.agent_record.contract import SOURCE_SIDECAR, SOURCE_TERMINAL

    return {
        grant.identity: files_digest(repo_root, [login_file(grant.login)])
        for grant in grants
        if grant.source_class in (SOURCE_TERMINAL, SOURCE_SIDECAR)
    }


# --- listing -------------------------------------------------------------------------

_HOLDS: Mapping[str, str] = {
    SERVICE_FILE: "the agent record's login verifier and the secrets its ingest scrubs",
    CLICKHOUSE_FILE: "the password the agent record keeps its ClickHouse store with",
    CLICKHOUSE_ADMIN_FILE: "the agent record ClickHouse store's administrator password",
    HISTORY_GATE_FILE: "the secret the history page is gated on",
}


def existing_credential_files(repo_root: Path | str) -> list[tuple[str, str]]:
    """Every agent-record credential file at *repo_root*, as ``(file name, what it holds)``.

    Read from the directory rather than from a config, so a file whose login
    left the config long ago is still named.
    """
    listed: list[tuple[str, str]] = []
    for suffix in _suffixes(Path(repo_root), CREDENTIAL_PREFIX):
        name = f"{CREDENTIAL_PREFIX}{suffix}"
        if name in _HOLDS:
            holds = _HOLDS[name]
        elif name.startswith(LOGIN_FILE_PREFIX):
            holds = f"the agent record login of {name[len(LOGIN_FILE_PREFIX) :]}"
        elif name.startswith(READER_FILE_PREFIX):
            reader, mark, _ = name[len(READER_FILE_PREFIX) :].partition(_WITHDRAWN_MARK)
            holds = (
                f"the key of the withdrawn agent record reader {reader}; the record no "
                "longer accepts it"
                if mark
                else f"the key of the agent record reader {reader}"
            )
        else:
            holds = "an agent record credential"
        listed.append((name, holds))
    return listed


# --- checks --------------------------------------------------------------------------


def credential_file_problems(
    repo_root: Path | str, config: Mapping[str, Any] | None
) -> list[tuple[str, str]]:
    """Credential files a start cannot go on without, or cannot trust.

    Asked of the files a container is handed whole: the history gate file
    while the history page is deployed. Pure: reads names and modes, writes
    nothing. Asked after :func:`ensure_agent_record_credentials` has run, so a
    missing file here is one that could not be minted, and a file with the
    wrong mode is one somebody else wrote.

    Returns:
        ``(problem, remedy)`` pairs, empty when every such file is a regular
        file readable by its owner alone.
    """
    root = Path(repo_root)
    names = [HISTORY_GATE_FILE] if history_page_deployed(config) else []
    problems: list[tuple[str, str]] = []
    for name in names:
        try:
            status = (root / name).lstat()
        except OSError:
            problems.append(
                (
                    f"{name} is missing",
                    "Run `osprey up` from an account that may write the deployment's "
                    "repo root; it creates the file",
                )
            )
            continue
        if not stat.S_ISREG(status.st_mode):
            problems.append(
                (
                    f"{name} is not a regular file",
                    "Remove it and run `osprey up`; the file is created again with a new secret",
                )
            )
            continue
        if status.st_mode & 0o077:
            problems.append(
                (
                    f"{name} has mode {stat.S_IMODE(status.st_mode):04o}: a credential file "
                    "is readable by its owner alone",
                    f"chmod 600 {name}",
                )
            )
    return problems


# --- host launches -------------------------------------------------------------------


def export_built_host_identity(
    environ: MutableMapping[str, str], repo_root: Path | str
) -> list[str]:
    """Tell a session launched on the host who it records as.

    Sets :data:`~osprey_connectors.identity.AUDIT_IDENTITY_ENV` to
    ``deployment.host_identity`` of the config the build rendered, so the
    record files the session under the name the build recorded whichever
    account runs the command. For the commands that launch a session from a
    repo: one call, after the env chain is loaded and before anything is
    spawned.

    A rendered container is left alone: it carries a terminal user, and its
    identity was set by the compose file that started it. A repo with no
    readable render, or a render with no host identity, exports nothing; the
    command that needs the render says so itself.

    Returns:
        The names of the variables set, for the caller to report. Never a value.
    """
    if (environ.get(TERMINAL_USER_ENV) or "").strip():
        return []
    import yaml

    from osprey_connectors.workspace import rendered_config_path

    try:
        config = yaml.safe_load(rendered_config_path(Path(repo_root)).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return []
    identity = ids.host_identity(config if isinstance(config, Mapping) else None)
    if identity is None:
        return []
    environ[AUDIT_IDENTITY_ENV] = identity
    return [AUDIT_IDENTITY_ENV]
