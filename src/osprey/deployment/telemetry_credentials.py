"""The telemetry store's credential files, and the one function that owns them.

Every account the telemetry store and gateway know has its secret in a file of
its own at the repo root, mode 0600, written once and never rewritten:

==================================  ===========================================
``.env.telemetry-admin``            the store's bootstrap account (its only
                                    ``env_file``)
``.env.telemetry-ingest``           the gateway's store account
``.env.telemetry-senders.htpasswd`` one line per sender; the gateway reads it
``.env.telemetry-archive``          the archive's store account
``.env.telemetry-history``          the history page's store account
``.env.telemetry-history-gate``     the secret the history page is gated on
``.env.telemetry-sender.<id>``      one sender's login; that sender's
                                    ``env_file``
``.env.telemetry-reader.<name>``    one reader's key; no container's file
==================================  ===========================================

One file per account is the containment rule: a container is handed exactly
the files of the accounts it acts as, as ``env_file`` or as a read-only bind,
so a leak from one container names one account. None is a member of the env
chain, and every one is a ``.env*`` name, so the guards that keep the chain
out of git, image contexts and the source zone keep these out too.

Two identity sets
-----------------
The **sender set** (:func:`sender_identities`) is everything that EMITS:
roster terminals, dispatch workers, and host launches under
``deployment.host_identity``. Each holds a login. The fixed services, the
queueservers, the archive and the proxy record to the ledger but emit nothing,
so they hold no login.

Recreate triggers
-----------------
A container reads an ``env_file`` when it is created, so a changed credential
file reaches it only if the compose document changed. Every owner therefore
carries a :data:`DIGEST_LABEL` label interpolated from a variable
:func:`digest_env` computes; :func:`~osprey.deployment.runtime_helper.runtime_env`
sets them on every compose invocation, and a terminal's is baked into the web
overlay at render time (:func:`sender_digests`).

Nothing here prints, and nothing here returns a secret:
:func:`ensure_telemetry_credentials` returns file names.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import os
import secrets
import stat
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.deployment import telemetry_config as tc

__all__ = [
    "ADMIN_FILE",
    "ARCHIVE_FILE",
    "AUDIT_IDENTITY_ENV",
    "CREDENTIAL_GLOB",
    "CredentialReport",
    "DIGEST_ENV_PREFIX",
    "DIGEST_LABEL",
    "HISTORY_FILE",
    "HISTORY_GATE_FILE",
    "HTPASSWD_FILE",
    "INGEST_FILE",
    "LOGIN_ENV",
    "READER_FILE_PREFIX",
    "SENDER_FILE_PREFIX",
    "credential_file_problems",
    "digest_env",
    "digest_env_name",
    "ensure_telemetry_credentials",
    "existing_credential_files",
    "expected_files",
    "export_built_host_identity",
    "export_host_launch_identity",
    "files_digest",
    "host_launch_env",
    "is_credential_file",
    "owner_files",
    "read_env_value",
    "read_sender_login",
    "reader_file",
    "render_htpasswd",
    "sender_digests",
    "sender_file",
    "sender_identities",
    "terminal_identities",
]

ADMIN_FILE = ".env.telemetry-admin"
INGEST_FILE = ".env.telemetry-ingest"
HTPASSWD_FILE = ".env.telemetry-senders.htpasswd"
ARCHIVE_FILE = ".env.telemetry-archive"
HISTORY_FILE = ".env.telemetry-history"
HISTORY_GATE_FILE = ".env.telemetry-history-gate"
SENDER_FILE_PREFIX = ".env.telemetry-sender."
READER_FILE_PREFIX = ".env.telemetry-reader."
#: Matches every file this module writes, and nothing else at a repo root.
CREDENTIAL_GLOB = ".env.telemetry-*"

#: The variable a sender's file carries: ``<identity>:<password>``.
LOGIN_ENV = "OSPREY_TELEMETRY_LOGIN"
#: Rung two of the ledger writers' identity ladder.
AUDIT_IDENTITY_ENV = "OSPREY_AUDIT_IDENTITY"
_TERMINAL_USER_ENV = "OSPREY_TERMINAL_USER"

#: The store's bootstrap account.
ADMIN_USER = "osprey_admin"
INGEST_PASSWORD_ENV = "OSPREY_TELEMETRY_INGEST_PASSWORD"
ARCHIVE_PASSWORD_ENV = "OSPREY_TELEMETRY_ARCHIVE_PASSWORD"
HISTORY_PASSWORD_ENV = "OSPREY_TELEMETRY_HISTORY_PASSWORD"
#: Spelled so the proxy's ``^OSPREY_TERMINAL_SECRET_`` substitution filter
#: passes it and the history app's roster-secret seam accepts it.
HISTORY_GATE_ENV = "OSPREY_TERMINAL_SECRET_HISTORY"
#: The key a reader's file carries.
READER_KEY = "password"

#: The label every owner of a credential file carries, and the prefix of the
#: variable it is interpolated from.
DIGEST_LABEL = "osprey.telemetry.cred.digest"
DIGEST_ENV_PREFIX = "OSPREY_TELEMETRY_DIGEST_"

_FILE_MODE = 0o600
_WITHDRAWN_MARK = ".withdrawn-"
_ARCHIVE_SERVICE = "archive"
_PROXY_SERVICE = "nginx"

#: The owners whose files do not depend on the config, and what each is handed.
#: Always computed, deployed or not: a label interpolated from an unset
#: variable reads as a document change.
_FIXED_OWNER_FILES: Mapping[str, tuple[str, ...]] = {
    tc.STORE_SERVICE: (ADMIN_FILE,),
    tc.GATEWAY_SERVICE: (INGEST_FILE, HTPASSWD_FILE),
    _ARCHIVE_SERVICE: (ARCHIVE_FILE,),
    tc.HISTORY_SERVICE: (HISTORY_FILE, HISTORY_GATE_FILE),
    _PROXY_SERVICE: (HISTORY_GATE_FILE,),
}


@dataclass
class CredentialReport:
    """What one :func:`ensure_telemetry_credentials` call did, by file name.

    Attributes:
        minted: Files created by this call.
        purged: Files removed because their account left the config.
        reader_keys: Names of the readers whose key this call minted.
        withdrawn: ``(reader name, new file name)`` for every reader whose
            file was set aside because the reader left the config.
    """

    minted: list[str] = field(default_factory=list)
    purged: list[str] = field(default_factory=list)
    reader_keys: list[str] = field(default_factory=list)
    withdrawn: list[tuple[str, str]] = field(default_factory=list)


def sender_file(identity: str) -> str:
    """The name of the file holding *identity*'s login."""
    return f"{SENDER_FILE_PREFIX}{identity}"


def reader_file(name: str) -> str:
    """The name of the file holding reader *name*'s key."""
    return f"{READER_FILE_PREFIX}{name}"


def is_credential_file(name: str) -> bool:
    """Whether a file called *name* is one this module writes or sets aside."""
    return name.startswith(".env.telemetry-")


def _check_identity(identity: str, source: str) -> str:
    """Refuse an identity that cannot be a login name and a file-name suffix."""
    bad = (
        not identity
        or identity != identity.strip()
        or identity in (".", "..")
        or any(ch in identity for ch in "/\\:$\0")
        or any(ch.isspace() for ch in identity)
    )
    if bad:
        raise ValueError(
            f"{source} {identity!r} cannot be a telemetry sender: a sender's name is the "
            "user half of its login and the suffix of its credential file, so it may not "
            "be empty or hold whitespace, '/', '\\', ':' or '$'"
        )
    return identity


def terminal_identities(config: Mapping[str, Any] | None) -> list[str]:
    """The roster terminals (people and shared cards alike) the config renders.

    Empty when web terminals are not enabled.
    """
    web_terminals = ((config or {}).get("modules") or {}).get("web_terminals") or {}
    if not (isinstance(web_terminals, Mapping) and web_terminals.get("enabled")):
        return []
    from osprey.deployment.web_terminals.personas import resolve_personas

    registry = (config or {}).get("registry") or {}
    entries = resolve_personas(dict(web_terminals), registry, strict=False)
    return [entry["name"] for entry in entries]


def sender_identities(config: Mapping[str, Any] | None) -> list[str]:
    """Every identity that emits to the gateway, in a stable order.

    Roster terminals (people and shared cards alike), dispatch workers, then
    ``deployment.host_identity``. Empty when the gateway is not deployed.

    Raises:
        ValueError: If an identity cannot be a login name.
    """
    if not tc.gateway_deployed(config):
        return []
    from osprey.deployment.compose_generator import dispatch_worker_audit_identities

    config = config or {}
    identities = [_check_identity(name, "roster name") for name in terminal_identities(config)]
    identities.extend(
        _check_identity(identity, "dispatch worker")
        for identity in dispatch_worker_audit_identities(config)
    )
    host = tc.host_identity(config)
    if host is not None:
        identities.append(_check_identity(host, "deployment.host_identity"))
    return list(dict.fromkeys(identities))


def _dispatch_worker_identities(config: Mapping[str, Any] | None) -> list[str]:
    if not tc.gateway_deployed(config):
        return []
    from osprey.deployment.compose_generator import dispatch_worker_audit_identities

    return list(dispatch_worker_audit_identities(config or {}))


def owner_files(config: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    """Which credential files each compose service is handed, by service name.

    The fixed owners, always, then one entry per dispatch worker the config
    renders. A roster terminal's is its own sender file and is not here: its
    label is baked by the web render (:func:`sender_digests`).
    """
    owners = dict(_FIXED_OWNER_FILES)
    for identity in _dispatch_worker_identities(config):
        owners[identity] = (sender_file(identity),)
    return owners


def digest_env_name(service: str) -> str:
    """The variable a service's :data:`DIGEST_LABEL` is interpolated from."""
    return DIGEST_ENV_PREFIX + "".join(ch if ch.isalnum() else "_" for ch in service).upper()


def files_digest(repo_root: Path | str, files: Iterable[str]) -> str:
    """sha256 over the named files' bytes, concatenated in name order.

    A file that is missing or unreadable contributes nothing, so the digest of
    a set that does not exist yet is a constant rather than an error.
    """
    root = Path(repo_root)
    digest = hashlib.sha256()
    for name in sorted(files):
        with contextlib.suppress(OSError):
            digest.update((root / name).read_bytes())
    return digest.hexdigest()


def digest_env(repo_root: Path | str, config: Mapping[str, Any] | None) -> dict[str, str]:
    """Every ``OSPREY_TELEMETRY_DIGEST_*`` variable a rendered compose file reads.

    One per entry of :func:`owner_files`. Never raises: a config this module
    cannot read yields the fixed owners alone.
    """
    try:
        owners = owner_files(config)
    except Exception:
        owners = dict(_FIXED_OWNER_FILES)
    return {
        digest_env_name(service): files_digest(repo_root, files)
        for service, files in owners.items()
    }


def sender_digests(repo_root: Path | str, config: Mapping[str, Any] | None) -> dict[str, str]:
    """The digest of each sender's login file, by identity.

    Empty when the gateway is not deployed or the sender set cannot be
    resolved; the caller renders no label then.
    """
    try:
        identities = sender_identities(config)
    except Exception:
        return {}
    return {identity: files_digest(repo_root, [sender_file(identity)]) for identity in identities}


# --- file contents -----------------------------------------------------------


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


def read_sender_login(repo_root: Path | str, identity: str) -> str | None:
    """*identity*'s ``<identity>:<password>`` login, or ``None`` when it has none.

    ``None`` too when the file names another identity: a login is only ever
    used under the name its file is filed under.
    """
    login = read_env_value(Path(repo_root) / sender_file(identity), LOGIN_ENV)
    if login is None or not login.startswith(f"{identity}:") or len(login) <= len(identity) + 1:
        return None
    return login


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


def render_htpasswd(repo_root: Path | str, identities: Iterable[str]) -> str:
    """The gateway's sender list: one ``<identity>:{SHA}<digest>`` line each.

    The digest is the base64 SHA-1 of the password. The passwords are random
    tokens of full strength, so an unsalted digest gives nothing away, and
    having no salt is what makes two renders of the same logins the same
    bytes. Identities are listed in name order; one with no usable login file
    has no line and is therefore refused by the gateway.
    """
    lines = []
    for identity in sorted(set(identities)):
        login = read_sender_login(repo_root, identity)
        if login is None:
            continue
        password = login[len(identity) + 1 :]
        sha1 = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).digest()
        digest = base64.b64encode(sha1).decode("ascii")
        lines.append(f"{identity}:{{SHA}}{digest}\n")
    return "".join(lines)


def _write_htpasswd(root: Path, identities: Iterable[str]) -> None:
    """Bring the sender list in step with the login files, touching it only on a change."""
    path = root / HTPASSWD_FILE
    content = render_htpasswd(root, identities)
    with contextlib.suppress(OSError, UnicodeDecodeError):
        if path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == content:
            return
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _service_contents(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every service-account file this config calls for, with how to fill it."""
    contents: dict[str, Any] = {
        ADMIN_FILE: lambda: _env_lines(
            [
                ("CLICKHOUSE_USER", ADMIN_USER),
                ("CLICKHOUSE_PASSWORD", _token()),
                ("CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT", "1"),
            ]
        ),
        ARCHIVE_FILE: lambda: _env_lines([(ARCHIVE_PASSWORD_ENV, _token())]),
    }
    if tc.gateway_deployed(config):
        contents[INGEST_FILE] = lambda: _env_lines([(INGEST_PASSWORD_ENV, _token())])
    if tc.history_page_deployed(config):
        contents[HISTORY_FILE] = lambda: _env_lines([(HISTORY_PASSWORD_ENV, _token())])
        contents[HISTORY_GATE_FILE] = lambda: _env_lines([(HISTORY_GATE_ENV, _token())])
    return contents


def expected_files(config: Mapping[str, Any] | None, *, reader_keys: bool = False) -> list[str]:
    """Every credential file a started deployment of *config* reads, by name.

    Args:
        config: The rendered config.
        reader_keys: Include the readers' key files, which only ``osprey up``
            mints.

    Returns:
        Empty when the store is not deployed.
    """
    if not tc.store_deployed(config):
        return []
    names = list(_service_contents(config))
    senders = sender_identities(config)
    names.extend(sender_file(identity) for identity in senders)
    if tc.gateway_deployed(config):
        names.append(HTPASSWD_FILE)
    if reader_keys:
        names.extend(reader_file(name) for name in tc.reader_names(config))
    return names


def _suffixes(root: Path, prefix: str) -> list[str]:
    """What follows *prefix* in the name of every regular file under *root* carrying it."""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    return [
        entry.name[len(prefix) :]
        for entry in entries
        if entry.name.startswith(prefix) and entry.is_file() and not entry.name.endswith(".tmp")
    ]


def _withdraw(root: Path, name: str, today: datetime.date) -> str:
    """Set reader *name*'s key file aside under a dated name, and return that name."""
    source = root / reader_file(name)
    base = f"{reader_file(name)}{_WITHDRAWN_MARK}{today.isoformat()}"
    target, attempt = root / base, 1
    while target.exists() or target.is_symlink():
        attempt += 1
        target = root / f"{base}.{attempt}"
    os.rename(source, target)
    return target.name


def ensure_telemetry_credentials(
    repo_root: Path | str,
    config: Mapping[str, Any] | None,
    *,
    reader_keys: bool,
    today: datetime.date | None = None,
) -> CredentialReport:
    """Bring the credential files at *repo_root* in step with *config*.

    Mints every file that is missing, removes the files of accounts the config
    no longer has, and renders the gateway's sender list from the login files.
    A file that exists is never rewritten, so calling this again changes
    nothing — rotating one credential is deleting its file and calling this.

    * The service accounts: admin and archive always; ingest with the gateway;
      the history page's two only while the page is deployed, removed
      otherwise.
    * One login per member of :func:`sender_identities`; a login whose
      identity left the set is removed.
    * With ``reader_keys``: one key per configured reader, and the key file of
      a reader that left the config is renamed ``.withdrawn-<date>`` — kept
      for the operator, and out of the way, so adding the name back later
      mints a new key instead of reviving the one that was handed out.

    A deployment without the store is left exactly as it is: nothing is
    minted, and nothing is removed either, because a store volume that
    outlives its config entry is only reachable with the admin file.

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
        ValueError: If a sender identity cannot be a login name.
    """
    report = CredentialReport()
    if not tc.store_deployed(config):
        return report
    root = Path(repo_root)

    for name, content in _service_contents(config).items():
        if _create(root / name, content()):
            report.minted.append(name)
    if not tc.history_page_deployed(config):
        for name in (HISTORY_FILE, HISTORY_GATE_FILE):
            if _remove(root / name):
                report.purged.append(name)

    senders = sender_identities(config)
    for identity in senders:
        if _create(
            root / sender_file(identity), _env_lines([(LOGIN_ENV, f"{identity}:{_token()}")])
        ):
            report.minted.append(sender_file(identity))
    for identity in _suffixes(root, SENDER_FILE_PREFIX):
        if identity not in senders and _remove(root / sender_file(identity)):
            report.purged.append(sender_file(identity))
    if tc.gateway_deployed(config):
        _write_htpasswd(root, senders)
    elif _remove(root / HTPASSWD_FILE):
        report.purged.append(HTPASSWD_FILE)

    if reader_keys:
        configured = tc.reader_names(config)
        for suffix in _suffixes(root, READER_FILE_PREFIX):
            # A name holds no dot, so a suffix with one is a key already set aside.
            if "." not in suffix and suffix not in configured:
                report.withdrawn.append(
                    (suffix, _withdraw(root, suffix, today or datetime.date.today()))
                )
        for name in configured:
            _check_identity(name, "services.telemetry_store.readers name")
            if _create(root / reader_file(name), _env_lines([(READER_KEY, _token())])):
                report.minted.append(reader_file(name))
                report.reader_keys.append(name)
    return report


def _remove(path: Path) -> bool:
    """Delete *path* if it is there. True when this call removed it."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def credential_file_problems(
    repo_root: Path | str, config: Mapping[str, Any] | None
) -> list[tuple[str, str]]:
    """Credential files a start cannot go on without, or cannot trust.

    Pure: reads names, modes and contents, writes nothing. Asked after
    :func:`ensure_telemetry_credentials` has run, so a missing file here is one
    that could not be minted, and a file with the wrong mode is one somebody
    else wrote.

    Returns:
        ``(problem, remedy)`` pairs, empty when every expected file is a
        regular file readable by its owner alone and every sender file holds
        its own login.
    """
    if not tc.store_deployed(config):
        return []
    root = Path(repo_root)
    try:
        names = expected_files(config, reader_keys=True)
    except ValueError as exc:
        return [(str(exc), "Rename it in the config, then run `osprey build`")]
    problems: list[tuple[str, str]] = []
    logins = {sender_file(identity): identity for identity in sender_identities(config)}
    for name in names:
        path = root / name
        try:
            status = path.lstat()
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
        if name in logins and read_sender_login(root, logins[name]) is None:
            problems.append(
                (
                    f"{name} holds no {LOGIN_ENV} for {logins[name]!r}",
                    "Remove it and run `osprey up`; the file is created again with a new secret",
                )
            )
    return problems


#: What each kind of file holds, for the commands that list them.
_HOLDS: Mapping[str, str] = {
    ADMIN_FILE: "the telemetry store's administrator password",
    INGEST_FILE: "the password the telemetry gateway stores records with",
    HTPASSWD_FILE: "the list of senders the telemetry gateway admits",
    ARCHIVE_FILE: "the password the archive reads the telemetry store with",
    HISTORY_FILE: "the password the history page reads the telemetry store with",
    HISTORY_GATE_FILE: "the secret the history page is gated on",
}


def existing_credential_files(repo_root: Path | str) -> list[tuple[str, str]]:
    """Every credential file present at *repo_root*, as ``(file name, what it holds)``.

    Read from the directory rather than from a config, so a file whose account
    left the config long ago is still named.
    """
    listed: list[tuple[str, str]] = []
    for suffix in _suffixes(Path(repo_root), ".env.telemetry-"):
        name = f".env.telemetry-{suffix}"
        if name in _HOLDS:
            holds = _HOLDS[name]
        elif name.startswith(SENDER_FILE_PREFIX):
            holds = f"the telemetry login of {name[len(SENDER_FILE_PREFIX) :]}"
        elif name.startswith(READER_FILE_PREFIX):
            reader, mark, _ = name[len(READER_FILE_PREFIX) :].partition(_WITHDRAWN_MARK)
            holds = (
                f"the key of the withdrawn telemetry reader {reader}; the store no longer accepts it"
                if mark
                else f"the key of the telemetry reader {reader}"
            )
        else:
            holds = "a telemetry credential"
        listed.append((name, holds))
    return listed


# --- host launches -----------------------------------------------------------


def host_launch_env(repo_root: Path | str, config: Mapping[str, Any] | None) -> dict[str, str]:
    """What a session launched on the host is told about who it is.

    :data:`AUDIT_IDENTITY_ENV` is ``deployment.host_identity``, the name the
    build recorded, and :data:`LOGIN_ENV` is that identity's login. Both come
    from the one recorded name, so the directory the ledger is filed under and
    the user the gateway stamps are the same string whichever account runs the
    command.

    Returns:
        Empty when the config records no host identity. Without a login file
        the identity alone: the session still records under the right name.
    """
    identity = tc.host_identity(config)
    if identity is None:
        return {}
    env = {AUDIT_IDENTITY_ENV: identity}
    login = read_sender_login(repo_root, identity)
    if login is not None:
        env[LOGIN_ENV] = login
    return env


def export_host_launch_identity(
    environ: MutableMapping[str, str], repo_root: Path | str, config: Mapping[str, Any] | None
) -> list[str]:
    """Put :func:`host_launch_env` into *environ*, for a launch on the host.

    A rendered container is left alone. It says it is one by carrying the
    gateway's address (:data:`~osprey.deployment.telemetry_config.GATEWAY_HOST_ENV`)
    or a terminal user, and its identity and login were set, together, by the
    compose file that started it.

    Returns:
        The names of the variables set, for the caller to report. Never a value.
    """
    if (environ.get(tc.GATEWAY_HOST_ENV) or "").strip() or (
        environ.get(_TERMINAL_USER_ENV) or ""
    ).strip():
        return []
    env = host_launch_env(repo_root, config)
    environ.update(env)
    return sorted(env)


def export_built_host_identity(
    environ: MutableMapping[str, str], repo_root: Path | str
) -> list[str]:
    """:func:`export_host_launch_identity`, reading the config the build rendered.

    For the commands that launch a session from a repo: one call, after the
    env chain is loaded and before anything is spawned. A repo with no
    readable render exports nothing, and the command that needs the render
    says so itself.
    """
    import yaml

    from osprey.utils.workspace import rendered_config_path

    try:
        config = yaml.safe_load(rendered_config_path(Path(repo_root)).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return []
    return export_host_launch_identity(
        environ, repo_root, config if isinstance(config, Mapping) else None
    )
