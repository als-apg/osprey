"""The dispatch bus's CURVE keys: one env file per container, minted by ``osprey up``.

A deployment on the bus (``services.event_dispatcher.transport: bus``) hands
each container its own keys in a file only that container's ``env_file:``
names, at the repo root beside the agent record's login files::

    .env.dispatch-bus.event-dispatcher      DISPATCH_BUS_SECRET_KEY   the dispatcher's secret
                                            DISPATCH_BUS_WORKER_KEYS  every worker's public key, by name
    .env.dispatch-bus.dispatch-worker-<i>   DISPATCH_BUS_SECRET_KEY   worker i's secret
                                            DISPATCH_BUS_SERVER_KEY   the dispatcher's public key

Not in the shared env chain on purpose: every container that runs an agent
reads that chain, and an agent holding a worker's key could join the bus as
that worker and be handed its runs. Keys are hex, which no env-file reader
rewrites (see :mod:`osprey.dispatch.bus.keys`).

**Existing keys win.** A secret already on disk is kept, so a redeploy never
rotates keys under a running stack; deleting a file is how one is rotated.
Everything derived — public keys, the dispatcher's worker list — is rewritten
from the secrets on every start, so a worker added to or removed from the
profile is accepted or refused by the next dispatcher that starts. Each file's
digest is interpolated into its container's ``osprey.dispatch-bus.digest``
label, so a file that changed recreates its container and nothing else.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey_connectors.safety_file import atomic_replace

#: Every file this module owns starts with this.
FILE_PREFIX = ".env.dispatch-bus."
#: The dispatcher's compose service key, and the name its file is under.
DISPATCHER_SERVICE = "event-dispatcher"
#: A worker's compose service key and bus name is this plus its index.
WORKER_PREFIX = "dispatch-worker-"
#: Prefix of the variable a service's digest label is interpolated from.
DIGEST_ENV_PREFIX = "OSPREY_DISPATCH_BUS_DIGEST_"

_SECRET = "DISPATCH_BUS_SECRET_KEY"  # noqa: S105 -- names an environment variable, holds no secret
_WORKER_KEYS = "DISPATCH_BUS_WORKER_KEYS"
_SERVER_KEY = "DISPATCH_BUS_SERVER_KEY"
_FILE_MODE = 0o600


@dataclass
class BusKeysReport:
    """What one provisioning pass did, by file name."""

    minted: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def key_file(service: str) -> str:
    """The name of the file ``service``'s keys are in."""
    return f"{FILE_PREFIX}{service}"


def bus_workers(config: Mapping[str, Any] | None) -> list[str] | None:
    """The workers' bus names when this deployment runs the bus, else ``None``."""
    deployed = (config or {}).get("deployed_services") or []
    if "event_dispatcher" not in [str(s) for s in deployed]:
        return None
    services = (config or {}).get("services")
    services = services if isinstance(services, Mapping) else {}
    dispatcher = services.get("event_dispatcher")
    if not isinstance(dispatcher, Mapping) or dispatcher.get("transport") != "bus":
        return None
    worker = services.get("dispatch_worker")
    count = worker.get("worker_count", 1) if isinstance(worker, Mapping) else 1
    count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 1 else 1
    return [f"{WORKER_PREFIX}{i}" for i in range(1, count + 1)]


def _read(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    return values


def _kept_secret(path: Path) -> str | None:
    """The secret ``path`` holds, when it is a well-formed key."""
    from osprey.dispatch.bus.keys import BusKeyError, public_of

    secret = _read(path).get(_SECRET)
    if not secret:
        return None
    try:
        public_of(secret)
    except BusKeyError:
        return None
    return secret


def _write(path: Path, content: str) -> bool:
    """Replace ``path`` with ``content`` at mode 0600, unless it already holds it.

    Replaced whole, so no reader opens a half-written key.
    """
    with contextlib.suppress(OSError):  # missing or unreadable: write it
        if path.read_text(encoding="utf-8") == content:
            return False
    atomic_replace(path, content.encode("utf-8"), mode=_FILE_MODE)
    return True


def _lines(pairs: Mapping[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in pairs.items())


def ensure_dispatch_bus_keys(
    repo_root: Path | str, config: Mapping[str, Any] | None
) -> BusKeysReport:
    """Mint what is missing, rewrite what is derived, and drop what left the config.

    Does nothing for a deployment that is not on the bus.
    """
    from osprey.dispatch.bus.keys import format_worker_keys, new_secret, public_of

    report = BusKeysReport()
    workers = bus_workers(config)
    if workers is None:
        return report
    root = Path(repo_root)

    def secret_of(service: str) -> str:
        kept = _kept_secret(root / key_file(service))
        if kept is not None:
            return kept
        report.minted.append(key_file(service))
        return new_secret()

    server_secret = secret_of(DISPATCHER_SERVICE)
    server_public = public_of(server_secret)
    worker_secrets = {name: secret_of(name) for name in workers}

    files = {
        DISPATCHER_SERVICE: {
            _SECRET: server_secret,
            _WORKER_KEYS: format_worker_keys(
                {name: public_of(secret) for name, secret in worker_secrets.items()}
            ),
        },
        **{
            name: {_SECRET: secret, _SERVER_KEY: server_public}
            for name, secret in worker_secrets.items()
        },
    }
    for service, values in files.items():
        if _write(root / key_file(service), _lines(values)):
            report.written.append(key_file(service))

    current = {key_file(name) for name in workers}
    for stale in sorted(root.glob(f"{FILE_PREFIX}{WORKER_PREFIX}*")):
        if stale.name not in current and not stale.name.endswith(".tmp"):
            stale.unlink(missing_ok=True)
            report.removed.append(stale.name)
    return report


def digest_env(repo_root: Path | str, config: Mapping[str, Any] | None) -> dict[str, str]:
    """Every :data:`DIGEST_ENV_PREFIX` variable a rendered bus deployment reads.

    Empty for a deployment that is not on the bus. Never raises: a file that
    cannot be read contributes an empty digest, which a later start corrects.
    """
    from osprey.agent_record.credentials import digest_env_name, files_digest

    workers = bus_workers(config)
    if workers is None:
        return {}
    return {
        digest_env_name(service, prefix=DIGEST_ENV_PREFIX): files_digest(
            repo_root, [key_file(service)]
        )
        for service in (DISPATCHER_SERVICE, *workers)
    }
