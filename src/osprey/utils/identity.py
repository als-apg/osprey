"""Who an audit record names — one ladder, shared by every surface that records.

An audit record uses its identity twice: once as the ``actor`` field inside the
envelope, and once as the ``<identity>`` path component of
``var/audit/<identity>/<surface>.jsonl``. If those two were resolved by
separate code they would drift by a rung the first time one of them grew a
fallback, and the ledger would start naming one actor while filing under
another. This module is the single resolution, so both uses read the same
answer by construction.

The ladder, most specific first:

1. ``OSPREY_TERMINAL_USER`` — the multi-user deployment's per-container user
   name. When it is set, a real person is behind the container.
2. ``OSPREY_AUDIT_IDENTITY`` — what a container that hosts no single user is
   named instead: a framework service's key, ``sidecar`` for auth. Rendered by
   the same derivation that names the audit mount path, so the writer's path
   and the mounted path cannot diverge. Without this rung a shared or framework
   service container would file everything under the process account —
   ``osprey`` or ``root``, which names nobody.
3. The local account (:func:`getpass.getuser`) — the single-user laptop case,
   where the process account *is* the person.
4. :data:`UNKNOWN_IDENTITY` — an honest floor. An unresolvable identity must
   never cost the record.

**Never the hostname.** It looks like an identity and is not one: a
host-network container reports the shared host's name, so every service on the
box would collide into one file, and a bridge-network container reports a
random container id, so the same service would scatter across a new file per
restart.

Like :mod:`osprey.utils.sensitive_env`, this module deliberately imports
nothing from ``osprey`` — only the standard library. It is imported from
``mcp_server``, from the interface apps and from services, and any osprey
import here would risk a cycle between those packages.
"""

import getpass
import os

#: The multi-user deployment's per-container user name. Spelled locally rather
#: than imported from ``interfaces.common_middleware``, which owns the same
#: constant for URL-prefix purposes: importing it would cost this module its
#: leaf property, and the name is a deployment contract that does not move.
TERMINAL_USER_ENV: str = "OSPREY_TERMINAL_USER"

#: The identity of a container that hosts no single user — a framework
#: service's key, or ``sidecar`` for auth. Rendered into every containerized
#: service's ``environment:`` by the same derivation that names its audit
#: mount. Audit-critical: it is stripped from every MCP server spec's ``env:``
#: so that no spec can repoint another service's records at itself.
AUDIT_IDENTITY_ENV: str = "OSPREY_AUDIT_IDENTITY"

#: The environment rungs, in the order they are consulted. Exposed so a test
#: or a drift check can assert the order rather than re-encode it.
IDENTITY_ENV_LADDER: tuple[str, ...] = (TERMINAL_USER_ENV, AUDIT_IDENTITY_ENV)

#: What a record names when no rung resolves. A slim image often has no
#: ``pwd`` entry for its uid, which is normal rather than exceptional, so this
#: is a routine outcome and not an error path.
UNKNOWN_IDENTITY: str = "unknown"

# Characters that would let an identity escape its own directory or split into
# several. The identity is a single path component; anything carrying one of
# these is not one.
_PATH_SEPARATORS: tuple[str, ...] = ("/", "\\", "\0")

# Names that are a path component syntactically but resolve elsewhere.
_RESERVED_NAMES: tuple[str, ...] = (".", "..")


def _usable(value: str | None) -> str:
    """Return *value* stripped, or ``""`` if it cannot serve as an identity.

    A rung must produce a value that works for *both* uses or it has not
    produced an identity at all. Empty and whitespace-only values are the unset
    case spelled differently — a rendered-but-blank ``environment:`` entry, an
    env var exported as ``""``. Values carrying a path separator or naming a
    relative directory are rejected because the same string becomes a directory
    name under ``var/audit/``, where ``../elsewhere`` would file one service's
    records into another's mount.

    Rejection is deliberately narrow: it covers what breaks path semantics and
    nothing else. A stricter character allowlist would turn a legitimate
    account name into :data:`UNKNOWN_IDENTITY`, which loses more than it
    protects.
    """
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate or candidate in _RESERVED_NAMES:
        return ""
    if any(separator in candidate for separator in _PATH_SEPARATORS):
        return ""
    return candidate


def acting_identity() -> str:
    """Return the identity an audit record should carry, per the module ladder.

    Reads the environment on every call rather than caching at import: the
    markers are set per process by compose and by the entrypoint, and a value
    frozen at import time would be whatever the first importer happened to see.

    Takes no arguments on purpose. A caller-supplied override would be a second
    ladder — the exact drift this module exists to prevent — so every surface
    gets the same answer from the same source or none at all.

    Never raises: :func:`getpass.getuser` raises ``KeyError`` (Python 3.12 and
    earlier) or ``OSError`` (3.13+) for a uid with no passwd entry, which is
    ordinary in a slim container, and any other failure is equally not worth an
    audit record. Every failure lands on :data:`UNKNOWN_IDENTITY`.
    """
    for env_name in IDENTITY_ENV_LADDER:
        candidate = _usable(os.environ.get(env_name))
        if candidate:
            return candidate

    try:
        local_account = getpass.getuser()
    except Exception:  # noqa: BLE001 — an unnamed account must not cost the record
        return UNKNOWN_IDENTITY

    return _usable(local_account) or UNKNOWN_IDENTITY
