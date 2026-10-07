"""How a roster username keys its per-user environment variables.

This is the one definition of the suffix a web-terminal roster username is
keyed by in ``OSPREY_AUTH_PW_HASH_<SUFFIX>``, ``OSPREY_AUTH_OIDC_SUBJECT_<SUFFIX>``,
``OSPREY_AUTH_ROSTER_ROLE_<SUFFIX>``, ``OSPREY_AUTH_ROSTER_ACCESS_<SUFFIX>`` and
``OSPREY_TERMINAL_SECRET_<SUFFIX>``. The deployment layer mints under these
names (credential provisioning, render, lint, lifecycle) and the auth sidecar
reads them, so both import the mapping from here. It also defines the
stored-hash stem those names start with, for the same reason: both ends import
it.

It lives in the sidecar package because the deployment layer imports the
sidecar's leaf modules and never the other way round, which keeps the sidecar's
process free of the render and CLI code. It imports only the standard library.
"""

from __future__ import annotations

from collections.abc import Iterable

PW_HASH_VAR_PREFIX = "OSPREY_AUTH_PW_HASH_"
"""Stem of a roster user's stored password hash, completed by :func:`env_var_suffix`.

The one definition the credential writer, the sidecar and lint share.
"""


def env_var_suffix(username: str) -> str:
    """Map a roster username to the suffix its per-user env vars are keyed by.

    Uppercase, with ``-`` replaced by ``_`` — so ``alice-b`` keys
    ``OSPREY_AUTH_PW_HASH_ALICE_B``. This is the single definition of that
    mapping; credential provisioning, the sidecar's env lookup, and lint all
    route through it so a username can never be keyed one way at mint time and
    another at verify time.

    The mapping is intentionally total and lossy: it neither validates the
    username charset nor rejects anything. Two distinct usernames can therefore
    collide onto one suffix (``alice-b`` and ``alice_b``), which is exactly what
    :func:`env_var_suffix_collisions` exists to detect — enforcement is the
    caller's (a hard raise on the deploy preflight path, an ERROR in lint), not
    this function's.
    """
    return username.upper().replace("-", "_")


def env_var_suffix_collisions(usernames: Iterable[str]) -> dict[str, list[str]]:
    """Find roster usernames that :func:`env_var_suffix` maps onto one suffix.

    Without this check ``alice-b`` and ``alice_b`` would silently share a single
    ``OSPREY_AUTH_PW_HASH_ALICE_B`` entry — one user's password would open the
    other's terminal, which is precisely the isolation the auth feature exists to
    establish.

    A username repeated verbatim in the roster is *not* a collision here: it is
    one user listed twice (a duplicate-name config error reported separately),
    not two users sharing a credential. Only distinct names count.

    Args:
        usernames: Roster usernames — typically ``entry["name"]`` for each
            :func:`~osprey.deployment.web_terminals.personas.normalize_users`
            entry. Non-string items are ignored, matching the roster readers'
            drop-don't-raise convention.

    Returns:
        ``{suffix: [colliding usernames]}`` for suffixes claimed by two or more
        distinct usernames; empty when the roster is unambiguous. Suffix keys and
        the names under each are sorted, so a lint or preflight message built
        from this is byte-stable across runs.
    """
    by_suffix: dict[str, set[str]] = {}
    for username in usernames:
        if isinstance(username, str):
            by_suffix.setdefault(env_var_suffix(username), set()).add(username)
    return {suffix: sorted(names) for suffix, names in sorted(by_suffix.items()) if len(names) > 1}
