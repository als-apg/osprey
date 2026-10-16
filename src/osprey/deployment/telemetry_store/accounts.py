"""The telemetry store's accounts, profiles, quotas and readers' workload, as SQL.

:func:`render_account_statements` turns the config, the account passwords
and the store's current access entities (:class:`LiveAccounts`) into one
ordered script. The script stops at its first error, so its order is the
fail-closed part of the design, at two levels:

* **Global phase order** (:class:`Phase`): every withdrawal first — owned
  readers and view users no longer configured, and the accounts of parts of
  the deployment that are gone — then every narrowing revoke, and only then
  profiles, quotas, the workload, creates and re-asserts. A failing create can
  therefore never leave a withdrawn person able to read.
* **Per entity**: a row policy is created before the grant it scopes, and a
  view user gets its roles before its SELECT grants.

Before rendering, :func:`adoption_problems` refuses any configured reader or
view-user name that already exists as a store user OSPREY does not own: a
hand-made user is never adopted. A user that carries its kind's profile but
lost its ownership role (a cut run) is owned, so the next run completes it.

Every identifier is backquoted (:func:`quote_ident`) and every value escaped
(:func:`quote_string`); passwords travel as ``sha256_hash`` with a salt
computed here, never in clear.

Each configured reader is one user (:func:`reader_statements`): its
credential, default database and profile in the one ``CREATE USER``, then the
re-assert, the ``readers`` role, and per table its row policy before the
SELECT that policy scopes. A table dropped from a reader's list is revoked,
then its policy dropped, in the revoke phase.

View users exist only while the history page is deployed
(:func:`osprey.deployment.telemetry_config.history_page_deployed`). The page
reads every row as one of them, through ``EXECUTE AS``; none can sign in
(``no_password``, ``HOST NONE``) and each holds its SELECT grants itself, never
through a role (:func:`view_statements`):

* ``view_<person>`` per roster person: the three agent-record tables, scoped to
  that person's own rows by the ``own_*`` policies on role ``views_own``;
* ``view__all``: the same tables, unscoped;
* ``view__reader_<name>`` per reader: exactly that reader's tables and days,
  through the reader's own routine with ``viewreader_<name>_<table>`` policies.

Every view user gets role ``views`` (the ownership marker) and its scoping
role or policies before any SELECT grant, so a run cut anywhere leaves it able
to read less than it should, never more.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_credentials as creds
from osprey.deployment.telemetry_store import ddl

#: Roles that mark who OSPREY owns, and the profile each kind carries.
READERS_ROLE = "readers"
VIEWS_ROLE = "views"
VIEWS_OWN_ROLE = "views_own"
READER_PROFILE = "reader_limits"
VIEW_PROFILE = "view_limits"
HISTORY_PROFILE = "history_limits"
READER_QUOTA = "reader_quota"
VIEW_QUOTA = "view_quota"

INGEST_USER = "ingest"
ARCHIVE_USER = "archive"
HISTORY_USER = "history"

#: The query resource every reader query takes a slot of, and the workload
#: that bounds the readers' shared pool. Queries outside the workload (the
#: archive, the reconcile) take no slot of it.
QUERY_RESOURCE = "osprey_queries"
READERS_WORKLOAD = "readers"

#: Query-page settings ``/play`` sends, which a readonly user may only send
#: when they are marked changeable. The page sends ``send_logs_level`` on every
#: run, and its Download sends ``http_response_headers``.
PLAY_SETTINGS = (
    "add_http_cors_header",
    "enable_http_compression",
    "framing_output_format",
    "extremes",
    "use_query_cache",
    "enable_reads_from_query_cache",
    "enable_writes_to_query_cache",
    "query_cache_ttl",
    "query_cache_nondeterministic_function_handling",
    "query_cache_system_table_handling",
    "send_logs_level",
    "http_response_headers",
)

#: The tables a reader reads when its entry names none.
DEFAULT_READER_TABLES = ("otel_logs", "otel_traces", "audit_ledger")

#: Rows per Parquet row group in the archive's daily exports.
ARCHIVE_PARQUET_ROW_GROUP = 100_000
#: How long one archive query may run.
ARCHIVE_SECONDS = 600
#: Page queries per minute per person (``KEYED BY client_key``).
VIEW_QUERIES_PER_MINUTE = 120
#: The page's time limit, and the limit on its outer context.
HISTORY_SECONDS = 30

#: The tables every person's and the ``all`` view user read.
VIEW_TABLES = ("otel_logs", "otel_traces", "audit_ledger")
#: ``view__all``: the leading underscore cannot come from a roster name.
VIEW_ALL_USER = tc.RESERVED_ACCOUNT_PREFIX + "_all"
#: ``view__reader_<name>``.
VIEW_READER_PREFIX = tc.RESERVED_ACCOUNT_PREFIX + "_reader_"
#: Policy-name prefixes, namespaced by kind so no configured name can produce
#: another kind's policy: ``reader_<r>_<t>``, ``viewreader_<r>_<t>``, ``own_<t>``.
READER_POLICY_PREFIX = "reader_"
VIEW_READER_POLICY_PREFIX = "viewreader_"
OWN_POLICY_PREFIX = "own_"
#: The person a ``view_<person>`` user stands for, in SQL: ``currentUser()``
#: names the impersonated user under ``EXECUTE AS``.
_VIEW_PERSON = f"substring(currentUser(), {len(tc.RESERVED_ACCOUNT_PREFIX) + 1})"
#: The one ledger surface a queueserver's ``on_behalf_of`` counts on.
PREFLIGHT_SURFACE = "preflight"


class Phase(enum.IntEnum):
    """The global order of the accounts script."""

    DROP = 1
    REVOKE = 2
    CREATE = 3


@dataclass(frozen=True)
class Statement:
    """One statement of the accounts script and the phase it belongs to."""

    phase: Phase
    sql: str


@dataclass
class LiveAccounts:
    """The store's access entities as ``system.*`` reports them."""

    users: set[str] = field(default_factory=set)
    #: role -> users granted it.
    role_members: dict[str, set[str]] = field(default_factory=dict)
    #: user -> the settings profiles it carries.
    user_profiles: dict[str, set[str]] = field(default_factory=dict)
    #: (user, access_type, database, table) per row of ``system.grants``.
    grants: list[tuple[str, str, str, str]] = field(default_factory=list)
    #: (short name, database, table) per row of ``system.row_policies``.
    policies: set[tuple[str, str, str]] = field(default_factory=set)

    def members(self, role: str) -> set[str]:
        return self.role_members.get(role, set())


#: The queries :meth:`LiveAccounts.from_rows` reads, keyed by its argument.
LIVE_QUERIES: Mapping[str, str] = {
    "users": "SELECT name FROM system.users FORMAT JSONEachRow",
    "role_grants": (
        "SELECT user_name, granted_role_name FROM system.role_grants "
        "WHERE user_name IS NOT NULL FORMAT JSONEachRow"
    ),
    "profiles": (
        "SELECT user_name, inherit_profile FROM system.settings_profile_elements "
        "WHERE user_name IS NOT NULL AND inherit_profile IS NOT NULL FORMAT JSONEachRow"
    ),
    "grants": (
        "SELECT user_name, access_type, database, table FROM system.grants "
        "WHERE user_name IS NOT NULL FORMAT JSONEachRow"
    ),
    "policies": "SELECT short_name, database, table FROM system.row_policies FORMAT JSONEachRow",
}


def live_from_rows(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> LiveAccounts:
    """Assemble :class:`LiveAccounts` from the rows of every :data:`LIVE_QUERIES` query."""
    live = LiveAccounts(users={str(row["name"]) for row in rows.get("users", ())})
    for row in rows.get("role_grants", ()):
        live.role_members.setdefault(str(row["granted_role_name"]), set()).add(
            str(row["user_name"])
        )
    for row in rows.get("profiles", ()):
        live.user_profiles.setdefault(str(row["user_name"]), set()).add(str(row["inherit_profile"]))
    live.grants = [
        (
            str(row["user_name"]),
            str(row["access_type"]),
            str(row.get("database") or ""),
            str(row.get("table") or ""),
        )
        for row in rows.get("grants", ())
    ]
    live.policies = {
        (str(row["short_name"]), str(row.get("database") or ""), str(row.get("table") or ""))
        for row in rows.get("policies", ())
    }
    return live


# --------------------------------------------------------------------------- quoting


def quote_ident(name: str) -> str:
    """*name* as a backquoted ClickHouse identifier, any backquote or backslash escaped."""
    return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"


def quote_string(value: str) -> str:
    """*value* as a single-quoted ClickHouse string literal."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def table_ident(table: str) -> str:
    """``telemetry.<table>``, both parts backquoted."""
    return f"{quote_ident(ddl.DATABASE)}.{quote_ident(table)}"


def identified(user: str, password: str) -> str:
    """The ``IDENTIFIED WITH sha256_hash`` clause for *password*.

    The store checks ``sha256(password || salt)``. The salt is derived from the
    user and the password, so a render is deterministic and a rotated password
    gets a new salt with it.
    """
    salt = hashlib.sha256(f"osprey-salt\0{user}\0{password}".encode()).hexdigest()[:32]
    digest = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"IDENTIFIED WITH sha256_hash BY {quote_string(digest)} SALT {quote_string(salt)}"


def _settings(pairs: Sequence[str]) -> str:
    return ", ".join(pairs)


# --------------------------------------------------------------------------- passwords


@dataclass(frozen=True)
class Passwords:
    """The service accounts' passwords, as the repo-root credential files hold them."""

    ingest: str | None = None
    archive: str | None = None
    history: str | None = None
    #: Reader name -> its key, one per configured reader.
    readers: Mapping[str, str] = field(default_factory=dict)


class AccountsError(Exception):
    """The accounts script cannot be rendered."""


def read_passwords(repo_root: Path | str, config: Mapping[str, Any] | None) -> Passwords:
    """Read every service account's password this config renders an account for.

    Every configured reader's key is read from its own file; a withdrawn
    reader's file, set aside under a dated name, is never read.

    Raises:
        AccountsError: A file the config calls for is missing or empty, naming it.
    """
    root = Path(repo_root)

    def need(filename: str, key: str) -> str:
        value = creds.read_env_value(root / filename, key)
        if value is None:
            raise AccountsError(
                f"{filename} is missing or carries no {key}; `osprey up` mints it "
                f"before the store is staged, so rerun `osprey up`"
            )
        return value

    return Passwords(
        ingest=need(creds.INGEST_FILE, creds.INGEST_PASSWORD_ENV)
        if tc.gateway_deployed(config)
        else None,
        archive=need(creds.ARCHIVE_FILE, creds.ARCHIVE_PASSWORD_ENV),
        history=need(creds.HISTORY_FILE, creds.HISTORY_PASSWORD_ENV)
        if tc.history_page_deployed(config)
        else None,
        readers={
            name: need(creds.reader_file(name), creds.READER_KEY)
            for name in tc.reader_names(config)
        },
    )


# --------------------------------------------------------------------------- owned entities


@dataclass(frozen=True)
class OwnedKind:
    """One kind of store user OSPREY creates per config entry."""

    role: str
    profile: str


READER_KIND = OwnedKind(READERS_ROLE, READER_PROFILE)
VIEW_KIND = OwnedKind(VIEWS_ROLE, VIEW_PROFILE)


def roster_persons(config: Mapping[str, Any] | None) -> list[str]:
    """Every roster entry that is one person's, in roster order.

    A shared card is no one person: a sign-in as the card is governed by the
    card's own roster entry, so it gets no view user of its own.
    """
    from osprey.deployment.web_terminals.personas import entry_is_shared

    users: Any = config
    for key in ("modules", "web_terminals", "users"):
        users = users.get(key) if isinstance(users, Mapping) else None
    persons: list[str] = []
    for entry in users if isinstance(users, list) else []:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            continue
        if not entry_is_shared(dict(entry)):
            persons.append(entry["name"])
    return list(dict.fromkeys(persons))


def person_view_user(person: str) -> str:
    """``view_<person>``."""
    return tc.RESERVED_ACCOUNT_PREFIX + person


def reader_view_user(reader: str) -> str:
    """``view__reader_<reader>``."""
    return VIEW_READER_PREFIX + reader


def configured_view_users(config: Mapping[str, Any] | None) -> list[str]:
    """Every view user this config calls for: persons, ``all``, then readers.

    None when the history page is not deployed, so every owned one is withdrawn.
    """
    if not tc.history_page_deployed(config):
        return []
    return [
        *(person_view_user(person) for person in roster_persons(config)),
        VIEW_ALL_USER,
        *(reader_view_user(name) for name in tc.reader_names(config)),
    ]


def owned(live: LiveAccounts, kind: OwnedKind) -> set[str]:
    """The store users of *kind* OSPREY owns: role members, or profile carriers."""
    by_profile = {user for user, profiles in live.user_profiles.items() if kind.profile in profiles}
    return (live.members(kind.role) | by_profile) & live.users


def adoption_problems(config: Mapping[str, Any] | None, live: LiveAccounts) -> list[str]:
    """Every configured reader or view-user name that is a store user OSPREY does not own."""
    problems: list[str] = []
    for kind, names, label in (
        (READER_KIND, tc.reader_names(config), "reader"),
        (VIEW_KIND, configured_view_users(config), "view user"),
    ):
        mine = owned(live, kind)
        problems.extend(
            f"{label} {name!r} names a store user OSPREY did not create (it is neither a "
            f"member of role {kind.role} nor carries profile {kind.profile}); rename the "
            f"{label} or drop that user by hand"
            for name in names
            if name in live.users and name not in mine
        )
    return problems


def _drop_policies(prefix: str) -> list[str]:
    return [
        f"DROP ROW POLICY IF EXISTS {quote_ident(f'{prefix}{table}')} ON {table_ident(table)}"
        for table in ddl.TIME_COLUMNS
    ]


def _view_policies(name: str) -> list[str]:
    if not name.startswith(VIEW_READER_PREFIX):
        return []
    return _drop_policies(f"{VIEW_READER_POLICY_PREFIX}{name[len(VIEW_READER_PREFIX) :]}_")


def _withdraw_view_user(name: str, live: LiveAccounts) -> list[str]:
    """A view user's withdrawal: its SELECT grants, then ``views_own``, then the user.

    The scoping role goes only after the grants it scopes, so a run cut
    between them leaves the user reading less, never every person's rows.
    """
    user = quote_ident(name)
    granted = _granted_tables(live, name)
    sql = [
        f"REVOKE SELECT ON {table_ident(table)} FROM {user}"
        for table in ddl.TIME_COLUMNS
        if table in granted
    ]
    if name in live.members(VIEWS_OWN_ROLE):
        sql.append(f"REVOKE {quote_ident(VIEWS_OWN_ROLE)} FROM {user}")
    sql.append(f"DROP USER IF EXISTS {user}")
    return [*sql, *_view_policies(name)]


def _withdrawals(config: Mapping[str, Any] | None, live: LiveAccounts) -> list[Statement]:
    statements: list[Statement] = []
    for name in sorted(owned(live, READER_KIND) - set(tc.reader_names(config))):
        statements.append(Statement(Phase.DROP, f"DROP USER IF EXISTS {quote_ident(name)}"))
        statements.extend(
            Statement(Phase.DROP, sql) for sql in _drop_policies(f"{READER_POLICY_PREFIX}{name}_")
        )
    for name in sorted(owned(live, VIEW_KIND) - set(configured_view_users(config))):
        statements.extend(Statement(Phase.DROP, sql) for sql in _withdraw_view_user(name, live))
    if not tc.history_page_deployed(config) and HISTORY_USER in live.users:
        statements.append(Statement(Phase.DROP, f"DROP USER IF EXISTS {quote_ident(HISTORY_USER)}"))
    if not tc.gateway_deployed(config) and INGEST_USER in live.users:
        statements.append(Statement(Phase.DROP, f"DROP USER IF EXISTS {quote_ident(INGEST_USER)}"))
    return statements


@dataclass(frozen=True)
class ReaderScope:
    """What one configured reader may read: tables, and how many days back."""

    name: str
    #: Registry tables, in registry order.
    tables: tuple[str, ...]
    days_back: int


def reader_scopes(config: Mapping[str, Any] | None) -> list[ReaderScope]:
    """Every configured reader's scope, defaults filled in, in config order.

    ``tables`` defaults to :data:`DEFAULT_READER_TABLES`, ``days_back`` to
    ``retention_days``.

    Raises:
        AccountsError: A reader names a table the store does not have, or
            states ``tables`` as anything but a list of names.
    """
    retention = tc.retention_days(config)
    scopes: list[ReaderScope] = []
    for entry in tc.readers(config):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            continue
        name = entry["name"]
        stated = entry.get("tables", DEFAULT_READER_TABLES)
        if not isinstance(stated, list | tuple) or not all(isinstance(t, str) for t in stated):
            raise AccountsError(
                f"services.telemetry_store.readers entry {name!r} tables must be a list "
                f"of table names, got {stated!r}"
            )
        unknown = sorted(set(stated) - set(ddl.TIME_COLUMNS))
        if unknown:
            raise AccountsError(
                f"services.telemetry_store.readers entry {name!r} names "
                f"{', '.join(unknown)}, which the store does not have; the tables are "
                f"{', '.join(ddl.TIME_COLUMNS)}"
            )
        days = entry.get("days_back")
        scopes.append(
            ReaderScope(
                name=name,
                tables=tuple(table for table in ddl.TIME_COLUMNS if table in stated),
                days_back=days
                if isinstance(days, int) and not isinstance(days, bool)
                else retention,
            )
        )
    return scopes


def _granted_tables(live: LiveAccounts, user: str) -> set[str]:
    return {
        table
        for grantee, access, database, table in live.grants
        if grantee == user and access == "SELECT" and database == ddl.DATABASE and table
    }


def _scoped_user(
    name: str,
    head: Sequence[str],
    tables: Sequence[str],
    live: LiveAccounts,
    *,
    policy_prefix: str | None = None,
    days_back: int = 0,
) -> list[Statement]:
    """One user's creates and its narrowing revokes: the readers' routine.

    *head* (the user and its roles) comes first; then per table in *tables* its
    ``<policy_prefix><table>`` row policy, when there is a prefix, before the
    SELECT grant that policy scopes. For every registry table not in *tables*,
    a SELECT the store reports granted is revoked, then the stale policy
    dropped, both in the revoke phase. The diff is computed from
    ``system.grants``, never ``REVOKE ALL``.
    """
    user = quote_ident(name)
    create = list(head)
    for table in tables:
        if policy_prefix is not None:
            # The policy before the grant it scopes: a run cut between them
            # leaves a policy and no access, never access without the policy.
            create.append(
                f"CREATE ROW POLICY OR REPLACE {quote_ident(policy_prefix + table)} "
                f"ON {table_ident(table)} AS RESTRICTIVE USING "
                f"{quote_ident(ddl.TIME_COLUMNS[table])} >= now() - INTERVAL "
                f"{days_back} DAY TO {user}"
            )
        create.append(f"GRANT SELECT ON {table_ident(table)} TO {user}")
    statements: list[Statement] = []
    granted = _granted_tables(live, name) if name in live.users else set()
    for table in ddl.TIME_COLUMNS:
        if table in tables:
            continue
        # Narrowing, from what the store holds: the revoke before the policy
        # drop, so a run cut between them leaves the policy and no access.
        if table in granted:
            statements.append(
                Statement(Phase.REVOKE, f"REVOKE SELECT ON {table_ident(table)} FROM {user}")
            )
        if policy_prefix is None:
            continue
        policy = policy_prefix + table
        if (policy, ddl.DATABASE, table) in live.policies:
            statements.append(
                Statement(
                    Phase.REVOKE,
                    f"DROP ROW POLICY IF EXISTS {quote_ident(policy)} ON {table_ident(table)}",
                )
            )
    statements.extend(Statement(Phase.CREATE, sql) for sql in create)
    return statements


def reader_statements(
    config: Mapping[str, Any] | None, passwords: Passwords, live: LiveAccounts
) -> list[Statement]:
    """Each configured reader's user, policies and grants, and its narrowing revokes.

    Raises:
        AccountsError: A configured reader's key is not in *passwords*, or its
            scope names a table the store does not have.
    """
    statements: list[Statement] = []
    for scope in reader_scopes(config):
        password = passwords.readers.get(scope.name)
        if password is None:
            raise AccountsError(
                f"no key for reader {scope.name!r}: {creds.reader_file(scope.name)} is "
                "missing; `osprey up` mints it before the store is staged"
            )
        head = [
            *_user(scope.name, password, profile=READER_PROFILE),
            f"GRANT {quote_ident(READERS_ROLE)} TO {quote_ident(scope.name)}",
        ]
        statements += _scoped_user(
            scope.name,
            head,
            scope.tables,
            live,
            policy_prefix=f"{READER_POLICY_PREFIX}{scope.name}_",
            days_back=scope.days_back,
        )
    return statements


def own_scope_policies(config: Mapping[str, Any] | None) -> list[str]:
    """The three own-scope policies, on role ``views_own`` only.

    A person's view user reads that person's rows. On the ledger it also reads
    the sidecar's rows on the person's behalf, and a queueserver's preflight
    rows on the person's behalf (and only those: agent-authored plans run
    inside the queueserver, so the page corroborates each such row).
    """
    from osprey.deployment.telemetry_gateway import trusted_on_behalf_of

    sidecar, queueservers = trusted_on_behalf_of(config)
    own = f"{quote_ident('User')} = {_VIEW_PERSON}"
    on_behalf = f"{quote_ident('OnBehalfOf')} = {_VIEW_PERSON}"
    ledger = f"{own} OR ({quote_ident('User')} = {quote_string(sidecar)} AND {on_behalf})"
    if queueservers:
        names = ", ".join(quote_string(name) for name in queueservers)
        ledger += (
            f" OR ({quote_ident('User')} IN ({names}) AND {quote_ident('Surface')} = "
            f"{quote_string(PREFLIGHT_SURFACE)} AND {on_behalf})"
        )
    return [
        f"CREATE ROW POLICY OR REPLACE {quote_ident(OWN_POLICY_PREFIX + table)} "
        f"ON {table_ident(table)} AS RESTRICTIVE USING "
        f"{ledger if table == 'audit_ledger' else own} TO {quote_ident(VIEWS_OWN_ROLE)}"
        for table in VIEW_TABLES
    ]


def _view_user(name: str) -> list[str]:
    ident = quote_ident(name)
    # Nobody signs in as a view user: the page reaches it only by EXECUTE AS.
    tail = (
        f"IDENTIFIED WITH no_password HOST NONE DEFAULT DATABASE {quote_ident(ddl.DATABASE)} "
        f"SETTINGS PROFILE {quote_string(VIEW_PROFILE)}"
    )
    return [
        f"CREATE USER IF NOT EXISTS {ident} {tail}",
        f"ALTER USER {ident} {tail}",
        f"GRANT {quote_ident(VIEWS_ROLE)} TO {ident}",
    ]


def _impersonate(name: str) -> Statement:
    return Statement(
        Phase.CREATE, f"GRANT IMPERSONATE ON {quote_ident(name)} TO {quote_ident(HISTORY_USER)}"
    )


def view_statements(config: Mapping[str, Any] | None, live: LiveAccounts) -> list[Statement]:
    """Each configured view user in its create order, and its narrowing revokes.

    Per view user: ``CREATE USER`` (and its re-assert), ``GRANT views``, then
    ``GRANT views_own`` (a person) or its ``viewreader_*`` policies (a reader),
    then the SELECT grants, then the ``history`` account's ``IMPERSONATE``. The
    own-scope policies come before any of them. Nothing when the history page
    is not deployed.
    """
    if not tc.history_page_deployed(config):
        return []
    statements = [Statement(Phase.CREATE, sql) for sql in own_scope_policies(config)]
    for person in roster_persons(config):
        name = person_view_user(person)
        # views_own before any SELECT: holding SELECT without it, the user
        # would read every person's rows.
        head = [*_view_user(name), f"GRANT {quote_ident(VIEWS_OWN_ROLE)} TO {quote_ident(name)}"]
        statements += [*_scoped_user(name, head, VIEW_TABLES, live), _impersonate(name)]
    statements += [
        *_scoped_user(VIEW_ALL_USER, _view_user(VIEW_ALL_USER), VIEW_TABLES, live),
        _impersonate(VIEW_ALL_USER),
    ]
    for scope in reader_scopes(config):
        name = reader_view_user(scope.name)
        statements += _scoped_user(
            name,
            _view_user(name),
            scope.tables,
            live,
            policy_prefix=f"{VIEW_READER_POLICY_PREFIX}{scope.name}_",
            days_back=scope.days_back,
        )
        statements.append(_impersonate(name))
    return statements


# --------------------------------------------------------------------------- core


def _profile(name: str, settings: Sequence[str]) -> list[str]:
    ident = quote_ident(name)
    return [
        f"CREATE SETTINGS PROFILE IF NOT EXISTS {ident}",
        f"ALTER SETTINGS PROFILE {ident} SETTINGS {_settings(settings)}",
    ]


def _memory_settings(query_memory: int) -> list[str]:
    """A query's memory cap, with sorts and aggregations spilling to disk at half of it."""
    spill = query_memory // 2
    return [
        f"max_memory_usage = {query_memory} CONST",
        f"max_bytes_before_external_sort = {spill} CONST",
        f"max_bytes_before_external_group_by = {spill} CONST",
    ]


def _reader_limits(limits: Mapping[str, int]) -> list[str]:
    return [
        "readonly = 1 CONST",
        f"max_execution_time = {limits['seconds']} MAX {limits['seconds']} CHANGEABLE_IN_READONLY",
        "timeout_overflow_mode = 'throw' CONST",
        f"max_result_rows = {limits['result_rows']} CONST",
        f"max_result_bytes = {limits['result_bytes']} CONST",
        "result_overflow_mode = 'throw' CONST",
        f"max_rows_to_read = {limits['rows_read']} CONST",
        f"max_bytes_to_read = {limits['bytes_read']} CONST",
        "read_overflow_mode = 'throw' CONST",
        *_memory_settings(limits["query_memory"]),
        f"max_concurrent_queries_for_user = {limits['concurrent']} CONST",
        "log_queries = 1 CONST",
        "log_queries_min_type = 'QUERY_START' CONST",
        "log_queries_probability = 1 CONST",
        "http_wait_end_of_query = 1 MIN 1 MAX 1 CHANGEABLE_IN_READONLY",
        "log_comment CHANGEABLE_IN_READONLY",
        *(f"{name} CHANGEABLE_IN_READONLY" for name in PLAY_SETTINGS),
        f"workload = {quote_string(READERS_WORKLOAD)} CONST",
    ]


def _view_limits(config: Mapping[str, Any] | None, limits: Mapping[str, int]) -> list[str]:
    return [
        "readonly = 1 CONST",
        f"max_execution_time = {HISTORY_SECONDS} CONST",
        f"max_result_rows = {limits['result_rows']} CONST",
        f"max_result_bytes = {2 * tc.gateway_max_log_size(config)} CONST",
        "result_overflow_mode = 'throw' CONST",
        f"max_rows_to_read = {limits['rows_read']} CONST",
        f"max_bytes_to_read = {limits['bytes_read']} CONST",
        "read_overflow_mode = 'throw' CONST",
        # Governed on the outer context by history_limits; these copies
        # matter only for a direct use of a view user, which HOST NONE prevents.
        *_memory_settings(limits["query_memory"]),
        "http_wait_end_of_query = 1 MIN 1 MAX 1",
        "log_queries = 1 CONST",
        # The page's quota_key is no setting: the HTTP handler reads it from
        # the URL as the quota key, so it needs no CHANGEABLE_IN_READONLY.
        "log_comment CHANGEABLE_IN_READONLY",
    ]


def _history_limits(limits: Mapping[str, int]) -> list[str]:
    # readonly = 0: IMPERSONATE is a write-DCL flag a readonly user is refused.
    # http_wait_end_of_query and max_memory_usage live here because the HTTP
    # handler and the query's memory tracker read them from the authenticated
    # outer context, before the EXECUTE AS subquery context exists.
    return [
        "readonly = 0 CONST",
        "log_queries = 1 CONST",
        f"max_execution_time = {HISTORY_SECONDS} CONST",
        "http_wait_end_of_query = 1 MIN 1 MAX 1",
        f"max_memory_usage = {limits['query_memory']} CONST",
    ]


def _user(
    name: str, password: str, *, settings: Sequence[str] = (), profile: str = ""
) -> list[str]:
    ident = quote_ident(name)
    tail = f"{identified(name, password)} DEFAULT DATABASE {quote_ident(ddl.DATABASE)}"
    if profile:
        tail += f" SETTINGS PROFILE {quote_string(profile)}"
    elif settings:
        tail += f" SETTINGS {_settings(settings)}"
    # Credential, database and settings in the one CREATE, so a run cut right
    # after it leaves a user that cannot do anything, never a passwordless one;
    # the ALTER re-asserts them on an existing user (a rotated password).
    return [f"CREATE USER IF NOT EXISTS {ident} {tail}", f"ALTER USER {ident} {tail}"]


def _core(config: Mapping[str, Any] | None, passwords: Passwords) -> list[str]:
    limits = tc.store_limits(config)
    retention = tc.retention_days(config)
    sql: list[str] = [
        f"CREATE ROLE IF NOT EXISTS {quote_ident(role)}"
        for role in (READERS_ROLE, VIEWS_ROLE, VIEWS_OWN_ROLE)
    ]
    sql += [
        f"CREATE RESOURCE IF NOT EXISTS {quote_ident(QUERY_RESOURCE)} (QUERY)",
        f"CREATE OR REPLACE WORKLOAD {quote_ident(READERS_WORKLOAD)} "
        f"SETTINGS max_concurrent_queries = {limits['reader_slots']}",
    ]
    sql += _profile(READER_PROFILE, _reader_limits(limits))
    sql += [
        f"CREATE QUOTA IF NOT EXISTS {quote_ident(READER_QUOTA)}",
        f"ALTER QUOTA {quote_ident(READER_QUOTA)} KEYED BY user_name "
        f"FOR INTERVAL 1 MINUTE MAX queries = {limits['per_minute']} "
        f"TO {quote_ident(READERS_ROLE)}",
    ]
    if tc.history_page_deployed(config):
        sql += _profile(VIEW_PROFILE, _view_limits(config, limits))
        sql += [
            f"CREATE QUOTA IF NOT EXISTS {quote_ident(VIEW_QUOTA)}",
            f"ALTER QUOTA {quote_ident(VIEW_QUOTA)} KEYED BY client_key "
            f"FOR INTERVAL 1 MINUTE MAX queries = {VIEW_QUERIES_PER_MINUTE} "
            f"TO {quote_ident(VIEWS_ROLE)}",
        ]
        sql += _profile(HISTORY_PROFILE, _history_limits(limits))
    if passwords.ingest is not None:
        sql += _user(
            INGEST_USER,
            passwords.ingest,
            settings=[
                "log_queries = 0 CONST",
                f"max_partitions_per_insert_block = {max(100, retention + 2)} CONST",
            ],
        )
        sql.append(f"GRANT INSERT ON {quote_ident(ddl.DATABASE)}.* TO {quote_ident(INGEST_USER)}")
    if passwords.archive is not None:
        sql += _user(
            ARCHIVE_USER,
            passwords.archive,
            settings=[
                "readonly = 1 CONST",
                f"max_execution_time = {ARCHIVE_SECONDS} CONST",
                *_memory_settings(limits["query_memory"]),
                f"output_format_parquet_row_group_size = {ARCHIVE_PARQUET_ROW_GROUP} CONST",
                "log_queries = 1 CONST",
            ],
        )
        sql.append(f"GRANT SELECT ON {quote_ident(ddl.DATABASE)}.* TO {quote_ident(ARCHIVE_USER)}")
        sql.append(f"GRANT SELECT ON `system`.`query_log` TO {quote_ident(ARCHIVE_USER)}")
    if passwords.history is not None:
        # No SELECT and no temporary tables: the page reads only as a view
        # user; each view user's IMPERSONATE follows that user's grants.
        sql += _user(HISTORY_USER, passwords.history, profile=HISTORY_PROFILE)
    return sql


def render_account_statements(
    config: Mapping[str, Any] | None,
    passwords: Passwords,
    live: LiveAccounts,
) -> list[Statement]:
    """The whole accounts script, in its global phase order.

    Raises:
        AccountsError: A configured name is a store user OSPREY does not own.
    """
    problems = adoption_problems(config, live)
    if problems:
        raise AccountsError("; ".join(problems))
    statements = [
        *_withdrawals(config, live),
        *(Statement(Phase.CREATE, sql) for sql in _core(config, passwords)),
        *reader_statements(config, passwords, live),
        *view_statements(config, live),
    ]
    # Stable: the per-entity order inside each phase is kept.
    return sorted(statements, key=lambda statement: statement.phase)
