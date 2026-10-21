"""The history page's view users: per-user statement order, scope, and the deployed gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment import telemetry_credentials as creds
from osprey.deployment.telemetry_store import accounts
from osprey.deployment.telemetry_store.accounts import LiveAccounts, Passwords, Phase

_PASSWORDS = Passwords(ingest="i", archive="a", history="h", readers={"own": "o", "jean-luc": "j"})
_ROSTER = [
    {"name": "jean-luc"},
    {"name": "alice"},
    {"name": "ops-desk", "access": "any"},
]


def _config(*, auth: str | None = "password", bluesky: bool = True) -> dict:
    web: dict = {"enabled": True, "users": _ROSTER}
    if auth is not None:
        web["auth"] = {"method": auth}
    return {
        "deployed_services": [
            "telemetry_store",
            "telemetry_gateway",
            *(["bluesky"] if bluesky else []),
        ],
        "modules": {"web_terminals": web},
        "services": {
            "telemetry_store": {
                "readers": [
                    {"name": "own", "tables": ["otel_logs"], "days_back": 3},
                    {"name": "jean-luc"},
                ]
            }
        },
    }


def _sql(config: dict, live: LiveAccounts | None = None) -> list[str]:
    return [
        s.sql
        for s in accounts.render_account_statements(config, _PASSWORDS, live or LiveAccounts())
    ]


def _user_block(sql: list[str], name: str) -> list[str]:
    """The statements that name *name* as a user, in script order."""
    ident = f"`{name}`"
    return [
        s
        for s in sql
        if f"TO {ident}" in s
        or f"USER IF NOT EXISTS {ident}" in s
        or s.startswith(f"ALTER USER {ident}")
        or f"IMPERSONATE ON {ident}" in s
    ]


def test_view_users_are_roster_persons_all_and_readers() -> None:
    assert accounts.configured_view_users(_config()) == [
        "view_jean-luc",
        "view_alice",
        "view__all",
        "view__reader_own",
        "view__reader_jean-luc",
    ]


def test_a_shared_card_gets_no_view_user() -> None:
    sql = _sql(_config())
    assert not any("view_ops-desk" in s for s in sql)


@pytest.mark.parametrize("person", ["jean-luc", "alice"])
def test_statement_order_for_a_persons_view_user(person: str) -> None:
    name = f"view_{person}"
    block = _user_block(_sql(_config()), name)
    assert block[0].startswith(
        f"CREATE USER IF NOT EXISTS `{name}` IDENTIFIED WITH no_password HOST NONE"
    )
    assert block[0].endswith("SETTINGS PROFILE 'view_limits'")
    assert block[1].startswith(f"ALTER USER `{name}` IDENTIFIED WITH no_password HOST NONE")
    assert block[2:] == [
        f"GRANT `views` TO `{name}`",
        f"GRANT `views_own` TO `{name}`",
        f"GRANT SELECT ON `telemetry`.`otel_logs` TO `{name}`",
        f"GRANT SELECT ON `telemetry`.`otel_traces` TO `{name}`",
        f"GRANT SELECT ON `telemetry`.`audit_ledger` TO `{name}`",
        f"GRANT IMPERSONATE ON `{name}` TO `history`",
    ]


def test_statement_order_for_the_all_view_user() -> None:
    block = _user_block(_sql(_config()), "view__all")
    assert block[2:] == [
        "GRANT `views` TO `view__all`",
        "GRANT SELECT ON `telemetry`.`otel_logs` TO `view__all`",
        "GRANT SELECT ON `telemetry`.`otel_traces` TO `view__all`",
        "GRANT SELECT ON `telemetry`.`audit_ledger` TO `view__all`",
        "GRANT IMPERSONATE ON `view__all` TO `history`",
    ]


def test_statement_order_for_a_readers_view_user() -> None:
    block = _user_block(_sql(_config()), "view__reader_own")
    assert block[2:] == [
        "GRANT `views` TO `view__reader_own`",
        "CREATE ROW POLICY OR REPLACE `viewreader_own_otel_logs` ON `telemetry`.`otel_logs` "
        "AS RESTRICTIVE USING `Timestamp` >= now() - INTERVAL 3 DAY TO `view__reader_own`",
        "GRANT SELECT ON `telemetry`.`otel_logs` TO `view__reader_own`",
        "GRANT IMPERSONATE ON `view__reader_own` TO `history`",
    ]
    assert "views_own` TO `view__reader_own`" not in " ".join(_sql(_config()))


def test_every_view_user_is_created_after_the_own_scope_policies_and_the_history_user() -> None:
    sql = _sql(_config())
    last_policy = max(i for i, s in enumerate(sql) if "ROW POLICY OR REPLACE `own_" in s)
    history = next(
        i for i, s in enumerate(sql) if s.startswith("CREATE USER IF NOT EXISTS `history`")
    )
    first_view = next(
        i for i, s in enumerate(sql) if s.startswith("CREATE USER IF NOT EXISTS `view_")
    )
    assert history < first_view and last_policy < first_view


def test_own_scope_policies_are_on_views_own_only() -> None:
    sql = _sql(_config())
    own = [s for s in sql if "ROW POLICY OR REPLACE `own_" in s]
    assert len(own) == 3
    assert all(s.endswith("TO `views_own`") for s in own)
    ledger = next(s for s in own if "`own_audit_ledger`" in s)
    assert ledger == (
        "CREATE ROW POLICY OR REPLACE `own_audit_ledger` ON `telemetry`.`audit_ledger` AS "
        "RESTRICTIVE USING `User` = substring(currentUser(), 6) OR (`User` = 'sidecar' AND "
        "`OnBehalfOf` = substring(currentUser(), 6)) OR (`User` IN ('queueserver') AND "
        "`Surface` = 'preflight' AND `OnBehalfOf` = substring(currentUser(), 6)) TO `views_own`"
    )


def test_without_a_queueserver_the_preflight_arm_is_absent() -> None:
    ledger = next(s for s in _sql(_config(bluesky=False)) if "`own_audit_ledger`" in s)
    assert "Surface" not in ledger
    assert "'sidecar'" in ledger


def test_a_reader_named_own_never_touches_the_own_scope_policies() -> None:
    sql = _sql(_config())
    for table in ("otel_logs", "otel_traces", "audit_ledger"):
        named = [s for s in sql if f"`own_{table}`" in s]
        assert named == [
            next(s for s in sql if s.startswith(f"CREATE ROW POLICY OR REPLACE `own_{table}`"))
        ]
    assert any("`reader_own_otel_logs`" in s for s in sql)
    assert any("`viewreader_own_otel_logs`" in s for s in sql)


def test_withdrawal_order_for_a_persons_view_user() -> None:
    live = LiveAccounts(
        users={"view_bob"},
        role_members={"views": {"view_bob"}, "views_own": {"view_bob"}},
        grants=[
            ("view_bob", "SELECT", "telemetry", "otel_logs"),
            ("view_bob", "SELECT", "telemetry", "otel_traces"),
        ],
    )
    statements = accounts.render_account_statements(_config(), _PASSWORDS, live)
    drops = [s.sql for s in statements if s.phase is Phase.DROP]
    assert drops == [
        "REVOKE SELECT ON `telemetry`.`otel_logs` FROM `view_bob`",
        "REVOKE SELECT ON `telemetry`.`otel_traces` FROM `view_bob`",
        "REVOKE `views_own` FROM `view_bob`",
        "DROP USER IF EXISTS `view_bob`",
    ]
    assert statements[: len(drops)] == [accounts.Statement(Phase.DROP, sql) for sql in drops]


def test_withdrawing_a_reader_withdraws_its_view_user_and_policies() -> None:
    config = _config()
    config["services"]["telemetry_store"]["readers"] = [{"name": "jean-luc"}]
    live = LiveAccounts(
        users={"own", "view__reader_own"},
        role_members={"readers": {"own"}, "views": {"view__reader_own"}},
        grants=[("view__reader_own", "SELECT", "telemetry", "otel_logs")],
    )
    drops = [
        s.sql
        for s in accounts.render_account_statements(config, _PASSWORDS, live)
        if s.phase is Phase.DROP
    ]
    i = drops.index("DROP USER IF EXISTS `view__reader_own`")
    assert drops[i - 1] == "REVOKE SELECT ON `telemetry`.`otel_logs` FROM `view__reader_own`"
    assert drops[i + 1].startswith("DROP ROW POLICY IF EXISTS `viewreader_own_otel_logs`")


def test_narrowing_a_readers_view_user_revokes_then_drops_its_policy() -> None:
    live = LiveAccounts(
        users={"view__reader_own"},
        role_members={"views": {"view__reader_own"}},
        grants=[
            ("view__reader_own", "SELECT", "telemetry", "otel_logs"),
            ("view__reader_own", "SELECT", "telemetry", "audit_ledger"),
        ],
        policies={("viewreader_own_audit_ledger", "telemetry", "audit_ledger")},
    )
    statements = accounts.render_account_statements(_config(), _PASSWORDS, live)
    assert [s.sql for s in statements if s.phase is Phase.REVOKE] == [
        "REVOKE SELECT ON `telemetry`.`audit_ledger` FROM `view__reader_own`",
        "DROP ROW POLICY IF EXISTS `viewreader_own_audit_ledger` ON `telemetry`.`audit_ledger`",
    ]


@pytest.mark.parametrize("auth", ["password", "oidc"])
def test_deployed_page_renders_view_users_and_history(auth: str) -> None:
    sql = _sql(_config(auth=auth))
    assert any(s.startswith("CREATE USER IF NOT EXISTS `history`") for s in sql)
    assert any(s.startswith("CREATE USER IF NOT EXISTS `view__all`") for s in sql)
    assert any(s.startswith("ALTER SETTINGS PROFILE `view_limits`") for s in sql)


@pytest.mark.parametrize("auth", [None, "none"])
def test_without_the_page_no_view_user_history_or_its_files(
    auth: str | None, tmp_path: Path
) -> None:
    config = _config(auth=auth)
    live = LiveAccounts(
        users={"history", "view_alice", "view__all"},
        role_members={"views": {"view_alice", "view__all"}, "views_own": {"view_alice"}},
        grants=[("view__all", "SELECT", "telemetry", "otel_logs")],
    )
    statements = accounts.render_account_statements(
        config, Passwords(archive="a", readers=_PASSWORDS.readers), live
    )
    creates = [s.sql for s in statements if s.phase is Phase.CREATE]
    assert not any("`view_" in s or "`history`" in s or "view_limits" in s for s in creates)
    drops = [s.sql for s in statements if s.phase is Phase.DROP]
    assert "DROP USER IF EXISTS `view_alice`" in drops
    assert "DROP USER IF EXISTS `view__all`" in drops
    assert "DROP USER IF EXISTS `history`" in drops
    assert drops.index("REVOKE `views_own` FROM `view_alice`") < drops.index(
        "DROP USER IF EXISTS `view_alice`"
    )
    # The history account's files are purged in the same up.
    for name in (creds.HISTORY_FILE, creds.HISTORY_GATE_FILE):
        (tmp_path / name).write_text("X=1\n")
    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    assert set(report.purged) >= {creds.HISTORY_FILE, creds.HISTORY_GATE_FILE}
    assert not (tmp_path / creds.HISTORY_FILE).exists()
    assert accounts.read_passwords(tmp_path, config).history is None


def test_a_hand_made_view_user_is_never_adopted() -> None:
    with pytest.raises(accounts.AccountsError, match="view user 'view_alice' names a store user"):
        _sql(_config(), LiveAccounts(users={"view_alice"}))
    cut = LiveAccounts(users={"view_alice"}, user_profiles={"view_alice": {"view_limits"}})
    assert accounts.adoption_problems(_config(), cut) == []
