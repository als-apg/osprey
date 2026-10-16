"""The store's accounts script: goldens, the global phase order, adoption, quoting.

Regenerate the goldens with:

    PYTHONPATH=src ./.venv/bin/python tests/deployment/telemetry_store/test_store_accounts.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment.telemetry_store import accounts
from osprey.deployment.telemetry_store.accounts import LiveAccounts, Passwords, Phase

_GOLDENS = Path(__file__).parent / "goldens"
_PASSWORDS = Passwords(
    ingest="ingest-pw",
    archive="archive-pw",
    history="history-pw",
    readers={"2nd-shift": "shift-pw", "jean-luc": "jl-pw", "own": "own-pw"},
)
_OWN_POLICIES = {
    ("own_otel_logs", "telemetry", "otel_logs"),
    ("own_otel_traces", "telemetry", "otel_traces"),
    ("own_audit_ledger", "telemetry", "audit_ledger"),
}
#: A store holding ``jean-luc`` from a run that granted it the traces, a stale
#: policy of the reader ``own`` on the traces, and the own-scope policies.
_READERS_LIVE = LiveAccounts(
    users={"jean-luc"},
    role_members={"readers": {"jean-luc"}},
    user_profiles={"jean-luc": {"reader_limits"}},
    grants=[
        ("jean-luc", "SELECT", "telemetry", "otel_logs"),
        ("jean-luc", "SELECT", "telemetry", "otel_traces"),
        ("jean-luc", "SELECT", "telemetry", "audit_ledger"),
    ],
    policies={
        ("reader_jean-luc_otel_logs", "telemetry", "otel_logs"),
        ("reader_jean-luc_otel_traces", "telemetry", "otel_traces"),
        ("reader_jean-luc_audit_ledger", "telemetry", "audit_ledger"),
        ("reader_own_otel_traces", "telemetry", "otel_traces"),
        *_OWN_POLICIES,
    },
)
_READERS_CONFIG = {
    "deployed_services": ["telemetry_store"],
    "services": {
        "telemetry_store": {
            "readers": [
                {"name": "2nd-shift"},
                {"name": "jean-luc", "tables": ["otel_logs", "audit_ledger"], "days_back": 3},
                {"name": "own", "tables": ["otel_logs"]},
            ]
        }
    },
}
_WEB = {"enabled": True, "auth": {"method": "password"}}

#: name -> (config, live state) of each golden.
SCENARIOS = {
    "core_store_only": ({"deployed_services": ["telemetry_store"]}, LiveAccounts()),
    "core_with_gateway_and_history": (
        {
            "deployed_services": ["telemetry_store", "telemetry_gateway"],
            "modules": {"web_terminals": _WEB},
            "services": {"telemetry_store": {"retention_days": 120, "limits": {"reader_slots": 3}}},
        },
        LiveAccounts(),
    ),
    # Owned readers no longer configured, with hyphen and digit identifiers,
    # and a history account left from a deployment that served the page.
    "withdrawals": (
        {
            "deployed_services": ["telemetry_store"],
            "services": {"telemetry_store": {"readers": [{"name": "jean-luc"}]}},
        },
        LiveAccounts(
            users={"2nd-shift", "jean-luc", "cut-run", "history", "ingest", "view__reader_x"},
            role_members={"readers": {"2nd-shift", "jean-luc"}, "views": {"view__reader_x"}},
            user_profiles={"cut-run": {"reader_limits"}},
        ),
    ),
    # A new reader with the default scope, a reader narrowed from three tables
    # to two, and a reader whose name is the own-scope policies' prefix.
    "readers": (_READERS_CONFIG, _READERS_LIVE),
    # The history page's view users: roster persons (a shared card gets none),
    # all, and one per reader incl. one named `own`; a withdrawn person still
    # holding its grants, and a reader's view user narrowed from three tables.
    "views": (
        {
            **_READERS_CONFIG,
            "deployed_services": ["telemetry_store", "telemetry_gateway", "bluesky"],
            "modules": {
                "web_terminals": {
                    **_WEB,
                    "users": [
                        {"name": "jean-luc"},
                        {"name": "2nd-shift"},
                        {"name": "ops-desk", "access": "any"},
                    ],
                }
            },
        },
        LiveAccounts(
            users={"view_bob", "view__reader_jean-luc"},
            role_members={
                "views": {"view_bob", "view__reader_jean-luc"},
                "views_own": {"view_bob"},
            },
            grants=[
                ("view_bob", "SELECT", "telemetry", "otel_logs"),
                ("view_bob", "SELECT", "telemetry", "audit_ledger"),
                ("view__reader_jean-luc", "SELECT", "telemetry", "otel_logs"),
                ("view__reader_jean-luc", "SELECT", "telemetry", "otel_traces"),
            ],
            policies={
                ("viewreader_jean-luc_otel_traces", "telemetry", "otel_traces"),
                *_OWN_POLICIES,
            },
        ),
    ),
}


def _render(name: str) -> str:
    config, live = SCENARIOS[name]
    passwords = _PASSWORDS
    statements = accounts.render_account_statements(config, passwords, live)
    return "".join(f"-- phase {s.phase.value}\n{s.sql};\n" for s in statements)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_golden(name: str) -> None:
    assert _render(name) == (_GOLDENS / f"accounts_{name}.sql").read_text()


def test_phase_order_is_global() -> None:
    config, live = SCENARIOS["withdrawals"]
    statements = accounts.render_account_statements(config, _PASSWORDS, live)
    phases = [s.phase for s in statements]
    assert phases == sorted(phases)
    drops = [s.sql for s in statements if s.phase is Phase.DROP]
    assert drops[0] == "DROP USER IF EXISTS `2nd-shift`"
    assert "DROP USER IF EXISTS `cut-run`" in drops
    assert "DROP USER IF EXISTS `jean-luc`" not in drops
    assert "DROP USER IF EXISTS `history`" in drops
    assert "DROP USER IF EXISTS `ingest`" in drops
    assert "DROP ROW POLICY IF EXISTS `viewreader_x_otel_logs` ON `telemetry`.`otel_logs`" in drops
    # A user's policies go after the user.
    first_user = drops.index("DROP USER IF EXISTS `2nd-shift`")
    assert drops[first_user + 1].startswith("DROP ROW POLICY IF EXISTS `reader_2nd-shift_")


def test_stable_order_inside_a_phase() -> None:
    statements = accounts.render_account_statements(
        {"deployed_services": ["telemetry_store", "telemetry_gateway"]}, _PASSWORDS, LiveAccounts()
    )
    sql = [s.sql for s in statements]
    assert sql.index("CREATE ROLE IF NOT EXISTS `readers`") < sql.index(
        "ALTER QUOTA `reader_quota` KEYED BY user_name FOR INTERVAL 1 MINUTE MAX queries = 60 "
        "TO `readers`"
    )
    create = next(
        i for i, s in enumerate(sql) if s.startswith("CREATE USER IF NOT EXISTS `ingest`")
    )
    assert sql[create + 1].startswith("ALTER USER `ingest` IDENTIFIED WITH sha256_hash")
    assert sql[create + 2] == "GRANT INSERT ON `telemetry`.* TO `ingest`"


def test_hand_made_user_is_never_adopted() -> None:
    config = {
        "deployed_services": ["telemetry_store"],
        "services": {"telemetry_store": {"readers": [{"name": "alice"}]}},
    }
    with pytest.raises(accounts.AccountsError, match="'alice' names a store user OSPREY did not"):
        accounts.render_account_statements(config, _PASSWORDS, LiveAccounts(users={"alice"}))
    # A cut run (profile, no role) is owned.
    cut = LiveAccounts(users={"alice"}, user_profiles={"alice": {"reader_limits"}})
    assert accounts.adoption_problems(config, cut) == []


def test_workload_binds_the_readers_pool() -> None:
    sql = [
        s.sql
        for s in accounts.render_account_statements(
            {"deployed_services": ["telemetry_store"]}, _PASSWORDS, LiveAccounts()
        )
    ]
    assert "CREATE RESOURCE IF NOT EXISTS `osprey_queries` (QUERY)" in sql
    assert "CREATE OR REPLACE WORKLOAD `readers` SETTINGS max_concurrent_queries = 2" in sql
    profile = next(s for s in sql if s.startswith("ALTER SETTINGS PROFILE `reader_limits`"))
    assert profile.endswith("workload = 'readers' CONST")


def test_a_reader_may_change_the_settings_play_sends() -> None:
    """``/play`` sends ``send_logs_level`` on every run and ``http_response_headers``
    on Download; a ``readonly = 1`` reader may send them only when marked changeable."""
    assert {"send_logs_level", "http_response_headers"} <= set(accounts.PLAY_SETTINGS)
    sql = [
        s.sql
        for s in accounts.render_account_statements(
            {"deployed_services": ["telemetry_store"]}, _PASSWORDS, LiveAccounts()
        )
    ]
    profile = next(s for s in sql if s.startswith("ALTER SETTINGS PROFILE `reader_limits`"))
    assert "readonly = 1 CONST" in profile
    for name in ("send_logs_level", "http_response_headers"):
        assert f"{name} CHANGEABLE_IN_READONLY" in profile


def _sql(config=_READERS_CONFIG, live=_READERS_LIVE, passwords=_PASSWORDS) -> list[str]:
    return [s.sql for s in accounts.render_account_statements(config, passwords, live)]


def test_a_reader_is_created_in_one_statement_with_its_profile() -> None:
    sql = _sql()
    create = next(
        i for i, s in enumerate(sql) if s.startswith("CREATE USER IF NOT EXISTS `2nd-shift`")
    )
    assert sql[create].endswith("DEFAULT DATABASE `telemetry` SETTINGS PROFILE 'reader_limits'")
    assert "IDENTIFIED WITH sha256_hash BY '" in sql[create]
    assert "shift-pw" not in "".join(sql)
    assert sql[create + 1].startswith("ALTER USER `2nd-shift` IDENTIFIED WITH sha256_hash")
    assert sql[create + 1].endswith("SETTINGS PROFILE 'reader_limits'")
    assert sql[create + 2] == "GRANT `readers` TO `2nd-shift`"


def test_each_policy_comes_before_the_grant_it_scopes() -> None:
    sql = _sql()
    for table, column in (
        ("otel_logs", "Timestamp"),
        ("otel_traces", "Timestamp"),
        ("audit_ledger", "Timestamp"),
    ):
        policy = (
            f"CREATE ROW POLICY OR REPLACE `reader_2nd-shift_{table}` ON `telemetry`.`{table}` "
            f"AS RESTRICTIVE USING `{column}` >= now() - INTERVAL 14 DAY TO `2nd-shift`"
        )
        grant = f"GRANT SELECT ON `telemetry`.`{table}` TO `2nd-shift`"
        assert sql.index(policy) + 1 == sql.index(grant)
    # The default scope reads no metrics table.
    assert not any("otel_metrics" in s and "2nd-shift" in s for s in sql)


def test_days_back_and_tables_scope_the_reader() -> None:
    sql = _sql()
    assert (
        "CREATE ROW POLICY OR REPLACE `reader_jean-luc_audit_ledger` ON `telemetry`.`audit_ledger` "
        "AS RESTRICTIVE USING `Timestamp` >= now() - INTERVAL 3 DAY TO `jean-luc`"
    ) in sql
    assert "GRANT SELECT ON `telemetry`.`otel_traces` TO `jean-luc`" not in sql


def test_narrowing_revokes_then_drops_the_stale_policy_before_any_create() -> None:
    statements = accounts.render_account_statements(_READERS_CONFIG, _PASSWORDS, _READERS_LIVE)
    revokes = [s.sql for s in statements if s.phase is Phase.REVOKE]
    assert revokes == [
        "REVOKE SELECT ON `telemetry`.`otel_traces` FROM `jean-luc`",
        "DROP ROW POLICY IF EXISTS `reader_jean-luc_otel_traces` ON `telemetry`.`otel_traces`",
        "DROP ROW POLICY IF EXISTS `reader_own_otel_traces` ON `telemetry`.`otel_traces`",
    ]
    phases = [s.phase for s in statements]
    assert phases == sorted(phases)
    first_create = phases.index(Phase.CREATE)
    assert all(p is Phase.REVOKE for p in phases[:first_create])


def test_a_grant_the_store_never_held_is_not_revoked() -> None:
    sql = _sql(live=LiveAccounts())
    assert not any(s.startswith(("REVOKE", "DROP")) for s in sql)


def test_a_reader_named_own_leaves_the_own_scope_policies_alone() -> None:
    sql = _sql()
    for short, _db, _table in _OWN_POLICIES:
        assert not any(f"`{short}`" in s for s in sql), short
    assert "GRANT SELECT ON `telemetry`.`otel_logs` TO `own`" in sql
    assert any(s.startswith("CREATE ROW POLICY OR REPLACE `reader_own_otel_logs`") for s in sql)


def test_a_rotated_key_is_reasserted_with_its_new_hash() -> None:
    old = _sql(live=_READERS_LIVE)
    rotated = Passwords(readers={**_PASSWORDS.readers, "jean-luc": "new-pw"}, archive="archive-pw")
    new = _sql(passwords=rotated)
    alter_old = next(s for s in old if s.startswith("ALTER USER `jean-luc`"))
    alter_new = next(s for s in new if s.startswith("ALTER USER `jean-luc`"))
    assert alter_old != alter_new
    assert accounts.identified("jean-luc", "new-pw") in alter_new


def test_a_reader_without_a_key_is_refused_by_file_name() -> None:
    with pytest.raises(accounts.AccountsError, match=r"\.env\.telemetry-reader\.own"):
        _sql(passwords=Passwords(archive="a", readers={"2nd-shift": "x", "jean-luc": "y"}))


def test_an_unknown_reader_table_is_refused() -> None:
    config = {
        "deployed_services": ["telemetry_store"],
        "services": {"telemetry_store": {"readers": [{"name": "x", "tables": ["nope"]}]}},
    }
    with pytest.raises(accounts.AccountsError, match="names nope"):
        accounts.reader_scopes(config)


def test_reader_slots_set_the_workload_and_the_server_ceiling() -> None:
    from osprey.deployment import telemetry_config as tc
    from osprey.deployment.telemetry_store import render

    config = {
        "deployed_services": ["telemetry_store"],
        "services": {"telemetry_store": {"limits": {"reader_slots": 5}}},
    }
    sql = _sql(config=config, live=LiveAccounts())
    assert "CREATE OR REPLACE WORKLOAD `readers` SETTINGS max_concurrent_queries = 5" in sql
    context = render.render_context(config)
    assert context["max_concurrent_select_queries"] == 5 + tc.STORE_SPARE_QUERY_SLOTS == 7


def test_live_policies_are_read_from_the_store_rows() -> None:
    live = accounts.live_from_rows(
        {
            "policies": [
                {"short_name": "own_otel_logs", "database": "telemetry", "table": "otel_logs"}
            ]
        }
    )
    assert live.policies == {("own_otel_logs", "telemetry", "otel_logs")}


def test_quoting() -> None:
    assert accounts.quote_ident("jean-luc") == "`jean-luc`"
    assert accounts.quote_ident("a`b\\c") == "`a\\`b\\\\c`"
    assert accounts.quote_string("it's") == "'it\\'s'"


def test_password_hash_matches_the_store_check() -> None:
    import hashlib
    import re

    clause = accounts.identified("ingest", "secret")
    digest, salt = re.findall(r"'([0-9a-f]+)'", clause)
    assert hashlib.sha256(("secret" + salt).encode()).hexdigest() == digest
    assert "secret" not in clause


def test_read_passwords_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(accounts.AccountsError, match=r"\.env\.telemetry-archive"):
        accounts.read_passwords(tmp_path, {"deployed_services": ["telemetry_store"]})


def _store_with_readers(*names: str) -> dict:
    return {
        "deployed_services": ["telemetry_store"],
        "services": {"telemetry_store": {"readers": [{"name": n} for n in names]}},
    }


def test_a_withdrawn_readers_key_file_is_renamed_in_the_same_up(tmp_path: Path) -> None:
    """The start sets the key aside, then stages a script that drops the user."""
    import datetime

    from osprey.deployment import telemetry_credentials as creds

    today = datetime.date(2026, 3, 4)
    creds.ensure_telemetry_credentials(
        tmp_path, _store_with_readers("ops", "eve"), reader_keys=True, today=today
    )
    ops_key = (tmp_path / creds.reader_file("ops")).read_text()

    # The next up, with ops gone from the config: credentials first, then the
    # accounts script, exactly as `osprey up` runs them.
    config = _store_with_readers("eve")
    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True, today=today)
    aside = f"{creds.reader_file('ops')}.withdrawn-2026-03-04"
    assert report.withdrawn == [("ops", aside)]
    assert not (tmp_path / creds.reader_file("ops")).exists()
    assert (tmp_path / aside).read_text() == ops_key

    passwords = accounts.read_passwords(tmp_path, config)
    assert set(passwords.readers) == {"eve"}
    live = LiveAccounts(users={"ops", "eve"}, role_members={"readers": {"ops", "eve"}})
    statements = accounts.render_account_statements(config, passwords, live)
    assert statements[0] == accounts.Statement(Phase.DROP, "DROP USER IF EXISTS `ops`")
    assert not any("`ops`" in s.sql for s in statements if s.phase is Phase.CREATE)


def test_a_deleted_key_file_rotates_the_readers_key(tmp_path: Path) -> None:
    from osprey.deployment import telemetry_credentials as creds

    config = _store_with_readers("eve")
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    before = accounts.read_passwords(tmp_path, config).readers["eve"]
    (tmp_path / creds.reader_file("eve")).unlink()
    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    assert report.reader_keys == ["eve"]
    after = accounts.read_passwords(tmp_path, config).readers["eve"]
    assert after != before


if __name__ == "__main__":
    _GOLDENS.mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        (_GOLDENS / f"accounts_{scenario}.sql").write_text(_render(scenario))
