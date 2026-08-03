"""The ``ariel_dsn_port`` cross-check between an explicit DSN and ``port_host``.

``ariel.database.uri`` is optional: with it unset the DSN is derived from
``services.postgresql``, so it follows a port move by construction. An explicit
uri is a second, independent copy of the same facts — a config rendered before
the project moved ``port_host`` keeps its literal ``:5432`` and quietly points
ARIEL at a database that is no longer listening there.

These tests pin the two halves of that: the stale explicit uri warns with advice
naming both keys, and every configuration that is *not* a stale duplicate — a
derived DSN, an external database, a matching port — stays silent.
"""

from __future__ import annotations

from pathlib import Path

from osprey.health.core.configuration import ConfigState, configuration
from osprey.health.models import CheckResult, Status


def _rows(config: dict) -> dict[str, CheckResult]:
    """Run the category over a cleanly parsed config and index rows by name."""
    state = ConfigState(
        config_path=Path("/proj/config.yml"),
        exists=True,
        cwd=Path("/proj"),
        config=config,
    )
    return {r.name: r for r in configuration(state)()}


def _config(*, database: dict | None = None, postgresql: dict | None = None) -> dict:
    """A config carrying only the keys this cross-check reads."""
    config: dict = {}
    if database is not None:
        config["ariel"] = {"database": database}
    if postgresql is not None:
        config["services"] = {"postgresql": postgresql}
    return config


#: The rendered Postgres service after a project moved its published port.
_MOVED_PORT = {"port_host": 5433, "username": "ariel", "database_name": "ariel"}

#: The uri a config rendered before that move keeps verbatim.
_STALE_URI = "postgresql://ariel:ariel@localhost:5432/ariel"


class TestStaleExplicitUriWarns:
    def test_port_mismatch_warns(self):
        row = _rows(_config(database={"uri": _STALE_URI}, postgresql=_MOVED_PORT))["ariel_dsn_port"]
        assert row.status is Status.WARNING

    def test_message_reports_both_ports(self):
        row = _rows(_config(database={"uri": _STALE_URI}, postgresql=_MOVED_PORT))["ariel_dsn_port"]
        assert "5432" in row.message
        assert "5433" in row.message

    def test_remediation_names_both_keys(self):
        row = _rows(_config(database={"uri": _STALE_URI}, postgresql=_MOVED_PORT))["ariel_dsn_port"]
        assert "ariel.database.uri" in row.details
        assert "services.postgresql.port_host" in row.details

    def test_remediation_states_both_honest_fixes(self):
        """Delete the duplicate to derive it, or correct the port it names."""
        row = _rows(_config(database={"uri": _STALE_URI}, postgresql=_MOVED_PORT))["ariel_dsn_port"]
        assert "delete" in row.details.lower()
        assert "5433" in row.details

    def test_legacy_connection_string_is_checked_like_an_explicit_uri(self):
        """The retired alias is honored as the DSN, so it can go stale the same way."""
        row = _rows(_config(database={"connection_string": _STALE_URI}, postgresql=_MOVED_PORT))[
            "ariel_dsn_port"
        ]
        assert row.status is Status.WARNING
        assert "ariel.database.connection_string" in row.message
        assert "ariel.database.uri" in row.details
        assert "services.postgresql.port_host" in row.details

    def test_loopback_ip_form_is_checked_too(self):
        row = _rows(
            _config(
                database={"uri": "postgresql://ariel:ariel@127.0.0.1:5432/ariel"},
                postgresql=_MOVED_PORT,
            )
        )["ariel_dsn_port"]
        assert row.status is Status.WARNING

    def test_string_port_host_is_compared_numerically(self):
        """YAML quoting of the port must not silence the check."""
        row = _rows(_config(database={"uri": _STALE_URI}, postgresql={"port_host": "5433"}))[
            "ariel_dsn_port"
        ]
        assert row.status is Status.WARNING


class TestMatchingPortIsSilent:
    def test_matching_port_reports_ok(self):
        row = _rows(_config(database={"uri": _STALE_URI}, postgresql={"port_host": 5432}))[
            "ariel_dsn_port"
        ]
        assert row.status is Status.OK

    def test_matching_port_does_not_raise_the_exit_code(self):
        rows = _rows(_config(database={"uri": _STALE_URI}, postgresql={"port_host": 5432}))
        assert rows["ariel_dsn_port"].status is not Status.WARNING


class TestNothingToCrossCheck:
    def test_derived_dsn_emits_no_row(self):
        """No explicit uri: the DSN follows port_host by construction."""
        rows = _rows(_config(database={}, postgresql=_MOVED_PORT))
        assert "ariel_dsn_port" not in rows

    def test_absent_ariel_section_emits_no_row(self):
        rows = _rows(_config(postgresql=_MOVED_PORT))
        assert "ariel_dsn_port" not in rows

    def test_no_postgresql_service_emits_no_row(self):
        """An explicit uri with no local Postgres is a legitimate external DB."""
        rows = _rows(_config(database={"uri": _STALE_URI}))
        assert "ariel_dsn_port" not in rows

    def test_postgresql_service_without_port_host_emits_no_row(self):
        rows = _rows(_config(database={"uri": _STALE_URI}, postgresql={"username": "ariel"}))
        assert "ariel_dsn_port" not in rows

    def test_external_database_host_emits_no_row(self):
        """port_host publishes *this* project's Postgres, not a remote server's."""
        rows = _rows(
            _config(
                database={"uri": "postgresql://user:pw@db.example.org:5432/ariel"},
                postgresql=_MOVED_PORT,
            )
        )
        assert "ariel_dsn_port" not in rows

    def test_portless_uri_emits_no_row(self):
        """Nothing explicit to have gone stale."""
        rows = _rows(
            _config(database={"uri": "postgresql://localhost/ariel"}, postgresql=_MOVED_PORT)
        )
        assert "ariel_dsn_port" not in rows


class TestUnparseableValuesAreHandledGracefully:
    def test_garbage_uri_emits_no_row(self):
        rows = _rows(_config(database={"uri": "not a dsn at all"}, postgresql=_MOVED_PORT))
        assert "ariel_dsn_port" not in rows

    def test_non_numeric_port_in_uri_emits_no_row(self):
        rows = _rows(
            _config(
                database={"uri": "postgresql://ariel@localhost:not-a-port/ariel"},
                postgresql=_MOVED_PORT,
            )
        )
        assert "ariel_dsn_port" not in rows

    def test_unresolved_placeholder_port_emits_no_row(self, monkeypatch):
        monkeypatch.delenv("ARIEL_DB_PORT", raising=False)
        rows = _rows(
            _config(
                database={"uri": "postgresql://ariel@localhost:${ARIEL_DB_PORT}/ariel"},
                postgresql=_MOVED_PORT,
            )
        )
        assert "ariel_dsn_port" not in rows

    def test_resolved_placeholder_port_is_compared(self, monkeypatch):
        monkeypatch.setenv("ARIEL_DB_PORT", "5432")
        row = _rows(
            _config(
                database={"uri": "postgresql://ariel@localhost:${ARIEL_DB_PORT}/ariel"},
                postgresql=_MOVED_PORT,
            )
        )["ariel_dsn_port"]
        assert row.status is Status.WARNING

    def test_non_string_uri_emits_no_row(self):
        rows = _rows(_config(database={"uri": 5432}, postgresql=_MOVED_PORT))
        assert "ariel_dsn_port" not in rows

    def test_non_numeric_port_host_emits_no_row(self):
        rows = _rows(_config(database={"uri": _STALE_URI}, postgresql={"port_host": "auto"}))
        assert "ariel_dsn_port" not in rows

    def test_null_database_block_emits_no_row(self):
        rows = _rows({"ariel": {"database": None}, "services": {"postgresql": _MOVED_PORT}})
        assert "ariel_dsn_port" not in rows
