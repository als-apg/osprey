"""Unit tests for the graph MCP server's connection lifecycle and read path.

The context is exercised against a stub driver rather than a live store: the
only thing that genuinely needs one is the fast-fail pin at the bottom of this
file, which dials an unroutable address on purpose. Everything else — the
configured/absent/malformed gate, the two tuning keys, the row cap, the
JSON-safety pass and the whole error map — is a decision the context makes
before or after the wire, so a fake session pins it exactly.

The module is imported module-qualified as ``graph_ctx``. ``tests/mcp_server/
conftest.py`` already imports a ``reset_server_context`` (the control-system
one) into this package's fixture namespace, and a bare ``from ... import
reset_server_context`` here would read as the wrong one.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from osprey.mcp_server.graph import server_context as graph_ctx

# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_ambient_graph_password(monkeypatch):
    """Keep the resolver off whatever ``GRAPHDB_PASSWORD`` the host carries."""
    monkeypatch.delenv("GRAPHDB_PASSWORD", raising=False)


@pytest.fixture(autouse=True)
def _reset_graph_context():
    """Drop the graph singleton around every test so none leaks a driver."""
    graph_ctx.reset_server_context()
    yield
    graph_ctx.reset_server_context()


def _configured_block(**overrides: Any) -> dict[str, Any]:
    """Build a config mapping describing a graph store.

    Args:
        **overrides: Keys to merge into the ``services.graphdb`` block.

    Returns:
        A config mapping the context reads as configured.
    """
    block: dict[str, Any] = {"path": "./services/graphdb", "port_host": 7687}
    block.update(overrides)
    return {"services": {"graphdb": block}}


def _build_context(monkeypatch, config: dict[str, Any] | None, **settings: Any):
    """Construct an initialized :class:`GraphContext` over a config mapping.

    Both seams the context reads config through are module globals, so patching
    them here is what lets every test below name its own store and its own
    tuning values without a config file on disk.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        config: The mapping ``load_osprey_config`` should return.
        **settings: Values ``get_config_value`` should serve, keyed by dotted
            config key — pass the module's key constants through ``**{...}``.

    Returns:
        The initialized context.
    """
    monkeypatch.setattr(graph_ctx, "load_osprey_config", lambda: config)
    values = dict(settings)
    monkeypatch.setattr(
        graph_ctx,
        "get_config_value",
        lambda key, default=None: values.get(key, default),
    )
    context = graph_ctx.GraphContext()
    context.initialize()
    return context


class _FakeRecord:
    """A driver record whose ``data()`` returns a preset mapping."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        """Return the record as a mapping, as the real driver does."""
        return dict(self._data)


class _FakeTransaction:
    """A transaction that answers any query with a fixed record list."""

    def __init__(self, records: list[_FakeRecord]) -> None:
        self._records = records
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, cypher: str, params: dict[str, Any]) -> list[_FakeRecord]:
        """Record the call and return an iterable of records."""
        self.calls.append((cypher, params))
        return list(self._records)


class _FakeSession:
    """A session that runs the transaction function, or raises instead."""

    def __init__(
        self, records: list[_FakeRecord] | None = None, error: Exception | None = None
    ) -> None:
        self.records = records or []
        self.error = error
        self.tx_function: Any = None
        self.transaction = _FakeTransaction(self.records)
        self.closed = False

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        self.closed = True
        return False

    def execute_read(self, tx_function: Any) -> Any:
        """Capture the transaction function, then run it or raise."""
        self.tx_function = tx_function
        if self.error is not None:
            raise self.error
        return tx_function(self.transaction)


class _FakeDriver:
    """A driver that hands out one prepared session."""

    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.session_kwargs: dict[str, Any] | None = None
        self.closed = False

    def session(self, **kwargs: Any) -> _FakeSession:
        """Record how the session was opened and return it."""
        self.session_kwargs = kwargs
        return self._session

    def close(self) -> None:
        """Mark the driver closed."""
        self.closed = True


def _with_driver(context, session: _FakeSession) -> _FakeDriver:
    """Attach a fake driver to *context* so no real connection is opened.

    Args:
        context: The context to install the driver on.
        session: The session the driver should hand out.

    Returns:
        The installed fake driver.
    """
    driver = _FakeDriver(session)
    context._driver = driver
    return driver


#: Every error the context can raise, so a new one cannot be added without the
#: taxonomy tests below noticing it.
_ERROR_CLASSES = (
    graph_ctx.GraphNotConfigured,
    graph_ctx.GraphUnreachable,
    graph_ctx.GraphAuthFailed,
    graph_ctx.GraphQueryTimeout,
    graph_ctx.GraphReadOnlyViolation,
    graph_ctx.GraphQueryError,
)


def _hook_error_classes() -> dict[str, str]:
    """Read ``ERROR_CLASS_MAP`` out of the error-guidance hook.

    Parsed from source rather than imported: the hook is a standalone script
    that inserts its own directory onto ``sys.path`` at import time, and a test
    that pins its taxonomy should not have that side effect.

    Returns:
        The hook's ``error_type`` to error-class mapping.
    """
    import ast
    import pathlib

    import osprey

    hook = (
        pathlib.Path(osprey.__file__).parent
        / "templates/claude_code/claude/hooks/osprey_error_guidance.py"
    )
    tree = ast.parse(hook.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ERROR_CLASS_MAP"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"ERROR_CLASS_MAP not found in {hook}")


def _server_error(code: str, message: str) -> Exception:
    """Build the exception the driver itself would raise for a server error.

    Goes through the driver's own hydration entry point rather than
    constructing a class directly: that is what picks ``AuthError`` over
    ``ClientError`` for a security code, and it is the only way to get an
    instance whose ``.code`` and ``.message`` read like a real one.

    Args:
        code: The Neo4j status code, e.g. ``Neo.ClientError.Statement.SyntaxError``.
        message: The server's message.

    Returns:
        The hydrated exception.
    """
    from neo4j.exceptions import Neo4jError

    return Neo4jError._hydrate_neo4j(code=code, message=message)


# ---------------------------------------------------------------------------
# Configuration gate
# ---------------------------------------------------------------------------


class TestConfigurationGate:
    """``configured`` is decided once, from the ``services.graphdb`` block."""

    def test_block_present_resolves_a_connection(self, monkeypatch):
        """A well-formed block makes the context configured and dialable."""
        context = _build_context(monkeypatch, _configured_block(port_host=7999))

        assert context.configured is True
        assert context.uri == "bolt://localhost:7999"

    def test_explicit_uri_wins_over_the_derived_address(self, monkeypatch):
        """An external store is dialed at the address it was given."""
        context = _build_context(
            monkeypatch, _configured_block(uri="bolt://graph.example.org:7687")
        )

        assert context.uri == "bolt://graph.example.org:7687"

    def test_absent_block_is_not_configured(self, monkeypatch):
        """A project with no graph store answers reads without dialing."""
        context = _build_context(monkeypatch, {"services": {"postgresql": {}}})

        assert context.configured is False
        assert context.uri is None

    def test_absent_services_section_is_not_configured(self, monkeypatch):
        """A config with no services at all is the same state, not an error."""
        context = _build_context(monkeypatch, {})

        assert context.configured is False

    def test_malformed_block_is_not_configured_and_warns(self, monkeypatch, caplog):
        """A typo reads as unconfigured, but it is logged loudly rather than swallowed."""
        with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.graph.server_context"):
            context = _build_context(monkeypatch, _configured_block(port_host="abc"))

        assert context.configured is False
        assert context.uri is None
        assert "malformed" in caplog.text
        assert "services.graphdb.port_host" in caplog.text

    def test_unreadable_config_degrades_instead_of_failing_startup(self, monkeypatch, caplog):
        """A config that will not load leaves the context unconfigured, not broken."""

        def _boom() -> dict[str, Any]:
            raise OSError("config.yml is unreadable")

        monkeypatch.setattr(graph_ctx, "load_osprey_config", _boom)
        monkeypatch.setattr(graph_ctx, "get_config_value", lambda key, default=None: default)

        with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.graph.server_context"):
            context = graph_ctx.GraphContext()
            context.initialize()

        assert context.configured is False
        assert "could not load config.yml" in caplog.text

    def test_initialize_is_idempotent(self, monkeypatch):
        """A second initialize() does not re-read config."""
        calls: list[int] = []

        def _load() -> dict[str, Any]:
            calls.append(1)
            return _configured_block()

        monkeypatch.setattr(graph_ctx, "load_osprey_config", _load)
        monkeypatch.setattr(graph_ctx, "get_config_value", lambda key, default=None: default)

        context = graph_ctx.GraphContext()
        context.initialize()
        context.initialize()

        assert len(calls) == 1

    def test_password_never_reaches_the_log(self, monkeypatch, caplog):
        """Initialization logs the address, never the credential."""
        monkeypatch.setenv("GRAPHDB_PASSWORD", "s3cret-value")

        with caplog.at_level(logging.DEBUG, logger="osprey.mcp_server.graph.server_context"):
            _build_context(monkeypatch, _configured_block())

        assert "s3cret-value" not in caplog.text


# ---------------------------------------------------------------------------
# Tuning keys
# ---------------------------------------------------------------------------


class TestTuningKeys:
    """``query_timeout_s`` and ``query_max_rows`` come from config with defaults."""

    def test_keys_are_spelled_as_the_contract_states(self):
        """The dotted keys other surfaces cite are these exact strings."""
        assert graph_ctx.QUERY_TIMEOUT_CONFIG_KEY == "services.graphdb.query_timeout_s"
        assert graph_ctx.QUERY_MAX_ROWS_CONFIG_KEY == "services.graphdb.query_max_rows"

    def test_defaults_when_unset(self, monkeypatch):
        """An unconfigured project still has a bounded timeout and row cap."""
        context = _build_context(monkeypatch, _configured_block())

        assert context.query_timeout_s == graph_ctx.DEFAULT_QUERY_TIMEOUT_S == 15
        assert context.query_max_rows == graph_ctx.DEFAULT_QUERY_MAX_ROWS == 200

    def test_config_overrides_both(self, monkeypatch):
        """Configured values replace the built-in defaults."""
        context = _build_context(
            monkeypatch,
            _configured_block(),
            **{
                graph_ctx.QUERY_TIMEOUT_CONFIG_KEY: 45,
                graph_ctx.QUERY_MAX_ROWS_CONFIG_KEY: 25,
            },
        )

        assert context.query_timeout_s == 45
        assert context.query_max_rows == 25

    @pytest.mark.parametrize("bad", ["abc", 0, -5, True, 2.5])
    def test_malformed_value_warns_and_falls_back(self, monkeypatch, caplog, bad):
        """A bad tuning value must not leave the agent with no graph tools."""
        with caplog.at_level(logging.WARNING, logger="osprey.mcp_server.graph.server_context"):
            context = _build_context(
                monkeypatch,
                _configured_block(),
                **{graph_ctx.QUERY_MAX_ROWS_CONFIG_KEY: bad},
            )

        assert context.query_max_rows == graph_ctx.DEFAULT_QUERY_MAX_ROWS
        assert "services.graphdb.query_max_rows" in caplog.text
        assert "positive integer" in caplog.text


# ---------------------------------------------------------------------------
# The read path
# ---------------------------------------------------------------------------


class TestRunRead:
    """``run_read`` bounds every answer and never leaks a driver type."""

    def test_unconfigured_raises_not_configured(self, monkeypatch):
        """No store means an immediate refusal, not a dial."""
        context = _build_context(monkeypatch, {})

        with pytest.raises(graph_ctx.GraphNotConfigured) as excinfo:
            context.run_read("RETURN 1")

        assert excinfo.value.error_type == "not_configured"
        assert any("services.graphdb" in s for s in excinfo.value.suggestions)

    def test_rows_and_parameters_reach_the_transaction(self, monkeypatch):
        """Records come back as plain dicts, with parameters passed through."""
        session = _FakeSession([_FakeRecord({"name": "SR:MAG:DIPOLE:01"})])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        result = context.run_read("MATCH (d) RETURN d.name AS name", {"limit": 5})

        assert result.rows == [{"name": "SR:MAG:DIPOLE:01"}]
        assert result.truncated is False
        assert session.transaction.calls == [("MATCH (d) RETURN d.name AS name", {"limit": 5})]

    def test_session_names_the_database_explicitly(self, monkeypatch):
        """Naming the database skips the driver's home-database round trip."""
        session = _FakeSession([_FakeRecord({"n": 1})])
        context = _build_context(monkeypatch, _configured_block())
        driver = _with_driver(context, session)

        context.run_read("RETURN 1 AS n")

        assert driver.session_kwargs == {"database": "neo4j"}

    def test_row_cap_truncates_and_flags(self, monkeypatch):
        """More rows than the cap allows come back clipped and flagged."""
        session = _FakeSession([_FakeRecord({"i": i}) for i in range(10)])
        context = _build_context(
            monkeypatch, _configured_block(), **{graph_ctx.QUERY_MAX_ROWS_CONFIG_KEY: 3}
        )
        _with_driver(context, session)

        result = context.run_read("MATCH (n) RETURN n")

        assert result.rows == [{"i": 0}, {"i": 1}, {"i": 2}]
        assert result.truncated is True

    def test_exactly_the_cap_is_not_truncated(self, monkeypatch):
        """A result that just fits must not be reported as clipped."""
        session = _FakeSession([_FakeRecord({"i": i}) for i in range(3)])
        context = _build_context(
            monkeypatch, _configured_block(), **{graph_ctx.QUERY_MAX_ROWS_CONFIG_KEY: 3}
        )
        _with_driver(context, session)

        result = context.run_read("MATCH (n) RETURN n")

        assert len(result.rows) == 3
        assert result.truncated is False

    def test_per_call_max_rows_overrides_the_default(self, monkeypatch):
        """A caller may ask for fewer rows than the configured cap."""
        session = _FakeSession([_FakeRecord({"i": i}) for i in range(10)])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        result = context.run_read("MATCH (n) RETURN n", max_rows=2)

        assert len(result.rows) == 2
        assert result.truncated is True

    def test_max_rows_below_one_is_raised_to_one(self, monkeypatch):
        """A zero-row answer would be indistinguishable from an empty graph."""
        session = _FakeSession([_FakeRecord({"i": 0}), _FakeRecord({"i": 1})])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        result = context.run_read("MATCH (n) RETURN n", max_rows=0)

        assert len(result.rows) == 1

    def test_transaction_function_carries_the_unit_of_work_timeout(self, monkeypatch):
        """The timeout is enforced server-side, via neo4j's own decorator."""
        session = _FakeSession([_FakeRecord({"n": 1})])
        context = _build_context(
            monkeypatch, _configured_block(), **{graph_ctx.QUERY_TIMEOUT_CONFIG_KEY: 7}
        )
        _with_driver(context, session)

        context.run_read("RETURN 1 AS n")

        # neo4j.unit_of_work stashes its arguments on the wrapped function; the
        # driver reads them back when it begins the transaction.
        assert session.tx_function.timeout == 7
        assert hasattr(session.tx_function, "metadata")


class TestJsonSafety:
    """Driver-native values must not reach a JSON envelope unconverted."""

    def test_temporal_and_spatial_values_become_strings(self, monkeypatch):
        """DateTime, Duration and Point all render as readable text."""
        from neo4j.spatial import WGS84Point
        from neo4j.time import Date, DateTime, Duration

        session = _FakeSession(
            [
                _FakeRecord(
                    {
                        "when": DateTime(2026, 8, 20, 12, 30, 0),
                        "day": Date(2026, 8, 20),
                        "how_long": Duration(hours=2),
                        "where": WGS84Point((13.4, 52.5)),
                    }
                )
            ]
        )
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        row = context.run_read("MATCH (n) RETURN n").rows[0]

        assert all(isinstance(value, str) for value in row.values())
        assert "2026-08-20" in row["when"]
        assert row["day"] == "2026-08-20"
        # A Point is a tuple subclass, so recursing into it as a container would
        # leave a bare [13.4, 52.5] with its coordinate system dropped. The
        # string form keeps that in the class name.
        assert row["where"] == "WGS84Point(13.4, 52.5)"
        assert row["how_long"].startswith("P")  # ISO-8601 duration

    def test_nested_containers_are_converted_recursively(self, monkeypatch):
        """Lists and maps inside a record get the same pass."""
        from neo4j.time import Date

        session = _FakeSession([_FakeRecord({"outer": {"inner": [Date(2026, 1, 2), 3, "text"]}})])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        row = context.run_read("MATCH (n) RETURN n").rows[0]

        assert row == {"outer": {"inner": ["2026-01-02", 3, "text"]}}

    def test_json_natives_pass_through_unchanged(self, monkeypatch):
        """Nothing that JSON already carries is stringified."""
        session = _FakeSession([_FakeRecord({"i": 3, "f": 1.5, "b": True, "s": "x", "none": None})])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        row = context.run_read("MATCH (n) RETURN n").rows[0]

        assert row == {"i": 3, "f": 1.5, "b": True, "s": "x", "none": None}


class TestErrorMapping:
    """Every way the store can refuse maps to one class carrying one remedy."""

    def _run_with_error(self, monkeypatch, error: Exception):
        """Run a read whose session raises *error* and return the context."""
        session = _FakeSession(error=error)
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)
        return context

    def test_auth_error_maps_to_auth_failed(self, monkeypatch):
        """A rejected credential names GRAPHDB_PASSWORD, never its value."""
        error = _server_error("Neo.ClientError.Security.Unauthorized", "bad credentials")
        context = self._run_with_error(monkeypatch, error)

        with pytest.raises(graph_ctx.GraphAuthFailed) as excinfo:
            context.run_read("RETURN 1")

        assert excinfo.value.error_type == "connection_error"
        remedies = " ".join(excinfo.value.suggestions)
        assert "GRAPHDB_PASSWORD" in remedies
        assert ".env" in remedies

    def test_auth_error_is_matched_before_client_error(self):
        """AuthError subclasses ClientError, so its arm has to come first."""
        from neo4j.exceptions import AuthError, ClientError

        assert issubclass(AuthError, ClientError)

    @pytest.mark.parametrize(
        "kind", ["service_unavailable", "session_expired", "os_error", "socket_timeout"]
    )
    def test_dial_failures_map_to_unreachable(self, monkeypatch, kind):
        """A store that does not answer names the port key and `osprey up`."""
        from neo4j.exceptions import ServiceUnavailable, SessionExpired

        errors: dict[str, Exception] = {
            "service_unavailable": ServiceUnavailable("cannot reach the store"),
            "session_expired": SessionExpired("session expired"),
            "os_error": OSError("connection refused"),
            # socket.timeout is an alias of TimeoutError, itself an OSError.
            "socket_timeout": TimeoutError("socket timed out"),
        }
        context = self._run_with_error(monkeypatch, errors[kind])

        with pytest.raises(graph_ctx.GraphUnreachable) as excinfo:
            context.run_read("RETURN 1")

        assert excinfo.value.error_type == "service_unavailable"
        remedies = " ".join(excinfo.value.suggestions)
        assert "services.graphdb.port_host" in remedies
        assert "services.graphdb.uri" in remedies
        assert "osprey up" in remedies

    def test_access_mode_maps_to_read_only_violation(self, monkeypatch):
        """A write that got past the gate is named as a write, not a syntax error."""
        error = _server_error(
            "Neo.ClientError.Statement.AccessMode",
            "Write queries cannot be performed in READ access mode.",
        )
        context = self._run_with_error(monkeypatch, error)

        with pytest.raises(graph_ctx.GraphReadOnlyViolation) as excinfo:
            context.run_read("CREATE (n)")

        assert excinfo.value.error_type == "validation_error"
        assert excinfo.value.details == {
            "kind": "read_only",
            "code": "Neo.ClientError.Statement.AccessMode",
        }
        assert "READ access mode" in str(excinfo.value)

    @pytest.mark.parametrize(
        "code",
        [
            "Neo.ClientError.Transaction.TransactionTimedOut",
            "Neo.TransientError.Transaction.TransactionTimedOutClientConfiguration",
        ],
    )
    def test_transaction_timeout_codes_map_to_timeout(self, monkeypatch, code):
        """Both codes the server uses for a terminated transaction land here."""
        context = self._run_with_error(monkeypatch, _server_error(code, "transaction ended"))

        with pytest.raises(graph_ctx.GraphQueryTimeout) as excinfo:
            context.run_read("MATCH (a)-[*]->(b) RETURN a, b")

        assert excinfo.value.error_type == "timeout_error"
        remedies = " ".join(excinfo.value.suggestions)
        assert "Narrow the query" in remedies
        assert "services.graphdb.query_timeout_s" in remedies

    @pytest.mark.parametrize("message", ["The transaction timed out", "Transaction terminated"])
    def test_timeout_recognised_from_the_message(self, monkeypatch, message):
        """An unrecognised code still reads as a timeout when the message says so."""
        error = _server_error("Neo.TransientError.General.SomethingElse", message)
        context = self._run_with_error(monkeypatch, error)

        with pytest.raises(graph_ctx.GraphQueryTimeout):
            context.run_read("MATCH (n) RETURN n")

    def test_syntax_error_maps_to_query_error(self, monkeypatch):
        """A rejected query points at the schema, not at the service."""
        error = _server_error("Neo.ClientError.Statement.SyntaxError", "Invalid input 'MATHC'")
        context = self._run_with_error(monkeypatch, error)

        with pytest.raises(graph_ctx.GraphQueryError) as excinfo:
            context.run_read("MATHC (n) RETURN n")

        assert excinfo.value.error_type == "validation_error"
        assert excinfo.value.details == {
            "kind": "query_error",
            "code": "Neo.ClientError.Statement.SyntaxError",
        }
        assert "Invalid input 'MATHC'" in str(excinfo.value)
        assert any("get_schema" in s for s in excinfo.value.suggestions)

    def test_database_error_maps_to_query_error(self, monkeypatch):
        """A server-side failure is still a reached store, so it is not 'unreachable'."""
        error = _server_error("Neo.DatabaseError.General.UnknownError", "internal failure")
        context = self._run_with_error(monkeypatch, error)

        with pytest.raises(graph_ctx.GraphQueryError):
            context.run_read("MATCH (n) RETURN n")

    def test_every_error_is_a_graph_store_error(self):
        """One base class is what lets the tool layer catch the family at once."""
        for cls in _ERROR_CLASSES:
            assert issubclass(cls, graph_ctx.GraphStoreError)


class TestErrorTaxonomy:
    """``error_type`` values have to be ones the error-guidance hook knows."""

    def test_error_types_are_hook_taxonomy_keys(self):
        """An unknown value classes as 'Internal' and mis-advises the agent.

        The one deliberate exception is GraphNotConfigured: 'Internal' is the
        correct class for it, because no rewording of the query adds a
        services.graphdb block — an operator has to.
        """
        known = _hook_error_classes()

        for cls in _ERROR_CLASSES:
            if cls is graph_ctx.GraphNotConfigured:
                assert cls.error_type not in known
                continue
            assert cls.error_type in known, (
                f"{cls.__name__}.error_type={cls.error_type!r} is not in the hook's "
                f"ERROR_CLASS_MAP, so the hook would class it as 'Internal'"
            )

    def test_error_types_map_to_the_intended_classes(self):
        """Each cause lands in the hook class that matches its remedy."""
        known = _hook_error_classes()

        assert known[graph_ctx.GraphUnreachable.error_type] == "Connection"
        assert known[graph_ctx.GraphAuthFailed.error_type] == "Connection"
        assert known[graph_ctx.GraphQueryTimeout.error_type] == "Connection"
        assert known[graph_ctx.GraphReadOnlyViolation.error_type] == "Validation"
        assert known[graph_ctx.GraphQueryError.error_type] == "Validation"

    def test_shared_error_types_stay_distinguishable_in_details(self):
        """Two causes share validation_error, so details['kind'] separates them."""
        assert graph_ctx.GraphReadOnlyViolation.error_type == graph_ctx.GraphQueryError.error_type
        assert graph_ctx.GraphReadOnlyViolation("x").details == {"kind": "read_only"}
        assert graph_ctx.GraphQueryError("x").details == {"kind": "query_error"}

    def test_details_default_to_none_where_the_type_is_unique(self):
        """A cause its error_type already identifies carries no details."""
        assert graph_ctx.GraphUnreachable("x").details is None
        assert graph_ctx.GraphAuthFailed("x").details is None
        assert graph_ctx.GraphQueryTimeout("x").details is None
        assert graph_ctx.GraphNotConfigured("x").details is None

    def test_caller_supplied_details_are_merged_with_the_kind(self):
        """The kind is stamped on without displacing what the mapper passed."""
        error = graph_ctx.GraphQueryError("x", details={"code": "Neo.ClientError.X.Y"})

        assert error.details == {"code": "Neo.ClientError.X.Y", "kind": "query_error"}


class TestIsEmpty:
    """The bootstrapped-but-unseeded state is a question of its own."""

    def test_zero_resources_is_empty(self, monkeypatch):
        """A store with no Resource nodes was never seeded."""
        session = _FakeSession([_FakeRecord({"n": 0})])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        assert context.is_empty() is True
        assert session.transaction.calls[0][0] == "MATCH (r:Resource) RETURN count(r) AS n"

    def test_a_populated_store_is_not_empty(self, monkeypatch):
        """Any Resource node means the corpus is there."""
        session = _FakeSession([_FakeRecord({"n": 8123})])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        assert context.is_empty() is False

    def test_no_rows_reads_as_empty(self, monkeypatch):
        """A count that returned nothing is not evidence of a corpus."""
        session = _FakeSession([])
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        assert context.is_empty() is True

    def test_unreachable_store_is_not_reported_as_empty(self, monkeypatch):
        """'Empty' and 'down' are different states with different remedies."""
        from neo4j.exceptions import ServiceUnavailable

        session = _FakeSession(error=ServiceUnavailable("down"))
        context = _build_context(monkeypatch, _configured_block())
        _with_driver(context, session)

        with pytest.raises(graph_ctx.GraphUnreachable):
            context.is_empty()


class TestSingleton:
    """The module trio matches the bluesky server_context shape."""

    def test_get_before_initialize_raises(self):
        """A tool that runs before startup gets a clear error, not None."""
        with pytest.raises(RuntimeError, match="not initialized"):
            graph_ctx.get_server_context()

    def test_initialize_then_get_returns_the_same_object(self, monkeypatch):
        """One context per process, shared by every tool."""
        monkeypatch.setattr(graph_ctx, "load_osprey_config", lambda: _configured_block())
        monkeypatch.setattr(graph_ctx, "get_config_value", lambda key, default=None: default)

        created = graph_ctx.initialize_server_context()

        assert graph_ctx.get_server_context() is created
        assert created.configured is True

    def test_reset_closes_the_driver(self, monkeypatch):
        """Resetting must not leak the connection pool between tests."""
        monkeypatch.setattr(graph_ctx, "load_osprey_config", lambda: _configured_block())
        monkeypatch.setattr(graph_ctx, "get_config_value", lambda key, default=None: default)

        context = graph_ctx.initialize_server_context()
        driver = _with_driver(context, _FakeSession())

        graph_ctx.reset_server_context()

        assert driver.closed is True
        with pytest.raises(RuntimeError):
            graph_ctx.get_server_context()

    def test_shutdown_survives_a_driver_that_refuses_to_close(self, monkeypatch):
        """Teardown is best-effort; a failing close must not propagate."""

        class _AngryDriver(_FakeDriver):
            def close(self) -> None:
                raise RuntimeError("close failed")

        context = _build_context(monkeypatch, _configured_block())
        context._driver = _AngryDriver(_FakeSession())

        context.shutdown()

        assert context._driver is None


# ---------------------------------------------------------------------------
# Fast-fail pin (real driver, unroutable address)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_unreachable_store_fails_within_the_budget(monkeypatch):
    """An unreachable store raises GraphUnreachable in seconds, not minutes.

    Success Criterion 5 allows eight seconds, and the whole point of
    ``connection_timeout=3.0`` plus ``max_transaction_retry_time=0.0`` is to
    stay well inside it. This is the one test that uses the real driver:
    192.0.2.1 is TEST-NET-1, reserved for documentation and unroutable, so the
    dial can only end in a timeout.
    """
    context = _build_context(monkeypatch, _configured_block(uri="bolt://192.0.2.1:7687"))
    assert context.configured is True

    started = time.perf_counter()
    with pytest.raises(graph_ctx.GraphUnreachable):
        context.run_read("RETURN 1")
    elapsed = time.perf_counter() - started

    assert elapsed <= 8.0, f"unreachable store took {elapsed:.1f}s to fail"
    context.shutdown()
