"""Tests for the ``services.graphdb`` config schema resolver and its preflight."""

import dataclasses

import pytest

from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.graphdb_service import (
    DEFAULT_BIND_ADDRESS,
    DEFAULT_HEAP_INITIAL_SIZE,
    DEFAULT_HEAP_MAX_SIZE,
    DEFAULT_HTTP_PORT,
    DEFAULT_IMAGE,
    DEFAULT_PAGECACHE_SIZE,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USERNAME,
    GRAPHDB_HTTP_PORT_CONFIG_KEY,
    GRAPHDB_PASSWORD_ENV,
    GRAPHDB_PORT_CONFIG_KEY,
    GRAPHDB_SERVICE_NAME,
    GraphdbConnection,
    GraphdbServiceConfig,
    preflight_graphdb_config,
    resolve_bind_address,
    resolve_graphdb_connection,
    resolve_graphdb_service_config,
)


@pytest.fixture(autouse=True)
def _no_ambient_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the resolver off whatever ``GRAPHDB_PASSWORD`` the host carries.

    :func:`resolve_graphdb_connection` reads the process environment whenever no
    ``env`` mapping is passed, so a developer or runner that exports the
    variable would fail the default-password cases below for a reason that has
    nothing to do with the code under test. The one test that wants the ambient
    value sets it itself, which still wins — this only clears what the host
    brought.
    """
    monkeypatch.delenv(GRAPHDB_PASSWORD_ENV, raising=False)


class TestAbsentBlock:
    """A deployment without a graph store resolves to ``None``, not defaults."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"services": {}},
            {"services": {"postgresql": {"port_host": 5432}}},
            {"services": None},
            {"services": {"graphdb": None}},
            {"services": ["graphdb"]},
        ],
        ids=[
            "none",
            "empty",
            "no-graphdb",
            "other-service",
            "services-null",
            "graphdb-null",
            "services-not-a-mapping",
        ],
    )
    def test_absent_resolves_to_none(self, config: dict | None) -> None:
        assert resolve_graphdb_service_config(config) is None


class TestDefaults:
    """A present-but-empty block fills in every default."""

    def test_empty_block_gets_defaults(self) -> None:
        resolved = resolve_graphdb_service_config({"services": {"graphdb": {}}})
        assert resolved == GraphdbServiceConfig(
            image=DEFAULT_IMAGE,
            port_host=DEFAULT_PORT,
            http_port_host=DEFAULT_HTTP_PORT,
            bind_address=DEFAULT_BIND_ADDRESS,
            heap_initial_size=DEFAULT_HEAP_INITIAL_SIZE,
            heap_max_size=DEFAULT_HEAP_MAX_SIZE,
            pagecache_size=DEFAULT_PAGECACHE_SIZE,
            ttl_path=None,
        )

    def test_partial_block_keeps_the_stated_key_and_defaults_the_rest(self) -> None:
        """The half an operator wrote wins; the half they did not is defaulted."""
        resolved = resolve_graphdb_service_config({"services": {"graphdb": {"port_host": 17687}}})
        assert resolved is not None
        assert resolved.port_host == 17687
        assert resolved.http_port_host == DEFAULT_HTTP_PORT
        assert resolved.image == DEFAULT_IMAGE

    def test_explicit_values_win(self) -> None:
        resolved = resolve_graphdb_service_config(
            {
                "deployment": {"bind_address": "0.0.0.0"},
                "services": {
                    "graphdb": {
                        "image": "mirror.local/neo4j:5.20-community",
                        "port_host": 17687,
                        "http_port_host": 17474,
                        "heap_initial_size": "2G",
                        "heap_max_size": "4G",
                        "pagecache_size": "2G",
                        "ttl_path": "./data/demo_machine.ttl",
                    }
                },
            }
        )
        assert resolved == GraphdbServiceConfig(
            image="mirror.local/neo4j:5.20-community",
            port_host=17687,
            http_port_host=17474,
            bind_address="0.0.0.0",
            heap_initial_size="2G",
            heap_max_size="4G",
            pagecache_size="2G",
            ttl_path="./data/demo_machine.ttl",
        )

    def test_unknown_keys_are_ignored(self) -> None:
        """``path`` and the external-store keys are read elsewhere, not here."""
        resolved = resolve_graphdb_service_config(
            {
                "services": {
                    "graphdb": {
                        "path": "./services/graphdb",
                        "uri": "bolt://graph.example.org:7687",
                        "username": "neo4j",
                    }
                }
            }
        )
        assert resolved == GraphdbServiceConfig()

    def test_config_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            GraphdbServiceConfig().port_host = 1234  # type: ignore[misc]


class TestConstants:
    """The dotted keys and defaults other modules import from here."""

    def test_dotted_port_keys_are_spelled_exactly_once(self) -> None:
        assert GRAPHDB_PORT_CONFIG_KEY == "services.graphdb.port_host"
        assert GRAPHDB_HTTP_PORT_CONFIG_KEY == "services.graphdb.http_port_host"

    def test_the_two_port_keys_are_distinct(self) -> None:
        """A port-conflict remedy has to name which of the two ports to move."""
        assert GRAPHDB_PORT_CONFIG_KEY != GRAPHDB_HTTP_PORT_CONFIG_KEY

    def test_default_ports_match_neo4j_container_ports(self) -> None:
        assert (DEFAULT_PORT, DEFAULT_HTTP_PORT) == (7687, 7474)

    def test_image_is_pinned_to_the_last_release_neosemantics_builds_for(self) -> None:
        """n10s publishes no build for 5.23+, and without n10s nothing seeds."""
        assert DEFAULT_IMAGE == "neo4j:5.20-community"

    def test_service_name_matches_the_dotted_keys(self) -> None:
        assert GRAPHDB_SERVICE_NAME == "graphdb"
        assert GRAPHDB_PORT_CONFIG_KEY.startswith(f"services.{GRAPHDB_SERVICE_NAME}.")


class TestBindAddress:
    """``bind_address`` is project-wide, never a per-service key."""

    def test_reuses_the_qmd_reading(self) -> None:
        """One reader for one key: this module imports it rather than re-spelling it."""
        from osprey.deployment import qmd_service

        assert resolve_bind_address is qmd_service.resolve_bind_address

    @pytest.mark.parametrize(
        "config",
        [None, {}, {"deployment": {}}, {"deployment": None}],
        ids=["none", "empty", "no-address", "deployment-null"],
    )
    def test_defaults_to_loopback(self, config: dict | None) -> None:
        resolved = resolve_graphdb_service_config({**(config or {}), "services": {"graphdb": {}}})
        assert resolved is not None
        assert resolved.bind_address == "127.0.0.1"

    def test_per_service_bind_address_is_ignored(self) -> None:
        """A hand-written ``services.graphdb.bind_address`` must not take effect.

        Honouring it would let the store publish on an interface the rest of the
        stack does not, which is exactly the split the single project-wide key
        exists to prevent.
        """
        resolved = resolve_graphdb_service_config(
            {"services": {"graphdb": {"bind_address": "0.0.0.0"}}}
        )
        assert resolved is not None
        assert resolved.bind_address == "127.0.0.1"


class TestValidation:
    """A malformed scalar refuses rather than silently defaulting."""

    @pytest.mark.parametrize("bad", [0, -1, 65536, "7687", 7687.0, True, [7687]])
    def test_bad_bolt_port_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.graphdb\.port_host"):
            resolve_graphdb_service_config({"services": {"graphdb": {"port_host": bad}}})

    @pytest.mark.parametrize("bad", [0, -1, 65536, "7474", 7474.0, True])
    def test_bad_http_port_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.graphdb\.http_port_host"):
            resolve_graphdb_service_config({"services": {"graphdb": {"http_port_host": bad}}})

    @pytest.mark.parametrize(
        "key",
        ["heap_initial_size", "heap_max_size", "pagecache_size"],
    )
    @pytest.mark.parametrize("bad", ["", "   ", "512 m", "half a gig", "512mb", 512, True])
    def test_bad_memory_size_raises(self, key: str, bad: object) -> None:
        with pytest.raises(ValueError, match=rf"services\.graphdb\.{key}"):
            resolve_graphdb_service_config({"services": {"graphdb": {key: bad}}})

    @pytest.mark.parametrize("good", ["512m", "1G", "2048k", "1024", " 512m "])
    def test_accepted_memory_sizes(self, good: str) -> None:
        resolved = resolve_graphdb_service_config(
            {"services": {"graphdb": {"heap_max_size": good}}}
        )
        assert resolved is not None
        assert resolved.heap_max_size == good.strip()

    @pytest.mark.parametrize("bad", ["", "   ", 5, True, ["neo4j:5.20-community"]])
    def test_bad_image_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.graphdb\.image"):
            resolve_graphdb_service_config({"services": {"graphdb": {"image": bad}}})

    @pytest.mark.parametrize("bad", ["", "   ", 7, True, ["demo_machine.ttl"]])
    def test_bad_ttl_path_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.graphdb\.ttl_path"):
            resolve_graphdb_service_config({"services": {"graphdb": {"ttl_path": bad}}})

    def test_ttl_path_is_kept_verbatim(self) -> None:
        """Relative stays relative: consumers resolve it against config.yml's directory."""
        resolved = resolve_graphdb_service_config(
            {"services": {"graphdb": {"ttl_path": " ./data/demo_machine.ttl "}}}
        )
        assert resolved is not None
        assert resolved.ttl_path == "./data/demo_machine.ttl"


class TestDeployedWithoutBlockPreflight:
    """A deployed graph store must be described, not defaulted."""

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {},
            {"deployed_services": []},
            {"deployed_services": None},
            {"deployed_services": ["postgresql", "qmd"]},
            {"deployed_services": ["postgresql"], "services": {"graphdb": {}}},
            {"deployed_services": ["graphdb"], "services": {"graphdb": {}}},
            {"deployed_services": ("graphdb",), "services": {"graphdb": {"port_host": 7687}}},
        ],
        ids=[
            "none",
            "empty",
            "nothing-deployed",
            "deployed-null",
            "graphdb-not-deployed",
            "declared-but-not-deployed",
            "deployed-with-block",
            "deployed-with-block-tuple",
        ],
    )
    def test_satisfied_preflight_is_a_no_op(self, config: dict | None) -> None:
        assert preflight_graphdb_config(config) is None

    @pytest.mark.parametrize(
        "config",
        [
            {"deployed_services": ["graphdb"]},
            {"deployed_services": ["graphdb"], "services": {}},
            {"deployed_services": ["graphdb"], "services": {"graphdb": None}},
            {"deployed_services": ["postgresql", "graphdb"], "services": {"qmd": {}}},
        ],
        ids=["no-services", "empty-services", "graphdb-null", "other-service-only"],
    )
    def test_deployed_without_block_refuses(self, config: dict) -> None:
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_graphdb_config(config)
        assert "services.graphdb" in excinfo.value.reason

    def test_refusal_names_the_missing_block_and_both_ways_out(self) -> None:
        with pytest.raises(DeploymentPreconditionError) as excinfo:
            preflight_graphdb_config({"deployed_services": ["graphdb"]})
        assert "services.graphdb" in excinfo.value.reason
        assert "deployed_services" in excinfo.value.reason
        remedy = excinfo.value.remedy
        assert "services:" in remedy and "graphdb:" in remedy
        assert str(DEFAULT_PORT) in remedy
        assert str(DEFAULT_HTTP_PORT) in remedy
        assert "deployed_services" in remedy

    def test_malformed_block_still_raises_valueerror(self) -> None:
        """The preflight resolves the block, so a typo in it is not masked."""
        with pytest.raises(ValueError, match=r"services\.graphdb\.port_host"):
            preflight_graphdb_config(
                {"deployed_services": ["graphdb"], "services": {"graphdb": {"port_host": "7687"}}}
            )


class TestConnectionPrecedence:
    """An explicit ``uri`` wins verbatim; everything else derives from the port."""

    def test_connection_explicit_uri_wins_verbatim(self) -> None:
        """An external store has no local port to derive from, so nothing is rebuilt."""
        resolved = resolve_graphdb_connection(
            {"uri": "neo4j+s://graph.example.org:7687", "port_host": 17687}
        )
        assert resolved.uri == "neo4j+s://graph.example.org:7687"
        assert resolved.username == DEFAULT_USERNAME

    def test_connection_explicit_uri_pairs_with_the_stated_username(self) -> None:
        resolved = resolve_graphdb_connection(
            {"uri": "bolt://graph.example.org:7687", "username": "okf_reader"}
        )
        assert (resolved.uri, resolved.username) == (
            "bolt://graph.example.org:7687",
            "okf_reader",
        )

    def test_connection_explicit_uri_is_stripped_not_rewritten(self) -> None:
        resolved = resolve_graphdb_connection({"uri": "  bolt://graph.example.org:7687  "})
        assert resolved.uri == "bolt://graph.example.org:7687"

    @pytest.mark.parametrize("bad", ["", "   ", 7687, True, ["bolt://x"]])
    def test_connection_blank_uri_raises(self, bad: object) -> None:
        """A half-finished edit refuses rather than silently deriving a local URI."""
        with pytest.raises(ValueError, match=r"services\.graphdb\.uri"):
            resolve_graphdb_connection({"uri": bad})

    @pytest.mark.parametrize("bad", ["", "   ", 4, True])
    def test_connection_blank_username_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match=r"services\.graphdb\.username"):
            resolve_graphdb_connection({"uri": "bolt://graph.example.org:7687", "username": bad})

    @pytest.mark.parametrize(
        "section",
        [None, {}, {"path": "./services/graphdb"}, {"port_host": None}],
        ids=["none", "empty", "unrelated-keys", "port-null"],
    )
    def test_connection_derives_the_default_bolt_url(self, section: dict | None) -> None:
        resolved = resolve_graphdb_connection(section)
        assert resolved == GraphdbConnection(
            uri=f"bolt://localhost:{DEFAULT_PORT}",
            username=DEFAULT_USERNAME,
            password=DEFAULT_PASSWORD,
        )
        assert resolved.uri == "bolt://localhost:7687"

    def test_connection_derives_a_moved_port(self) -> None:
        """A project that moved ``port_host`` stays connectable without a second edit."""
        resolved = resolve_graphdb_connection({"port_host": 17687})
        assert resolved.uri == "bolt://localhost:17687"

    def test_connection_reads_the_port_from_the_services_argument(self) -> None:
        """The service block may reach the resolver separately from the connection keys."""
        resolved = resolve_graphdb_connection({}, services={"port_host": 17687})
        assert resolved.uri == "bolt://localhost:17687"

    def test_connection_services_argument_supplies_the_port_not_the_uri(self) -> None:
        resolved = resolve_graphdb_connection(
            {"uri": "bolt://graph.example.org:7687"}, services={"port_host": 17687}
        )
        assert resolved.uri == "bolt://graph.example.org:7687"

    @pytest.mark.parametrize("bad", [0, -1, 65536, "17687", 7687.0, True])
    def test_connection_bad_port_raises(self, bad: object) -> None:
        """The same refusal the schema gives — not ``bolt://localhost:True``."""
        with pytest.raises(ValueError, match=r"services\.graphdb\.port_host"):
            resolve_graphdb_connection({"port_host": bad})

    def test_connection_derived_username_ignores_a_configured_one(self) -> None:
        """The store this deployment runs authenticates as ``neo4j`` and nothing else.

        ``NEO4J_AUTH`` is the composite ``neo4j/${GRAPHDB_PASSWORD}``, so honouring a
        ``username`` without a ``uri`` would hand back a credential the container it
        points at does not have.
        """
        resolved = resolve_graphdb_connection({"username": "okf_reader"})
        assert resolved.username == DEFAULT_USERNAME
        assert resolved.uri == f"bolt://localhost:{DEFAULT_PORT}"


class TestConnectionPassword:
    """The password comes from the environment mapping, never from config.yml."""

    def test_connection_password_comes_from_the_env_mapping(self) -> None:
        resolved = resolve_graphdb_connection({}, env={GRAPHDB_PASSWORD_ENV: "minted-secret"})
        assert resolved.password == "minted-secret"

    def test_connection_password_applies_to_the_explicit_uri_path_too(self) -> None:
        """External stores get their credential the same way — the archiver's pattern."""
        resolved = resolve_graphdb_connection(
            {"uri": "neo4j+s://graph.example.org:7687", "username": "okf_reader"},
            env={GRAPHDB_PASSWORD_ENV: "minted-secret"},
        )
        assert resolved.password == "minted-secret"

    def test_connection_password_falls_back_to_the_template_default(self) -> None:
        """Matches the compose fallback, so a project is dialable before `osprey up`."""
        resolved = resolve_graphdb_connection({}, env={})
        assert resolved.password == DEFAULT_PASSWORD == "ospreygraph"

    def test_connection_password_reads_os_environ_when_no_mapping_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(GRAPHDB_PASSWORD_ENV, "ambient-secret")
        assert resolve_graphdb_connection({}).password == "ambient-secret"

    def test_connection_password_env_mapping_shuts_out_the_ambient_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deploy acting ON a project must not reach some other deployment's store."""
        monkeypatch.setenv(GRAPHDB_PASSWORD_ENV, "ambient-secret")
        assert resolve_graphdb_connection({}, env={}).password == DEFAULT_PASSWORD
        assert (
            resolve_graphdb_connection({}, env={GRAPHDB_PASSWORD_ENV: "project-secret"}).password
            == "project-secret"
        )

    def test_connection_password_env_var_is_spelled_once(self) -> None:
        assert GRAPHDB_PASSWORD_ENV == "GRAPHDB_PASSWORD"


class TestConnectionDataclass:
    """The shape consumers hand to ``neo4j.GraphDatabase.driver(uri, auth=...)``."""

    def test_connection_field_order_is_uri_username_password(self) -> None:
        connection = GraphdbConnection("bolt://localhost:7687", "neo4j", "secret")
        assert connection.uri == "bolt://localhost:7687"
        assert connection.username == "neo4j"
        assert connection.password == "secret"

    def test_connection_is_frozen(self) -> None:
        connection = resolve_graphdb_connection({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            connection.password = "swapped"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "section",
        [{}, {"port_host": 17687}, {"uri": "bolt://graph.example.org:7687"}],
        ids=["derived-default", "derived-moved-port", "explicit"],
    )
    def test_connection_never_embeds_credentials_in_the_uri(self, section: dict) -> None:
        """The driver takes auth separately, so no ``user:password@host`` DSN."""
        resolved = resolve_graphdb_connection(section, env={GRAPHDB_PASSWORD_ENV: "minted-secret"})
        assert "minted-secret" not in resolved.uri
        assert "@" not in resolved.uri

    def test_connection_repr_does_not_leak_the_password(self) -> None:
        resolved = resolve_graphdb_connection({}, env={GRAPHDB_PASSWORD_ENV: "minted-secret"})
        assert "minted-secret" not in repr(resolved)
        assert "bolt://localhost:7687" in repr(resolved)
