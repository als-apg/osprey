"""Each service backend reads its block into its own settings model, on the one base.

A spec keeps its block as written in the base
:class:`~osprey_connectors.config_sections.ConnectorSettings`, which refuses
nothing; the backend reads it into its ``settings_model`` when it is built, so a
block its type refuses raises there, in the words it always had.
"""

from __future__ import annotations

import pytest

from osprey_connectors import logger as logger_module
from osprey_connectors.alarm_state._phoebus_kafka import PhoebusKafkaSettings
from osprey_connectors.alarm_state._replay import ReplayAlarmStateSettings
from osprey_connectors.archiver._doocs_archiver_connector import DoocsArchiverSettings
from osprey_connectors.archiver._epics_archiver_connector import EpicsArchiverSettings
from osprey_connectors.archiver._mock_archiver_connector import MockArchiverSettings
from osprey_connectors.archiver._mongodb_archiver_connector import MongoStoreSettings
from osprey_connectors.archiver._mya_archiver_connector import MyaArchiverSettings
from osprey_connectors.config import using_config
from osprey_connectors.config_sections import ConnectorSettings, Unreadable
from osprey_connectors.connection import Credential, PasswordLogin, TokenLogin
from osprey_connectors.directory._channelfinder import ChannelFinderSettings
from osprey_connectors.errors import ConfigurationError
from osprey_connectors.service_answers import BackendSpec
from osprey_connectors.testing import config_of

WHERE = "archiver.settings"

MONGO = {
    "host": "store",
    "port": 27017,  # osprey:not-a-port - a block under test, never dialled
    "name": "db",
    "collection": "c",
    "auth": {"source": "admin", "username": "u", "password_env": "P"},
}


def test_a_spec_keeps_its_block_as_written_and_refuses_nothing():
    spec = BackendSpec(
        kind="archive",
        type="epics_archiver",
        target="live",
        settings={"timeout_s": "not a number", "retrieval_path": 5},
        rehearsal=False,
    )
    assert type(spec.settings) is ConnectorSettings
    assert spec.settings.model_dump() == {"timeout_s": "not a number", "retrieval_path": 5}


class TestTheArchives:
    def test_the_appliance_reads_its_connection_and_prefix(self):
        read = EpicsArchiverSettings.from_block(
            {"url": "https://a.example", "retrieval_path": "proxy/retrieval/"}, key=WHERE
        )
        assert read.connection.url == "https://a.example"
        assert read.retrieval_path == "/proxy/retrieval"

    def test_the_appliance_without_a_url_is_refused_in_its_words(self):
        with pytest.raises(ConfigurationError, match="archiver URL is required") as raised:
            EpicsArchiverSettings.from_block({}, key=WHERE)
        assert raised.value.key == "archiver.settings.url"

    def test_the_store_reads_its_address_and_login(self):
        read = MongoStoreSettings.from_block(MONGO, key=WHERE)
        assert (read.host, read.port, read.database, read.collection) == ("store", 27017, "db", "c")
        assert read.login == PasswordLogin(username="u", password_env="P")
        assert read.with_env_overrides() == read
        assert (read.auth_source, read.timeout_s) == ("admin", 60)

    def test_a_numeric_string_port_is_its_number(self):
        """Behaviour change: pymongo refused a string port at the first call."""
        assert MongoStoreSettings.from_block({**MONGO, "port": "27017"}, key=WHERE).port == 27017

    def test_a_port_that_names_none_is_refused_at_build(self):
        """Behaviour change: it was handed to pymongo, which failed at the first call."""
        read = MongoStoreSettings.from_block({**MONGO, "port": "x"}, key=WHERE)
        with pytest.raises(ConfigurationError) as raised:
            read.with_env_overrides()
        assert raised.value.key == "archiver.settings.port"

    def test_a_missing_port_is_named_before_a_missing_login_as_it_always_was(self):
        block = {"host": "h", "name": "db", "collection": "c", "username": "u"}
        with pytest.raises(ConfigurationError, match="port is required for MongoDB archiver"):
            MongoStoreSettings.from_block(block, key=WHERE).with_env_overrides()

    def test_the_overrides_win_and_a_missing_host_is_refused_after_them(self, monkeypatch):
        monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_HOST", "mongo")
        monkeypatch.setenv("OSPREY_ARCHIVER_MONGODB_PORT", "27018")
        without_host = {key: value for key, value in MONGO.items() if key != "host"}
        read = MongoStoreSettings.from_block(without_host, key=WHERE).with_env_overrides()
        assert (read.host, read.port) == ("mongo", 27018)
        monkeypatch.delenv("OSPREY_ARCHIVER_MONGODB_HOST")
        with pytest.raises(ConfigurationError, match="host is required for MongoDB archiver"):
            MongoStoreSettings.from_block(without_host, key=WHERE).with_env_overrides()

    def test_mya_reads_its_text_settings(self):
        read = MyaArchiverSettings.from_block({"deployment": "", "myquery_server": "h"}, key=WHERE)
        assert (read.deployment, read.myquery_server, read.protocol) == ("ops", "h", None)

    def test_a_smoothing_window_that_is_not_a_number_is_refused_at_build(self):
        """Behaviour change: it failed every history read instead."""
        with pytest.raises(ConfigurationError) as raised:
            DoocsArchiverSettings.from_block({"avg_window": "20s"}, key=WHERE)
        assert raised.value.key == "archiver.settings.avg_window"

    def test_the_mock_rate_keeps_its_words(self):
        with pytest.raises(ConfigurationError, match=r"sample_rate_hz must be > 0 \(got 0\)"):
            MockArchiverSettings.from_block({"sample_rate_hz": 0}, key=WHERE)


class TestTheServices:
    def test_a_connection_block_that_breaks_the_shape_is_kept_with_its_reason(self):
        read = ChannelFinderSettings.from_block({"timeout": 5}, key="pv_directory.settings")
        assert isinstance(read.connection, Unreadable)
        assert read.connection.problem == (
            "`pv_directory.settings.timeout` is spelled `pv_directory.settings.timeout_s`"
        )

    def test_channelfinder_falls_back_to_its_default_properties(self):
        read = ChannelFinderSettings.from_block(
            {"url": "https://cf.example", "description_property": " ", "exact_properties": "x"},
            key="pv_directory.settings",
        )
        assert read.description_property == "recordDesc"
        assert read.exact_properties == frozenset({"pvStatus", "recordType"})

    @pytest.mark.parametrize(
        "block", [{}, {"bootstrap_servers": 5, "topic": "t"}, {"bootstrap_servers": "k:9092"}]
    )
    def test_a_feed_naming_no_broker_or_topic_reads_as_unset(self, block):
        read = PhoebusKafkaSettings.from_block(block, key="pv_directory.alarm_state.settings")
        assert not (read.bootstrap_servers and read.topic)

    def test_a_replay_file_that_is_not_a_path_is_unreadable(self):
        read = ReplayAlarmStateSettings.from_block(
            {"replay_file": 5}, key="pv_directory.alarm_state.settings"
        )
        assert isinstance(read.replay_file, Unreadable)


class TestTheLogin:
    def test_a_token_is_sent_as_a_bearer(self):
        credential = Credential(login=TokenLogin(token_env="T"), secret="s")
        assert credential.authorization_header() == "Bearer s"
        assert "'s'" not in repr(credential)

    def test_a_password_login_is_sent_as_basic(self):
        credential = Credential(login=PasswordLogin(username="u", password_env="P"), secret="p")
        assert credential.authorization_header() == "Basic dTpw"

    def test_a_variable_name_is_checked(self):
        with pytest.raises(ValueError):
            TokenLogin(token_env="1BAD")


class TestTheLoadedConfigSections:
    def test_every_section_the_package_reads_is_read_at_load(self):
        loaded = config_of(
            {
                "archiver": {"type": "epics_archiver", "settings": {"url": "u"}},
                "pv_directory": {"type": "channelfinder"},
                "simulation": {"tick_s": 2},
                "logging": {"show_full_paths": "true"},
                "agent_data": {"base_dir": "_agent_data"},
            },
            environ={},
        )
        assert loaded.archiver.type == "epics_archiver"
        assert loaded.pv_directory.present
        assert loaded.simulation.tick_s == 2.0
        assert loaded.logging.show_full_paths is True
        assert loaded.agent_data.base_dir == "_agent_data"

    def test_a_section_no_reader_can_use_does_not_stop_the_load(self):
        loaded = config_of(
            {
                "archiver": {"settings": "x"},
                "simulation": "fast",
                "logging": {"rich_tracebacks": 7},
            },
            environ={},
        )
        assert isinstance(loaded.archiver.settings, Unreadable)
        assert isinstance(loaded.simulation.unreadable, Unreadable)
        assert isinstance(loaded.logging.rich_tracebacks, Unreadable)


class TestTheLoggerReadsTheLoggingSection:
    def test_a_quoted_false_is_off(self):
        """Behaviour change: ``bool('false')`` read it as on."""
        with using_config(config_of({"logging": {"rich_tracebacks": "false"}}, environ={})):
            settings, problem = logger_module._read_handler_settings()
        assert settings.rich_tracebacks is False
        assert problem is None

    def test_an_unreadable_switch_renders_by_its_default_and_is_reported(self):
        with using_config(config_of({"logging": {"show_full_paths": "maybe"}}, environ={})):
            settings, problem = logger_module._read_handler_settings()
        assert settings.show_full_paths is False
        assert isinstance(problem, ConfigurationError)
        assert problem.key == "logging.show_full_paths"


@pytest.mark.parametrize(
    ("model", "block"),
    [
        (EpicsArchiverSettings, {"url": "https://a.example", "auth": {"token_env": "T"}}),
        (MongoStoreSettings, {**MONGO, "tls": {"ca_bundle": "/etc/ca.pem"}, "timeout_s": 5}),
        (MyaArchiverSettings, {"myquery_server": "h", "timeout_s": 9}),
        (DoocsArchiverSettings, {"avg_window": 20}),
        (MockArchiverSettings, {"sample_rate_hz": 2, "simulator_view": "/v"}),
        (ChannelFinderSettings, {"url": "https://cf.example", "exact_properties": ["a"]}),
        (PhoebusKafkaSettings, {"bootstrap_servers": ["k:1", "k:2"], "topic": "t"}),
    ],
)
def test_settings_read_again_from_their_dump_are_the_same(model, block):
    """A backend handed another model's settings reads them from its dump, to the same answer."""
    read = model.from_block(block, key=WHERE)
    assert model.from_block(read.model_dump(), key=WHERE) == read
