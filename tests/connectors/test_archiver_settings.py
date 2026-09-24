"""The one block an archiver's settings are read from, on every route that selects it."""

import pytest

from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries
from osprey_connectors.types import (
    archiver_settings_errors,
    archiver_settings_key,
    resolve_archiver_settings,
)
from tests.connectors._mock_dynamic_connector import RecordingArchiver

RECORDING = "tests.connectors._mock_dynamic_connector.RecordingArchiver"


@pytest.fixture(autouse=True)
def clean_factory():
    """Run each test against empty factory registries and a fresh recording.

    Snapshot/restore brackets the clear so registrations made elsewhere in the
    process survive this module's teardown.
    """
    RecordingArchiver.settings = None
    with isolated_connector_registries(clear=True):
        yield


@pytest.mark.asyncio
async def test_a_dotted_path_archiver_is_connected_with_its_settings_block():
    await ConnectorFactory.create_archiver_connector(
        {"type": RECORDING, "settings": {"server": "history.example.org"}}
    )
    assert RecordingArchiver.settings == {"server": "history.example.org"}


@pytest.mark.asyncio
async def test_a_block_keyed_by_the_dotted_path_is_not_read():
    await ConnectorFactory.create_archiver_connector(
        {"type": RECORDING, RECORDING: {"server": "history.example.org"}}
    )
    assert RecordingArchiver.settings == {}


def test_a_builtin_reads_the_settings_block():
    section = {"type": "epics_archiver", "settings": {"url": "u"}}
    assert resolve_archiver_settings(section) == {"url": "u"}


def test_a_builtin_still_reads_its_own_name_block():
    section = {"type": "epics_archiver", "epics_archiver": {"url": "u"}}
    assert resolve_archiver_settings(section) == {"url": "u"}


def test_a_registered_short_name_reads_its_own_name_block():
    section = {"type": "moat_archiver", "moat_archiver": {"x": 1}}
    assert resolve_archiver_settings(section) == {"x": 1}


def test_an_unselected_builtin_block_is_not_read():
    section = {"type": "mock_archiver", "mongodb_archiver": {"host": "h"}}
    assert resolve_archiver_settings(section) == {}


@pytest.mark.parametrize(
    "section",
    [None, {}, {"type": "mock_archiver"}, {"type": "a.b.C", "settings": None}],
)
def test_no_settings_is_an_empty_block(section):
    assert resolve_archiver_settings(section) == {}


def test_both_spellings_at_once_are_refused():
    section = {"type": "epics_archiver", "settings": {}, "epics_archiver": {"url": "u"}}
    with pytest.raises(ValueError) as refusal:
        resolve_archiver_settings(section)
    assert "`archiver.settings`" in str(refusal.value)
    assert "`archiver.epics_archiver`" in str(refusal.value)


def test_a_settings_value_that_is_not_a_block_is_refused():
    with pytest.raises(ValueError, match="not a block"):
        resolve_archiver_settings({"type": "a.b.C", "settings": "x"})


def test_the_settings_key_names_the_block_that_answered():
    assert archiver_settings_key({"type": "a.b.C", "settings": {}}) == "archiver.settings"
    assert (
        archiver_settings_key({"type": "epics_archiver", "epics_archiver": {"url": "u"}})
        == "archiver.epics_archiver"
    )
    assert archiver_settings_key({}) == "archiver.settings"


class TestArchiverSettingsErrors:
    """The build's lint over a rendered ``archiver:`` section."""

    def test_a_block_keyed_by_the_dotted_path_is_named(self):
        assert archiver_settings_errors({"type": "a.b.C", "a.b.C": {"s": 1}}) == [
            "`a.b.C:` under `archiver:` is a block no archiver reads; the selected "
            "archiver, a.b.C, takes its settings from `archiver.settings`"
        ]

    def test_a_block_keyed_by_the_last_path_segment_is_named(self):
        (error,) = archiver_settings_errors({"type": "a.b.C", "C": {"s": 1}})
        assert "`C:`" in error

    def test_a_misspelled_builtin_block_is_named(self):
        (error,) = archiver_settings_errors(
            {"type": "epics_archiver", "epcis_archiver": {"url": "u"}}
        )
        assert "`epcis_archiver:`" in error

    def test_both_spellings_at_once_are_named(self):
        section = {"type": "epics_archiver", "settings": {}, "epics_archiver": {"url": "u"}}
        with pytest.raises(ValueError) as refusal:
            resolve_archiver_settings(section)
        assert archiver_settings_errors(section) == [str(refusal.value)]

    def test_a_settings_value_that_is_not_a_block_is_named(self):
        section = {"type": "a.b.C", "settings": "x"}
        with pytest.raises(ValueError) as refusal:
            resolve_archiver_settings(section)
        assert archiver_settings_errors(section) == [str(refusal.value)]

    @pytest.mark.parametrize(
        "section",
        [
            {"type": "a.b.C", "settings": {"server": "s"}},
            {"type": "epics_archiver", "epics_archiver": {"url": "u"}},
            {"type": "moat_archiver", "moat_archiver": {"x": 1}},
            {"type": "a.b.C", "settings": {}, "epics_archiver": {"url": "u"}},
            {"type": "mock_archiver", "auto_bin_points": 10000},
            {},
            None,
        ],
    )
    def test_what_a_reader_takes_passes(self, section):
        assert archiver_settings_errors(section) == []

    def test_a_scalar_is_not_a_block(self):
        assert archiver_settings_errors({"type": "a.b.C", "url": "x"}) == []
