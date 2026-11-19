"""The one block an archiver's settings are read from, on every route that selects it."""

import pytest

from osprey_connectors import archiver_selection
from osprey_connectors.config_sections import ArchiverSection
from osprey_connectors.connector_types import TARGET_LIVE
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors_tests._harness import isolated_connector_registries
from osprey_connectors_tests._mock_dynamic_connector import RecordingArchiver
from osprey_connectors_tests._service_table import table_of

RECORDING = "osprey_connectors_tests._mock_dynamic_connector.RecordingArchiver"


def resolve_archiver_settings(section: object) -> dict[str, object]:
    return archiver_selection.resolve_archiver_settings(ArchiverSection.read(section))


def archiver_settings_key(section: object) -> str:
    return archiver_selection.archiver_settings_key(ArchiverSection.read(section))


def section_problems(section: object) -> list[str]:
    """The build's lint: the section's problems, as it reads them."""
    return list(ArchiverSection.read(section).problems)


@pytest.fixture(autouse=True)
def clean_factory():
    """Run each test against empty factory registries and a fresh recording.

    Snapshot/restore brackets the clear so registrations made elsewhere in the
    process survive this module's teardown.
    """
    RecordingArchiver.settings = None
    with isolated_connector_registries(clear=True):
        yield


async def _built(section: dict) -> None:
    """Build the live archive a config with this ``archiver:`` *section* names."""
    spec = table_of({"archiver": section}).archive[TARGET_LIVE]
    assert spec is not None
    await ConnectorFactory.create_service(spec)


@pytest.mark.asyncio
async def test_a_dotted_path_archiver_is_built_with_its_settings_block():
    # A url makes the facility's archive serve the live machine.
    await _built({"type": RECORDING, "settings": {"url": "https://history.example.org"}})
    assert RecordingArchiver.settings == {"url": "https://history.example.org"}


@pytest.mark.asyncio
async def test_a_block_keyed_by_the_dotted_path_is_not_read():
    # Unread, the block names no address, so no target is served at all.
    table = table_of(
        {"archiver": {"type": RECORDING, RECORDING: {"url": "https://history.example.org"}}}
    )
    assert table.archive[TARGET_LIVE] is None
    assert RecordingArchiver.settings is None


def test_a_builtin_reads_the_settings_block():
    section = {"type": "epics_archiver", "settings": {"url": "u"}}
    assert resolve_archiver_settings(section) == {"url": "u"}


def test_a_builtin_still_reads_its_own_name_block():
    section = {"type": "epics_archiver", "epics_archiver": {"url": "u"}}
    assert resolve_archiver_settings(section) == {"url": "u"}


def test_a_registered_short_name_reads_its_own_name_block():
    section = {"type": "site_archiver", "site_archiver": {"x": 1}}
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
        assert section_problems({"type": "a.b.C", "a.b.C": {"s": 1}}) == [
            "`a.b.C:` under `archiver:` is a block no archiver reads; the selected "
            "archiver, a.b.C, takes its settings from `archiver.settings`"
        ]

    def test_a_block_keyed_by_the_last_path_segment_is_named(self):
        (error,) = section_problems({"type": "a.b.C", "C": {"s": 1}})
        assert "`C:`" in error

    def test_a_misspelled_builtin_block_is_named(self):
        (error,) = section_problems({"type": "epics_archiver", "epcis_archiver": {"url": "u"}})
        assert "`epcis_archiver:`" in error

    def test_both_spellings_at_once_are_named(self):
        section = {"type": "epics_archiver", "settings": {}, "epics_archiver": {"url": "u"}}
        with pytest.raises(ValueError) as refusal:
            resolve_archiver_settings(section)
        assert section_problems(section) == [str(refusal.value)]

    def test_a_settings_value_that_is_not_a_block_is_named(self):
        section = {"type": "a.b.C", "settings": "x"}
        with pytest.raises(ValueError) as refusal:
            resolve_archiver_settings(section)
        assert section_problems(section) == [str(refusal.value)]

    @pytest.mark.parametrize(
        "section",
        [
            {"type": "a.b.C", "settings": {"server": "s"}},
            {"type": "epics_archiver", "epics_archiver": {"url": "u"}},
            {"type": "site_archiver", "site_archiver": {"x": 1}},
            {"type": "a.b.C", "settings": {}, "epics_archiver": {"url": "u"}},
            {"type": "mock_archiver", "auto_bin_points": 10000},
            {},
            None,
        ],
    )
    def test_what_a_reader_takes_passes(self, section):
        assert section_problems(section) == []

    def test_a_scalar_is_not_a_block(self):
        assert section_problems({"type": "a.b.C", "url": "x"}) == []


def test_the_known_archiver_names_are_the_factory_builtins():
    """A built-in's own name block is read, so every built-in has to be a known name."""
    from osprey_connectors import config_sections
    from osprey_connectors.factory import _BUILTIN_ARCHIVERS

    assert sorted(config_sections._ARCHIVER_TYPES) == sorted(_BUILTIN_ARCHIVERS)
