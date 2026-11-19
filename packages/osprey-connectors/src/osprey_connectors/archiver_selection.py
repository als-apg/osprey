"""Which archiver an ``archiver:`` section selects, and which block configures it.

:func:`resolve_archiver_type` is the archiver half of what a config selects,
beside :func:`~osprey_connectors.connector_types.resolve_control_system_type`,
so the framework's archiver-pairing rule and the factory read one answer.
:func:`resolve_archiver_settings` is the same arrangement for the other half of
an archiver selection: which block the archive is built from. The build
refuses what it would refuse, from the same section
(:attr:`~osprey_connectors.config_sections.ArchiverSection.problems`).

Each takes the section read
(:meth:`~osprey_connectors.config_sections.ArchiverSection.read`): the loaded
config's ``archiver``, or a mapping a caller holds, read.
"""

from osprey_connectors.config_sections import ArchiverSection, Unreadable
from osprey_connectors.connector_types import MOCK_ARCHIVER
from osprey_connectors.errors import ConfigurationError

__all__ = [
    "archiver_settings_key",
    "resolve_archiver_settings",
    "resolve_archiver_type",
]


def resolve_archiver_type(section: ArchiverSection) -> str:
    """The archiver an ``archiver:`` config section actually selects.

    Absent means the mock: a config that says nothing about its archiver is a
    config that gets the synthesizing one, which is the fact the honesty rule is
    really about.
    """
    return section.type or MOCK_ARCHIVER


def archiver_settings_key(section: ArchiverSection) -> str:
    """The dotted key of the block that configures *section*'s archiver.

    ``archiver.<type>`` when a short-named archiver is configured by its own-name
    block alone, else ``archiver.settings`` — so a message about a setting names
    the line an operator actually wrote.
    """
    if section.settings_home == "own_name":
        return f"archiver.{resolve_archiver_type(section)}"
    return "archiver.settings"


def resolve_archiver_settings(section: ArchiverSection) -> dict[str, object]:
    """The settings block the selected archiver is built from.

    ``archiver.settings`` for every archiver; for a short-named one, its own-name
    block when ``settings`` is absent. Absent, or a bare ``settings:``, is ``{}``.

    Raises:
        ConfigurationError: when both spellings are present, or the block is not a
            mapping. :attr:`~osprey_connectors.config_sections.ArchiverSection.problems`
            names the same configs at build time, with the same words.
    """
    settings = section.settings
    if isinstance(settings, Unreadable):
        raise ConfigurationError(settings.problem or settings.key, key=settings.key)
    return dict(settings)
