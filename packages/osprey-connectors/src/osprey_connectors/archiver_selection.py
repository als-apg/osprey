"""Which archiver an ``archiver:`` section selects, and which block configures it.

:func:`resolve_archiver_type` is the archiver half of what a config selects,
beside :func:`~osprey_connectors.connector_types.resolve_control_system_type`,
so the framework's archiver-pairing rule and the factory read one answer.
:func:`resolve_archiver_settings` is the same arrangement for the other half of
an archiver selection: which block its ``connect()`` is handed. The build
refuses what it would refuse, from the same helpers.
"""

from osprey_connectors.connector_types import BUILTIN_CONNECTORS, MOCK_ARCHIVER, _resolve_type
from osprey_connectors.errors import ConfigurationError

__all__ = [
    "archiver_settings_errors",
    "archiver_settings_key",
    "resolve_archiver_settings",
    "resolve_archiver_type",
]

#: Every built-in archiver's name, in registration order.
_CLI_ARCHIVER_TYPES = [row.name for row in BUILTIN_CONNECTORS if row.kind == "archiver"]


def resolve_archiver_type(section: object) -> str:
    """The archiver an ``archiver:`` config section actually selects.

    Absent means the mock: a config that says nothing about its archiver is a
    config that gets the synthesizing one, which is the fact the honesty rule is
    really about.
    """
    return _resolve_type(section, MOCK_ARCHIVER)


#: The one block under ``archiver:`` a connector's settings are read from,
#: whichever route selected the connector. It has no dot in it, so a profile's
#: ``archiver.settings.<leaf>`` renders to exactly the block the factory reads.
_ARCHIVER_SETTINGS_LEAF = "settings"

#: Keys of the ``archiver:`` section that are not a connector's settings block:
#: the selector, the settings block itself, and archiver_read's point budget.
_ARCHIVER_SECTION_LEAVES = ("type", _ARCHIVER_SETTINGS_LEAF, "auto_bin_points")


def _own_name_block(connector_type: str) -> str | None:
    """The older spelling of a short-named archiver's settings, or ``None``.

    A built-in, or a connector registered under a short name, may still keep
    its settings in a block carrying its own name (``archiver.epics_archiver``).
    A dotted module path has no such block: a profile splits keys on every dot,
    so a block keyed by one could never have been written from ``config:``.
    """
    return None if "." in connector_type else connector_type


def _archiver_settings_homes(section: object) -> list[tuple[str, object]]:
    """Every block in *section* that configures the selected archiver, in order."""
    if not isinstance(section, dict):
        return []
    homes: list[tuple[str, object]] = []
    if _ARCHIVER_SETTINGS_LEAF in section:
        homes.append((_ARCHIVER_SETTINGS_LEAF, section[_ARCHIVER_SETTINGS_LEAF]))
    own = _own_name_block(resolve_archiver_type(section))
    if own is not None and own != _ARCHIVER_SETTINGS_LEAF and own in section:
        homes.append((own, section[own]))
    return homes


def _two_homes(own: str) -> str:
    return (
        f"`archiver.settings` and `archiver.{own}` both configure {own}; keep one "
        f"— `archiver.settings` is the spelling for every archiver"
    )


def _not_a_block(key: str, value: object) -> str:
    return (
        f"`archiver.{key}` is {value!r}, not a block; it holds the selected "
        f"archiver's settings as a mapping"
    )


def archiver_settings_key(section: object) -> str:
    """The dotted key of the block that configures *section*'s archiver.

    ``archiver.<type>`` when a short-named archiver is configured by its own-name
    block alone, else ``archiver.settings`` — so a message about a setting names
    the line an operator actually wrote.
    """
    homes = _archiver_settings_homes(section)
    home = homes[0][0] if len(homes) == 1 else _ARCHIVER_SETTINGS_LEAF
    return f"archiver.{home}"


def resolve_archiver_settings(section: object) -> dict[str, object]:
    """The settings block the selected archiver's ``connect()`` is handed.

    ``archiver.settings`` for every archiver; for a short-named one, its own-name
    block when ``settings`` is absent. Absent, or a bare ``settings:``, is ``{}``.

    Raises:
        ConfigurationError: when both spellings are present, or the block is not a
            mapping. :func:`archiver_settings_errors` refuses the same configs
            at build time, with the same words.
    """
    homes = _archiver_settings_homes(section)
    if len(homes) > 1:
        raise ConfigurationError(_two_homes(homes[1][0]), key=f"archiver.{homes[1][0]}")
    if not homes or homes[0][1] is None:
        return {}
    key, value = homes[0]
    if not isinstance(value, dict):
        raise ConfigurationError(_not_a_block(key, value), key=f"archiver.{key}")
    return dict(value)


def archiver_settings_errors(section: object) -> list[str]:
    """Every block under a rendered ``archiver:`` section that no archiver reads, named.

    A lint, like :func:`~osprey_connectors.limits_posture.incomplete_limits_blocks`,
    and read by the build over the config a deployment runs. It reports, in
    section order: each mapping under a key no reader takes (anything but
    :data:`_ARCHIVER_SECTION_LEAVES`, a built-in archiver's name, or the selected
    type's own name when it has no dot); then the two refusals :func:`resolve_archiver_settings` raises, in its words. A
    scalar is not a block and is not reported. Never raises.
    """
    if not isinstance(section, dict):
        return []
    selected = resolve_archiver_type(section)
    own = _own_name_block(selected)
    known = {*_ARCHIVER_SECTION_LEAVES, *_CLI_ARCHIVER_TYPES, *([own] if own else [])}
    errors = [
        f"`{key}:` under `archiver:` is a block no archiver reads; the selected "
        f"archiver, {selected}, takes its settings from `archiver.settings`"
        for key, value in section.items()
        if isinstance(value, dict) and key not in known
    ]
    homes = _archiver_settings_homes(section)
    if len(homes) > 1:
        errors.append(_two_homes(homes[1][0]))
    errors.extend(
        _not_a_block(key, value)
        for key, value in homes
        if value is not None and not isinstance(value, dict)
    )
    return errors
