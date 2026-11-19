"""The sections of a rendered ``config.yml`` the package reads, each parsed once.

Every section is read by a classmethod ``read`` that **never raises**: loading a
config never refuses what using it refused, so a typo in ``archiver:`` does not
stop the controls server from starting. Where today's readers raise — the
factory, :func:`~osprey_connectors.connector_types.resolve_serving`,
:func:`~osprey_connectors.archiver_selection.resolve_archiver_settings` — they
still raise, at the same accessor, from what the parse kept.

Three rules hold for every section:

- **Written is not absent.** A leaf written as something no reader can use is
  :class:`Unreadable`, never ``None``: ``None`` is a key nobody wrote. The limits
  family needs the difference (an unreadable ``enabled`` read as unset would
  turn an attempt to switch checking *on* into unchecked writes), and so does the
  login-tier rung. A lone ``${VAR}`` that nothing expanded stays data: it is a
  value no reader can use, so it is :class:`Unreadable` where a leaf has a
  closed set of readable values.
- **Only a ``dict`` is a mapping**, as every reader of the YAML tree has always
  checked: a section or block that is anything else reads as written-but-
  unreadable (or, for a whole section, as empty).
- **The raw tree stays plain dicts.** A section keeps a deep copy of what it was
  read from in ``raw``, for the wire (``HostInit.control_system``) and for the
  framework-only keys no model names; a caller cannot change it through the
  section, and changing the caller's tree does not change the section.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Literal, Self, TypeAlias
from zoneinfo import ZoneInfo

from pydantic import ConfigDict, Field, ValidationError, ValidationInfo

from osprey_connectors._config_read import coerce_port, config_flag, is_unresolved_placeholder
from osprey_connectors.connector_types import (
    BUILTIN_CONNECTORS,
    IN_PROCESS,
    MOCK_ARCHIVER,
    SERVED,
    SERVING_LEAF,
    Serving,
)
from osprey_connectors.errors import ConfigurationError
from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, empty_mapping

__all__ = [
    "AgentDataSection",
    "ArchiverSection",
    "ConnectorBlock",
    "ConnectorSettings",
    "ControlSystemSection",
    "FacilityZone",
    "GatewayMode",
    "GatewayRole",
    "GatewayRow",
    "GatewaysDeclared",
    "LimitsBlock",
    "LimitsMode",
    "LoggingSection",
    "PvDirectorySection",
    "SelectedRole",
    "SimulationSection",
    "SystemSection",
    "TierWord",
    "Unreadable",
    "settings_key",
]

# -- closed sets the sections are written in ---------------------------------

#: One tier word: the three values of ``control_system.writes`` (and of each
#: ``control_system.login_tiers`` entry), narrowest first.
TierWord = Literal["never", "asked", "allowed"]

#: The two limits modes. ``exclusive``: the limits database is the complete
#: list of writable channels, and a channel with no record is refused.
#: ``optional``: a channel with a record is checked against it, and a channel
#: with no record is written with no limits.
LimitsMode = Literal["exclusive", "optional"]

#: A gateway role: the key of one endpoint a connector block derives. The two
#: :data:`SelectedRole` words key the ``gateways`` table; ``pva`` is the
#: ``pva_gateway`` a block routing PVAccess channels configures beside it.
GatewayRole = Literal["read_only", "write_access", "pva"]

#: The gateway roles a connector can select: the keys of the ``gateways`` table.
#: EPICS keeps one process-wide context, so exactly one of them is used; ``pva``
#: is never the selected role.
SelectedRole = Literal["read_only", "write_access"]

#: How a gateway reaches EPICS: ``name_server`` (``use_name_server: true``:
#: EPICS_CA_NAME_SERVERS / EPICS_PVA_NAME_SERVERS) or ``addr_list``
#: (``use_name_server: false``: EPICS_CA_ADDR_LIST / EPICS_PVA_ADDR_LIST).
GatewayMode = Literal["name_server", "addr_list"]

#: How a connector block spells its ``gateways`` table: an explicit
#: ``gateways: []`` (nothing to probe), no table anybody filled in, or rows.
GatewaysDeclared = Literal["none_declared", "missing", "rows"]

#: The two :data:`GatewaysDeclared` words a block with no rows is read as.
_NoRows: TypeAlias = Literal["none_declared", "missing"]

_SELECTED_ROLES: tuple[SelectedRole, ...] = ("read_only", "write_access")


class Unreadable(FrozenModel):
    """A key that was written, as something no reader can use: never read as absent.

    ``raw`` is what was written, a YAML ``null`` among the values (``None``
    here is a value, not "the key is absent"); a refusal quotes it.
    """

    #: The dotted key, e.g. ``control_system.connector.epics.limits_checking.mode``.
    key: str
    raw: object
    #: Why it cannot be read, in a reader's own words, when one said so.
    problem: str | None = None


def _dict(value: object) -> dict[object, object] | None:
    """*value* when it is a ``dict``, the one shape every reader takes as a mapping."""
    return value if isinstance(value, dict) else None


def _str_keyed(value: dict[object, object]) -> dict[str, object]:
    """The items of *value* whose key is a string, deep-copied."""
    return {key: copy.deepcopy(item) for key, item in value.items() if isinstance(key, str)}


# -- the settings a connector or service backend reads -------------------------

#: The validation-context key the block's dotted key travels under.
_BLOCK_KEY = "block_key"


class ConnectorSettings(FrozenModel):
    """One connector's or service backend's own block, read: the base of every settings model.

    A control-system connector's block (``control_system.connector.<type>``), an
    archive's (``archiver.settings``) and a service backend's
    (``pv_directory.settings`` and its kinds' ``settings``) are each read into a
    subclass of this, the ``settings_model`` its class declares, with a field
    for every key it reads. Every key is also kept as written, so a connector or
    backend that declares no model of its own reads its keys from
    :attr:`~pydantic.BaseModel.model_extra`, and a key the package reads
    elsewhere stays there too. Reading a block into this base refuses nothing.

    Read with :meth:`from_block` (or :meth:`read`), which names the block in a
    refusal.
    """

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_block(cls, block: Mapping[str, object], *, key: str) -> Self:
        """*block* read into this model, as the block under the dotted *key*.

        Raises:
            ConfigurationError: A key the model reads is written as something it
                cannot use: the refusal a validator raised, words and key as it
                spelled them, or one naming the first key at fault.
        """
        try:
            return cls.model_validate(dict(block), context={_BLOCK_KEY: key})
        except ValidationError as error:
            raise _settings_refusal(error, key) from None

    @classmethod
    def read(cls, settings: ConnectorSettings | Mapping[str, object], *, key: str) -> Self:
        """*settings* as this model: itself when it is one, else its keys read by :meth:`from_block`.

        Raises:
            ConfigurationError: As :meth:`from_block`.
        """
        if isinstance(settings, cls):
            return settings
        written = settings if isinstance(settings, Mapping) else settings.model_dump()
        return cls.from_block(written, key=key)


def settings_key(info: ValidationInfo[object]) -> str | None:
    """The dotted key of the block a settings validation reads, as :meth:`ConnectorSettings.from_block` gave it."""
    context = info.context
    key = context.get(_BLOCK_KEY) if isinstance(context, Mapping) else None
    return key if isinstance(key, str) else None


def _settings_refusal(error: ValidationError, key: str) -> ConfigurationError:
    """The :class:`ConfigurationError` one failed settings validation is refused with.

    A validator that raised one is the refusal, words and key as it spelled
    them; any other failure names the first key at fault under *key*.
    """
    first = error.errors()[0]
    raised = (first.get("ctx") or {}).get("error")
    if isinstance(raised, ConfigurationError):
        return raised
    at = ".".join([key, *(str(part) for part in first["loc"])])
    return ConfigurationError(f"{at}: {first['msg']}", key=at)


# -- control_system ----------------------------------------------------------


class LimitsBlock(FrozenModel):
    """One ``limits_checking`` block, deployment-wide or inside a connector block.

    Each leaf is tri-state plus unreadable: absent (``None``), readable, or
    :class:`Unreadable`. ``enabled`` is readable only as a literal ``true`` or
    ``false``, and ``mode`` only as one of :data:`LimitsMode`, matched exactly: a
    quoted ``'true'``, a ``1``, a bare ``enabled:`` and an unexpanded
    ``'${LIMITS_ON}'`` are unreadable.
    """

    enabled: bool | Unreadable | None = None
    mode: LimitsMode | Unreadable | None = None
    database_path: str | Unreadable | None = None

    @classmethod
    def read(cls, value: Mapping[object, object], key: str) -> Self:
        """The block *value* (a mapping) as written, under the dotted *key*."""
        enabled: bool | Unreadable | None = None
        mode: LimitsMode | Unreadable | None = None
        database_path: str | Unreadable | None = None
        if "enabled" in value:
            written = value["enabled"]
            enabled = (
                written
                if written is True or written is False
                else Unreadable(key=f"{key}.enabled", raw=written)
            )
        if "mode" in value:
            written = value["mode"]
            if written == "exclusive":
                mode = "exclusive"
            elif written == "optional":
                mode = "optional"
            else:
                mode = Unreadable(key=f"{key}.mode", raw=written)
        if "database_path" in value:
            written = value["database_path"]
            database_path = (
                written
                if isinstance(written, str)
                else Unreadable(key=f"{key}.database_path", raw=written)
            )
        return cls(enabled=enabled, mode=mode, database_path=database_path)


def _limits(block: Mapping[object, object], key: str) -> LimitsBlock | Unreadable | None:
    """The ``limits_checking`` leaf of *block*: absent, a block, or written as a non-mapping."""
    if "limits_checking" not in block:
        return None
    written = block["limits_checking"]
    mapping = _dict(written)
    if mapping is None:
        return Unreadable(key=f"{key}.limits_checking", raw=written)
    return LimitsBlock.read(mapping, f"{key}.limits_checking")


class GatewayRow(FrozenModel):
    """One gateway a connector block configures: where it is and how EPICS reaches it.

    One parse (:meth:`read`) for the derivation and for ``connect()``, so the
    row a roster names is the row a child connects through.
    """

    #: The gateway host, as written: a non-blank string, never stripped.
    address: str
    #: The port; ``None`` when the block names none and a default applies.
    port: int | None = None
    #: ``use_name_server``, read by :func:`~osprey_connectors.config.config_flag`.
    use_name_server: bool = False

    @property
    def mode(self) -> GatewayMode:
        """How EPICS reaches this gateway: a name server or an address list."""
        return "name_server" if self.use_name_server else "addr_list"

    @staticmethod
    def refusal(row: Unreadable) -> ConfigurationError:
        """The refusal of a gateway entry no connection can be made through."""
        return ConfigurationError(
            row.problem
            or f"{row.key} is {row.raw!r}; a gateway names a non-blank `address`, an "
            "integer `port` and a `use_name_server` of true or false",
            key=row.key,
        )

    @classmethod
    def read(cls, value: object, key: str) -> GatewayRow | Unreadable | None:
        """One gateway entry, or ``None`` for an entry ``connect()`` ignores.

        ``connect()`` guards its environment derivation with ``if gateway_config:``,
        so a missing, ``null`` or empty entry configures nothing. Any other value
        is a row or :class:`Unreadable`: a row needs a non-blank string
        ``address``, a ``port`` that is an integer or a numeric string (a bool
        is refused), and a ``use_name_server`` that spells true or false.
        """
        if isinstance(value, GatewayRow):
            return value
        if not value:
            return None
        mapping = _dict(value)
        if mapping is None:
            return Unreadable(key=key, raw=value)
        address = mapping.get("address")
        if not isinstance(address, str) or not address.strip():
            return Unreadable(key=key, raw=value)
        written_port = mapping.get("port")
        port = None if written_port is None else coerce_port(written_port)
        if written_port is not None and port is None:
            return Unreadable(key=key, raw=value)
        try:
            use_name_server = config_flag(
                mapping.get("use_name_server"), key=f"{key}.use_name_server"
            )
        except ValueError as error:
            return Unreadable(key=key, raw=value, problem=str(error))
        return cls(address=address, port=port, use_name_server=use_name_server)


def _gateways(
    block: Mapping[object, object], key: str
) -> ReadOnlyMapping[SelectedRole, GatewayRow | Unreadable] | _NoRows:
    """The ``gateways`` table: its rows by role, or how a block with none spells it.

    Only the explicit empty list is ``none_declared``; a table that is absent,
    ``null``, ``{}`` or anything but a non-empty mapping is ``missing``. A
    non-empty mapping keeps the roles it names with a value (a falsy entry
    configures nothing, as ``connect()`` reads it).
    """
    written = block.get("gateways")
    if written == []:
        return "none_declared"
    table = _dict(written)
    if not table:
        return "missing"
    rows: dict[SelectedRole, GatewayRow | Unreadable] = {}
    for role in _SELECTED_ROLES:
        row = GatewayRow.read(table.get(role), f"{key}.gateways.{role}")
        if row is not None:
            rows[role] = row
    return rows


def _pva_globs(value: object, key: str) -> tuple[str, ...] | Unreadable:
    """The PVA routing globs: a string or a list of them, blank entries dropped.

    Absent and empty are no routing. Anything that is not a string, list or
    tuple is unreadable.
    """
    if not value:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        return Unreadable(key=key, raw=value)
    return tuple(str(pattern).strip() for pattern in value if str(pattern).strip())


def _serving(block: Mapping[object, object], key: str) -> Serving | Unreadable | None:
    written = block.get(SERVING_LEAF)
    if written is None:
        return None
    if written == SERVED:
        return SERVED
    if written == IN_PROCESS:
        return IN_PROCESS
    return Unreadable(key=f"{key}.{SERVING_LEAF}", raw=written)


class ConnectorBlock(FrozenModel):
    """One ``control_system.connector.<type>`` block, as the section reads it.

    ``settings`` is the block as written, with its string keys: what the type's
    :attr:`~osprey_connectors.control_system.ControlSystemConnector.settings_model`
    validates when the factory builds the connector. A key that is not a string
    is named in ``unreadable_keys``, and the factory refuses the block.
    """

    #: The dotted key, ``control_system.connector.<type>``.
    key: str
    settings: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)
    unreadable_keys: FrozenTuple[str] = ()
    limits_checking: LimitsBlock | Unreadable | None = None
    gateways: ReadOnlyMapping[SelectedRole, GatewayRow | Unreadable] | _NoRows = "missing"
    pva_channels: FrozenTuple[str] | Unreadable = ()
    pva_gateway: GatewayRow | Unreadable | None = None
    #: Where the simulator runs; read only on the ``virtual_accelerator`` block.
    serving: Serving | Unreadable | None = None

    @property
    def declared_gateways(self) -> GatewaysDeclared:
        """How the block spells its ``gateways`` table."""
        return self.gateways if isinstance(self.gateways, str) else "rows"

    @property
    def written(self) -> bool:
        """Whether the block names anything at all: it configures a connector."""
        return bool(self.settings) or bool(self.unreadable_keys)

    @classmethod
    def read(cls, value: Mapping[object, object], key: str) -> Self:
        """The block *value* (a mapping) under the dotted *key*."""
        return cls(
            key=key,
            settings=_str_keyed(dict(value)),
            unreadable_keys=tuple(repr(name) for name in value if not isinstance(name, str)),
            limits_checking=_limits(value, key),
            gateways=_gateways(value, key),
            pva_channels=_pva_globs(value.get("pva_channels"), f"{key}.pva_channels"),
            pva_gateway=GatewayRow.read(value.get("pva_gateway"), f"{key}.pva_gateway"),
            serving=_serving(value, key),
        )


def _type_word(value: object) -> str | None:
    """``type`` as written: empty and ``None`` are absent, anything else is its text."""
    return str(value) if value else None


def _tier_word(value: object) -> TierWord | None:
    """*value* as a tier word, matched exactly, or ``None``."""
    if value == "never":
        return "never"
    if value == "asked":
        return "asked"
    if value == "allowed":
        return "allowed"
    return None


def _writes(value: object) -> TierWord:
    """``writes``: only ``asked`` and ``allowed`` arm; every other value is ``never``."""
    word = _tier_word(value)
    return "never" if word is None else word


def _login_tiers(
    section: Mapping[object, object],
) -> ReadOnlyMapping[str, TierWord] | Unreadable | None:
    """``login_tiers``: absent, the map, or unreadable as a whole.

    The map is one render's output, so one entry that cannot be read makes the
    whole map unreadable: a key that is not a non-empty string, or a value that
    is not one of :data:`TierWord`.
    """
    if "login_tiers" not in section:
        return None
    written = section["login_tiers"]
    table = _dict(written)
    if table is None:
        return Unreadable(key="control_system.login_tiers", raw=written)
    tiers: dict[str, TierWord] = {}
    for login, tier in table.items():
        word = _tier_word(tier) if isinstance(tier, str) else None
        if not (isinstance(login, str) and login and word is not None):
            return Unreadable(key="control_system.login_tiers", raw=written)
        tiers[login] = word
    return tiers


def _connectors(
    section: Mapping[object, object],
) -> ReadOnlyMapping[str, ConnectorBlock | Unreadable] | Unreadable:
    """The ``connector`` table by type; absent is empty, a non-mapping is unreadable.

    A key that is not a string names no type a resolver can ask for, so it is
    not kept here (the raw tree keeps it).
    """
    if "connector" not in section:
        return {}
    written = section["connector"]
    table = _dict(written)
    if table is None:
        return Unreadable(key="control_system.connector", raw=written)
    blocks: dict[str, ConnectorBlock | Unreadable] = {}
    for name, block in table.items():
        if not isinstance(name, str):
            continue
        key = f"control_system.connector.{name}"
        mapping = _dict(block)
        blocks[name] = (
            Unreadable(key=key, raw=block) if mapping is None else ConnectorBlock.read(mapping, key)
        )
    return blocks


class ControlSystemSection(FrozenModel):
    """The ``control_system:`` section, read once.

    What every resolver, posture reader and verdict takes. :meth:`read` never
    raises: a section that is not a mapping reads as an empty one, which every
    reader answers the fail-safe way (the simulator in process, tier ``never``).
    """

    #: ``control_system.type`` as written, ``None`` when empty or absent. Any other
    #: value is its text, so a typo reaches the factory's "Unknown … type" error.
    type: str | None = None
    #: The login tier, ``control_system.writes``: only the exact strings
    #: ``asked`` and ``allowed`` arm; everything else, absence included, is ``never``.
    writes: TierWord = "never"
    #: ``control_system.login_tiers``, the build-rendered login → tier map.
    login_tiers: ReadOnlyMapping[str, TierWord] | Unreadable | None = None
    #: The deployment-wide ``control_system.limits_checking`` block.
    limits_checking: LimitsBlock | Unreadable | None = None
    #: The ``control_system.connector`` table, by type.
    connectors: ReadOnlyMapping[str, ConnectorBlock | Unreadable] | Unreadable = Field(
        default_factory=empty_mapping
    )
    #: The section as written (its string keys, deep-copied): the wire, and the
    #: framework-only keys no field names.
    raw: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*, as a rendered ``config.yml`` holds it. Never raises.

        A section that is already read is returned as it is. A mapping that
        cannot be read at all (one that raises when it is asked for a key) is
        unreadable everywhere a reader must not read silence as permission:
        tier ``never``, the login rung and the limits block unreadable, and no
        connector table.
        """
        if isinstance(value, cls):
            return value
        section = _dict(value)
        if section is None:
            section = {}
        try:
            return cls(
                type=_type_word(section.get("type")),
                writes=_writes(section.get("writes")),
                login_tiers=_login_tiers(section),
                limits_checking=_limits(section, "control_system"),
                connectors=_connectors(section),
                raw=_str_keyed(section),
            )
        except Exception as error:  # a section that cannot be read arms nothing
            unreadable = Unreadable(key="control_system", raw=None, problem=repr(error))
            return cls(login_tiers=unreadable, limits_checking=unreadable, connectors=unreadable)

    def block(self, connector_type: object) -> ConnectorBlock | Unreadable | None:
        """The connector block *connector_type* keys, as the table holds it, or ``None``."""
        if isinstance(self.connectors, Unreadable) or not isinstance(connector_type, str):
            return None
        return self.connectors.get(connector_type)


# -- the other sections the package reads --------------------------------------


class ArchiverSection(FrozenModel):
    """The ``archiver:`` section, read once.

    ``settings_home`` is where the selected archiver's settings are written:
    ``settings`` (``archiver.settings``), ``own_name`` (``archiver.<type>``, for
    a short-named archiver), or ``None`` for neither. ``settings`` is that
    block, ``{}`` when absent or a bare ``settings:``, and :class:`Unreadable`
    when it cannot be the archiver's settings: both spellings are written, or
    the block is not a mapping. Its ``problem`` is the refusal
    :func:`~osprey_connectors.archiver_selection.resolve_archiver_settings`
    raises. ``problems`` names every block no archiver reads and that refusal,
    in the build lint's words.
    """

    type: str | None = None
    settings_home: Literal["settings", "own_name"] | None = None
    settings: ReadOnlyMapping[str, object] | Unreadable = Field(default_factory=empty_mapping)
    auto_bin_points: int | Unreadable | None = None
    problems: FrozenTuple[str] = ()
    raw: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*. Never raises."""
        if isinstance(value, cls):
            return value
        section = _dict(value) or {}
        selected = _type_word(section.get("type")) or MOCK_ARCHIVER
        homes = _archiver_homes(section, selected)
        home: Literal["settings", "own_name"] | None = None
        settings: ReadOnlyMapping[str, object] | Unreadable = {}
        if len(homes) > 1:
            second = homes[1][0]
            home = "settings"
            settings = Unreadable(
                key=f"archiver.{second}", raw=homes[1][1], problem=_two_archiver_homes(second)
            )
        elif homes:
            name, written = homes[0]
            home = "settings" if name == _ARCHIVER_SETTINGS_LEAF else "own_name"
            mapping = _dict(written)
            if mapping is not None:
                settings = _str_keyed(mapping)
            elif written is not None:
                settings = Unreadable(
                    key=f"archiver.{name}",
                    raw=written,
                    problem=_not_an_archiver_block(name, written),
                )
        points: int | Unreadable | None = None
        if section.get("auto_bin_points") is not None:
            written_points = section["auto_bin_points"]
            points = (
                written_points
                if isinstance(written_points, int) and not isinstance(written_points, bool)
                else Unreadable(key="archiver.auto_bin_points", raw=written_points)
            )
        return cls(
            type=_type_word(section.get("type")),
            settings_home=home,
            settings=settings,
            auto_bin_points=points,
            problems=tuple(_archiver_problems(section, selected, homes)),
            raw=_str_keyed(section),
        )


#: The block under ``archiver:`` a connector's settings are read from.
_ARCHIVER_SETTINGS_LEAF = "settings"

#: Keys of the ``archiver:`` section that are not a connector's settings block.
_ARCHIVER_SECTION_LEAVES = ("type", _ARCHIVER_SETTINGS_LEAF, "auto_bin_points")

#: Every built-in archiver's name, in registration order.
_ARCHIVER_TYPES = tuple(row.name for row in BUILTIN_CONNECTORS if row.kind == "archiver")


def _archiver_homes(section: Mapping[object, object], selected: str) -> list[tuple[str, object]]:
    """Every block in *section* that configures the *selected* archiver, in order.

    ``archiver.settings``, then a short-named archiver's own-name block. A
    dotted type has no own-name block: a profile splits keys on every dot.
    """
    homes: list[tuple[str, object]] = []
    if _ARCHIVER_SETTINGS_LEAF in section:
        homes.append((_ARCHIVER_SETTINGS_LEAF, section[_ARCHIVER_SETTINGS_LEAF]))
    own = None if "." in selected else selected
    if own is not None and own != _ARCHIVER_SETTINGS_LEAF and own in section:
        homes.append((own, section[own]))
    return homes


def _two_archiver_homes(own: str) -> str:
    return (
        f"`archiver.settings` and `archiver.{own}` both configure {own}; keep one "
        "— `archiver.settings` is the spelling for every archiver"
    )


def _not_an_archiver_block(key: str, value: object) -> str:
    return (
        f"`archiver.{key}` is {value!r}, not a block; it holds the selected "
        "archiver's settings as a mapping"
    )


def _archiver_problems(
    section: Mapping[object, object], selected: str, homes: list[tuple[str, object]]
) -> list[str]:
    """The build lint's lines for *section*, in section order (see :class:`ArchiverSection`)."""
    own = None if "." in selected else selected
    known = {*_ARCHIVER_SECTION_LEAVES, *_ARCHIVER_TYPES, *([own] if own else [])}
    problems = [
        f"`{key}:` under `archiver:` is a block no archiver reads; the selected "
        f"archiver, {selected}, takes its settings from `archiver.settings`"
        for key, value in section.items()
        if isinstance(value, dict) and key not in known
    ]
    if len(homes) > 1:
        problems.append(_two_archiver_homes(homes[1][0]))
    problems.extend(
        _not_an_archiver_block(key, value)
        for key, value in homes
        if value is not None and not isinstance(value, dict)
    )
    return problems


class PvDirectorySection(FrozenModel):
    """The ``pv_directory:`` section, read once: one block per service kind.

    Each of ``directory``, ``alarm_log``, ``alarm_state`` and ``put_log`` is the
    block as written (the ``directory`` block is the section itself, which
    carries its ``type`` and ``settings`` at the top), empty when absent or not
    a mapping.
    """

    present: bool = False
    directory: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)
    alarm_log: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)
    alarm_state: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)
    put_log: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*. Never raises."""
        section = _dict(value)
        if section is None:
            return cls()

        def kind(name: str) -> dict[str, object]:
            return _str_keyed(_dict(section.get(name)) or {})

        return cls(
            present=True,
            directory=_str_keyed(section),
            alarm_log=kind("alarm_log"),
            alarm_state=kind("alarm_state"),
            put_log=kind("put_log"),
        )


def _positive_number(value: object, key: str) -> float | Unreadable | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return Unreadable(key=key, raw=value)
    return float(value)


def _text(value: object, key: str) -> str | Unreadable | None:
    if value is None:
        return None
    return value if isinstance(value, str) else Unreadable(key=key, raw=value)


class SimulationSection(FrozenModel):
    """The ``simulation:`` section, read once.

    ``tick_s`` and ``fast_tick_s`` are numbers of seconds greater than zero;
    ``state_dir`` a path string. Each is ``None`` when absent. ``unreadable``
    holds the section itself when it is written as something other than a
    mapping.
    """

    tick_s: float | Unreadable | None = None
    fast_tick_s: float | Unreadable | None = None
    state_dir: str | Unreadable | None = None
    unreadable: Unreadable | None = None

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*. Never raises."""
        if not value:
            return cls()
        section = _dict(value)
        if section is None:
            return cls(unreadable=Unreadable(key="simulation", raw=value))
        return cls(
            tick_s=_positive_number(section.get("tick_s"), "simulation.tick_s"),
            fast_tick_s=_positive_number(section.get("fast_tick_s"), "simulation.fast_tick_s"),
            state_dir=_text(section.get("state_dir"), "simulation.state_dir"),
        )


class LoggingSection(FrozenModel):
    """The ``logging:`` section, read once: its three switches.

    Each is read by :func:`~osprey_connectors.config.config_flag`, so a quoted
    ``'false'`` is off; a value that spells neither is :class:`Unreadable`.
    """

    rich_tracebacks: bool | Unreadable = True
    show_traceback_locals: bool | Unreadable = False
    show_full_paths: bool | Unreadable = False

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*. Never raises."""
        section = _dict(value) or {}

        def flag(name: str, default: bool) -> bool | Unreadable:
            if section.get(name) is None:
                return default
            try:
                return config_flag(section[name], key=f"logging.{name}")
            except ValueError:
                return Unreadable(key=f"logging.{name}", raw=section[name])

        return cls(
            rich_tracebacks=flag("rich_tracebacks", True),
            show_traceback_locals=flag("show_traceback_locals", False),
            show_full_paths=flag("show_full_paths", False),
        )


#: Where a facility zone came from: written in ``system.timezone``, the UTC a
#: config that writes none gets, or the UTC a written value that names no zone
#: falls back to.
_FacilityZoneSource = Literal["configured", "default", "fallback"]

#: The zone a config that names none, or names one that is not a zone, runs on.
_DEFAULT_ZONE = "UTC"


class FacilityZone(FrozenModel):
    """The facility clock a config names, read when it is loaded.

    Lenient by design: a written value that names no zone (a typo, a number, an
    unset ``${VAR}``) is recorded as a fallback to UTC with the ``problem`` it
    has, never refused, so a hand-edited render cannot stop every config load —
    the write path's included. ``osprey build`` still refuses a bad zone when it
    renders one.

    Attributes:
        name: The IANA zone name, e.g. ``America/Los_Angeles``; ``UTC`` for a
            default or a fallback.
        source: ``"configured"`` when ``system.timezone`` names a zone,
            ``"default"`` when the config names none, ``"fallback"`` when it
            names something that is not a zone.
        problem: Why a fallback fell back; ``None`` otherwise.
    """

    name: str
    source: _FacilityZoneSource
    problem: str | None = None

    @property
    def zone(self) -> ZoneInfo:
        """The zone itself."""
        return ZoneInfo(self.name)


class SystemSection(FrozenModel):
    """The ``system:`` section the package reads, read once: the facility clock."""

    #: The zone ``system.timezone`` names; UTC, with its source, when it names none.
    timezone: FacilityZone = Field(
        default_factory=lambda: FacilityZone(name=_DEFAULT_ZONE, source="default")
    )

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*. Never raises.

        A section that is not a mapping, or that writes no ``timezone``, is the
        default UTC. A written value that names no zone (a typo, a number, a
        null, an unset ``${VAR}``) falls back to UTC with the problem it has.
        """
        section = _dict(value)
        if section is None or "timezone" not in section:
            return cls()
        written = section["timezone"]
        if not isinstance(written, str):
            problem = f"system.timezone must be a string, got {type(written).__name__}"
        else:
            try:
                ZoneInfo(written)
            # ZoneInfoNotFoundError is a KeyError; a malformed key is a ValueError,
            # and a key naming a directory of the zone database an OSError.
            except (KeyError, ValueError, OSError) as exc:
                problem = f"system.timezone {written!r} names no zone ({type(exc).__name__}: {exc})"
                if is_unresolved_placeholder(written):
                    problem += " (an unset environment variable)"
            else:
                return cls(timezone=FacilityZone(name=written, source="configured"))
        return cls(timezone=FacilityZone(name=_DEFAULT_ZONE, source="fallback", problem=problem))


class AgentDataSection(FrozenModel):
    """The ``agent_data:`` section, read once: ``base_dir``, ``None`` when unset or empty."""

    base_dir: str | None = None

    @classmethod
    def read(cls, value: object) -> Self:
        """The section *value*. Never raises."""
        section = value if isinstance(value, Mapping) else {}
        base_dir = section.get("base_dir")
        return cls(base_dir=str(base_dir) if base_dir else None)
