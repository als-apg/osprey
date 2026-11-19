"""Connector type names, one row per built-in type, and the fallbacks a factory applies to them.

Single source of truth for built-in connector type name strings.
Custom connectors use dotted module paths (e.g., 'mypackage.SiteConnector')
and don't need constants here.

What a built-in type *is* — the wire it speaks, the kind of machine it serves,
the target it is the baseline of, whether its history is invented, which
commands offer it, its default port — is one :class:`BuiltinConnector` row in
:data:`BUILTIN_CONNECTORS`. Every list and table below that answers one of
those questions is derived from the rows, so adding a type is one row.

The resolvers are here rather than in :mod:`osprey_connectors.factory` so that
a guard deciding what a deployment *will build* can share them with the factory
that builds it. A guard which re-implements "what does this config select" is a
guard that can disagree with the answer, and the disagreement is a bypass
rather than a discrepancy — see the framework's ``osprey.deployment.honesty``.

The control-target vocabulary (:data:`ControlTarget`, :data:`MachineKind`)
lives here too, because a built-in row is written in it: it names the target it
is the baseline of and the kind of machine it serves. Which machine a target
means in one deployment is :mod:`osprey_connectors.targets`'s answer.
"""

from typing import TYPE_CHECKING, Final, Literal, Self, get_args

from pydantic import model_validator
from typing_extensions import TypeIs

from osprey_connectors.errors import ConfigurationError
from osprey_connectors.model import FrozenModel

if TYPE_CHECKING:
    from osprey_connectors.config_sections import ConnectorBlock, ControlSystemSection

__all__ = [
    "BUILTIN_CONNECTORS",
    "BuiltinConnector",
    "CLI_CONTROL_SYSTEM_TYPES",
    "CONTROL_TARGETS",
    "ControlSystemTypeName",
    "ControlTarget",
    "DEFAULT_VA_PORT",
    "DOOCS",
    "DOOCS_ARCHIVER",
    "EPICS",
    "EPICS_ARCHIVER",
    "INVENTED_HISTORY_TYPES",
    "IN_PROCESS",
    "LIVE_STANDIN",
    "MOCK_ARCHIVER",
    "MONGODB_ARCHIVER",
    "MYA_ARCHIVER",
    "MachineKind",
    "RETIRED_CONTROL_SYSTEM_TYPES",
    "SERVED",
    "SERVING_KEY",
    "SERVING_LEAF",
    "SERVING_MODES",
    "SET_CONTROL_SYSTEM_TYPES",
    "SIMULATED_TYPES",
    "STANDIN_TYPES",
    "Serving",
    "TANGO",
    "TARGET_LIVE",
    "TARGET_STANDIN",
    "TARGET_VA",
    "TRANSPORT_CA",
    "TRANSPORT_IN_PROCESS",
    "Transport",
    "UNSET_TYPE_WARNING",
    "VIRTUAL_ACCELERATOR",
    "connector_transport",
    "is_control_target",
    "is_simulated",
    "resolve_control_system_type",
    "resolve_serving",
    "retired_type_message",
    "speaks_channel_access",
    "talks_to_network",
]

# -- Control system connector types (have implementations) --
#: Every built-in control-system type name, as a type.
ControlSystemTypeName = Literal["epics", "virtual_accelerator", "live_standin", "doocs", "tango"]

EPICS = "epics"
VIRTUAL_ACCELERATOR = "virtual_accelerator"
DOOCS = "doocs"
TANGO = "tango"

#: The live stand-in: a facility-shaped soft IOC a deployment runs for itself.
#: A connector type of its own — served by the EPICS connector, but keyed apart
#: from ``epics`` so that the facility's authored ``epics`` block stays the one
#: thing ``live`` can mean, and the stand-in gets its own
#: ``control_system.connector.live_standin`` block to be configured from.
LIVE_STANDIN = "live_standin"

# -- Archiver connector types --
MOCK_ARCHIVER = "mock_archiver"
EPICS_ARCHIVER = "epics_archiver"
MONGODB_ARCHIVER = "mongodb_archiver"
DOOCS_ARCHIVER = "doocs_archiver"
MYA_ARCHIVER = "mya_archiver"

# -- Where the simulator runs --
#: The leaf under ``control_system.connector.virtual_accelerator`` that says
#: where the simulator runs: ``served`` from its container over Channel Access,
#: or ``in_process`` inside the process that asks, with no network at all.
SERVING_LEAF = "serving"
#: The two venues the simulator runs in: one word of :data:`SERVING_LEAF`.
Serving = Literal["served", "in_process"]
SERVED: Final = "served"
IN_PROCESS: Final = "in_process"
#: Every :data:`Serving` word, ``served`` first.
SERVING_MODES: tuple[Serving, ...] = get_args(Serving)
SERVING_KEY = f"control_system.connector.{VIRTUAL_ACCELERATOR}.{SERVING_LEAF}"

# -- Transports: the wire a connector speaks --
#: The wire a connector speaks: Channel Access (``ca``, which carries PVAccess
#: beside it), nothing at all (``in_process``), or a facility protocol.
Transport = Literal["ca", "in_process", "doocs", "tango"]
TRANSPORT_IN_PROCESS: Final = "in_process"
TRANSPORT_CA: Final = "ca"

# -- Control targets --
# What a *deployment* can be pointed at, as opposed to what a *config* selects.
# Deliberately three names and not one per connector type: a target names a
# machine (the facility's own, the virtual accelerator, the stand-in soft IOC),
# and which connector type reaches that machine is
# :mod:`osprey_connectors.targets`'s job to answer, not the caller's.
ControlTarget = Literal["live", "va", "standin"]
TARGET_LIVE: Final = "live"
TARGET_VA: Final = "va"
TARGET_STANDIN: Final = "standin"
#: Every :data:`ControlTarget`, in the order every render, roster and state
#: file lists them.
CONTROL_TARGETS: list[ControlTarget] = list(get_args(ControlTarget))

#: The kind of machine a control target is: the facility's own (``real``), the
#: deployment's stand-in once the controls server reports it so
#: (``rehearsal``), or the simulator.
MachineKind = Literal["real", "rehearsal", "simulator"]


def is_control_target(value: object) -> TypeIs[ControlTarget]:
    """Whether *value* is one of the three target words, spelled exactly.

    The parse point for a target a caller read from somewhere untyped — an
    environment variable, a request, a file — so a refusal that records
    whatever was asked can still hand a resolver a :data:`ControlTarget`.
    """
    return isinstance(value, str) and value in CONTROL_TARGETS


# -- The built-in connectors, one row each --
class BuiltinConnector(FrozenModel):
    """One connector this package ships, as a registry names it.

    The module path names an implementation module of this package, which only
    this table spells: a registry that loads connectors lazily by path takes
    its entries from here rather than restating the private paths.

    A control-system row also says what its type is, and every list of types
    this module offers is read off those fields. An archiver row states none
    of them: they describe a machine, and an archiver serves none.
    """

    #: The type name a config selects it by, and the key it registers under.
    name: str
    #: Which registry it goes into, and the ``connector_type`` a registration names.
    kind: Literal["control_system", "archiver"]
    module_path: str
    class_name: str
    description: str
    #: The wire the type speaks when it is served over a network. There is no
    #: PVAccess transport: the EPICS-family connectors carry PVAccess channels
    #: beside Channel Access, behind the same gateway gate, so ``ca`` covers
    #: both. A connector that speaks only PVAccess adds its own transport.
    transport: Transport | None = None
    #: The kind of machine the type serves: the facility's own, the stand-in a
    #: deployment rehearses on, or the simulator. Only a ``real`` type is ever
    #: a candidate for a deployment's ``live`` target.
    machine: MachineKind | None = None
    #: The target a deployment whose own ``control_system.type`` is this type
    #: sits on before anybody switches.
    baseline_target: ControlTarget | None = None
    #: Whether the type's history is invented rather than recorded: a machine
    #: the deployment stands up for itself, with no archive of its own.
    invents_history: bool = False
    #: Whether ``osprey init`` offers the type for a fresh project.
    offered_by_init: bool = False
    #: Whether the ``connector:`` shorthand of ``osprey set`` accepts it.
    offered_by_set: bool = False
    #: The port the type's server listens on when the config names none.
    default_port: int | None = None

    @model_validator(mode="after")
    def _describes_its_kind(self) -> Self:
        machine_fields = (self.transport, self.machine, self.baseline_target)
        if self.kind == "control_system":
            if self.name not in get_args(ControlSystemTypeName):
                raise ValueError(f"{self.name!r} is not a ControlSystemTypeName")
            if None in machine_fields:
                raise ValueError(f"{self.name!r} must state its transport, machine and target")
        elif machine_fields != (None, None, None) or (
            self.invents_history or self.offered_by_init or self.offered_by_set
        ):
            raise ValueError(f"archiver {self.name!r} states a control-system field")
        return self


#: Every built-in connector, in registration order.
BUILTIN_CONNECTORS: tuple[BuiltinConnector, ...] = (
    BuiltinConnector(
        name=EPICS,
        kind="control_system",
        module_path="osprey_connectors.control_system._epics_connector",
        class_name="EPICSConnector",
        description="EPICS Channel Access control system connector",
        transport=TRANSPORT_CA,
        machine="real",
        baseline_target=TARGET_LIVE,
        offered_by_init=True,
        offered_by_set=True,
    ),
    BuiltinConnector(
        name=VIRTUAL_ACCELERATOR,
        kind="control_system",
        module_path="osprey_connectors.control_system._va_connector",
        class_name="VirtualAcceleratorConnector",
        description="Virtual Accelerator connector for PyAT-backed soft-IOC simulations",
        transport=TRANSPORT_CA,
        machine="simulator",
        baseline_target=TARGET_VA,
        invents_history=True,
        offered_by_init=True,
        offered_by_set=True,
        # The Channel Access port the shipped Virtual Accelerator soft-IOC
        # serves on. It is here rather than only in the VA connector because
        # the deployment layer fills the same default when it dials the VA, and
        # it must not import a concrete connector (and the EPICS client stack
        # behind it) to learn one integer.
        default_port=5064,
    ),
    BuiltinConnector(
        name=DOOCS,
        kind="control_system",
        module_path="osprey_connectors.control_system._doocs_connector",
        class_name="DOOCSConnector",
        description="DOOCS control system connector (requires doocs4py)",
        transport="doocs",
        machine="real",
        baseline_target=TARGET_LIVE,
        offered_by_init=True,
        offered_by_set=True,
    ),
    BuiltinConnector(
        name=TANGO,
        kind="control_system",
        module_path="osprey_connectors.control_system._tango_connector",
        class_name="TangoConnector",
        description="TANGO Controls control system connector (requires PyTango)",
        transport="tango",
        machine="real",
        baseline_target=TARGET_LIVE,
        offered_by_init=True,
        offered_by_set=True,
    ),
    # The live stand-in is a soft IOC, so the EPICS connector is what reaches
    # it — but it registers under its own name rather than sharing 'epics'.
    # The registry key is what build_control_system_connector() names as the
    # context's connector type, and that type is what selects the
    # connector block ('control_system.connector.live_standin') and the write
    # posture read from it. Registering the stand-in as 'epics' would give it
    # the facility's authored block and the facility's arming.
    #
    # Settable, not initable. A deployment may be pointed at its stand-in once
    # it has one — ``osprey set connector=live_standin`` is how a session
    # starts on the soft IOC — but ``osprey init`` never offers it: a fresh
    # project has no stand-in to point at, so a project that began on one
    # would name a machine that does not exist.
    BuiltinConnector(
        name=LIVE_STANDIN,
        kind="control_system",
        module_path="osprey_connectors.control_system._epics_connector",
        class_name="EPICSConnector",
        description="Live stand-in connector: the facility-shaped soft IOC a deployment runs "
        "for itself, served over Channel Access",
        transport=TRANSPORT_CA,
        machine="rehearsal",
        baseline_target=TARGET_STANDIN,
        invents_history=True,
        offered_by_set=True,
    ),
    BuiltinConnector(
        name=MOCK_ARCHIVER,
        kind="archiver",
        module_path="osprey_connectors.archiver._mock_archiver_connector",
        class_name="MockArchiverConnector",
        description="Mock archiver connector for development and testing",
    ),
    BuiltinConnector(
        name=EPICS_ARCHIVER,
        kind="archiver",
        module_path="osprey_connectors.archiver._epics_archiver_connector",
        class_name="EPICSArchiverConnector",
        description="EPICS Archiver Appliance connector",
    ),
    BuiltinConnector(
        name=MONGODB_ARCHIVER,
        kind="archiver",
        module_path="osprey_connectors.archiver._mongodb_archiver_connector",
        class_name="MongoDBArchiverConnector",
        description="MongoDB archiver connector for time-series PV data",
    ),
    BuiltinConnector(
        name=DOOCS_ARCHIVER,
        kind="archiver",
        module_path="osprey_connectors.archiver._doocs_archiver_connector",
        class_name="DOOCSArchiverConnector",
        description="DOOCS local history connector (requires doocs4py)",
    ),
    BuiltinConnector(
        name=MYA_ARCHIVER,
        kind="archiver",
        module_path="osprey_connectors.archiver._mya_archiver_connector",
        class_name="MYAArchiverConnector",
        description="MYA archiver connector, read over the myquery HTTP service "
        "(requires jlab-archiver-client)",
    ),
)

#: The control-system rows, in registration order: what every list below reads.
_CONTROL_SYSTEMS = tuple(row for row in BUILTIN_CONNECTORS if row.kind == "control_system")


def _default_port(name: str) -> int:
    port = next(row.default_port for row in _CONTROL_SYSTEMS if row.name == name)
    if port is None:
        raise ValueError(f"{name!r} states no default port")
    return port


#: Channel Access port the shipped Virtual Accelerator soft-IOC serves on: the
#: simulator row's :attr:`BuiltinConnector.default_port`.
DEFAULT_VA_PORT = _default_port(VIRTUAL_ACCELERATOR)

#: The wire each built-in type speaks when it is served over a network, read
#: off each row's :attr:`BuiltinConnector.transport`.
_TRANSPORTS: dict[str, Transport | None] = {row.name: row.transport for row in _CONTROL_SYSTEMS}

# -- Retired type names --
#: Type names no deployment may state any more, each with the type that now
#: means it. A table rather than a missing registration, so that every site
#: refusing one says what to write instead.
RETIRED_CONTROL_SYSTEM_TYPES = {"mock": VIRTUAL_ACCELERATOR}

#: What a factory says when it builds from a section that states no type.
UNSET_TYPE_WARNING = (
    f"control_system.type is not set; defaulting to '{VIRTUAL_ACCELERATOR}' "
    "served in process (no network). Set control_system.type explicitly to "
    "select a connector."
)

# -- CLI choice lists (only types with implementations) --
#: What ``osprey init`` will materialize: every row offered by init.
CLI_CONTROL_SYSTEM_TYPES = [row.name for row in _CONTROL_SYSTEMS if row.offered_by_init]

#: What the ``connector:`` shorthand accepts: every row offered by set. Wider
#: than :data:`CLI_CONTROL_SYSTEM_TYPES` by the stand-in, which a deployment
#: may be pointed at once it has one but a fresh project never begins on.
SET_CONTROL_SYSTEM_TYPES = [row.name for row in _CONTROL_SYSTEMS if row.offered_by_set]

#: Types that serve a machine nobody has to be careful around. They are the
#: reason ``live`` cannot simply be "whatever the config selects": a deployment
#: whose baseline is one of these has not yet said what its real machine is.
SIMULATED_TYPES = tuple(row.name for row in _CONTROL_SYSTEMS if row.machine == "simulator")


def is_simulated(connector_type: str) -> bool:
    """Whether a connector type serves the simulator rather than a machine.

    A type predicate: the simulator answers True whether it is served or runs
    in process. It never says whether a connector dials a network; that is
    :func:`speaks_channel_access` and :func:`talks_to_network`.
    """
    return connector_type in SIMULATED_TYPES


#: Types that serve the live stand-in. Reachable only through the ``standin``
#: target: a stand-in is a machine in its own right, so it is never a candidate
#: for a deployment's ``live`` one — not as the baseline the derivation starts
#: from, and not as a block it falls back to. A deployment running the stand-in
#: beside its facility's own block still has exactly one real machine.
STANDIN_TYPES = tuple(row.name for row in _CONTROL_SYSTEMS if row.machine == "rehearsal")

#: Types whose history is invented rather than recorded, so pairing one with a
#: synthesizing archiver leaves nothing that ever tells a reader the data is
#: made up. Both are machines a deployment stands up for itself, and neither has
#: an archive of its own — see the framework's ``osprey.deployment.honesty``.
INVENTED_HISTORY_TYPES = tuple(row.name for row in _CONTROL_SYSTEMS if row.invents_history)


def resolve_control_system_type(section: "ControlSystemSection") -> str:
    """The control system a ``control_system:`` config section actually selects.

    A section that states no type selects the simulator, and
    :func:`resolve_serving` runs it in process, so it dials nothing. The
    transport (:func:`connector_transport`), never the type, says whether a
    connector dials a network. Any other value is returned as written (see
    :attr:`~osprey_connectors.config_sections.ControlSystemSection.type`), so a
    typo reaches the factory's "Unknown … type" error.
    """
    return section.type or VIRTUAL_ACCELERATOR


def resolve_serving(section: "ControlSystemSection") -> Serving:
    """Where the simulator a ``control_system:`` section selects runs.

    A section that states no type is the fail-safe default, the simulator in
    process, and the leaf is not read: a section that names nothing must dial
    nothing. A stated ``virtual_accelerator`` answers
    :data:`SERVING_KEY`, else :data:`SERVED`. Any other stated type is
    :data:`SERVED`: only the simulator has an in-process venue.

    Raises:
        ConfigurationError: when the leaf holds a value outside :data:`SERVING_MODES`.
    """
    if section.type is None:
        return IN_PROCESS
    if resolve_control_system_type(section) != VIRTUAL_ACCELERATOR:
        return SERVED
    block = section.block(VIRTUAL_ACCELERATOR)
    value = block.serving if _is_block(block) else None
    if value is None or isinstance(value, str):
        return value or SERVED
    raise ConfigurationError(
        f"{SERVING_KEY} is {value.raw!r}; it is {SERVED!r} or {IN_PROCESS!r}.", key=SERVING_KEY
    )


def _is_block(value: object) -> "TypeIs[ConnectorBlock]":
    """Whether *value* is a connector block the section could read."""
    from osprey_connectors.config_sections import ConnectorBlock

    return isinstance(value, ConnectorBlock)


def connector_transport(
    section: "ControlSystemSection", connector_type: str | None = None
) -> Transport | None:
    """The wire a connector built from *section* speaks.

    Computed once, here, from the type and the serving leaf, so every check that
    asks "does this speak Channel Access" or "does this dial a network" reads
    one derived fact and never the type word. *connector_type* defaults to
    :func:`resolve_control_system_type` of *section*; a caller building for a
    target passes the type the target resolves to.

    Returns:
        :data:`TRANSPORT_IN_PROCESS` for the simulator served in process, the
        type's row's :attr:`BuiltinConnector.transport` otherwise, and ``None``
        for a type with no row (a dotted custom connector).

    Raises:
        ConfigurationError: when the serving leaf holds an unknown value.
    """
    if connector_type is None:
        connector_type = resolve_control_system_type(section)
    if connector_type == VIRTUAL_ACCELERATOR and resolve_serving(section) == IN_PROCESS:
        return TRANSPORT_IN_PROCESS
    return _TRANSPORTS.get(connector_type)


def speaks_channel_access(
    section: "ControlSystemSection", connector_type: str | None = None
) -> bool:
    """Whether a connector built from *section* speaks real Channel Access.

    The facility's own EPICS machine, the simulator served from its container,
    or the live stand-in soft IOC. The queue worker builds its devices over
    Channel Access, so these are the connectors plans can execute against today
    and every other one browses. That is a property of the worker's device
    layer, not of the plan stack: a facility whose machine speaks another
    protocol executes plans once a device layer for it exists, and its
    transport joins this one — this is not a statement that no other protocol
    can ever run plans.
    """
    return connector_transport(section, connector_type) == TRANSPORT_CA


def talks_to_network(section: "ControlSystemSection", connector_type: str | None = None) -> bool:
    """Whether a connector built from *section* dials anything at all.

    False only for the simulator served in process. A transport this module has
    no row for (a dotted custom connector) answers True: what it dials is not
    known, so it is not assumed to dial nothing.
    """
    return connector_transport(section, connector_type) != TRANSPORT_IN_PROCESS


def retired_type_message(value: str) -> str:
    """The one sentence every site refusing a retired type name quotes."""
    replacement = RETIRED_CONTROL_SYSTEM_TYPES.get(value, VIRTUAL_ACCELERATOR)
    return (
        f"`{value}` is retired: the simulator in process is `control_system.type: "
        f"{replacement}` with `{SERVING_KEY}: {IN_PROCESS}` (`osprey set "
        f"connector={replacement} config.{SERVING_KEY}={IN_PROCESS}`), then "
        "rebuild with `osprey build`."
    )
