"""
Connector factory for creating control system and archiver connectors.

Provides centralized factory for instantiating and configuring connectors
based on configuration. Connectors are registered through the Osprey registry
system for unified component management and lazy loading.

"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from osprey_connectors import types
from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    # Import only for annotations: pulling in the archiver base eagerly would
    # make pandas a hard dependency of the lean control-system import chain.
    from osprey_connectors.archiver._base import ArchiverConnector

__all__ = [
    "BUILTIN_CONNECTORS",
    "BuiltinConnector",
    "ConnectorFactory",
    "ConnectorRegistrySnapshot",
    "isolated_connector_registries",
    "register_builtin_connectors",
    "restore_connector_registries",
    "snapshot_connector_registries",
]

logger = get_logger("connector_factory")


class ConnectorFactory:
    """
    Factory for creating control system and archiver connectors.

    Provides centralized management of available connectors. Connectors are
    registered automatically through the Osprey registry system during
    framework initialization.

    Example:
        >>> # Using default config
        >>> cs_connector = await ConnectorFactory.create_control_system_connector()
        >>>
        >>> # Using custom config
        >>> config = {'type': 'epics', 'connector': {'epics': {...}}}
        >>> cs_connector = await ConnectorFactory.create_control_system_connector(config)
    """

    _control_system_connectors: dict[str, type[ControlSystemConnector]] = {}
    _archiver_connectors: dict[str, type[ArchiverConnector]] = {}

    @classmethod
    def register_control_system(
        cls, name: str, connector_class: type[ControlSystemConnector]
    ) -> None:
        """
        Register a control system connector.

        Args:
            name: Unique name for the connector (e.g., 'epics', 'tango')
            connector_class: Connector class implementing ControlSystemConnector
        """
        cls._control_system_connectors[name] = connector_class
        logger.debug(f"Registered control system connector: {name}")

    @classmethod
    def register_archiver(cls, name: str, connector_class: type[ArchiverConnector]) -> None:
        """
        Register an archiver connector.

        Args:
            name: Unique name for the connector (e.g., 'epics_archiver', 'mock_archiver')
            connector_class: Connector class implementing ArchiverConnector
        """
        cls._archiver_connectors[name] = connector_class
        logger.debug(f"Registered archiver connector: {name}")

    @classmethod
    def build_control_system_connector(
        cls, config: dict[str, Any] | None = None, *, control_target: str | None = None
    ) -> tuple[ControlSystemConnector, dict[str, Any]]:
        """
        Build a control system connector and its type config, without connecting.

        Args:
            config: Control system configuration dict with keys:
                - type: Connector type (e.g., 'epics', 'virtual_accelerator').
                  Defaults to the simulator served in process, with a warning,
                  when unset, so an under-specified config never reaches live
                  hardware.
                - connector: Dict with connector-specific configs
                If None, loads from global config
            control_target: The control target this connector is being built
                for — ``live``, ``va`` or ``standin``. Stamped on the instance
                as ``_control_target`` and used to index the per-target
                narrowing in the control-context record, and nothing else: the
                deployment half of the write posture stays keyed on the
                connector *type*. Callers
                that have no target to name leave it ``None``, which is the
                behaviour every caller had before the stamp existed.

        Returns:
            ``(connector, type_config)`` — the stamped but *unconnected*
            connector, and the type-specific config block ``connect()`` takes.
            Both halves are needed to finish the build, and the second is not
            derivable from the first, so they travel together.

        Raises:
            ValueError: If connector type is unknown or config is invalid

        Example:
            >>> connector, type_config = ConnectorFactory.build_control_system_connector(config)
            >>> await connector.connect(type_config)
        """
        # Load config if not provided
        if config is None:
            try:
                from osprey_connectors.config import get_config_value

                config = get_config_value("control_system", {})
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        # Fail closed: an under-specified config must never bring up a live
        # control system. A blank value is treated as absent for the same
        # reason. The fallback lives in types.resolve_control_system_type so
        # that anything deciding what this call WILL build — the honesty rule's
        # three refusals above all — resolves it identically.
        connector_type = types.resolve_control_system_type(config)
        if not config.get("type"):
            logger.warning(types.UNSET_TYPE_WARNING)

        connector_class = cls._control_system_connectors.get(connector_type)
        # The venue is the factory's choice, made from the same function every
        # check reads the connector's wire from: the simulator served in process
        # is built here, and nowhere else decides it.
        if (
            connector_type == types.VIRTUAL_ACCELERATOR
            and types.resolve_serving(config) == types.IN_PROCESS
        ):
            from osprey_connectors.control_system._va_in_process_connector import (
                VAInProcessConnector,
            )

            connector_class = VAInProcessConnector

        if not connector_class:
            if "." in connector_type:
                module_path, class_name = connector_type.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    connector_class = getattr(module, class_name)
                except ImportError as e:
                    raise ValueError(
                        f"Could not import connector module '{module_path}': {e}"
                    ) from e
                except AttributeError as e:
                    raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from e
                cls._control_system_connectors[connector_type] = connector_class
            else:
                if connector_type in types.RETIRED_CONTROL_SYSTEM_TYPES:
                    raise ValueError(types.retired_type_message(connector_type))
                available = list(cls._control_system_connectors.keys())
                raise ValueError(
                    f"Unknown control system type: '{connector_type}'. "
                    f"Available types: {available}. "
                    f"Use a dotted module path for custom connectors."
                )

        # Create connector instance
        connector = connector_class()
        # The one seam between construction and connect(): the write posture and
        # the limits posture are both per connector type, and connect() itself
        # already consults the stamp to load them.
        connector._connector_type = connector_type
        # On the same seam, the wire this instance speaks: computed once from the
        # type and the serving leaf, and echoed by a host child so its parent can
        # refuse a child that built the other venue.
        connector._transport = types.connector_transport(config, connector_type)
        # Beside it, and for the same reason: the session half of the posture is
        # per target, so the target has to be on the instance before connect()
        # reads it. The two stamps answer different questions — the type selects
        # the deployment's connector block, the target indexes the session
        # posture store — so neither is derivable from the other.
        connector._control_target = control_target

        # Get type-specific configuration
        connector_configs = config.get("connector", {})
        type_config = connector_configs.get(connector_type, {})

        logger.debug(f"Built control system connector: {connector_type}")
        return connector, type_config

    @classmethod
    async def create_control_system_connector(
        cls, config: dict[str, Any] | None = None, *, control_target: str | None = None
    ) -> ControlSystemConnector:
        """
        Create, configure and connect a control system connector.

        The whole build in one call, and what almost every caller wants.
        :meth:`build_control_system_connector` is the same build stopped one
        step short, for the one caller that has to reach the instance between
        the type stamp and ``connect()``.

        Args:
            config: Control system configuration dict, as
                :meth:`build_control_system_connector` documents it.
            control_target: The control target this connector is being built
                for, stamped on the instance as
                :meth:`build_control_system_connector` documents it.

        Returns:
            Initialized and connected ControlSystemConnector

        Raises:
            ValueError: If connector type is unknown or config is invalid
            ConnectionError: If connection fails

        Example:
            >>> config = {
            >>>     'type': 'epics',
            >>>     'connector': {
            >>>         'epics': {
            >>>             'timeout_s': 5.0,
            >>>             'gateways': {'read_only': {...}}
            >>>         }
            >>>     }
            >>> }
            >>> connector = await ConnectorFactory.create_control_system_connector(config)
        """
        connector, type_config = cls.build_control_system_connector(
            config, control_target=control_target
        )
        await connector.connect(type_config)
        logger.debug(f"Created control system connector: {connector._connector_type}")
        return connector

    @classmethod
    async def create_archiver_connector(
        cls, config: dict[str, Any] | None = None
    ) -> ArchiverConnector:
        """
        Create and configure an archiver connector.

        Args:
            config: Archiver configuration dict with keys:
                - type: Connector type (e.g., 'epics_archiver', 'mock_archiver').
                  Defaults to 'mock_archiver' with a warning when unset, so a
                  config that omits the archiver section stays serviceable.
                - settings: The selected connector's settings, handed to
                  connect() whole. A short-named connector's own-name block is
                  read when settings is absent; both at once raise ValueError.
                If None, loads from global config

        Returns:
            Initialized and connected ArchiverConnector

        Raises:
            ValueError: If connector type is unknown or config is invalid
            ConnectionError: If connection fails

        Example:
            >>> config = {
            >>>     'type': 'epics_archiver',
            >>>     'settings': {
            >>>         'url': 'https://archiver.example.org:8443',
            >>>         'timeout_s': 60
            >>>     }
            >>> }
            >>> connector = await ConnectorFactory.create_archiver_connector(config)
        """
        # Load config if not provided
        if config is None:
            try:
                from osprey_connectors.config import get_config_value

                config = get_config_value("archiver", {})
            except Exception as e:
                logger.warning(f"Could not load config: {e}, using defaults")
                config = {}

        # Fail closed: a config with no `archiver:` section is under-specified,
        # not a declaration that a facility archiver exists. A blank value is
        # treated as absent for the same reason. The fallback lives in
        # types.resolve_archiver_type because the honesty rule's refusals have
        # to answer "what will this build?" with exactly this answer.
        connector_type = types.resolve_archiver_type(config)
        if not config.get("type"):
            logger.warning(
                f"archiver.type is not set; defaulting to '{types.MOCK_ARCHIVER}'. "
                f"Set archiver.type explicitly to select a connector."
            )

        connector_class = cls._archiver_connectors.get(connector_type)

        if not connector_class:
            if "." in connector_type:
                module_path, class_name = connector_type.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    connector_class = getattr(module, class_name)
                except ImportError as e:
                    raise ValueError(
                        f"Could not import connector module '{module_path}': {e}"
                    ) from e
                except AttributeError as e:
                    raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from e
                cls._archiver_connectors[connector_type] = connector_class
            else:
                available = list(cls._archiver_connectors.keys())
                raise ValueError(
                    f"Unknown archiver type: '{connector_type}'. "
                    f"Available types: {available}. "
                    f"Use a dotted module path for custom connectors."
                )

        # Create connector instance
        connector = connector_class()

        # The selected archiver's settings block. The rule is in types so that the
        # build's refusal and this read give the same answer.
        type_config = types.resolve_archiver_settings(config)

        # Connect with configuration
        await connector.connect(type_config)

        logger.info(f"Created archiver connector: {connector_type}")
        return connector

    @classmethod
    def list_control_systems(cls) -> list:
        """List available control system connector types."""
        return list(cls._control_system_connectors.keys())

    @classmethod
    def list_archivers(cls) -> list:
        """List available archiver connector types."""
        return list(cls._archiver_connectors.keys())


@dataclass(frozen=True)
class ConnectorRegistrySnapshot:
    """Captured contents of both :class:`ConnectorFactory` registries.

    Treat the two dicts as read-only. They are the restore point, not a working
    copy: mutating either one changes what
    :func:`restore_connector_registries` puts back, so the registries would be
    restored to a state the process was never in.
    """

    control_system: dict[str, type[ControlSystemConnector]] = field(default_factory=dict)
    archiver: dict[str, type[ArchiverConnector]] = field(default_factory=dict)


def snapshot_connector_registries() -> ConnectorRegistrySnapshot:
    """Copy the current control system and archiver registrations.

    Returns:
        Snapshot that :func:`restore_connector_registries` can replay.
    """
    return ConnectorRegistrySnapshot(
        control_system=dict(ConnectorFactory._control_system_connectors),
        archiver=dict(ConnectorFactory._archiver_connectors),
    )


def restore_connector_registries(snapshot: ConnectorRegistrySnapshot) -> None:
    """Replace both registries with the contents of ``snapshot``.

    The restore is total: registrations added after the snapshot are dropped
    and registrations removed after it are put back, so any mutation made in
    between is undone.

    Args:
        snapshot: Value returned by :func:`snapshot_connector_registries`.
    """
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._control_system_connectors.update(snapshot.control_system)
    ConnectorFactory._archiver_connectors.clear()
    ConnectorFactory._archiver_connectors.update(snapshot.archiver)


@contextmanager
def isolated_connector_registries(*, clear: bool = False) -> Iterator[ConnectorRegistrySnapshot]:
    """Bracket registry mutations so the surrounding process state survives.

    This is the sanctioned way for tests to mutate the factory registries.
    Clearing them directly is destructive: registrations made elsewhere (by
    ``initialize_registry()``, by another test module, by a facility preset)
    are gone for the rest of the process, and
    :func:`register_builtin_connectors` cannot tell a deliberately customized
    registry from a truncated one.

    Args:
        clear: Empty both registries on entry, for code that needs to observe
            registration from scratch. The clear is undone on exit like any
            other mutation.

    Yields:
        The snapshot taken on entry, before any ``clear``. It is the restore
        point for the ``finally`` below — read it, do not mutate it.

    Example:
        >>> with isolated_connector_registries(clear=True):
        ...     register_builtin_connectors()
    """
    snapshot = snapshot_connector_registries()
    if clear:
        ConnectorFactory._control_system_connectors.clear()
        ConnectorFactory._archiver_connectors.clear()
    try:
        yield snapshot
    finally:
        restore_connector_registries(snapshot)


@dataclass(frozen=True)
class BuiltinConnector:
    """One connector this package ships, as a registry names it.

    The module path names an implementation module of this package, which only
    this table spells: a registry that loads connectors lazily by path takes
    its entries from here rather than restating the private paths.
    """

    #: The type name a config selects it by, and the key it registers under.
    name: str
    #: Which registry it goes into, and the ``connector_type`` a registration names.
    kind: Literal["control_system", "archiver"]
    module_path: str
    class_name: str
    description: str

    def load(self) -> Any:
        """Import the module and return the connector class."""
        return getattr(importlib.import_module(self.module_path), self.class_name)


#: Every built-in connector, in registration order.
BUILTIN_CONNECTORS: tuple[BuiltinConnector, ...] = (
    BuiltinConnector(
        types.EPICS,
        "control_system",
        "osprey_connectors.control_system._epics_connector",
        "EPICSConnector",
        "EPICS Channel Access control system connector",
    ),
    BuiltinConnector(
        types.VIRTUAL_ACCELERATOR,
        "control_system",
        "osprey_connectors.control_system._va_connector",
        "VirtualAcceleratorConnector",
        "Virtual Accelerator connector for PyAT-backed soft-IOC simulations",
    ),
    BuiltinConnector(
        types.DOOCS,
        "control_system",
        "osprey_connectors.control_system._doocs_connector",
        "DOOCSConnector",
        "DOOCS control system connector (requires doocs4py)",
    ),
    BuiltinConnector(
        types.TANGO,
        "control_system",
        "osprey_connectors.control_system._tango_connector",
        "TangoConnector",
        "TANGO Controls control system connector (requires PyTango)",
    ),
    # The live stand-in is a soft IOC, so the EPICS connector is what reaches
    # it — but it registers under its own name rather than sharing 'epics'.
    # The registry key is what build_control_system_connector() stamps onto
    # the instance as its connector type, and that stamp is what selects the
    # connector block ('control_system.connector.live_standin') and the write
    # posture read from it. Registering the stand-in as 'epics' would give it
    # the facility's authored block and the facility's arming.
    BuiltinConnector(
        types.LIVE_STANDIN,
        "control_system",
        "osprey_connectors.control_system._epics_connector",
        "EPICSConnector",
        "Live stand-in connector: the facility-shaped soft IOC a deployment runs "
        "for itself, served over Channel Access",
    ),
    BuiltinConnector(
        types.MOCK_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._mock_archiver_connector",
        "MockArchiverConnector",
        "Mock archiver connector for development and testing",
    ),
    BuiltinConnector(
        types.EPICS_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._epics_archiver_connector",
        "EPICSArchiverConnector",
        "EPICS Archiver Appliance connector",
    ),
    BuiltinConnector(
        types.MONGODB_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._mongodb_archiver_connector",
        "MongoDBArchiverConnector",
        "MongoDB archiver connector for time-series PV data",
    ),
    BuiltinConnector(
        types.DOOCS_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._doocs_archiver_connector",
        "DOOCSArchiverConnector",
        "DOOCS local history connector (requires doocs4py)",
    ),
    BuiltinConnector(
        types.MYA_ARCHIVER,
        "archiver",
        "osprey_connectors.archiver._mya_archiver_connector",
        "MYAArchiverConnector",
        "MYA archiver connector, read over the myquery HTTP service "
        "(requires jlab-archiver-client)",
    ),
)

_BUILTIN_CONTROL_SYSTEMS = tuple(c.name for c in BUILTIN_CONNECTORS if c.kind == "control_system")
_BUILTIN_ARCHIVERS = tuple(c.name for c in BUILTIN_CONNECTORS if c.kind == "archiver")


def register_builtin_connectors() -> None:
    """Register all built-in control system and archiver connectors.

    Safe to call multiple times and convergent: every call ensures each
    built-in name is registered, and an existing registration under one of
    those names is never replaced. A registry holding only part of the
    built-ins (e.g., one that was cleared and then partially repopulated)
    therefore heals on the next call instead of keeping the truncated set.
    """
    missing_control_systems = [
        name
        for name in _BUILTIN_CONTROL_SYSTEMS
        if name not in ConnectorFactory._control_system_connectors
    ]
    missing_archivers = [
        name for name in _BUILTIN_ARCHIVERS if name not in ConnectorFactory._archiver_connectors
    ]
    if not missing_control_systems and not missing_archivers:
        return

    # The DOOCS and MongoDB connectors import their optional drivers (doocs4py,
    # pymongo) inside connect(), not at module scope, so they register
    # unconditionally like any other built-in. A machine without the driver
    # only finds out when it tries to connect — which is the point at which it
    # would have failed anyway. Registering MongoDB conditionally instead used
    # to leave it out of _BUILTIN_ARCHIVERS, which broke convergence: a registry
    # holding the other built-ins satisfied the early return above and never
    # healed the missing mongodb_archiver entry.
    for builtin in BUILTIN_CONNECTORS:
        if builtin.kind == "control_system":
            if builtin.name not in ConnectorFactory._control_system_connectors:
                ConnectorFactory.register_control_system(builtin.name, builtin.load())
        elif builtin.kind == "archiver":
            if builtin.name not in ConnectorFactory._archiver_connectors:
                ConnectorFactory.register_archiver(builtin.name, builtin.load())
        else:
            raise ValueError(
                f"built-in connector {builtin.name!r} has unknown kind {builtin.kind!r}"
            )
