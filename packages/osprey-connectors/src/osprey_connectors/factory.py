"""
Connector factory for creating control system and archiver connectors.

Provides centralized factory for instantiating and configuring connectors
based on configuration. Connectors are registered through the Osprey registry
system for unified component management and lazy loading.

"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from osprey_connectors import types
from osprey_connectors.control_system._base import ControlSystemConnector
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from osprey_connectors.alarm_log import AlarmLog
    from osprey_connectors.alarm_state import AlarmState

    # Import only for annotations: pulling in the archiver base eagerly would
    # make pandas a hard dependency of the lean control-system import chain.
    from osprey_connectors.archiver._base import ArchiverConnector
    from osprey_connectors.directory import ChannelDirectory
    from osprey_connectors.put_log import PutLog
    from osprey_connectors.service_answers import BackendSpec

__all__ = [
    "BUILTIN_CONNECTORS",
    "BuiltinConnector",
    "ConnectorFactory",
    "register_builtin_connectors",
]

logger = get_logger("connector_factory")

#: One directory kind's backend base; a create call returns the kind it builds.
_DirectoryBackend = TypeVar(
    "_DirectoryBackend", "ChannelDirectory", "AlarmLog", "AlarmState", "PutLog"
)

#: A directory kind's name, as ``BackendSpec.kind`` spells it.
_DirectoryKind = Literal["directory", "alarm_log", "alarm_state", "put_log"]


def _dotted_class(connector_type: str) -> Any:
    """Import the class a dotted out-of-tree type names.

    Raises:
        ValueError: The module does not import, or has no such class.
    """
    module_path, class_name = connector_type.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ValueError(f"Could not import connector module '{module_path}': {e}") from e
    except AttributeError as e:
        raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from e


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
    _pv_directory_connectors: dict[str, type[ChannelDirectory]] = {}
    _alarm_log_connectors: dict[str, type[AlarmLog]] = {}
    _alarm_state_connectors: dict[str, type[AlarmState]] = {}
    _put_log_connectors: dict[str, type[PutLog]] = {}

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
    def register_pv_directory(cls, name: str, connector_class: type[ChannelDirectory]) -> None:
        """Register a PV directory backend under the type name a config selects it by."""
        cls._pv_directory_connectors[name] = connector_class
        logger.debug(f"Registered PV directory connector: {name}")

    @classmethod
    def register_alarm_log(cls, name: str, connector_class: type[AlarmLog]) -> None:
        """Register an alarm log backend under the type name a config selects it by."""
        cls._alarm_log_connectors[name] = connector_class
        logger.debug(f"Registered alarm log connector: {name}")

    @classmethod
    def register_alarm_state(cls, name: str, connector_class: type[AlarmState]) -> None:
        """Register an alarm state backend under the type name a config selects it by."""
        cls._alarm_state_connectors[name] = connector_class
        logger.debug(f"Registered alarm state connector: {name}")

    @classmethod
    def register_put_log(cls, name: str, connector_class: type[PutLog]) -> None:
        """Register a put log backend under the type name a config selects it by."""
        cls._put_log_connectors[name] = connector_class
        logger.debug(f"Registered put log connector: {name}")

    @classmethod
    def build_control_system_connector(
        cls, config: dict[str, Any] | None = None, *, control_target: str | None = None
    ) -> tuple[ControlSystemConnector, dict[str, Any]]:
        """
        Build a control system connector and its type config, without connecting.

        Args:
            config: Control system configuration dict, or None to load it
                from the global config. Its keys:

                - type: Connector type (e.g., 'epics', 'virtual_accelerator').
                  Defaults to the simulator served in process, with a warning,
                  when unset, so an under-specified config never reaches live
                  hardware.
                - connector: Dict with connector-specific configs
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
                connector_class = _dotted_class(connector_type)
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
            config: Archiver configuration dict, or None to load it from the
                global config. Its keys:

                - type: Connector type (e.g., 'epics_archiver', 'mock_archiver').
                  Defaults to 'mock_archiver' with a warning when unset, so a
                  config that omits the archiver section stays serviceable.
                - settings: The selected connector's settings, handed to
                  connect() whole. A short-named connector's own-name block is
                  read when settings is absent; both at once raise ValueError.

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
                connector_class = _dotted_class(connector_type)
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
    def _build_directory_backend(
        cls,
        registry: dict[str, type[_DirectoryBackend]],
        spec: BackendSpec,
        project_dir: Path | None,
        seams: dict[str, Any],
    ) -> _DirectoryBackend:
        """Build the backend a spec names, registering the builtins first.

        The panel and the directory MCP server never register builtins themselves,
        and registration converges, so every create call can afford to.

        Raises:
            ValueError: The type is unknown, or a dotted type does not resolve.
        """
        register_builtin_connectors()
        backend_class = registry.get(spec.type)
        if backend_class is None:
            if "." not in spec.type:
                raise ValueError(
                    f"Unknown {spec.kind} type: '{spec.type}'. "
                    f"Available types: {list(registry)}. "
                    f"Use a dotted module path for custom connectors."
                )
            backend_class = _dotted_class(spec.type)
            registry[spec.type] = backend_class
        given = {name: seam for name, seam in seams.items() if seam is not None}
        return backend_class(spec, project_dir=project_dir, **given)

    @classmethod
    async def create_pv_directory_connector(
        cls, spec: BackendSpec, *, project_dir: Path | None = None, **seams: Any
    ) -> ChannelDirectory:
        """Build a PV directory backend. Backends connect nothing here: no I/O happens.

        Args:
            spec: The backend to build, from ``service_targets``.
            project_dir: The rendered config's directory.
            **seams: Test seams, passed on only when not ``None``.

        Raises:
            ValueError: If the type is unknown or does not resolve.
        """
        return cls._build_directory_backend(cls._pv_directory_connectors, spec, project_dir, seams)

    @classmethod
    async def create_alarm_log_connector(
        cls, spec: BackendSpec, *, project_dir: Path | None = None, **seams: Any
    ) -> AlarmLog:
        """Build an alarm log backend, as :meth:`create_pv_directory_connector` does."""
        return cls._build_directory_backend(cls._alarm_log_connectors, spec, project_dir, seams)

    @classmethod
    async def create_alarm_state_connector(
        cls, spec: BackendSpec, *, project_dir: Path | None = None, **seams: Any
    ) -> AlarmState:
        """Build an alarm state backend, as :meth:`create_pv_directory_connector` does."""
        return cls._build_directory_backend(cls._alarm_state_connectors, spec, project_dir, seams)

    @classmethod
    async def create_put_log_connector(
        cls, spec: BackendSpec, *, project_dir: Path | None = None, **seams: Any
    ) -> PutLog:
        """Build a put log backend, as :meth:`create_pv_directory_connector` does."""
        return cls._build_directory_backend(cls._put_log_connectors, spec, project_dir, seams)

    @classmethod
    def list_control_systems(cls) -> list:
        """List available control system connector types."""
        return list(cls._control_system_connectors.keys())

    @classmethod
    def list_archivers(cls) -> list:
        """List available archiver connector types."""
        return list(cls._archiver_connectors.keys())


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


@dataclass(frozen=True)
class _DirectoryBuiltin:
    """One directory-kind backend this package ships.

    Apart from :class:`BuiltinConnector` because the framework registry reads
    that table, and these kinds must never reach a framework that predates them.
    """

    #: The type name a ``BackendSpec`` selects it by, and the key it registers under.
    name: str
    kind: _DirectoryKind
    module_path: str
    class_name: str
    description: str

    def load(self) -> Any:
        """Import the module and return the backend class."""
        return getattr(importlib.import_module(self.module_path), self.class_name)


#: Every built-in directory-kind backend, in registration order. ``simulator_view``
#: and ``alarm_replay`` are internal: ``service_targets`` selects them, and the
#: framework's section resolver refuses them as a configured ``type``.
_DIRECTORY_BUILTINS: tuple[_DirectoryBuiltin, ...] = (
    _DirectoryBuiltin(
        "channelfinder",
        "directory",
        "osprey_connectors.directory._channelfinder",
        "ChannelFinderDirectory",
        "ChannelFinder directory service",
    ),
    _DirectoryBuiltin(
        "simulator_view",
        "directory",
        "osprey_connectors.directory._simulator",
        "SimulatorDirectory",
        "Every PV the virtual accelerator serves, from the build's simulator view",
    ),
    _DirectoryBuiltin(
        "phoebus_alarm_logger",
        "alarm_log",
        "osprey_connectors.alarm_log._phoebus",
        "PhoebusAlarmLogger",
        "Phoebus alarm logger service",
    ),
    _DirectoryBuiltin(
        "phoebus_alarm_server",
        "alarm_state",
        "osprey_connectors.alarm_state._phoebus_kafka",
        "PhoebusAlarmState",
        "Phoebus alarm server state, from its Kafka topic (requires aiokafka)",
    ),
    _DirectoryBuiltin(
        "alarm_replay",
        "alarm_state",
        "osprey_connectors.alarm_state._replay",
        "ReplayAlarmState",
        "Alarm state replayed from a file, for the simulator",
    ),
    _DirectoryBuiltin(
        "caputlog",
        "put_log",
        "osprey_connectors.put_log._caputlog",
        "CaPutLog",
        "caPutLog put history service",
    ),
)

_BUILTIN_PV_DIRECTORIES = tuple(c.name for c in _DIRECTORY_BUILTINS if c.kind == "directory")
_BUILTIN_ALARM_LOGS = tuple(c.name for c in _DIRECTORY_BUILTINS if c.kind == "alarm_log")
_BUILTIN_ALARM_STATES = tuple(c.name for c in _DIRECTORY_BUILTINS if c.kind == "alarm_state")
_BUILTIN_PUT_LOGS = tuple(c.name for c in _DIRECTORY_BUILTINS if c.kind == "put_log")


def _directory_registry(
    kind: _DirectoryKind,
) -> (
    dict[str, type[ChannelDirectory]]
    | dict[str, type[AlarmLog]]
    | dict[str, type[AlarmState]]
    | dict[str, type[PutLog]]
):
    """The factory's registry for one directory kind."""
    if kind == "directory":
        return ConnectorFactory._pv_directory_connectors
    if kind == "alarm_log":
        return ConnectorFactory._alarm_log_connectors
    if kind == "alarm_state":
        return ConnectorFactory._alarm_state_connectors
    return ConnectorFactory._put_log_connectors


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
    missing_directory_rows = [
        row for row in _DIRECTORY_BUILTINS if row.name not in _directory_registry(row.kind)
    ]
    if not missing_control_systems and not missing_archivers and not missing_directory_rows:
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

    # Their optional client (aiokafka) is imported on start, not at module scope.
    for row in missing_directory_rows:
        _directory_registry(row.kind)[row.name] = row.load()
