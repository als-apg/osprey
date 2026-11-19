"""The service table of a whole config held as a mapping, for tests that write one.

:func:`~osprey_connectors.service_targets.service_targets` takes the sections
it reads; a test that writes a whole config reads them from it here, the way
the framework's edge does.
"""

from collections.abc import Mapping

from osprey_connectors.config_sections import ArchiverSection, PvDirectorySection
from osprey_connectors.deployment.simulator import has_simulator, virtual_accelerator_port
from osprey_connectors.service_targets import ServiceTable, service_targets


def table_of(config: object) -> ServiceTable:
    """The service table of *config*, a whole config mapping (anything else has no sections)."""
    tree: Mapping[str, object] = config if isinstance(config, Mapping) else {}
    return service_targets(
        PvDirectorySection.read(tree.get("pv_directory")),
        ArchiverSection.read(tree.get("archiver")),
        simulator_port=virtual_accelerator_port(tree),
        built_simulator=has_simulator(tree),
    )
