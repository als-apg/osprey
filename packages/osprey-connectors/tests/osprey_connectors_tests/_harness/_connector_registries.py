"""Bracket a test's changes to the connector factory's registries.

``ConnectorFactory`` keeps its registrations in two class-level dicts that live
for the whole process. A test that registers, replaces or clears connectors
does it inside :func:`isolated_connector_registries`, which puts both dicts back
as they were on the way out.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def isolated_connector_registries(*, clear: bool = False) -> Iterator[None]:
    """Bracket registry mutations so the surrounding process state survives.

    This is the sanctioned way for tests to mutate the factory registries.
    Clearing them directly is destructive: registrations made elsewhere (by
    ``initialize_registry()``, by another test module, by a facility preset)
    are gone for the rest of the process, and ``register_builtin_connectors``
    cannot tell a deliberately customized registry from a truncated one.

    The restore is total: registrations added inside the block are dropped
    and registrations removed inside it are put back.

    Args:
        clear: Empty both registries on entry, for code that needs to observe
            registration from scratch. The clear is undone on exit like any
            other mutation.

    Example:
        >>> with isolated_connector_registries(clear=True):
        ...     register_builtin_connectors()
    """
    # Imported here: the harness package is loaded by every conftest, and the
    # factory brings the control-system contract with it.
    from osprey_connectors.factory import ConnectorFactory

    control_system = dict(ConnectorFactory._control_system_connectors)
    archiver = dict(ConnectorFactory._archiver_connectors)
    if clear:
        ConnectorFactory._control_system_connectors.clear()
        ConnectorFactory._archiver_connectors.clear()
    try:
        yield
    finally:
        ConnectorFactory._control_system_connectors.clear()
        ConnectorFactory._control_system_connectors.update(control_system)
        ConnectorFactory._archiver_connectors.clear()
        ConnectorFactory._archiver_connectors.update(archiver)
