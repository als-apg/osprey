"""Serve a ``get_config_value``-shaped stand-in as the process config.

The write-path readers (the reference monitor, the EPICS read-only refusal, the
limits validator, the connector host's write report and the factory with no
section given) read the ``control_system`` section from
:func:`osprey_connectors.config.current_config`, not through
``get_config_value``. A test that fakes the deployment with a lookup function
therefore serves both: the lookup itself, for the readers that still go through
``get_config_value``, and a loaded config built from what the lookup answers for
``control_system``, for the readers that take the loaded config's section.

The config is built at each read, so a lookup that answers differently later in
the test, or that raises to stand for a config that cannot be read, is seen by
both kinds of reader alike.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from osprey_connectors import config as config_module
from osprey_connectors.config import LoadedConfig
from osprey_connectors.errors import ConfigLoadError, ConfigurationError
from osprey_connectors.testing import config_of

#: A ``get_config_value`` stand-in: ``(dotted_key, default) -> value``.
ConfigLookup = Callable[..., Any]

_GET_CONFIG_VALUE = "osprey_connectors.config.get_config_value"
_CURRENT_CONFIG = "osprey_connectors.config.current_config"


#: The dotted key of the limits database path, as the limits validator reads it.
_DATABASE_PATH_KEY = "control_system.limits_checking.database_path"

#: What a lookup answers for a key it does not hold, told apart from a ``None``.
_ABSENT = object()

#: The real process default, captured before any test patches it.
_real_current_config = config_module.current_config


def config_from_lookup(lookup: ConfigLookup) -> LoadedConfig:
    """The process config with *lookup*'s answer as its ``control_system`` section.

    The rest of the tree, and the path every relative path derives from, are the
    config the process would load without the lookup (none when none loads), as
    a lookup that delegated every other key to the real one answered them. The
    section is taken as the lookup answers it: placeholders in it stay verbatim.
    """
    section = lookup("control_system", {})
    try:
        base = _real_current_config()
    except (ConfigLoadError, ConfigurationError):
        tree: dict[str, Any] = {}
        path: Path | str = "config.yml"
    else:
        tree, path = base.raw_tree(), base.path
    database_path = lookup(_DATABASE_PATH_KEY, _ABSENT)
    limits = section.get("limits_checking") if isinstance(section, dict) else None
    if database_path is not _ABSENT and isinstance(limits, dict) and "database_path" not in limits:
        # The limits database path is read from the loaded config, where it is a
        # line of the deployment-wide block: a lookup that answers it dotted
        # beside the section describes that one line.
        section = {**section, "limits_checking": {**limits, "database_path": database_path}}
    tree["control_system"] = section
    # The path anchor a relative limits database resolves against, as the
    # lookup answers it: a top-level key, read from the loaded config.
    project_root = lookup("project_root", _ABSENT)
    if project_root is not _ABSENT:
        tree["project_root"] = project_root
    return config_of(tree, path=path, environ={})


def serve_config_lookup(monkeypatch: Any, lookup: ConfigLookup) -> None:
    """Serve *lookup* as the process config for the rest of the test."""
    monkeypatch.setattr(_GET_CONFIG_VALUE, lookup)
    monkeypatch.setattr(_CURRENT_CONFIG, lambda: config_from_lookup(lookup))


def _raising(error: BaseException) -> ConfigLookup:
    """A lookup that raises *error* for every key: a config that cannot be read."""

    def lookup(*_: Any, **__: Any) -> Any:
        raise error

    return lookup


@contextmanager
def config_lookup_patched(lookup: ConfigLookup | BaseException) -> Iterator[MagicMock]:
    """Serve *lookup* as the process config for the block.

    An exception in place of a lookup is raised by every read, as
    ``patch(..., side_effect=error)`` raised it.

    Yields:
        The mock standing in for ``get_config_value``, for a test that asserts on
        its calls.
    """
    if isinstance(lookup, BaseException):
        lookup = _raising(lookup)
    with (
        patch(_GET_CONFIG_VALUE, side_effect=lookup) as mocked,
        patch(_CURRENT_CONFIG, side_effect=lambda: config_from_lookup(lookup)),
    ):
        yield mocked


def constant_lookup(value: Any) -> ConfigLookup:
    """A lookup answering *value* for every key, as ``patch(..., return_value=value)`` did."""

    def lookup(*_: Any, **__: Any) -> Any:
        return value

    return lookup
