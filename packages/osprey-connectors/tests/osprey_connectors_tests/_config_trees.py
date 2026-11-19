"""Configs a test writes in code, served as the process config.

A test that fakes the deployment builds the config a file holding it would
load as (:func:`osprey_connectors.testing.config_of`) and makes it the config
the code under test reads: for the rest of the test with
:func:`serve_config` (the suites' autouse reset puts the process back), or for
a block with :func:`installed`. Both install it process-wide
(:func:`osprey_connectors.config.install_config`), so the threads a server or an
executor runs on read it too, which a ``using_config`` block, scoped to one
context, does not reach.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from osprey_connectors.config import LoadedConfig, current_config, install_config
from osprey_connectors.errors import ConfigLoadError, ConfigurationError
from osprey_connectors.testing import config_of


def config_tree(
    values: Mapping[str, Any], *, base: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The nested tree whose dotted keys answer *values*, over a copy of *base*.

    ``{"a.b": 1, "c": {"d": 2}}`` is ``{"a": {"b": 1}, "c": {"d": 2}}``; a
    dotted key written after a mapping at its prefix is merged into it.
    """
    tree: dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in values.items():
        node = tree
        *parents, leaf = key.split(".")
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = node[part] = {}
            node = child
        node[leaf] = copy.deepcopy(value)
    return tree


def control_system_config(
    section: Mapping[str, Any], *, path: Path | str = "config.yml", **rest: Any
) -> LoadedConfig:
    """A loaded config whose ``control_system`` section is *section*.

    ``${VAR}`` placeholders stay as written, as a section a test spells out
    means them. *rest* adds other top-level sections.
    """
    return config_of({"control_system": dict(section), **rest}, path=path, environ={})


def serve_config(config: LoadedConfig) -> LoadedConfig:
    """Make *config* the process config for the rest of the test."""
    install_config(config)
    return config


def serve_control_system(section: Mapping[str, Any], **rest: Any) -> LoadedConfig:
    """:func:`serve_config` of :func:`control_system_config`."""
    return serve_config(control_system_config(section, **rest))


def current_tree() -> tuple[dict[str, Any], Path]:
    """The process config's tree and path, or an empty tree when none loads."""
    try:
        config = current_config()
    except (ConfigLoadError, ConfigurationError):
        return {}, Path("config.yml")
    return config.raw_tree(), config.path


def with_values(values: Mapping[str, Any]) -> LoadedConfig:
    """The process config with *values* (dotted keys) set over it."""
    tree, path = current_tree()
    return config_of(config_tree(values, base=tree), path=path, environ={})


@contextmanager
def installed(config: LoadedConfig) -> Iterator[LoadedConfig]:
    """Make *config* the process config for the block; the previous one is back after."""
    previous = install_config(config)
    try:
        yield config
    finally:
        install_config(previous)
