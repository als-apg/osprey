"""Hooks for tests of code that reads the control state or the config.

Nothing in the package or the framework calls these: they exist so a test can
put a process back the way it started between cases, or hand code a config
without writing a file, which production never needs to do.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path

from osprey_connectors import config as config_module
from osprey_connectors import control_context
from osprey_connectors import dotenv as dotenv_module
from osprey_connectors.config import LoadedConfig, _loaded

__all__ = ["clear_config_state", "config_of", "invalidate_cache"]


def invalidate_cache() -> None:
    """Forget every control-context record and server report this process has parsed.

    Every read re-stats its file and re-parses only when the signature moved
    (rule 4 of :mod:`~osprey_connectors.control_context`), so a test that writes
    a file inside one filesystem clock tick of the last, or that moves the
    agent-data root under a reader, drops the cache through this before it reads.
    """
    control_context._invalidate_cache()


def clear_config_state() -> None:
    """Forget all config state this process holds, as at its start.

    The process default, every cached load, the current context's
    ``using_config`` scope, the env chain's shell-override record and every
    warn-once latch (the facility-timezone fallback, the ignored shell-only
    variables). The latches are process state in exactly the way a cache is:
    left set, whichever test happened to run first would consume the single
    warning and every later test asserting on it would see none — an
    order-dependent pass that says nothing about the code.
    """
    config_module._clear_process_state()
    dotenv_module._clear_process_state()


def config_of(
    mapping: Mapping[str, object],
    *,
    path: Path | str = "config.yml",
    environ: Mapping[str, str] | None = None,
) -> LoadedConfig:
    """*mapping* as the config a file holding it would load as.

    The load after the parse, unchanged: ``${VAR}`` placeholders resolve
    against *environ* (the process environment by default), the load's
    refusals apply, and the repo root and facility zone are read. *mapping*
    itself is copied, never held.

    Args:
        mapping: The document, as a YAML parse of ``config.yml`` would give it.
        path: The path the config claims to come from; nothing is read there,
            but the repo root and every path relative to the config derive from
            it.
        environ: What placeholders resolve against.

    Raises:
        ConfigurationError: What :func:`osprey_connectors.config.load_config`
            refuses after the parse.
    """
    return _loaded(
        copy.deepcopy(dict(mapping)),
        Path(path),
        environ=os.environ if environ is None else environ,
    )
