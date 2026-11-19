"""The whole web-terminal app, started over a config the test writes in code.

The render-table suites — theme, rail position, tour — each carried a
generator that booted ``create_app`` under a couple of config patches and was
torn down by a hand-driven ``next(gen)``, and two of them answered *every*
config key with the one value under test. This module centralizes that boot
as :func:`started_client`: the config the lifespan reads, the ``web:`` section
and every other key it asks for, comes from the test, key by key, and teardown
is an ordinary ``with`` block.

:func:`terminal_client` is the boot under it, and the one most route suites use
directly: ``_load_web_config`` answers only the watched directory, any further
patches the test hands over are live across ``create_app`` and the lifespan,
and every other keyword goes to ``create_app``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey_connectors.testing import config_of
from osprey_connectors_tests._config_trees import config_tree, installed


@contextmanager
def terminal_client(
    watch_dir: Path | str,
    *patches: AbstractContextManager[Any],
    shell_command: Sequence[str] = ("echo",),
    **create_app_kw: Any,
) -> Iterator[TestClient]:
    """Yield a ``TestClient`` whose lifespan has run over *watch_dir*.

    Args:
        watch_dir: What ``_load_web_config`` answers as ``watch_dir``.
        *patches: Context managers entered, in order, before ``create_app``
            and left after the lifespan has shut down.
        shell_command: Passed to ``create_app``.
        **create_app_kw: Passed to ``create_app``.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(watch_dir)},
            )
        )
        for extra in patches:
            stack.enter_context(extra)
        app = create_app(shell_command=list(shell_command), **create_app_kw)
        with TestClient(app) as client:
            yield client


@contextmanager
def started_client(
    workspace_dir: Path,
    *,
    web: Mapping[str, Any] | None = None,
    config_values: Mapping[str, Any] | None = None,
) -> Iterator[TestClient]:
    """Yield a ``TestClient`` whose lifespan has run over the given config.

    Args:
        workspace_dir: The watched directory.
        web: The config's ``web:`` section; ``None`` is an empty section.
        config_values: The config's other values, per dotted key (one under
            ``web.`` is merged into that section). A key not named here is
            absent, so no test inherits a value it did not set.
    """
    tree = config_tree({"web": dict(web or {}), **dict(config_values or {})})
    with terminal_client(workspace_dir, installed(config_of(tree, environ={}))) as client:
        yield client
