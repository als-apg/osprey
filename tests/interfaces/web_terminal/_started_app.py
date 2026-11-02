"""The whole web-terminal app, started over a config the test writes in code.

The render-table suites — theme, rail position, tour — each carried a
generator that booted ``create_app`` under a couple of config patches and was
torn down by a hand-driven ``next(gen)``, and two of them answered *every*
``get_config_value`` key with the one value under test. This module
centralizes that boot as :func:`started_client`: the ``web:`` section the
lifespan reads and the ``get_config_value`` keys it asks for both come from
the test, key by key, and teardown is an ordinary ``with`` block.

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
        web: The ``web:`` section ``load_osprey_config`` answers; ``None`` is
            an empty section.
        config_values: What ``get_config_value`` answers, per dotted key. A key
            not named here answers the caller's default, so no test inherits a
            value it did not set.
    """
    values = dict(config_values or {})

    def get_config_value(path: str, default: Any = None, config_path: str | None = None) -> Any:  # noqa: ARG001
        return values.get(path, default)

    with terminal_client(
        workspace_dir,
        patch(
            "osprey_connectors.workspace.load_osprey_config",
            return_value={"web": dict(web or {})},
        ),
        patch("osprey_connectors.config.get_config_value", get_config_value),
    ) as client:
        yield client
