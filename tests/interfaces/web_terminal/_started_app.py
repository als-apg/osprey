"""The whole web-terminal app, started over a config the test writes in code.

The render-table suites — theme, UI mode, rail position, tour — each carried a
generator that booted ``create_app`` under a couple of config patches and was
torn down by a hand-driven ``next(gen)``, and two of them answered *every*
``get_config_value`` key with the one value under test. This module
centralizes that boot as :func:`started_client`: the ``web:`` section the
lifespan reads and the ``get_config_value`` keys it asks for both come from
the test, key by key, and teardown is an ordinary ``with`` block.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app


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

    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": dict(web or {})},
        ),
        patch("osprey.utils.config.get_config_value", get_config_value),
    ):
        app = create_app(shell_command=["echo"])
        with TestClient(app) as client:
            yield client
