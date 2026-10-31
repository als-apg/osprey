"""A real web-terminal hub for the browser suites, its companion backends bypassed.

The browser suites that serve the hub in-process boot ``create_app`` here,
under three patches: the watched directory, a fixed panel roster, and a
``_launch_panel_server`` that publishes a URL instead of starting a backend
process. :func:`hub_app` applies them and hands back the app before it is
served, for a suite that installs middleware or wraps it first;
:func:`hub_server` also serves it. A suite may add patches of its own, or
replace the launcher with ``launch``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi import FastAPI

from osprey.interfaces._serving import run_app_server
from osprey.interfaces.web_terminal.app import create_app
from tests.interfaces._panel_launch import publish_artifact_url

#: The shell a hub that never reads its terminal is launched with.
HUB_SHELL = ("echo", "hello")

#: The shell a hub whose terminal must stay up is launched with. It outlives the
#: test AND tolerates the ``--session-id <id>`` / ``--resume <id>`` (and any
#: ``--effort <level>``) arguments the websocket route appends: ``echo``,
#: ``sleep`` and ``cat`` each exit or error on them, and an exit trips
#: terminal.js's auto-resume failover, which clears the stored session id
#: mid-test.
LONG_LIVED_SHELL = (sys.executable, "-c", "import time; time.sleep(3600)")


@contextmanager
def hub_app(
    watch_dir: Path | str,
    *patches: AbstractContextManager[Any],
    enabled_panels: Iterable[str] = ("artifacts",),
    custom_panels: Sequence[dict] = (),
    launch: Callable[[Any, str], None] | None = None,
    shell_command: Sequence[str] = HUB_SHELL,
    **create_app_kw: Any,
) -> Iterator[FastAPI]:
    """Yield the hub app, built and not yet served, with its patches live.

    Args:
        watch_dir: What ``_load_web_config`` answers as ``watch_dir``.
        *patches: Further context managers, entered after the hub's own three
            and before ``create_app``.
        enabled_panels: The built-in panel ids ``_load_panel_config`` enables.
        custom_panels: The config-declared panels it answers.
        launch: The ``_launch_panel_server`` side effect; ``None`` publishes
            the standard artifact URL (:func:`publish_artifact_url`).
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
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=(set(enabled_panels), list(custom_panels), None),
            )
        )
        stack.enter_context(
            patch(
                "osprey.interfaces.web_terminal.app._launch_panel_server",
                side_effect=publish_artifact_url() if launch is None else launch,
            )
        )
        for extra in patches:
            stack.enter_context(extra)
        yield create_app(shell_command=list(shell_command), **create_app_kw)


@contextmanager
def hub_server(
    watch_dir: Path | str, *patches: AbstractContextManager[Any], **hub_kw: Any
) -> Iterator[tuple[str, FastAPI]]:
    """Serve :func:`hub_app` and yield ``(base_url, app)``.

    The server is up and its lifespan has run when this yields, and it is
    joined while the patches are still live.
    """
    with hub_app(watch_dir, *patches, **hub_kw) as app, run_app_server(app) as base_url:
        yield base_url, app
