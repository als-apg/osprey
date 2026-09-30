"""The notebook sidecar's confined contents manager: its shape and its import.

Confinement itself — symlink escapes refused on ``/api/contents``, ``/files``
and checkpoints, in-root paths served — is asserted over real HTTP against a
running sidecar in ``test_jupyter_sidecar.py``. What stays here is what a
running server cannot show: that the manager keeps the async base, and that
importing it does not pull the web application into the sidecar process.
"""

from __future__ import annotations

import subprocess
import sys

from jupyter_server.services.contents.filemanager import AsyncFileContentsManager

from osprey.interfaces.web_terminal.jupyter_contents import ConfinedFileContentsManager


def test_the_sidecar_uses_an_async_manager() -> None:
    assert issubclass(ConfinedFileContentsManager, AsyncFileContentsManager)


def test_importing_the_module_does_not_build_the_web_application() -> None:
    probe = (
        "import sys\n"
        "import osprey.interfaces.web_terminal.jupyter_contents\n"
        "print('osprey.interfaces.web_terminal.app' in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
