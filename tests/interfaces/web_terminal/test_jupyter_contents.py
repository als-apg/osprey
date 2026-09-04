"""Tests for the notebook sidecar's confined contents manager.

The upstream root check is textual, so these cover the case it misses — a name
under the root whose target is not — alongside the ordinary paths that must
keep working, and the import isolation the sidecar process depends on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from jupyter_server.base.handlers import AuthenticatedFileHandler
from jupyter_server.files.handlers import FilesHandler
from jupyter_server.services.contents.filecheckpoints import AsyncFileCheckpoints
from jupyter_server.services.contents.filemanager import AsyncFileContentsManager
from tornado.web import HTTPError

from osprey.interfaces.web_terminal.jupyter_contents import (
    ConfinedFileCheckpoints,
    ConfinedFileContentsManager,
    RawFilesHandler,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Return a contents root with a sibling directory to escape into."""
    contents_root = tmp_path / "notebooks"
    contents_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    return contents_root


@pytest.fixture
def manager(root: Path) -> ConfinedFileContentsManager:
    return ConfinedFileContentsManager(root_dir=str(root))


def test_the_sidecar_uses_an_async_manager() -> None:
    assert issubclass(ConfinedFileContentsManager, AsyncFileContentsManager)


def test_a_file_in_the_root_resolves(manager: ConfinedFileContentsManager, root: Path) -> None:
    (root / "run.ipynb").write_text("{}")

    assert manager._get_os_path("run.ipynb") == str(root / "run.ipynb")


def test_a_path_that_does_not_exist_yet_resolves(
    manager: ConfinedFileContentsManager, root: Path
) -> None:
    assert manager._get_os_path("new.ipynb") == str(root / "new.ipynb")


def test_the_root_itself_resolves(manager: ConfinedFileContentsManager, root: Path) -> None:
    assert manager._get_os_path("") == str(root)


def test_a_symlink_inside_the_root_resolves(
    manager: ConfinedFileContentsManager, root: Path
) -> None:
    (root / "real.ipynb").write_text("{}")
    (root / "alias.ipynb").symlink_to(root / "real.ipynb")

    assert manager._get_os_path("alias.ipynb") == str(root / "alias.ipynb")


def test_a_symlink_out_of_the_root_is_not_found(
    manager: ConfinedFileContentsManager, root: Path
) -> None:
    (root / "escape.txt").symlink_to(root.parent / "outside" / "secret.txt")

    with pytest.raises(HTTPError) as excinfo:
        manager._get_os_path("escape.txt")

    assert excinfo.value.status_code == 404


def test_a_symlinked_directory_out_of_the_root_is_not_found(
    manager: ConfinedFileContentsManager, root: Path
) -> None:
    (root / "elsewhere").symlink_to(root.parent / "outside", target_is_directory=True)

    with pytest.raises(HTTPError) as excinfo:
        manager._get_os_path("elsewhere/secret.txt")

    assert excinfo.value.status_code == 404


def test_a_parent_traversal_is_not_found(manager: ConfinedFileContentsManager) -> None:
    with pytest.raises(HTTPError) as excinfo:
        manager._get_os_path("../outside/secret.txt")

    assert excinfo.value.status_code == 404


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


def test_raw_files_are_served_through_the_contents_manager(
    manager: ConfinedFileContentsManager, root: Path
) -> None:
    """``/files/`` must reach ``_get_os_path``; the static handler never does."""
    assert manager.files_handler_class is RawFilesHandler
    assert issubclass(RawFilesHandler, FilesHandler)
    assert not issubclass(manager.files_handler_class, AuthenticatedFileHandler)

    handlers = manager.get_extra_handlers()

    assert handlers == [(r"/files/(.*)", RawFilesHandler, {"path": str(root)})]


def test_checkpoints_share_the_confinement(
    manager: ConfinedFileContentsManager, root: Path
) -> None:
    """The checkpoints resolve paths with their own copy of the mixin."""
    (root / "elsewhere").symlink_to(root.parent / "outside", target_is_directory=True)
    checkpoints = manager.checkpoints

    assert isinstance(checkpoints, ConfinedFileCheckpoints)
    assert isinstance(checkpoints, AsyncFileCheckpoints)
    assert checkpoints.root_dir == str(root)
    assert checkpoints._get_os_path("run.ipynb") == str(root / "run.ipynb")
    with pytest.raises(HTTPError) as excinfo:
        checkpoints.checkpoint_path("checkpoint", "elsewhere/x.ipynb")
    assert excinfo.value.status_code == 404
    assert not (root.parent / "outside" / ".ipynb_checkpoints").exists()
