"""Contents manager that keeps the notebook sidecar inside its own root.

``jupyter_server`` 2.21 resolves an API path with
:func:`~jupyter_server.utils.to_os_path` and then checks that the *textual*
result sits under ``root_dir``. Text is not the filesystem: a symlink whose
name lives under the root but whose target does not passes that check, so the
contents API would happily read and write through it. The confinement here
resolves the symlink chain and compares the real path, which is the property
the sidecar's root actually needs to hold.

The base class is the one ``ServerApp`` picks by default —
``AsyncLargeFileManager`` — so the sidecar keeps async contents handling and
chunked uploads and never trips the synchronous-manager deprecation warning.
``_get_os_path`` is defined once, on ``FileManagerMixin``, and every class
that touches the disk inherits that single definition; the override lives in
one mixin so that every such class gets the same check. Three classes reach
the disk:

* the contents manager itself, for every ``/api/contents`` operation;
* the checkpoints, which carry their own ``root_dir`` and their own copy of
  the mixin: listing and deleting a checkpoint resolve the notebook's parent
  directory through it, and would create ``.ipynb_checkpoints`` there;
* the raw file handler (``/files/<path>``, behind Download and Open in New
  Browser Tab). ``FileContentsManager`` serves it through
  ``AuthenticatedFileHandler``, a ``StaticFileHandler`` that reads straight off
  ``root_dir`` and never asks the contents manager, so tornado's own textual
  root check lets a symlink through. The manager's ``files_handler_class`` is
  therefore pointed at ``jupyter_server``'s generic ``FilesHandler``, which
  fetches the file through ``ContentsManager.get`` — and so through
  ``_get_os_path``. The price is that a raw file is read into memory rather
  than streamed, which the contents API already does for the same files.

Importing this module must not build the web application: it is imported only
inside the sidecar process, which serves no FastAPI route.
"""

from __future__ import annotations

import os

from jupyter_server.files.handlers import FilesHandler
from jupyter_server.services.contents.filecheckpoints import AsyncFileCheckpoints
from jupyter_server.services.contents.fileio import FileManagerMixin
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
from tornado.web import HTTPError, RequestHandler
from traitlets import default

__all__ = ["ConfinedFileCheckpoints", "ConfinedFileContentsManager", "RawFilesHandler"]


class _ConfinedPaths(FileManagerMixin):
    """``_get_os_path`` that refuses a path whose real location escapes ``root_dir``."""

    root_dir: str  # the trait every concrete class below declares

    def _get_os_path(self, path: str) -> str:
        """Return the OS path for an API path, refusing one that escapes the root.

        Args:
            path: Relative API path to a file or directory.

        Returns:
            Native, absolute OS path, unresolved, exactly as the base class
            returns it.

        Raises:
            HTTPError: 404 when the path, with every symlink resolved, is not
                the root directory or somewhere beneath it.
        """
        os_path: str = super()._get_os_path(path)
        root = os.path.realpath(self.root_dir)
        resolved = os.path.realpath(os_path)
        if resolved != root and not resolved.startswith(root + os.sep):
            raise HTTPError(404, f"{path} is outside root contents directory")
        return os_path


class ConfinedFileCheckpoints(_ConfinedPaths, AsyncFileCheckpoints):
    """File checkpoints whose paths must resolve inside ``root_dir``."""


class RawFilesHandler(FilesHandler):
    """``FilesHandler`` whose ``HEAD`` answers.

    Upstream ``head`` calls the async ``get`` without awaiting it, and the
    inherited ``compute_etag`` asserts a static-file attribute ``get`` never
    sets, so every ``HEAD`` was a 500.
    """

    def compute_etag(self) -> str | None:
        return None

    async def head(self, path: str) -> None:  # type: ignore[override]
        response = self.get(path, include_body=False)
        if response is not None:
            await response


class ConfinedFileContentsManager(_ConfinedPaths, AsyncLargeFileManager):  # type: ignore[misc]
    """A contents manager whose paths must resolve inside ``root_dir``.

    The ignore is upstream's own: ``AsyncFileContentsManager`` carries it for
    the same sync/async ``resolve_path`` pair, which surfaces again whenever a
    class names two bases.
    """

    @default("checkpoints_class")
    def _checkpoints_class_default(self) -> type[AsyncFileCheckpoints]:
        return ConfinedFileCheckpoints

    @default("files_handler_class")
    def _files_handler_class_default(self) -> type[RequestHandler]:
        """Serve ``/files/`` through :meth:`get`, so :meth:`_get_os_path` confines it."""
        return RawFilesHandler
