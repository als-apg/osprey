"""ARIEL qmd export enhancement module.

Mirrors each enhanced entry into the markdown tree the qmd sidecar indexes, so
that entries become searchable through the ``qmd`` search mode without a second
ingestion path. The rendering, sharding, containment and byte-compare rules all
live in :mod:`writer`; this module is the pipeline adapter around them.

The freshness marker is the reason this module exists as a pipeline step rather
than a periodic dump: the sidecar watches ``<mirror>/.qmd-touch`` and reindexes
when its mtime advances, so a marker touched on every entry — including entries
whose rendered bytes did not change — would keep the sidecar reindexing the
whole corpus forever. The marker is therefore advanced only when
:func:`~osprey.services.ariel_search.enhancement.qmd_export.writer.write_entry`
reports a real write.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
from osprey.services.ariel_search.enhancement.qmd_export.writer import write_entry
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.database.migrations import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

#: Fixed-name freshness marker written at the mirror root after every real
#: write. The qmd sidecar remembers the last mtime it acted on and reindexes
#: when that mtime advances, so a mirrored entry becomes searchable promptly
#: instead of waiting out the sidecar's interval sweep. The file is only ever
#: replaced, never deleted, so a consumer cannot observe it missing.
TOUCH_MARKER_NAME = ".qmd-touch"


def resolve_mirror_path(raw: str | Path, config_dir: Path | None = None) -> Path:
    """Resolve a configured ``mirror_path`` to an absolute path.

    Follows the same rule as
    :func:`osprey.services.facility_knowledge.bundle_path.resolve_bundle_path`,
    which is what every config-relative path in a rendered project promises:
    expand ``~``, return an absolute value unchanged, and resolve a relative one
    against the directory holding ``config.yml``. The exporter, the resync CLI
    and the compose mount must all land on the same directory, or the sidecar
    indexes a tree nobody writes to.

    Args:
        raw: The configured value, as read from
            ``enhancement_modules.qmd_export.mirror_path``.
        config_dir: Directory containing ``config.yml``. When omitted it is
            derived from
            :func:`osprey.utils.workspace.resolve_config_path`, which falls
            back to the process CWD when ``OSPREY_CONFIG`` is unset.

    Returns:
        Absolute :class:`~pathlib.Path` to the mirror root.
    """
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    if config_dir is not None:
        return (config_dir / path).resolve()

    try:
        from osprey.utils.workspace import resolve_config_path

        config_dir = Path(resolve_config_path()).parent
        return (config_dir / path).resolve()
    except Exception:  # noqa: BLE001 — resolution must never break ingestion.
        return path.resolve()


class QmdExportModule(BaseEnhancementModule):
    """Mirror enhanced entries into the markdown tree qmd indexes.

    Writes one ``<mirror>/YYYY/MM/<encoded entry_id>.md`` file per entry and
    advances the sidecar's freshness marker when — and only when — a file
    actually changed on disk.
    """

    def __init__(self) -> None:
        """Initialize the module with no mirror configured."""
        self._mirror_root: Path | None = None

    @property
    def name(self) -> str:
        """Return module identifier."""
        return "qmd_export"

    @property
    def migration(self) -> type[BaseMigration] | None:
        """Return migration class for this module.

        Returns:
            ``None`` — the mirror lives on the filesystem, not in Postgres.
        """
        return None

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the module with settings from config.yml.

        Args:
            config: The ``enhancement_modules.qmd_export`` config dict. Only
                ``mirror_path`` is read; a relative value is resolved against
                the directory holding ``config.yml``.

        Raises:
            ValueError: If no ``mirror_path`` is configured. An enabled
                exporter with nowhere to write would report every entry
                mirrored while qmd search stayed empty, so it fails at
                configuration time instead.
        """
        raw = config.get("mirror_path")
        if not raw:
            raise ValueError(
                "ariel.enhancement_modules.qmd_export.mirror_path is required "
                "when the qmd_export enhancement is enabled. Set it to the "
                "directory the qmd sidecar indexes."
            )
        self._mirror_root = resolve_mirror_path(raw)

    async def enhance(
        self,
        entry: EnhancedLogbookEntry,
        conn: AsyncConnection,
    ) -> None:
        """Mirror one entry to the markdown tree and refresh the sidecar marker.

        The write is skipped when the file on disk already holds exactly the
        rendered bytes, and the freshness marker is advanced only when a write
        really happened. Advancing it unconditionally would ask the sidecar to
        reindex the whole corpus on every ingestion tick.

        A module that was never configured mirrors nothing; a module configured
        without a ``mirror_path`` never gets this far, because
        :meth:`configure` refuses it.

        Args:
            entry: The entry to mirror.
            conn: Database connection from pool. Unused — this module writes to
                the filesystem, not to Postgres.

        Raises:
            ValueError: If the entry has no mirrorable ``entry_id``. The row is
                recorded as a qmd_export failure rather than silently skipped,
                because a missing mirror file is a search gap.
            OSError: If the mirror file cannot be written.
        """
        del conn  # Filesystem-only module; kept for the pipeline signature.

        if self._mirror_root is None:
            return

        if write_entry(self._mirror_root, entry):
            self._touch_marker()

    def _touch_marker(self) -> None:
        """Advance the mirror's :data:`TOUCH_MARKER_NAME` freshness marker.

        Atomically rewrites ``<mirror>/.qmd-touch`` with the current UTC
        timestamp, which advances its mtime. Marker resolution is the
        filesystem's, so several entries landing inside one mtime tick may
        present to the sidecar as a single advance; the sidecar's interval
        sweep is the backstop for that.

        A failure here does not invalidate the entry that was just mirrored —
        the markdown file is on disk and the interval sweep will still find it
        — so the error is logged rather than raised.
        """
        if self._mirror_root is None:
            return

        marker = self._mirror_root / TOUCH_MARKER_NAME
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(marker, f"{datetime.now(UTC).isoformat()}\n")
        except OSError as e:
            logger.warning(
                f"qmd_export: could not update freshness marker {marker}: {e} — "
                "the sidecar will pick the change up on its next interval sweep"
            )

    def mirror_now(self, entry: Mapping[str, Any]) -> bool:
        """Mirror one entry synchronously, outside the enhancement pipeline.

        The write-path twin of :meth:`enhance`, for callers that hold a plain
        entry mapping rather than a pipeline row — most importantly the
        service's ``create_entry``, whose optimistic upsert would otherwise be
        invisible to hybrid search until the next batch enhancement run.

        Args:
            entry: The entry to mirror.

        Returns:
            ``True`` when a file changed on disk (and the freshness marker was
            advanced), ``False`` when the mirror already held these bytes or
            the module is unconfigured.
        """
        if self._mirror_root is None:
            return False
        if write_entry(self._mirror_root, entry):
            self._touch_marker()
            return True
        return False

    async def health_check(self) -> tuple[bool, str]:
        """Check if module is ready.

        Returns:
            Tuple of (healthy, message). Unhealthy when no ``mirror_path`` is
            configured or the mirror root cannot be created.
        """
        if self._mirror_root is None:
            return (False, "mirror_path is not configured")
        try:
            self._mirror_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return (False, f"mirror path {self._mirror_root} is not writable: {e}")
        return (True, "OK")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to a path atomically, replacing any existing file.

    The content goes to a temp file in the same directory (so the rename stays
    within one filesystem), is flushed to disk, and is then moved over the
    destination. A reader sees either the previous file or the complete new
    one, never a partially written one.

    Args:
        path: Destination file. Its parent directory must already exist.
        text: Full file content to write, UTF-8 encoded.

    Raises:
        OSError: If the temp file cannot be created, written, or renamed. The
            temp file is removed on any failure, leaving the destination
            untouched.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def mirror_entry_best_effort(config: Any, entry: Mapping[str, Any]) -> bool:
    """Mirror one entry into the qmd markdown tree without ever raising.

    The service's ``create_entry`` calls this after its optimistic upsert so an
    agent- or operator-written entry becomes hybrid-searchable within one
    sidecar poll instead of waiting for the next batch enhancement run. It is
    best-effort by contract: the entry is already durable in Postgres, and a
    mirror problem must not fail the write that caused it — the batch resync
    remains the backstop for anything skipped here.

    Args:
        config: The loaded ARIEL configuration (duck-typed: only
            ``is_enhancement_module_enabled`` and
            ``get_enhancement_module_config`` are read, avoiding an import
            cycle with :mod:`osprey.services.ariel_search.config`).
        entry: The entry to mirror, as an ``enhanced_entries``-shaped mapping.

    Returns:
        ``True`` when a mirror file changed on disk, ``False`` otherwise —
        including every failure path.
    """
    try:
        if not config.is_enhancement_module_enabled("qmd_export"):
            return False
        module = QmdExportModule()
        module.configure(config.get_enhancement_module_config("qmd_export") or {})
        return module.mirror_now(entry)
    except Exception as e:  # noqa: BLE001 — see docstring: never fail the write path.
        logger.warning(
            f"qmd_export: could not mirror entry {entry.get('entry_id')!r} inline: {e} — "
            "it stays searchable by keyword and will reach the mirror on the next "
            "enhancement run or `osprey ariel qmd-resync`."
        )
        return False
