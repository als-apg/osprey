"""The execution-side half of the artifact store.

Agent code runs in a subprocess that has no artifact store: it cannot reach the
index, the gallery server, or the listeners. So ``save_artifact()`` inside that
subprocess writes files into ``<execution_folder>/artifacts/`` and appends an
entry to ``artifacts/manifest.json``; the parent reads the manifest back with
:func:`collect_artifacts` and registers each file properly.

Both ends of that contract live here. :data:`SAVE_ARTIFACT_SOURCE` is the
source text every execution wrapper injects — the python executor's and the
visualization sandbox's alike — and it delegates type detection to
:func:`osprey.stores.artifact_store.serialize_object`, the same serializer the
in-process store uses. There is one answer to "how is this object saved",
whichever path saved it.

The manifest and everything beside it are written by agent code, so the parent
reads them as untrusted input: :func:`collect_artifacts` returns only regular
files that sit directly in ``artifacts/``, and :func:`read_collected_file` is
how a collected file's bytes are read. Neither follows a link out of the folder.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger("osprey.stores.artifact_manifest")

#: Name of the per-execution manifest inside ``artifacts/``.
MANIFEST_FILENAME = "manifest.json"

#: Name of the folder an execution saves its artifacts into.
ARTIFACTS_DIRNAME = "artifacts"

#: The largest file the parent reads back from an execution, in bytes.
MAX_COLLECTED_FILE_BYTES = 512 * 1024 * 1024

#: The most bytes the parent reads back from one execution's artifacts, in all.
MAX_COLLECTED_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

#: The most files the parent collects from one execution's artifacts.
MAX_COLLECTED_ENTRIES = 1000

#: The largest manifest the parent parses, in bytes.
MAX_MANIFEST_BYTES = 16 * 1024 * 1024

#: ``save_artifact()`` as injected into an execution subprocess. Plain source
#: text, no template placeholders: the wrappers splice it in verbatim. It reads
#: ``_execution_dir`` from the module globals the wrapper defines and falls back
#: to the working directory.
SAVE_ARTIFACT_SOURCE = '''\
# save_artifact() — the execution-side half of the artifact store
def save_artifact(obj, title="Untitled", description="", artifact_type=None, category=""):
    """Save an object as a gallery artifact.

    Supported types:
      - plotly Figure -> interactive HTML
      - bokeh Model -> dashboard HTML
      - matplotlib Figure -> PNG image
      - pandas DataFrame -> HTML table
      - str -> markdown or HTML (auto-detected)
      - dict / list -> JSON
      - bytes -> binary file
      - numpy ndarray -> .npy file

    Args:
        obj: The object to save.
        title: Human-readable title shown in the gallery.
        description: Optional longer description.
        artifact_type: Override the auto-detected type.
        category: Optional category key for gallery grouping.
    """
    import json as _json
    import uuid as _uuid
    from pathlib import Path as _Path

    from osprey.stores.artifact_store import serialize_object as _serialize_object

    _art_base = _Path(globals().get("_execution_dir", _Path.cwd()))
    artifacts_dir = _art_base / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    art_id = _uuid.uuid4().hex[:12]
    content, detected_type, filename, mime_type = _serialize_object(obj, title)
    # Prefix with the id so two saves under one title never overwrite each other.
    filename = f"{art_id}_{filename}"
    final_type = artifact_type or detected_type

    (artifacts_dir / filename).write_bytes(content)

    manifest_path = artifacts_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    entry = {
        "id": art_id,
        "filename": filename,
        "title": title,
        "description": description,
        "artifact_type": final_type,
        "mime_type": mime_type,
        "size_bytes": len(content),
    }
    if category:
        entry["category"] = category
    manifest.append(entry)
    manifest_path.write_text(_json.dumps(manifest, indent=2))

    print(f"Artifact saved: {title} ({final_type}, {len(content)} bytes)")
'''


class CollectedFileRefused(OSError):
    """A file an execution left is not one the parent reads.

    Raised for a name that is not a bare file name, a directory or file reached
    through a link, a file that is not regular, one with a second hard link,
    and one larger than the cap.
    """


def _is_bare_filename(name: str) -> bool:
    """Whether *name* names one entry of a directory and nothing else."""
    return (
        name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
        and name == os.path.basename(name)
    )


def _open_regular(directory: Path, name: str, max_bytes: int) -> int:
    """Open the regular file *name* directly inside *directory*; the open descriptor.

    *directory* is opened without following a link at its last component and
    *name* is opened relative to it, also without following a link, so the file
    read is the one checked. Opened non-blocking, so a FIFO planted under the
    name cannot hold the parent.

    Raises:
        CollectedFileRefused: The name, the directory or the file fails a rule.
    """
    if not _is_bare_filename(name):
        raise CollectedFileRefused(f"{name!r} is not a bare file name")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        dir_fd = os.open(directory, flags | os.O_DIRECTORY)
    except OSError as exc:
        raise CollectedFileRefused(f"{directory} is not a directory to read from: {exc}") from exc
    try:
        try:
            linked = os.lstat(name, dir_fd=dir_fd)
        except OSError as exc:
            raise CollectedFileRefused(f"{directory / name} cannot be read: {exc}") from exc
        _check_regular(directory / name, linked, max_bytes)
        try:
            fd = os.open(name, flags | os.O_NONBLOCK, dir_fd=dir_fd)
        except OSError as exc:
            raise CollectedFileRefused(f"{directory / name} cannot be opened: {exc}") from exc
    finally:
        os.close(dir_fd)
    try:
        opened = os.fstat(fd)
        _check_regular(directory / name, opened, max_bytes)
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise CollectedFileRefused(f"{directory / name} changed while it was opened")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _check_regular(path: Path, status: os.stat_result, max_bytes: int) -> None:
    """Refuse *status* unless it is a regular file with one hard link, within the cap."""
    if stat.S_ISLNK(status.st_mode):
        raise CollectedFileRefused(f"{path} is a link")
    if not stat.S_ISREG(status.st_mode):
        raise CollectedFileRefused(f"{path} is not a regular file")
    if status.st_nlink != 1:
        raise CollectedFileRefused(f"{path} has more than one hard link")
    if status.st_size > max_bytes:
        raise CollectedFileRefused(f"{path} holds {status.st_size} bytes, more than {max_bytes}")


def read_collected_file(path: Path, *, max_bytes: int = MAX_COLLECTED_FILE_BYTES) -> bytes:
    """The bytes of a file an execution left, read without following a link.

    The file must be a regular file with one hard link, at most *max_bytes*
    long, and its directory must not be a link. The checks are made on the open
    descriptor the bytes are read from.

    Args:
        path: The file, as :func:`collect_artifacts` or a figure scan returned it.
        max_bytes: The largest file read.

    Returns:
        The file's bytes.

    Raises:
        CollectedFileRefused: The file fails a rule; nothing was read.
    """
    fd = _open_regular(path.parent, path.name, max_bytes)
    with os.fdopen(fd, "rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise CollectedFileRefused(f"{path} holds more than {max_bytes} bytes")
    return data


def is_collectable_file(path: Path, *, max_bytes: int = MAX_COLLECTED_FILE_BYTES) -> bool:
    """Whether :func:`read_collected_file` would read *path*; never raises.

    Args:
        path: The file to test.
        max_bytes: The largest file read.

    Returns:
        True for a regular file with one hard link, at most *max_bytes* long,
        whose directory is not a link.
    """
    try:
        os.close(_open_regular(path.parent, path.name, max_bytes))
    except OSError:
        return False
    return True


def collect_artifacts(execution_folder: Path) -> list[dict]:
    """Read back what ``save_artifact()`` wrote during one execution.

    Returns one dict per manifest entry whose file can be collected, with
    ``path`` the file on disk; the caller reads it with
    :func:`read_collected_file` and registers it with the store. A missing or
    unreadable manifest, or one that is not a list, means nothing was saved; an
    entry that is not a mapping naming its ``filename`` is skipped.

    A file is collected only when its ``filename`` is a bare file name other
    than the manifest's, it sits directly in ``artifacts/``, ``artifacts/`` is
    not a link, and it is a regular file with one hard link. At most
    :data:`MAX_COLLECTED_ENTRIES` files and :data:`MAX_COLLECTED_TOTAL_BYTES`
    bytes are collected from one execution. Any other entry is skipped with a
    warning; none raises.
    """
    artifacts_dir = execution_folder / ARTIFACTS_DIRNAME
    try:
        raw = read_collected_file(artifacts_dir / MANIFEST_FILENAME, max_bytes=MAX_MANIFEST_BYTES)
    except CollectedFileRefused as refused:
        if not isinstance(refused.__cause__, FileNotFoundError):
            logger.warning("Artifact manifest not read: %s", refused)
        return []
    except OSError:
        logger.debug("Failed to read artifact manifest", exc_info=True)
        return []

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except Exception:
        logger.debug("Failed to read artifact manifest", exc_info=True)
        return []

    if not isinstance(manifest, list):
        return []

    contained_in = os.path.realpath(artifacts_dir)
    artifacts: list[dict] = []
    total_bytes = 0
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
            logger.debug("Skipping a malformed artifact manifest entry: %r", entry)
            continue
        filename = entry["filename"]
        if not _is_bare_filename(filename) or filename == MANIFEST_FILENAME:
            logger.warning("Artifact not collected: %r is not a file name in artifacts/", filename)
            continue
        file_path = artifacts_dir / filename
        if os.path.dirname(os.path.realpath(file_path)) != contained_in:
            logger.warning("Artifact not collected: %s is outside artifacts/", file_path)
            continue
        if not os.path.lexists(file_path):
            logger.debug("Artifact file missing: %s", file_path)
            continue
        if not is_collectable_file(file_path):
            logger.warning(
                "Artifact not collected: %s is not a regular file of artifacts/", file_path
            )
            continue
        if len(artifacts) >= MAX_COLLECTED_ENTRIES:
            logger.warning(
                "Artifact not collected: %s is past the %d files one run may hand over",
                file_path,
                MAX_COLLECTED_ENTRIES,
            )
            continue
        size = file_path.lstat().st_size
        if total_bytes + size > MAX_COLLECTED_TOTAL_BYTES:
            logger.warning(
                "Artifact not collected: %s would take the run past %d bytes",
                file_path,
                MAX_COLLECTED_TOTAL_BYTES,
            )
            continue
        total_bytes += size
        art = {
            "path": file_path,
            "title": entry.get("title", "Untitled"),
            "description": entry.get("description", ""),
            "artifact_type": entry.get("artifact_type", "file"),
            "mime_type": entry.get("mime_type", "application/octet-stream"),
        }
        if entry.get("category"):
            art["category"] = entry["category"]
        artifacts.append(art)

    return artifacts
