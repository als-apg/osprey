"""Alarm replays packaged with OSPREY, named in a config as ``demo:<name>``.

``alarm_state.replay_file`` is either a path relative to the project directory or
``demo:<name>``, the packaged ``<name>.ndjson`` beside this module. Only the
framework reads the ``demo:`` spelling: the replay backend takes the path this
module resolves.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__all__ = ["DEMO_PREFIX", "resolve_replay_file"]

#: The prefix naming a packaged replay instead of a project file.
DEMO_PREFIX = "demo:"

_SUFFIX = ".ndjson"


def _packaged_dir() -> Path:
    """The directory holding the packaged replays."""
    return Path(str(files(__name__)))


def resolve_replay_file(value: str, project_dir: Path | None) -> Path:
    """The file *value* names, whether or not it exists.

    Args:
        value: ``demo:<name>`` or a path; a relative path is against *project_dir*.
        project_dir: The rendered project's directory, or ``None`` to leave a
            relative path relative.

    Raises:
        ValueError: *value* is ``demo:`` with a name that is empty or holds a
            path separator.
    """
    if value.startswith(DEMO_PREFIX):
        name = value[len(DEMO_PREFIX) :]
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"{value!r} names no packaged replay")
        return _packaged_dir() / f"{name}{_SUFFIX}"
    path = Path(value)
    return path if project_dir is None else Path(project_dir) / path
