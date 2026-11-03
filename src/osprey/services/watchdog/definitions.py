"""Load watchdog definitions: the build's enabled ones, or a profile's whole block.

Both readers go through :func:`~osprey.services.watchdog.config.parse_watchdogs_block`,
and differ in what they answer. At run time :func:`load_watchdogs` reads the rendered
``config.yml`` map with its files copied under the build directory; the build
already validated it, so a file that has since gone costs that one watchdog and a
warning while the others keep watching, and any other fault is a broken build and
is refused. :func:`load_profile_watchdogs` reads a profile's block with ``enabled``
ignored, because a watchdog is replayed before it is enabled.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.services.watchdog.config import (
    INSTRUCTIONS_FILENAME,
    WATCHDOG_NAME_RE,
    WATCHDOGS_BLOCK,
    WatchdogDefinition,
    instructions_error,
    parse_watchdogs_block,
)

logger = logging.getLogger(__name__)

#: Where a render sits under a repo. A rendered project, a container's among them,
#: holds its config at its root.
_CONFIG_CANDIDATES = ("build/config.yml", "config.yml")


def load_rendered_config(root: Path) -> tuple[dict[str, Any], Path] | None:
    """The rendered config under *root* and the build directory holding it, if it has one.

    ``OSPREY_CONFIG`` is set to the same file, so everything this process resolves
    from the config (the state directory among them) answers for this project.
    """
    for candidate in _CONFIG_CANDIDATES:
        config_path = (root / candidate).resolve()
        if config_path.is_file():
            os.environ["OSPREY_CONFIG"] = str(config_path)
            from osprey_connectors.config import get_config_builder

            builder = get_config_builder(config_path=str(config_path), set_as_default=True)
            return builder.raw_config, config_path.parent
    return None


def _loadable_entries(raw: Mapping[Any, Any], build_dir: Path) -> dict[Any, Any]:
    """The entries to parse: not disabled, and not missing their files.

    An entry that is malformed stays in, so the parser refuses it; only a
    well-named, enabled entry whose instructions are unavailable is dropped.
    """
    entries: dict[Any, Any] = {}
    for name, entry in raw.items():
        if isinstance(entry, Mapping) and entry.get("enabled", False) is False:
            continue
        if (
            isinstance(name, str)
            and WATCHDOG_NAME_RE.fullmatch(name)
            and instructions_error(name, build_dir) is not None
        ):
            logger.warning(
                "Watchdog %r is not running: %s has no readable, non-empty %s",
                name,
                build_dir / WATCHDOGS_BLOCK / name,
                INSTRUCTIONS_FILENAME,
            )
            continue
        entries[name] = entry
    return entries


def load_watchdogs(config: Mapping[str, Any], build_dir: Path) -> tuple[WatchdogDefinition, ...]:
    """Load the watchdogs a build runs.

    Args:
        config: The build's parsed ``config.yml``.
        build_dir: The build directory holding ``watchdogs/<name>/``.

    Returns:
        One definition per enabled watchdog whose files are present, in block order.

    Raises:
        BuildProfileError: If the rendered block breaks a rule other than its
            files being present.
    """
    raw = config.get(WATCHDOGS_BLOCK)
    if isinstance(raw, Mapping):
        raw = _loadable_entries(raw, build_dir)
    return parse_watchdogs_block(raw, base_dir=build_dir)


def load_profile_watchdogs(profile_dir: Path) -> tuple[WatchdogDefinition, ...]:
    """Load every watchdog a profile declares, enabled or not.

    Only the block is read: its inheritance is resolved, and the rest of the
    profile need not be ready to build for its watchdogs to be replayed.

    Args:
        profile_dir: The directory holding ``profile.yml``.

    Returns:
        One definition per entry in the resolved block, in block order.

    Raises:
        BuildProfileError: If the profile is absent or unreadable, or the block
            breaks a rule, a missing file included.
    """
    from osprey.cli.build_profile_document import _read_profile_document
    from osprey.cli.build_profile_merge import resolve_profile_document
    from osprey.cli.repo_resolver import PROFILE_FILENAME
    from osprey_connectors.errors import BuildProfileError

    path = profile_dir / PROFILE_FILENAME
    if not path.exists():
        raise BuildProfileError(f"Profile not found: {path}")
    raw = _read_profile_document(path)
    if not isinstance(raw, dict):
        raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")
    document = resolve_profile_document(raw, path)
    return parse_watchdogs_block(document.raw.get(WATCHDOGS_BLOCK), base_dir=document.root_dir)
