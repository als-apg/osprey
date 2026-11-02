"""What an application directory contributes to the profile fingerprint.

An application lives in ``apps/<name>/`` of a deployment repo, and its files —
``app.yml``, its fragments, its config — are build input the fingerprint folds.
The vendor files the build fetches into that directory are its output, not its
input, and are skipped. One home for both answers, so the build's material fold
and the staleness digests cannot disagree about which files count.
"""

from __future__ import annotations

import posixpath
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import yaml

from osprey.utils.workspace import APPS_DIRNAME
from osprey_connectors import yaml_loader

#: The directory inside an application that the build fetches vendor files
#: into by default, relative to ``apps/<name>/``.
_APP_PANEL_VENDOR_DIR = "panel/vendor/"


def app_build_output(app_dir: Path) -> Callable[[str], bool]:
    """Which files under ``apps/<name>/`` the build writes, as a predicate over their paths.

    ``panel/vendor/`` and every ``vendor[].into`` target the application's
    ``app.yml`` names are fetched by the build, so they are its output and never
    its input: folding them would move the fingerprint between a fresh clone and
    a built host, and between the two fingerprint reads of one build. The
    manifest is read as plain YAML, so an ``app.yml`` the loader would refuse
    still skips what it names.

    Args:
        app_dir: One application directory.

    Returns:
        A predicate over a file's posix path relative to *app_dir*.
    """
    from osprey.apps.loader import MANIFEST_FILE

    targets: set[str] = set()
    try:
        manifest = yaml_loader.safe_load((app_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        manifest = None
    vendor = manifest.get("vendor") if isinstance(manifest, dict) else None
    for entry in vendor if isinstance(vendor, list) else []:
        into = entry.get("into") if isinstance(entry, dict) else None
        if isinstance(into, str) and into:
            targets.add(PurePosixPath(posixpath.normpath(into)).as_posix())

    def is_output(rel: str) -> bool:
        return rel.startswith(_APP_PANEL_VENDOR_DIR) or rel in targets

    return is_output


def application_dirs(profile_dir: Path) -> list[Path]:
    """Every ``apps/<name>/`` directory of a profile root, sorted by name.

    Dot-prefixed entries and ``__pycache__`` are skipped, the walk the
    application loader makes.
    """
    apps_dir = profile_dir / APPS_DIRNAME
    if not apps_dir.is_dir():
        return []
    return [
        entry
        for entry in sorted(apps_dir.iterdir(), key=lambda entry: entry.name)
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != "__pycache__"
    ]
