"""Discover and validate a deployment's applications."""

from __future__ import annotations

from pathlib import Path

import yaml

from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import AppManifest, parse_manifest

APPS_DIR = "apps"
MANIFEST_FILE = "app.yml"


def load(deployment_root: Path) -> list[AppManifest]:
    """Load every ``apps/<name>/app.yml`` under ``deployment_root``, sorted by name.

    Only ``app.yml`` is read; other files in an application directory (its
    ``.provenance`` sidecar, fragments, code) are left alone. Hidden entries and
    ``__pycache__`` in ``apps/`` are skipped.

    Raises:
        AppManifestError: A manifest is refused, its ``name`` differs from its
            directory, a directory holds no ``app.yml``, or ``apps/`` holds a
            plain ``<name>.yml`` login file.
    """
    apps_dir = deployment_root / APPS_DIR
    if not apps_dir.is_dir():
        return []
    manifests: list[AppManifest] = []
    for entry in sorted(apps_dir.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_file():
            if entry.suffix in (".yml", ".yaml"):
                raise AppManifestError(
                    entry.stem,
                    "",
                    "plain_app_file",
                    f"{APPS_DIR}/{entry.name} is a login file; apps/ holds application "
                    "directories only. Run `osprey init --revalidate` to move it to tiers/",
                )
            continue
        manifest_path = entry / MANIFEST_FILE
        if not manifest_path.is_file():
            raise AppManifestError(
                entry.name,
                "",
                "missing_manifest",
                f"{APPS_DIR}/{entry.name}/ has no {MANIFEST_FILE}",
            )
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise AppManifestError(entry.name, "", "yaml", f"not valid YAML: {exc}") from None
        manifest = parse_manifest(data, app=entry.name)
        if manifest.name != entry.name:
            raise AppManifestError(
                entry.name,
                "name",
                "name_matches_directory",
                f"name {manifest.name!r} must equal the directory name {entry.name!r}",
            )
        manifests.append(manifest)
    return sorted(manifests, key=lambda m: m.name)
