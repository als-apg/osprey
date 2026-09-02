"""Every bundled service directory must be catalogued.

Guards against a new ``templates/services/<name>/`` directory shipping with a
compose template but no corresponding ``BuildArtifact`` entry — which would
silently drop it from ``osprey build`` / ``osprey scaffold list``.
"""

from __future__ import annotations

from pathlib import Path

from osprey.services.build_artifacts.catalog import BuildArtifactCatalog


def _service_dirs_with_compose_template(services_root: Path) -> set[str]:
    """Return names of directories directly under ``services_root`` that ship
    a ``docker-compose.yml.j2``."""
    return {
        child.name
        for child in services_root.iterdir()
        if child.is_dir() and (child / "docker-compose.yml.j2").exists()
    }


def test_every_bundled_service_directory_is_catalogued() -> None:
    """Every directory under templates/services/ with a compose template has a catalog entry.

    Deliberately one-directional: a catalog entry referencing a directory that
    does not yet exist on disk (e.g. mid-build-out) is not asserted against
    here — that is covered separately by the "template paths exist" check.
    """
    services_root = (
        Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "services"
    )
    shipped = _service_dirs_with_compose_template(services_root)

    catalog = BuildArtifactCatalog.default()
    catalogued_service_dirs = {
        artifact.template_path
        for artifact in catalog.all_artifacts()
        if artifact.template_root == "services" and artifact.is_directory
    }

    missing = shipped - catalogued_service_dirs
    assert not missing, (
        f"Service directories with a docker-compose.yml.j2 but no BuildArtifact catalog "
        f"entry: {sorted(missing)}"
    )
