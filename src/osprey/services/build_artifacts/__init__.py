"""Build artifact catalog and ownership helpers — shared service layer.

Re-exports the public API so callers can write::

    from osprey.services.build_artifacts import BuildArtifactCatalog, BuildArtifact
    from osprey.services.build_artifacts import get_user_owned
"""

from osprey.services.build_artifacts.catalog import BuildArtifact, BuildArtifactCatalog
from osprey.services.build_artifacts.ownership import (
    get_user_owned,
    update_config_add_user_owned,
    update_config_remove_user_owned,
    update_manifest_add_user_owned,
    update_manifest_remove_user_owned,
)

__all__ = [
    "BuildArtifact",
    "BuildArtifactCatalog",
    "get_user_owned",
    "update_config_add_user_owned",
    "update_config_remove_user_owned",
    "update_manifest_add_user_owned",
    "update_manifest_remove_user_owned",
]
