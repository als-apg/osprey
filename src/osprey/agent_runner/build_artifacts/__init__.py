"""Build artifact catalog and ownership helpers — the harness adapter's catalog of
what ``osprey build`` renders.

Re-exports the public API so callers can write::

    from osprey.agent_runner.build_artifacts import BuildArtifactCatalog, BuildArtifact
    from osprey.agent_runner.build_artifacts import get_user_owned
"""

from osprey.agent_runner.build_artifacts.catalog import BuildArtifact, BuildArtifactCatalog
from osprey.agent_runner.build_artifacts.ownership import (
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
