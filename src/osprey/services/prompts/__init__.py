"""Prompt catalog and ownership helpers — shared service layer.

Re-exports the public API so callers can write::

    from osprey.services.prompts import PromptCatalog, PromptArtifact
    from osprey.services.prompts import get_user_owned
"""

from osprey.services.prompts.catalog import PromptArtifact, PromptCatalog
from osprey.services.prompts.ownership import (
    get_user_owned,
    update_config_add_user_owned,
    update_config_remove_user_owned,
    update_manifest_add_user_owned,
    update_manifest_remove_user_owned,
)

__all__ = [
    "PromptArtifact",
    "PromptCatalog",
    "get_user_owned",
    "update_config_add_user_owned",
    "update_config_remove_user_owned",
    "update_manifest_add_user_owned",
    "update_manifest_remove_user_owned",
]
