"""OSPREY storage infrastructure.

Cross-cutting storage layer used by MCP servers, interfaces, and hooks.
Provides file-backed artifact storage with JSON indexing and a type registry
for artifact/tool metadata, extension types, and the legacy
category labels old entries' badges still render.

Modules:
    base_store — Generic file-backed indexed store
    artifact_store — Artifact storage singleton
    type_registry — Canonical type definitions (artifact, tool, extension types)
    notebook_renderer — Jupyter notebook creation and HTML rendering
"""

from osprey.stores.artifact_store import (
    ArtifactEntry,
    ArtifactStore,
    get_artifact_store,
    initialize_artifact_store,
    register_artifact_listener,
    reset_artifact_store,
    unregister_artifact_listener,
)
from osprey.stores.base_store import BaseStore
from osprey.stores.notebook_renderer import (
    create_notebook_from_code,
    get_or_render_html,
    render_notebook_to_html,
)
from osprey.stores.type_registry import (
    ExtensionType,
    TypeDef,
    get_artifact_types,
    get_categories,
    get_extension_types,
    get_tool_types,
    load_extension_types,
    registry_to_api_dict,
    viewer_path,
)

__all__ = [
    "ArtifactEntry",
    "ArtifactStore",
    "BaseStore",
    "ExtensionType",
    "TypeDef",
    "create_notebook_from_code",
    "get_artifact_store",
    "get_artifact_types",
    "get_categories",
    "get_extension_types",
    "get_or_render_html",
    "get_tool_types",
    "initialize_artifact_store",
    "load_extension_types",
    "register_artifact_listener",
    "registry_to_api_dict",
    "render_notebook_to_html",
    "reset_artifact_store",
    "unregister_artifact_listener",
    "viewer_path",
]
