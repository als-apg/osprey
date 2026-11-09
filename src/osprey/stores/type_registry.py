"""OSPREY Type Registry — single source of truth for artifact, data, and tool types.

Provides canonical definitions (key, label, color) for every type used in
the artifact gallery.  Python tools and the ``/api/type-registry`` endpoint
reference this module.  The gallery JS
fetches the registry at startup and overlays its lookup tables, so adding a
new type here is all that's needed — no JS or CSS changes required.

Icons (SVGs) remain in ``interfaces/artifacts/static/js/types.js`` where they
belong; this module only stores display metadata (label + color).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TypeDef:
    """A single type definition: display label + badge colour."""

    key: str
    label: str
    color: str  # hex colour for badges


# Artifact types
ARTIFACT_TYPES: dict[str, TypeDef] = {
    "plot_html": TypeDef("plot_html", "Plotly", "#8b5cf6"),
    "plot_png": TypeDef("plot_png", "Matplotlib", "#e8c9a0"),
    "table_html": TypeDef("table_html", "Table", "#22c55e"),
    "html": TypeDef("html", "HTML", "#c084fc"),
    "markdown": TypeDef("markdown", "Markdown", "#fbbf24"),
    "json": TypeDef("json", "JSON", "#3b82f6"),
    "image": TypeDef("image", "Image", "#a78bfa"),
    "text": TypeDef("text", "Text", "#8b9ab5"),
    "file": TypeDef("file", "File", "#506380"),
    # What `serialize_object` stamps on raw bytes. Distinct from "file", which
    # is a saved-from-disk artifact whose extension says what it holds: nothing
    # can be inferred about these bytes, and the published `save_artifact()`
    # contract names the type ("bytes -> binary file"), so remapping them onto
    # "file" would break a promise the docstring makes.
    "binary": TypeDef("binary", "Binary", "#64748b"),
    "notebook": TypeDef("notebook", "Notebook", "#e879f9"),
    "dashboard_html": TypeDef("dashboard_html", "Dashboard", "#06b6d4"),
}

# Categories — retired: nothing writes one any more. Kept read-only because an
# entry saved before the retirement still carries its ``category``, and the
# gallery badge (``types.js`` ``typeBadge``/``typeColor``) labels it from here.
CATEGORIES: dict[str, TypeDef] = {
    "archiver_data": TypeDef("archiver_data", "Archiver Data", "#2563eb"),
    "channel_values": TypeDef("channel_values", "Channel Values", "#14b8a6"),
    "write_results": TypeDef("write_results", "Write Results", "#e8c9a0"),
    "code_output": TypeDef("code_output", "Code Output", "#c084fc"),
    "visualization": TypeDef("visualization", "Visualization", "#fb923c"),
    "dashboard": TypeDef("dashboard", "Dashboard", "#06b6d4"),
    "document": TypeDef("document", "Document", "#a3e635"),
    "screenshot": TypeDef("screenshot", "Screenshot", "#a78bfa"),
    # The fallback when a submitting agent names no category. "Agent Response"
    # read as a content category, which it is not — every subagent artifact is
    # an agent response. The honest label makes an undeclared hand-in look like
    # what it is: something that forgot to say what it holds.
    "agent_response": TypeDef("agent_response", "Uncategorized", "#f472b6"),
    "channel_addresses": TypeDef("channel_addresses", "Channel Addresses", "#2dd4bf"),
    "channel_finder": TypeDef("channel_finder", "Channel Finder", "#10b981"),
    "logbook_research": TypeDef("logbook_research", "Logbook Research", "#e879f9"),
    "search_results": TypeDef("search_results", "Search Results", "#fb7185"),
    "notebook": TypeDef("notebook", "Notebook", "#d946ef"),
    "user_artifact": TypeDef("user_artifact", "User Artifact", "#94a3b8"),
    "diagnostic_report": TypeDef("diagnostic_report", "Diagnostic Report", "#ef4444"),
    "lattice_analysis": TypeDef("lattice_analysis", "Lattice Analysis", "#0ea5e9"),
    # Both facility-knowledge agents submit here — the document-bundle one and
    # the knowledge-graph one. They answer different questions from different
    # stores, but the operator asking them is after the same thing; telling the
    # two apart is source_agent's job, not the category's.
    "facility_knowledge": TypeDef("facility_knowledge", "Facility Knowledge", "#84cc16"),
}

# Tool types
TOOL_TYPES: dict[str, TypeDef] = {
    "channel_read": TypeDef("channel_read", "Channel Read", "#4fd1c5"),
    "channel_write": TypeDef("channel_write", "Channel Write", "#e8c9a0"),
    "archiver_read": TypeDef("archiver_read", "Archiver Read", "#3b82f6"),
    "execute": TypeDef("execute", "Python Execute", "#c084fc"),
    "channel_find": TypeDef("channel_find", "Channel Find", "#22c55e"),
    # memory_save and memory_recall removed — replaced by Claude Code native memory
    "ariel_search": TypeDef("ariel_search", "ARIEL Search", "#e879f9"),
    "attachment_to_artifact": TypeDef("attachment_to_artifact", "Logbook Picture", "#e879f9"),
    "screenshot_capture": TypeDef("screenshot_capture", "Screenshot Capture", "#a78bfa"),
    "facility_description": TypeDef("facility_description", "Facility Description", "#fbbf24"),
    "artifact_register": TypeDef("artifact_register", "Artifact Register", "#94a3b8"),
    "artifact_delete": TypeDef("artifact_delete", "Artifact Delete", "#94a3b8"),
    "artifact_export": TypeDef("artifact_export", "Artifact Export", "#94a3b8"),
    "artifact_focus": TypeDef("artifact_focus", "Artifact Focus", "#60a5fa"),
    "submit_response": TypeDef("submit_response", "Submit Response", "#f472b6"),
    "channel-finder": TypeDef("channel-finder", "Channel Finder", "#2dd4bf"),
    "logbook-deep-research": TypeDef("logbook-deep-research", "Logbook Deep Research", "#e879f9"),
    "logbook-search": TypeDef("logbook-search", "Logbook Search", "#f0abfc"),
    "create_static_plot": TypeDef("create_static_plot", "Static Plot", "#fb923c"),
    "create_interactive_plot": TypeDef("create_interactive_plot", "Interactive Plot", "#38bdf8"),
    "create_dashboard": TypeDef("create_dashboard", "Dashboard", "#06b6d4"),
    "create_document": TypeDef("create_document", "Document", "#a3e635"),
}


def get_artifact_types() -> dict[str, TypeDef]:
    """Return the canonical artifact type definitions."""
    return dict(ARTIFACT_TYPES)


def get_tool_types() -> dict[str, TypeDef]:
    """Return the canonical tool type definitions."""
    return dict(TOOL_TYPES)


def get_categories() -> dict[str, TypeDef]:
    """Return the legacy category labels an old entry's badge still renders."""
    return dict(CATEGORIES)


def _typedef_to_dict(td: TypeDef) -> dict[str, str]:
    """Serialise a TypeDef to a plain dict (label + color only)."""
    return {"label": td.label, "color": td.color}


def registry_to_api_dict() -> dict[str, Any]:
    """Return the full registry as a JSON-serialisable dict for the API.

    ``base_forms`` lists every base form with its label; ``extension_types``
    lists the registered extension types (loaded lazily on first use).
    ``viewer_url`` stays ``None`` here: serving a viewer is the artifacts app's
    route, which reads only :func:`viewer_path`.
    """
    from osprey.stores.artifact_forms import BASE_FORMS, form_label

    return {
        "artifact_types": {k: _typedef_to_dict(v) for k, v in ARTIFACT_TYPES.items()},
        "tool_types": {k: _typedef_to_dict(v) for k, v in TOOL_TYPES.items()},
        "categories": {k: _typedef_to_dict(v) for k, v in CATEGORIES.items()},
        "base_forms": [{"key": key, "label": form_label(key)} for key in BASE_FORMS],
        "extension_types": [
            {
                "key": ext.key,
                "label": ext.label,
                "base_form": ext.base_form,
                "viewer_url": None,
            }
            for ext in get_extension_types().values()
        ],
    }


# ---------------------------------------------------------------------------
# Extension types
# ---------------------------------------------------------------------------

#: Entry-point group an installed package declares extension types under.
ENTRY_POINT_GROUP = "osprey.artifact_types"

#: Suffixes a viewer module may carry.
VIEWER_SUFFIXES = frozenset({".js", ".mjs"})


@dataclass(frozen=True)
class ExtensionType:
    """A declared extension type: a ``kind`` that conforms to one base form.

    ``viewer`` is the resolved, contained path of the type's ES-module viewer,
    or ``None`` when it declares none (or declared one that did not resolve).
    """

    key: str
    label: str
    base_form: str
    viewer: Path | None = None


#: The process's extension registry; ``None`` until first use.
_EXTENSION_TYPES: dict[str, ExtensionType] | None = None


class _Declaration:
    """One raw declaration, with where it came from and its viewer root."""

    __slots__ = ("key", "spec", "source", "viewer_base", "viewer_root")

    def __init__(
        self,
        key: Any,
        spec: Any,
        source: str,
        viewer_base: Path | None,
        viewer_root: Path | None,
    ) -> None:
        self.key = key
        self.spec = spec
        self.source = source
        self.viewer_base = viewer_base
        self.viewer_root = viewer_root


def _spec_field(spec: Any, name: str) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(name)
    return getattr(spec, name, None)


def _resolve_viewer(raw: Any, base: Path | None, root: Path | None) -> tuple[Path | None, str]:
    """Resolve a declared viewer under Decision 2's containment rule.

    Returns ``(path, "")`` when the viewer resolves inside ``root`` to an
    existing ``.js``/``.mjs`` file, else ``(None, reason)``. Symlinks are
    followed before the containment check, so a link out of the tree fails.
    """
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        return None, f"viewer must be a non-empty path (got {raw!r})"
    if base is None or root is None:
        return None, "viewer has no declaring root to resolve against"
    try:
        resolved = (base / Path(raw)).resolve()
        root_resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"viewer {raw!r} does not resolve: {exc}"
    if not resolved.is_relative_to(root_resolved):
        return None, f"viewer {raw!r} resolves outside {root_resolved}"
    if resolved.suffix not in VIEWER_SUFFIXES:
        return None, f"viewer {raw!r} is not a .js/.mjs file"
    if not resolved.is_file():
        return None, f"viewer {raw!r} is missing"
    return resolved, ""


def _config_declarations(
    block: Any, project_root: Path | None, errors: list[str]
) -> list[_Declaration]:
    if block is None:
        return []
    if not isinstance(block, Mapping):
        errors.append(
            f"artifact_server.extension_types must be a mapping (got {type(block).__name__})"
        )
        return []
    return [
        _Declaration(key, spec, "config", project_root, project_root) for key, spec in block.items()
    ]


def _package_dirs(module_name: str) -> tuple[Path | None, Path | None]:
    """Directory of ``module_name``'s package, and of its top-level package."""
    try:
        module = importlib.import_module(module_name)
        top = importlib.import_module(module_name.split(".", 1)[0])
    except Exception:
        return None, None
    mod_file = getattr(module, "__file__", None)
    top_file = getattr(top, "__file__", None)
    if not mod_file or not top_file:
        return None, None
    base = Path(mod_file).parent
    root = Path(top_file).parent if hasattr(top, "__path__") else base
    return base, root


def _entry_point_declarations(eps: Iterable[Any] | None, errors: list[str]) -> list[_Declaration]:
    """Declarations from the ``osprey.artifact_types`` entry points.

    An entry point loads to a mapping ``{label, base_form, viewer?}`` (or an
    object with those attributes, e.g. an :class:`ExtensionType`), or to a
    zero-argument callable returning one; its key is the spec's ``key`` or,
    absent that, the entry point's name. A viewer is a package resource,
    relative to the entry point's module and contained in its top-level package.
    """
    if eps is None:
        from importlib.metadata import entry_points

        try:
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:
            errors.append(f"entry points {ENTRY_POINT_GROUP!r} could not be listed: {exc}")
            return []
    out: list[_Declaration] = []
    for ep in eps:
        name = getattr(ep, "name", repr(ep))
        try:
            spec = ep.load()
            if callable(spec) and not isinstance(spec, (Mapping, ExtensionType)):
                spec = spec()
        except Exception as exc:
            errors.append(f"extension type {name!r} (entry point) failed to load: {exc}")
            continue
        key = _spec_field(spec, "key") or name
        module_name = getattr(ep, "module", None) or str(getattr(ep, "value", "")).split(":")[0]
        base, root = _package_dirs(module_name) if module_name else (None, None)
        out.append(_Declaration(key, spec, f"entry point {name!r}", base, root))
    return out


def _read_config_block() -> tuple[Any, Path | None]:
    """(``artifact_server`` block, project root) from this process's config.

    A process whose ConfigBuilder was primed (both startup sites) reads that
    one; an unprimed process (lazy first use, e.g. the hub) reads the config the
    ordinary rule resolves (``OSPREY_CONFIG``, then the render under the cwd)
    without making it the process default.
    """
    from osprey_connectors import config as config_mod
    from osprey_connectors.workspace import resolve_config_path, resolve_project_root

    if getattr(config_mod, "_default_config", None) is not None:
        cb = config_mod.get_config_builder()
    else:
        path = resolve_config_path()
        if not path.is_file():
            return {}, None
        cb = config_mod.get_config_builder(config_path=str(path))
    raw = getattr(cb, "raw_config", None)
    block = cb.get("artifact_server", {})
    return block, resolve_project_root(raw if isinstance(raw, Mapping) else None)


def load_extension_types(
    *,
    validate: bool = False,
    config: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
    entry_points: Iterable[Any] | None = None,
) -> dict[str, ExtensionType]:
    """Load extension types from config and the ``osprey.artifact_types`` group.

    Sources are config ``artifact_server.extension_types: {key: {label,
    base_form, viewer?}}`` and the installed entry points. Config wins a key
    collision only by coming first; a later duplicate is a bad declaration.

    A bad declaration (unknown base form, a label colliding with a base form or
    another type, a duplicate or empty key, a non-mapping spec) is logged and
    skipped at runtime; a bad viewer (missing, not ``.js``/``.mjs``, outside its
    declaring root) drops only the viewer. With ``validate=True`` (``osprey
    build``) every problem, plus any ``artifact_server.categories``, is
    collected and raised as one :class:`ValueError` naming the offending keys.

    Args:
        validate: Raise instead of logging and skipping.
        config: A loaded config mapping to read instead of the global
            ConfigBuilder. When given, the result is returned without
            replacing the process registry.
        project_root: Containment root for config viewers; defaults to
            :func:`osprey_connectors.workspace.resolve_project_root`.
        entry_points: Entry points to read instead of the installed ones.

    Returns:
        The registry, ``{key: ExtensionType}``.
    """
    global _EXTENSION_TYPES

    from osprey.stores.artifact_forms import BASE_FORMS, form_label

    errors: list[str] = []
    install = config is None

    if config is None:
        try:
            block, root = _read_config_block()
        except Exception as exc:
            if validate:
                raise ValueError(f"extension types: config could not be read: {exc}") from exc
            logger.warning("Extension types: config could not be read (%s); none loaded", exc)
            block, root = {}, None
    else:
        block = config.get("artifact_server", {})
        root = None
        if project_root is None:
            from osprey_connectors.workspace import resolve_project_root

            root = resolve_project_root(config)
    if project_root is not None:
        root = Path(project_root)
    if not isinstance(block, Mapping):
        block = {}

    if block.get("categories"):
        errors.append(
            "artifact_server.categories is retired; declare artifact_server.extension_types instead"
        )

    declarations = _config_declarations(block.get("extension_types"), root, errors)
    declarations += _entry_point_declarations(entry_points, errors)

    reserved_labels = {form_label(f).casefold(): f for f in BASE_FORMS}
    reserved_labels.update({f.casefold(): f for f in BASE_FORMS})
    registry: dict[str, ExtensionType] = {}
    labels: dict[str, str] = {}

    for decl in declarations:
        key, spec, where = decl.key, decl.spec, decl.source
        if not isinstance(key, str) or not key.strip():
            errors.append(f"extension type {key!r} ({where}): key must be a non-empty string")
            continue
        if spec is None or isinstance(spec, (str, bytes, int, float, list, tuple)):
            errors.append(f"extension type {key!r} ({where}): declaration must be a mapping")
            continue
        if key in registry:
            errors.append(f"extension type {key!r} ({where}): key already declared")
            continue
        if key in BASE_FORMS:
            errors.append(f"extension type {key!r} ({where}): key is a base form")
            continue
        label = _spec_field(spec, "label")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"extension type {key!r} ({where}): label must be a non-empty string")
            continue
        folded = label.strip().casefold()
        if folded in reserved_labels:
            errors.append(
                f"extension type {key!r} ({where}): label {label!r} collides with "
                f"base form {reserved_labels[folded]!r}"
            )
            continue
        if folded in labels:
            errors.append(
                f"extension type {key!r} ({where}): label {label!r} collides with "
                f"extension type {labels[folded]!r}"
            )
            continue
        base_form = _spec_field(spec, "base_form")
        if base_form not in BASE_FORMS or base_form == "runs":
            errors.append(f"extension type {key!r} ({where}): unknown base form {base_form!r}")
            continue
        viewer: Path | None = None
        raw_viewer = _spec_field(spec, "viewer")
        if raw_viewer is not None:
            viewer, reason = _resolve_viewer(raw_viewer, decl.viewer_base, decl.viewer_root)
            if viewer is None:
                if validate:
                    errors.append(f"extension type {key!r} ({where}): {reason}")
                else:
                    logger.warning(
                        "Extension type %r (%s): dropping viewer: %s", key, where, reason
                    )
        registry[key] = ExtensionType(key, label, base_form, viewer)
        labels[folded] = key

    if errors:
        if validate:
            raise ValueError("Invalid extension types:\n  - " + "\n  - ".join(errors))
        for message in errors:
            logger.warning("Skipping %s", message)

    if install:
        _EXTENSION_TYPES = registry
        if registry:
            logger.info("Loaded %d extension type(s): %s", len(registry), ", ".join(registry))
    return dict(registry)


def get_extension_types() -> dict[str, ExtensionType]:
    """The process's extension registry, loaded on first use.

    The returned dict satisfies :func:`osprey.stores.artifact_forms.form_of`'s
    duck-typed registry (``.get(kind).base_form``). A load failure yields an
    empty registry rather than raising.
    """
    if _EXTENSION_TYPES is None:
        try:
            load_extension_types()
        except Exception as exc:  # runtime never raises
            logger.warning("Extension type loading failed (non-fatal): %s", exc)
            return {}
    return dict(_EXTENSION_TYPES or {})


def viewer_path(key: str) -> Path | None:
    """Resolved viewer module for extension type ``key``, or ``None``."""
    ext = get_extension_types().get(key)
    return ext.viewer if ext is not None else None


def reset_extension_types() -> None:
    """Forget the loaded registry, so the next use reloads (tests, reconfig)."""
    global _EXTENSION_TYPES
    _EXTENSION_TYPES = None
