"""Design-token model: load, flatten, and alias-resolve a DTCG token tree.

Loads the hand-authored DTCG (Design Tokens Community Group format) JSON
files under a ``tokens/`` source directory into a flat, dot-path-addressed
token model. Groups nest arbitrarily; a JSON object becomes a leaf *token*
once it defines ``$value``, otherwise it is a *group* whose non-``$`` keys
are recursed into. A group's own ``$type`` is inherited by descendant
tokens that do not declare their own ``$type``.

Alias references (``{group.path.to.primitive}``) are resolved exactly one
hop, and only against the primitives document (``core.json``). Aliases
that would require a second hop (the target is itself alias-shaped) or
that do not resolve to a primitive (dangling, or pointing at a semantic or
extension token instead) are left unresolved and annotated with an
:class:`AliasStatus` so ``generator/validate.py`` can reject them with a
precise, located error.

This module performs no semantic validation (theme completeness, color
syntax, WCAG contrast, namespace collisions, ...) — that is
``generator/validate.py``'s job. It only establishes the token model those
checks, and the CSS/JS emitters, operate on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "PRIMITIVES_FILENAME",
    "THEMES_DIRNAME",
    "INTERFACES_DIRNAME",
    "TokenModelError",
    "AliasStatus",
    "RawToken",
    "ResolvedToken",
    "TokenTree",
    "default_flagged_stem",
    "load_json_document",
    "flatten_document",
    "resolve_document",
    "load_token_tree",
]

#: Name of the primitives document directly under the tokens/ directory.
PRIMITIVES_FILENAME = "core.json"
#: Name of the subdirectory holding per-theme semantic-token documents.
THEMES_DIRNAME = "themes"
#: Name of the subdirectory holding per-interface extension-token documents.
INTERFACES_DIRNAME = "interfaces"

_ALIAS_PATTERN = re.compile(r"^\{([A-Za-z0-9_.-]+)\}$")


class TokenModelError(ValueError):
    """Raised for structural problems while loading a DTCG token source.

    Always carries the source file, and where applicable the dot-path
    within it, so callers can surface a precise location without
    re-deriving it from a bare exception message.

    Attributes:
        source_file: The token file the error was raised while processing.
        path: The dot-path within that file, or ``""`` if not applicable
            (e.g. a file-level read/parse error).
    """

    def __init__(self, message: str, *, source_file: Path, path: str = "") -> None:
        location = str(source_file) + (f" ({path})" if path else "")
        super().__init__(f"{location}: {message}")
        self.source_file = source_file
        self.path = path


class AliasStatus(StrEnum):
    """Classification of a resolved token's ``$value`` alias status."""

    #: ``$value`` is a literal, not an ``{alias.path}`` reference.
    NOT_ALIAS = "not_alias"
    #: ``$value`` referenced a primitive one hop away; resolved successfully.
    RESOLVED = "resolved"
    #: The alias target exists among the primitives but is itself
    #: alias-shaped, i.e. resolving it would take a second hop.
    MULTI_HOP = "multi_hop"
    #: The alias target exists somewhere in the tree, but not among the
    #: primitives (e.g. it points at a semantic or extension token).
    NOT_PRIMITIVE = "not_primitive"
    #: The alias target does not exist anywhere in the loaded tree.
    DANGLING = "dangling"


@dataclass(frozen=True)
class RawToken:
    """A single flattened DTCG token before alias resolution.

    Attributes:
        path: Dot-separated path from the document root, e.g.
            ``"color.teal.300"``.
        value: The raw ``$value`` exactly as authored: a literal, or an
            ``{alias.path}`` reference string.
        type: The token's ``$type``, own or inherited from an ancestor
            group; ``None`` if neither declares one.
        description: The token's ``$description``, if present.
        extensions: The token's ``$extensions`` mapping (empty if absent).
        source_file: The file this token was parsed from.
    """

    path: str
    value: Any
    type: str | None
    description: str | None
    extensions: dict[str, Any]
    source_file: Path


@dataclass(frozen=True)
class ResolvedToken:
    """A flattened DTCG token after one-hop alias resolution.

    Attributes:
        path: Dot-separated path from the document root.
        value: The literal value. Equal to the resolved primitive's value
            when ``alias_status`` is :attr:`AliasStatus.RESOLVED`; equal to
            the original raw ``$value`` for plain literals *and* for
            unresolved aliases (validate.py reports on those, it does not
            receive a resolved value for them).
        type: The token's ``$type``, own or inherited.
        description: The token's ``$description``, if present.
        extensions: The token's ``$extensions`` mapping.
        source_file: The file this token was parsed from.
        alias_status: Whether/how ``value`` relates to an alias reference.
        alias_target: The referenced dot-path, if the raw value was
            alias-shaped; ``None`` for plain literals.
    """

    path: str
    value: Any
    type: str | None
    description: str | None
    extensions: dict[str, Any]
    source_file: Path
    alias_status: AliasStatus
    alias_target: str | None

    @property
    def has_literal_value(self) -> bool:
        """Whether :attr:`value` is a usable literal to read or emit.

        True for a plain literal (:attr:`AliasStatus.NOT_ALIAS`) and for a
        successfully one-hop-resolved alias (:attr:`AliasStatus.RESOLVED`);
        False for every unresolved alias (dangling, multi-hop, or
        non-primitive), whose :attr:`value` is only the raw reference text.

        This is the single source of truth for the pipeline's core
        invariant — a tree that passes ``validate.assert_valid`` has this
        True for every token, so the emitters may rely on it. Both the
        validator (which skips value-reading checks when it is False) and
        the CSS emitter (which refuses to emit when it is False) consult
        this one predicate, so the two can never drift apart.
        """
        return self.alias_status in (AliasStatus.NOT_ALIAS, AliasStatus.RESOLVED)


@dataclass(frozen=True)
class TokenTree:
    """The fully loaded, alias-resolved token model for a ``tokens/`` tree.

    Attributes:
        primitives: Flattened, resolved primitives from ``core.json``,
            keyed by dot-path.
        themes: Flattened, resolved semantic tokens from
            ``themes/*.json``, keyed by theme file stem (e.g. ``"dark"``)
            then by dot-path.
        interfaces: Flattened, resolved extension tokens from
            ``interfaces/*.json``, keyed by interface file stem (e.g.
            ``"web_terminal"``) then by dot-path.
        theme_metadata: Each theme document's root-level ``$extensions``
            (e.g. ``{"mode": "dark", "id": "dark", "label": "Dark"}``),
            keyed by theme file stem.
        interface_metadata: Each interface document's root-level
            ``$extensions``, keyed by interface file stem.
    """

    primitives: dict[str, ResolvedToken]
    themes: dict[str, dict[str, ResolvedToken]]
    interfaces: dict[str, dict[str, ResolvedToken]]
    theme_metadata: dict[str, dict[str, Any]]
    interface_metadata: dict[str, dict[str, Any]]


def default_flagged_stem(tree: TokenTree) -> str | None:
    """Stem of the theme flagged ``$extensions.default: true``, if any.

    The single source for "which theme is the product default" — both the
    CSS emitter (the ``:root`` fallback theme) and the JS emitters
    (``DEFAULT_FAMILY``) resolve their defaults from this flag, so sharing
    the lookup is what guarantees the two generated artifacts agree.
    ``validate.py``'s ``check_default_flag`` enforces that at most one
    theme carries the flag (and that it is a dark theme), so the
    first-in-metadata-order tiebreak here is unreachable on a valid tree.

    Args:
        tree: The loaded token tree.

    Returns:
        The flagged theme's stem, or ``None`` when no theme is flagged.
    """
    for stem, metadata in tree.theme_metadata.items():
        if metadata.get("default") is True:
            return stem
    return None


def load_json_document(path: Path) -> dict[str, Any]:
    """Read and parse a single DTCG JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON document.

    Raises:
        TokenModelError: If the file cannot be read, is not valid JSON, or
            its top-level structure is not a JSON object.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TokenModelError(f"could not read token file: {exc}", source_file=path) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TokenModelError(f"invalid JSON: {exc}", source_file=path) from exc
    if not isinstance(document, dict):
        raise TokenModelError("top-level token document must be a JSON object", source_file=path)
    return document


def flatten_document(document: dict[str, Any], *, source_file: Path) -> dict[str, RawToken]:
    """Flatten a parsed DTCG document into dot-path-addressed raw tokens.

    Recurses through nested groups (objects without ``$value``); each
    object that defines ``$value`` is a leaf token. A group's own
    ``$type`` is inherited by descendant tokens that do not declare their
    own ``$type``. Other root/group-level ``$``-prefixed keys (such as
    ``$extensions`` describing an entire theme document) are not tokens
    and are ignored here.

    Args:
        document: The parsed JSON document, as returned by
            :func:`load_json_document`.
        source_file: The origin file, recorded on every token and used in
            any error raised while flattening.

    Returns:
        A mapping from dot-path to :class:`RawToken`, in document order.

    Raises:
        TokenModelError: If a node defines both ``$value`` and nested
            group children (structurally ambiguous), if ``$value`` appears
            at the document root, if ``$type``/``$description`` is present
            with the wrong type, if ``$extensions`` is present but not an
            object, or if a group child is not itself a JSON object.
    """
    tokens: dict[str, RawToken] = {}
    _flatten_node(document, prefix=(), inherited_type=None, source_file=source_file, out=tokens)
    return tokens


def _flatten_node(
    node: dict[str, Any],
    *,
    prefix: tuple[str, ...],
    inherited_type: str | None,
    source_file: Path,
    out: dict[str, RawToken],
) -> None:
    """Recursively flatten one DTCG group/token node into ``out``."""
    path = ".".join(prefix)
    child_keys = [key for key in node if not key.startswith("$")]
    is_leaf = "$value" in node

    if is_leaf and not prefix:
        raise TokenModelError(
            "a bare $value cannot appear at the document root", source_file=source_file
        )
    if is_leaf and child_keys:
        raise TokenModelError(
            f"token defines both $value and nested group keys {child_keys!r}",
            source_file=source_file,
            path=path,
        )
    if "$type" in node and not isinstance(node["$type"], str):
        raise TokenModelError("$type must be a string", source_file=source_file, path=path)
    node_type = node.get("$type", inherited_type)

    if is_leaf:
        description = node.get("$description")
        if description is not None and not isinstance(description, str):
            raise TokenModelError(
                "$description must be a string", source_file=source_file, path=path
            )
        extensions = node.get("$extensions", {})
        if not isinstance(extensions, dict):
            raise TokenModelError(
                "$extensions must be an object", source_file=source_file, path=path
            )
        out[path] = RawToken(
            path=path,
            value=node["$value"],
            type=node_type,
            description=description,
            extensions=dict(extensions),
            source_file=source_file,
        )
        return

    for key in child_keys:
        child = node[key]
        child_path = ".".join((*prefix, key))
        if not isinstance(child, dict):
            raise TokenModelError(
                f"expected an object for group/token {key!r}, got {type(child).__name__}",
                source_file=source_file,
                path=child_path,
            )
        _flatten_node(
            child,
            prefix=(*prefix, key),
            inherited_type=node_type,
            source_file=source_file,
            out=out,
        )


def resolve_document(
    raw_tokens: dict[str, RawToken],
    primitives: dict[str, RawToken],
    all_known_paths: set[str],
) -> dict[str, ResolvedToken]:
    """Resolve one-hop ``{alias.path}`` references against primitives.

    A token's ``$value`` is alias-shaped if it is a string of the exact
    form ``{path.to.token}``. Resolution succeeds only when the referenced
    path exists in ``primitives`` *and* that primitive's own value is not
    itself alias-shaped (which would require a second hop). Every other
    case is left unresolved and classified via :class:`AliasStatus` for
    ``validate.py`` to reject.

    Args:
        raw_tokens: The tokens to resolve, as returned by
            :func:`flatten_document`.
        primitives: The primitives document's raw tokens — the only valid
            alias target. Pass ``raw_tokens`` itself when resolving the
            primitives document, so intra-primitive one-hop aliases (e.g.
            a semantic alias primitive referencing a color-ramp primitive)
            are supported the same way as any other document.
        all_known_paths: The union of every token dot-path across the
            whole tree (primitives, all themes, all interfaces), used only
            to distinguish a truly dangling reference from one that points
            at a real but non-primitive token.

    Returns:
        A mapping from dot-path to :class:`ResolvedToken`, one entry per
        input raw token, in the same order.
    """
    resolved: dict[str, ResolvedToken] = {}
    for path, raw in raw_tokens.items():
        alias_target = _alias_target(raw.value)

        if alias_target is None:
            resolved[path] = ResolvedToken(
                path=raw.path,
                value=raw.value,
                type=raw.type,
                description=raw.description,
                extensions=raw.extensions,
                source_file=raw.source_file,
                alias_status=AliasStatus.NOT_ALIAS,
                alias_target=None,
            )
            continue

        primitive = primitives.get(alias_target)
        if primitive is None:
            status = (
                AliasStatus.NOT_PRIMITIVE
                if alias_target in all_known_paths
                else AliasStatus.DANGLING
            )
            value = raw.value
        elif _alias_target(primitive.value) is not None:
            status = AliasStatus.MULTI_HOP
            value = raw.value
        else:
            status = AliasStatus.RESOLVED
            value = primitive.value

        resolved[path] = ResolvedToken(
            path=raw.path,
            value=value,
            type=raw.type,
            description=raw.description,
            extensions=raw.extensions,
            source_file=raw.source_file,
            alias_status=status,
            alias_target=alias_target,
        )
    return resolved


def _alias_target(value: Any) -> str | None:
    """Return the referenced dot-path if ``value`` is an ``{alias.path}`` string."""
    if not isinstance(value, str):
        return None
    match = _ALIAS_PATTERN.match(value)
    return match.group(1) if match else None


def _root_extensions(document: dict[str, Any], *, source_file: Path) -> dict[str, Any]:
    """Extract and validate a document's root-level ``$extensions`` mapping."""
    extensions = document.get("$extensions", {})
    if not isinstance(extensions, dict):
        raise TokenModelError("$extensions must be an object", source_file=source_file)
    return dict(extensions)


def _load_raw_documents(directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Read and parse every ``*.json`` file directly under ``directory``.

    Args:
        directory: A directory such as ``tokens/themes`` or
            ``tokens/interfaces``. A missing directory yields an empty
            mapping (e.g. an interface with no extension tokens yet).

    Returns:
        A mapping from file stem (e.g. ``"dark"``) to its path and parsed
        JSON document, ordered by filename.
    """
    if not directory.is_dir():
        return {}
    return {
        path.stem: (path, load_json_document(path)) for path in sorted(directory.glob("*.json"))
    }


def load_token_tree(tokens_dir: Path) -> TokenTree:
    """Load, flatten, and alias-resolve an entire ``tokens/`` source directory.

    Expects ``tokens_dir`` to contain a primitives document
    (:data:`PRIMITIVES_FILENAME`), a ``themes/`` subdirectory of theme
    documents, and an ``interfaces/`` subdirectory of interface-extension
    documents.

    Args:
        tokens_dir: The ``tokens/`` source directory.

    Returns:
        The fully loaded :class:`TokenTree`, with every alias one-hop
        resolved against primitives where possible.

    Raises:
        TokenModelError: If the primitives file is missing, or any
            document fails to parse or flatten (see
            :func:`load_json_document` and :func:`flatten_document`).
    """
    primitives_path = tokens_dir / PRIMITIVES_FILENAME
    if not primitives_path.is_file():
        raise TokenModelError(
            f"primitives file not found: {primitives_path}", source_file=primitives_path
        )
    primitives_document = load_json_document(primitives_path)
    primitives_raw = flatten_document(primitives_document, source_file=primitives_path)

    themes_documents = _load_raw_documents(tokens_dir / THEMES_DIRNAME)
    interfaces_documents = _load_raw_documents(tokens_dir / INTERFACES_DIRNAME)

    themes_raw = {
        stem: flatten_document(document, source_file=path)
        for stem, (path, document) in themes_documents.items()
    }
    interfaces_raw = {
        stem: flatten_document(document, source_file=path)
        for stem, (path, document) in interfaces_documents.items()
    }

    all_known_paths: set[str] = set(primitives_raw)
    for document_tokens in (*themes_raw.values(), *interfaces_raw.values()):
        all_known_paths.update(document_tokens)

    primitives = resolve_document(primitives_raw, primitives_raw, all_known_paths)
    themes = {
        stem: resolve_document(doc, primitives_raw, all_known_paths)
        for stem, doc in themes_raw.items()
    }
    interfaces = {
        stem: resolve_document(doc, primitives_raw, all_known_paths)
        for stem, doc in interfaces_raw.items()
    }
    theme_metadata = {
        stem: _root_extensions(document, source_file=path)
        for stem, (path, document) in themes_documents.items()
    }
    interface_metadata = {
        stem: _root_extensions(document, source_file=path)
        for stem, (path, document) in interfaces_documents.items()
    }

    return TokenTree(
        primitives=primitives,
        themes=themes,
        interfaces=interfaces,
        theme_metadata=theme_metadata,
        interface_metadata=interface_metadata,
    )
