"""Provider catalog loading — the ``providers.yml`` sibling of ``profile.yml``.

The catalog is the deployment's list of model providers: the entries the build
renders into ``api.providers``, and the names ``provider:`` in ``profile.yml``
may pick from. It lives beside the profile rather than inside it because it is a
catalog of endpoints, not a facility decision — one operator edit adds a
gateway, and the packaged copy keeps receiving framework refreshes for everyone
who never touched it.

Resolution is a *replacement*, not a merge: a ``providers.yml`` next to the
profile is the whole catalog, and the packaged copy is used only when the repo
has none. Merging would make a deleted entry come back, which is exactly the
edit an operator removing a decommissioned gateway is trying to make.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from osprey.errors import BuildProfileError

#: Filename of the catalog, both packaged and in a deployment repo.
PROVIDERS_FILENAME = "providers.yml"

#: Top-level key holding the entries. The file is a mapping so it has room for
#: future file-level keys without a provider named after one of them.
_PROVIDERS_KEY = "providers"

#: The two wire protocols an entry may declare. The framework compares against
#: ``"anthropic"`` exactly, so a misspelling — ``Anthropic``, ``antropic`` —
#: reads as "not Anthropic" and silently inserts the translation hop the config
#: template warns about. Two values, checked as an enum where the catalog is
#: read; :mod:`osprey.infrastructure.proxy.lifecycle` imports this set rather
#: than keeping a second copy of it.
VALID_API_PROTOCOLS = frozenset({"anthropic", "openai"})

#: Keys the contract documents. ``base_url`` is the only required one — an entry
#: without it names no endpoint, so nothing downstream can call it. ``api_key``,
#: ``models`` and ``api_protocol`` are optional, and an entry may carry further
#: keys the framework renders through without reading.
_REQUIRED_ENTRY_KEYS = ("base_url",)


@dataclass(frozen=True)
class ProviderCatalog:
    """A resolved provider catalog and where it came from.

    Attributes:
        entries: Provider name → entry mapping, as spelled in the file.
        source: ``"repo"`` when a ``providers.yml`` beside the profile supplied
            it, ``"packaged"`` when the framework's own copy did.
        path: The file the entries were read from. Build output names it, and
            :func:`compute_providers_hash` stamps provenance from it.
    """

    entries: dict[str, Any]
    source: Literal["repo", "packaged"]
    path: Path


def packaged_catalog_path() -> Path:
    """Path of the catalog shipped inside the ``osprey.profiles`` package."""
    return Path(str(importlib.resources.files("osprey.profiles"))) / PROVIDERS_FILENAME


def load_provider_catalog(profile_dir: Path | None) -> ProviderCatalog:
    """Resolve the provider catalog for a profile directory.

    Args:
        profile_dir: Directory holding ``profile.yml``. A ``providers.yml``
            there REPLACES the packaged catalog. ``None`` (or a directory with
            no catalog) resolves to the packaged copy.

    Returns:
        The resolved :class:`ProviderCatalog`.

    Raises:
        BuildProfileError: The file is unreadable, is not a mapping, has no
            ``providers:`` mapping, or holds a malformed entry. The message
            names the file and the offending key.
    """
    path = None
    source: Literal["repo", "packaged"] = "packaged"
    if profile_dir is not None:
        candidate = Path(profile_dir) / PROVIDERS_FILENAME
        if candidate.is_file():
            path, source = candidate, "repo"
    if path is None:
        path = packaged_catalog_path()
    return ProviderCatalog(entries=_read_entries(path), source=source, path=path)


def compute_providers_hash(path: Path) -> str:
    """Content hash of a resolved provider catalog.

    Stamped into the build manifest as ``provenance.providers_hash``. Hashes the
    *entries*, not the file bytes, so rewording a comment does not read as a
    catalog change while an added, removed, or re-pointed provider does. Same
    spelling as :func:`~osprey.cli.build_profile_merge.compute_preset_hash`, so
    the two hashes sit comparably beside each other in provenance.

    Args:
        path: The catalog file to hash — normally ``ProviderCatalog.path``.

    Returns:
        ``"sha256:<hexdigest>"``.

    Raises:
        BuildProfileError: The catalog at ``path`` is missing or malformed.
    """
    canonical = json.dumps(_read_entries(path), sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _read_entries(path: Path) -> dict[str, Any]:
    """Read and validate the entries mapping out of a catalog file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildProfileError(f"Provider catalog not found: {path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise BuildProfileError(f"Cannot read provider catalog {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise BuildProfileError(
            f"Provider catalog {path} must be a YAML mapping with a "
            f"`{_PROVIDERS_KEY}:` key, got {type(raw).__name__}."
        )
    if _PROVIDERS_KEY not in raw:
        raise BuildProfileError(f"Provider catalog {path} has no `{_PROVIDERS_KEY}:` key.")

    entries = raw[_PROVIDERS_KEY]
    if not isinstance(entries, dict):
        raise BuildProfileError(
            f"Provider catalog {path}: `{_PROVIDERS_KEY}:` must be a mapping of "
            f"provider name to entry, got {type(entries).__name__}."
        )

    for name, entry in entries.items():
        _validate_entry(path, str(name), entry)
    return entries


def _validate_entry(path: Path, name: str, entry: Any) -> None:
    """Validate one catalog entry, naming the file and the key on refusal."""
    if not isinstance(entry, dict):
        raise BuildProfileError(
            f"Provider catalog {path}: `{_PROVIDERS_KEY}.{name}` must be a mapping, "
            f"got {type(entry).__name__}."
        )
    for key in _REQUIRED_ENTRY_KEYS:
        if key not in entry:
            raise BuildProfileError(
                f"Provider catalog {path}: `{_PROVIDERS_KEY}.{name}` is missing "
                f"required key `{key}`."
            )
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise BuildProfileError(
                f"Provider catalog {path}: `{_PROVIDERS_KEY}.{name}.{key}` must be a "
                f"non-empty string."
            )
    # Keys beyond the documented contract are passed through untouched rather
    # than refused: the entries are rendered verbatim into `api.providers`, and
    # an operator's gateway may legitimately carry a key this framework version
    # does not read yet.
    protocol = entry.get("api_protocol")
    if protocol is not None and protocol not in VALID_API_PROTOCOLS:
        raise BuildProfileError(
            f"Provider catalog {path}: `{_PROVIDERS_KEY}.{name}.api_protocol` is "
            f"{protocol!r}; expected one of {', '.join(sorted(VALID_API_PROTOCOLS))}."
        )
    models = entry.get("models")
    if models is not None and not isinstance(models, dict):
        raise BuildProfileError(
            f"Provider catalog {path}: `{_PROVIDERS_KEY}.{name}.models` must be a "
            f"mapping of tier to model ID, got {type(models).__name__}."
        )
