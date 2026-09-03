"""PyYAML safe loading on the libyaml scanner when it is installed.

``yaml.safe_load`` always runs PyYAML's pure-Python scanner, even when the
``yaml`` wheel shipped the libyaml bindings — the C scanner is only reached by
naming ``yaml.CSafeLoader`` explicitly. The two loaders share the constructor
and the resolver, so a document parses to the same objects either way; only the
scanning speed differs, by an order of magnitude. Every hot parse of a
deployment's YAML (``osprey build`` re-reads its rendered ``config.yml`` many
times per build) goes through :func:`safe_load` here rather than through
``yaml.safe_load`` directly.
"""

from __future__ import annotations

from typing import IO, Any

import yaml


def safe_loader() -> type[Any]:
    """The safe loader class to parse with: libyaml's when present, else PyYAML's.

    Resolved at call time rather than at import, so a build that runs without
    the C bindings falls back on the same code path a test can exercise by
    removing ``yaml.CSafeLoader``.
    """
    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def safe_load(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Parse one YAML document the way ``yaml.safe_load`` does, on the fastest loader.

    Args:
        stream: The document — a string, bytes, or an open file.

    Returns:
        The document as plain Python objects; ``None`` for an empty document.

    Raises:
        yaml.YAMLError: On a document that does not parse.
    """
    return yaml.load(stream, Loader=safe_loader())
