"""Artifact library: discover, validate, and resolve built-in Claude Code artifacts.

Built-in artifacts live under osprey/templates/claude_code/claude/:
  hooks/         — Python scripts (.py files); short name = filename stem
  rules/         — Markdown files (.md, .md.j2); short name = stem without .j2
  skills/        — Subdirectories; short name = directory name
  agents/        — Markdown files (.md.j2); short name = stem without .j2
  output-styles/ — Markdown files (.md.j2); short name = stem without .j2
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger("osprey.cli.templates")

# Artifact type → subdirectory name under claude_code/claude/
_TYPE_TO_SUBDIR: dict[str, str] = {
    "hooks": "hooks",
    "rules": "rules",
    "skills": "skills",
    "agents": "agents",
    "output_styles": "output-styles",
}

ARTIFACT_TYPES = list(_TYPE_TO_SUBDIR.keys())


def _claude_code_root() -> Path:
    """Return path to the built-in claude_code template directory."""
    try:
        import osprey.templates

        base = Path(osprey.templates.__file__).parent
        candidate = base / "claude_code"
        if candidate.is_dir():
            return candidate
    except (ImportError, AttributeError):
        pass

    # Development fallback: this file is at src/osprey/cli/templates/artifact_library.py
    fallback = Path(__file__).parent.parent.parent / "templates" / "claude_code"
    if fallback.is_dir():
        return fallback

    raise RuntimeError(
        "Cannot locate the osprey built-in claude_code template directory. "
        "Ensure osprey is properly installed."
    )


def _artifact_dir(artifact_type: str) -> Path:
    """Return the directory for a given artifact type."""
    subdir = _TYPE_TO_SUBDIR.get(artifact_type)
    if subdir is None:
        raise ValueError(
            f"Unknown artifact type '{artifact_type}'. "
            f"Valid types: {', '.join(ARTIFACT_TYPES)}"
        )
    return _claude_code_root() / "claude" / subdir


def _hook_short_name(stem: str) -> str:
    """Normalize a hook file stem to a user-facing short name.

    Hook files are named with an ``osprey_`` prefix and underscores
    (e.g. ``osprey_approval.py``, ``osprey_hook_log.py``).  Profile authors
    use the shorter hyphenated form without the prefix (e.g. ``approval``,
    ``hook-log``).  This function applies that normalization.

    Examples::
        osprey_approval     → approval
        osprey_hook_log     → hook-log
        osprey_writes_check → writes-check
    """
    if stem.startswith("osprey_"):
        stem = stem[len("osprey_"):]
    return stem.replace("_", "-")


def parse_hook_frontmatter(hook_path: Path) -> dict | None:
    """Extract YAML frontmatter from a hook file's docstring.

    Hook files embed metadata in a YAML block delimited by ``---`` inside
    the module docstring.  This function parses that block and returns the
    metadata for hooks that declare ``wiring: standalone``.

    Returns:
        A dict with keys ``name``, ``event``, ``tools``, ``safety_layer``,
        ``timeout``, ``wiring`` — or ``None`` for utility hooks (no
        frontmatter), hooks without an ``event`` field, or hooks that do
        not declare ``wiring: standalone``.
    """
    try:
        text = hook_path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Find the first docstring block
    start = text.find('"""')
    if start < 0:
        return None
    end = text.find('"""', start + 3)
    if end < 0:
        return None
    docstring = text[start + 3 : end]

    # Extract content between --- delimiters
    parts = docstring.split("---")
    if len(parts) < 3:
        return None
    yaml_block = parts[1]

    try:
        meta = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        logger.warning("Malformed YAML frontmatter in %s", hook_path)
        return None

    if not isinstance(meta, dict):
        return None

    # Must have event + tools fields and wiring: standalone
    if "event" not in meta:
        return None
    if "tools" not in meta:
        logger.warning("Hook %s has wiring: standalone but no tools: field — skipping", hook_path)
        return None
    if meta.get("wiring") != "standalone":
        return None

    # Apply defaults
    meta.setdefault("timeout", 5)
    meta.setdefault("safety_layer", 99)
    return meta


def _short_name_from_path(path: Path, artifact_type: str) -> str | None:
    """Derive the short name from a path in the artifact directory.

    Returns None if the path should be excluded (e.g. __pycache__, _ prefixed dirs).
    """
    if path.name.startswith("_") or path.name == "__pycache__":
        return None

    if artifact_type == "skills":
        # Skills are directories; the directory name is the short name
        if path.is_dir():
            return path.name
        return None
    else:
        # All other types are files
        if not path.is_file():
            return None
        if artifact_type == "hooks":
            # Hook Python scripts: strip osprey_ prefix and convert underscores
            # to hyphens so profiles use readable short names (e.g. "approval").
            if path.suffix == ".py":
                return _hook_short_name(path.stem)
            # Hook JSON config template: hook_config.json.j2 → hook-config
            if path.name.endswith(".json.j2"):
                stem = path.name[: -len(".json.j2")]
                return stem.replace("_", "-")
        else:
            # rules, agents, output_styles: .md or .md.j2
            name = path.name
            if name.endswith(".md.j2"):
                return name[: -len(".md.j2")]
            if name.endswith(".md"):
                return name[: -len(".md")]
    return None


def list_artifacts(artifact_type: str) -> list[str]:
    """List available artifact short names for a given type.

    Args:
        artifact_type: One of "hooks", "rules", "skills", "agents", "output_styles"

    Returns:
        Sorted list of short names available in the built-in library.

    Raises:
        ValueError: If artifact_type is not recognised.
    """
    directory = _artifact_dir(artifact_type)
    names: list[str] = []
    for path in directory.iterdir():
        name = _short_name_from_path(path, artifact_type)
        if name is not None:
            names.append(name)
    return sorted(names)


def validate_artifacts(artifacts: dict[str, list[str]]) -> None:
    """Validate that all requested artifact short names exist in the library.

    Args:
        artifacts: Mapping of artifact_type → list of short names to validate.
            Example: {"hooks": ["osprey_approval", "nonexistent"], "rules": ["safety"]}

    Raises:
        ValueError: If any short name is unknown, with a helpful message listing
            the invalid names and the available alternatives.
    """
    errors: list[str] = []

    for artifact_type, names in artifacts.items():
        available = list_artifacts(artifact_type)
        available_set = set(available)

        for name in names:
            if name not in available_set:
                # Build a "did you mean" suggestion
                suggestion = _suggest(name, available)
                hint = f"did you mean '{suggestion}'? " if suggestion else ""
                errors.append(
                    f"  {artifact_type}/{name}: not found. "
                    f"{hint}Available {artifact_type}: {', '.join(available)}"
                )

    if errors:
        raise ValueError("Invalid artifact(s) in profile:\n" + "\n".join(errors))


def resolve_artifact(artifact_type: str, short_name: str) -> Path:
    """Resolve a short name to the source path in the built-in library.

    For skills (directories), returns the directory path.
    For all other types, returns the path of the matching file.

    Args:
        artifact_type: One of "hooks", "rules", "skills", "agents", "output_styles"
        short_name: The short name of the artifact (e.g. "osprey_approval", "safety")

    Returns:
        Absolute Path to the artifact file or directory.

    Raises:
        ValueError: If the artifact type or short name is not found.
    """
    directory = _artifact_dir(artifact_type)

    if artifact_type == "skills":
        candidate = directory / short_name
        if candidate.is_dir():
            return candidate
    else:
        for path in directory.iterdir():
            name = _short_name_from_path(path, artifact_type)
            if name == short_name:
                return path

    available = list_artifacts(artifact_type)
    suggestion = _suggest(short_name, available)
    hint = f" Did you mean '{suggestion}'?" if suggestion else ""
    raise ValueError(
        f"Artifact {artifact_type}/{short_name} not found in the built-in library.{hint} "
        f"Available {artifact_type}: {', '.join(available)}"
    )


def _suggest(name: str, candidates: list[str]) -> str | None:
    """Return the best matching candidate by edit distance, or None if none is close."""
    if not candidates:
        return None

    def _distance(a: str, b: str) -> int:
        # Simple Levenshtein distance
        if a == b:
            return 0
        la, lb = len(a), len(b)
        prev = list(range(lb + 1))
        for i, ca in enumerate(a):
            curr = [i + 1] + [0] * lb
            for j, cb in enumerate(b):
                curr[j + 1] = min(
                    prev[j + 1] + 1,  # deletion
                    curr[j] + 1,  # insertion
                    prev[j] + (0 if ca == cb else 1),  # substitution
                )
            prev = curr
        return prev[lb]

    best, best_dist = min(((c, _distance(name, c)) for c in candidates), key=lambda x: x[1])
    # Only suggest if reasonably close (within half the length of the query)
    threshold = max(2, len(name) // 2)
    return best if best_dist <= threshold else None
