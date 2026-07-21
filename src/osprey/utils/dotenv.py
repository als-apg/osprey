"""Minimal ``.env`` file parsing shared across the CLI and deployment layers.

A tiny dependency-free parser (no ``python-dotenv`` import) used where we only
need to read the current ``KEY=VALUE`` pairs from a project ``.env`` — e.g. the
build lifecycle env injection and the deploy-time dispatch-token bootstrap.
"""

from __future__ import annotations

from pathlib import Path


def parse_dotenv_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a dict of environment variables.

    Handles ``KEY=VALUE`` lines, ``#`` comments, blank lines, an optional
    ``export`` prefix, and quoted values (single or double quotes stripped from
    the value boundaries).
    """
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip lines without =
        if "=" not in line:
            continue
        # Skip `export` prefix (common in .env files)
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip matching surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env[key] = value
    return env


def _dotenv_raw_lines(text: str) -> dict[str, str]:
    """Map ``KEY`` -> its raw ``KEY=VALUE`` line (quoting intact) from ``text``."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        candidate = stripped[7:] if stripped.startswith("export ") else stripped
        key = candidate.partition("=")[0].strip()
        if key:
            raw[key] = stripped
    return raw


def merge_env_preserving_existing(rendered_text: str, existing_text: str) -> str:
    """Merge a freshly rendered ``.env`` with an existing one; existing wins.

    Used when a build re-renders a project in place (``osprey build --force``)
    or a profile ships a template ``.env``: the rendered text provides the
    structure, comments, and any newly introduced variables, while every value
    the user already has keeps its existing setting (their secrets, and the
    service tokens/passwords that live containers and docker volumes were
    initialized with). Keys present only in the existing file are appended at
    the end so nothing the user set is ever dropped.
    """
    existing = _dotenv_raw_lines(existing_text)
    consumed: set[str] = set()
    out_lines: list[str] = []
    for line in rendered_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            candidate = stripped[7:] if stripped.startswith("export ") else stripped
            key = candidate.partition("=")[0].strip()
            if key in existing:
                out_lines.append(existing[key])
                consumed.add(key)
                continue
        out_lines.append(line)
    leftovers = [existing[key] for key in existing if key not in consumed]
    if leftovers:
        out_lines.extend(["", "# Preserved from existing .env"])
        out_lines.extend(leftovers)
    return "\n".join(out_lines) + "\n"
