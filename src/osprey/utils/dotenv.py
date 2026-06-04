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
