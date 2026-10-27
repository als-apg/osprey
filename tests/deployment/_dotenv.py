"""Read back the ``.env`` a deploy wrote, for tests that assert on what it minted."""

from __future__ import annotations

from pathlib import Path


def read_dotenv(path: Path) -> dict[str, str]:
    """``path`` parsed as a dotenv file, or ``{}`` while nothing has written it."""
    from osprey_connectors.dotenv import parse_dotenv_file

    return parse_dotenv_file(path) if path.is_file() else {}


def project_env(project_dir: Path) -> dict[str, str]:
    """The project's ``.env``, parsed, or ``{}`` before the first deploy writes one."""
    return read_dotenv(project_dir / ".env")
