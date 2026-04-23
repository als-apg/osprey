"""Skills install CLI for Osprey Framework.

Copies bundled skills from inside the installed wheel to ``~/.claude/skills/``
using ``importlib.resources`` so it works in both editable and installed
(zipped) wheel modes.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

import click

_SKILL_SOURCES: dict[str, str] = {
    "build-interview": "templates/skills/build-interview",
}


@click.group()
def skills() -> None:
    """Manage bundled Osprey skills."""


@skills.command()
@click.argument("name", type=str)
def install(name: str) -> None:
    """Install a bundled skill into ~/.claude/skills/<name>/.

    \b
    Currently supported skills:
      build-interview

    On an existing non-empty target, the prior content is renamed to
    <name>.bak.<YYYYMMDD-HHMMSS>/ before the new copy is written, so a
    previous version of the skill is never lost.
    """
    if name not in _SKILL_SOURCES:
        click.echo(
            f"Unknown skill '{name}'. Available: {sorted(_SKILL_SOURCES)}",
            err=True,
        )
        sys.exit(1)

    skills_dir = Path.home() / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    target = skills_dir / name

    if target.exists() and any(target.iterdir()):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = skills_dir / f"{name}.bak.{ts}"
        target.rename(backup)
        click.echo(f"Warning: existing '{name}' moved to {backup}", err=True)
    elif target.exists():
        target.rmdir()

    src_traversable = files("osprey").joinpath(_SKILL_SOURCES[name])
    with as_file(src_traversable) as src_path:
        shutil.copytree(src_path, target)

    click.echo(f"Installed '{name}' to {target}")
