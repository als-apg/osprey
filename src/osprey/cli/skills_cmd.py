"""Skills install CLI for Osprey Framework.

Copies bundled skills from inside the installed wheel to a target
``.claude/skills/`` directory using ``importlib.resources`` so it works in both
editable and installed (zipped) wheel modes.

The default target is ``~/.claude/skills/`` (global, available in any Claude
Code session). With ``--target``, the skill can be installed into a specific
``.claude/skills/`` directory — used by the build-interview skill to drop
``osprey-build-deploy`` into a freshly generated facility profile repo.
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
    "osprey-build-deploy": "templates/skills/osprey-build-deploy",
    "osprey-contribute": "templates/skills/osprey-contribute",
    "osprey-pre-commit": "templates/skills/osprey-pre-commit",
    "osprey-release": "templates/skills/osprey-release",
}


@click.group()
def skills() -> None:
    """Manage bundled Osprey skills."""


@skills.command()
@click.argument("name", type=str)
@click.option(
    "--target",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory to install the skill into. Defaults to ~/.claude/skills/ "
        "(global). Use a project-local .claude/skills/ path to scope the skill "
        "to one repo."
    ),
)
def install(name: str, target: Path | None) -> None:
    """Install a bundled skill into <target>/<name>/.

    \b
    Currently supported skills:
      build-interview      Author OSPREY build profiles (global)
      osprey-build-deploy  Operate a facility profile repo's deploy pipeline
      osprey-contribute    Walk a contributor through the GitHub Flow journey
      osprey-pre-commit    Run quick / ci / premerge check scripts at the right gate
      osprey-release       Cut a CalVer release: bump PR, tag, verify publish

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

    skills_dir = (target or Path.home() / ".claude" / "skills").expanduser().resolve()
    skills_dir.mkdir(parents=True, exist_ok=True)
    install_path = skills_dir / name

    if install_path.exists() and any(install_path.iterdir()):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = skills_dir / f"{name}.bak.{ts}"
        install_path.rename(backup)
        click.echo(f"Warning: existing '{name}' moved to {backup}", err=True)
    elif install_path.exists():
        install_path.rmdir()

    src_traversable = files("osprey").joinpath(_SKILL_SOURCES[name])
    with as_file(src_traversable) as src_path:
        shutil.copytree(src_path, install_path)

    click.echo(f"Installed '{name}' to {install_path}")
