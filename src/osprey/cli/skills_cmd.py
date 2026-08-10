"""Skills install CLI for Osprey Framework.

Copies bundled skills from inside the installed wheel to a target
``.claude/skills/`` directory using ``importlib.resources`` so it works in both
editable and installed (zipped) wheel modes.

The default target is ``~/.claude/skills/`` (global, available in any agent
session). With ``--target``, the skill goes into a specific ``.claude/skills/``
directory instead, scoping it to one repo.

A deployment's own lifetime is split across two of the bundled skills, and the
split is by the question being asked rather than by command group.
``osprey-build-interview`` is build-time: what the facility repo should say,
its ``deploy:`` block included, up to the point where a project builds from it.
``osprey-deploy-ops`` is operate-time: emitting the deploy scaffolding, running
the stack and diagnosing it. Anything ending in "the profile should say
something different" is the first; anything ending in "the deployment is
misbehaving" is the second.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

import click

_SKILL_SOURCES: dict[str, str] = {
    "osprey-build-interview": "templates/skills/osprey-build-interview",
    "osprey-deploy-ops": "templates/skills/osprey-deploy-ops",
    "osprey-contribute": "templates/skills/osprey-contribute",
    "osprey-pre-commit": "templates/skills/osprey-pre-commit",
    "osprey-release": "templates/skills/osprey-release",
    "osprey-design-philosophy": "templates/skills/osprey-design-philosophy",
    "creating-an-osprey-panel": "templates/skills/creating-an-osprey-panel",
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
      osprey-build-interview  Author OSPREY build profiles (global)
      osprey-deploy-ops       Operate a deployed stack: triage, re-scaffold, secrets
      osprey-contribute       Walk a contributor through the GitHub Flow journey
      osprey-pre-commit       Run quick / ci / premerge check scripts at the right gate
      osprey-release          Cut a CalVer release: bump PR, tag, verify publish
      osprey-design-philosophy  OSPREY's design and architecture principles for review/design
      creating-an-osprey-panel  Author a themed, token-only OSPREY web-terminal panel

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
