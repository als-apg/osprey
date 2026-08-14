"""Allow ``python -m osprey`` to run the CLI.

Anything that re-enters the CLI as a subprocess must invoke it as
``[sys.executable, "-m", "osprey", ...]`` rather than as a bare ``"osprey"``:
PATH may resolve to a *different* install (a global pipx/uv tool, another venv)
whose presets and code diverge from the running interpreter's.
"""

from osprey.cli.main import cli

if __name__ == "__main__":
    cli()
