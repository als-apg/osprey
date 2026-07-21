"""Allow ``python -m osprey`` to run the CLI.

Subprocesses that need to re-enter the CLI (e.g. persona auto-rendering during
``osprey deploy up``) must invoke ``[sys.executable, "-m", "osprey", ...]``
rather than a bare ``"osprey"``: PATH may resolve to a *different* install
(a global pipx/uv tool, another venv) whose presets and code diverge from the
running interpreter's.
"""

from osprey.cli.main import cli

if __name__ == "__main__":
    cli()
