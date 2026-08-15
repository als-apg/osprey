"""Main CLI entry point for Osprey Framework.

This module provides the main CLI group that organizes all osprey
commands under the `osprey` command namespace.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from .phase_reporter import PhaseReporter

# Ensure UTF-8 on Windows for Unicode CLI output
if sys.platform == "win32":
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding
        # This fixes the 'charmap' codec error on Windows when printing Unicode
        import io

        # Only reconfigure if not already UTF-8
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except (AttributeError, OSError):
        # If reconfiguration fails (e.g., no buffer attribute), continue
        # The CLI should still work, just without fancy Unicode characters
        pass

try:
    from osprey import __version__
except ImportError:
    # Only reachable from a broken/partial install. A sentinel rather than a
    # release number: any literal release named here is stale the moment the
    # next version ships, and `osprey --version` would report it as fact.
    __version__ = "0.0.0+unknown"


class LazyGroup(click.Group):
    """Click group that lazily loads subcommands only when invoked."""

    def get_command(self, ctx, cmd_name):
        """Lazily import and return the command when it's invoked."""
        # Map command names to their module paths
        commands = {
            "init": "osprey.cli.init_cmd",  # Create the deployment repo
            "build": "osprey.cli.build_cmd",
            "up": "osprey.cli.deploy_cmd",  # Start the deployment from build/
            "down": "osprey.cli.deploy_cmd",  # Stop it, keeping volumes
            "restart": "osprey.cli.deploy_cmd",  # Stop and start, recreating
            "status": "osprey.cli.deploy_cmd",  # Read-only report on the deployment
            "logs": "osprey.cli.deploy_cmd",  # Compose logs for the deployment
            "reset": "osprey.cli.reset_cmd",  # Factory reset (destructive)
            "profile": "osprey.cli.profile_cmd",  # Build-profile authoring
            "config": "osprey.cli.config_cmd",
            "set": "osprey.cli.set_cmd",  # Profile write-back
            "validate": "osprey.cli.validate_cmd",
            "chat": "osprey.cli.chat_cmd",  # Agent session against build/
            "health": "osprey.cli.health_cmd",
            "eject": "osprey.cli.eject_cmd",
            "channel-finder": "osprey.cli.channel_finder_cmd",
            "ariel": "osprey.cli.ariel",  # ARIEL search service
            "sim": "osprey.cli.sim",  # Simulation scenarios
            "artifacts": "osprey.cli.artifacts_cmd",  # Artifact Gallery
            "web": "osprey.cli.web_cmd",  # Web Terminal
            "theme-lab": "osprey.cli.theme_lab_cmd",  # Design-system theme workbench
            "scaffold": "osprey.cli.scaffold_cmd",  # Build artifact overrides
            "audit": "osprey.cli.audit_cmd",  # Safety auditor
            "skills": "osprey.cli.skills_cmd",  # Bundled skill management
            "vendor": "osprey.cli.vendor_cmd",  # Vendor asset management
            "knowledge": "osprey.cli.knowledge_cmd",  # OKF facility knowledge
            "query": "osprey.cli.query_cmd",  # Headless agent query
            "users": "osprey.cli.users_cmd",  # Web-terminal roster
        }

        if cmd_name not in commands:
            return None

        import importlib

        mod = importlib.import_module(commands[cmd_name])

        if cmd_name == "channel-finder":
            cmd_func = mod.channel_finder
        elif cmd_name == "ariel":
            cmd_func = mod.ariel_group
        elif cmd_name == "sim":
            cmd_func = mod.sim_group
        elif cmd_name == "artifacts":
            cmd_func = mod.artifacts
        elif cmd_name == "web":
            cmd_func = mod.web
        elif cmd_name == "theme-lab":
            cmd_func = mod.theme_lab
        elif cmd_name == "scaffold":
            cmd_func = mod.scaffold
        elif cmd_name == "profile":
            cmd_func = mod.profile
        elif cmd_name in ("up", "down", "restart", "status", "logs"):
            # Every lifecycle verb in deploy_cmd is defined as `<name>_verb`,
            # so it is resolved by that name rather than by the bare one. The
            # module has no bare `up`/`down`/`restart`/`status`/`logs` to fall
            # back to, and the generic getattr below would raise AttributeError.
            cmd_func = getattr(mod, f"{cmd_name}_verb")
        else:
            cmd_func = getattr(mod, cmd_name)

        return cmd_func

    def list_commands(self, ctx):
        """Return list of available commands (for --help)."""
        return [
            "init",
            "build",
            "up",
            "down",
            "restart",
            "status",
            "logs",
            "reset",
            "profile",
            "set",
            "validate",
            "chat",
            "config",
            "health",
            "channel-finder",
            "eject",
            "ariel",
            "sim",
            "artifacts",
            "web",
            "theme-lab",
            "scaffold",
            "audit",
            "skills",
            "vendor",
            "knowledge",
            "query",
            "users",
        ]


@click.group(cls=LazyGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="osprey")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show debug output, including every container command run.",
)
@click.pass_context
def cli(ctx, verbose):
    """Osprey Framework CLI - Capability-Based Agentic Framework.

    A deployment is a git repo: profile.yml at its root is the source, build/ is
    what a build renders from it, var/ is durable state. Every verb below finds
    that repo by walking up from where you are standing, so none of them needs
    to be told where it is.

    Use 'osprey COMMAND --help' for more information on a specific command.

    Examples:

    \b
      osprey init my-agent --preset control-assistant
                                       Create a deployment repo from a preset
      osprey set connector=epics       Edit profile.yml
      osprey validate                  Check profile.yml without building
      osprey build                     Render build/ from profile.yml
      osprey up -d                     Start it, in the background
      osprey -v up -d                  Same, with every command echoed
      osprey status                    What is running, and is it current
      osprey logs -f                   Follow the container logs
      osprey chat                      Talk to this deployment's agent
      osprey restart                   Stop and start it again, as built
      osprey down                      Stop it, keeping all data
      osprey users remove alice        Retire one web-terminal user
      osprey reset                     Factory reset (asks first)
      osprey web                       Launch web terminal
      osprey health                    Check system health
    """
    import logging

    from osprey.utils.config import load_project_dotenv
    from osprey.utils.logger import configure_logging

    from .styles import initialize_theme_from_config

    # The CLI is a process entry point: nothing else populates the environment
    # from `.env`, and importing the framework deliberately does not. Do this
    # first — logging and theme config below resolve `${VAR}` placeholders, and
    # subcommands may read os.environ before touching config at all.
    load_project_dotenv()

    # Likewise a process entry point for logging: nothing else configures it,
    # and importing the framework deliberately does not. Records go to stderr,
    # so `--json` subcommand output on stdout stays machine-readable.
    #
    # `-v` is the escape hatch for everything demoted to DEBUG so that a normal
    # run reads as a report rather than a transcript — most visibly the
    # container commands the deploy path shells out to, which are what someone
    # needs when reproducing a deploy step by hand.
    configure_logging(logging.DEBUG if verbose else logging.INFO)

    initialize_theme_from_config()

    if ctx.invoked_subcommand is None:
        # The bare command lists what there is to run. Every verb is
        # zero-argument, so the help is the menu — an interactive wrapper would
        # be a surface with nothing to wrap.
        click.echo(ctx.get_help())


@contextmanager
def lifecycle_reporter() -> Iterator["PhaseReporter"]:
    """Install the phase reporter for the duration of one lifecycle verb.

    Every lifecycle verb (``init``, ``build``, ``up``, ``restart``, ``down``,
    ``reset``) opens this at entry. The verb that opens it OUTERMOST owns the
    reporter: a verb that finds one already installed reuses it and installs
    nothing, so ``init --up`` — which chains ``build`` and ``up`` through
    ``ctx.invoke`` — reports as one continuous run rather than three, and does
    it through the module-level handle instead of threading a reporter through
    every signature.

    "Already installed" is judged as "not the quiet default": the default is a
    :class:`~osprey.cli.phase_reporter.NullReporter` with ``verbose`` False, and
    that is the one shape no verb ever installs. Under the global ``--verbose``
    the installed reporter is ``NullReporter(verbose=True)`` — verbose there is
    not decoration, it is what :func:`~osprey.cli.phase_reporter.is_verbose`
    reports to the capture helper, which streams its subprocesses instead of
    spooling them.

    The flag is read off the root context rather than taken as an argument,
    because the verbs reach here three different ways — a plain invocation, a
    ``ctx.invoke`` chain, and a direct call in a test — and only the context
    knows about all three. ``build`` declares no ``pass_context`` at all.
    """
    from .phase_reporter import NullReporter, PhaseReporter, current_reporter, install_reporter

    active = current_reporter()
    if not (isinstance(active, NullReporter) and not active.verbose):
        yield active
        return

    ctx = click.get_current_context(silent=True)
    verbose = bool(ctx.find_root().params.get("verbose")) if ctx is not None else False
    reporter: PhaseReporter = NullReporter(verbose=True) if verbose else _tty_aware_reporter()
    previous = install_reporter(reporter)
    try:
        reporter.start_rendering()
        yield reporter
    finally:
        # The swap stops whatever the reporter had running: the monitor thread
        # first, then the live region. Nothing else here needs to know which.
        install_reporter(previous)


def _tty_aware_reporter() -> "PhaseReporter":
    """The reporter for this stdout: live on a terminal, plain lines otherwise.

    Asked once, here, rather than per line: a verb whose output is piped or
    redirected gets the plain-line reporter for its whole run, and the live
    region is never mounted at all. ``stdout`` is asked directly rather than the
    console, which is ``force_terminal`` on win32.

    The reporter is built immediately before it is installed, which is what
    binds the live region's console to the themed one the rest of the CLI is
    already using.
    """
    from .phase_reporter import LiveReporter, PhaseReporter

    return LiveReporter() if sys.stdout.isatty() else PhaseReporter()


def main():
    """Entry point for the osprey CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nGoodbye!", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
