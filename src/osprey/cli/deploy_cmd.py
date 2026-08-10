"""Service deployment CLI commands wrapping osprey.deployment.container_manager.

``osprey deploy`` is a Click group: one subcommand per verb, each carrying only
the options it acts on. A flag that a verb ignores is a parse error rather than
a silent no-op, and ``osprey deploy VERB --help`` documents exactly that verb.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import click

from osprey.cli.styles import Styles, console
from osprey.deployment.container_manager import (
    clean_deployment,
    deploy_down,
    deploy_restart,
    deploy_up,
    prepare_compose_files,
    rebuild_deployment,
    show_status,
)
from osprey.deployment.errors import DevModeUnavailableError
from osprey.utils.config import load_project_config
from osprey.utils.logger import get_logger

from .project_utils import resolve_config_path, resolve_project_path

logger = get_logger("deploy")

F = TypeVar("F", bound=Callable[..., Any])

# Shared option surfaces. ``click.option`` builds a fresh Option per
# application, so these decorators can be reused across subcommands.
_config_option = click.option(
    "--config",
    "-c",
    type=click.Path(),
    default="config.yml",
    help="Configuration file (default: config.yml in project directory)",
)
_project_option = click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
_detached_option = click.option(
    "--detached",
    "-d",
    is_flag=True,
    help="Run services in detached mode",
)
_dev_option = click.option(
    "--dev",
    is_flag=True,
    help="Development mode: copy local osprey package to containers instead of using PyPI version. Use this when testing local osprey changes.",
)
_expose_option = click.option(
    "--expose",
    is_flag=True,
    help="Expose services to all network interfaces (0.0.0.0). WARNING: This exposes services to the network! Only use with proper authentication configured.",
)
_archive_option = click.option(
    "--archive",
    is_flag=True,
    help="Archive a user's workspace before removing it.",
)
_purge_option = click.option(
    "--purge",
    is_flag=True,
    help="Permanently delete a user's workspace without archiving it.",
)
_yes_option = click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Assume yes to confirmation prompts.",
)


def _project_options(f: F) -> F:
    """Attach the ``--project``/``--config`` pair every deploy verb shares."""
    return _project_option(_config_option(f))


def _check_archive_purge(archive: bool, purge: bool) -> None:
    """Reject ``--archive --purge``: a workspace is either kept or it isn't."""
    if archive and purge:
        raise click.UsageError("--archive and --purge are mutually exclusive.")


def _abort_missing_config(config_path: str, action: str) -> NoReturn:
    """Explain a missing config file, pointing at any project dirs nearby."""
    console.print(
        f"\n✗ Configuration file not found: [accent]{config_path}[/accent]",
        style=Styles.ERROR,
    )
    console.print("\nHint: Are you in a project directory?", style=Styles.WARNING)
    console.print(f"   Current directory: [dim]{Path.cwd()}[/dim]\n")

    # Look for nearby project directories with config.yml
    # Exclude common non-project directories
    excluded_dirs = {
        "docs",
        "tests",
        "test",
        "build",
        "dist",
        "venv",
        ".venv",
        "node_modules",
        ".git",
        "__pycache__",
        "src",
        "lib",
    }
    nearby_projects = []
    try:
        for item in Path.cwd().iterdir():
            if (
                item.is_dir()
                and item.name not in excluded_dirs
                and not item.name.startswith(".")
                and (item / "config.yml").exists()
            ):
                nearby_projects.append(item.name)
    except PermissionError:
        pass  # Skip if can't read directory

    if nearby_projects:
        console.print("   Found project(s) in current directory:", style=Styles.WARNING)
        for proj in nearby_projects[:5]:  # Limit to 5 suggestions
            console.print(f"     • [command]cd {proj} && osprey deploy {action}[/command] or: ")
            console.print(f"       [command]osprey deploy {action} --project {proj}[/command]")
    else:
        console.print("   Try:", style=Styles.WARNING)
        console.print("     • Navigate to your project directory first")
        console.print("     • Use [command]--project[/command] flag to specify project location")

    console.print("\n   Or use interactive menu: [command]osprey[/command]\n")
    raise click.Abort()


@contextmanager
def _deploy_session(project: str | None, config: str, *, announce: bool = True) -> Iterator[str]:
    """Resolve the project and config file, then run one deploy verb in it.

    Changes into the project directory so that all CWD-relative operations
    (template loading, .env lookup, build/ output) resolve against the project
    root, and renders the failures every verb shares — a cancelled run, an
    unusable ``--dev``, and anything else the deployment layer raises.

    Args:
        project: Value of ``--project``, or None to fall back to OSPREY_PROJECT
            / the current directory.
        config: Value of ``--config``.
        announce: Print the "Service management: VERB" banner. Off for verbs
            whose own output arrives immediately, such as ``status``.

    Yields:
        Path to the project's configuration file, guaranteed to exist.
    """
    action = click.get_current_context().info_name or "deploy"
    if announce:
        console.print(f"Service management: [bold]{action}[/bold]")

    try:
        # Both paths are resolved against the directory the operator typed them
        # in, BEFORE the chdir. Resolving the config afterwards would resolve
        # --project a second time against the project it already selected, so a
        # relative `--project build/my-agent` would look for its config under
        # `build/my-agent/build/my-agent/` and abort — which is exactly the
        # spelling the emitted CI pipeline and the missing-config hint both use.
        project_dir = resolve_project_path(project)
        config_path = resolve_config_path(project, config)
        os.chdir(project_dir)

        if not Path(config_path).exists():
            _abort_missing_config(config_path, action)

        yield config_path
    except (click.Abort, click.ClickException):
        raise
    except KeyboardInterrupt:
        console.print("\n!  Operation cancelled by user", style=Styles.WARNING)
        raise click.Abort() from None
    except DevModeUnavailableError as e:
        # Rendered ahead of the generic handler: the reason alone is not
        # actionable, and this failure is precisely the one that used to be a
        # warning nobody saw. Print the remedy and say plainly that nothing was
        # deployed, so it cannot be mistaken for a partial success.
        console.print(f"\n✗ --dev cannot be honored: {e.reason}\n", style=Styles.ERROR)
        for line in e.remedy.splitlines():
            console.print(f"  {line}" if line else "")
        console.print("\n  Nothing was deployed.\n", style=Styles.WARNING)
        raise click.Abort() from None
    except Exception as e:
        console.print(f"✗ Deployment failed: {e}", style=Styles.ERROR)
        # Show more details in verbose mode
        if os.environ.get("DEBUG"):
            import traceback

            console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort() from None


@click.group()
def deploy() -> None:
    """Manage Docker/Podman services for Osprey projects.

    Controls service deployment, status and cleanup, plus per-user
    web-terminal lifecycle management. The services to deploy are defined in
    your config.yml under the 'deployed_services' key.

    Every verb acts on one project: run it from the project directory, pass
    --project, or set the OSPREY_PROJECT environment variable. The exception is
    'scaffold', which emits a facility repo's deployment files and therefore
    acts on the repo rather than on a project built from it.

    Examples:

    \b
      # Everyday service control
      $ osprey deploy up -d
      $ osprey deploy up --dev --project ~/projects/my-agent
      $ osprey deploy down
      $ osprey deploy status
      $ osprey deploy rebuild --dev

    \b
      # Web-terminal workspaces: one user, the stale ones, the whole stack
      $ osprey deploy decommission alice --archive
      $ osprey deploy prune --dry-run
      $ osprey deploy prune --purge --yes
      $ osprey deploy nuke --yes
      $ osprey deploy seed alice

    \b
      # Re-emit the facility repo's CI pipeline and health check
      $ osprey deploy scaffold

    Run 'osprey deploy VERB --help' for the options a single verb takes.
    """


@deploy.command()
@_project_options
@_detached_option
@_dev_option
@_expose_option
def up(project: str | None, config: str, detached: bool, dev: bool, expose: bool) -> None:
    """Start all configured services."""
    with _deploy_session(project, config) as config_path:
        deploy_up(config_path, detached=detached, dev_mode=dev, expose_network=expose)


@deploy.command()
@_project_options
@_dev_option
def down(project: str | None, config: str, dev: bool) -> None:
    """Stop all services."""
    with _deploy_session(project, config) as config_path:
        deploy_down(config_path, dev_mode=dev)


@deploy.command()
@_project_options
@_detached_option
@_expose_option
def restart(project: str | None, config: str, detached: bool, expose: bool) -> None:
    """Restart all services."""
    with _deploy_session(project, config) as config_path:
        deploy_restart(config_path, detached=detached, expose_network=expose)


@deploy.command()
@_project_options
def status(project: str | None, config: str) -> None:
    """Show service status."""
    # No banner: the status table is the output, and announcing the verb that
    # produced it only pushes the table down the screen.
    with _deploy_session(project, config, announce=False) as config_path:
        show_status(config_path, console=console, styles=Styles)


@deploy.command()
@_project_options
@_dev_option
@_expose_option
def build(project: str | None, config: str, dev: bool, expose: bool) -> None:
    """Build/prepare compose files without starting services."""
    with _deploy_session(project, config) as config_path:
        console.print("Building compose files...")
        _, compose_files = prepare_compose_files(config_path, dev_mode=dev, expose_network=expose)
        console.print("\n✓ Compose files built successfully:")
        for compose_file in compose_files:
            console.print(f"  • {compose_file}")


@deploy.command()
@_project_options
@_dev_option
@_expose_option
def clean(project: str | None, config: str, dev: bool, expose: bool) -> None:
    """Remove containers and volumes (WARNING: destructive)."""
    with _deploy_session(project, config) as config_path:
        # clean_deployment expects compose_files list, so prepare them first.
        # Keep the loaded config and pass it through so cleanup runs under
        # this deploy's COMPOSE_PROJECT_NAME (resolve_project_name(config))
        # rather than the shared "unnamed-project" default.
        cfg, compose_files = prepare_compose_files(config_path, dev_mode=dev, expose_network=expose)
        clean_deployment(compose_files, cfg)


@deploy.command()
@_project_options
@_detached_option
@_dev_option
@_expose_option
def rebuild(project: str | None, config: str, detached: bool, dev: bool, expose: bool) -> None:
    """Clean, rebuild, and restart services."""
    with _deploy_session(project, config) as config_path:
        rebuild_deployment(config_path, detached=detached, dev_mode=dev, expose_network=expose)


@deploy.command()
@click.argument("user")
@_project_options
@_archive_option
@_purge_option
@_yes_option
def decommission(
    user: str, project: str | None, config: str, archive: bool, purge: bool, yes: bool
) -> None:
    """Remove a single user's web-terminal workspace."""
    # Validated before the session so the check fires without a project,
    # a config file, or the web_terminals package being importable.
    _check_archive_purge(archive, purge)

    with _deploy_session(project, config) as config_path:
        from osprey.deployment.web_terminals.lifecycle import decommission_user

        decommission_user(config_path, user, archive=archive, purge=purge, assume_yes=yes)


@deploy.command()
@_project_options
@_archive_option
@_purge_option
@_yes_option
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without making changes.",
)
def prune(
    project: str | None, config: str, archive: bool, purge: bool, yes: bool, dry_run: bool
) -> None:
    """Remove workspaces for users no longer in the user index."""
    _check_archive_purge(archive, purge)

    with _deploy_session(project, config) as config_path:
        from osprey.deployment.web_terminals.lifecycle import prune_users

        prune_users(config_path, dry_run=dry_run, archive=archive, purge=purge, assume_yes=yes)


@deploy.command()
@_project_options
@_yes_option
def nuke(project: str | None, config: str, yes: bool) -> None:
    """Tear down the whole multi-user web-terminal stack.

    WARNING: destructive — this removes every user's workspace, not just the
    stale ones. Use 'prune' to remove only users who left the index.
    """
    with _deploy_session(project, config) as config_path:
        from osprey.deployment.web_terminals.lifecycle import nuke_stack

        nuke_stack(config_path, assume_yes=yes)


@deploy.command()
@click.argument("user", required=False)
@_project_options
def seed(user: str | None, project: str | None, config: str) -> None:
    """(Re)seed web-terminal workspaces from the user index.

    USER targets one user; omit it to reseed every user in the index.
    """
    with _deploy_session(project, config) as config_path:
        from osprey.deployment.web_terminals.seeding import seed_web_terminals

        seed_web_terminals(config_path, user)


@deploy.command()
@click.argument("user")
@_project_options
def passwd(user: str, project: str | None, config: str) -> None:
    """Change one web-terminal user's login password.

    Prompts for the new password and ends that user's sessions.
    """
    with _deploy_session(project, config) as config_path:
        from osprey.deployment.web_terminals.lifecycle import rotate_user_password

        # hide_input: the password is never echoed to the terminal, never
        # placed on the argv (where it would land in shell history and in
        # every `ps` listing on the host), and never logged.
        # confirmation_prompt: a mistyped password would otherwise lock the
        # user out of a terminal that was working a moment ago.
        password = click.prompt(
            f"New web-terminal password for {user}",
            hide_input=True,
            confirmation_prompt=True,
        )
        rotate_user_password(config_path, user, password)
        # Deliberately does not claim the old sessions are already dead:
        # when nothing has been deployed from this root yet, the recreate
        # is skipped (with its own warning) and the change takes effect at
        # the next `deploy up`. Stating the consequence without asserting
        # the timing keeps this true on both paths.
        console.print(
            f"\n✓ Password changed for [accent]{user}[/accent]. Any session they "
            "still hold stops working as soon as the sidecar is running this change."
        )


@deploy.command()
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Facility repo to emit into (default: the one the current directory is in)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing file that the scaffolder did not write.",
)
def scaffold(repo: str | None, force: bool) -> None:
    """Emit this facility repo's deployment files.

    Renders two files from the profile's 'deploy:' block: the CI pipeline at
    the repo root, and the post-deploy health check inside the profile's
    project/ mirror, which every build copies to scripts/verify.sh. Run it
    again whenever the block changes.

    Unlike the other deploy verbs this one acts on a facility repo rather than
    on a built project, so it takes --repo instead of --project. Run from
    anywhere inside the repo and it finds the root on its own.

    Re-running is safe. A file whose content already matches is left untouched,
    stamp included, so an OSPREY upgrade alone produces no diff. A file the
    scaffolder did not write is reported and left alone; --force replaces it.

    Examples:

    \b
      $ osprey deploy scaffold
      $ osprey deploy scaffold --repo ~/facility/demo-facility
      $ osprey deploy scaffold --force
    """
    # Resolved before anything is emitted: an unfindable repo is a mistake in
    # where the command was run, not a deployment failure.
    from osprey.errors import ConfigurationError

    from .deploy_scaffold import find_facility_repo_root, scaffold_deploy_files

    repo_root = Path(repo).resolve() if repo else find_facility_repo_root(Path.cwd())
    if repo_root is None:
        logger.error(
            "No facility repo found from %s. 'osprey deploy scaffold' emits into "
            "a repo holding profile/profile.yml -- run it from inside one, or "
            "point --repo at it.",
            Path.cwd(),
        )
        raise click.Abort()

    try:
        emitted = scaffold_deploy_files(repo_root, force=force)
    except ConfigurationError as e:
        logger.error("Cannot scaffold %s: %s", repo_root, e)
        raise click.Abort() from None

    for result in emitted:
        shown = result.path.relative_to(repo_root)
        if result.action == "unchanged":
            logger.info("Unchanged: %s", shown)
        elif result.refused:
            logger.error("Left alone: %s %s", shown, result.reason)
        else:
            logger.key_info("%s: %s", result.action.capitalize(), shown)

    if any(result.refused for result in emitted):
        raise click.Abort()


@deploy.command("render-env-production")
@_project_options
@click.option(
    "--env-file",
    type=click.Path(),
    help="Secrets file to render from (default: .env in the project directory)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Write the result here (mode 0600) instead of to stdout",
)
def render_env_production(
    project: str | None, config: str, env_file: str | None, output: str | None
) -> None:
    """Render .env.production to stdout or a file.

    .env.production is the env file every per-user web-terminal container runs
    with. This renders the same subset a deploy would generate, from the same
    two inputs: the deploy config, and one secrets file. It applies no rule of
    its own, so a file rendered here and one generated by 'osprey deploy up'
    cannot disagree.

    Secret values come only from the secrets file, never from the surrounding
    environment, so the result depends on the named file and nothing else.
    Unlike a deploy, which never overwrites an existing .env.production, an
    explicit --output is taken as an instruction and replaces what is there.

    Examples:

    \b
      $ osprey deploy render-env-production > .env.production
      $ osprey deploy render-env-production --output .env.production
      $ osprey deploy render-env-production --env-file ../secrets/prod.env
    """
    # Resolve caller-supplied paths BEFORE the session chdirs into the project:
    # a relative path means what the operator typed it against, not whatever
    # the project directory happens to hold.
    env_file_path = Path(env_file).resolve() if env_file else None
    output_path = Path(output).resolve() if output else None

    # No banner: in the default mode stdout IS the rendered file, so anything
    # else printed there would corrupt a `> .env.production` redirect.
    with _deploy_session(project, config, announce=False) as config_path:
        from osprey.deployment.web_terminals.env_production import (
            _build_env_production_subset,
            _claude_code_auth_secret_vars,
        )
        from osprey.utils.dotenv import parse_dotenv_file

        project_root = Path.cwd()
        env_path = env_file_path or project_root / ".env"
        if not env_path.is_file():
            logger.error(
                f"Cannot render .env.production: {env_path} does not exist. Point "
                "--env-file at the secrets file to render from, or create .env in "
                "the project directory."
            )
            raise click.Abort()

        deploy_config = load_project_config(config_path)
        dotenv = parse_dotenv_file(env_path)

        required_vars, extra_vars = _claude_code_auth_secret_vars(deploy_config, project_root)
        missing = {var: origin for var, origin in required_vars.items() if var not in dotenv}
        if missing:
            # Warn rather than refuse: whether an absent auth secret is fatal is
            # the deploy path's call (see ensure_env_production, which does
            # refuse). Rendering anyway is what lets an operator SEE the gap.
            needs = "; ".join(f"{origin} needs {var}" for var, origin in missing.items())
            logger.warning(
                f"Rendering without provider auth secret(s) absent from {env_path}: "
                f"{needs}. Web terminals started with this file will fail "
                "authentication unless they authenticate another way."
            )

        subset = _build_env_production_subset(
            deploy_config, dotenv, {**required_vars, **extra_vars}
        )
        rendered = "".join(f"{key}={value}\n" for key, value in subset.items())

        if output_path is None:
            # click.echo, not the Rich console: these are file bytes, and Rich
            # would wrap a long secret across lines.
            click.echo(rendered, nl=False)
            return

        # Create at 0600 from the first byte on disk rather than write-then-chmod,
        # the same convention ensure_env_production uses for this file: there is
        # no instant at which the secrets exist at a wider mode.
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        os.chmod(output_path, 0o600)
        logger.key_info("Rendered %s (mode 0600): %s", output_path, ", ".join(subset))
