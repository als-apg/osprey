"""The lifecycle verbs that act on a running deployment.

Five top-level commands live here: ``up``, ``down``, ``restart``, ``status`` and
``logs``. Each takes a deployment repo and nothing else — it is found by walking
up from the working directory, or named with ``--repo`` — and each acts on
``build/`` as the last ``osprey build`` rendered it. None of them renders.

``up`` and ``restart`` share one gate, :func:`gate_start_from_build`: a start
verb must never quietly deploy a profile edit that was never built.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import click

from osprey.cli import output
from osprey.cli.styles import Styles, console
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.utils.config import load_project_config
from osprey.utils.logger import get_logger

from .repo_resolver import repo_option

if TYPE_CHECKING:
    from .phase_reporter import PhaseReporter

logger = get_logger("deploy")


def _report_fact(message: str) -> None:
    """Report ``message`` under this module's logger.

    The promotion contract lives in :func:`osprey.cli.output.report_fact`; this
    binds it to the deploy logger so call sites pass the line alone.

    Args:
        message: The finished line, built by the caller.
    """
    output.report_fact(logger, message)


def _warn_fact(summary: str, detail: str | None = None, remedy: str | None = None) -> None:
    """Warn under this module's logger, in the run's own column.

    The promotion contract lives in :func:`osprey.cli.output.warn_fact`; this
    binds it to the deploy logger so call sites pass the copy alone. Promotion is
    not optional for a warning raised during a lifecycle verb: the altitude gate
    drops raw WARNING records while a reporter owns the terminal, so
    ``logger.warning`` here would reach nobody.
    """
    output.warn_fact(logger, summary, detail, remedy)


# ---------------------------------------------------------------------------
# osprey up — the four-zone start verb
# ---------------------------------------------------------------------------

#: Section header the ``.env`` preflight writes its harvested keys under. Says
#: the verb that actually wrote them: a file claiming to have been seeded by a
#: command the operator never ran is the kind of small lie that costs an hour
#: when someone goes looking for where a key came from.
_UP_SEEDED_ENV_BANNER = "# ── Seeded by `osprey up` from your shell ──"


def _abort(
    summary: str,
    cause: str | None = None,
    remedy: str | None = None,
    *,
    mark: bool = True,
) -> NoReturn:
    """Render a refusal in the CLI's one failure shape, then stop.

    :func:`osprey.cli.output.fail` only prints, so the ``click.Abort`` below is
    what ends the run: no traceback, exit code 1, exactly as before.

    Args:
        summary: What the verb will not do, in one line and without the glyph.
        cause: Why, and what is unchanged as a result. Multi-line is fine.
        remedy: The one thing to do about it, or ``None`` when there is nothing
            honest to name.
        mark: Whether to open with ``✗``. Pass ``False`` only where a phase has
            already printed one for this same failure — the block then continues
            that line instead of reading as a second thing having gone wrong.
    """
    output.fail(summary, cause, remedy, mark=mark)
    raise click.Abort()


def _abort_unmet_precondition(
    exc: DeploymentPreconditionError,
    *,
    nothing_done: str | None = None,
    mark: bool = True,
) -> NoReturn:
    """Render an unmet deploy precondition: what is not true, then the one fix.

    Every :class:`~osprey.deployment.errors.DeploymentPreconditionError` carries
    the same three fields, so ONE renderer serves all of them: no verb
    hand-writes a per-error string, and a new precondition needs no new handler
    here. The fields land on :func:`_abort`'s three in order, which is why the
    error type was given them.

    Args:
        exc: The refusal, carrying ``summary``/``reason``/``remedy``.
        nothing_done: What is unchanged as a result ("Nothing was started."),
            for the verbs that act. ``None`` for the read-only verbs, which
            changed nothing whether they refused or not.
        mark: ``False`` where the refusal left through an open phase, whose ``✗``
            is already the run's one failure marker. Decided by the CALLER, not
            here: this same renderer serves the read-only verbs, which refuse
            outside any phase and so have no marker to continue.
    """
    cause = exc.reason if nothing_done is None else f"{exc.reason}\n{nothing_done}"
    _abort(exc.summary, cause, exc.remedy, mark=mark)


def _stdin_is_a_terminal() -> bool:
    """Whether there is a person here to answer a question.

    Its own function so that "can this prompt?" is one decision with one seam,
    rather than an ``isatty`` call buried in a branch — and because ``sys.stdin``
    is itself replaced during a Click test run, so the object to ask has to be
    looked up when the question is asked, not captured at import.
    """
    import sys

    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError):
        # A closed or exotic stream. "Nobody is here" is the safe reading: it
        # refuses with a remedy instead of blocking on a prompt no one sees.
        return False


def gate_start_from_build(
    ctx: click.Context,
    repo_root: Path,
    *,
    chain_build: bool,
    as_built: bool,
    verb: str,
    dev: bool = False,
) -> None:
    """Decide whether a start verb may start this repo's ``build/``.

    The gate behind ``up`` and ``restart``: a start verb must never quietly
    deploy a profile edit that was never built. It reads ``profile.yml`` for
    exactly one purpose — the drift fingerprint — and starts ``build/``
    unchanged either way.

    Both start verbs call it, and gate identically (CC-5); it is parameterised
    on the verb name so each refusal names the command the operator actually
    typed. For ``restart`` the gate matters more than it does for ``up``,
    because a refusal there also means the running stack is left alone: the
    check happens before anything is stopped.

    Four outcomes, from :func:`~osprey.deployment.staleness.check_drift`:

    * ``CLEAN`` — proceed.
    * ``DRIFT`` — refuse, naming the keys that moved, with both exits.
    * ``UNRESOLVABLE`` — refuse; ``--as-built`` is the way through, because
      ``build/`` is self-contained and starting it as rendered stays a knowing,
      valid choice even when nothing can vouch for it.
    * ``NO_BUILD`` — refuse. ``--as-built`` is NOT an escape here and saying so
      is the point: there is no build to start as-built. ``--build`` is, because
      it makes one.

    Version skew warns on every outcome and blocks none: a framework upgrade
    must not stand between an operator and the stack they already have.

    Args:
        ctx: The invoking command's context, used to chain ``osprey build``
            through the same command object an operator would type.
        repo_root: The deployment repo.
        chain_build: ``--build`` — re-render, then start.
        as_built: ``--as-built`` — start the existing render regardless.
        verb: The verb spelling the refusal's remedies use (``"up"`` or
            ``"restart"``).
        dev: ``--dev`` on the start verb, forwarded to the chained build so
            ``--build --dev`` renders the dev build the start then demands.

    Raises:
        click.Abort: On any refusal. Nothing has been started.
        click.UsageError: When both flags were passed.
    """
    from osprey.deployment.staleness import DriftState, check_drift

    if chain_build and as_built:
        raise click.UsageError(
            "--build and --as-built are opposites: one re-renders build/ from "
            "profile.yml before starting, the other starts build/ without "
            "re-rendering. Pass at most one."
        )

    if chain_build:
        # Re-rendering settles the question the gate exists to ask, so it runs
        # instead of the check rather than after it. Through the root group's
        # own `build` command: chaining anything else would be a second way to
        # render a deployment, which is exactly what this feature removes.
        _chain_build(ctx, repo_root, dev=dev)
        return

    report = check_drift(repo_root)
    if report.version_skew:
        output.warn(report.version_skew.message)

    if report.state is DriftState.NO_BUILD:
        note = ""
        if as_built:
            note = "\n--as-built starts a build that already exists; this repo has none to start."
        _abort(
            f"No build found in {report.build_dir}",
            f"`osprey {verb}` starts what a build rendered, and never renders one "
            f"itself.{note}\nNothing was started.",
            "run `osprey build` first",
        )

    if not report.refuses:
        return

    if as_built:
        # Two different facts, so two different sentences. On DRIFT the mismatch
        # is established, and saying so is the warning. On UNRESOLVABLE nothing
        # is known either way — the build may match the profile perfectly — and
        # claiming a mismatch there would be inventing a finding out of a failed
        # comparison.
        consequence = (
            "The running stack will not match profile.yml until the next `osprey build`."
            if report.state is DriftState.DRIFT
            else "Whether the running stack matches profile.yml is unknown, and stays "
            "unknown until the comparison can be made."
        )
        output.warn(
            "Starting build/ as it was rendered (--as-built)", f"{report.message} {consequence}"
        )
        return

    # No `→` remedy line here, deliberately: there are two valid ways through
    # and they are not interchangeable. Naming one of them as *the* remedy
    # would be a recommendation this verb cannot honestly make, so both are
    # spelled in the cause as an aligned block and the operator picks.
    _abort(
        report.message,
        f"osprey {verb} --build      re-render build/ from profile.yml, then start it\n"
        f"osprey {verb} --as-built   start build/ as it was rendered, leaving the "
        f"change unbuilt\n"
        "Nothing was started.",
    )


def _chain_build(ctx: click.Context, repo_root: Path, *, dev: bool = False) -> None:
    """Run ``osprey build`` for *repo_root*, as ``--build`` promises.

    Looked up on the root group by name rather than imported, so this is one
    call into the same command an operator would type — there is no second code
    path that renders a deployment.

    Args:
        ctx: The invoking command's context.
        repo_root: The deployment repo.
        dev: Whether the start verb was given ``--dev``. Forwarded to the
            chained build so the render it produces is the dev render the
            start is about to demand — a chain that dropped it would render a
            pinned build and then refuse to start it.

    Raises:
        click.ClickException: When the verb is unavailable in this installation,
            which is a framework problem rather than an operator mistake.
        click.Abort: Propagated from the build itself. ``build/`` is left as the
            previous render, and nothing is started.
    """
    group = ctx.find_root().command
    build_cmd = group.get_command(ctx, "build") if isinstance(group, click.Group) else None
    if build_cmd is None:
        raise click.ClickException(
            "--build cannot run: `osprey build` is not available in this "
            "installation. Nothing was started."
        )
    ctx.invoke(build_cmd, repo=repo_root, dev=dev)


def ensure_repo_env(repo_root: Path, config: dict[str, Any], *, mark: bool = True) -> None:
    """Refuse to start a deployment repo that has no ``.env``, offering to seed one.

    ``.env`` at the repo root is the deployment's whole secret store and the
    file every compose invocation is pointed at with ``--env-file``. Without it
    every ``${VAR}`` in every rendered compose file substitutes to empty and the
    stack comes up authenticating with nothing — services that fail closed
    refuse, services that do not come up wide open. Compose says nothing about
    it, which is why this is a refusal rather than a warning.

    On an interactive terminal the operator is offered ``osprey init``'s
    shell-harvest instead: the auth variable this deployment's own provider
    authenticates with, taken from the exported environment and written to
    ``.env`` through the same append-only 0600 writer every other secret path
    uses. Values are never echoed — only the variable name.

    Only the deployment's own provider, deliberately. A persona that
    authenticates elsewhere needs its key in the same file, but that gap belongs
    to ``.env.users`` generation further down the deploy, which names the
    persona and the variable it is missing; anticipating it here would mean
    re-deriving the persona sweep for a prompt whose whole job is to get the
    file into existence.

    Args:
        repo_root: The deployment repo.
        config: The rendered ``build/config.yml``, which names the provider.
        mark: Whether the refusal opens with ``✗``. ``False`` from inside an
            open phase, whose own ``✗`` is already the run's failure marker.
            Defaults to ``True`` because this function is also called on its
            own, with no phase above it to carry the mark.

    Raises:
        click.Abort: When there is no ``.env`` and none was seeded.
    """
    import os

    env_path = repo_root / ".env"
    if env_path.exists():
        return

    from osprey.build.claude_code_resolver import provider_auth_secret_env

    provider = (config.get("claude_code") or {}).get("provider")
    api_providers = (config.get("api") or {}).get("providers")
    secret_var = (
        provider_auth_secret_env(
            provider, api_providers if isinstance(api_providers, dict) else None
        )
        if isinstance(provider, str) and provider
        else None
    )
    exported = os.environ.get(secret_var) if secret_var else None

    if exported and _stdin_is_a_terminal():
        from .phase_reporter import current_reporter

        # The prompt and the reporter want the same terminal: a live region
        # left mounted repaints over the question while the operator is still
        # reading it. Suspended for the prompt only — the seed write below is
        # the verb's own work and belongs back under the reporter.
        with current_reporter().suspended():
            seed_it = click.confirm(
                f"No .env in {repo_root}. Seed one from your shell ({secret_var})?", default=True
            )
        if seed_it:
            from osprey.utils.dotenv import append_profile_env

            append_profile_env(env_path, {secret_var: exported}, _UP_SEEDED_ENV_BANNER)
            _report_fact(f"Seeded {env_path} (mode 0600) with {secret_var}")
            return

    needed = f" It needs {secret_var} for provider {provider!r}." if secret_var else ""
    # The remedy is only "copy the example" when there is one. A repo whose
    # .env.example has been removed would otherwise be told to copy a file that
    # is not there — a small lie that costs an operator a minute of hunting.
    remedy = (
        "cp .env.example .env, then fill it in and re-run"
        if (repo_root / ".env.example").is_file()
        else f"create {env_path} with that variable in it, then re-run"
    )
    _abort(
        f"No .env in {repo_root}",
        f"It is this deployment's only secret store, and every compose invocation "
        f"reads it. Without it the stack starts with every credential "
        f"empty.{needed}\nNothing was started.",
        remedy,
        mark=mark,
    )


def _warn_if_host_networking_is_off(config: dict) -> None:
    """Say up front when Docker Desktop cannot forward the web terminals' port.

    :func:`~osprey.deployment.web_terminals.postup_hooks.warn_if_web_stack_unreachable`
    stays the authority on whether the web tier is actually reachable, because it
    tests the thing itself rather than a setting. The one thing it cannot be is
    early: it runs after the images are built and the containers are up, which on
    a first deploy is a quarter of an hour after the operator could have fixed
    this with one checkbox. So this is the same finding, stated before any of that
    work is done.

    Only a definite "off" earns a word here. A setting that reads as on, a host
    that cannot be asked, a deployment with no web terminals: all silent, and
    left to the post-up probe. A preflight that warned on suspicion would fire on
    every deployment whose web tier is perfectly fine, and an operator who is
    warned about nothing stops reading warnings.

    Advisory, like the probe it front-runs. The backend services do not care
    about this setting, so refusing a deploy over it would be deciding for an
    operator that they wanted the web terminals more than they wanted their
    services running.

    :param config: The as-built deploy config, already loaded by the caller.
    """
    from osprey.deployment.docker_desktop import (
        HOST_NETWORKING_REMEDY,
        host_networking_enabled,
        on_docker_desktop,
    )

    # The same reading of "this deployment has web terminals" the post-up probe
    # uses: a rendered nginx port is what the module's presence amounts to here.
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    nginx_port = web_terminals.get("nginx_port")
    if not isinstance(nginx_port, int):
        return
    if not on_docker_desktop(config):
        return
    if host_networking_enabled() is not False:
        return

    _warn_fact(
        "Docker Desktop will not be able to reach the web terminals",
        f"host networking is turned off in Docker Desktop, so the web terminal on port "
        f"{nginx_port} binds inside the Docker Linux VM and stays unreachable from this "
        "machine. The containers will start and report themselves healthy either way, "
        "and http://127.0.0.1:"
        f"{nginx_port}/ will not load in a browser. Everything else in this deployment "
        "publishes its ports normally and is unaffected.",
        f"{HOST_NETWORKING_REMEDY}, and run this again",
    )


def _preflight(repo_root: Path, reporter: PhaseReporter, *, nothing_done: str) -> None:
    """Check what a start verb needs before it starts anything.

    Reported as one phase because it is one question to the operator: can this
    deployment start at all? Two facts answer it, a ``build/`` to start and the
    secret store every compose invocation is pointed at, and both refusals name
    the same remedies they always did. The phase only adds a ✗ line naming the
    step that stopped.

    One thing here refuses nothing and is reported anyway:
    :func:`_warn_if_host_networking_is_off`. It belongs to this phase rather than
    to the start because its whole value is arriving before the images are built.

    Args:
        repo_root: The deployment repo.
        reporter: The reporter this verb installed.
        nothing_done: The sentence the ``build/`` refusal ends with, spelled for
            the verb the operator typed (``up`` starts nothing, ``restart`` also
            stops nothing).

    Raises:
        click.Abort: When there is no build, or no ``.env`` and none was seeded.
    """
    from osprey.deployment.container_lifecycle import as_built_config_path

    # Both refusals below leave through the open phase, which fails it with its
    # own ✗ on the way out — so neither prints one of its own.
    with reporter.phase("Preflight") as phase:
        config_path = as_built_config_path(repo_root)
        if not config_path.is_file():
            _abort(
                f"No build found at {config_path.parent}",
                nothing_done,
                "run `osprey build` to render it",
                mark=False,
            )
        config = load_project_config(str(config_path), wrap_errors=True)
        ensure_repo_env(repo_root, config, mark=False)
        # Last, so a refusal above still leaves through the phase rather than
        # being preceded by a warning about a deployment that is not going to
        # start at all.
        _warn_if_host_networking_is_off(config)
        phase.done("build/ and .env are in place")


@click.command("up")
@repo_option
@click.option("--detached", "-d", is_flag=True, help="Run services in the background.")
@click.option(
    "--dev",
    is_flag=True,
    help="Start the dev render with freshly built images running the local osprey "
    "checkout. Needs a dev build (osprey build --dev), or --build to chain one.",
)
@click.option(
    "--build",
    "chain_build",
    is_flag=True,
    help="Re-render build/ from profile.yml first, then start it.",
)
@click.option(
    "--as-built",
    "as_built",
    is_flag=True,
    help="Start build/ as it was rendered, even though profile.yml has moved on.",
)
@click.option(
    "--keep-archiver-base",
    is_flag=True,
    help="Keep the existing archiver history even when the profile's retention/cadence knobs no longer match it. Without this, changed knobs rebuild the base series and discard recorded samples.",
)
@click.option(
    "--reuse-stores",
    is_flag=True,
    help="Adopt data volumes left by an earlier deployment of this name, keeping their contents: each store's original credential is restored to .env in place of the one just generated. Only possible while the store's container survives.",
)
@click.pass_context
def up_verb(
    ctx: click.Context,
    repo: Path | None,
    detached: bool,
    dev: bool,
    chain_build: bool,
    as_built: bool,
    keep_archiver_base: bool,
    reuse_stores: bool,
) -> None:
    """Start this deployment from build/, as built.

    Run with no arguments, anywhere inside a deployment repo. It starts what the
    last `osprey build` rendered, and re-renders nothing from profile.yml — so
    the services that come up are always the ones you can read on disk.

    One exception, by design: a deployment with web terminals re-renders that
    stack at every start (its compose file, nginx config, landing page, and any
    persona whose project is missing). Those follow the user roster rather than
    the build, so a roster edit takes effect on the next start.

    It reads profile.yml for one thing: a fingerprint. If the profile has
    changed since the build, up refuses and says what moved, because starting
    would deploy something other than what the profile now describes. Pass
    --build to re-render first, or --as-built to start the old render knowingly.

    Whether this deployment is reachable off-host is a property of the build,
    not of this command: the bind address is rendered into every published port.
    Change it with `osprey set deployment.bind_address=0.0.0.0`, then rebuild.
    The fail-closed service-token rules read what the build actually publishes,
    so an exposed deployment is treated as exposed either way.

    Examples:

    \b
      # Start it, in the background
      $ osprey up -d

    \b
      # Pick up a profile edit, then start
      $ osprey up --build -d

    \b
      # Start the existing render anyway
      $ osprey up --as-built -d

    \b
      # Test local osprey changes in the containers (dev render + start)
      $ osprey up --build --dev
    """
    from osprey.cli.main import lifecycle_reporter
    from osprey.cli.repo_resolver import find_repo_root
    from osprey.cli.summary_card import owns_summary_card, print_summary_card
    from osprey.deployment.container_lifecycle import up_as_built

    repo_root = find_repo_root(repo)
    # Asked before the reporter is installed, so a start chained from `init --up`
    # leaves the card to the verb that owns the run (see `owns_summary_card`).
    owns_card = owns_summary_card()
    with lifecycle_reporter() as reporter:
        # The gate opens no phase of its own: with --build it chains a whole
        # `osprey build`, which reports its own phases into this same reporter,
        # and a phase held open around them would close after them and time the
        # build as part of the start.
        gate_start_from_build(
            ctx, repo_root, chain_build=chain_build, as_built=as_built, verb="up", dev=dev
        )

        _preflight(repo_root, reporter, nothing_done="Nothing was started.")

        # Into the repo for the duration: the compose-file lookup and the runtime's
        # own relative-path handling both resolve against the working directory, and
        # a start verb has to be correct from any directory inside the repo. Restored
        # on the detached path; the attached one os.execvpe-replaces this process.
        previous = Path.cwd()
        os.chdir(repo_root)
        try:
            # Attached starts never close this phase: `up_as_built` hands the
            # terminal to compose with os.execvpe and this process is gone. That
            # is the documented shape — phases up to the exec point, then the
            # live log stream, and no summary card.
            with reporter.phase(f"Starting {repo_root.name}"):
                up_as_built(
                    repo_root,
                    detached=detached,
                    dev_mode=dev,
                    keep_archiver_base=keep_archiver_base,
                    reuse_stores=reuse_stores,
                )
        except DeploymentPreconditionError as e:
            # One handler for every unmet precondition on this path — a missing
            # render, a --dev that cannot be honored, an unreleased pin. They
            # differ only in their reason and remedy, which the renderer reads
            # off the exception. The start phase closed with its own ✗ on the
            # way out, so this block continues that line rather than marking it
            # a second time.
            _abort_unmet_precondition(e, nothing_done="Nothing was deployed.", mark=False)
        except KeyboardInterrupt:
            output.warn("Operation cancelled by user")
            raise click.Abort() from None
        except (click.Abort, click.ClickException):
            raise
        except Exception as e:
            # Some deploy failures carry a cause OSPREY can name and the raw
            # text cannot -- a registry 401 for a password never sent. Local
            # import to keep this module's import cost off every CLI start.
            from osprey.deployment.subprocess_capture import diagnose_captured_failure

            _abort("Deployment failed", str(e), diagnose_captured_failure(e))
        finally:
            os.chdir(previous)

        # Reached only on a detached start that succeeded: every refusal above
        # aborts, and an attached one never comes back from `up_as_built`.
        if owns_card and detached:
            print_summary_card(repo_root, "running")


@click.command("down")
@repo_option
def down_verb(repo: Path | None) -> None:
    """Stop this deployment, keeping all data.

    Run with no arguments, anywhere inside a deployment repo. It stops what the
    last build rendered, in the order that leaves nothing behind: a deployment
    with web terminals has that stack stopped first, because it is a separate
    compose invocation whose containers take host-global names that the next web
    deployment on this machine would otherwise collide with.

    Volumes are kept. Every one of them: the databases, the artifact store, the
    per-user terminal workspaces. Stopping a deployment is not a way to lose its
    data, and destroying data is `osprey reset`, which asks first.

    It renders nothing. If build/ is gone or was never rendered, down does not
    re-derive the compose files from profile.yml to stop with -- those files
    describe what would be started now, not what is running. Instead it stops the
    containers this repo labelled as its own, which is the recovery path for a
    build/ deleted while the stack was up.

    That fallback has one honest limit. Containers are labelled when they are
    CREATED, so a stack started before this version of OSPREY carries no label
    and cannot be found this way. Run `osprey build` to restore build/, and down
    works normally again.

    Examples:

    \b
      # Stop it
      $ osprey down

    \b
      # Stop a deployment you are not standing in
      $ osprey down --repo ~/deployments/my-agent
    """
    from osprey.cli.main import lifecycle_reporter
    from osprey.cli.repo_resolver import find_repo_root
    from osprey.cli.summary_card import print_summary_card
    from osprey.deployment.container_lifecycle import down_deployment

    repo_root = find_repo_root(repo)

    # Into the repo for the duration, for the same reason `up` does it: compose
    # resolves relative paths against the working directory, and a repo-scoped
    # verb has to be correct from any directory inside the repo. The restore in
    # the `finally` actually runs here, unlike on the attached `up` path, which
    # os.execvpe-replaces this process before it can.
    previous = Path.cwd()
    os.chdir(repo_root)
    try:
        with lifecycle_reporter() as reporter:
            # The card is the verb's, not the library's: `down_deployment` is
            # also what `restart` and `reset` stop with, and a "stopped" card
            # printed from inside it would land in the middle of both.
            with reporter.phase(f"Stopping {repo_root.name}"):
                down_deployment(repo_root)
            print_summary_card(repo_root, "stopped")
    except KeyboardInterrupt:
        output.warn("Operation cancelled by user")
        raise click.Abort() from None
    except (click.Abort, click.ClickException):
        raise
    except Exception as e:
        _abort("Could not stop this deployment", str(e))
    finally:
        os.chdir(previous)


@click.command("restart")
@repo_option
@click.option("--detached", "-d", is_flag=True, help="Run services in the background.")
@click.option(
    "--dev",
    is_flag=True,
    help="Start the dev render with freshly built images running the local osprey "
    "checkout. Needs a dev build (osprey build --dev), or --build to chain one.",
)
@click.option(
    "--build",
    "chain_build",
    is_flag=True,
    help="Re-render build/ from profile.yml first, then stop and start.",
)
@click.option(
    "--as-built",
    "as_built",
    is_flag=True,
    help="Restart build/ as it was rendered, even though profile.yml has moved on.",
)
@click.option(
    "--keep-archiver-base",
    is_flag=True,
    help="Keep the existing archiver history even when the profile's retention/cadence knobs no longer match it. Without this, changed knobs rebuild the base series and discard recorded samples.",
)
@click.option(
    "--reuse-stores",
    is_flag=True,
    help="Adopt data volumes left by an earlier deployment of this name, keeping their contents: each store's original credential is restored to .env in place of the one just generated. Checked before the stop, since stopping removes the containers this reads from.",
)
@click.pass_context
def restart_verb(
    ctx: click.Context,
    repo: Path | None,
    detached: bool,
    dev: bool,
    chain_build: bool,
    as_built: bool,
    keep_archiver_base: bool,
    reuse_stores: bool,
) -> None:
    """Stop and start this deployment again.

    Run with no arguments, anywhere inside a deployment repo. It is a stop
    followed by a start, not a container restart: the containers are recreated,
    so a rebuilt image, an edited compose file or a freshly minted token is
    actually picked up. Restarting containers in place would leave every one of
    those changes on the floor.

    Because it ends in a start, it obeys the same rule as `osprey up`. It starts
    what the last build rendered and re-renders nothing from profile.yml, and if
    profile.yml or a file it points at has changed since the build it refuses and
    says what moved, rather than stopping a running stack to bring up something
    you did not build.
    Pass --build to re-render first, or --as-built to restart the old render
    knowingly. Web terminals are the same documented exception they are for up:
    that stack's compose file, nginx config and landing page are re-rendered at
    start, because those three follow the user roster.

    Nothing is stopped until the start is known to be possible. A refused drift
    check, a build with nothing in it, a --dev that cannot stage a wheel: each of
    those leaves the running stack exactly as it was.

    Volumes survive, as they do for down. Service tokens are minted again on the
    way back up, into this repo's .env.

    With --build the stop uses the newly rendered compose files, so a service you
    deleted from profile.yml in that same edit is no longer named in them and
    keeps running. Run `osprey down` before `osprey build` when an edit removes a
    service.

    Examples:

    \b
      # Restart it, in the background
      $ osprey restart -d

    \b
      # Pick up a profile edit, then restart
      $ osprey restart --build -d

    \b
      # Restart the existing render anyway
      $ osprey restart --as-built -d
    """
    from osprey.cli.main import lifecycle_reporter
    from osprey.cli.repo_resolver import find_repo_root
    from osprey.cli.summary_card import owns_summary_card, print_summary_card
    from osprey.deployment.container_lifecycle import restart_deployment

    repo_root = find_repo_root(repo)
    owns_card = owns_summary_card()
    with lifecycle_reporter() as reporter:
        gate_start_from_build(
            ctx, repo_root, chain_build=chain_build, as_built=as_built, verb="restart", dev=dev
        )

        _preflight(repo_root, reporter, nothing_done="Nothing was stopped.")

        previous = Path.cwd()
        os.chdir(repo_root)
        try:
            # One phase for the stop and the start together, because that is what
            # the verb is: `restart_deployment` recreates the containers, and a
            # stop reported as finished while the start is still to come would
            # invite reading the ✓ as "it is down now".
            with reporter.phase(f"Restarting {repo_root.name}"):
                restart_deployment(
                    repo_root,
                    detached=detached,
                    dev_mode=dev,
                    keep_archiver_base=keep_archiver_base,
                    reuse_stores=reuse_stores,
                )
        except DeploymentPreconditionError as e:
            # Same single handler as `up`; the restart phase's ✗ is the run's
            # marker, so the renderer adds none of its own.
            _abort_unmet_precondition(e, nothing_done="Nothing was stopped.", mark=False)
        except KeyboardInterrupt:
            output.warn("Operation cancelled by user")
            raise click.Abort() from None
        except (click.Abort, click.ClickException):
            raise
        except Exception as e:
            _abort("Restart failed", str(e))
        finally:
            os.chdir(previous)

        # Detached only, for the same reason `up` has it: an attached restart
        # ends inside compose's log stream, which the card cannot follow.
        if owns_card and detached:
            print_summary_card(repo_root, "running")


@click.command("status")
@repo_option
@click.option(
    "--agents",
    "show_agents",
    is_flag=True,
    help="Also list which model each of the agent's subagents resolves to.",
)
def status_verb(repo: Path | None, show_agents: bool) -> None:
    """Show what this deployment is doing.

    Run with no arguments, anywhere inside a deployment repo. It reads and
    reports; it starts nothing, stops nothing and renders nothing, so it is safe
    to run against a live stack at any time.

    Four sections. Build says whether build/ still matches profile.yml -- the
    same check `osprey up` refuses on, so a refusal there is never a surprise
    here -- and which version of osprey rendered it. Containers is what the
    container runtime reports, not what compose thinks should exist. Endpoints
    is where the services are declared to answer. Agent is the provider, whether
    its credential can be found, and whether the rendered agent files still
    match the config.

    Containers are matched by the label a build bakes into them, so a second
    checkout of the same deployment on this host is reported as a second
    checkout instead of being folded in. One limit, stated where it matters: a
    container created before this labelling existed carries no label and can
    only be matched by project name. Status says which rows those are.

    Examples:

    \b
      # What is this deployment doing?
      $ osprey status

    \b
      # Same, plus the per-subagent model assignments
      $ osprey status --agents

    \b
      # A deployment you are not standing in
      $ osprey status --repo ~/deployments/my-agent
    """
    from osprey.cli.repo_resolver import find_repo_root
    from osprey.deployment.status_display import show_repo_status

    repo_root = find_repo_root(repo)
    try:
        show_repo_status(repo_root, console=console, styles=Styles, show_agents=show_agents)
    except KeyboardInterrupt:
        output.warn("Operation cancelled by user")
        raise click.Abort() from None
    except (click.Abort, click.ClickException):
        raise
    except Exception as e:
        _abort("Could not report this deployment's status", str(e))


@click.command("logs")
@repo_option
@click.argument("service", required=False)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    help="Keep streaming new output until interrupted.",
)
@click.option(
    "--tail",
    type=int,
    default=None,
    help="Show only the last N lines per container. Default: the runtime's own (all of them).",
)
def logs_verb(repo: Path | None, service: str | None, follow: bool, tail: int | None) -> None:
    """Show this deployment's container logs.

    Run with no arguments, anywhere inside a deployment repo, to see every
    container's output; name a service to see just that one. The service names
    are the ones in the Containers table of `osprey status`.

    This is a thin wrapper: it hands the invocation to your container runtime
    and steps out of the way, so -f streams, Ctrl-C stops it, piping into other
    commands works, and the exit code is the runtime's own.

    Both halves of a deployment are covered -- the services and, when there is
    one, the web-terminal stack -- because they are one compose project even
    though starting them takes two separate invocations.

    Examples:

    \b
      # Everything, from the beginning
      $ osprey logs

    \b
      # Follow one service
      $ osprey logs event-dispatcher -f

    \b
      # The last 50 lines of each container
      $ osprey logs --tail 50
    """
    from osprey.cli.repo_resolver import find_repo_root
    from osprey.deployment.status_display import follow_logs

    repo_root = find_repo_root(repo)
    try:
        follow_logs(repo_root, service=service, follow=follow, tail=tail)
    except DeploymentPreconditionError as e:
        # Keeps its ✗: `logs` reads, so it opens no phase, and a refusal with no
        # phase line above it is the only marker this run will get.
        _abort_unmet_precondition(e)
    except KeyboardInterrupt:
        output.warn("Operation cancelled by user")
        raise click.Abort() from None
    except (click.Abort, click.ClickException):
        raise
    except Exception as e:
        _abort("Could not read this deployment's logs", str(e))
