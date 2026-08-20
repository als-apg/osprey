"""The ``osprey chat`` verb — talk to this deployment's agent.

Chat is the product's first-class verb: everything else in the lifecycle exists
so that this one starts an agent wired to the right control system, with the
right provider, against the right facility knowledge.

What it launches is ``build/`` **as it was built**. The render is where the
agent's own materials live — ``config.yml``, ``.mcp.json``, the ``.claude/``
tree — so that directory is the working directory the agent CLI is handed, and
the CLI treats its working directory as the project root. Nothing is rendered
here: ``osprey build`` owns rendering, and a chat that quietly re-rendered first
would mean the thing an operator talked to was never the thing ``osprey up``
would deploy.

That makes drift a *warning* rather than a refusal. A start verb refuses on
drift because a stack deployed from a half-finished edit outlives the command
that started it; a chat session is read-oriented and ends when the operator
closes it, so being told "the profile has moved on since this build" is the
useful thing, and being blocked is not. Drift's one refusal is having nothing
to launch at all — no ``build/``, no chat.
"""

from __future__ import annotations

import os

# Module level so tests can scope a stand-in to this module without patching the
# stdlib for every thread in the worker (see tests/cli/_scoped_subprocess.py).
import subprocess
from pathlib import Path

import click

from osprey.cli import output

from .altitude import lift_gate
from .phase_reporter import NullReporter, install_reporter
from .repo_resolver import find_repo_root, repo_option


def _stand_down_output_policy() -> None:
    """Take OSPREY's output policy off the terminal the agent is about to own.

    One switch, called once, for the whole session. Past this point the screen
    belongs to the agent's own interface, and what OSPREY has to say about how a
    verb's output should read stops applying to it: the altitude gate that
    decides which log records are painted comes off, and the phase reporter that
    decorates a verb's progress is replaced by the quiet one. Agent text reaches
    the terminal exactly as the agent wrote it.

    What this does not do is silence the process. The pin it replaces set the
    root logger to CRITICAL, which stopped records from being emitted at all --
    so a warning worth having was dropped rather than merely unpainted, and the
    companion-server failure in :func:`_launch_companion_servers` had to be
    printed by hand to get past it. No level is touched here, so every record is
    still emitted and still reaches every sink the deployment configured.
    """
    lift_gate()
    install_reporter(NullReporter())


def _overlay_repo_env(repo_root: Path) -> None:
    """Load the repo's env chain into ``os.environ``, overriding what is there.

    The SECRETS zone is at the repo root while the render is under ``build/``,
    so the env chain and the ``config.yml`` this verb needs do not live in one
    directory. This overlay is what closes that gap *before* the provider spec
    is resolved: a custom provider's ``base_url: ${ARGO_PROD_URL}`` is expanded
    at spec-resolution time, and the value it expands from is a chain key.

    The chain is loaded in ascending precedence — ``.env.shared`` then ``.env``,
    each with ``override=True`` — so the host-local file wins over the committed
    defaults, and both win over a stale shell export. That is the same
    local-wins precedence every other launch path applies. Every key is copied,
    not a declared subset — the agent CLI expands ``.mcp.json`` ``${VAR}``
    references (Channel Access addressing among them) out of this environment,
    so a narrowed copy would silently mis-address the control system rather
    than fail.

    Copying everything is not the same as *honoring* everything. The invariant
    on both paths out of this overlay is one of provenance: ``.env`` cannot
    introduce or change a backend-selection var, while the inherited shell
    environment is the operator's own and is left alone. With a provider
    configured, ``inject_provider_env`` scrubs ``MANAGED_ENV_VARS`` and
    re-injects them from the resolved spec; with none configured, the caller
    reverts them to their pre-overlay values (see
    :func:`_restore_managed_env`). What survives either way is the non-managed
    remainder that ``.mcp.json`` expansion needs.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from osprey.utils.dotenv import chain_files

    for env_file in chain_files(repo_root):
        load_dotenv(env_file, override=True)


def _snapshot_managed_env() -> dict[str, str | None]:
    """Record every ``MANAGED_ENV_VARS`` value, ``None`` for the unset ones.

    Taken *before* the repo ``.env`` is overlaid, so it is the record of what
    the operator's shell — and nothing else — had to say about the backend.
    """
    from osprey.build.claude_code_resolver import MANAGED_ENV_VARS

    return {var: os.environ.get(var) for var in MANAGED_ENV_VARS}


def _restore_managed_env(snapshot: dict[str, str | None]) -> None:
    """Put the managed vars back the way :func:`_snapshot_managed_env` found them.

    This runs only when no provider is configured, and only after the overlay —
    before it there is nothing to undo. It is a revert by *provenance* rather
    than a scrub: a managed var the ``.env`` introduced or overwrote goes back
    to what the shell had (or to being unset), while one the operator exported
    themselves is left exactly as it was.

    A scrub would be the wrong instrument here. A deployment that configures no
    provider is one that expects the agent to authenticate the way it always
    did — from the shell — so clearing those exports would trade a silent
    redirect for a silent auth failure. What the deployment never sanctioned is
    a ``.env`` key steering the backend, and that is precisely what this undoes.
    """
    for var, value in snapshot.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def _unreadable_config(config_path: Path, exc: Exception) -> click.ClickException:
    """The refusal for a render whose ``config.yml`` will not parse.

    Same shape as "no build found": a build that cannot be read is a build that
    cannot be launched, and the answer to both is to render it again.
    """
    return click.ClickException(
        f"could not read {config_path} — run `osprey build` to render it again.\n\n"
        "`osprey chat` talks to the rendered deployment, and this render's "
        f"config is not valid YAML:\n{exc}"
    )


def _launch_companion_servers(project_dir: Path) -> list[tuple[str, str]]:
    """Launch the companion web servers this build's config enables.

    Sets ``OSPREY_CONFIG``, resets the config cache, then offers every
    registered server the chance to start; each server's ``auto_launch_checker``
    decides whether it actually does.

    Args:
        project_dir: The rendered project — ``build/``, which holds config.yml.

    Returns:
        ``(display_name, url)`` for each server that ended up running.
    """
    from osprey.infrastructure.server_launcher import _launchers, ensure_web_server
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS
    from osprey.utils.workspace import reset_config_cache

    config_file = project_dir / "config.yml"
    if config_file.exists():
        os.environ["OSPREY_CONFIG"] = str(config_file)
    reset_config_cache()

    started: list[tuple[str, str]] = []
    for key, defn in FRAMEWORK_WEB_SERVERS.items():
        try:
            ensure_web_server(key)
            launcher = _launchers[key]
            if launcher._launched:
                host, port = launcher._config_reader()
                started.append((defn.name, f"http://{host}:{port}"))
        except Exception as exc:
            # Fail OPEN — a companion panel that will not start is not a reason
            # to withhold the agent session — but not fail SILENT. Reported
            # here, before the TUI takes the terminal, so the operator learns
            # which panel is absent instead of discovering it as a dead link
            # mid-session.
            output.warn(
                f"{defn.name} did not start",
                f"{type(exc).__name__}: {exc}\n"
                "The session continues without it. Run `osprey web` to see why.",
            )
    return started


def _report_drift(repo_root: Path, build_dir: Path) -> None:
    """Say what the build no longer matches, then let the launch proceed.

    Refusing here is the start verbs' job. What this owes an operator is the
    knowledge that the session they are about to open answers from an older
    render than the profile they last edited — reported before the agent's TUI
    takes the terminal, since afterwards nobody would see it.

    Raises:
        click.ClickException: When there is no build to launch at all.
    """
    from osprey.deployment.staleness import DriftState, check_drift

    report = check_drift(repo_root)

    # NO_BUILD is the one drift state chat cannot warn its way past, and the
    # config.yml check beside it is the same condition read one file deeper: a
    # manifest with no rendered config beside it is a build in name only.
    if report.state is DriftState.NO_BUILD or not (build_dir / "config.yml").is_file():
        raise click.ClickException(
            f"no build found at {build_dir} — run `osprey build` first.\n\n"
            "`osprey chat` talks to the rendered deployment, and there is "
            "nothing rendered yet."
        )

    if report.refuses:
        output.warn(report.message, "Starting the agent against build/ as it was last rendered.")

    # Independent of the verdict above: a build can be a faithful render of the
    # current profile and still predate the installed framework.
    if report.version_skew:
        output.warn(report.version_skew.message)


@click.command()
@repo_option
@click.option("--resume", default=None, help="Resume a previous agent session by ID.")
@click.option("--print", "print_mode", is_flag=True, help="Print the answer and exit.")
@click.option(
    "--effort",
    type=click.Choice(["low", "medium", "high", "max"]),
    default=None,
    help="Reasoning effort. Default: claude_code.effort from the build.",
)
@click.option(
    "--no-pin",
    is_flag=True,
    help="Ignore claude_code.cli_version and use the installed agent CLI.",
)
@click.argument("prompt", nargs=-1)
def chat(
    repo: Path | None,
    resume: str | None,
    print_mode: bool,
    effort: str | None,
    no_pin: bool,
    prompt: tuple[str, ...],
) -> None:
    """Talk to this deployment's agent.

    Starts the agent in the deployment's build/ directory, wired to the
    provider, control system and facility knowledge that build was rendered
    with. PROMPT, when given, is the opening message — with --print it is
    answered and the command exits, which is the shape a script wants.

    Nothing is re-rendered: `osprey build` owns that. When the profile has
    changed since the last build, a warning says so and the session starts
    anyway against the build as it stands.

    Examples:

    \b
      $ osprey chat
      $ osprey chat --print "what is the stored beam current?"
      $ osprey chat --resume SESSION_ID
      $ osprey chat --repo ~/als-assistant
    """
    import sys

    import yaml

    from osprey.build.claude_code_resolver import (
        detect_managed_policy_conflicts,
        format_managed_policy_conflicts,
        inject_provider_env,
        load_provider_spec,
    )
    from osprey.build.claude_code_telemetry import (
        ObservabilityCredentialError,
        telemetry_creds_are_store_issued,
    )
    from osprey.deployment.staleness import BUILD_DIRNAME
    from osprey.utils.claude_launcher import build_claude_launch_argv

    repo_root = find_repo_root(repo)
    build_dir = repo_root / BUILD_DIRNAME

    # ── Provider isolation: inject env block + auth, scrub managed vars ──
    #
    # Managed (enterprise) policy settings outrank the process environment AND
    # the --setting-sources project restriction below, so a policy `env` block
    # setting a provider variable would silently redirect the agent to a backend
    # the deployment did not configure. For a framework driving control systems,
    # refuse to launch rather than start against the wrong provider.
    policy_conflicts = detect_managed_policy_conflicts()
    if policy_conflicts:
        output.fail("Refusing to launch", format_managed_policy_conflicts(policy_conflicts))
        raise SystemExit(1)

    # This call refuses when there is no build, so it has to run before anything
    # with a side effect — the proxy, the companion servers, the environment
    # overlay. Moving it below any of them would start something on behalf of a
    # command that then declines to launch.
    _report_drift(repo_root, build_dir)

    # Before the spec is resolved: `${VAR}` in a provider's base_url expands out
    # of this environment. The snapshot is what tells the two provenances apart
    # afterwards — it has to be taken while the overlay's values are not yet in.
    shell_managed_env = _snapshot_managed_env()
    _overlay_repo_env(repo_root)

    config_path = build_dir / "config.yml"
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise _unreadable_config(config_path, exc) from exc
    cc_config: dict = config.get("claude_code", {}) or {}
    if not effort:
        effort = cc_config.get("effort")

    # load_provider_spec re-reads the rendered config.yml and expands ${VAR} in
    # provider config (e.g. a custom provider's base_url: ${ARGO_PROD_URL})
    # before resolving — so the proxy below gets a real upstream URL, not a
    # literal. The parse above already refused on a corrupt file; the guard is
    # repeated because this is a second, independent read of it.
    try:
        spec = load_provider_spec(build_dir)
    except yaml.YAMLError as exc:
        raise _unreadable_config(config_path, exc) from exc
    except ObservabilityCredentialError as exc:
        # Keep this arm ahead of any broader one added later: it subclasses
        # ValueError, so an `except ValueError` below would swallow it whole.
        #
        # Resolving the provider also resolves the telemetry block, so an
        # observability credential that does not exist yet arrives as a failure
        # to read the provider — and on a deployment that has never run
        # `osprey up`, the store-issued token in the shipped telemetry block is
        # exactly that. Drop telemetry for this session and start; the session
        # is read-oriented, ends when the operator closes it, and losing its
        # traces is not a reason to withhold the agent. Anything else — a
        # credential an operator has to set, or one that is simply blank —
        # keeps raising untouched.
        if not telemetry_creds_are_store_issued(exc):
            raise
        output.warn(
            "Telemetry is off for this session",
            f"`osprey up` issues {', '.join(exc.unresolved_vars)} when it starts the "
            "telemetry store, and this deployment has not been started yet.",
        )
        spec = load_provider_spec(build_dir, include_telemetry=False)

    if spec is None:
        # No provider configured, so inject_provider_env — the only caller of
        # the managed-var scrub — never runs. Without this, the `.env` overlay
        # above would be the one path on which an operator's ANTHROPIC_BASE_URL
        # reaches the agent and silently chooses its backend. Undo the overlay's
        # contribution to those vars and nothing else: the shell keeps
        # authenticating this deployment, and the non-managed keys `.mcp.json`
        # expands from stay.
        _restore_managed_env(shell_managed_env)
    else:
        if spec.auth_secret_env and not os.environ.get(spec.auth_secret_env):
            output.warn(
                f"${spec.auth_secret_env} is not set",
                f"The '{spec.provider}' provider may not authenticate.",
            )
        # The repo root, not the render: `.env` is the durable SECRETS zone and
        # deliberately does not live in the disposable build output.
        #
        # `os.environ` is a MutableMapping rather than a dict, and handing over
        # the real one is the point — the overlay mutates the environment this
        # process will hand to the agent.
        injected = inject_provider_env(
            os.environ,  # type: ignore[arg-type]
            spec,
            project_dir=repo_root,
        )
        if injected:
            output.note(f"Injected: {', '.join(injected)}")
        if spec.auth_secret_env and os.environ.get(spec.auth_env_var):
            output.note(f"Set ${spec.auth_env_var} from ${spec.auth_secret_env}")

        # Start translation proxy for OpenAI-compatible providers
        if spec.needs_proxy and spec.upstream_base_url:
            from osprey.infrastructure.proxy.lifecycle import start_proxy

            proxy_port = start_proxy(
                spec.upstream_base_url,
                os.environ.get(spec.auth_env_var),
            )
            os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
            output.note(f"Translation proxy on :{proxy_port} forwards to {spec.upstream_base_url}")

    # Build the agent CLI args (it uses cwd as the project root — there is no
    # --project-dir flag). When claude_code.cli_version is set,
    # build_claude_launch_argv() returns an ``npx -y @anthropic-ai/claude-code@<v>``
    # prefix instead of a bare ``claude`` so each deployment can pin the CLI
    # version (issue #218). ``--no-pin`` opts out of the pin but not the
    # ``--setting-sources project`` provider isolation.
    args = build_claude_launch_argv(cc_config, no_pin=no_pin)
    if resume:
        args.extend(["--resume", resume])
    if print_mode:
        args.append("--print")
    if effort:
        args.extend(["--effort", effort])
    # Last, and positional: the agent CLI reads a single trailing argument as
    # the opening message, so an unquoted `osprey chat what is the current?` is
    # rejoined into the one message the operator meant rather than forwarded as
    # four arguments of which only the first would be read.
    if prompt:
        args.append(" ".join(prompt))

    # build/ IS the rendered project, and the agent CLI uses the working
    # directory as its project root — this is what points it at this
    # deployment's .mcp.json, .claude/ tree and CLAUDE.md.
    os.chdir(build_dir)

    started_servers = _launch_companion_servers(build_dir)
    if started_servers:
        output.section("Companion servers", started_servers)
        output.report("")

    output.note(f"Launching the agent in {build_dir}...")
    output.report("")

    # Everything this verb had to say is now on screen. The switch takes
    # OSPREY's output policy off the terminal, and the flush empties what is
    # still buffered, before the agent's TUI takes it over.
    _stand_down_output_policy()
    sys.stdout.flush()
    sys.stderr.flush()

    # Companion servers and the translation proxy run in daemon threads, so the
    # parent process must stay alive — always subprocess.run, never os.execvp,
    # which would replace it.
    raise SystemExit(subprocess.run(args).returncode)
