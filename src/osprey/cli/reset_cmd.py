"""``osprey reset`` — the factory reset, and the only verb that destroys data.

The command layer is deliberately thin: it resolves the repo, hands the work to
:mod:`osprey.deployment.reset`, and turns that module's two refusals into
operator-facing exits. Everything that decides what gets removed — the identity
scoping, the plan, the typed gate — lives in the deployment layer, where it can
be tested without a container runtime and where the plan an operator reads and
the program that runs are the same object.
"""

from __future__ import annotations

from pathlib import Path

import click

from . import output
from .repo_resolver import find_repo_root, repo_option


@click.command("reset")
@repo_option
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    help="Show the removal plan and stop. Nothing is stopped, removed, or written.",
)
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    help="Skip the typed confirmation. The plan is still printed.",
)
@click.option(
    "--purge-audit",
    "purge_audit",
    is_flag=True,
    help="Destroy var/audit/ as well. It is kept by default.",
)
def reset(repo: Path | None, dry_run: bool, assume_yes: bool, purge_audit: bool) -> None:
    """Wipe this deployment back to a fresh state.

    Run with no arguments, anywhere inside a deployment repo. It stops the
    stack, removes the containers and volumes carrying this checkout's identity
    along with the images this deployment built, destroys the agent's memory
    under var/agent_data/, strips the tokens `osprey up` minted out of .env, and
    deletes build/.

    Two things survive, and the plan says so before you confirm:

    \b
      var/audit/   the safety audit log — what the agent was asked to do and
                   what it did. Pass --purge-audit to destroy it too.
      .env         everything in it that a deploy did not mint, which is your
                   provider keys.

    The source zone is never touched: profile.yml, data/, personas/, triggers.yml
    and .git are exactly what they were. After a reset, `osprey build && osprey
    up -d` gives you the same deployment with no state.

    A compose project name comes from this directory's NAME, so two clones or
    worktrees of one deployment share it. Reset removes a container or volume
    only when it also carries this repo's identity label — a hash of its path —
    and refuses outright, removing nothing, when it finds same-named resources
    created from a different path. Renaming or moving this directory changes
    that hash, so its own older resources read as foreign too; the refusal says
    so and still will not remove them.

    Exit status, so `osprey reset && ...` means what it looks like:

    \b
      0   the reset did what it said, or there was nothing to do
      1   you declined, or it refused — nothing was touched
      3   it ran, but this deployment is only partly reset (each
          survivor is named above the exit)

    2 is never this command's: it is what any command line tool returns
    for a bad flag, and a half-reset deployment must not look like a typo.

    Examples:

    \b
      # See what would go, without touching anything
      $ osprey reset --dry-run

    \b
      # Start over
      $ osprey reset

    \b
      # Complete uninstall, from the parent directory afterwards
      $ osprey reset --purge-audit -y && rm -rf <this directory>
    """
    from osprey.deployment.reset import (
        EXIT_CODES,
        PARTIAL_RESET_EXIT_CODE,
        ForeignCheckoutError,
        reset_deployment,
    )

    from .main import lifecycle_reporter

    repo_root = find_repo_root(repo)

    # Installed, but not reported through: reset says what it is doing on its
    # own `emit` callback, which is the plan an operator reads before confirming
    # and the record of what actually went. The reporter is here for what reset
    # reaches — the shared teardown helpers, which capture their subprocesses
    # and need a reporter to report a failure to, and which must stream instead
    # of spooling when `--verbose` was asked for.
    with lifecycle_reporter() as reporter:
        try:
            # The typed destruction confirmation is inside `reset_deployment`,
            # and it reads the terminal through `input()` — which writes its
            # prompt to the real stdout, underneath anything Rich has mounted
            # there. So the region comes down for the call that contains it.
            #
            # The block spans the whole call rather than the prompt alone
            # because the prompt is not reachable from here, and it costs
            # nothing to hold: reset opens no phase and registers no build, so
            # its region is zero-height from the first line to the last. What
            # `suspended()` guarantees — it restarts only what it actually
            # paused, and it is reentrant — is what makes the wider scope safe
            # for anything reset later reaches that does open a phase.
            with reporter.suspended():
                # `output.report`, never the reporter's `emit`: report is
                # echo-class, so it survives the NullReporter the global
                # `--verbose` installs, and the plan is the one thing this verb
                # may never go quiet about. It is also the printer
                # `init --reset` hands the same call, so the plan an operator
                # reads is line-identical on both paths.
                outcome = reset_deployment(
                    repo_root,
                    dry_run=dry_run,
                    assume_yes=assume_yes,
                    purge_audit=purge_audit,
                    emit=output.report,
                )
        except ForeignCheckoutError as e:
            # Shown verbatim rather than summarized: the message is already the
            # whole explanation, and it is carefully scoped to what the labels
            # actually prove. Rewording it here would be how a claim it does not
            # make gets reintroduced.
            raise click.ClickException(str(e)) from None
        except KeyboardInterrupt:
            # An interrupt AT THE PROMPT is handled inside reset_deployment, which
            # can prove nothing ran and says so. Reaching here means the interrupt
            # landed after the confirmation, mid-teardown — so this must not claim
            # the deployment is untouched, because it very likely is not.
            #
            # It exits with the PARTIAL code rather than 1 for the same reason the
            # partial-failure ending does: what a script needs from `$?` is the
            # state on disk, and this path leaves exactly the state a stuck removal
            # leaves. Reporting 1 here would tell a caller "nothing was touched"
            # about a half-wiped deployment.
            #
            # `output.fail` rather than `click.echo(err=True)`: this is the one
            # shape every refusal in the CLI wears, and it puts the notice on
            # stderr verbatim, which is the channel a caller reading stdout for
            # the plan watches for trouble.
            output.fail(
                "Reset interrupted while it was removing things",
                "It does not roll back, so this deployment is in a partly-reset state.",
                "re-run `osprey reset` to finish; it re-reads what is actually there",
            )
            raise click.exceptions.Exit(PARTIAL_RESET_EXIT_CODE) from None
        except RuntimeError as e:
            # ClickException is deliberately NOT re-raised above this: it does not
            # inherit from RuntimeError, so it already passes through untouched.
            raise click.ClickException(str(e)) from None

    # The exit status is this verb's contract in machine-readable form: 0 says
    # the deployment is now what the plan described. Both non-zero endings have
    # already printed their own explanation — the decline notice, or the list of
    # survivors — so this raises Exit rather than a ClickException, which would
    # only restate them under an "Error:" prefix.
    code = EXIT_CODES[outcome]
    if code:
        raise click.exceptions.Exit(code)
