"""Web-terminal roster commands for one deployment repo.

``osprey users`` is the roster half of the lifecycle surface: who has a web
terminal in this deployment, and what happens to their workspace when they
leave. The service half — start, stop, rebuild — belongs to the top-level
lifecycle verbs, and factory teardown belongs to ``osprey reset``.

Every verb here acts on the deployment repo enclosing the working directory
(``--repo`` overrides the starting point, never the walk) and reads the
rendered ``build/config.yml``, which is what the running stack was started
from. Acting on the *source* profile would let these verbs disagree with the
containers they are about to touch.

The destructive verbs keep their typed-confirmation gates: the gate lives in
:mod:`osprey.deployment.web_terminals.lifecycle` beside the removal it guards,
and these commands only forward ``--yes``.

The engines behind these verbs resolve their durable state against the working
directory, which is why :func:`_users_session` chdirs into the repo root rather
than merely passing a path. Four things resolve that way, and each of them
lands in the zone it belongs in — see :func:`_users_session` for the list.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import click

from osprey.cli.output import fail, note, report, warn
from osprey.cli.repo_resolver import PROFILE_FILENAME, find_repo_root, repo_option
from osprey.cli.styles import Styles
from osprey.utils.logger import get_logger

logger = get_logger("users")

#: The rendered config every verb here acts through, relative to the repo root.
#: ``build/`` is the rendered project, so this is the file the running stack was
#: started from — see :data:`osprey.deployment.staleness.BUILD_DIRNAME`.
_RENDERED_CONFIG = "config.yml"

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


class _Repo(NamedTuple):
    """What every roster verb resolves before it runs.

    Attributes:
        root: The deployment repo root, which is also the working directory the
            verb runs in.
        config: The rendered ``build/config.yml`` the engines are handed.
    """

    root: Path
    config: str


#: The two spellings a profile's ``config:`` block can address the roster by.
#: A profile writes ``modules.web_terminals:`` as ONE literal dotted key holding
#: the whole module subtree (the presets say so in a comment, because a nested
#: ``modules:`` mapping would wholesale-replace the rendered subtree). A deeper
#: ``modules.web_terminals.users:`` key is the other legal form — the emitter
#: folds it into the subtree at render time, deeper wins.
_ROSTER_SUBTREE_KEY = "modules.web_terminals"
_ROSTER_USERS_KEY = "modules.web_terminals.users"


def _check_archive_purge(archive: bool, purge: bool) -> None:
    """Reject ``--archive --purge``: a workspace is either kept or it isn't."""
    if archive and purge:
        raise click.UsageError("--archive and --purge are mutually exclusive.")


def _locate_profile_roster(config_block: Any) -> tuple[Any, Any] | None:
    """Find the ``(container, key)`` holding the roster list in a ``config:`` block.

    Returns the node the roster list is stored under and the key it is stored
    at, so the caller edits the entry the profile actually uses instead of
    adding a second one beside it. ``None`` when this profile does not spell the
    roster at all — the roster then comes from an ``extends:`` preset, and there
    is nothing here to edit.
    """
    if not isinstance(config_block, dict):
        return None
    if isinstance(config_block.get(_ROSTER_USERS_KEY), list):
        return config_block, _ROSTER_USERS_KEY
    subtree = config_block.get(_ROSTER_SUBTREE_KEY)
    if isinstance(subtree, dict) and isinstance(subtree.get("users"), list):
        return subtree, "users"
    return None


def _profile_roster_lists(profile_path: Path, user: str) -> bool:
    """Whether ``profile.yml``'s roster still names *user*."""
    import yaml

    try:
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    located = _locate_profile_roster(document.get("config") if isinstance(document, dict) else None)
    if located is None:
        return False
    container, key = located
    return any(_entry_name(entry) == user for entry in container[key])


def _rendered_roster_lists(config_path: str, user: str) -> bool:
    """Whether the rendered ``build/config.yml`` roster still names *user*.

    This is the roster the engine acts on, so it answers "is there anything for
    :func:`decommission_user` to do?" — distinct from what the profile says.
    """
    import yaml

    try:
        document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    modules = document.get("modules") if isinstance(document, dict) else None
    web_terminals = modules.get("web_terminals") if isinstance(modules, dict) else None
    roster = web_terminals.get("users") if isinstance(web_terminals, dict) else None
    return (
        any(_entry_name(entry) == user for entry in roster) if isinstance(roster, list) else False
    )


def _entry_name(entry: Any) -> str | None:
    """A roster entry's username, for either spelling (bare string or mapping)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        name = entry.get("name")
        if isinstance(name, str):
            return name
    return None


class _RosterEdit(NamedTuple):
    """The outcome of a profile roster edit.

    Attributes:
        changed: Whether ``profile.yml`` was rewritten.
        expanded_bare_entries: Whether bare-string roster entries were expanded
            into ``{name, index}`` mappings to freeze their ports. Worth telling
            the operator about: their file gained lines they did not type.
    """

    changed: bool
    expanded_bare_entries: bool


def _drop_user_from_profile_roster(profile_path: Path, user: str) -> _RosterEdit:
    """Remove *user* from the roster in ``profile.yml``, in place.

    The profile is the source of truth: a removal that edits only the rendered
    ``build/config.yml`` is undone by the next ``osprey build``, which would
    re-render the departed user straight back onto the roster and recreate their
    terminal at the next ``up``.

    The edit is made where the roster already lives rather than by writing a
    fresh ``config.modules.web_terminals.users`` key beside it. Both spellings
    render the same config — the emitter folds the deeper key into the subtree —
    but a profile carrying both lists the removed user in one place and omits
    them in the other, and deleting the seemingly-redundant key (the obvious
    tidy-up) would silently bring them back.

    Surviving entries are edited in place, never rebuilt from
    :func:`~osprey.deployment.web_terminals.personas.normalize_users`: that
    normalizer keeps only ``name``/``index`` plus three optional fields and
    drops ``persona``, so writing its output back would silently re-assign every
    surviving user to the default persona.

    Indices are frozen first, exactly as the engine freezes them: a bare-string
    entry takes its list position as an explicit ``index`` so that removing an
    earlier entry cannot shift a later user's ports.

    Args:
        profile_path: The repo's ``profile.yml``.
        user: The roster name being removed.

    Returns:
        A :class:`_RosterEdit`: whether the profile was rewritten, and whether
        doing so expanded bare-string entries into mapping form.
    """
    from ruamel.yaml.comments import CommentedMap

    from osprey.utils.config_writer import _load, _save

    data = _load(profile_path)
    located = _locate_profile_roster(data.get("config") if isinstance(data, dict) else None)
    if located is None:
        return _RosterEdit(changed=False, expanded_bare_entries=False)

    container, key = located
    roster = container[key]

    # Freeze first, drop second: an entry's index must not depend on a position
    # that the removal is about to change. Dicts are mutated in place so each
    # surviving entry keeps its own keys and its own comments.
    expanded = False
    for position, entry in enumerate(roster):
        if isinstance(entry, str):
            roster[position] = CommentedMap({"name": entry, "index": position})
            expanded = True
        elif isinstance(entry, dict) and "index" not in entry:
            entry["index"] = position

    survivors = [
        entry for entry in roster if not (isinstance(entry, dict) and entry.get("name") == user)
    ]
    if len(survivors) == len(roster):
        return _RosterEdit(changed=False, expanded_bare_entries=False)

    # Deleting from the existing list node rather than assigning a new one keeps
    # the block comment that documents the roster attached to it.
    for position in range(len(roster) - 1, -1, -1):
        entry = roster[position]
        if isinstance(entry, dict) and entry.get("name") == user:
            del roster[position]

    _save(profile_path, data)
    return _RosterEdit(changed=True, expanded_bare_entries=expanded)


def _rendered_config_path(repo_root: Path) -> Path:
    """Where the running stack's config lives in a deployment repo."""
    from osprey.deployment.staleness import BUILD_DIRNAME

    return repo_root / BUILD_DIRNAME / _RENDERED_CONFIG


@contextmanager
def _users_session(repo: Path | None, *, announce: bool = True) -> Iterator[_Repo]:
    """Resolve the deployment repo, then run one roster verb inside it.

    Changes into the repo root for the duration, because the roster engines
    resolve what they touch against the working directory rather than against
    the config path they are handed. Four things resolve that way:

    1. The root ``.env`` — ``users env``'s default secrets source, and the
       file the plaintext-password warning reads. Correct at the repo root:
       per TR-3 that single ``.env`` is what the containers mount.
    2. ``.env.auth``, the web terminals' credential store. Compose resolves
       every relative path a compose file spells — the sidecar's ``env_file:``
       among them — against the pinned compose *project directory*, which is the
       repo root, so the repo root is this file's correct home even though the
       compose file that names it is rendered a level down in ``build/``.
    3. ``--archive`` tarballs, written to ``var/web_terminal_archives``. An
       archived workspace is durable state, so it belongs in the ``var/`` zone
       and not in the tracked source zone at the repo root.
    4. The rendered web-terminal artifacts — ``docker-compose.web.yml`` and its
       ``nginx/`` subtree, re-rendered by ``remove`` and read back by the nginx
       reload and the sidecar recreate. These live under ``build/``, and every
       writer and reader of them resolves that destination from the repo root
       rather than from a bare relative path:
       :func:`~osprey.deployment.web_terminals.artifacts.write_web_terminal_artifacts`
       writes into ``<repo>/build``, and
       :func:`~osprey.deployment.web_terminals.provision.web_stack_compose_cmd`
       addresses the file under ``build/`` off the same pinned root. So the
       re-render and the reload act on exactly the files the running stack was
       started from.

    Args:
        repo: Value of ``--repo``, or None to walk up from the working
            directory.
        announce: Print the "Web-terminal roster: VERB" banner. Off for verbs
            whose stdout is data, such as ``env``.

    Yields:
        The repo root and the rendered ``build/config.yml``, which is
        guaranteed to exist.

    Raises:
        click.ClickException: If no deployment repo encloses the search start.
        click.Abort: If the repo has no build, or the verb failed.
    """
    action = click.get_current_context().info_name or "users"
    repo_root = find_repo_root(repo)
    config_path = _rendered_config_path(repo_root)

    if not config_path.is_file():
        fail(
            f"No build found in {repo_root}",
            f"{config_path} does not exist. Web-terminal users are managed through "
            "the rendered config the stack runs from.",
            "run `osprey build` first",
        )
        raise click.Abort()

    if announce:
        report(f"Web-terminal roster: {action} in {repo_root}")

    # Restored afterwards so a verb cannot leave the process somewhere else than
    # it was invoked from. Nothing here defers work past the yield, so the
    # restore is invisible to the engines.
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        yield _Repo(root=repo_root, config=str(config_path))
    except (click.Abort, click.ClickException):
        raise
    except KeyboardInterrupt:
        warn("Operation cancelled by user")
        raise click.Abort() from None
    except Exception as e:
        cause = str(e)
        if os.environ.get("DEBUG"):
            import traceback

            cause = f"{cause}\n{traceback.format_exc()}"
        fail(f"{action} failed", cause)
        raise click.Abort() from None
    finally:
        os.chdir(previous_cwd)


@click.group()
def users() -> None:
    """Manage this repo's web-terminal users.

    A multi-user deployment gives each person on the roster their own web
    terminal, container and workspace volumes. These verbs act on that roster:
    remove one person, clear out the ones who already left, re-seed workspaces,
    rotate a login password, and render the env file the terminals run with.

    Every verb acts on the deployment repo you are standing in; pass --repo to
    act on another one. The roster itself lives in the profile.

    Examples:

    \b
      $ osprey users remove alice --archive
      $ osprey users prune --dry-run
      $ osprey users prune --purge --yes
      $ osprey users seed alice
      $ osprey users passwd alice
      $ osprey users env --output .env.users

    Run 'osprey users VERB --help' for the options a single verb takes.
    """


# ORDER OF OPERATIONS for `remove`, and what a failure leaves behind.
#
# The engine runs FIRST, then the profile is written. That is deliberately the
# opposite of "write the reversible thing before the irreversible one", because
# the typed confirmation lives inside the engine (decommission_user gates on
# confirm_destroy after checking the roster and before any mutation). Writing
# the profile first would commit the roster edit before the operator is asked to
# type the username, so declining would leave the deployment's source of truth
# already changed — breaking the engine's guarantee that a decline is a true
# no-op. Hoisting the gate into this command would mean two typed gates that can
# drift apart, which is worse than either ordering.
#
# The failure states that follow from running the engine first:
#
#   - Gate declined, or the engine fails before mutating: nothing happened at
#     all. The profile write is never reached.
#   - The engine fails partway (it removes the rendered roster entry and
#     re-renders before touching the container): the destruction it completed
#     stands, and the profile still lists the user. Re-running this command
#     converges — the branch below sees the user gone from the rendered roster
#     but still in the profile, skips the engine, and finishes the profile
#     write. Any container or volume the interrupted run left behind is what
#     `osprey users prune` removes.
#   - The engine succeeds but the profile write fails (unwritable file, full
#     disk, a profile.yml that no longer parses): this does not fall to the
#     generic "remove failed" handler, which would read as though nothing had
#     happened. It reports the true state instead — the container and volumes
#     were removed, profile.yml still lists the user, and the next build would
#     put them back — followed by the two remedies, re-running the command
#     (which converges via the branch above) or editing profile.yml by hand.
#     The command still exits non-zero.
#
# So every partial state is reachable by re-running the same command, and no
# partial state silently reverts: the one outcome that must never happen — a
# profile edited for a removal the operator refused — is the one the ordering
# rules out.
@users.command()
@click.argument("user")
@repo_option
@_archive_option
@_purge_option
@_yes_option
def remove(user: str, repo: Path | None, archive: bool, purge: bool, yes: bool) -> None:
    """Remove one user's web-terminal workspace.

    Drops USER from the roster in profile.yml, removes their container, and
    applies the workspace policy: kept by default, --archive tars the volumes
    first, --purge deletes them. Destroying volumes asks you to type the
    username. Run 'osprey build' to carry the roster change into build/.

    The workspace is destroyed first, then profile.yml is updated, so:

    \b
      • Decline the typed confirmation and nothing happens at all.
      • If the run is interrupted partway, re-run this command: it detects the
        half-finished removal and completes just the profile edit.
      • If the workspace was removed but profile.yml could not be written, the
        command says exactly that and exits non-zero. Re-run it, or drop the
        entry from profile.yml by hand.

    Containers or volumes an interrupted run left behind are removed by
    'osprey users prune'.
    """
    # Validated before the session so the check fires without a repo, a build,
    # or the web_terminals package being importable.
    _check_archive_purge(archive, purge)

    with _users_session(repo) as session:
        from osprey.deployment.staleness import check_drift
        from osprey.deployment.web_terminals.lifecycle import decommission_user

        profile_path = session.root / PROFILE_FILENAME
        # A removal whose profile write never landed leaves the user off the
        # rendered roster and on the profile's. The engine has nothing left to
        # act on there and would only raise its "not present" error, so the
        # re-run skips it and finishes the half-done job. When neither roster
        # lists the user the engine IS called, so that its own message — the
        # single definition of that wording — is what the operator reads.
        resuming = not _rendered_roster_lists(session.config, user) and _profile_roster_lists(
            profile_path, user
        )
        if resuming:
            warn(
                f"{user} is already off the deployed roster",
                "Completing the removal in profile.yml. Any container or volume left "
                "behind is removed by `osprey users prune`.",
            )
        else:
            # The engine, unchanged: it owns the typed confirmation and every
            # container/volume effect. Its own roster edit lands in the rendered
            # build/config.yml; that write is now redundant (the profile is what
            # the next build renders from) but harmless, since this command
            # makes the same removal in both files.
            decommission_user(session.config, user, archive=archive, purge=purge, assume_yes=yes)

        try:
            edit = _drop_user_from_profile_roster(profile_path, user)
        except Exception as exc:
            # Deliberately NOT left to the session's generic "✗ remove failed"
            # handler, which would describe the opposite of what happened: by
            # this point the destruction has succeeded and only the bookkeeping
            # failed. Every way this write can fail — an unwritable file, a full
            # disk, a profile.yml that no longer parses — leaves the same true
            # state and has the same way forward, so they are caught together.
            fail(
                f"{user}'s workspace WAS removed, but this repo's profile.yml could not be updated",
                f"{exc}\n"
                "Done: the container and volumes were removed.\n"
                f"Not done: profile.yml still lists {user}, so the next `osprey build` "
                "would put them back on the roster.\n"
                "Re-running notices the half-finished removal and only edits the "
                f"profile; failing that, delete {user}'s entry from profile.yml by hand.",
                f"re-run `osprey users remove {user}`",
            )
            raise click.Abort() from None

        if edit.changed:
            report("")
            report(f"✓ Removed {user} from the roster in {profile_path}", style=Styles.SUCCESS)
            if edit.expanded_bare_entries:
                note(
                    "The remaining roster entries were written out with explicit "
                    "'index:' values, which keeps each survivor's ports exactly where "
                    "they were."
                )
        elif resuming:
            # Unreachable in practice — `resuming` was true because the profile
            # listed the user — but a roster that changed under us must not be
            # reported as a successful edit.
            warn(f"{user} is no longer in this repo's profile.yml roster; nothing to do.")
        else:
            # Said plainly rather than passed over: the containers and volumes
            # are already gone, so silence here would leave the operator
            # believing the removal was complete while the next build re-adds
            # the user and the next up gives them a terminal again.
            warn(
                f"{user} was removed from the deployment, but this repo's profile.yml "
                "does not spell the web-terminal roster",
                "There was nothing to edit there. If the roster comes from an "
                "'extends:' preset, remove the user in the profile by hand, or the "
                "next build will restore them.",
            )

        report(check_drift(session.root).status_line)


@users.command()
@repo_option
@_archive_option
@_purge_option
@_yes_option
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without making changes.",
)
def prune(repo: Path | None, archive: bool, purge: bool, yes: bool, dry_run: bool) -> None:
    """Remove workspaces for users no longer on the roster.

    Finds containers and volumes whose user has left the roster, prints the
    plan, and asks you to type 'prune' before removing anything.
    """
    _check_archive_purge(archive, purge)

    with _users_session(repo) as session:
        from osprey.deployment.web_terminals.lifecycle import prune_users

        prune_users(session.config, dry_run=dry_run, archive=archive, purge=purge, assume_yes=yes)


@users.command()
@click.argument("user", required=False)
@repo_option
def seed(user: str | None, repo: Path | None) -> None:
    """(Re)seed web-terminal workspaces from the roster.

    USER targets one user; omit it to reseed every user on the roster.
    """
    with _users_session(repo) as session:
        from osprey.deployment.web_terminals.seeding import seed_web_terminals

        seed_web_terminals(session.config, user)


@users.command()
@click.argument("user")
@repo_option
def passwd(user: str, repo: Path | None) -> None:
    """Change one web-terminal user's login password.

    Prompts for the new password and ends that user's sessions.
    """
    with _users_session(repo) as session:
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
        rotate_user_password(session.config, user, password)
        # Deliberately does not claim the old sessions are already dead:
        # when nothing has been deployed from this repo yet, the recreate
        # is skipped (with its own warning) and the change takes effect at
        # the next `osprey up`. Stating the consequence without asserting
        # the timing keeps this true on both paths.
        report("")
        report(f"✓ Password changed for {user}", style=Styles.SUCCESS)
        note(
            "Any session they still hold stops working as soon as the sidecar is "
            "running this change."
        )


@users.command("env")
@repo_option
@click.option(
    "--env-file",
    type=click.Path(),
    help="Single secrets file to render from (default: the repo root's .env chain)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Write the result here (mode 0600) instead of to stdout",
)
def env_production(repo: Path | None, env_file: str | None, output: str | None) -> None:
    """Render .env.users to stdout or a file.

    .env.users is the env file every per-user web-terminal container runs
    with. This renders the same subset a deploy would generate, from the same
    two inputs: the rendered deploy config, and the repo root's env files
    (.env.shared then .env, so .env wins on a key both set). It applies no
    rule of its own, so a file rendered here and one generated by 'osprey up'
    cannot disagree. Pass --env-file to render from one named file instead.

    Values come only from those files, never from the surrounding environment,
    so the result depends on what is on disk and nothing else.
    Unlike a deploy, which never overwrites an existing .env.users, an
    explicit --output is taken as an instruction and replaces what is there.

    Examples:

    \b
      $ osprey users env > .env.users
      $ osprey users env --output .env.users
      $ osprey users env --env-file ../secrets/prod.env
    """
    # Resolve caller-supplied paths BEFORE the session chdirs into the repo: a
    # relative path means what the operator typed it against, not whatever the
    # repo root happens to hold.
    env_file_path = Path(env_file).resolve() if env_file else None
    output_path = Path(output).resolve() if output else None

    # No banner: in the default mode stdout IS the rendered file, so anything
    # else printed there would corrupt a `> .env.users` redirect.
    with _users_session(repo, announce=False) as session:
        from osprey.deployment.web_terminals.env_production import (
            USERS_ENV_FILENAME,
            _build_env_production_subset,
            _claude_code_auth_secret_vars,
        )
        from osprey.utils.config import load_project_config
        from osprey.utils.dotenv import (
            ENV_LOCAL_FILENAME,
            ENV_SHARED_FILENAME,
            chain_files,
            format_env_line,
            merge_chain,
            parse_dotenv_file,
        )

        # The repo root, which the session has already chdir'd into: the env
        # chain and everything else durable lives there, not under build/.
        project_root = Path.cwd()
        if env_file_path is not None:
            # An explicitly named file is the whole source, chain or no chain:
            # the operator pointed at it, and rendering from something else
            # would silently answer a different question than the one asked.
            if not env_file_path.is_file():
                fail(
                    f"Cannot render {USERS_ENV_FILENAME}",
                    f"{env_file_path} does not exist.\n"
                    "Dropping the option renders from the repo root's env chain "
                    "instead.",
                    "point --env-file at the secrets file to render from",
                )
                raise click.Abort()
            dotenv = parse_dotenv_file(env_file_path)
            source_desc = str(env_file_path)
        else:
            # The merged chain, the same source a deploy derives from (see
            # ensure_env_production): `.env` wins on a key both files set, and a
            # key only the committed defaults carry still reaches the terminals.
            sources = chain_files(project_root)
            if not sources:
                fail(
                    f"Cannot render {USERS_ENV_FILENAME}",
                    f"{project_root / ENV_LOCAL_FILENAME} does not exist, and neither "
                    f"does {project_root / ENV_SHARED_FILENAME}.\n"
                    "Or point --env-file at the secrets file to render from.",
                    "create .env at the repo root",
                )
                raise click.Abort()
            dotenv = merge_chain(project_root)
            source_desc = " + ".join(str(path) for path in sources)

        deploy_config = load_project_config(session.config)

        required_vars, extra_vars = _claude_code_auth_secret_vars(deploy_config, project_root)
        missing = {var: origin for var, origin in required_vars.items() if var not in dotenv}
        if missing:
            # Warn rather than refuse: whether an absent auth secret is fatal is
            # the deploy path's call (see ensure_env_production, which does
            # refuse). Rendering anyway is what lets an operator SEE the gap.
            needs = "; ".join(f"{origin} needs {var}" for var, origin in missing.items())
            warn(
                f"Rendering without provider auth secret(s) absent from {source_desc}",
                f"{needs}. Web terminals started with this file will fail "
                "authentication unless they authenticate another way.",
            )

        subset = _build_env_production_subset(
            deploy_config, dotenv, {**required_vars, **extra_vars}
        )
        # Same line renderer the deploy path writes through, so a value needing
        # quotes to survive a re-read is rendered identically either way.
        rendered = "".join(f"{format_env_line(key, value)}\n" for key, value in subset.items())

        if output_path is None:
            # ALLOWLISTED raw click.echo, not a renderer primitive: stdout here
            # is a file, byte for byte, the way a `--json` verb's stdout is one
            # document. Nothing may decorate it, and nothing else may print to
            # this stream, which is why `users env` runs with `announce=False`.
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
