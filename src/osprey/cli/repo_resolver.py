"""Deployment-repo discovery — the one rule every repo-scoped command uses.

A deployment repo is a directory holding a ``profile.yml`` at its root. Finding
it works the way ``git`` finds its repository: walk up from where the operator
is standing until the marker appears, and act on that repo no matter which
subdirectory the command was typed in.

One rule, in one module, is the point. Three divergent contracts — a
``profile/`` dirname match, an ancestor walk keyed off ``profile/profile.yml``,
and a ``--project`` flag — meant the same command line could act on different
directories depending on which verb ran it, and that an operator had to know
which verb used which rule. Every repo-scoped command now calls
:func:`find_repo_root`, and the ones that need no repo at all are named in
:data:`REPO_FREE_COMMANDS` rather than deciding case by case.

:mod:`osprey.cli.profile_root` is not one of those contracts and is not
replaced by this one. It answers a different question — given a profile FILE
that was named explicitly, is it a facility profile or a persona delta, and
which directory do its relative paths anchor at — which no amount of repo
discovery decides.

The override is :func:`repo_option`'s ``--repo PATH``, which does not bypass the
walk: it substitutes a starting directory for the working directory and the
same rule runs from there. ``--repo`` naming a subdirectory of a repo therefore
resolves to that repo, exactly as standing in the subdirectory would.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import click

#: The marker that makes a directory a deployment repo root. It sits at the
#: repo root rather than one level down: the root is the deployment, and
#: everything beside the manifest — ``data/``, ``personas/``, the CI files —
#: is content the manifest refers to.
PROFILE_FILENAME = "profile.yml"

#: Where ``osprey init --force`` holds the source zone it is replacing while the
#: new one renders. Spelled here, beside the marker it temporarily displaces,
#: because two modules need the same name for opposite reasons: ``init`` creates
#: and consumes the directory, and this module has to RECOGNIZE it — a repo
#: whose zone is held aside has no ``profile.yml`` at its root, so the walk below
#: passes straight over the deployment the operator is standing in.
HELD_SOURCE_ZONE_DIRNAME = ".osprey-replaced-source-zone"

#: Commands that need no deployment repo, so they carry no ``--repo`` and never
#: call :func:`find_repo_root`. ``init`` creates a repo (there is none to find
#: yet); the rest act on the installed framework or the machine, not on a
#: deployment.
REPO_FREE_COMMANDS: frozenset[str] = frozenset(
    {
        "init",
        "eject",
        "vendor",
        "theme-lab",
    }
)

#: Tooling that resolves its own inputs through ``--project`` rather than
#: ``--repo``, so it does not call :func:`find_repo_root` directly. Listed
#: rather than left unmentioned so that "no ``--repo`` here" reads as a recorded
#: decision instead of an oversight.
#:
#: Two different reasons live here. ``skills``, ``knowledge``, ``ariel`` and
#: ``artifacts`` predate the lifecycle surface and were not rewired onto it.
#: ``health`` and ``channel-finder`` are here because a deployment repo is not
#: the only subject they accept: both also act on a RENDERED project directory,
#: which holds a ``config.yml`` at its own root and no ``profile.yml`` anywhere
#: above it, so this module's marker cannot find one. They resolve through
#: :func:`osprey.cli.project_utils.resolve_project_path`, which applies exactly
#: the walk below whenever the directory in hand is not itself a project — which
#: is what makes ``osprey health`` run from ``<repo>/data/raw`` report on that
#: repo, like every repo-scoped verb. ``health`` reports ON a deployment; it was
#: previously filed under :data:`REPO_FREE_COMMANDS`, whose members act on the
#: machine or the installed framework, and that claim was simply false.
SELF_DISCOVERING_COMMANDS: frozenset[str] = frozenset(
    {
        "skills",
        "knowledge",
        "channel-finder",
        "ariel",
        "artifacts",
        "health",
    }
)

#: Commands whose target is a required ARGUMENT, so there is nothing to
#: discover: the operator has already said which profile or which directory to
#: act on. Distinct from :data:`SELF_DISCOVERING_COMMANDS`, which decide for
#: themselves — these decide nothing, which is why no ``--repo`` could improve
#: them. ``profile`` authors a profile at a path it is handed; ``audit`` reads
#: the profile or built project named on its command line. Recorded for the
#: same reason as the sets above: the exemption picture is only useful if it is
#: complete, and a command absent from every set reads as one nobody looked at.
EXPLICIT_TARGET_COMMANDS: frozenset[str] = frozenset(
    {
        "profile",
        "audit",
    }
)

#: Repo-scoped commands that have one documented repo-free *mode*, keyed to the
#: flag selecting it. They still take ``--repo``, because their default mode
#: acts on a repo; the flag is what drops the requirement. Recorded here so the
#: full exemption picture stays readable in one place even though the branch
#: itself belongs to the command that owns the flag.
MODE_EXEMPT_FLAGS: Mapping[str, tuple[str, ...]] = {
    "config": ("--defaults",),
}

_EXEMPT_COMMANDS = REPO_FREE_COMMANDS | SELF_DISCOVERING_COMMANDS | EXPLICIT_TARGET_COMMANDS

_F = TypeVar("_F", bound=Callable[..., Any])


class RepoNotFoundError(click.ClickException):
    """Raised when no deployment repo encloses the directory searched from.

    A :class:`click.ClickException` because every raiser is a CLI command and
    the condition is an operator-facing one — a wrong working directory or a
    wrong ``--repo`` — that deserves the message and exit code 1, not a
    traceback. Callers that want to handle it themselves still can: it is a
    named type, not a bare exception.
    """

    def __init__(
        self, message: str, *, searched_from: Path, held_aside: Path | None = None
    ) -> None:
        super().__init__(message)
        #: The directory the upward walk started from, for callers that report
        #: the failure their own way.
        self.searched_from = searched_from
        #: The deployment repo found mid-replacement, if that is why the walk
        #: came up empty (see :data:`HELD_SOURCE_ZONE_DIRNAME`). Carried on the
        #: exception rather than left to the message, because "there is a repo
        #: here after all" is a fact a caller may need to ACT on: ``osprey
        #: init`` refuses to nest a new deployment inside one, and a repo whose
        #: zone is held aside is still one repo.
        self.held_aside = held_aside


def find_repo_root(start: Path | None = None) -> Path:
    """Find the deployment repo enclosing *start*.

    Walks up from *start* to the first directory holding a ``profile.yml``
    file, so a command run from a nested subdirectory acts on the same repo as
    the same command run from the root. With nested repos the innermost one
    wins — the nearest marker is the one the operator is standing in.

    Args:
        start: Directory to search from. Defaults to the working directory,
            which is what an operator with no ``--repo`` means. Passing the
            value of ``--repo`` here is the whole override mechanism: it moves
            the starting point, it does not skip the walk.

    Returns:
        The absolute, symlink-resolved repo root.

    Raises:
        RepoNotFoundError: If *start* is not an existing directory, or if no
            ancestor of it holds a ``profile.yml``. A directory holding
            :data:`HELD_SOURCE_ZONE_DIRNAME` instead gets its own message: the
            repo is there, its zone is one directory down, and "create one with
            osprey init" would be the wrong advice to give its owner.
    """
    origin = Path.cwd() if start is None else Path(start)
    searched_from = origin.resolve()

    if not searched_from.exists():
        raise RepoNotFoundError(
            f"No such directory: {searched_from}\n\n"
            "--repo must name a directory inside an OSPREY deployment repo.",
            searched_from=searched_from,
        )
    if not searched_from.is_dir():
        raise RepoNotFoundError(
            f"Not a directory: {searched_from}\n\n"
            f"--repo names the repo itself (the directory holding {PROFILE_FILENAME}), "
            "or any directory inside it — not a file.",
            searched_from=searched_from,
        )

    # The walk also notes a repo whose zone is HELD ASIDE — see the refusal
    # below. Innermost wins there too, for the same reason the marker does.
    interrupted: Path | None = None
    for candidate in (searched_from, *searched_from.parents):
        if (candidate / PROFILE_FILENAME).is_file():
            return candidate
        if interrupted is None and (candidate / HELD_SOURCE_ZONE_DIRNAME).is_dir():
            interrupted = candidate

    if interrupted is not None:
        raise RepoNotFoundError(
            f"{interrupted} is a deployment repo whose source zone is held aside.\n\n"
            f"There is no {PROFILE_FILENAME} at its root, but {HELD_SOURCE_ZONE_DIRNAME}/ "
            f"is — and that is where `osprey init --force` keeps the zone it is "
            f"replacing while the new one renders. An init that was killed "
            f"mid-replacement leaves exactly this. Nothing has been lost: the "
            f"zone is in that directory, whole.\n\n"
            f"Re-run the init that was interrupted:\n"
            f"    osprey init {interrupted} --preset <NAME>\n"
            f"It puts the held zone back before it decides anything else, so "
            f"without --force it then stops and leaves the restored zone alone.",
            searched_from=searched_from,
            held_aside=interrupted,
        )

    raise RepoNotFoundError(
        f"No OSPREY deployment repo found: no {PROFILE_FILENAME} in {searched_from} "
        "or any parent directory.\n\n"
        "A deployment repo is the directory holding its profile.yml. Either cd into "
        "one, point at one with --repo PATH, or create one with 'osprey init <name>'.",
        searched_from=searched_from,
    )


def repo_option(func: _F) -> _F:
    """Attach the shared ``--repo PATH`` override to a repo-scoped command.

    One decorator rather than a hand-written option per command: the flag name,
    its type, and its help text are what an operator learns once and expects
    everywhere, so they are defined once. The decorated callback receives
    ``repo: Path | None`` and is expected to pass it straight to
    :func:`find_repo_root`, which applies the default-to-cwd rule.

    A path that does not exist, or names a file, is rejected by Click as a
    usage error before the command body runs — a mistyped flag is a usage
    problem, distinct from standing outside a repo.
    """
    return click.option(
        "--repo",
        "repo",
        type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
        default=None,
        help="Deployment repo to act on. Default: nearest profile.yml at or above cwd.",
    )(func)


def is_repo_free(command_path: str) -> bool:
    """Whether *command_path* runs without resolving a deployment repo.

    The single predicate behind the exemption list, so that "does this command
    need a repo?" is answered from one table rather than re-decided at each
    command. A command is exempt when it, or any command group above it, is
    exempt — ``osprey users seed`` inherits nothing, but ``osprey ariel search``
    is exempt because ``ariel`` is.

    Commands with a repo-free *mode* (see :data:`MODE_EXEMPT_FLAGS`) are not
    exempt: their default behaviour is repo-scoped, so they carry ``--repo``.

    Args:
        command_path: The command as invoked, with or without the ``osprey``
            prefix — ``"osprey users seed"``, ``"users seed"``, and ``"build"``
            are all accepted. Click's ``ctx.command_path`` can be passed
            straight in.

    Returns:
        ``True`` when the command never calls :func:`find_repo_root`.
    """
    parts = command_path.split()
    if parts and parts[0] == "osprey":
        parts = parts[1:]

    return any(" ".join(parts[: i + 1]) in _EXEMPT_COMMANDS for i in range(len(parts)))
