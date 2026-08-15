"""``osprey init`` — create the deployment repo.

This is the one way an OSPREY deployment comes into existence. It writes a git
repository that is the deployment: one directory, four zones.

    als-assistant/
    │  ═ SOURCE — tracked, user-edited ═══════════════
    ├── profile.yml  triggers.yml  README.md
    ├── data/  personas/  web-terminal-context/
    ├── .gitignore  .env.example  ci-extra.yml
    ├── .gitlab-ci.yml  scripts/verify.sh   (with deploy coordinates)
    │  ═ SECRETS — ignored, durable ══════════════════
    ├── .env                       (seeded from the shell, when it has keys)
    │  ═ OUTPUT — ignored, disposable ════════════════
    ├── build/                     (absent until the first `osprey build`)
    │  ═ STATE — ignored, durable ════════════════════
    └── var/agent_data/  var/audit/

The source zone is materialized by the same machinery every other
materialization path uses (:func:`~.profile_cmd._materialize_profile_directory`,
in its repo-root layout). What this module adds is the repo around it: the
anchored three-zone ``.gitignore``, the README explaining the zones, the state
skeleton, the CI emission, and the initial commit.

Usage::

    osprey init                              # in-place, into an empty directory
    osprey init als-assistant --preset control-assistant
    osprey init demo --preset control-assistant --up -d --dev
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import click

from osprey.errors import BuildProfileError
from osprey.utils.logger import get_logger
from osprey.utils.workspace import STATE_ZONE_DIRS

from .profile_conventions import BUILD_OUTPUT_DIR, STATE_DIR
from .repo_resolver import HELD_SOURCE_ZONE_DIRNAME

if TYPE_CHECKING:
    # Annotation only — both modules are imported lazily inside the command
    # body to keep `osprey --help` off the build-profile import chain (the
    # lazy-import budget test in tests/cli/test_main.py pins this).
    from .deploy_scaffold import ScaffoldedFile
    from .profile_cmd import _MaterializedProfile

logger = get_logger("init")

#: Where the post-deploy health check lands in a three-zone repo. The source
#: zone IS the repo root here, so the check sits exactly where the pipeline
#: that invokes it says it does — there is no ``project/`` mirror in between.
REPO_VERIFY_PATH: tuple[str, ...] = ("scripts", "verify.sh")

#: The STATE zone, created empty. Git-ignored, so deliberately WITHOUT a
#: ``.gitkeep``: a marker file there would be ignored too, and would be the one
#: thing ``osprey reset``'s wipe had to work around. A build recreates these
#: when they are absent, so a fresh clone and a reset repo look identical —
#: which is why the pair is imported rather than restated here.
_STATE_DIRS: tuple[str, ...] = STATE_ZONE_DIRS

#: The facility's own CI jobs. The generated pipeline ``include:``s this file,
#: so it is the supported way to extend the pipeline without editing a
#: generated one — and it is written once and never rewritten.
_CI_EXTRA_FILENAME = "ci-extra.yml"


# ---------------------------------------------------------------------------
# The files the repo owns, as opposed to the ones the profile owns
# ---------------------------------------------------------------------------


def _repo_gitignore() -> str:
    """The repo's ``.gitignore`` — one entry per generated or secret zone.

    Every zone entry is ANCHORED with a leading slash, which is the whole
    subtlety of the file: an unanchored ``build/`` or ``.env*`` also matches a
    same-named path anywhere deeper in the tree, including files moved there
    later, and it does it silently. The editor noise at the end is the one
    deliberate exception, being a name pattern rather than a path.
    """
    return f"""\
# This repo is the deployment: the source zone is tracked, and the three
# generated or secret zones below never are. A fresh deployment has a clean
# `git status` from birth.

# OUTPUT — rendered by `osprey build` from the source zone. Regenerable in
# full, so it is never committed.
/{BUILD_OUTPUT_DIR}/

# STATE — the agent's memory, sessions, and audit log. Durable, host-local,
# and nobody else's business.
/{STATE_DIR}/

# The source zone `osprey init --force` is replacing, while the new one
# renders. A successful run removes it; one that is killed outright leaves it,
# and the next `osprey init` puts its contents back. Never committed either
# way — for the seconds it exists it is a second copy of files already tracked.
/{HELD_SOURCE_ZONE_DIRNAME}/

# SECRETS — provider keys you set plus the tokens `osprey up` mints, and the
# lock file the write-back path creates beside them. .env.example carries no
# values and is the single exception.
#
# Every zone entry above is anchored to the repo root with a leading slash. An
# unanchored `{BUILD_OUTPUT_DIR}/` or `.env*` would also swallow a same-named path anywhere
# deeper in the tree — including files moved there later — and it would do it
# silently.
/.env*
!/.env.example

# OS / editor noise. Deliberately unanchored: these are junk at any depth.
.DS_Store
*.swp
*.swo
"""


def _repo_readme(name: str) -> str:
    """The README an operator meets this layout through.

    Its subject is the repo, not the profile: which zone survives what, and the
    handful of commands the deployment is operated with. Everything specific to
    a single key lives in ``profile.yml``'s own comments, where the key is.
    """
    return f"""\
# {name}

This folder is your OSPREY assistant. Everything it is made of lives here, and
the folder name is the assistant's name.

## What is in here

| What | Where | In git? | Kept? |
| --- | --- | --- | --- |
| Your settings | `profile.yml`, `data/`, `personas/` | yes | yes |
| Your API keys | `.env` | no | yes |
| Generated files | `{BUILD_OUTPUT_DIR}/` | no | no, safe to delete |
| The agent's memory and audit log | `{STATE_DIR}/agent_data/`, `{STATE_DIR}/audit/` | no | yes |

In full, the first row is: {_source_zone_prose()}.

`{BUILD_OUTPUT_DIR}/` is generated from your settings every time you run `osprey build`.
Deleting it is always safe: no settings, no keys and no agent memory live there.

## Everyday commands

```bash
osprey build          # turn your settings into something runnable
osprey up -d          # start it in the background
osprey status         # what is running, and is it up to date
osprey logs           # watch the logs
osprey down           # stop it
```

Run these from anywhere inside this folder. They find their way to the top on
their own, so they need no arguments. `--repo PATH` points them somewhere else.

## Changing something

Edit `profile.yml` (or run `osprey set model=sonnet` to change one setting),
then:

```bash
osprey build && osprey up -d
```

`osprey up` starts what `osprey build` last produced. If you change your
settings without rebuilding, `up` stops and tells you what changed, so a
half-finished edit cannot reach a running system. `osprey up --build` does both
steps; `osprey up --as-built` starts the previous build anyway.

## Running it on a server

To run this somewhere other than your own machine, fill in the `deploy:` section
at the end of `profile.yml` (which server, which CI system), then run:

```bash
osprey scaffold ci
```

That writes the pipeline files. Your own extra CI jobs go in `ci-extra.yml`,
which nothing ever overwrites.

## Starting over

```bash
osprey reset          # stops everything, then deletes containers, agent data and {BUILD_OUTPUT_DIR}/
```

`reset` keeps `{STATE_DIR}/audit/` and your API keys. `osprey reset --purge-audit`
deletes the audit log as well; that plus deleting this folder removes it all.

## Backups

Git covers your settings. `{STATE_DIR}/` and `.env` are everything else, so a backup
is a copy of those two, and a restore is:

```bash
git clone <this repo> && tar xf state.tar.gz && osprey build && osprey up -d
```
"""


def _ci_extra_text(name: str) -> str:
    """The starter ``ci-extra.yml`` — an include point with nothing in it yet.

    Written by this command and by nothing else, ever: the pipeline beside it
    is regenerated, and a facility needs one file in the CI surface that is
    safe to edit. The placeholder job exists because an empty file is not valid
    YAML for an ``include:`` to resolve.
    """
    return f"""\
# {name}'s own pipeline jobs.
#
# .gitlab-ci.yml is emitted by `osprey scaffold ci` and will be overwritten the
# next time it runs. This file never is — put anything facility-specific here:
# extra tests, an IOC smoke check, a notification hook. It is included after
# the scaffolded pipeline, so it can also override a job by redefining it under
# the same name.
#
# Example:
#
#   ioc-smoke-test:
#     stage: validate
#     image: python:3.11-slim
#     script:
#       - ./ci/ioc_smoke_test.sh

# Placeholder so the include always parses. Delete it when you add a job.
.facility-jobs-go-here: {{}}
"""


# ---------------------------------------------------------------------------
# Where the repo goes, and whether it may go there
# ---------------------------------------------------------------------------


_IN_PLACE_NOT_EMPTY = (
    "Refusing to initialize in place: {target} is not empty.\n\n"
    "`osprey init` with no DIR writes the deployment into the current "
    "directory, which is the shape a freshly cloned empty repository has. "
    "Name a directory instead — `osprey init <name> --preset <NAME>` — or cd "
    "somewhere empty."
)

_NOT_A_DIRECTORY = "Not a directory: {target}. `osprey init` creates a deployment repo there."

# ---------------------------------------------------------------------------
# Every file `init` writes belongs to exactly one of three categories
# ---------------------------------------------------------------------------
#
# The categories are what make "--force is safe" checkable rather than asserted.
# Each one is a table that DRIVES the code path implementing it, so a file
# cannot be written by a path that no category names — which is exactly how the
# CI pair came to be regenerated by `--force` while the prose promised it was
# untouched. A file in no category is the bug; the test that enumerates a
# rendered repo against these tables is what catches it.
#
#   1. REPLACED   `profile_cmd.MATERIALIZED_SOURCE_ENTRIES` — the source zone.
#                 Drives `_replacing_source_zone`'s hold-aside and `_cleanup`'s
#                 rollback. The ONLY thing `--force` removes, and it removes it
#                 only once the replacement has been rendered and validated.
#   2. WRITE-ONCE :data:`WRITE_ONCE_FILES` below — the repo's own shell.
#                 Drives the write loop in `init`. Authored when absent, never
#                 rewritten, `--force` or not: from the moment they exist they
#                 are the facility's.
#   3. SCAFFOLDED :data:`CI_EMITTED_PATHS` below — the CI pair. Written by the
#                 scaffolding engine under its marker contract and never
#                 forced from here (see `_emit_ci`); regenerating an unmarked
#                 one is `osprey scaffold ci`'s job, with its own `--force`.
#
# Nothing else in the repo is written by this command at all, so everything
# else — `.env`, `var/`, `build/`, `.git` — survives by construction rather
# than by a promise anybody has to maintain.


def _repo_gitignore_for(_name: str) -> str:
    """:func:`_repo_gitignore`, with the uniform signature the table needs.

    The zone paths are the layout's, not the deployment's, so this is the one
    write-once file whose text does not vary with the name.
    """
    return _repo_gitignore()


#: Files ``init`` authors and never rewrites, each with the builder producing
#: its text. This mapping DRIVES the writing — ``init`` loops over it rather
#: than naming the three files again — so a file that is written is a file that
#: is listed, and the ``--force`` promise below cannot describe a set the code
#: does not implement.
WRITE_ONCE_FILES: Mapping[str, Callable[[str], str]] = {
    ".gitignore": _repo_gitignore_for,
    "README.md": _repo_readme,
    _CI_EXTRA_FILENAME: _ci_extra_text,
}

#: Where the scaffolding engine puts the CI pair in a three-zone repo. Spelled
#: here rather than imported because ``deploy_scaffold`` pulls the build-profile
#: chain in with it (TR-2); a test cross-checks these against the engine's own
#: ``CI_OUTPUT_NAMES`` and :data:`REPO_VERIFY_PATH` so the two cannot drift.
CI_EMITTED_PATHS: tuple[str, ...] = (".gitlab-ci.yml", "/".join(REPO_VERIFY_PATH))


def _source_zone_prose() -> str:
    """The SOURCE row of the README's zone table, derived from the categories above.

    The source zone is exactly what a materialization owns
    (:data:`~.profile_cmd.MATERIALIZED_SOURCE_ENTRIES`), plus the repo shell
    ``init`` authors once (:data:`WRITE_ONCE_FILES`), plus the CI pair the
    scaffolding engine emits (:data:`CI_EMITTED_PATHS`) — the same
    derive-from-the-tables rule :data:`PRESERVED_BY_FORCE` follows, for the same
    reason. Naming the zone by hand is how this table came to advertise
    ``scripts/`` (not a materialized entry at all, and never replaced) while
    omitting ``web-terminal-context/`` (which is one, and is), so that the README
    and this module's own docstring described two different repos.

    A materialized entry with no filename suffix is a directory and is shown with
    a trailing slash — ``.env.example`` has one, so the split holds for every
    entry in that table. Write-once and CI entries are files by construction.

    Imported inside the body rather than at module scope: ``profile_cmd`` pulls
    the build-profile chain in with it, which ``osprey --help`` must stay off
    (TR-2), and this is only ever called while a repo is being written.
    """
    from .profile_cmd import MATERIALIZED_SOURCE_ENTRIES

    entries = [
        f"{name}/" if not Path(name).suffix else name for name in MATERIALIZED_SOURCE_ENTRIES
    ]
    entries.extend(WRITE_ONCE_FILES)
    entries.extend(CI_EMITTED_PATHS)
    return ", ".join(f"`{name}`" for name in entries)


#: The durable content ``init`` never writes at all, and therefore never risks.
_UNTOUCHED_BY_INIT: tuple[str, ...] = (".env", ".git", f"{STATE_DIR}/", f"{BUILD_OUTPUT_DIR}/")

#: Everything a re-materialization leaves intact — DERIVED from the categories
#: above rather than declared beside them. A new write-once file appears here
#: the moment it is added to the table that writes it, which is the property a
#: hand-maintained list cannot offer.
PRESERVED_BY_FORCE: tuple[str, ...] = (
    *_UNTOUCHED_BY_INIT,
    *WRITE_ONCE_FILES,
    *CI_EMITTED_PATHS,
)

_PRESERVED_PROSE = ", ".join(PRESERVED_BY_FORCE)

_ALREADY_A_REPO = (
    "{target} is already an OSPREY deployment repo (it has a profile.yml).\n\n"
    "Re-run with --force to re-materialize its source zone from the preset — "
    "which replaces profile.yml, data/, personas/, triggers.yml, "
    "web-terminal-context/, and .env.example, losing any edit to them. To start "
    "over on this name entirely, re-run with --reset: that re-materializes the "
    "source zone the same way AND destroys the previous deployment's "
    "containers, data volumes and images. These are left alone either way: "
    f"{_PRESERVED_PROSE}."
)

_TARGET_NOT_EMPTY = (
    "{target} already exists, is not empty, and is not an OSPREY deployment "
    "repo.\n\n"
    "A deployment repo is one directory that holds nothing else, so this "
    "command will not write into a directory that is already someone's. "
    "Choose an empty or new path."
)

_NESTED_REPO = (
    "Refusing to create a deployment repo inside another one.\n\n"
    "{enclosing} is already an OSPREY deployment repo, and one repo is exactly "
    "one deployment — a nested one would be discovered by whichever profile.yml "
    "the command happened to reach first. For a variant, create a second repo "
    "beside this one from the same preset with different --set values."
)

#: Appended to :data:`_NESTED_REPO` when the enclosing repo has no profile.yml
#: at this instant because an interrupted ``--force`` left its source zone held
#: aside. Without it the refusal names a directory the operator can see is
#: missing its manifest, and reads as simply wrong.
_NESTED_HELD_ASIDE = (
    "Its profile.yml is not there to see right now: an `osprey init --force` "
    "was interrupted and its source zone is held aside inside that directory. "
    "Re-run that init to put it back — nothing has been lost."
)


def _resolve_target(directory: Path | None) -> Path:
    """The repo root this run writes, from the argument or from where we stand.

    With no DIR the deployment is written IN PLACE, into the working directory.
    That is the clone-your-empty-repository-first workflow: the operator made
    the repository on their forge, cloned it, and is standing in it. In-place
    therefore requires the directory to be empty apart from a ``.git`` — with a
    DIR the caller named a target, but without one they only named a location,
    and turning whatever they happened to be standing in into a deployment is
    not a thing to guess at.
    """
    if directory is not None:
        return Path(directory).resolve()

    here = Path.cwd().resolve()
    if any(entry.name != ".git" for entry in here.iterdir()):
        raise click.UsageError(_IN_PLACE_NOT_EMPTY.format(target=here))
    return here


def _refuse_enclosing_repo(target: Path) -> None:
    """Refuse a target that would nest one deployment repo inside another.

    Asked of the target's PARENT, not the target: a target that is itself
    already a repo is the ``--force`` re-materialization case, which
    :func:`_prepare_repo_root` answers, and reporting it as nesting would name
    the wrong problem. The walk starts at the nearest ancestor that exists,
    because ``osprey init a/b/c`` may name three directories at once.

    A parent whose source zone is HELD ASIDE is still a deployment repo, and
    nesting inside one is refused exactly as nesting inside a whole one is. It
    reaches this function as a failed lookup — there is no ``profile.yml`` up
    there for the moment — so the answer comes off
    :attr:`~.repo_resolver.RepoNotFoundError.held_aside` rather than from the
    return value. Letting it through would build a second deployment inside a
    facility's interrupted one, at the moment they are least able to see it.

    Raises:
        click.UsageError: When any ancestor holds a ``profile.yml``, or holds
            the source zone that one was replacing.
    """
    from .repo_resolver import RepoNotFoundError, find_repo_root

    start = target.parent
    while not start.is_dir() and start != start.parent:
        start = start.parent

    try:
        enclosing = find_repo_root(start)
    except RepoNotFoundError as e:
        if e.held_aside is None:
            # The happy path: nothing above us is a deployment.
            return
        raise click.UsageError(
            _NESTED_REPO.format(enclosing=e.held_aside) + f"\n\n{_NESTED_HELD_ASIDE}"
        ) from e
    raise click.UsageError(_NESTED_REPO.format(enclosing=enclosing))


def _prepare_repo_root(target: Path, *, force: bool) -> bool:
    """Settle whether the source zone may be written into ``target``.

    Four cases, and none of them writes or removes anything: the target does not
    exist (the materializer creates it); it exists and is empty apart from a
    ``.git`` (an empty clone, or the operator's own ``mkdir``); it exists and is
    already a deployment repo (``--force`` re-materializes it); or it exists
    and is something else, which is refused.

    Deciding is ALL this does. What ``--force`` replaces is
    :data:`~.profile_cmd.MATERIALIZED_SOURCE_ENTRIES`, and the replacement is
    :func:`_replacing_source_zone`'s to carry out, at the point where a
    replacement exists to put there. The repo's ``.git``, its ``var/`` state,
    and its CI files are outside that set entirely, and its ``.env`` is only
    ever appended to — a key already on file keeps its value, whatever the
    shell exports. So re-running this command over a live deployment can never
    cost a secret or an agent's memory.

    Returns:
        Whether this run is creating the directory, so a failure can undo it.

    Raises:
        click.UsageError: When the target must not be written into.
    """
    if not target.exists():
        return True
    if not target.is_dir():
        raise click.UsageError(_NOT_A_DIRECTORY.format(target=target))

    if not any(entry.name != ".git" for entry in target.iterdir()):
        return False

    if not (target / "profile.yml").is_file():
        raise click.UsageError(_TARGET_NOT_EMPTY.format(target=target))
    if not force:
        raise click.UsageError(_ALREADY_A_REPO.format(target=target))
    return False


def _reinstate_held_source_zone(target: Path) -> None:
    """Put back a source zone a killed ``--force`` run left held aside.

    :func:`_replacing_source_zone` restores what it moved on every exception,
    Ctrl-C included. It cannot restore anything if the process is killed
    outright, and the holding directory
    (:data:`~.repo_resolver.HELD_SOURCE_ZONE_DIRNAME`) exists for as long as a
    materialization takes — so what that leaves is a repo whose source zone is
    intact but one directory down, where nothing can find it: the marker every
    verb discovers a repo by is in the holding directory too.

    This is the only code that puts it back. Every other verb reaches
    :func:`~.repo_resolver.find_repo_root`, which recognizes the same holding
    directory and says what happened — but recognizing is all it does, because a
    read path that repaired the repo underneath ``osprey status`` would be a
    surprise nobody asked for. It names this command as the repair instead.

    Mirrors :func:`~.build_cmd._repair_interrupted_swap`, which answers the same
    question for the render's staging directory.

    THE HELD COPY WINS. An entry standing in the repo AND held aside is a
    killed run's half-written output sitting where the facility's own zone
    belongs, so the output is removed and the held entry moves back over it.
    The rule is worth stating as a rule, because the tempting answer is the
    wrong one: when two copies of a zone exist, keep the one nothing can
    regenerate. Re-running the command reproduces the output; nothing
    reproduces a facility's edits.

    Only the names in :data:`~.profile_cmd.MATERIALIZED_SOURCE_ENTRIES` are
    reinstated — an unrecognized name is not something to move into a repo
    root sight unseen. Neither is it something to delete: the way one gets
    there is a crash under an osprey whose source zone included an entry this
    version has since renamed, which makes it a facility's file that this
    version simply has no place for. So the holding directory is removed only
    when the restore emptied it, and otherwise stays, named, for its owner to
    decide about. It is git-ignored, so leaving it costs nothing.
    """
    import shutil

    from .profile_cmd import MATERIALIZED_SOURCE_ENTRIES

    stash = target / HELD_SOURCE_ZONE_DIRNAME
    if not stash.is_dir():
        return

    for name in MATERIALIZED_SOURCE_ENTRIES:
        held = stash / name
        if not held.exists():
            continue
        destination = target / name
        if destination.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
        else:
            destination.unlink(missing_ok=True)
        held.rename(destination)

    unrecognized = sorted(entry.name for entry in stash.iterdir())
    if unrecognized:
        click.echo(
            f"\n⚠ Put your files back from {stash}, but left "
            f"{', '.join(unrecognized)} in it. This version of osprey has no "
            f"place for anything by those names, and nothing will read them. "
            f"Move them out, or delete that directory yourself.",
            err=True,
        )
        return
    shutil.rmtree(stash, ignore_errors=True)


@contextmanager
def _replacing_source_zone(target: Path, *, active: bool) -> Iterator[None]:
    """Hold the existing source zone aside for the duration of the block.

    ``--force`` replaces the source zone, and the replacement only comes into
    existence INSIDE this block: the preset resolves there, the ``-O``/``--set``
    layers merge there, the persona deltas are emitted there, and the profile
    that was written is validated there — the last of those after files are on
    disk. So the old zone is moved rather than removed. The block returns and
    the holding directory goes; it raises and every entry goes back exactly as
    it was, byte for byte.

    Delete-first could not offer that at any ordering. There is no point in the
    sequence where every way the materialization can fail is already behind it,
    so a mistyped preset would cost a facility its edited ``profile.yml``: the
    clearing has to run somewhere, and everything that validates the operator's
    input runs after it.

    Only the entries in :data:`~.profile_cmd.MATERIALIZED_SOURCE_ENTRIES` move.
    Everything else in the repo — :data:`PRESERVED_BY_FORCE` — this command
    either never writes at all or, in the single case of ``.env``, only ever
    appends to: the materialization adds provider keys the shell exports and
    that the file does not already carry, and never rewrites a value that is
    already there. So no edit of an operator's is at stake in any of it, and
    none of it is this context manager's to hold aside.

    Args:
        target: The repo root whose source zone is being replaced.
        active: Whether this run replaces anything, i.e. ``--force`` over an
            existing repo. False makes the block a plain pass-through: a fresh
            materialization has nothing to hold aside.

    Raises:
        click.ClickException: When the holding directory's own path is occupied
            by something this command cannot clear. Raised before a single
            entry moves, so the repo is exactly as it was.
    """
    import shutil

    from .profile_cmd import MATERIALIZED_SOURCE_ENTRIES

    if not active:
        yield
        return

    # BEFORE the survey, not after: reinstating a killed run's zone puts entries
    # back into the repo, and those entries are precisely the ones this run has
    # to hold aside. Surveying first would leave a facility's restored zone
    # standing where the new one is about to be written.
    _reinstate_held_source_zone(target)

    present = [name for name in MATERIALIZED_SOURCE_ENTRIES if (target / name).exists()]
    if not present:
        yield
        return

    stash = target / HELD_SOURCE_ZONE_DIRNAME
    try:
        stash.mkdir()
    except FileExistsError as e:
        # Only reachable when the reinstate above could not clear the path — it
        # is a file, or its removal failed. Refusing here costs the operator a
        # re-run; moving the zone into a directory whose contents are unknown
        # could cost them the zone, because a rename onto an existing file
        # replaces it silently.
        raise click.ClickException(
            f"{stash} is in the way: `osprey init --force` needs that path to "
            f"hold your current source zone while the new one renders, and "
            f"something is already there. If it is what an interrupted run left, "
            f"its contents are yours — move them somewhere you can read them, "
            f"then re-run."
        ) from e

    for name in present:
        (target / name).rename(stash / name)

    try:
        yield
    except BaseException:
        for name in present:
            # The materializer removes what it wrote before it raises, but it
            # does so best-effort — anything it could not remove is in the way
            # of the entry that has to go back.
            written = target / name
            if written.is_dir():
                shutil.rmtree(written, ignore_errors=True)
            else:
                written.unlink(missing_ok=True)
            (stash / name).rename(written)
        shutil.rmtree(stash, ignore_errors=True)
        raise

    # The replacement is complete, so a holding directory that will not go is
    # not worth failing over — it is a stale copy of files that are all present
    # and correct. It IS worth saying out loud: `.gitignore` covers it in a repo
    # this command wrote, but the file is write-once, so a repo created before
    # this entry existed would sweep it into the next `git add --all`.
    shutil.rmtree(stash, ignore_errors=True)
    if stash.exists():
        click.echo(
            f"\n⚠ Could not remove {stash}. It holds the files that were just "
            f"replaced, so everything in it is an older copy of what is now in "
            f"the repo. Remove it before you commit.",
            err=True,
        )


def _discard_created_root(target: Path, *, created: bool) -> None:
    """Undo the directory this run created, when nothing was left in it.

    Without this a failed run leaves an empty directory behind that the next
    attempt then refuses — the caller having done nothing wrong twice.
    """
    if not created or not target.is_dir():
        return
    try:
        if not any(target.iterdir()):
            target.rmdir()
    except OSError:
        # Cleanup must never mask the failure that brought us here.
        pass


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _enclosing_git_dir(target: Path) -> Path | None:
    """The git repository already covering ``target``, if there is one.

    Includes ``target`` itself: the clone-first workflow puts the deployment
    inside a repository that exists precisely to hold it.
    """
    for candidate in (target, *target.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _bootstrap_git(target: Path, *, no_git: bool) -> str:
    """``git init`` plus an initial commit, and say what happened either way.

    A deployment repo is a repository: the pipeline resolves its ``include:``
    through git, the deploy host gets its copy by cloning, and the source zone
    is the record of what the deployment IS. Committing it here means the
    operator's first ``git status`` is clean, which is the property the
    three-zone ``.gitignore`` exists to give them.

    Nothing is done when a repository already encloses the target. Two
    different situations reach that branch and both want the same answer:
    an empty clone, where the operator will review and commit themselves; and
    an ``osprey init`` run inside some unrelated checkout, where committing
    would add a deployment to somebody else's history.

    Every failure degrades to a note. No git on PATH, a git that errors, no
    configured commit identity — the repo is complete without any of it, so
    none is worth failing the command over.

    Returns:
        One line for the summary.
    """
    if no_git:
        return "Skipped `git init` (--no-git). Run it yourself to keep this in version control."

    enclosing = _enclosing_git_dir(target)
    if enclosing is not None:
        where = "here" if enclosing == target else f"at {enclosing}"
        return (
            f"Found a git repository already {where}, and left it alone. Commit when you are ready."
        )

    import subprocess

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=target, capture_output=True, text=True, timeout=30
        )

    steps = (
        ("init", "--quiet", "--initial-branch", "main"),
        ("add", "--all"),
        ("commit", "--quiet", "-m", "Initial deployment"),
    )
    try:
        for step in steps:
            completed = run(*step)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip().splitlines()
                logger.warning(
                    "`git %s` failed in %s: %s", step[0], target, detail[-1] if detail else ""
                )
                return (
                    f"`git {step[0]}` failed. Everything was created, but nothing is committed yet."
                )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Could not run git in %s: %s", target, e)
        return "No git found, so `git init` was skipped. Run it yourself to keep this in\n  version control."

    return "Started a git repo and made the first commit."


# ---------------------------------------------------------------------------
# --up
# ---------------------------------------------------------------------------


def _chain_up(ctx: click.Context, target: Path, *, detached: bool, dev: bool) -> None:
    """Render the deployment and start it, as ``--up`` promises.

    Both verbs are looked up on the root group rather than imported, so this
    stays one call into the same commands an operator would type — there is no
    second code path that starts a deployment.

    Raises:
        click.ClickException: When either verb is unavailable, or when a flag
            this command accepted has nowhere to go. Both are framework
            problems rather than operator mistakes, so they are named as such
            instead of being dropped in silence.
    """
    import os

    group = ctx.find_root().command
    verbs: dict[str, click.Command | None] = (
        {name: group.get_command(ctx, name) for name in ("build", "up")}
        if isinstance(group, click.Group)
        else {"build": None, "up": None}
    )
    missing = sorted(name for name, command in verbs.items() if command is None)
    if missing:
        raise click.ClickException(
            f"--up cannot run: `osprey {'` and `osprey '.join(missing)}` "
            f"{'is' if len(missing) == 1 else 'are'} not available in this "
            f"installation. The deployment repo was created — build and start "
            f"it once the verb lands."
        )

    build_cmd = verbs["build"]
    up_cmd = verbs["up"]
    assert build_cmd is not None and up_cmd is not None  # guarded by `missing` above
    # Both verbs discover their repo by walking up from the working directory,
    # so standing in it is what makes the chain act on the repo just created.
    # `--repo` is passed as well where the verb declares it, which makes the
    # chained call the same call an operator would type.
    previous = Path.cwd()
    os.chdir(target)
    try:
        ctx.invoke(build_cmd, **_forwarded(build_cmd, target, {"dev": dev}))
        ctx.invoke(up_cmd, **_forwarded(up_cmd, target, {"detached": detached, "dev": dev}))
    finally:
        os.chdir(previous)


def _forwarded(command: click.Command, target: Path, flags: dict[str, bool]) -> dict[str, object]:
    """The keyword arguments to invoke ``command`` with, for the ``--up`` chain.

    A verb that declares no ``--repo`` is fine — the chain runs from inside the
    repo either way. A verb that declares no home for a flag in ``flags`` is
    NOT, and that is judged on the NAME alone, whether or not the operator set
    it. A guard that only fired on a flag someone happened to pass would go
    quiet exactly when it mattered least — the first run after ``up`` renamed
    ``--dev`` — and then, months later, silently start a deployment in the
    wrong mode for whoever finally passed it. In a control-system CLI a
    renamed flag should break the chain loudly, at the first test that runs it.

    Raises:
        click.ClickException: When a flag this chain forwards has nowhere to go.
    """
    declared = {param.name for param in command.params}
    homeless = sorted(name for name in flags if name not in declared)
    if homeless:
        raise click.ClickException(
            f"--up cannot forward {', '.join(homeless)} to `osprey {command.name}`: "
            f"it declares no such option."
        )

    kwargs: dict[str, object] = {name: value for name, value in flags.items() if name in declared}
    if "repo" in declared:
        kwargs["repo"] = target
    return kwargs


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _list_presets_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Eager --list-presets: print the bundled presets and exit before anything parses."""
    if not value or ctx.resilient_parsing:
        return
    from .build_profile import list_presets

    for name in list_presets():
        click.echo(name)
    ctx.exit(0)


@click.command()
@click.argument("directory", required=False, type=click.Path(path_type=Path))
@click.option(
    "--preset",
    default=None,
    metavar="NAME",
    help="Bundled preset to materialize (see --list-presets).",
)
@click.option(
    "--override",
    "-O",
    "overrides",
    multiple=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Layer a YAML file on top of the preset before writing (repeatable).",
)
@click.option(
    "--set",
    "set_pairs",
    multiple=True,
    metavar="KEY.PATH=VALUE",
    help="Inline scalar/list override baked into the emitted profile (repeatable). "
    "RHS parsed as YAML. Top-level shorthands: provider, model, "
    "channel_finder_mode, connector (the control system to talk to — mock, "
    "epics, virtual_accelerator, doocs).",
)
@click.option(
    "--list-presets",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_list_presets_callback,
    help="List bundled preset names and exit.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-materialize the source zone of an existing deployment repo, "
    "discarding edits to profile.yml, data/, personas/, triggers.yml, "
    f"web-terminal-context/, and .env.example. Never touches: {_PRESERVED_PROSE}.",
)
@click.option(
    "--no-git",
    "no_git",
    is_flag=True,
    help="Skip `git init` and the initial commit.",
)
@click.option(
    "--reset",
    "reset",
    is_flag=True,
    help="Start over on this name: destroy the containers, data volumes and "
    "images left by a previous deployment of it, and re-materialize the source "
    "zone as --force does when the repo directory still exists. Removing a "
    "deployment's directory never removed its runtime state — it is keyed on "
    "the project name and outlives the directory — so re-creating under a used "
    f"name inherits its stores. Discards their data. Never touches: {_PRESERVED_PROSE}.",
)
@click.option("--up", "start", is_flag=True, help="Build the deployment and start it.")
@click.option("-d", "--detach", "detached", is_flag=True, help="With --up: run in the background.")
@click.option("--dev", is_flag=True, help="With --up: start in development mode.")
@click.pass_context
def init(
    ctx: click.Context,
    directory: Path | None,
    preset: str | None,
    overrides: tuple[Path, ...],
    set_pairs: tuple[str, ...],
    force: bool,
    no_git: bool,
    reset: bool,
    start: bool,
    detached: bool,
    dev: bool,
) -> None:
    """Create a deployment repo from a preset.

    DIRECTORY is the repository the deployment lives in, and its name is the
    deployment's name. Omit it to initialize the current directory in place,
    which is how a repository cloned empty from a forge is filled in.

    The repo holds four zones — source you edit, secrets, disposable build
    output, and durable state:

    \b
      profile.yml     the manifest; everything the preset configures, explicit
      data/ personas/ triggers.yml  the material it names — yours to edit
      .env            provider keys, seeded from your shell where it has them
      build/          rendered by `osprey build`; gitignored, 100% disposable
      var/            agent memory and audit log; gitignored, durable

    `git init` and an initial commit run at the end, unless a git repository
    already encloses the target or --no-git is given.

    Examples:

    \b
      $ osprey init --list-presets
      $ osprey init als-assistant --preset control-assistant
      $ osprey init demo --preset control-assistant --up -d --dev
    """
    if preset is None:
        raise click.UsageError(
            "Missing --preset. A deployment starts from a bundled preset — run "
            "`osprey init --list-presets` to see them."
        )
    if (detached or dev) and not start:
        raise click.UsageError(
            "-d/--dev only mean something with --up, which is what starts the deployment."
        )

    from .main import lifecycle_reporter
    from .profile_cmd import _directory_derived_name, _materialize_profile_directory
    from .summary_card import print_summary_card

    target = _resolve_target(directory)
    _refuse_enclosing_repo(target)
    # Before anything is decided: a repo whose last `--force` run was killed
    # mid-replacement is a whole repo again, so the refusals below judge the
    # deployment the operator has rather than the wreck of one.
    _reinstate_held_source_zone(target)
    # `--reset` implies `--force`'s file half. Its promise is "start over on
    # this name", and a source zone left standing from the last deployment is
    # not a start over — an edited profile.yml or a file an older preset wrote
    # would carry into the new deployment silently. It is also what lets the
    # flag work at all on the case it exists for: without it, a used name is
    # refused right here, before the reset it asked for could ever run.
    replacing = force or reset
    created = _prepare_repo_root(target, force=replacing)

    # The reporter is installed around the `--up` chain as well as around the
    # creation itself: `_chain_up` invokes `build` and `up`, each of which finds
    # this one already installed and reports into it, so `init --up` reads as
    # one run of phases rather than three commands' worth.
    with lifecycle_reporter() as reporter:
        # The phase opens only once the refusals above are past, so a run that
        # is turned away prints its reason and nothing else.
        with reporter.phase(f"Creating {target.name}") as phase:
            # Everything that can reject this run happens inside the block — the
            # preset resolves in there — so the zone being replaced is held aside
            # for it rather than removed ahead of it.
            try:
                with _replacing_source_zone(target, active=replacing):
                    materialized = _materialize_profile_directory(
                        target,
                        preset,
                        overrides,
                        set_pairs,
                        profile_name=_directory_derived_name(target.name),
                    )
            except BuildProfileError as e:
                # Reaching here means a packaging problem, not a user mistake —
                # the helper raises UsageError for everything the caller could
                # have got wrong. Abort (exit 1) keeps that distinct from usage
                # errors (exit 2).
                _discard_created_root(target, created=created)
                logger.error("✗ %s", e)
                raise click.Abort() from e
            except BaseException:
                _discard_created_root(target, created=created)
                raise
            phase.step(f"settings and data from preset {preset}")

            name = materialized.profile_name
            # Driven off the table rather than three calls written out: the set of
            # files this command authors and the set the --force promise names are
            # then the same object, not two lists that agree today.
            for filename, build_text in WRITE_ONCE_FILES.items():
                _write_if_absent(target / filename, build_text(name))
            for relative in _STATE_DIRS:
                (target / relative).mkdir(parents=True, exist_ok=True)

            # Emitted through the same engine `osprey scaffold ci` re-runs, so a
            # repo created today and one re-scaffolded a year from now carry the
            # same files.
            deploy_files = _emit_ci(target, declared=materialized.deploy_declared)
            phase.step("repo scaffolding")

            # No step of its own: whatever git did or did not do, `_report`
            # below says so in a sentence, and saying it twice in two shapes is
            # exactly the duplication this feature is removing.
            git_note = _bootstrap_git(target, no_git=no_git)

        _report(target, materialized, deploy_files, git_note)

        # After the repo exists and before anything is built or started. The
        # repo root is what `reset_deployment` reads to resolve the project
        # name and plan the removals, so it cannot run earlier; and the state
        # it removes is what a `--up` below would otherwise inherit, so it
        # must not run later.
        if reset:
            # Imported here, not at module scope: this command is lazy-loaded,
            # and `reset` pulls in the whole deployment stack.
            from osprey.deployment.reset import reset_deployment

            with reporter.phase(f"Discarding the previous {target.name}"):
                reset_deployment(target, assume_yes=True, emit=logger.key_info)

            # `reset` removes only what carries this checkout's repo-id label,
            # so a deployment predating that label survives it — correctly, since
            # ownership cannot be proved. But this flag promised a clean start,
            # and continuing without one walks into the stale-volume refusal
            # further down, whose remedy is `osprey reset`: the very thing that
            # just declined. Stop here and name the actual obstacle instead.
            survivors = _surviving_project_resources(target)
            if survivors:
                _abort_incomplete_reset(target, survivors)

        if start:
            _chain_up(ctx, target, detached=detached, dev=dev)

        # The chain's single card, printed by the verb that owns the run: the
        # `build` and `up` inside `_chain_up` find this reporter installed and
        # leave it to here, so `init --up -d` ends with one card describing the
        # deployment that is now running rather than three. An ATTACHED
        # `init --up` never arrives — compose replaced this process.
        print_summary_card(target, "running" if start else "created")


def _surviving_project_resources(target: Path) -> list[str]:
    """Containers and volumes still labelled for this project after a reset.

    Read-only, and label-scoped to the compose project rather than to the
    checkout: the point is to find exactly what ``reset``'s ownership proof
    could NOT account for, so the narrower filter would report nothing every
    time.

    An unreachable runtime yields ``[]`` — a reset that could not run at all has
    already failed loudly, and inventing a second failure here would only bury
    the first.
    """
    from osprey.deployment.compose_generator import resolve_project_name
    from osprey.deployment.reset import RuntimeProbe
    from osprey.deployment.runtime_helper import get_runtime_command

    project = resolve_project_name({"project_name": target.name})
    try:
        probe = RuntimeProbe(get_runtime_command({})[0])
        return [
            f"{resource.kind} {resource.name}"
            for resource in (
                *probe.containers_for_project(project),
                *probe.volumes_for_project(project),
            )
        ]
    except Exception:  # noqa: BLE001 — see docstring: never mask the real failure
        return []


def _abort_incomplete_reset(target: Path, survivors: list[str]) -> None:
    """Stop a ``--reset`` that could not actually clear the name it was given."""
    logger.error(
        "✗ --reset could not clear %s: %d resource(s) of this project remain.\n\n%s\n\n"
        "`osprey reset` removes only what carries this checkout's `com.osprey.repo-id` "
        "label. These predate it — a deployment created before the label existed — so it "
        "cannot prove they are this repo's and declines to touch them, which is what keeps "
        "one checkout from destroying another's.\n\n"
        "Remove them yourself, once, and every later deployment of this name will carry the "
        "label and reset cleanly:\n"
        "    docker ps -aq --filter label=com.docker.compose.project=%s | xargs docker rm -f\n"
        "    docker volume ls -q --filter label=com.docker.compose.project=%s | xargs docker "
        "volume rm\n\n"
        "Nothing was built and nothing was started.",
        target.name,
        len(survivors),
        "\n".join(f"    {line}" for line in survivors),
        target.name,
        target.name,
    )
    raise click.Abort()


def _write_if_absent(path: Path, text: str) -> bool:
    """Write *path* only when nothing is there; report whether it was written.

    These files are the facility's from the moment they exist, so even --force
    leaves them: re-materializing a source zone must not discard a README
    somebody rewrote or a CI job somebody added.
    """
    if path.exists():
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _emit_ci(target: Path, *, declared: bool) -> list[ScaffoldedFile]:
    """Emit the CI pipeline and health check, if the profile says where to deploy.

    Deployment coordinates are opt-in and a fresh profile has none: every preset
    ships the ``deploy:`` block commented out, so a repo created from one is
    complete apart from these two files. That is decided from the materialized
    profile rather than by catching the engine's error, which reports a missing
    block as a failure — correct for ``osprey scaffold ci``, whose entire job it
    is, and wrong for a creation that never promised a pipeline.

    Emission is never forced, whatever ``osprey init --force`` was given.
    ``--force`` is scoped to the source zone, and the CI files are not in it:
    they are in :data:`PRESERVED_BY_FORCE`, which is a promise this function
    would break by threading ``force`` through. Regenerating a pipeline that
    carries no marker of ours — one somebody hand-wrote — is
    ``osprey scaffold ci``'s job, and it has a ``--force`` of its own to say so
    with. Without the marker the engine reports the file and leaves it alone,
    which is what the summary then shows.
    """
    if not declared:
        return []

    from .deploy_scaffold import scaffold_deploy_files

    # Both destinations are the engine's own now — the repo root's profile, and
    # the health check beside it — so this passes neither. The layout has one
    # shape, and a caller able to choose either path could put the check
    # somewhere the emitted pipeline does not look.
    return scaffold_deploy_files(target, force=False)


def _report(
    target: Path,
    materialized: _MaterializedProfile,
    deploy_files: list[ScaffoldedFile],
    git_note: str,
) -> None:
    """List what the new repo holds and which parts are the user's to edit.

    Someone meets this layout for the first time here, so the report names only
    the entries they make decisions about. The plumbing they never open --
    ``.gitignore``, ``.env.example``, ``ci-extra.yml``, ``build/`` -- is in the
    README, which the last row points at.

    Written through the reporter rather than with ``click.echo``: on a terminal
    the reporter's console owns a live region, and a raw write to the same
    stream lands INSIDE it -- half a report under a table that repaints over it.
    See :meth:`~osprey.cli.phase_reporter.PhaseReporter.echo`; off a terminal it
    is byte-for-byte what ``click.echo`` wrote, which is the parity the tests
    pin.
    """
    from .phase_reporter import current_reporter
    from .profile_cmd import _skipped_keys_note

    echo = current_reporter().echo

    echo(f"✓ Created {target.name}")
    echo("")
    for line in _entry_list(target, materialized):
        echo(line)
    echo(f"\n  {git_note}")

    if materialized.skipped_shell_keys:
        # Named rather than dropped in silence: they exported these, and have to
        # be able to account for the omission.
        echo(f"  {_skipped_keys_note(materialized.skipped_shell_keys)}")

    for line in _ci_report(deploy_files, target):
        echo(line)


def _entry_list(target: Path, materialized: _MaterializedProfile) -> list[str]:
    """The entries someone edits, one line each, with what the entry is for."""
    personas = sorted(path.stem for path in (target / "personas").glob("*.yml"))
    rows: list[tuple[str, str]] = [
        ("profile.yml", "your assistant's settings; edit this"),
        ("data/", "channel lists and facility docs; edit these"),
    ]
    if personas:
        rows.append(("personas/", f"one per web login: {', '.join(personas)}"))
    rows += [
        (".env", _env_note(target, materialized)),
        ("README.md", "what everything here does"),
    ]

    width = max(len(name) for name, _ in rows)
    return [f"  {name.ljust(width)}   {note}" for name, note in rows]


def _env_note(target: Path, materialized: _MaterializedProfile) -> str:
    """What happened to the secrets file, in one clause.

    Three outcomes, and the remedy differs: keys were taken from the shell,
    nothing was exported at all, or what was exported belongs to providers this
    assistant does not use.
    """
    from osprey.utils.dotenv import parse_dotenv_file

    env_path = target / ".env"
    if env_path.is_file():
        taken = ", ".join(sorted(parse_dotenv_file(env_path)))
        return f"from your shell: {taken}. Not in git"
    if materialized.skipped_shell_keys:
        return "empty; no key for the providers this assistant uses"
    return "empty; copy .env.example and add your API key"


def _ci_report(emitted: list[ScaffoldedFile], target: Path) -> list[str]:
    """What the CI emission did, when it did anything.

    The no-pipeline case says nothing at all. It is by far the common one on a
    fresh repo, and it is not a problem: running the assistant on a shared
    server is a later job, and someone creating their first deployment has no
    question here to answer. The README covers it when they get there.

    A refusal already names its remedy (the engine's per-file reason ends with
    the ``osprey scaffold ci --force`` re-run), so the trailer only has to say
    why *this* command's ``--force`` is not it: init's flag cannot regenerate a
    CI file (see :func:`_emit_ci`).
    """
    if not emitted:
        return []

    lines = ["\nDeployment files, generated from profile.yml:"]
    for scaffolded in emitted:
        relative = scaffolded.path.relative_to(target)
        if scaffolded.refused:
            lines.append(f"  {relative} not written: {scaffolded.reason}")
        else:
            lines.append(f"  {relative} ({scaffolded.action})")

    if any(scaffolded.refused for scaffolded in emitted):
        lines.append(
            "  (`osprey init` never regenerates a CI file. `osprey scaffold ci --force` does.)"
        )
    return lines
