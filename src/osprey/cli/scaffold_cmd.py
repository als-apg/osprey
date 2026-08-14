"""CLI subcommands for the generated half of a deployment repo.

Two jobs, one group, because both are about material a deployment repo
generates rather than authors:

* ``osprey scaffold ci`` emits the repo's CI pipeline and post-deploy health
  check from the profile's ``deploy:`` block — the same engine
  ``osprey init`` runs at creation, so a repo scaffolded years apart carries
  the same two files.
* ``osprey scaffold list|claim|diff|unclaim`` inspect and take over the build
  artifacts a build renders into ``build/``.

Claiming is a *move into the source zone*: the profile is the source of truth
for what it builds, so an artifact an operator wants to own is moved out of the
disposable build zone and into the profile convention directory that owns its
destination, where the next build's convention scan registers it (see
:func:`osprey.cli.build_persistence._register_convention_artifacts`). Nothing
here writes ``scaffold.user_owned`` — a claim that edited the built config.yml
would be the second source of truth this design removes, and the next build
would wipe it with the zone it lives in.

Every verb here is repo-scoped: it takes the shared ``--repo`` and resolves
through :func:`osprey.cli.repo_resolver.find_repo_root`, so all of them act on
the repo the operator is standing in, whichever subdirectory that is.
"""

from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import click

from osprey.cli.profile_conventions import (
    BUILD_OUTPUT_DIR,
    CONVENTION_DIRS,
    ConventionDir,
    EntryShape,
)
from osprey.cli.repo_resolver import find_repo_root, repo_option
from osprey.cli.styles import console
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import ConfigurationError
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog
from osprey.services.build_artifacts.ownership import (
    get_user_owned,
    update_config_remove_user_owned,
    update_manifest_remove_user_owned,
)
from osprey.utils.config import load_project_config
from osprey.utils.logger import get_logger

logger = get_logger("scaffold")

#: The command that renders the build zone again, copying a claimed artifact
#: back into it. One string, no arguments and no ``--force``: the repo IS the
#: deployment, and a build replaces its own build zone wholesale.
_REBUILD_HINT = "osprey build"


def _build_zone(repo: Path | None) -> Path:
    """The build zone of the repo *repo* selects, whether or not it was built.

    The one translation from the resolver's answer — a deployment repo — to what
    the ownership verbs read, which is what the last ``osprey build`` rendered.
    Kept in one function so ``list``, ``claim``, ``diff``, ``unclaim`` and the
    web-terminal verbs cannot end up reading different directories of the same
    repo.

    Existence is deliberately not checked here: ``list`` has something to say
    about a repo that has never been built, and each caller that does need the
    build reports its absence in its own words (see :func:`_load_config`).

    Args:
        repo: The ``--repo`` value, or ``None`` for the working directory.

    Returns:
        ``<repo root>/build``, absolute.

    Raises:
        RepoNotFoundError: If no deployment repo encloses the search path.
    """
    return find_repo_root(repo) / BUILD_OUTPUT_DIR


def _load_config(project_dir: Path) -> dict[str, Any]:
    """Load ``config.yml`` from ``project_dir`` through the shared project loader.

    The single config loader in this module, and deliberately the same one the
    deploy path uses (:func:`osprey.utils.config.load_project_config`), so these
    verbs report on — and render from — exactly the config a deploy would have
    produced.

    The existence check stays ahead of the load, which is this call site's one
    divergence from the shared loader: callers catch ``ClickException`` to mean
    "nothing built here", and ``ConfigBuilder``'s own ``FileNotFoundError``
    advises setting ``CONFIG_FILE``, which is the wrong instruction for a verb
    that resolves its own repo.

    Args:
        project_dir: The repo's build zone.

    Returns:
        The env-var-expanded config mapping.

    Raises:
        click.ClickException: If ``project_dir`` holds no ``config.yml``.
    """
    config_file = project_dir / "config.yml"
    if not config_file.exists():
        raise click.ClickException(
            f"No build at {project_dir}.\n\n"
            "The build zone is disposable, so this repo has either never been built "
            "or was reset. Run 'osprey build' to render it."
        )

    return load_project_config(config_file)


class ScaffoldClaimError(Exception):
    """A claim cannot proceed, with an operator-facing reason.

    Raised by :func:`claim_into_profile` — the callable form of
    ``osprey scaffold claim``, shared with the Scaffold Gallery's host mode —
    so callers that are not a Click command can report the failure in their own
    idiom. Every message names what was refused *and* what to do instead; a
    claim that fails leaves both the project and the profile untouched.
    """


#: Convention directories a claim can move an artifact into. Ownership follows
#: the DESTINATION, not the channel (see ``ownership_canonical`` in
#: :mod:`osprey.cli.build_persistence`), so this is exactly the set of
#: conventions whose destination the ownership system covers: the ``.claude/``
#: subtrees a later render could rewrite or prune, plus ``services/``. The
#: per-user context directory is roster-derived, never claimed by name.
_CLAIMABLE_CONVENTIONS: tuple[ConventionDir, ...] = tuple(
    convention
    for convention in CONVENTION_DIRS
    if not convention.per_user
    and (convention.destination.startswith(".claude/") or convention.destination == "services")
)

_CLAIMABLE_PREFIXES: str = ", ".join(f"{c.source}/" for c in _CLAIMABLE_CONVENTIONS)


@dataclass(frozen=True)
class ClaimTarget:
    """Where one artifact name lives on both sides of a claim.

    Attributes:
        name: The canonical artifact name (``rules/safety``,
            ``commands/osprey/scan``, ``skills/orbit-check``) — the same
            vocabulary the build registers ownership under and a persona delta
            excludes by.
        project_rel: Project-relative posix path of the artifact.
        profile_rel: Profile-relative posix path of its convention slot.
        is_directory: Whether the artifact is a whole directory (a skill, a
            service) rather than a single file.
        entry_noun: What one entry is, for messages ("rule", "skill").
    """

    name: str
    project_rel: str
    profile_rel: str
    is_directory: bool
    entry_noun: str


@dataclass(frozen=True)
class ClaimOutcome:
    """What a completed claim moved, and where.

    Attributes:
        target: The resolved claim target.
        profile_path: Absolute path the artifact now lives at.
        profile_root: The profile root it was moved into.
        is_persona_delta: The project was built from a persona delta, so the
            destination is the delta's *root* profile — every persona built
            from it sees the artifact.
    """

    target: ClaimTarget
    profile_path: Path
    profile_root: Path
    is_persona_delta: bool


def _claimable_convention(category: str) -> ConventionDir | None:
    """Return the claimable convention a name's first segment selects, if any."""
    for convention in _CLAIMABLE_CONVENTIONS:
        if convention.source == category:
            return convention
    return None


def _unclaimable_message(name: str) -> str:
    """Explain why ``name`` has no profile slot, naming what does own it."""
    from osprey.cli.profile_conventions import FRAMEWORK_RENDER_CHANNEL, reserved_path_channel

    artifact = BuildArtifactCatalog.default().get(name)
    if artifact is None:
        return (
            f"Unknown artifact '{name}'.\n"
            f"  Claimable names are <convention>/<path> under: {_CLAIMABLE_PREFIXES}\n"
            "  Run `osprey scaffold list` to see this project's artifacts."
        )
    channel = reserved_path_channel(artifact.output_path) or FRAMEWORK_RENDER_CHANNEL
    return (
        f"'{name}' ({artifact.output_path}) has no profile convention directory: "
        f"it is written by {channel}.\n"
        f"  Only artifacts under {_CLAIMABLE_PREFIXES} can be claimed into a profile."
    )


def _generated_path_message(name: str, target: ClaimTarget) -> str:
    """Explain why a path inside a claimable channel is still not claimable.

    Distinct from :func:`_unclaimable_message`, which says a whole class has no
    channel. Here the channel exists and works — this one *file* is generated,
    so the honest thing is to say so and leave the channel's reputation intact.
    """
    from osprey.cli.profile_conventions import reserved_path_channel

    channel = reserved_path_channel(target.project_rel) or "the build itself"
    convention = target.profile_rel.split("/", 1)[0]
    return (
        f"'{name}' ({target.project_rel}) is generated, not authored: "
        f"it is written by {channel}.\n"
        "  Claiming it would move a generated file into the profile and freeze it "
        "against the build that regenerates it, so it would go on shadowing every "
        "later change to what generates it.\n"
        f"  Change that instead. Other {convention}/ artifacts can still be claimed."
    )


def _refuse_generated(name: str, target: ClaimTarget) -> ClaimTarget:
    """Pass a resolved target through, unless its path is generated.

    The single choke point both spellings reach, so a reserved path cannot be
    claimed by naming it one way when it is refused named the other.

    The paths come from
    :data:`~osprey.cli.profile_conventions.RESERVED_EXACT_PATHS` rather than a
    list here, so this stays one table with one owner: a generated file added
    under a convention directory later is protected the moment it is reserved,
    without a second list to remember.
    """
    from osprey.cli.profile_conventions import RESERVED_EXACT_PATHS

    if target.project_rel in RESERVED_EXACT_PATHS:
        raise ScaffoldClaimError(_generated_path_message(name, target))
    return target


def _target_below(convention: ConventionDir, tail: tuple[str, ...]) -> ClaimTarget:
    """Build the claim target for a path ``tail`` below a convention's destination.

    The single place the shape rules live, so the two ways in — an artifact
    name, and a catalog artifact's output path — cannot drift apart. Each shape
    decides how much of the tail is one artifact and what its name is, matching
    :func:`~osprey.cli.build_persistence.ownership_canonical` exactly:

    * ``DIRECTORY`` — the first tail segment is the whole artifact (a skill is
      its directory, however many files are inside it).
    * ``MARKDOWN`` — the full tail, ``.md`` carried by the path but not by the
      name. Markdown categories legitimately nest (``commands/osprey/scan``).
    * ``FILE`` — the full tail, suffix and all. A hook is known by its
      filename, because that is what ``settings.json`` wires.

    Raises:
        ScaffoldClaimError: For a shape this OSPREY has no rule for. Guessing
            one would move the artifact to a path nothing reads.
    """
    if convention.shape is EntryShape.DIRECTORY:
        rel = tail[0]
        return ClaimTarget(
            name=f"{convention.source}/{rel}",
            project_rel=f"{convention.destination}/{rel}",
            profile_rel=f"{convention.source}/{rel}",
            is_directory=True,
            entry_noun=convention.entry_noun,
        )

    rel = "/".join(tail)
    if convention.shape is EntryShape.MARKDOWN:
        # The name carries no suffix; the path always does. Accept either
        # spelling rather than resolving to a `.md.md` path.
        stem = rel[: -len(".md")] if rel.endswith(".md") else rel
        return ClaimTarget(
            name=f"{convention.source}/{stem}",
            project_rel=f"{convention.destination}/{stem}.md",
            profile_rel=f"{convention.source}/{stem}.md",
            is_directory=False,
            entry_noun=convention.entry_noun,
        )

    if convention.shape is EntryShape.FILE:
        return ClaimTarget(
            name=f"{convention.source}/{rel}",
            project_rel=f"{convention.destination}/{rel}",
            profile_rel=f"{convention.source}/{rel}",
            is_directory=False,
            entry_noun=convention.entry_noun,
        )

    raise ScaffoldClaimError(
        f"This OSPREY does not know how to claim {convention.source}/ entries one "
        f"at a time. Move the {convention.entry_noun} into the profile's "
        f"{convention.source}/ directory yourself."
    )


def _target_for_output_path(output_path: str) -> ClaimTarget | None:
    """The claim target for a project-relative path, or ``None`` if unclaimable.

    Lets a catalog artifact be claimed under the name ``osprey scaffold list``
    prints even where that name is not the artifact's path. Hooks are the case
    that forces this: the catalog calls one ``hooks/cf-feedback-capture`` while
    the file is ``.claude/hooks/osprey_cf_feedback_capture.py``, and ownership
    — like ``settings.json`` — knows it by its filename. Resolving through the
    *destination* keeps the claimed file's name intact and yields the name the
    build will register, whichever spelling the operator typed.
    """
    normalized = PurePosixPath(output_path).as_posix()
    for convention in _CLAIMABLE_CONVENTIONS:
        prefix = f"{convention.destination}/"
        if not normalized.startswith(prefix):
            continue
        tail = PurePosixPath(normalized[len(prefix) :]).parts
        if tail:
            return _target_below(convention, tail)
    return None


def resolve_claim_target(name: str) -> ClaimTarget:
    """Map an artifact name onto its project path and profile convention slot.

    The inverse of the build's ownership derivation: a name is
    ``<convention>/<path below the destination>``, so the project path is the
    convention's destination plus that path and the profile slot is the
    convention source plus the same path, each shape deciding the details (see
    :func:`_target_below`).

    A catalog canonical that differs from its own path resolves through the
    path, so both spellings of a framework hook name the same artifact and the
    returned :attr:`ClaimTarget.name` is always the name ownership uses.

    A claimable channel does not make everything inside it claimable: a
    generated file that the reserved-path table assigns to another writer is
    refused whichever spelling names it, because a claim would freeze it
    against the build that regenerates it.

    Raises:
        ScaffoldClaimError: If the name is malformed, names no claimable
            convention, names a whole convention directory rather than one
            artifact in it, or names a generated path another channel owns.
    """
    trimmed = name.strip().strip("/")
    parts = PurePosixPath(trimmed).parts if trimmed else ()
    if not parts or any(part in (".", "..") for part in parts) or "\\" in trimmed:
        raise ScaffoldClaimError(
            f"'{name}' is not an artifact name. Expected <convention>/<path>, "
            f"e.g. rules/safety or skills/orbit-check ({_CLAIMABLE_PREFIXES})."
        )

    artifact = BuildArtifactCatalog.default().get(trimmed)
    if artifact is not None:
        target = _target_for_output_path(artifact.output_path)
        if target is None:
            raise ScaffoldClaimError(_unclaimable_message(trimmed))
        return _refuse_generated(trimmed, target)

    convention = _claimable_convention(parts[0])
    if convention is None:
        raise ScaffoldClaimError(_unclaimable_message(trimmed))

    rest = parts[1:]
    if not rest:
        raise ScaffoldClaimError(
            f"'{name}' names the whole {convention.source}/ convention directory. "
            f"Claim one {convention.entry_noun} at a time, e.g. {convention.source}/<name>."
        )
    if convention.shape is EntryShape.DIRECTORY and len(rest) > 1:
        # Truncating here would silently claim the enclosing directory. A
        # catalog artifact's path may legitimately point inside one (a skill's
        # SKILL.md), but a name the operator typed is a name, not a path.
        raise ScaffoldClaimError(
            f"'{name}' is nested: {convention.source}/ holds one directory per "
            f"{convention.entry_noun}, so a name has exactly one segment below it "
            f"({convention.source}/{rest[0]})."
        )

    return _refuse_generated(trimmed, _target_below(convention, rest))


#: Why ownership cannot simply be written down wherever the operator is
#: standing — the paragraph every refusal about a missing or unreachable
#: profile has to make. Shared rather than retyped: the Scaffold Gallery
#: refuses on the same grounds when a container has no durable surface, and two
#: hand-written copies of this explanation had already begun to disagree about
#: what the next build does to an edit made in the wrong place.
OWNERSHIP_LIVES_IN_THE_PROFILE: str = (
    "  Ownership lives in the profile: `osprey build` copies its convention\n"
    "  directories into the project and registers what it copied. Recording it\n"
    "  anywhere else would be overwritten by the next build."
)


def still_supplied_by_profile_message(profile_slot: str) -> str:
    """Why releasing an artifact the profile supplies does not release it.

    Ownership of such an artifact is derived, not stored: the build's
    convention scan re-registers it from the profile every time. Dropping the
    project's record of it therefore changes nothing that survives the next
    build, and saying "removed" would be a plain untruth.

    Shared with the Scaffold Gallery, which reaches the same dead end by a
    different route and has to say the same thing. The gallery passes a bare
    path and the CLI passes a styled one, so the sentence is identical and only
    the rendering differs.

    Args:
        profile_slot: Where the profile holds it, already rendered for the
            surface that will print it.
    """
    return (
        f"The profile still supplies it ({profile_slot}), so the next build copies it in "
        "and registers it again.\n"
        "  Remove it from the profile to give it up for good."
    )


def _no_profile_message(project_dir: Path) -> str:
    return (
        f"{project_dir} names no profile, so there is nowhere to claim into.\n\n"
        f"{OWNERSHIP_LIVES_IN_THE_PROFILE}\n\n"
        "  Its manifest records no build_args.profile_path_abs — it was built by an\n"
        "  older OSPREY, or is not a build zone at all.\n\n"
        "  Render it from this repo's profile, then claim again:\n"
        f"    {_REBUILD_HINT}"
    )


def _project_profile_root(project_dir: Path) -> tuple[Path, bool]:
    """Resolve the profile root a project's claims land in.

    Follows the manifest's ``build_args.profile_path_abs`` — the path the build
    actually resolved — and anchors it the way the build does: a persona delta
    anchors at its *root* profile, because convention directories live there
    and a delta can only exclude from them.

    The claim-side reading of
    :func:`~osprey.cli.profile_root.resolve_project_profile`, with
    ``require_profile_file=True`` because a claim is about to write into the
    profile: a directory that no longer holds one is not somewhere an artifact
    may be moved. Each way the resolver can answer "no" gets its own wording
    here — they are different situations with different remedies, and the
    resolver stays free of claim vocabulary.

    Returns:
        ``(profile_root, is_persona_delta)``.

    Raises:
        ScaffoldClaimError: If the project names no profile, the profile it
            names is gone, or its root is a directory a claim must not write.
    """
    import osprey
    from osprey.cli.profile_root import ProjectProfileProblem, resolve_project_profile

    resolved = resolve_project_profile(project_dir, require_profile_file=True)

    if resolved.problem is ProjectProfileProblem.NO_PROFILE_RECORDED:
        raise ScaffoldClaimError(_no_profile_message(project_dir))

    if resolved.problem is ProjectProfileProblem.PROFILE_FILE_MISSING:
        raise ScaffoldClaimError(
            f"The profile this project was built from is missing:\n"
            f"    {resolved.profile_path}\n"
            "  (recorded in the project manifest as build_args.profile_path_abs)\n\n"
            "  Nothing was moved. Restore that profile — or rebuild this project from\n"
            "  the profile you want to own the artifact — and claim again."
        )

    if resolved.root is None:
        # All that is left is a persona delta whose root profile is gone. That
        # error already names the file that has to exist, and says it better
        # than a claim-flavoured rewording would.
        raise ScaffoldClaimError(str(resolved.root_error)) from resolved.root_error

    profile_root, is_delta = resolved.root, resolved.is_persona_delta

    if not profile_root.is_dir():
        raise ScaffoldClaimError(
            f"The profile root {profile_root} does not exist. Nothing was moved."
        )

    package_root = Path(osprey.__file__).resolve().parent
    if profile_root.resolve().is_relative_to(package_root):
        # The build refuses to read conventions from a package directory, so a
        # claim writing one would put the artifact somewhere no build reads —
        # inside the installed wheel, at that.
        raise ScaffoldClaimError(
            f"The profile root {profile_root} is inside the installed osprey package. "
            "Materialize an operator-owned deployment repo first: "
            "osprey init <name> --preset <preset>."
        )

    return profile_root, is_delta


def _same_content(left: Path, right: Path, *, is_directory: bool) -> bool:
    """Whether two artifacts hold identical content (recursively, for directories)."""
    import filecmp

    from osprey.services.build_artifacts.ownership import sha256_directory

    try:
        if is_directory:
            return sha256_directory(left) == sha256_directory(right)
        return filecmp.cmp(left, right, shallow=False)
    except OSError:
        return False


def _already_in_profile_message(target: ClaimTarget, source: Path, destination: Path) -> str:
    lines = [
        f"'{target.name}' is already in the profile:",
        f"    {destination}",
        "",
        "  Nothing was moved. Edit it there — the next build copies it into the project.",
    ]
    if source.exists() and not _same_content(source, destination, is_directory=target.is_directory):
        lines += [
            "",
            f"  The project copy at {target.project_rel} differs from it. A claim never",
            "  overwrites profile material; reconcile the two yourself if the project",
            "  copy is the one you want to keep.",
        ]
    return "\n".join(lines)


def _escaping_symlinks(source: Path) -> list[str]:
    """Symlinks in ``source`` that a move into the profile would break.

    A profile must be self-contained to be reproducible, so
    :func:`~osprey.cli.profile_conventions.validate_convention_sources` refuses
    a convention directory that symlinks outside the profile. A claim that
    moved such a link would report success and then fail *every* later build of
    that profile — with the project copy already gone.

    The artifact itself being a link is always fatal: it cannot resolve inside
    itself. Inside a directory artifact, a relative link pointing at a sibling
    within the same artifact survives the move unchanged, which is exactly what
    the build's own validation permits, so those are left alone.

    Returns:
        Artifact-relative posix paths of the offending links, sorted; ``["."]``
        when the artifact is itself a symlink.
    """
    if source.is_symlink():
        return ["."]
    if not source.is_dir():
        return []
    root = source.resolve()
    return sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_symlink() and not path.resolve().is_relative_to(root)
    )


def _symlink_message(target: ClaimTarget, links: list[str]) -> str:
    """Explain why a symlinked artifact cannot be carried into the profile."""
    lines: list[str]
    if links == ["."]:
        lines = [f"{target.project_rel} is a symlink, not {target.entry_noun} content."]
    else:
        lines = [f"{target.project_rel} contains symlinks pointing outside it:"]
        lines += [f"    {target.project_rel}/{link}" for link in links[:5]]
        if len(links) > 5:
            lines.append(f"    … and {len(links) - 5} more")
    lines += [
        "  A profile must be self-contained to be reproducible, so a build refuses a",
        "  convention directory that symlinks out of the profile. Claiming this as it",
        "  stands would move the link and break every later build of the profile.",
        "",
        "  Replace the link with a copy of what it points at, then claim again.",
        "  Nothing was moved.",
    ]
    return "\n".join(lines)


def _nothing_to_claim_message(project_dir: Path, target: ClaimTarget) -> str:
    known = BuildArtifactCatalog.default().get(target.name) is not None
    lines = [
        f"Nothing to claim: {target.project_rel} does not exist in {project_dir}.",
    ]
    if known:
        lines.append(
            "  The framework knows this artifact but did not render it here — its "
            "config conditions are not met in this project."
        )
    lines.append(
        f"  A claim moves an existing artifact into the profile. To author a new "
        f"{target.entry_noun}, create it in the profile directly at "
        f"{target.profile_rel} and rebuild."
    )
    return "\n".join(lines)


def claim_into_profile(project_dir: Path, name: str) -> ClaimOutcome:
    """Move a project artifact into the profile convention slot that owns it.

    The whole of ``osprey scaffold claim``, callable: resolves the name to a
    project path and profile slot, resolves the profile the project was built
    from, and moves the artifact there — a file as a file, a skill or service as
    a whole directory. Registration is *not* performed here: the next build's
    convention scan derives ``scaffold.user_owned`` from what it copied, so
    writing it now would create the second source of truth the profile model
    removes.

    Every refusal is raised before anything is touched, and every one of them
    is a :class:`ScaffoldClaimError` carrying an operator-facing reason — a
    filesystem error reaching the caller as a bare ``OSError`` would be a
    traceback where a message belongs. The unacceptable outcome is a half-moved
    artifact, so the checks — name resolvable, profile resolvable and present,
    source present, the right shape, and self-contained rather than symlinked,
    destination free — all run first, and the move itself is a single
    :func:`shutil.move`, which copies before it unlinks.

    Args:
        project_dir: Root of the built project holding the artifact.
        name: Canonical artifact name (``rules/safety``, ``skills/orbit-check``).

    Returns:
        What was moved and where.

    Raises:
        ScaffoldClaimError: With an operator-facing reason. Nothing was moved.
    """
    target = resolve_claim_target(name)
    profile_root, is_delta = _project_profile_root(project_dir)

    source = project_dir / target.project_rel
    destination = profile_root / target.profile_rel

    # `exists()` follows symlinks, so a broken link occupying the slot would
    # read as absent and be silently replaced. Occupancy is about what is *at*
    # the path, not what resolves through it.
    if destination.exists() or destination.is_symlink():
        raise ScaffoldClaimError(_already_in_profile_message(target, source, destination))

    if not source.exists():
        raise ScaffoldClaimError(_nothing_to_claim_message(project_dir, target))

    if target.is_directory and not source.is_dir():
        raise ScaffoldClaimError(
            f"{target.project_rel} is a file, but a {target.entry_noun} is a whole "
            f"directory. Nothing was moved."
        )
    if not target.is_directory and not source.is_file():
        raise ScaffoldClaimError(
            f"{target.project_rel} is not a file, but a {target.entry_noun} is a single "
            f"markdown file. Nothing was moved."
        )

    escaping = _escaping_symlinks(source)
    if escaping:
        raise ScaffoldClaimError(_symlink_message(target, escaping))

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScaffoldClaimError(
            f"Could not create the profile directory {destination.parent}: {exc}\n"
            "  Nothing was moved. Clear whatever occupies that path — a file where a "
            "directory belongs, or a permission that denies writing it — then claim again."
        ) from exc

    try:
        shutil.move(str(source), str(destination))
    except OSError as exc:
        raise ScaffoldClaimError(
            f"Could not move {source} → {destination}: {exc}\n"
            "  Check both paths before retrying — the artifact is at one of them."
        ) from exc

    return ClaimOutcome(
        target=target,
        profile_path=destination,
        profile_root=profile_root,
        is_persona_delta=is_delta,
    )


def profile_root_or_none(project_dir: Path) -> Path | None:
    """The profile root a project's artifacts are owned from, or ``None``.

    The read-only counterpart of :func:`_project_profile_root`: where that one
    refuses with a reason because it is about to write, this one answers the
    question "is a profile reachable from here?" and lets the caller branch.
    ``None`` covers every way the answer is no — no profile recorded, the
    recorded one missing, a package directory — because to a caller choosing a
    strategy they are the same answer.

    Two callers rely on that distinction being a *routine* one, not an error:
    ``osprey scaffold list``, since a project with no profile still has
    artifacts worth listing, and the Scaffold Gallery, which reads ``None`` in
    a deployed container (the manifest's ``profile_path_abs`` is a build-machine
    path that does not exist there) as the signal to use its durable volume
    instead of the profile.

    Args:
        project_dir: Root of a built project.

    Returns:
        The profile root directory, or ``None`` if none resolves.
    """
    try:
        return _project_profile_root(project_dir)[0]
    except ScaffoldClaimError:
        return None


def profile_slot_for(project_dir: Path, name: str) -> Path | None:
    """Where an artifact lives — or would live — in the project's profile.

    Pure resolution: nothing is read from the artifact, nothing is moved, and
    the returned path need not exist. This is the mapping
    :func:`claim_into_profile` moves *to*, published so that callers which must
    follow a claimed artifact — the Scaffold Gallery reading and saving the
    profile's copy rather than the project copy a claim removed — share this
    module's convention and suffix rules instead of keeping a second copy of
    them that can drift.

    Returns ``None`` rather than raising when no profile resolves, mirroring
    :func:`profile_root_or_none`: a caller asking where the slot *is* is
    choosing a strategy, and "there is no profile" is one of the answers. An
    unresolvable *name* still raises, because that is a caller error rather
    than a property of the project — the same contract, and the same message,
    :func:`claim_into_profile` would refuse with.

    Args:
        project_dir: Root of a built project.
        name: Canonical artifact name, in either spelling a claim accepts
            (``rules/safety``, ``hooks/cf-feedback-capture``,
            ``hooks/osprey_cf_feedback_capture.py``).

    Returns:
        Absolute path to the artifact's profile convention slot, or ``None``
        when the project has no resolvable profile.

    Raises:
        ScaffoldClaimError: If the name is malformed or names nothing
            claimable.
    """
    target = resolve_claim_target(name)
    profile_root = profile_root_or_none(project_dir)
    if profile_root is None:
        return None
    return profile_root / target.profile_rel


@click.group(name="scaffold", invoke_without_command=True)
@click.pass_context
def scaffold(ctx):
    """Emit CI files and manage artifact ownership.

    The ci verb regenerates the repo's pipeline and health check from the
    profile's deploy: block. The other verbs cover build artifacts, which can
    be claimed per-facility: claiming moves the artifact out of the build zone
    and into the profile beside it, which is where you then edit it. Every
    build copies it back and marks it yours.

    Examples:

    \b
      osprey scaffold ci                          # Re-emit the CI files
      osprey scaffold list                        # Show all artifacts
      osprey scaffold claim agents/channel-finder # Move it into the profile
      osprey scaffold diff agents/channel-finder  # Compare yours vs framework
      osprey scaffold unclaim agents/channel-finder # Restore framework management
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@scaffold.command(name="ci")
@repo_option
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing file that the scaffolder did not write.",
)
def ci(repo: Path | None, force: bool) -> None:
    """Emit this repo's CI pipeline and health check.

    Renders two files from the profile's 'deploy:' block: the CI pipeline at
    the repo root, and the post-deploy health check at scripts/verify.sh. Run
    it again whenever the block changes. Each emitted file says in its own
    header when it runs and what it needs.

    Re-running is safe. A file whose content already matches is left untouched,
    stamp included, so an OSPREY upgrade alone produces no diff. A file the
    scaffolder did not write is reported and left alone; --force replaces it.
    ci-extra.yml is never touched: it is yours, and the pipeline includes it.

    Examples:

    \b
      $ osprey scaffold ci
      $ osprey scaffold ci --repo ~/deployments/als-exemplar
      $ osprey scaffold ci --force
    """
    # Imported here rather than at module scope: it pulls the whole
    # build-profile chain in behind it, and every other verb in this group
    # runs without any of it.
    from .deploy_scaffold import scaffold_deploy_files

    # Resolved before anything is emitted: standing outside a repo is a mistake
    # about where the command was run, not a scaffolding failure.
    repo_root = find_repo_root(repo)

    try:
        emitted = scaffold_deploy_files(repo_root, force=force)
    except ConfigurationError as exc:
        raise click.ClickException(str(exc)) from None

    for result in emitted:
        shown = result.path.relative_to(repo_root)
        if result.refused:
            console.print(f"  [error]✗[/error] {shown} — NOT written: {result.reason}")
        elif result.action == "unchanged":
            console.print(f"  [dim]= {shown} (unchanged)[/dim]")
        else:
            console.print(f"  [success]✓[/success] {shown} ({result.action})")

    if any(result.refused for result in emitted):
        # Non-zero, because an operator who asked for a re-emission did not get
        # one. The reason above already names the flag that overrides it.
        raise SystemExit(1)


def _owned_row(profile_root: Path | None, name: str) -> tuple[str, str, str]:
    """Render one ``list`` user-owned row: ``(name, project path, origin)``.

    Ownership is derived, not declared: most entries are there because a build
    copied the artifact out of the profile's convention tree, and the honest
    thing to show is *which* \u2014 a name whose profile slot is filled says
    "profile", one without says "in project" (a framework artifact claimed in
    place by an older OSPREY, or one the framework auto-registers).

    The path shown is the convention destination whenever the name has one,
    and only otherwise the catalog's output path. The two disagree exactly
    where the ownership does: the catalog lists a skill by its ``SKILL.md``,
    but ``skills/<name>`` in ``user_owned`` owns the whole directory, and that
    is what the row has to say.
    """
    try:
        target: ClaimTarget | None = resolve_claim_target(name)
    except ScaffoldClaimError:
        target = None

    artifact = BuildArtifactCatalog.default().get(name)
    if target is not None:
        project_rel = target.project_rel
    elif artifact is not None:
        project_rel = artifact.output_path
    else:
        project_rel = "?"

    profile_rel = target.profile_rel if target is not None else None
    if profile_root is not None and profile_rel and (profile_root / profile_rel).exists():
        # The slot is not spelled out: it is the name, and the profile root is
        # printed once at the foot of the listing.
        origin = "from profile"
    else:
        origin = "in project"
    return name, project_rel, origin


@scaffold.command(name="list")
@repo_option
def list_artifacts(repo):
    """List all build artifacts and their ownership status."""
    project_dir = _build_zone(repo)

    try:
        config = _load_config(project_dir)
    except click.ClickException:
        config = {}

    user_owned = get_user_owned(config)
    registry = BuildArtifactCatalog.default()
    profile_root = profile_root_or_none(project_dir)

    framework_managed = [
        art for art in registry.all_artifacts() if art.canonical_name not in user_owned
    ]
    # Walk `user_owned` itself, not the catalog: convention-derived ownership
    # covers artifacts the framework never renders (a profile's own skill), and
    # a catalog-only walk would show them nowhere at all.
    owned = [_owned_row(profile_root, str(name)) for name in user_owned]

    console.print("\n[bold]Build Artifacts[/bold]\n")

    if framework_managed:
        console.print("  [dim]Framework-managed:[/dim]")
        for art in framework_managed:
            console.print(
                f"    [success]\u2713[/success] {art.canonical_name:<35s} {art.description}"
            )

    if owned:
        console.print("\n  [dim]User-owned:[/dim]")
        for name, project_rel, origin in owned:
            console.print(
                f"    [bold]\u2605[/bold] {name:<35s} {project_rel:<34s} [dim]{origin}[/dim]"
            )

    if profile_root is not None:
        console.print(f"\n  [dim]Profile:[/dim] [path]{profile_root}[/path]")

    console.print()


@scaffold.command(name="claim")
@click.argument("name")
@repo_option
def claim(name, repo):
    """Move an artifact from the build zone to the profile.

    The profile is the source of truth, so claiming an artifact means putting it
    where the profile keeps that kind of artifact — rules/safety.md,
    skills/orbit-check/, services/postgresql/. A file moves as a file;
    skills and services move as whole directories. The next build copies it back
    into the build zone and registers it as user-owned, so a rebuild never
    overwrites it and prune never unlinks it.

    The built copy is *moved*, not copied: after a claim the artifact lives in
    one place only — the source zone, which is the tracked one — until the next
    build renders it again.

    Examples:

    \b
      osprey scaffold claim agents/channel-finder
      osprey scaffold claim rules/safety
      osprey scaffold claim skills/orbit-check
      osprey scaffold claim services/postgresql
    """
    project_dir = _build_zone(repo)

    try:
        outcome = claim_into_profile(project_dir, name)
    except ScaffoldClaimError as exc:
        raise click.ClickException(str(exc)) from exc

    noun = "directory" if outcome.target.is_directory else "file"
    console.print(
        f"\n  [success]✓[/success] Moved {noun} {outcome.target.project_rel} "
        f"→ [path]{outcome.profile_path}[/path]"
    )
    if outcome.is_persona_delta:
        console.print(
            "  [dim]This project builds a persona delta; convention directories live "
            "at the profile root, so every persona built from it gets this "
            f"{outcome.target.entry_noun}.[/dim]"
        )
    console.print(
        "\n  Edit it in the profile — every build registers "
        f"[bold]{outcome.target.name}[/bold] as user-owned when it copies it in.\n"
        "  Render it into the build zone again with:\n"
        f"    {_REBUILD_HINT}\n"
    )


@scaffold.command(name="diff")
@click.argument("name")
@repo_option
def diff(name, repo):
    """Show diff between a framework template and your file.

    Renders the current framework template and compares it against
    your file at the canonical output path using a unified diff.

    Examples:

    \b
      osprey scaffold diff agents/channel-finder
      osprey scaffold diff rules/facility
    """
    project_dir = _build_zone(repo)
    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name not in user_owned:
        # Deliberately not "run claim first": a claim moves the artifact out of
        # the project, so the file this command diffs would be gone until the
        # next build copies it back from the profile.
        raise click.ClickException(
            f"'{name}' is not user-owned in config.yml, so it is still the framework's "
            "render — there is nothing of yours to compare it against. An artifact "
            "becomes user-owned when a build copies it in from the profile "
            f"(`osprey scaffold claim {name}` puts it there)."
        )

    registry = BuildArtifactCatalog.default()
    artifact = registry.get(name)
    if artifact is None:
        raise click.ClickException(f"Unknown artifact '{name}'.")

    manager = TemplateManager()

    if artifact.is_directory:
        # Directory artifact: per-file unified diff of the packaged template
        # tree against the project copy \u2014 verbatim on both sides (the .j2
        # files inside are deploy-time templates, not build-time renders).
        template_dir = manager.template_root / artifact.template_root / artifact.template_path
        user_dir = project_dir / artifact.output_path
        if not user_dir.is_dir():
            raise click.ClickException(f"Directory not found: {artifact.output_path}")

        framework_files = {
            p.relative_to(template_dir).as_posix() for p in template_dir.rglob("*") if p.is_file()
        }
        user_files = {
            p.relative_to(user_dir).as_posix() for p in user_dir.rglob("*") if p.is_file()
        }

        output_parts = []
        for rel in sorted(framework_files | user_files):
            fw_file = template_dir / rel
            usr_file = user_dir / rel
            fw_lines = (
                fw_file.read_text(encoding="utf-8").splitlines(keepends=True)
                if rel in framework_files
                else []
            )
            usr_lines = (
                usr_file.read_text(encoding="utf-8").splitlines(keepends=True)
                if rel in user_files
                else []
            )
            output_parts.append(
                "".join(
                    difflib.unified_diff(
                        fw_lines,
                        usr_lines,
                        fromfile=f"framework:{artifact.template_path}/{rel}",
                        tofile=f"yours:{artifact.output_path}/{rel}",
                    )
                )
            )

        output = "".join(output_parts)
        if output:
            click.echo(output)
        else:
            console.print(
                "[success]\u2713[/success] Your directory matches the current framework "
                "template \u2014 no differences."
            )
        return

    # Read user's file from canonical output path
    user_file = project_dir / artifact.output_path
    if not user_file.exists():
        raise click.ClickException(f"File not found: {artifact.output_path}")
    user_lines = user_file.read_text(encoding="utf-8").splitlines(keepends=True)

    # Render framework template
    from osprey.cli.templates.claude_code import build_claude_code_context

    ctx = build_claude_code_context(manager.template_root, manager.jinja_env, project_dir, config)
    claude_code_dir = manager.template_root / "claude_code"
    template_file = claude_code_dir / artifact.template_path

    if template_file.suffix == ".j2":
        template_rel = f"claude_code/{artifact.template_path}"
        template = manager.jinja_env.get_template(template_rel)
        framework_content = template.render(**ctx)
    else:
        framework_content = template_file.read_text(encoding="utf-8")

    framework_lines = framework_content.splitlines(keepends=True)

    # Generate unified diff
    diff_lines = difflib.unified_diff(
        framework_lines,
        user_lines,
        fromfile=f"framework:{artifact.template_path}",
        tofile=f"yours:{artifact.output_path}",
    )

    output = "".join(diff_lines)
    if output:
        click.echo(output)
    else:
        console.print(
            "[success]\u2713[/success] Your file matches the current framework template "
            "\u2014 no differences."
        )


@scaffold.command(name="unclaim")
@click.argument("name")
@repo_option
def unclaim(name, repo):
    """Release ownership and restore framework management.

    Removes the artifact from the user_owned list in config.yml and
    .osprey-manifest.json. The next `osprey build` will
    overwrite the file with the framework template.

    Ownership that a build derived from the profile is re-registered by the
    next build, so releasing it here holds only until then — the profile is
    where such an artifact is given up, by removing it from its convention
    directory (or excluding it in a persona delta).

    Examples:

    \b
      osprey scaffold unclaim agents/channel-finder
      osprey scaffold unclaim rules/safety
    """
    project_dir = _build_zone(repo)
    config = _load_config(project_dir)
    user_owned = get_user_owned(config)

    if name not in user_owned:
        raise click.ClickException(f"'{name}' is not user-owned in config.yml.")

    # Remove from config.yml
    update_config_remove_user_owned(project_dir, name)

    # Remove from manifest
    update_manifest_remove_user_owned(project_dir, name)

    console.print(f"  [success]\u2713[/success] Released ownership of {name}")

    # Asked through the published resolver, not re-derived: where an artifact
    # lives in the profile is one mapping, and a second copy of it here could
    # stop naming the file a claim actually moved. An unclaimable name is not an
    # error on this path — ownership has already been released, and the only
    # question left is whether the profile still supplies the file.
    try:
        candidate = profile_slot_for(project_dir, name)
    except ScaffoldClaimError:
        candidate = None
    profile_slot = candidate if candidate is not None and candidate.exists() else None

    if profile_slot is not None:
        styled = f"[path]{profile_slot}[/path]"
        console.print(
            f"\n  [warning]\u26a0[/warning] {still_supplied_by_profile_message(styled)}\n"
        )
        return

    registry = BuildArtifactCatalog.default()
    artifact = registry.get(name)
    if artifact is not None and artifact.is_directory:
        console.print("\n  Next `osprey build` will refresh it from the framework template.\n")
    else:
        console.print("\n  Next `osprey build` will overwrite it with the framework template.\n")


@scaffold.group(name="web-terminals", invoke_without_command=True)
@click.pass_context
def web_terminals(ctx):
    """Validate and render the web_terminals stanza.

    Reads the stanza from the repo's built config.yml, which the profile's
    config: block produces. Point at a repo with --repo, or run from inside
    one.

    Examples:

    \b
      osprey scaffold web-terminals lint
      osprey scaffold web-terminals render --repo ~/deployments/demo -o deploy/
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


#: The retired ``--config PATH`` surface. Deliberately untyped and without
#: ``exists=True``: the option no longer names anything the CLI can read, and
#: the refusal below has to fire for a path that does not exist just as much as
#: for one that does — a Click existence check would pre-empt it with a message
#: that says nothing about where the stanza moved to.
_web_terminals_retired_config_option = click.option(
    "--config",
    "config_path",
    default=None,
    help="Removed. The stanza comes from the repo's built config.yml; see --repo.",
)


def _reject_retired_config_option(config_path: str | None) -> None:
    """Refuse ``--config``, naming what replaced it.

    ``--config`` pointed at a standalone facility-config.yml, a second source of
    truth for facts the profile already owns. That file is gone: the
    ``modules.web_terminals`` stanza now reaches the build zone through the
    profile's ``config:`` block, and the deployment artifacts around it through
    ``osprey scaffold ci`` and the profile's ``deploy:`` block. The option is
    kept for one release so an operator running the old command line is told
    where each half went instead of meeting an unknown-option error.

    Args:
        config_path: Whatever ``--config`` received, or ``None`` if unused.

    Raises:
        SystemExit: Whenever ``--config`` was supplied at all — valid path,
            missing path, or empty string. A bare exit rather than an exception
            carrying the text, because the message has already been reported on
            the operator's error channel by the line above; raising a
            :class:`~osprey.errors.ConfigurationError` here printed the refusal
            and then a traceback on top of it, which reads as a crash rather
            than as the deliberate tombstone it is.
    """
    if config_path is None:
        return
    message = (
        "--config is no longer supported: there is no facility-config.yml. "
        "The modules.web_terminals stanza lives in the repo's built "
        "config.yml, emitted from the profile's `config:` block — run this verb "
        "with --repo DIR, or from inside the repo. Its deployment artifacts "
        "come from `osprey scaffold ci`, driven by the profile's `deploy:` "
        "block. Run /osprey-build-interview to author both blocks."
    )
    logger.error("%s", message)
    raise SystemExit(1)


def _print_web_terminals_lint_errors(errors: list[Any]) -> None:
    """Print the ``modules.web_terminals lint`` "Errors:" block.

    Shared by ``lint`` (as part of its full errors+warnings report) and
    ``render``'s pre-render lint gate (errors only, no warnings section). Takes
    a list of ``osprey.deployment.web_terminals.lint.Finding`` (not type-hinted
    as such here to keep this module's lazy-import convention for that package).
    """
    console.print("  [dim]Errors:[/dim]")
    for finding in errors:
        console.print(f"    [error]\u2717[/error] [{finding.code}] {finding.message}")


@web_terminals.command(name="lint")
@repo_option
@_web_terminals_retired_config_option
def web_terminals_lint(repo, config_path):
    """Validate this repo's modules.web_terminals stanza.

    Checks port-family allocation, reserved service names, and duplicate
    users. Exits non-zero if any error-severity finding is reported;
    warnings (e.g. an enabled module with zero configured users) do not
    fail the check, so this is safe to wire into a CI gate.

    Examples:

    \b
      osprey scaffold web-terminals lint
      osprey scaffold web-terminals lint --repo ~/deployments/demo
    """
    _reject_retired_config_option(config_path)

    from osprey.deployment.web_terminals.lint import lint_web_terminals

    config = _load_config(_build_zone(repo))
    findings = lint_web_terminals(config)

    if not findings:
        console.print("[success]\u2713[/success] modules.web_terminals: no issues found\n")
        return

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]

    console.print("\n[bold]modules.web_terminals lint[/bold]\n")

    if errors:
        _print_web_terminals_lint_errors(errors)

    if warnings:
        console.print("\n  [dim]Warnings:[/dim]")
        for finding in warnings:
            console.print(f"    [warning]\u26a0[/warning] [{finding.code}] {finding.message}")

    console.print()

    if errors:
        raise SystemExit(1)


@web_terminals.command(name="render")
@repo_option
@_web_terminals_retired_config_option
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, resolve_path=True),
    required=True,
    help="Directory to write the rendered deployment artifacts into.",
)
@click.option(
    "--no-lint",
    is_flag=True,
    default=False,
    help="Skip the lint pre-check (by default, lint errors abort the render).",
)
def web_terminals_render(repo, config_path, output_dir, no_lint):
    """Render this repo's modules.web_terminals deployment artifacts.

    Writes the docker-compose overlay, nginx routing fragment, and static landing
    page into --output, creating the directory (and its nginx/ subdirectory) as
    needed. By default the stanza is linted first and error-severity findings
    abort the render; pass --no-lint to render anyway.

    Examples:

    \b
      osprey scaffold web-terminals render -o deploy/
      osprey scaffold web-terminals render --repo ~/deployments/demo -o deploy/
    """
    _reject_retired_config_option(config_path)

    from osprey.deployment.web_terminals.render import render_web_terminals

    config = _load_config(_build_zone(repo))

    if not no_lint:
        from osprey.deployment.web_terminals.lint import lint_web_terminals

        errors = [f for f in lint_web_terminals(config) if f.severity == "error"]
        if errors:
            console.print("\n[bold]modules.web_terminals lint[/bold]\n")
            _print_web_terminals_lint_errors(errors)
            console.print()
            raise click.ClickException(
                "modules.web_terminals has lint errors; fix them or pass --no-lint to render anyway."
            )

    try:
        artifacts = render_web_terminals(config)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = Path(output_dir)
    written = []
    for relative_path, content in artifacts.items():
        target = output_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)

    console.print("\n[bold]modules.web_terminals render[/bold]\n")
    for target in written:
        console.print(f"  [success]\u2713[/success] {target}")
    console.print()
