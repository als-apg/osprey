"""Profile-root discovery — the anchor every profile-relative path resolves against.

A positional profile file is one of two things: a facility's own profile, or a
persona *delta* living in that profile's ``personas/`` directory. The two
anchor differently — a delta's ``data:``, convention dirs, ``.env`` and hash
material all belong to the profile root one level up, never to ``personas/``
itself — so the distinction has to be made once, in one place, rather than
re-derived by each consumer.

The trigger is deliberately narrow: the file's parent directory is named
``personas`` **and** a ``profile.yml`` sits beside that directory. One level,
no upward walk. A walk would quietly capture unrelated files (a profile passed
from a scratch directory that happens to sit under a ``personas`` ancestor) and
anchor them somewhere the caller never named; the narrow predicate leaves every
other path — including the bare temp-file profiles tests build — standalone,
with its own parent as root and no further requirements.

A file inside ``personas/`` whose root is missing is an error rather than a
standalone build: it is a delta, so building it alone would silently produce a
profile missing everything the root was meant to supply.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from osprey.errors import BuildProfileError

#: Directory name that marks a profile file as a persona delta.
PERSONA_DIRNAME = "personas"

#: Filename of the root profile a persona delta merges over.
ROOT_PROFILE_FILENAME = "profile.yml"


def resolve_profile_root(profile_path: Path) -> tuple[Path, bool]:
    """Resolve the profile root a profile file's relative paths anchor at.

    Args:
        profile_path: Path to the profile file being built. Need not exist —
            callers report a missing file themselves, with their own wording.

    Returns:
        ``(root_dir, is_persona_delta)``. ``root_dir`` is the absolute directory
        that ``data:``, convention dirs, ``.env`` and hash material resolve
        against: the profile root for a persona delta, the file's own parent
        otherwise. ``is_persona_delta`` tells callers whether the file's content
        is a delta that still has to be merged over the root profile.

    Raises:
        BuildProfileError: If the file sits in a ``personas/`` directory with no
            ``profile.yml`` beside it.
    """
    path = Path(profile_path).resolve()
    parent = path.parent

    if parent.name != PERSONA_DIRNAME:
        return parent, False

    root_dir = parent.parent
    root_profile = root_dir / ROOT_PROFILE_FILENAME
    if not root_profile.is_file():
        raise BuildProfileError(
            f"Persona profile {path} sits in a '{PERSONA_DIRNAME}/' directory but its "
            f"profile root is missing: expected {root_profile}. A persona file holds "
            "only a delta over the profile it belongs to, so it cannot be built on its "
            f"own — move it beside its {ROOT_PROFILE_FILENAME}, or move it out of a "
            f"'{PERSONA_DIRNAME}/' directory to build it as a standalone profile."
        )

    return root_dir, True


class ProjectProfileProblem(Enum):
    """Why a built project's profile root could not be resolved.

    Named rather than left implicit because the three are different situations
    with different remedies, and a caller that reports them has to tell them
    apart: nothing was ever recorded, what was recorded is no longer there, or
    it is there but does not anchor.
    """

    #: The manifest names no profile — a legacy manifest, or not a built project.
    NO_PROFILE_RECORDED = "no_profile_recorded"

    #: The manifest names a profile file that is not there any more.
    PROFILE_FILE_MISSING = "profile_file_missing"

    #: A persona delta whose root profile is gone; see :func:`resolve_profile_root`.
    ROOT_UNRESOLVABLE = "root_unresolvable"


@dataclass(frozen=True)
class ProjectProfile:
    """Where a built project's profile lives, per its own manifest.

    Always returned — a failure is reported in ``problem`` rather than as a
    bare ``None``, so a caller that only wants "is there a profile?" can read
    ``root is None`` while a caller that has to explain itself to an operator
    still has the reason and the paths to name.
    """

    #: The profile file the manifest records, or ``None`` when it records none.
    profile_path: Path | None = None

    #: The resolved profile root; ``None`` whenever ``problem`` is set.
    root: Path | None = None

    #: Whether ``profile_path`` is a persona delta anchored one level up.
    is_persona_delta: bool = False

    #: Set when no root resolved; ``None`` on success.
    problem: ProjectProfileProblem | None = None

    #: The root-discovery error behind ``ROOT_UNRESOLVABLE``, kept whole so a
    #: caller can re-raise with its own type and preserve the chain — its
    #: message already names the file that has to exist.
    root_error: BuildProfileError | None = None

    @property
    def env_path(self) -> Path | None:
        """The ``.env`` that owns this project's secrets, or ``None``.

        The ``.env`` sits at the profile *root*: a persona delta shares the root
        profile's secrets rather than keeping its own beside itself.
        """
        return None if self.root is None else self.root / ".env"


def resolve_project_profile(project_dir: Path, *, require_profile_file: bool) -> ProjectProfile:
    """Resolve the profile root a built project is anchored to.

    The manifest-anchored counterpart of :func:`resolve_profile_root`: it starts
    from a *project* rather than a profile file, following
    ``build_args.profile_path_abs`` — the path ``osprey build`` actually
    resolved — and then anchoring it the way the build does, so a project built
    from a persona delta reports the root one level above ``personas/`` rather
    than the delta's own directory.

    Callers ask the same question for different reasons, and the one thing they
    genuinely disagree on is whether the recorded profile file still has to be
    *there*, which ``require_profile_file`` makes explicit:

    * ``True`` for a caller acting on the profile — moving an artifact into it,
      reading a persona delta out of it. A directory that no longer holds a
      profile would otherwise be reported as the root and every file under it
      as individually missing: one problem told as several, none of them the
      real one.
    * ``False`` for a caller answering "where *would* this project's secrets
      live", which an operator being offered a place to put a secret needs
      answered even when the profile file is momentarily absent.

    ``require_profile_file=False`` deliberately does not make this a check a
    caller about to *write* unattended can lean on; that caller owes the
    stronger question separately (see
    :func:`osprey.deployment.container_lifecycle._stale_manifest_profile`),
    which is why this resolver stays a pure path resolution and answers no
    question about whether writing there is a good idea.

    :param project_dir: Project root (the directory holding the manifest).
    :param require_profile_file: Whether the recorded profile file must exist.
    :returns: The resolution, with ``root`` set on success and ``problem`` set
        on every way the answer is no. Never raises.
    """
    from osprey.cli.templates.manifest import manifest_profile_path

    profile_path = manifest_profile_path(project_dir)
    if profile_path is None:
        return ProjectProfile(problem=ProjectProfileProblem.NO_PROFILE_RECORDED)

    if require_profile_file and not profile_path.is_file():
        return ProjectProfile(
            profile_path=profile_path,
            problem=ProjectProfileProblem.PROFILE_FILE_MISSING,
        )

    try:
        root_dir, is_delta = resolve_profile_root(profile_path)
    except BuildProfileError as exc:
        return ProjectProfile(
            profile_path=profile_path,
            problem=ProjectProfileProblem.ROOT_UNRESOLVABLE,
            root_error=exc,
        )

    return ProjectProfile(profile_path=profile_path, root=root_dir, is_persona_delta=is_delta)
