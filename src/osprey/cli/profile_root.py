"""Profile-root discovery — the anchor every profile-relative path resolves against.

A positional profile file is one of three kinds, and its location alone says
which: a deployment's own **root** profile, a **tier** in ``tiers/`` beside that
profile's ``profile.yml``, or an **app** in ``apps/`` beside it. Tiers and apps
are layers: their ``data:``, convention dirs, ``.env`` and hash material all
belong to the profile root one level up, never to the layer's own directory, so
the distinction is made once, here, rather than re-derived by each consumer.
The package ships tier and app presets the same way, in ``presets/tiers/`` and
``presets/apps/`` beside the deployment presets; those anchor at the presets
directory, which holds no ``profile.yml``.

The trigger is deliberately narrow: the file's parent directory is named
``tiers`` or ``apps`` **and** a ``profile.yml`` sits beside that directory. One
level, no upward walk. A walk would quietly capture unrelated files (a profile
passed from a scratch directory that happens to sit under a ``tiers`` ancestor)
and anchor them somewhere the caller never named; the narrow predicate leaves
every other path — including the bare temp-file profiles tests build —
standalone, with its own parent as root and no further requirements.

A file inside a layer directory whose root is missing is an error rather than a
standalone build: it is a layer, so building it alone would silently produce a
profile missing everything the root was meant to supply. A stem present in both
``tiers/`` and ``apps/`` is refused: the name is the render's name, and one name
cannot be both an access level of the deployment's agent and a separate app.

The retired layer directory is read as ``tiers/`` for one release, with a single
warning naming the rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path

from osprey.utils.workspace import (
    APPS_DIRNAME,
    LAYER_DIR_KINDS,
    LAYER_DIRNAMES,
    PROFILE_FILENAME,
    RETIRED_LAYER_DIRS,
    TIERS_DIRNAME,
)
from osprey_connectors.errors import BuildProfileError

_LOGGER = logging.getLogger(__name__)


class ProfileKind(StrEnum):
    """What a profile file is, decided by where it sits.

    Truthy exactly when the kind is a layer that still has to be merged over a
    root profile, so ``if kind:`` reads as "is this a layer?".
    """

    ROOT = "root"
    TIER = "tier"
    APP = "app"

    @property
    def is_layer(self) -> bool:
        """Whether a file of this kind is merged over the root profile."""
        return self is not ProfileKind.ROOT

    def __bool__(self) -> bool:
        return self.is_layer


#: Layer directory name -> the kind its files resolve as. Built from the
#: layout names in ``osprey_connectors.workspace``; a kind word there that
#: :class:`ProfileKind` does not know fails here, at import.
_LAYER_KINDS: dict[str, ProfileKind] = {
    **{d: ProfileKind(word) for d, word in LAYER_DIR_KINDS.items()},
    **{d: ProfileKind(LAYER_DIR_KINDS[live]) for d, live in RETIRED_LAYER_DIRS.items()},
}


def symlink_escapes(path: Path, root: Path) -> bool:
    """True when ``path`` is a symlink resolving outside ``root``."""
    if not path.is_symlink():
        return False
    return not path.resolve().is_relative_to(root.resolve())


def layer_files(root_dir: Path) -> list[tuple[ProfileKind, Path]]:
    """Every layer file a profile root carries, as ``(kind, path)``.

    The direct children of ``tiers/``, ``apps/`` and the retired layer directory (in that
    order) that are files and are not dot-prefixed, sorted by name within each
    directory. Exactly the files :func:`resolve_profile_root` reads as layers —
    a file is a layer only when its parent directory IS a layer directory — so
    a tree nested below one is not build input. No suffix filter: a layer
    directory holds layers and nothing else, so a file in one that will not
    parse is a mistake the build names rather than walks past.

    Every listing that has to agree about the layer set uses this: the build
    loop that renders one project per layer, the content-hash fold, and the
    staleness fingerprint's per-file digests. A file one of them saw and
    another missed would either ship an edit silently or refuse a start with
    nothing to re-render.

    Purely lexical: it neither resolves symlinks nor checks that a file anchors
    back at *root_dir* — the build loop does that before rendering.

    Returns:
        An empty list when *root_dir* has no layer directory.
    """
    found: list[tuple[ProfileKind, Path]] = []
    for dirname in LAYER_DIRNAMES:
        layer_dir = root_dir / dirname
        if not layer_dir.is_dir():
            continue
        kind = _LAYER_KINDS[dirname]
        found.extend(
            (kind, entry)
            for entry in sorted(layer_dir.iterdir(), key=lambda entry: entry.name)
            if entry.is_file() and not entry.name.startswith(".")
        )
    return found


#: ``tiers/`` files already warned about in this process, so a build that
#: resolves the same file several times prints the deprecation line once.
_WARNED_LAYER_FILES: set[Path] = set()


def _is_bundled_preset_dir(path: Path) -> bool:
    # Imported here: the presets module sits above this one in the import graph.
    from osprey.cli.build_profile_presets import is_bundled_preset_dir

    return is_bundled_preset_dir(path)


def _refuse_stem_in_two_kinds(path: Path, root_dir: Path, kind: ProfileKind) -> None:
    """Raise when *path*'s stem also names a layer file of the other kind."""
    other_dirs = [d for d in LAYER_DIRNAMES if _LAYER_KINDS[d] is not kind]
    for other in other_dirs:
        twin = root_dir / other / path.name
        if twin.is_file():
            raise BuildProfileError(
                f"{path} and {twin} share the name {path.stem!r}. That name is both a "
                f"tier and an app, and a render is named by its file, so neither can be "
                f"built. Rename one of them, or delete the one this deployment does not "
                f"use: a tier in '{TIERS_DIRNAME}/' is an access level of the "
                f"deployment's own agent, an app in '{APPS_DIRNAME}/' is a separate agent "
                "the deployment offers."
            )


def _warn_layer_file(path: Path, root_dir: Path) -> None:
    if path in _WARNED_LAYER_FILES:
        return
    _WARNED_LAYER_FILES.add(path)
    successor = RETIRED_LAYER_DIRS[path.parent.name]
    _LOGGER.warning(
        "%s is read as the tier %s; the '%s/' directory is deprecated and will be "
        "refused in the next release. To move every file into '%s/', run "
        "`osprey init --revalidate`",
        path,
        root_dir / successor / path.name,
        path.parent.name,
        successor,
    )


def resolve_profile_root(profile_path: Path) -> tuple[Path, ProfileKind]:
    """Resolve the profile root a profile file's relative paths anchor at.

    Args:
        profile_path: Path to the profile file being built. Need not exist —
            callers report a missing file themselves, with their own wording.

    Returns:
        ``(root_dir, kind)``. ``root_dir`` is the absolute directory that
        ``data:``, convention dirs, ``.env`` and hash material resolve against:
        the profile root for a tier or app, the bundled presets directory for a
        tier or app preset, the file's own parent otherwise. ``kind`` is
        :attr:`ProfileKind.TIER` or :attr:`ProfileKind.APP` for a layer that
        still has to be merged over a root profile, :attr:`ProfileKind.ROOT`
        otherwise.

    Raises:
        BuildProfileError: If the file sits in ``tiers/``, ``apps/`` or a
            retired layer directory with no ``profile.yml`` beside it, or its
            stem names both a tier and an app.
    """
    path = Path(profile_path).resolve()
    parent = path.parent
    kind = _LAYER_KINDS.get(parent.name)

    if kind is None:
        return parent, ProfileKind.ROOT

    root_dir = parent.parent
    if parent.name not in RETIRED_LAYER_DIRS and _is_bundled_preset_dir(root_dir):
        return root_dir, kind

    root_profile = root_dir / PROFILE_FILENAME
    if not root_profile.is_file():
        raise BuildProfileError(
            f"{kind.value.capitalize()} profile {path} sits in a '{parent.name}/' "
            f"directory but its profile root is missing: expected {root_profile}. A "
            f"{kind.value} file holds only what it adds to the profile it belongs to, so "
            f"it cannot be built on its own — move it beside its {PROFILE_FILENAME}, "
            f"or move it out of a '{parent.name}/' directory to build it as a standalone "
            "profile."
        )

    _refuse_stem_in_two_kinds(path, root_dir, kind)
    if parent.name in RETIRED_LAYER_DIRS:
        _warn_layer_file(path, root_dir)
    return root_dir, kind


class ProjectProfileProblem(Enum):
    """Why a built project's profile root could not be resolved.

    Named rather than left implicit because the three are different situations
    with different remedies, and a caller that reports them has to tell them
    apart: nothing was ever recorded, what was recorded is no longer there, or
    it is there but does not anchor.
    """

    #: The manifest names no profile — a manifest without that field, or not a
    #: built project.
    NO_PROFILE_RECORDED = "no_profile_recorded"

    #: The manifest names a profile file that is not there any more.
    PROFILE_FILE_MISSING = "profile_file_missing"

    #: A tier or app whose root profile is gone; see :func:`resolve_profile_root`.
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

    #: What ``profile_path`` is; a tier or app anchors one level up.
    kind: ProfileKind = ProfileKind.ROOT

    #: Set when no root resolved; ``None`` on success.
    problem: ProjectProfileProblem | None = None

    #: The root-discovery error behind ``ROOT_UNRESOLVABLE``, kept whole so a
    #: caller can re-raise with its own type and preserve the chain — its
    #: message already names the file that has to exist.
    root_error: BuildProfileError | None = None

    @property
    def is_layer(self) -> bool:
        """Whether ``profile_path`` is a layer (tier or app) over the root."""
        return self.kind.is_layer

    @property
    def env_path(self) -> Path | None:
        """The ``.env`` that owns this project's secrets, or ``None``.

        The ``.env`` sits at the profile *root*: a tier or app shares the root
        profile's secrets rather than keeping its own beside itself.
        """
        return None if self.root is None else self.root / ".env"


def resolve_project_profile(project_dir: Path, *, require_profile_file: bool) -> ProjectProfile:
    """Resolve the profile root a built project is anchored to.

    The manifest-anchored counterpart of :func:`resolve_profile_root`: it starts
    from a *project* rather than a profile file, following
    ``build_args.profile_path_abs`` — the path ``osprey build`` actually
    resolved — and then anchoring it the way the build does, so a project built
    from a tier or app reports the root one level above ``tiers/`` or ``apps/``
    rather than the layer's own directory.

    Callers ask the same question for different reasons, and the one thing they
    genuinely disagree on is whether the recorded profile file still has to be
    *there*, which ``require_profile_file`` makes explicit:

    * ``True`` for a caller acting on the profile — moving an artifact into it,
      reading a tier or app out of it. A directory that no longer holds a
      profile would otherwise be reported as the root and every file under it
      as individually missing: one problem told as several, none of them the
      real one.
    * ``False`` for a caller answering "where *would* this project's secrets
      live", which an operator being offered a place to put a secret needs
      answered even when the profile file is momentarily absent.

    ``require_profile_file=False`` deliberately does not make this a check a
    caller about to *write* unattended can lean on; that caller owes the
    stronger question separately. This resolver stays a pure path resolution and
    answers no question about whether writing there is a good idea.

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
        root_dir, kind = resolve_profile_root(profile_path)
    except BuildProfileError as exc:
        return ProjectProfile(
            profile_path=profile_path,
            problem=ProjectProfileProblem.ROOT_UNRESOLVABLE,
            root_error=exc,
        )

    return ProjectProfile(profile_path=profile_path, root=root_dir, kind=kind)
