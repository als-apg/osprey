"""Host-local selection of which overlay profile a repo builds.

One deployment repo, many hosts: a test stand and a control-room machine build
the same tracked ``profile.yml`` but want different overlays on top of it. The
overlays themselves are tracked — ``profiles/<name>.yml`` beside ``profile.yml``
— because they are part of what the deployment *is*. What is host-local is only
the choice between them, and that choice is one line in one gitignored file:

    .env.variant:  OSPREY_PROFILE_VARIANT=teststand

The file rides the ``.gitignore``'s anchored ``/.env*`` family, so every repo
already ignores it with no migration. It is deliberately NOT a member of
:data:`~osprey_connectors.dotenv.ENV_CHAIN_FILENAMES`: the env chain is what a
deployment's processes read, and this setting is read once, by the build, to
pick a profile layer. A variant name is not an environment variable the running
agent should ever see.

Everything here reads; nothing writes. :func:`resolve_variant_selection` is the
whole surface — it answers "which overlay, if any, does this host want", warns
about the two states an operator would want to know about (the setting file
committed by accident, a variant named but no variants in the repo), and raises
only for a name the repo cannot honor.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from osprey.errors import BuildProfileError
from osprey.utils.dotenv import parse_dotenv_file
from osprey.utils.logger import get_logger

from .repo_resolver import PROFILE_FILENAME

logger = get_logger("build")

#: The host-local settings file, at the repo root. Named into the ``/.env*``
#: family on purpose — see the module docstring — and covered by the anchored
#: pattern `osprey init` writes, so it is ignored from birth.
VARIANT_SETTING_FILENAME = ".env.variant"

#: The one key the settings file carries. Anything else in the file is ignored:
#: the file's whole job is naming a profile layer, and a second key would be a
#: variable nothing reads.
VARIANT_SETTING_KEY = "OSPREY_PROFILE_VARIANT"

#: Repo-root directory holding the overlay profiles, one ``<name>.yml`` per
#: variant. Tracked, unlike the setting that picks among them.
VARIANT_DIRNAME = "profiles"


@dataclass(frozen=True)
class VariantSelection:
    """What ``.env.variant`` asked for, and what the repo could give it.

    Attributes:
        name: The variant named by the setting, or ``None`` when no setting is
            in force (file absent, key absent, or value empty).
        path: The overlay profile to merge over ``profile.yml``, or ``None``
            when nothing was selected — either because ``name`` is ``None`` or
            because the repo defines no variants at all, which is a notice
            rather than an error.
        notices: Non-fatal things an operator should know, already logged as
            warnings. Returned as well as logged so a caller rendering a
            summary can show them where the operator is looking.
    """

    name: str | None
    path: Path | None
    notices: tuple[str, ...] = ()

    @property
    def selected(self) -> bool:
        """True when an overlay profile was resolved and should be merged."""
        return self.path is not None


def variant_setting_path(repo_root: Path) -> Path:
    """Where the host-local setting file lives for ``repo_root``."""
    return repo_root / VARIANT_SETTING_FILENAME


def known_variants(repo_root: Path) -> list[str]:
    """Variant names the repo defines, sorted.

    The set is exactly the ``.yml`` files directly under ``profiles/`` — a
    variant is a profile layer, so a subdirectory is not one, and neither is a
    file with any other suffix.
    """
    root = repo_root / VARIANT_DIRNAME
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.glob("*.yml") if path.is_file())


def read_variant_setting(repo_root: Path) -> str | None:
    """The variant name ``.env.variant`` sets, or ``None``.

    Parses with the shared ``.env`` reader so the file's syntax is the syntax
    an operator already knows (comments, ``export`` prefix, quoted values). An
    unreadable file is treated as unset rather than fatal: this setting is an
    optional host-local preference, and a build that cannot read it should
    build the tracked default, not refuse.
    """
    path = variant_setting_path(repo_root)
    if not path.is_file():
        return None
    try:
        values = parse_dotenv_file(path)
    except OSError as exc:
        logger.warning("Could not read %s (%s) — building the tracked profile.", path, exc)
        return None
    name = values.get(VARIANT_SETTING_KEY, "").strip()
    return name or None


def variant_profile_path(repo_root: Path, name: str) -> Path:
    """Where the overlay named *name* lives, whether or not it exists.

    One spelling of the layout rule, so the resolver and the build fingerprint
    cannot disagree about which file a variant name means.
    """
    return repo_root / VARIANT_DIRNAME / f"{name}.yml"


def active_variant_overlay(repo_root: Path) -> Path | None:
    """The overlay a build of *repo_root* would merge, or ``None``.

    The *quiet* reading of the same selection :func:`resolve_variant_selection`
    makes: no warnings, no git probe, and no refusal — a name the repo cannot
    honor answers ``None`` here, because a build refuses on it long before
    anything asks what to fingerprint.

    It exists for the drift fingerprint
    (:func:`~osprey.cli.build_profile_merge.compute_profile_hash`), which is
    computed on both the stamp side and the check side and must therefore be
    cheap, silent and total. Warning twice about a tracked settings file — once
    when the build resolves it and once when the build hashes it — would be
    noise, and a hash that raises would turn a variant repo unverifiable.

    Args:
        repo_root: The directory holding ``profile.yml``, ``profiles/`` and the
            settings file. A directory that is not a deployment repo (a
            ``personas/`` directory, a preset package) simply has no setting
            and so selects nothing.
    """
    name = read_variant_setting(repo_root)
    if name is None:
        return None
    path = variant_profile_path(repo_root, name)
    return path if path.is_file() else None


def _is_git_tracked(repo_root: Path, relative: str) -> bool:
    """Whether git tracks ``relative`` inside ``repo_root``.

    Best-effort by construction: no git, no repository, or a git that takes too
    long all answer "not tracked". The probe exists to catch a real mistake, so
    it must never be the reason a build fails.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _unknown_variant_error(name: str, available: list[str]) -> BuildProfileError:
    """The refusal for a name the repo does not define."""
    listed = "\n".join(f"  {variant}" for variant in available)
    return BuildProfileError(
        f"{VARIANT_SETTING_FILENAME} selects the variant {name!r}, but "
        f"{VARIANT_DIRNAME}/{name}.yml does not exist.\n"
        f"Variants this repo defines:\n{listed}\n"
        f"  Set {VARIANT_SETTING_KEY} to one of those, or remove "
        f"{VARIANT_SETTING_FILENAME} to build {PROFILE_FILENAME} as tracked."
    )


def resolve_variant_selection(repo_root: Path) -> VariantSelection:
    """Resolve the host-local variant setting against the repo's variants.

    The four outcomes, in the order they are decided:

    * No setting in force — nothing selected, no notices. The ordinary case.
    * The setting file is tracked by git — the choice is host-local, so a
      committed one would follow the repo onto every other host. Warned about,
      never fatal, and resolution continues.
    * A variant is named but the repo defines none — the setting is ignored
      with a notice. This is what a host looks like after the variants are
      removed from the repo, and refusing there would break a build whose
      tracked profile is perfectly buildable.
    * A variant is named and the repo defines variants — it must be one of
      them, or resolution raises listing the names that would work.

    Args:
        repo_root: The deployment repo root — the directory holding
            ``profile.yml``, ``profiles/``, and the setting file.

    Returns:
        The selection. ``path`` is the overlay to merge over ``profile.yml``,
        or ``None`` when this host builds the tracked profile as-is.

    Raises:
        BuildProfileError: When the setting names a variant the repo defines
            no file for.
    """
    name = read_variant_setting(repo_root)
    if name is None:
        return VariantSelection(name=None, path=None)

    notices: list[str] = []

    if _is_git_tracked(repo_root, VARIANT_SETTING_FILENAME):
        notice = (
            f"{VARIANT_SETTING_FILENAME} is tracked by git — it names which variant "
            f"THIS host builds, so committing it hands the same choice to every other "
            f"host that clones this repo. Untrack it with "
            f"`git rm --cached {VARIANT_SETTING_FILENAME}`; the repo's `.gitignore` "
            f"already covers it."
        )
        logger.warning(notice)
        notices.append(notice)

    available = known_variants(repo_root)
    if not available:
        notice = (
            f"{VARIANT_SETTING_FILENAME} selects the variant {name!r}, but this repo "
            f"defines no variants ({VARIANT_DIRNAME}/ holds no .yml files) — building "
            f"{PROFILE_FILENAME} as tracked and ignoring the setting."
        )
        logger.warning(notice)
        notices.append(notice)
        return VariantSelection(name=name, path=None, notices=tuple(notices))

    if name not in available:
        raise _unknown_variant_error(name, available)

    return VariantSelection(
        name=name,
        path=variant_profile_path(repo_root, name),
        notices=tuple(notices),
    )
