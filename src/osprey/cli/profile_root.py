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
