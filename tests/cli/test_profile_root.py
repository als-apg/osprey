"""Tests for :func:`resolve_profile_root` — the profile-relative path anchor.

Pins the exact trigger: a profile file anchors one directory up only when its
parent is named ``personas`` *and* a ``profile.yml`` sits beside that directory.
The predicate looks exactly one level and never walks upward, so a profile that
merely happens to live under some ``personas`` ancestor stays standalone. A file
inside ``personas/`` with no root beside it is a delta that cannot stand alone,
so it raises rather than building something silently incomplete.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.profile_root import resolve_profile_root
from osprey.errors import BuildProfileError


def test_persona_file_anchors_at_the_profile_root(tmp_path: Path) -> None:
    """A personas/ file beside a profile.yml root anchors one level up."""
    root = tmp_path / "my-profile"
    (root / "personas").mkdir(parents=True)
    (root / "profile.yml").write_text("name: my-profile\n")
    persona = root / "personas" / "reader.yml"
    persona.write_text("name: reader\n")

    root_dir, is_persona_delta = resolve_profile_root(persona)

    assert root_dir == root.resolve()
    assert is_persona_delta is True


def test_persona_file_without_a_root_raises(tmp_path: Path) -> None:
    """A personas/ file with no profile.yml beside it is a hard error."""
    personas = tmp_path / "stray" / "personas"
    personas.mkdir(parents=True)
    orphan = personas / "reader.yml"
    orphan.write_text("name: reader\n")

    with pytest.raises(BuildProfileError, match="profile root is missing"):
        resolve_profile_root(orphan)


def test_persona_error_names_the_expected_root_profile(tmp_path: Path) -> None:
    """The error points at the exact profile.yml path that would fix it."""
    personas = tmp_path / "stray" / "personas"
    personas.mkdir(parents=True)
    orphan = personas / "reader.yml"
    orphan.write_text("name: reader\n")

    with pytest.raises(BuildProfileError) as excinfo:
        resolve_profile_root(orphan)

    assert str((tmp_path / "stray" / "profile.yml").resolve()) in str(excinfo.value)


def test_bare_file_is_standalone(tmp_path: Path) -> None:
    """A profile written straight into a temp dir is standalone, root = its parent."""
    profile = tmp_path / "profile.yml"
    profile.write_text("name: bare\n")

    root_dir, is_persona_delta = resolve_profile_root(profile)

    assert root_dir == tmp_path.resolve()
    assert is_persona_delta is False


def test_standalone_file_needs_no_sibling_profile_yml(tmp_path: Path) -> None:
    """An arbitrarily named file with no profile.yml beside it still resolves."""
    profile = tmp_path / "scratch-profile.yaml"
    profile.write_text("name: scratch\n")

    root_dir, is_persona_delta = resolve_profile_root(profile)

    assert root_dir == tmp_path.resolve()
    assert is_persona_delta is False


def test_nested_directory_that_is_not_personas_is_standalone(tmp_path: Path) -> None:
    """A nested dir with another name is standalone even beside a profile.yml."""
    root = tmp_path / "my-profile"
    (root / "variants").mkdir(parents=True)
    (root / "profile.yml").write_text("name: my-profile\n")
    nested = root / "variants" / "reader.yml"
    nested.write_text("name: reader\n")

    root_dir, is_persona_delta = resolve_profile_root(nested)

    assert root_dir == (root / "variants").resolve()
    assert is_persona_delta is False


def test_predicate_does_not_walk_upward(tmp_path: Path) -> None:
    """A file two levels below personas/ does not inherit the persona anchor."""
    root = tmp_path / "my-profile"
    deep = root / "personas" / "drafts"
    deep.mkdir(parents=True)
    (root / "profile.yml").write_text("name: my-profile\n")
    draft = deep / "reader.yml"
    draft.write_text("name: reader\n")

    root_dir, is_persona_delta = resolve_profile_root(draft)

    assert root_dir == deep.resolve()
    assert is_persona_delta is False


def test_personas_directory_alone_does_not_trigger(tmp_path: Path) -> None:
    """A personas/ dir whose parent holds a differently named profile stays standalone-erroring."""
    personas = tmp_path / "loose" / "personas"
    personas.mkdir(parents=True)
    (tmp_path / "loose" / "site.yml").write_text("name: loose\n")
    persona = personas / "reader.yml"
    persona.write_text("name: reader\n")

    with pytest.raises(BuildProfileError, match="profile root is missing"):
        resolve_profile_root(persona)


def test_relative_path_resolves_to_an_absolute_root(tmp_path: Path, monkeypatch) -> None:
    """A relative profile path yields an absolute root, so callers never join onto cwd."""
    root = tmp_path / "my-profile"
    (root / "personas").mkdir(parents=True)
    (root / "profile.yml").write_text("name: my-profile\n")
    (root / "personas" / "reader.yml").write_text("name: reader\n")
    monkeypatch.chdir(tmp_path)

    root_dir, is_persona_delta = resolve_profile_root(Path("my-profile/personas/reader.yml"))

    assert root_dir.is_absolute()
    assert root_dir == root.resolve()
    assert is_persona_delta is True


def test_missing_file_still_resolves_a_root(tmp_path: Path) -> None:
    """The helper anchors on path shape, not existence — callers report missing files."""
    root_dir, is_persona_delta = resolve_profile_root(tmp_path / "never-written.yml")

    assert root_dir == tmp_path.resolve()
    assert is_persona_delta is False
