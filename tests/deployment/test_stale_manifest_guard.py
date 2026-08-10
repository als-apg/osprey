"""Deploy write-back against a manifest whose profile has moved away.

``build_args.profile_path_abs`` is a build-machine absolute path, and nothing
keeps it true: an operator reorganizes a profile repo, moves a facility profile
one directory up, or renames the tree. What survives is the *old parent
directory* — the profile repo's other files are still there, only ``profile.yml``
has gone.

That is the case the write-back must not walk into. The profile root is derived
from the recorded profile file's parent, so a directory that still exists
resolves cleanly and ``append_profile_env`` happily *creates* a fresh ``.env``
there: a second, orphaned copy of the stack's secrets in a directory no profile
lives in any more, while the manifest is stamped ``secrets_synced_to_profile:
true`` — a claim about a profile that is not there.

So the recorded profile file's existence is checked before anything is written.
When it is gone the deploy degrades exactly as it does for a profile whose whole
directory vanished: secrets stay in the project ``.env``, a warning names both
the profile that moved and the ``.env`` that was not written, and the manifest
flag records that the project holds the only copy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from osprey.cli.templates.manifest import (
    SECRETS_SYNCED_KEY,
    read_secrets_synced_to_profile,
)
from osprey.deployment import container_lifecycle
from osprey.utils.dotenv import parse_dotenv_text

_DISPATCH_VARS = ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN")

#: A deploy whose services pull in ``_SERVICE_TOKEN_VARS`` entries — without one
#: the write-back never runs and the guard would never be reached.
_CONFIG = {"deployed_services": ["event_dispatcher"]}


@pytest.fixture(autouse=True)
def _clean_token_env(monkeypatch):
    """No exported token may win over the ``.env`` values under test."""
    for var in (*_DISPATCH_VARS, "ARIEL_DSN"):
        monkeypatch.delenv(var, raising=False)


def _profile(tmp_path: Path) -> Path:
    """A profile tree (``profile.yml`` plus a sibling file); returns its root."""
    root = tmp_path / "facility-profile"
    root.mkdir()
    (root / "profile.yml").write_text("name: facility\n", encoding="utf-8")
    # A profile repo holds more than the profile file, which is why the old
    # directory survives the move that strands the manifest.
    (root / "README.md").write_text("facility profile\n", encoding="utf-8")
    return root


def _project(tmp_path: Path, profile_path: Path) -> Path:
    """A built project dir whose manifest records ``profile_path``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("", encoding="utf-8")
    (project / ".osprey-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.2.0",
                "build_args": {
                    "project_name": "project",
                    "source": "profile",
                    "profile_path": "facility-profile/profile.yml",
                    "profile_path_abs": str(profile_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return project


def _move_profile_away(profile_root: Path) -> Path:
    """Move ``profile.yml`` out, keeping the old directory. Returns the new path."""
    moved_to = profile_root.parent / "moved-profile"
    moved_to.mkdir()
    (profile_root / "profile.yml").rename(moved_to / "profile.yml")
    return moved_to / "profile.yml"


def test_moved_profile_does_not_get_an_env_recreated_at_the_old_root(tmp_path, caplog):
    """The discriminating case: the directory survives the profile's move.

    Nothing here raises — the old root is a perfectly writable directory — so
    only an explicit check on the profile file itself stops the deploy from
    minting a stranded ``.env`` beside a profile that has left.
    """
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root / "profile.yml")
    moved = _move_profile_away(profile_root)
    assert profile_root.is_dir(), "the recipe requires the old parent directory to survive"
    assert not (profile_root / ".env").exists()

    with caplog.at_level(logging.WARNING):
        container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert not (profile_root / ".env").exists(), (
        "the write-back recreated a .env at a root the profile has left — a second, "
        "orphaned copy of the stack's secrets that no build will ever read"
    )
    assert not (moved.parent / ".env").exists(), (
        "the write-back followed the profile to its new location; the manifest is "
        "stale and only a rebuild may re-point it"
    )
    assert read_secrets_synced_to_profile(project) is False
    manifest = json.loads((project / ".osprey-manifest.json").read_text(encoding="utf-8"))
    assert manifest[SECRETS_SYNCED_KEY] is False
    assert manifest["schema_version"] == "1.2.0", "the stamp must not disturb the manifest"

    # The operator has to be able to find both ends of the break: the profile
    # that is no longer where the build left it, and the file that was not
    # written on its behalf.
    assert str(profile_root / "profile.yml") in caplog.text
    assert str(profile_root / ".env") in caplog.text


def test_secrets_still_reach_the_project_env_when_the_profile_moved(tmp_path):
    """Degrading is not failing: the deploy still comes up on real tokens."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root / "profile.yml")
    _move_profile_away(profile_root)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    project_env = parse_dotenv_text((project / ".env").read_text(encoding="utf-8"))
    for var in _DISPATCH_VARS:
        assert project_env[var], f"{var} must still be minted into the project .env"


def test_a_profile_still_in_place_syncs_as_before(tmp_path):
    """The guard is narrow: an intact profile is written exactly as it was."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root / "profile.yml")

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    profile_env = parse_dotenv_text((profile_root / ".env").read_text(encoding="utf-8"))
    for var in _DISPATCH_VARS:
        assert profile_env[var], f"{var} was not persisted to the profile"
    assert read_secrets_synced_to_profile(project) is True


def test_manifest_pointing_at_a_directory_is_treated_as_stale(tmp_path):
    """A recorded path that is not a *file* names no profile to write to."""
    profile_root = _profile(tmp_path)
    project = _project(tmp_path, profile_root)

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert not (profile_root / ".env").exists()
    assert read_secrets_synced_to_profile(project) is False


def test_persona_delta_that_moved_away_leaves_its_root_untouched(tmp_path):
    """A stranded persona must not seed the root profile's ``.env`` either.

    The root profile is intact here, so the root resolves fine and the write
    would land in a live profile — but on behalf of a persona this project can
    no longer be rebuilt from. Only the delta's own absence catches it.
    """
    profile_root = _profile(tmp_path)
    personas = profile_root / "personas"
    personas.mkdir()
    project = _project(tmp_path, personas / "readonly.yml")

    container_lifecycle._ensure_service_tokens(_CONFIG, False, env_path=project / ".env")

    assert not (profile_root / ".env").exists()
    assert not (personas / ".env").exists()
    assert read_secrets_synced_to_profile(project) is False
