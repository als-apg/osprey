"""The web tier's runtime secrets file is ``.env.users``, and ``up`` migrates.

Two things are pinned here. First, the artifact every per-user web-terminal
container reads is named ``.env.users`` — the generator writes that name, the
refusals name that name, and ``osprey reset`` discloses that name. Second, a
deployment carrying the pre-rename ``.env.production`` is carried onto the new
name by ``osprey up`` rather than being told its secrets are missing: the file
is gitignored and never regenerated once present, so on a host that deployed
successfully yesterday it is the operator's only copy. That second half is
pinned as a NON-destructive migration throughout — a leftover the current file
already stands in for is set aside under a name of its own, never deleted,
because ``up`` recognizes it by filename alone and a file at that name need
never have been this system's to remove.
"""

from __future__ import annotations

import os
import stat
import subprocess

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.reset import WEB_CREDENTIAL_FILES
from osprey.deployment.web_terminals.env_production import (
    LEGACY_USERS_ENV_FILENAME,
    SUPERSEDED_USERS_ENV_FILENAME,
    USERS_ENV_FILENAME,
    ensure_env_production,
    migrate_users_env,
)

# A local-mode web-terminals config with one credential to carry, so the
# generator has something to write and the refusals have something to name.
_LOCAL_CONFIG = {
    "facility": {"name": "Test Facility", "prefix": "test", "timezone": "UTC"},
    "llm": {"provider": "cborg", "api_key_env_var": "CBORG_API_KEY"},
    "modules": {"web_terminals": {"enabled": True, "image_source": "local"}},
}


# ---------------------------------------------------------------------------
# migrate_users_env -- the states a project root can be in.
# ---------------------------------------------------------------------------


def test_migration_moves_the_old_file_onto_the_new_name(tmp_path, caplog):
    """Plain migration: the old file is the only copy, so it becomes the new one."""
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=carried\n", encoding="utf-8")
    os.chmod(legacy, 0o600)

    with caplog.at_level("INFO"):
        result = migrate_users_env(tmp_path)

    users_path = tmp_path / USERS_ENV_FILENAME
    assert result == users_path
    assert not legacy.exists()
    # Contents carried verbatim -- this is a rename, not a re-derivation: the
    # values may be ones OSPREY never saw (CI-assembled or operator-authored).
    assert users_path.read_text(encoding="utf-8") == "CBORG_API_KEY=carried\n"
    # And carried at 0600, the mode the generator creates it with. A migration
    # that widened a secrets file on a multi-user host would be a regression
    # no downstream check catches.
    assert stat.S_IMODE(users_path.stat().st_mode) == 0o600

    # The record names both filenames, so an operator reading the deploy
    # output can see where their secrets went without diffing the directory.
    migration_lines = [rec.message for rec in caplog.records if USERS_ENV_FILENAME in rec.message]
    assert migration_lines, "the migration left no record"
    assert any(
        LEGACY_USERS_ENV_FILENAME in line and USERS_ENV_FILENAME in line for line in migration_lines
    )


def test_both_files_present_keeps_the_new_one_and_sets_the_leftover_aside(tmp_path, caplog):
    """Registry-mode CI's fresh render wins over a survivor of an earlier pipeline.

    CI writes ``.env.users`` on the host just before ``osprey up``, while the
    gitignored old file survives the ``git reset --hard`` that precedes the
    checkout. The fresh file is authoritative and the leftover stops sitting
    under a name a later reader could mistake for the live file.

    It gets there by a rename and NOT by a delete, which is what this pins:
    ``osprey up`` identifies the leftover by filename alone, so a file that was
    never OSPREY's — another tool's secrets under the same spelling — reaches
    this branch too, and a delete would destroy the operator's only copy of it
    without asking. The superseded copy is still secret-bearing, so it lands at
    0600 like every other write of this artifact.
    """
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=stale\n", encoding="utf-8")
    os.chmod(legacy, 0o644)
    users_path = tmp_path / USERS_ENV_FILENAME
    users_path.write_text("CBORG_API_KEY=fresh\n", encoding="utf-8")

    with caplog.at_level("INFO"):
        result = migrate_users_env(tmp_path)

    assert result == users_path
    assert not legacy.exists()
    assert users_path.read_text(encoding="utf-8") == "CBORG_API_KEY=fresh\n"

    # The leftover was moved, not destroyed: same bytes, under the set-aside
    # name, at the mode a secrets file is entitled to rather than the one it
    # happened to carry.
    superseded = tmp_path / SUPERSEDED_USERS_ENV_FILENAME
    assert superseded.read_text(encoding="utf-8") == "CBORG_API_KEY=stale\n"
    assert stat.S_IMODE(superseded.stat().st_mode) == 0o600

    supersede_lines = [rec.message for rec in caplog.records if USERS_ENV_FILENAME in rec.message]
    assert supersede_lines, "the supersede left no record"
    # Both paths in one line, so the operator can find the copy without diffing
    # the directory -- and nothing anywhere claiming a removal that no longer
    # happens.
    assert any(
        LEGACY_USERS_ENV_FILENAME in line and SUPERSEDED_USERS_ENV_FILENAME in line
        for line in supersede_lines
    )
    assert not any("removed" in line for line in supersede_lines)


def test_a_second_supersede_keeps_the_earlier_copy_and_skips_the_rename(tmp_path, caplog):
    """The set-aside name is fixed, so a repeated ``up`` must not overwrite it.

    A timestamped name would pile up copies of a secrets file, so there is one;
    the price is that a second deploy in this state finds it taken. The earlier
    copy is the one closer to the deployment that last worked, so it wins, the
    leftover stays where it is, and nothing is destroyed on either side.
    """
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=second\n", encoding="utf-8")
    superseded = tmp_path / SUPERSEDED_USERS_ENV_FILENAME
    superseded.write_text("CBORG_API_KEY=first\n", encoding="utf-8")
    users_path = tmp_path / USERS_ENV_FILENAME
    users_path.write_text("CBORG_API_KEY=fresh\n", encoding="utf-8")

    with caplog.at_level("INFO"):
        result = migrate_users_env(tmp_path)

    assert result == users_path
    assert superseded.read_text(encoding="utf-8") == "CBORG_API_KEY=first\n"
    assert legacy.read_text(encoding="utf-8") == "CBORG_API_KEY=second\n"
    assert users_path.read_text(encoding="utf-8") == "CBORG_API_KEY=fresh\n"

    # One line, naming both paths: which file was left alone and which copy it
    # was left alone for.
    skip_lines = [rec.message for rec in caplog.records if LEGACY_USERS_ENV_FILENAME in rec.message]
    assert len(skip_lines) == 1
    assert SUPERSEDED_USERS_ENV_FILENAME in skip_lines[0]


def test_no_old_file_is_a_silent_no_op(tmp_path, caplog):
    """Neither file, or only the current one: nothing to say and nothing to do."""
    with caplog.at_level("INFO"):
        assert migrate_users_env(tmp_path) is None
    assert not (tmp_path / USERS_ENV_FILENAME).exists()
    assert not any(USERS_ENV_FILENAME in rec.message for rec in caplog.records)

    users_path = tmp_path / USERS_ENV_FILENAME
    users_path.write_text("CBORG_API_KEY=fresh\n", encoding="utf-8")
    caplog.clear()
    with caplog.at_level("INFO"):
        assert migrate_users_env(tmp_path) is None
    # The current file is never touched by a migration that has no old file to
    # act on -- no rewrite, no log line claiming one happened.
    assert users_path.read_text(encoding="utf-8") == "CBORG_API_KEY=fresh\n"
    assert not any(USERS_ENV_FILENAME in rec.message for rec in caplog.records)


def test_a_directory_named_like_the_old_file_is_not_migrated(tmp_path):
    """``is_file`` gating: a directory of that name is not an env file to move."""
    (tmp_path / LEGACY_USERS_ENV_FILENAME).mkdir()

    assert migrate_users_env(tmp_path) is None
    assert (tmp_path / LEGACY_USERS_ENV_FILENAME).is_dir()
    assert not (tmp_path / USERS_ENV_FILENAME).exists()


def test_a_directory_at_the_new_name_never_moves_the_old_file(tmp_path):
    """The supersede branch needs a real file standing in, not merely something.

    A directory at the new name is not a copy of anything, so treating it as
    one would move the operator's only copy of the web tier's secrets out from
    under the deploy that is about to read it. The migration fails loudly
    instead, with the old file where it was and nothing set aside.
    """
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=carried\n", encoding="utf-8")
    (tmp_path / USERS_ENV_FILENAME).mkdir()

    with pytest.raises(OSError):
        migrate_users_env(tmp_path)

    assert legacy.read_text(encoding="utf-8") == "CBORG_API_KEY=carried\n"
    assert not (tmp_path / SUPERSEDED_USERS_ENV_FILENAME).exists()


def test_a_world_readable_old_file_is_migrated_to_0600(tmp_path):
    """Enforced, not inherited: a pre-0600 file does not arrive still readable.

    The artifact predates the mode convention on some hosts (hand-authored, or
    written at the umask), and a rename carries whatever mode it had.
    """
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=carried\n", encoding="utf-8")
    os.chmod(legacy, 0o644)

    users_path = migrate_users_env(tmp_path)

    assert users_path == tmp_path / USERS_ENV_FILENAME
    assert stat.S_IMODE(users_path.stat().st_mode) == 0o600
    assert users_path.read_text(encoding="utf-8") == "CBORG_API_KEY=carried\n"


# ---------------------------------------------------------------------------
# The writer and its refusals carry the new name.
# ---------------------------------------------------------------------------


def test_local_mode_generation_writes_env_users(tmp_path):
    (tmp_path / ".env").write_text("CBORG_API_KEY=secret\n", encoding="utf-8")

    written = ensure_env_production(_LOCAL_CONFIG, tmp_path)

    assert written == tmp_path / USERS_ENV_FILENAME
    assert "CBORG_API_KEY=secret" in written.read_text(encoding="utf-8")
    assert not (tmp_path / LEGACY_USERS_ENV_FILENAME).exists()


def test_an_existing_env_users_is_returned_untouched(tmp_path):
    users_path = tmp_path / USERS_ENV_FILENAME
    users_path.write_text("CBORG_API_KEY=authored\n", encoding="utf-8")
    (tmp_path / ".env").write_text("CBORG_API_KEY=derived\n", encoding="utf-8")

    assert ensure_env_production(_LOCAL_CONFIG, tmp_path) == users_path
    assert users_path.read_text(encoding="utf-8") == "CBORG_API_KEY=authored\n"


def test_registry_mode_refusal_names_env_users(tmp_path):
    config = {
        **_LOCAL_CONFIG,
        "modules": {"web_terminals": {"enabled": True, "image_source": "registry"}},
    }

    with pytest.raises(RuntimeError, match=r"\.env\.users not found"):
        ensure_env_production(config, tmp_path)


def test_local_mode_refusal_with_no_chain_names_env_users(tmp_path):
    with pytest.raises(RuntimeError, match=r"\.env\.users nor an env-chain file"):
        ensure_env_production(_LOCAL_CONFIG, tmp_path)


def test_reset_discloses_the_kept_file_under_the_new_name():
    """``osprey reset`` keeps this file and says so -- under the name on disk."""
    disclosed = [entry[0] for entry in WEB_CREDENTIAL_FILES]
    assert USERS_ENV_FILENAME in disclosed
    assert LEGACY_USERS_ENV_FILENAME not in disclosed
    # The set-aside copy is disclosed too, for the reason every entry in that
    # list is: it survives a reset, it holds live secrets, and an operator who
    # is not told it is there has no way to know they can drop it.
    assert SUPERSEDED_USERS_ENV_FILENAME in disclosed


# ---------------------------------------------------------------------------
# Wiring: `osprey up` migrates before the first thing that reads the file.
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_start_stack(monkeypatch):
    """Neutralize everything ``_start_stack`` does except the migration.

    The point of the tests below is ordering, so every provisioner, preflight
    and compose invocation is a no-op; the tests re-patch the one collaborator
    whose timing they are asking about.
    """
    for name in (
        "_check_shared_disk_preflight",
        "_preflight_host_ports",
        "_ensure_service_tokens",
        "_preflight_archiver_pymongo",
        "_ensure_bluesky_substrate_env",
        "_ensure_bluesky_control_plane_keys",
        "_ensure_bluesky_document_plane_certs",
        "_preflight_env_chain_drift",
        "_preflight_env_shadowing",
        "_build_project_image",
        "log_endpoint_summary",
    ):
        monkeypatch.setattr(container_lifecycle, name, lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(container_lifecycle, "deploy_up_web_terminals", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "preflight_web_terminals", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "_reconcile_exposure", lambda *a, **k: False)
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(
        container_lifecycle.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0),
    )


def _web_config(enabled: bool = True) -> dict:
    # A backend service alongside, so the disabled case still walks the whole
    # start path instead of taking the empty-deploy early return.
    return {
        "deployed_services": ["osprey-mcp"],
        "modules": {"web_terminals": {"enabled": enabled}},
    }


def test_up_migrates_before_the_web_terminal_preflight_reads_the_file(
    tmp_path, stubbed_start_stack, monkeypatch
):
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=carried\n", encoding="utf-8")

    # Recorded from inside the first reader, not after the call: the file has
    # to be under the new name by the time the preflight looks, or the deploy
    # refuses with "not found" on a host whose secrets are right there.
    at_preflight: list[bool] = []
    monkeypatch.setattr(
        container_lifecycle,
        "preflight_web_terminals",
        lambda *a, **k: at_preflight.append((tmp_path / USERS_ENV_FILENAME).is_file()),
    )

    container_lifecycle._start_stack(_web_config(), [], tmp_path, detached=True)

    assert at_preflight == [True]
    assert not legacy.exists()


def test_a_deploy_with_no_web_terminals_leaves_the_old_file_alone(tmp_path, stubbed_start_stack):
    """Nothing runs the web tier here, so nothing renames its artifact."""
    legacy = tmp_path / LEGACY_USERS_ENV_FILENAME
    legacy.write_text("CBORG_API_KEY=carried\n", encoding="utf-8")

    container_lifecycle._start_stack(_web_config(enabled=False), [], tmp_path, detached=True)

    assert legacy.read_text(encoding="utf-8") == "CBORG_API_KEY=carried\n"
    assert not (tmp_path / USERS_ENV_FILENAME).exists()
