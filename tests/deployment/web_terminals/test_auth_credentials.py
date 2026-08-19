"""Tests for deploy-time web-terminal auth credential provisioning."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from osprey.deployment.web_terminals.auth_credentials import (
    _HASH_HEADER,
    AUTH_ENV_FILENAME,
    PW_HASH_VAR_PREFIX,
    PW_PLAINTEXT_VAR_PREFIX,
    SESSION_SECRET_VAR,
    STATE_SECRET_VAR,
    ensure_auth_credentials,
    ensure_auth_session_secrets,
    purge_auth_credentials,
    seeded_logins,
    set_auth_password,
)
from osprey.services.auth_sidecar.passwords import (
    FIELD_SEP,
    SCHEME,
    generation_tag,
    verify_password,
)
from osprey.utils.dotenv import parse_dotenv_file

pytestmark = pytest.mark.unit

# The unwritable-file cases below rely on the OS honoring a read-only mode.
# root ignores it, so those assertions would be vacuous there.
running_as_root = hasattr(os, "geteuid") and os.geteuid() == 0


class Echo:
    """Collects the one-time minted-password notices."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        self.lines.append(message)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def read_auth_env(project_root: Path) -> dict[str, str]:
    return parse_dotenv_file(project_root / AUTH_ENV_FILENAME)


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_mints_hash_for_a_user_with_no_credential(tmp_path: Path) -> None:
    echo = Echo()

    result = ensure_auth_credentials(["alice"], tmp_path, echo=echo)

    assert result.minted == ("alice",)
    assert result.hashed_from_plaintext == ()
    assert result.preexisting == ()
    assert result.missing == ()
    assert result.changed is True
    assert result.env_auth_path == tmp_path / AUTH_ENV_FILENAME

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert stored.startswith(f"{SCHEME}{FIELD_SEP}")


def test_no_env_auth_value_contains_a_dollar_sign(tmp_path: Path) -> None:
    """Every value in ``.env.auth`` must encode without ``$`` — a whole-file rule.

    The rendered compose overlay hands this file to the sidecar as an
    ``env_file``, and Docker Compose interpolates ``${...}`` in env_file values:
    each ``$`` followed by an identifier character is replaced with the empty
    string. A ``$``-bearing secret therefore arrives at the container truncated,
    and the failure is invisible from the host — the file on disk is correct,
    every unit test that reads it passes, and only the login refuses.

    Asserted over the FILE rather than over the hash alone because the hash was
    merely the first value to hit it: the signing secrets share this file, and
    so will anything added later. Base64url (``A-Za-z0-9-_``) and the ``.``
    separator are both safe; a future encoding that reaches for ``$`` is not.
    """
    ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())
    ensure_auth_session_secrets(tmp_path)

    offenders = {name: value for name, value in read_auth_env(tmp_path).items() if "$" in value}
    assert offenders == {}, (
        f"{AUTH_ENV_FILENAME} values containing '$' would be truncated by compose "
        f"env_file interpolation before the sidecar ever reads them: {sorted(offenders)}"
    )


def test_env_auth_is_created_0600(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert file_mode(tmp_path / AUTH_ENV_FILENAME) == 0o600


def test_existing_env_auth_with_loose_mode_is_tightened(tmp_path: Path) -> None:
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text(f"{PW_HASH_VAR_PREFIX}ALICE=scrypt.16384.8.1.AAAA.AAAB\n")
    env_auth.chmod(0o644)

    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.changed is False  # a permission fix is not a content change
    assert file_mode(env_auth) == 0o600


def test_minted_password_is_printed_exactly_once_and_verifies(tmp_path: Path) -> None:
    echo = Echo()

    ensure_auth_credentials(["alice"], tmp_path, echo=echo)

    # Recover the printed password from the notice, then prove it is the one
    # the stored hash accepts.
    notice = next(line for line in echo.lines if "'alice'" in line)
    password = notice.rsplit(": ", 1)[1]
    assert password.isalnum()  # dotenv-safe, no transcription lookalikes
    assert echo.text.count(password) == 1

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert verify_password(password, stored)
    assert not verify_password("wrong-password", stored)


def test_minted_password_never_reaches_the_log(tmp_path: Path, caplog) -> None:
    echo = Echo()

    with caplog.at_level("DEBUG"):
        ensure_auth_credentials(["alice"], tmp_path, echo=echo)

    # Prove the instrument works before asserting on what it did NOT capture.
    assert f"{PW_HASH_VAR_PREFIX}ALICE" in caplog.text

    password = next(line for line in echo.lines if "'alice'" in line).rsplit(": ", 1)[1]
    assert password not in caplog.text
    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert stored not in caplog.text


def test_plaintext_from_project_dotenv_is_hashed_in(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=operator-choice\n")
    echo = Echo()

    result = ensure_auth_credentials(["alice"], tmp_path, echo=echo)

    assert result.hashed_from_plaintext == ("alice",)
    assert result.minted == ()
    assert result.changed is True

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert verify_password("operator-choice", stored)
    # The plaintext is neither echoed nor copied into the sidecar's env file.
    assert "operator-choice" not in echo.text
    assert "operator-choice" not in (tmp_path / AUTH_ENV_FILENAME).read_text()


def test_existing_hash_wins_over_plaintext(tmp_path: Path) -> None:
    existing = f"{PW_HASH_VAR_PREFIX}ALICE=scrypt.16384.8.1.AAAA.AAAB"
    (tmp_path / AUTH_ENV_FILENAME).write_text(f"{existing}\n")
    (tmp_path / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=operator-choice\n")

    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.preexisting == ("alice",)
    assert result.changed is False
    assert read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"] == existing.split("=", 1)[1]


def test_plaintext_wins_over_minting(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}BOB=bobs-password\n")
    echo = Echo()

    result = ensure_auth_credentials(["alice", "bob"], tmp_path, echo=echo)

    assert result.minted == ("alice",)
    assert result.hashed_from_plaintext == ("bob",)
    assert "'bob'" not in echo.text


def test_rerun_is_idempotent_and_reports_no_change(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())
    before = (tmp_path / AUTH_ENV_FILENAME).read_text()

    echo = Echo()
    result = ensure_auth_credentials(["alice", "bob"], tmp_path, echo=echo)

    assert result.changed is False
    assert result.preexisting == ("alice", "bob")
    assert result.minted == ()
    assert echo.lines == []
    assert (tmp_path / AUTH_ENV_FILENAME).read_text() == before


def test_only_the_missing_user_is_appended(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice"], tmp_path, echo=Echo())
    alice_hash = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]

    result = ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())

    assert result.preexisting == ("alice",)
    assert result.minted == ("bob",)
    stored = read_auth_env(tmp_path)
    assert stored[f"{PW_HASH_VAR_PREFIX}ALICE"] == alice_hash
    assert stored[f"{PW_HASH_VAR_PREFIX}BOB"].startswith(f"{SCHEME}{FIELD_SEP}")


def test_append_to_a_file_without_a_trailing_newline(tmp_path: Path) -> None:
    (tmp_path / AUTH_ENV_FILENAME).write_text("# hand-written, no trailing newline")

    ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert f"{PW_HASH_VAR_PREFIX}ALICE" in read_auth_env(tmp_path)
    text = (tmp_path / AUTH_ENV_FILENAME).read_text()
    assert "newline#" not in text  # the appended block starts on its own line


def test_a_hash_in_the_process_env_does_not_suppress_minting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sidecar reads .env.auth only, so a hash visible just to this process
    # must not stand in for a persisted one.
    monkeypatch.setenv(f"{PW_HASH_VAR_PREFIX}ALICE", "scrypt.16384.8.1.AAAA.AAAB")

    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.minted == ("alice",)
    assert read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"] != "scrypt.16384.8.1.AAAA.AAAB"


def test_plaintext_is_not_read_from_the_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stray exported variable must not silently become a deployed credential.
    monkeypatch.setenv(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE", "exported-password")

    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.minted == ("alice",)
    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert not verify_password("exported-password", stored)


def test_a_user_listed_twice_is_provisioned_once(tmp_path: Path) -> None:
    result = ensure_auth_credentials(["alice", "alice"], tmp_path, echo=Echo())

    assert result.minted == ("alice",)
    text = (tmp_path / AUTH_ENV_FILENAME).read_text()
    assert text.count(f"{PW_HASH_VAR_PREFIX}ALICE=") == 1


def test_empty_roster_writes_nothing(tmp_path: Path) -> None:
    result = ensure_auth_credentials([], tmp_path, echo=Echo())

    assert result.changed is False
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


@pytest.mark.parametrize("username", ["Alice", "al ice", "-alice", "alice.b", "_alice"])
def test_invalid_username_charset_raises(tmp_path: Path, username: str) -> None:
    with pytest.raises(RuntimeError, match="Refusing to deploy"):
        ensure_auth_credentials([username], tmp_path, echo=Echo())

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_username_with_a_trailing_newline_is_rejected(tmp_path: Path) -> None:
    """The charset gate is a `fullmatch`, not a `match`.

    Python's `$` also matches *before* a trailing newline, so a `match` would let
    "alice\\n" through this deploy-path gate and on into an nginx location key.
    """
    with pytest.raises(RuntimeError, match="Refusing to deploy"):
        ensure_auth_credentials(["alice\n"], tmp_path, echo=Echo())

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_env_var_suffix_collision_raises_naming_both_users(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        ensure_auth_credentials(["alice-b", "alice_b"], tmp_path, echo=Echo())

    message = str(excinfo.value)
    assert "'alice-b'" in message
    assert "'alice_b'" in message
    assert f"{PW_HASH_VAR_PREFIX}ALICE_B" in message
    # Nothing is written when the roster is ambiguous.
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_unwritable_env_auth_reports_missing_without_raising(tmp_path: Path) -> None:
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text("# pre-existing\n")
    env_auth.chmod(0o400)
    echo = Echo()
    try:
        result = ensure_auth_credentials(["alice", "bob"], tmp_path, echo=echo)
    finally:
        env_auth.chmod(0o600)

    assert result.missing == ("alice", "bob")
    assert result.changed is False
    assert result.minted == ()
    # A password that could not be persisted is never shown to the operator.
    assert echo.lines == []


def test_empty_hash_value_is_re_minted_and_the_new_entry_wins(tmp_path: Path) -> None:
    # An empty value is not an established credential — left alone it would
    # lock the user out permanently. The re-minted entry is appended after the
    # empty one, and last assignment wins for the parser the sidecar shares.
    (tmp_path / AUTH_ENV_FILENAME).write_text(f"{PW_HASH_VAR_PREFIX}ALICE=\n")
    echo = Echo()

    result = ensure_auth_credentials(["alice"], tmp_path, echo=echo)

    assert result.minted == ("alice",)
    assert result.preexisting == ()
    assert result.changed is True

    password = next(line for line in echo.lines if "'alice'" in line).rsplit(": ", 1)[1]
    assert verify_password(password, read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"])


def test_whitespace_padded_plaintext_is_trimmed_before_hashing(tmp_path: Path) -> None:
    # The trim makes the stored hash match what the operator types at the
    # prompt, rather than a value only a copy-paste could reproduce.
    (tmp_path / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=  padded-password  \n")

    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.hashed_from_plaintext == ("alice",)
    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert verify_password("padded-password", stored)
    assert not verify_password("  padded-password  ", stored)


def test_whitespace_only_plaintext_falls_through_to_minting(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f'{PW_PLAINTEXT_VAR_PREFIX}ALICE="   "\n')

    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.minted == ("alice",)
    assert result.hashed_from_plaintext == ()


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_a_failing_chmod_does_not_break_the_reported_not_raised_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a read-only filesystem the append AND the trailing permission fix both
    # fail. The write failure must still surface through `missing` rather than
    # as an escaping OSError from the last statement in the function.
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text("# pre-existing\n")
    env_auth.chmod(0o400)

    real_chmod = os.chmod

    def failing_chmod(path, mode, *args, **kwargs):
        if str(path) == str(env_auth):
            raise OSError(30, "Read-only file system")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", failing_chmod)
    try:
        result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())
    finally:
        monkeypatch.undo()
        env_auth.chmod(0o600)

    assert result.missing == ("alice",)
    assert result.changed is False


def test_a_failing_chmod_still_returns_a_successful_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_chmod = os.chmod

    def failing_chmod(path, mode, *args, **kwargs):
        if str(path).endswith(AUTH_ENV_FILENAME):
            raise OSError(1, "Operation not permitted")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", failing_chmod)
    result = ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    assert result.changed is True
    assert result.minted == ("alice",)
    assert result.missing == ()
    assert f"{PW_HASH_VAR_PREFIX}ALICE" in read_auth_env(tmp_path)


def test_purge_normalizes_crlf_input_but_keeps_line_content(tmp_path: Path) -> None:
    (tmp_path / AUTH_ENV_FILENAME).write_bytes(
        b"# header\r\n"
        b"OSPREY_AUTH_PW_HASH_ALICE=scrypt.16384.8.1.AAAA.AAAB\r\n"
        b"OSPREY_AUTH_PW_HASH_BOB=scrypt.16384.8.1.AAAA.AAAC\r\n"
    )

    assert purge_auth_credentials("alice", tmp_path) is True

    text = (tmp_path / AUTH_ENV_FILENAME).read_text()
    assert "ALICE" not in text
    assert "\r" not in text  # CRLF is normalized to LF on rewrite
    assert text == "# header\nOSPREY_AUTH_PW_HASH_BOB=scrypt.16384.8.1.AAAA.AAAC\n"


def test_purge_removes_only_the_named_users_entries(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())
    bob_hash = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}BOB"]

    changed = purge_auth_credentials("alice", tmp_path)

    assert changed is True
    stored = read_auth_env(tmp_path)
    assert f"{PW_HASH_VAR_PREFIX}ALICE" not in stored
    assert stored[f"{PW_HASH_VAR_PREFIX}BOB"] == bob_hash


def test_purge_removes_a_hand_placed_plaintext_entry(tmp_path: Path) -> None:
    (tmp_path / AUTH_ENV_FILENAME).write_text(
        f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=hand-placed\n"
        f"export {PW_HASH_VAR_PREFIX}ALICE=scrypt.16384.8.1.AAAA.AAAB\n"
        "# keep me\n"
    )

    assert purge_auth_credentials("alice", tmp_path) is True

    text = (tmp_path / AUTH_ENV_FILENAME).read_text()
    assert "ALICE" not in text
    assert "# keep me" in text


def test_purge_preserves_mode_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())

    purge_auth_credentials("alice", tmp_path)

    assert file_mode(tmp_path / AUTH_ENV_FILENAME) == 0o600
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_purge_is_a_no_op_for_an_unknown_user(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice"], tmp_path, echo=Echo())
    before = (tmp_path / AUTH_ENV_FILENAME).read_text()

    assert purge_auth_credentials("carol", tmp_path) is False
    assert (tmp_path / AUTH_ENV_FILENAME).read_text() == before


def test_purge_is_a_no_op_when_env_auth_is_absent(tmp_path: Path) -> None:
    assert purge_auth_credentials("alice", tmp_path) is False
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_purge_uses_the_same_suffix_mapping_as_provisioning(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice-b"], tmp_path, echo=Echo())
    assert f"{PW_HASH_VAR_PREFIX}ALICE_B" in read_auth_env(tmp_path)

    assert purge_auth_credentials("alice-b", tmp_path) is True
    assert read_auth_env(tmp_path) == {}


def test_readded_user_gets_a_fresh_credential_after_purge(tmp_path: Path) -> None:
    echo = Echo()
    ensure_auth_credentials(["alice"], tmp_path, echo=echo)
    old_password = next(line for line in echo.lines if "'alice'" in line).rsplit(": ", 1)[1]

    purge_auth_credentials("alice", tmp_path)
    ensure_auth_credentials(["alice"], tmp_path, echo=Echo())

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert not verify_password(old_password, stored)


# --- Session/state signing secrets (task 4.2) --------------------------------


def test_secrets_are_minted_into_env_auth(tmp_path: Path) -> None:
    result = ensure_auth_session_secrets(tmp_path)

    assert result.minted == (SESSION_SECRET_VAR, STATE_SECRET_VAR)
    assert result.preexisting == ()
    assert result.missing == ()
    assert result.changed is True

    stored = read_auth_env(tmp_path)
    # token_urlsafe(32) — 256 bits, the shared service-token recipe.
    assert len(stored[SESSION_SECRET_VAR]) >= 40
    assert stored[SESSION_SECRET_VAR] != stored[STATE_SECRET_VAR]


def test_secret_minting_creates_env_auth_0600(tmp_path: Path) -> None:
    ensure_auth_session_secrets(tmp_path)

    assert file_mode(tmp_path / AUTH_ENV_FILENAME) == 0o600


def test_secrets_are_idempotent_and_report_no_change(tmp_path: Path) -> None:
    ensure_auth_session_secrets(tmp_path)
    before = (tmp_path / AUTH_ENV_FILENAME).read_text()

    result = ensure_auth_session_secrets(tmp_path)

    # Re-minting would invalidate every live session.
    assert result.changed is False
    assert result.minted == ()
    assert result.preexisting == (SESSION_SECRET_VAR, STATE_SECRET_VAR)
    assert (tmp_path / AUTH_ENV_FILENAME).read_text() == before


def test_an_existing_secret_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / AUTH_ENV_FILENAME).write_text(f"{SESSION_SECRET_VAR}=operator-chosen-secret\n")

    result = ensure_auth_session_secrets(tmp_path)

    assert result.preexisting == (SESSION_SECRET_VAR,)
    assert result.minted == (STATE_SECRET_VAR,)
    assert read_auth_env(tmp_path)[SESSION_SECRET_VAR] == "operator-chosen-secret"


@pytest.mark.parametrize("empty_value", ["", "   "])
def test_an_empty_secret_is_re_minted(tmp_path: Path, empty_value: str) -> None:
    # The sidecar strips and rejects an empty value as missing configuration
    # and answers every request with 503, so preserving it would keep the whole
    # deployment locked out.
    (tmp_path / AUTH_ENV_FILENAME).write_text(f'{SESSION_SECRET_VAR}="{empty_value}"\n')

    result = ensure_auth_session_secrets(tmp_path)

    assert SESSION_SECRET_VAR in result.minted
    assert read_auth_env(tmp_path)[SESSION_SECRET_VAR].strip()


def test_secrets_never_reach_the_log(tmp_path: Path, caplog) -> None:
    with caplog.at_level("DEBUG"):
        ensure_auth_session_secrets(tmp_path)

    # The instrument works: variable names ARE logged.
    assert SESSION_SECRET_VAR in caplog.text
    stored = read_auth_env(tmp_path)
    assert stored[SESSION_SECRET_VAR] not in caplog.text
    assert stored[STATE_SECRET_VAR] not in caplog.text


def test_secrets_coexist_with_password_hashes_in_one_file(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice"], tmp_path, echo=Echo())
    secrets_result = ensure_auth_session_secrets(tmp_path)

    assert secrets_result.changed is True
    stored = read_auth_env(tmp_path)
    assert stored[f"{PW_HASH_VAR_PREFIX}ALICE"].startswith(f"{SCHEME}{FIELD_SEP}")
    assert stored[SESSION_SECRET_VAR]
    # Order-independent: minting secrets first must not disturb the hashes.
    assert ensure_auth_credentials(["alice"], tmp_path, echo=Echo()).changed is False


def test_secret_minting_is_not_wired_through_service_token_vars() -> None:
    # _SERVICE_TOKEN_VARS only mints for members of `deployed_services`, which
    # no web-terminal-stack service ever is — an entry there would silently
    # mint nothing.
    from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS

    declared = {var for vars_ in _SERVICE_TOKEN_VARS.values() for var in vars_}
    assert SESSION_SECRET_VAR not in declared
    assert STATE_SECRET_VAR not in declared


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_unwritable_env_auth_reports_missing_secrets_without_raising(tmp_path: Path) -> None:
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text("# pre-existing\n")
    env_auth.chmod(0o400)
    try:
        result = ensure_auth_session_secrets(tmp_path)
    finally:
        env_auth.chmod(0o600)

    assert result.missing == (SESSION_SECRET_VAR, STATE_SECRET_VAR)
    assert result.minted == ()
    assert result.changed is False


def test_an_invalid_existing_secret_raises_without_echoing_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The validator is imported from service_tokens, so a constraint registered
    # for these vars later takes effect on this path with no further work.
    from osprey.deployment import service_tokens

    monkeypatch.setitem(service_tokens._VAR_VALIDATORS, SESSION_SECRET_VAR, str.isalnum)
    monkeypatch.setitem(
        service_tokens._VAR_VALIDATOR_DESCRIPTIONS, SESSION_SECRET_VAR, "must be alphanumeric"
    )
    (tmp_path / AUTH_ENV_FILENAME).write_text(f"{SESSION_SECRET_VAR}=not+alphanumeric/value\n")

    with pytest.raises(RuntimeError) as excinfo:
        ensure_auth_session_secrets(tmp_path)

    message = str(excinfo.value)
    assert SESSION_SECRET_VAR in message
    assert "must be alphanumeric" in message
    assert "not+alphanumeric/value" not in message


# =============================================================================
# set_auth_password: rotation (osprey users passwd)
# =============================================================================


def _hash_lines(project_root: Path, user_suffix: str) -> list[str]:
    """Every raw line in .env.auth assigning that user's hash variable."""
    text = (project_root / AUTH_ENV_FILENAME).read_text(encoding="utf-8")
    var = f"{PW_HASH_VAR_PREFIX}{user_suffix}="
    return [line for line in text.splitlines() if line.startswith(var)]


def test_set_auth_password_stores_a_hash_that_verifies(tmp_path: Path) -> None:
    assert set_auth_password("alice", "new-password", tmp_path) == tmp_path / AUTH_ENV_FILENAME

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert verify_password("new-password", stored)


def test_set_auth_password_replaces_rather_than_appends(tmp_path: Path) -> None:
    """The rotated file holds ONE entry for the user, not a new one after the old.

    Appending would also "work" — the parser takes the last assignment — but the
    superseded hash would stay readable in the file, and the file's correctness
    would rest on parse order rather than on its content.
    """
    set_auth_password("alice", "first-password", tmp_path)
    first = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]

    set_auth_password("alice", "second-password", tmp_path)

    assert len(_hash_lines(tmp_path, "ALICE")) == 1
    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert stored != first
    assert first not in (tmp_path / AUTH_ENV_FILENAME).read_text(encoding="utf-8")


def test_set_auth_password_does_not_accumulate_across_repeated_rotations(tmp_path: Path) -> None:
    for password in ("one", "two", "three", "four"):
        set_auth_password("alice", password, tmp_path)

    assert len(_hash_lines(tmp_path, "ALICE")) == 1
    assert verify_password("four", read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"])


def test_set_auth_password_does_not_accumulate_headers(tmp_path: Path) -> None:
    """The file does not grow at all across rotations — not by entries, and not
    by orphaned header comments either.

    An append-style write emitted a fresh `# Auto-generated ...` header on every
    rotation, so a facility rotating on a schedule accumulated one comment line
    per rotation, each above nothing. The hash count stayed correct, which is
    why this needs its own pin.
    """
    set_auth_password("alice", "first", tmp_path)
    after_first = (tmp_path / AUTH_ENV_FILENAME).read_text(encoding="utf-8")

    for password in ("second", "third", "fourth", "fifth"):
        set_auth_password("alice", password, tmp_path)

    text = (tmp_path / AUTH_ENV_FILENAME).read_text(encoding="utf-8")
    assert text.count(_HASH_HEADER) == 1
    # Same shape, same line count — only the hash value differs.
    assert len(text.splitlines()) == len(after_first.splitlines())


def test_set_auth_password_rotation_moves_no_other_line(tmp_path: Path) -> None:
    """Substituted IN PLACE: every line except the user's own is byte-identical
    and in the same position, so a rotation cannot reorder or drop a neighbour's
    credential."""
    ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())
    ensure_auth_session_secrets(tmp_path)
    before = (tmp_path / AUTH_ENV_FILENAME).read_text(encoding="utf-8").splitlines()

    set_auth_password("alice", "alices-new-password", tmp_path)

    after = (tmp_path / AUTH_ENV_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before)
    differing = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(differing) == 1
    assert after[differing[0]].startswith(f"{PW_HASH_VAR_PREFIX}ALICE=")


def test_set_auth_password_only_the_old_password_stops_working(tmp_path: Path) -> None:
    set_auth_password("alice", "old-password", tmp_path)
    set_auth_password("alice", "new-password", tmp_path)

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert verify_password("new-password", stored)
    assert not verify_password("old-password", stored)


def test_set_auth_password_changes_the_generation_tag(tmp_path: Path) -> None:
    """The tag is what ends the user's live sessions.

    A session cookie carries the generation tag of the hash it was issued
    against, so a rotation that left the tag unchanged would leave every
    pre-rotation session valid — the opposite of what rotating a password means.
    """
    set_auth_password("alice", "old-password", tmp_path)
    before = generation_tag(read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"])

    set_auth_password("alice", "new-password", tmp_path)
    after = generation_tag(read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"])

    assert before != after


def test_set_auth_password_leaves_other_users_untouched(tmp_path: Path) -> None:
    ensure_auth_credentials(["alice", "bob"], tmp_path, echo=Echo())
    bobs_hash = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}BOB"]

    set_auth_password("alice", "alices-new-password", tmp_path)

    assert read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}BOB"] == bobs_hash
    assert len(_hash_lines(tmp_path, "BOB")) == 1


def test_set_auth_password_preserves_the_signing_secrets(tmp_path: Path) -> None:
    """Rotating one password must not invalidate every OTHER user's session,
    which is what dropping the session secret would do."""
    ensure_auth_session_secrets(tmp_path)
    before = read_auth_env(tmp_path)[SESSION_SECRET_VAR]

    set_auth_password("alice", "alices-password", tmp_path)

    assert read_auth_env(tmp_path)[SESSION_SECRET_VAR] == before


def test_set_auth_password_creates_the_file_0600_when_absent(tmp_path: Path) -> None:
    assert not (tmp_path / AUTH_ENV_FILENAME).exists()

    set_auth_password("alice", "alices-password", tmp_path)

    assert file_mode(tmp_path / AUTH_ENV_FILENAME) == 0o600


def test_set_auth_password_tightens_a_loose_mode(tmp_path: Path) -> None:
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text(f"{PW_HASH_VAR_PREFIX}ALICE=stale\n")
    env_auth.chmod(0o644)

    set_auth_password("alice", "alices-password", tmp_path)

    assert file_mode(env_auth) == 0o600


def test_set_auth_password_never_writes_the_cleartext(tmp_path: Path) -> None:
    set_auth_password("alice", "unmistakable-cleartext", tmp_path)

    assert "unmistakable-cleartext" not in (tmp_path / AUTH_ENV_FILENAME).read_text(
        encoding="utf-8"
    )


def test_set_auth_password_never_logs_the_cleartext(tmp_path: Path, caplog) -> None:
    with caplog.at_level("DEBUG"):
        set_auth_password("alice", "unmistakable-cleartext", tmp_path)

    assert "unmistakable-cleartext" not in caplog.text
    # Nor the hash it produced — a hash identifies a credential generation.
    assert read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"] not in caplog.text


def test_set_auth_password_strips_surrounding_whitespace(tmp_path: Path) -> None:
    """Same treatment the `.env` plaintext path gives, so a password set either
    way behaves identically at the login prompt."""
    set_auth_password("alice", "  padded-password\n", tmp_path)

    stored = read_auth_env(tmp_path)[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert verify_password("padded-password", stored)


@pytest.mark.parametrize("password", ["", "   ", "\n\t "])
def test_set_auth_password_rejects_an_empty_password(tmp_path: Path, password: str) -> None:
    with pytest.raises(ValueError, match="empty password"):
        set_auth_password("alice", password, tmp_path)

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


def test_set_auth_password_rejects_a_username_outside_the_charset(tmp_path: Path) -> None:
    """The same hard gate `ensure_auth_credentials` applies — a name that is not
    a legal nginx location key/URL segment can never be authorized."""
    with pytest.raises(RuntimeError, match="does not match"):
        set_auth_password("alice&bob", "some-password", tmp_path)

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_set_auth_password_raises_on_a_write_failure(tmp_path: Path) -> None:
    """A DELIBERATE divergence from this module's reported-not-raised contract.

    `ensure_*` reports through `missing` because a fail-closed deploy gate
    downstream owns the abort. Nothing owns this one — it is reached from an
    interactive command — so an operator who was just prompted for a password
    must never be told it took effect when the file could not be written.
    """
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / AUTH_ENV_FILENAME).write_text(f"{PW_HASH_VAR_PREFIX}ALICE=established\n")
    project_root.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match="NOT changed"):
            set_auth_password("alice", "new-password", project_root)
    finally:
        project_root.chmod(0o700)

    # Fail-closed: whatever survives, it is not the credential the operator was
    # told is now in force.
    stored = read_auth_env(project_root).get(f"{PW_HASH_VAR_PREFIX}ALICE", "")
    assert not verify_password("new-password", stored) or stored == ""


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_set_auth_password_write_failure_never_names_the_password(tmp_path: Path) -> None:
    """The raise carries the path and the username — never the cleartext, which
    would otherwise reach a traceback, a log, or a bug report."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / AUTH_ENV_FILENAME).write_text(f"{PW_HASH_VAR_PREFIX}ALICE=established\n")
    project_root.chmod(0o500)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            set_auth_password("alice", "unmistakable-cleartext", project_root)
    finally:
        project_root.chmod(0o700)

    assert "unmistakable-cleartext" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# seeded_logins: the narrow case where a password may be printed
# ---------------------------------------------------------------------------


def write_seeded_repo(root: Path, declared: str, in_env: str) -> None:
    """A deployment repo whose profile declares one login and whose .env sets it."""
    (root / "profile.yml").write_text(
        f"name: demo\nenv:\n  defaults:\n    {PW_PLAINTEXT_VAR_PREFIX}ALICE: {declared}\n"
    )
    (root / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE={in_env}\n")


def test_a_password_still_matching_the_profile_default_is_named(tmp_path: Path) -> None:
    """The whole point: `osprey init` wrote this value from the profile, so
    nobody was ever told it, and both files it lives in are tracked."""
    write_seeded_repo(tmp_path, "alice", "alice")

    assert seeded_logins(tmp_path, ["alice"]) == [("alice", "alice")]


def test_a_password_the_operator_changed_is_not_named(tmp_path: Path) -> None:
    """One edit to .env and the value is theirs, not the profile's."""
    write_seeded_repo(tmp_path, "alice", "s3cret-for-the-control-room")

    assert seeded_logins(tmp_path, ["alice"]) == []


def test_whitespace_around_the_env_value_does_not_hide_the_login(tmp_path: Path) -> None:
    """`ensure_auth_credentials` trims before hashing, so a padded value IS the
    default: the two readings have to agree about what was deployed."""
    write_seeded_repo(tmp_path, "alice", "alice  ")

    assert seeded_logins(tmp_path, ["alice"]) == [("alice", "alice")]


def test_a_password_the_profile_never_declared_is_not_named(tmp_path: Path) -> None:
    """A plaintext .env password with no profile default behind it is the
    operator's own, and stays unprinted."""
    (tmp_path / "profile.yml").write_text("name: demo\nenv:\n  defaults:\n    OTHER_VAR: x\n")
    (tmp_path / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=alice\n")

    assert seeded_logins(tmp_path, ["alice"]) == []


def test_a_repo_with_no_profile_names_nothing(tmp_path: Path) -> None:
    """Advisory: nothing here is load-bearing for a deployment already running."""
    (tmp_path / ".env").write_text(f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=alice\n")

    assert seeded_logins(tmp_path, ["alice"]) == []


def test_an_unparseable_profile_names_nothing(tmp_path: Path) -> None:
    write_seeded_repo(tmp_path, "alice", "alice")
    (tmp_path / "profile.yml").write_text("env:\n  defaults:\n   - not: a mapping\n  bad: [\n")

    assert seeded_logins(tmp_path, ["alice"]) == []


def test_logins_come_back_in_roster_order(tmp_path: Path) -> None:
    (tmp_path / "profile.yml").write_text(
        "env:\n"
        "  defaults:\n"
        f"    {PW_PLAINTEXT_VAR_PREFIX}ALICE: alice\n"
        f"    {PW_PLAINTEXT_VAR_PREFIX}BOB: bob\n"
    )
    (tmp_path / ".env").write_text(
        f"{PW_PLAINTEXT_VAR_PREFIX}ALICE=alice\n{PW_PLAINTEXT_VAR_PREFIX}BOB=bob\n"
    )

    assert seeded_logins(tmp_path, ["bob", "alice"]) == [("bob", "bob"), ("alice", "alice")]
