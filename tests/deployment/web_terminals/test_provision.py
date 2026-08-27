"""Unit tests for the web-terminal deploy orchestration entrypoints.

Covers ``osprey.deployment.web_terminals.provision`` in isolation: the web
stack's own ``compose down``, the single post-``up`` force-recreate (podman
image drift), the ``preflight_web_terminals`` auth-credential provisioning
(mint order and the fail-closed gate), and the auth sidecar's image
production (registry-mode ``auth.image`` requirement, local-mode build, pull
scoping) plus the shared force-recreate primitive. The
deploy_up-entry orchestration that wires the provisioning modules together
lives in ``tests/deployment/test_container_lifecycle.py``; the split-out
provisioning steps have their own modules and test files
(``test_persona_images.py``, ``test_env_production.py``,
``test_auth_credentials.py``, ``test_postup_hooks.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.deployment.web_terminals import provision
from osprey.deployment.web_terminals.artifacts import BashLaunchTokenConflictError
from osprey.deployment.web_terminals.auth_credentials import (
    AUTH_ENV_FILENAME,
    PW_HASH_VAR_PREFIX,
    SESSION_SECRET_VARS,
    TERMINAL_SECRET_VAR_PREFIX,
    AuthCredentialsResult,
    AuthSecretsResult,
    TerminalSecretsResult,
)
from osprey.utils.dotenv import ENV_LOCAL_FILENAME, parse_dotenv_file

# The unwritable-path cases below rely on the OS honoring a read-only mode.
# root ignores it, so those assertions would be vacuous there.
running_as_root = hasattr(os, "geteuid") and os.geteuid() == 0


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# deploy_down_web_terminals -- the web stack's own `compose down`
# ---------------------------------------------------------------------------


def test_deploy_down_web_terminals_runs_compose_down_on_web_file(monkeypatch, tmp_path):
    """With a rendered build/docker-compose.web.yml, the web stack gets its own
    `compose ... down` under the pinned compose project — the mirror of
    deploy_up_web_terminals' second invocation. Without it the fixed-name
    `<prefix>-web-<user>`/`<prefix>-nginx` containers outlive every
    `osprey down` and the next web-terminals deploy on the host dies at `up`
    with a container-name Conflict."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path, "services: {}\n")

    recorded: dict = {}

    def _fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["env"] = kwargs.get("env")
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])

    provision.deploy_down_web_terminals(
        {"project_name": "myproj"}, dict(os.environ), ["--env-file", ".env"]
    )

    assert recorded["cmd"] == [
        "docker",
        "compose",
        # Global flags precede the -f list; docker only (see
        # runtime_helper.with_plain_progress).
        "--progress",
        "plain",
        # The pinned base — repo root as project directory, the rendered file
        # addressed under build/ (compose_generator.compose_base_cmd).
        "--project-directory",
        str(tmp_path),
        "-f",
        str(tmp_path / "build" / "docker-compose.web.yml"),
        "--env-file",
        ".env",
        "down",
    ]
    assert recorded["env"]["COMPOSE_PROJECT_NAME"] == "myproj"


def test_deploy_down_web_terminals_noop_without_web_file(monkeypatch, tmp_path):
    """No rendered web compose file (nothing was ever deployed from this root,
    or the render predates web terminals) → no compose invocation at all."""
    monkeypatch.chdir(tmp_path)

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    provision.deploy_down_web_terminals({"project_name": "myproj"}, dict(os.environ), [])


# ---------------------------------------------------------------------------
# _reconcile_web_stack_recreates -- the single post-`up` force-recreate
# ---------------------------------------------------------------------------

_WEB_COMPOSE = (
    "services:\n"
    "  nginx:\n"
    "    image: nginx:1.27-alpine\n"
    "    container_name: als-nginx\n"
    "  web-alice:\n"
    "    image: reg/web-terminal:latest\n"
    "    container_name: als-web-alice\n"
    "  web-bob:\n"
    "    image: reg/web-terminal-analysis:latest\n"
    "    container_name: als-web-bob\n"
)

_WEB_CMD = ["podman", "compose", "-f", "docker-compose.web.yml"]
_RUN_ENV = {"COMPOSE_PROJECT_NAME": "als"}


def _write_web_compose(tmp_path, body: str = _WEB_COMPOSE):
    """Render a web compose file where every invocation addresses it: build/."""
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    (build / "docker-compose.web.yml").write_text(body, encoding="utf-8")


def _patch_ids(monkeypatch, image_ids, container_ids):
    monkeypatch.setattr(
        provision, "get_image_id", lambda runtime, image, env=None: image_ids[image]
    )
    monkeypatch.setattr(
        provision,
        "get_container_image_id",
        lambda runtime, container, env=None: container_ids[container],
    )


def test_reconcile_force_recreates_only_drifted_services(monkeypatch, tmp_path):
    """podman + a single service whose running image ID drifted from its tag →
    exactly one `up -d --force-recreate <that service>`, delta-scoped, under the
    same web_cmd/env as the preceding `up`."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    _patch_ids(
        monkeypatch,
        image_ids={
            "nginx:1.27-alpine": "idnginx",
            "reg/web-terminal:latest": "idNEW",
            "reg/web-terminal-analysis:latest": "idbob",
        },
        container_ids={
            "als-nginx": "idnginx",
            "als-web-alice": "idOLD",  # stale: still on the pre-pull image
            "als-web-bob": "idbob",
        },
    )
    recorded = []

    def _fake_run(cmd, **kwargs):
        recorded.append((cmd, kwargs.get("env")))
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._reconcile_web_stack_recreates({}, _WEB_CMD, _RUN_ENV)

    assert len(recorded) == 1
    cmd, env = recorded[0]
    assert cmd == [
        "podman",
        "compose",
        "-f",
        "docker-compose.web.yml",
        "up",
        "-d",
        "--force-recreate",
        "web-alice",
    ]
    assert env == _RUN_ENV


def test_reconcile_noop_when_all_images_match(monkeypatch, tmp_path):
    """podman + every running image ID matches its tag → no compose command."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    ids = {
        "nginx:1.27-alpine": "idnginx",
        "reg/web-terminal:latest": "idalice",
        "reg/web-terminal-analysis:latest": "idbob",
    }
    _patch_ids(
        monkeypatch,
        image_ids=ids,
        container_ids={"als-nginx": "idnginx", "als-web-alice": "idalice", "als-web-bob": "idbob"},
    )

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    provision._reconcile_web_stack_recreates({}, _WEB_CMD, _RUN_ENV)


def test_reconcile_skipped_entirely_on_docker(monkeypatch, tmp_path):
    """docker runtime → the reconcile is skipped before any inspect or compose
    call (docker compose already recreates after a re-pull)."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])

    def _boom(*args, **kwargs):
        raise AssertionError("must not inspect or run compose on docker")

    monkeypatch.setattr(provision, "get_image_id", _boom)
    monkeypatch.setattr(provision, "get_container_image_id", _boom)
    monkeypatch.setattr(provision.subprocess, "run", _boom)

    provision._reconcile_web_stack_recreates({}, _WEB_CMD, _RUN_ENV)


def test_reconcile_skips_service_on_inspect_error_without_raising(monkeypatch, tmp_path):
    """A service whose image or container can't be inspected (None) is skipped,
    never aborting the deploy; with the rest matching, no recreate runs."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    _patch_ids(
        monkeypatch,
        image_ids={
            "nginx:1.27-alpine": "idnginx",
            "reg/web-terminal:latest": None,  # image inspect failed / never pulled
            "reg/web-terminal-analysis:latest": "idbob",
        },
        container_ids={
            "als-nginx": "idnginx",
            "als-web-alice": "idOLD",  # would differ, but image side is None → skipped
            "als-web-bob": None,  # container not created yet → skipped
        },
    )

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    # No raise, no compose invocation.
    provision._reconcile_web_stack_recreates({}, _WEB_CMD, _RUN_ENV)


def test_reconcile_skipped_when_compose_file_unreadable(monkeypatch, tmp_path):
    """podman but no rendered docker-compose.web.yml at the root → advisory skip,
    no inspect, no compose command, no raise."""
    monkeypatch.chdir(tmp_path)  # no compose file written
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])

    def _boom(*args, **kwargs):
        raise AssertionError("must not inspect or run compose when the file is unreadable")

    monkeypatch.setattr(provision, "get_image_id", _boom)
    monkeypatch.setattr(provision.subprocess, "run", _boom)

    provision._reconcile_web_stack_recreates({}, _WEB_CMD, _RUN_ENV)


# ---------------------------------------------------------------------------
# preflight_web_terminals -- auth credential provisioning and the gate
# ---------------------------------------------------------------------------


def _auth_config(method: str, users=("alice", "bob"), auth_image="reg/osprey-auth:1") -> dict:
    """A registry-mode web-terminals config with `method` authentication.

    Carries an `auth.image` by default because registry mode requires one (see
    the sidecar-image section below); pass None to exercise that requirement.
    """
    auth: dict = {"method": method}
    if auth_image is not None:
        auth["image"] = auth_image
    return {
        "facility": {"prefix": "als"},
        "modules": {
            "web_terminals": {
                "users": list(users),
                "auth": auth,
            }
        },
    }


def _run_preflight(monkeypatch, project_root: Path, config: dict):
    """Run the preflight from `project_root` with the non-auth steps neutered.

    ``ensure_env_production`` has its own fail-closed gate (and would raise on
    a registry-mode root with no .env.production); it is covered by
    test_env_production.py, so stub it out to keep these assertions about auth.
    """
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(provision, "ensure_env_production", lambda config, root: None)
    return provision.preflight_web_terminals(config)


def _credentials_result(
    path: Path, *, changed=False, missing=(), users=()
) -> AuthCredentialsResult:
    return AuthCredentialsResult(
        env_auth_path=path,
        changed=changed,
        minted=(),
        hashed_from_plaintext=(),
        preexisting=tuple(users),
        missing=tuple(missing),
    )


def _secrets_result(path: Path, *, changed=False, missing=()) -> AuthSecretsResult:
    return AuthSecretsResult(
        env_auth_path=path,
        changed=changed,
        minted=(),
        preexisting=SESSION_SECRET_VARS,
        missing=tuple(missing),
    )


def _stub_clean_provisioning(monkeypatch, tmp_path):
    """Both provisioning calls succeed, change nothing, and leave nothing missing."""
    env_auth = tmp_path / AUTH_ENV_FILENAME
    monkeypatch.setattr(
        provision,
        "ensure_auth_credentials",
        lambda usernames, root, **kwargs: _credentials_result(env_auth),
    )
    monkeypatch.setattr(
        provision, "ensure_auth_session_secrets", lambda root: _secrets_result(env_auth)
    )


@pytest.mark.parametrize("image_source", ["local", "registry"])
def test_preflight_mints_credentials_then_secrets_for_the_roster(
    monkeypatch, tmp_path, image_source
):
    """Order is load-bearing and explicit: the roster's password hashes first,
    then the signing secrets, both keyed off the project root and both handed
    the plain roster names (never the normalized env-var suffixes).

    Run in BOTH image-source modes: the sidecar needs its credentials whatever
    the images come from, and only registry mode was exercised until the
    parametrize was added."""
    calls: list[tuple] = []
    env_auth = tmp_path / AUTH_ENV_FILENAME

    def _fake_credentials(usernames, project_root, **kwargs):
        calls.append(("credentials", list(usernames), str(project_root)))
        return _credentials_result(env_auth, users=usernames)

    def _fake_secrets(project_root):
        calls.append(("secrets", str(project_root)))
        return _secrets_result(env_auth)

    monkeypatch.setattr(provision, "ensure_auth_credentials", _fake_credentials)
    monkeypatch.setattr(provision, "ensure_auth_session_secrets", _fake_secrets)
    # Local mode resolves personas and verifies their renders; neither is this
    # test's subject, and local mode needs no auth.image.
    monkeypatch.setattr(provision, "resolve_personas", lambda *a, **kw: [])
    monkeypatch.setattr(provision, "verify_persona_renders", lambda *a, **kw: None)

    config = _auth_config("password", users=["alice", {"name": "bob-2", "index": 4}])
    config["modules"]["web_terminals"]["image_source"] = image_source
    _run_preflight(monkeypatch, tmp_path, config)

    assert [call[0] for call in calls] == ["credentials", "secrets"]
    assert calls[0][1] == ["alice", "bob-2"]
    assert calls[0][2] == str(tmp_path)
    assert calls[1][1] == str(tmp_path)


def test_preflight_mints_no_credential_for_a_login_false_entry(monkeypatch, tmp_path):
    """A roster entry with `login: false` is left out of the password mint: no
    gate ever asks the sidecar about it, so a hash for it would be a credential
    nothing checks — and a minted password printed for it would tell the
    operator the opposite of the truth."""
    calls: list[list[str]] = []
    env_auth = tmp_path / AUTH_ENV_FILENAME

    def _fake_credentials(usernames, project_root, **kwargs):
        calls.append(list(usernames))
        return _credentials_result(env_auth, users=usernames)

    monkeypatch.setattr(provision, "ensure_auth_credentials", _fake_credentials)
    monkeypatch.setattr(
        provision, "ensure_auth_session_secrets", lambda root: _secrets_result(env_auth)
    )

    config = _auth_config(
        "password",
        users=[
            "alice",
            {"name": "ariel", "index": 1, "login": False},
            {"name": "bob", "index": 2, "login": True},
        ],
    )
    _run_preflight(monkeypatch, tmp_path, config)

    assert calls == [["alice", "bob"]]


# ---------------------------------------------------------------------------
# preflight_web_terminals -- per-user terminal secrets. Provisioned for EVERY
# deployment, outside the `auth.method: none` early return that governs the
# login credentials: a terminal secret is not a login, it is what lets a
# terminal refuse everything that did not arrive through nginx.
# ---------------------------------------------------------------------------


def _terminal_result(path: Path, *, missing=(), minted=()) -> TerminalSecretsResult:
    return TerminalSecretsResult(
        env_path=path,
        changed=bool(minted),
        minted=tuple(minted),
        preexisting=(),
        missing=tuple(missing),
    )


def test_preflight_mints_terminal_secrets_when_auth_is_off(monkeypatch, tmp_path):
    """The CF-4 invariant, at the seam: the mint sits OUTSIDE
    `_provision_auth_secrets`, so `auth.method: none` — the default, and the
    shape with nothing else between one user's browser and another user's
    terminal — still gets a per-user secret."""
    _run_preflight(monkeypatch, tmp_path, _auth_config("none"))

    stored = parse_dotenv_file(tmp_path / ENV_LOCAL_FILENAME)
    assert stored[f"{TERMINAL_SECRET_VAR_PREFIX}ALICE"]
    assert stored[f"{TERMINAL_SECRET_VAR_PREFIX}BOB"]
    assert (
        stored[f"{TERMINAL_SECRET_VAR_PREFIX}ALICE"] != stored[f"{TERMINAL_SECRET_VAR_PREFIX}BOB"]
    )


def test_an_auth_off_deploy_refuses_a_bad_roster_name_without_naming_auth(monkeypatch, tmp_path):
    """The mint's charset gate now reaches every deployment, so its refusal has
    to make sense to an operator who configured no authentication.

    Running the mint outside the `auth.method: none` early return means
    `USERNAME_CHARSET_RE` is enforced on rosters that were previously only
    linted — a real tightening, and one whose message used to send an auth-off
    operator looking for auth credentials they never configured.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _run_preflight(monkeypatch, tmp_path, _auth_config("none", users=("Alice",)))

    message = str(excinfo.value)
    assert "web-terminal secrets" in message
    assert "auth credentials" not in message
    assert "'Alice'" in message
    # Nothing was written for a deploy that is being refused.
    assert not (tmp_path / ENV_LOCAL_FILENAME).exists()


def test_preflight_provisions_a_terminal_secret_for_a_login_false_entry(monkeypatch, tmp_path):
    """The opposite of the password mint's rule, and deliberately so: opting out
    of the login wall does not opt a terminal out of needing a front door, so
    every roster entry is passed to the terminal mint."""
    calls: list[list[str]] = []

    def _fake_terminal(project_root, usernames):
        calls.append(list(usernames))
        return _terminal_result(Path(project_root) / ENV_LOCAL_FILENAME)

    monkeypatch.setattr(provision, "ensure_terminal_secrets", _fake_terminal)
    monkeypatch.setattr(
        provision, "ensure_auth_credentials", lambda users, root, **kw: _credentials_result(root)
    )
    monkeypatch.setattr(
        provision,
        "ensure_auth_session_secrets",
        lambda root: _secrets_result(Path(root) / AUTH_ENV_FILENAME),
    )

    config = _auth_config(
        "password",
        users=[
            "alice",
            {"name": "ariel", "index": 1, "login": False},
            {"name": "bob", "index": 2, "login": True},
        ],
    )
    _run_preflight(monkeypatch, tmp_path, config)

    assert calls == [["alice", "ariel", "bob"]]


def test_preflight_aborts_naming_the_variable_when_a_secret_cannot_be_established(
    monkeypatch, tmp_path
):
    """The fail-closed gate. A roster user with no usable secret would get a
    terminal that refuses every request — including nginx's — so the deploy must
    stop here, and must name the variable an operator can set by hand."""
    var = f"{TERMINAL_SECRET_VAR_PREFIX}ALICE"
    monkeypatch.setattr(
        provision,
        "ensure_terminal_secrets",
        lambda root, users: _terminal_result(Path(root) / ENV_LOCAL_FILENAME, missing=(var,)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _run_preflight(monkeypatch, tmp_path, _auth_config("none"))

    message = str(excinfo.value)
    assert var in message
    assert str(tmp_path / ENV_LOCAL_FILENAME) in message


def test_the_terminal_mint_runs_after_the_auth_credential_mint(monkeypatch, tmp_path):
    """Ordering, pinned: the credential mint PRINTS a password once, and a
    refusal from the terminal gate arriving first would abort a deploy whose
    password the operator was never shown."""
    order: list[str] = []
    env_auth = tmp_path / AUTH_ENV_FILENAME

    def _fake_credentials(usernames, project_root, **kwargs):
        order.append("credentials")
        return _credentials_result(env_auth, users=usernames)

    def _fake_secrets(project_root):
        order.append("secrets")
        return _secrets_result(env_auth)

    def _fake_terminal(project_root, usernames):
        order.append("terminal")
        return _terminal_result(Path(project_root) / ENV_LOCAL_FILENAME)

    monkeypatch.setattr(provision, "ensure_auth_credentials", _fake_credentials)
    monkeypatch.setattr(provision, "ensure_auth_session_secrets", _fake_secrets)
    monkeypatch.setattr(provision, "ensure_terminal_secrets", _fake_terminal)

    _run_preflight(monkeypatch, tmp_path, _auth_config("password"))

    assert order == ["credentials", "secrets", "terminal"]


def test_the_terminal_mint_is_idempotent_across_deploys(monkeypatch, tmp_path):
    """A redeploy must not rotate a secret: nginx and the running terminal were
    both created holding the current value, so a fresh one locks every user out
    until the whole web stack is recreated."""
    _run_preflight(monkeypatch, tmp_path, _auth_config("none"))
    before = (tmp_path / ENV_LOCAL_FILENAME).read_text(encoding="utf-8")

    _run_preflight(monkeypatch, tmp_path, _auth_config("none"))

    assert (tmp_path / ENV_LOCAL_FILENAME).read_text(encoding="utf-8") == before


def test_preflight_with_auth_none_touches_no_AUTH_credential_state(monkeypatch, tmp_path, caplog):
    """`auth.method: none` (the default) mints no LOGIN credential: no hash, no
    signing secret, no .env.auth, and no gitignore warning about it — even from
    a project root whose .gitignore does not cover the file.

    Terminal secrets are the deliberate exception and are asserted here rather
    than merely tolerated: they are not a login, they are what makes nginx the
    only route to a terminal, and an auth-off multi-user deployment is exactly
    the one with nothing else in the way. They live in the deploy `.env`, which
    is why `.env.auth` still must not come into being."""

    def _unexpected(*args, **kwargs):
        raise AssertionError("auth provisioning must not run with auth.method: none")

    monkeypatch.setattr(provision, "ensure_auth_credentials", _unexpected)
    monkeypatch.setattr(provision, "ensure_auth_session_secrets", _unexpected)
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        _run_preflight(monkeypatch, tmp_path, _auth_config("none"))

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()
    assert AUTH_ENV_FILENAME not in caplog.text
    stored = parse_dotenv_file(tmp_path / ENV_LOCAL_FILENAME)
    assert stored[f"{TERMINAL_SECRET_VAR_PREFIX}ALICE"]
    assert stored[f"{TERMINAL_SECRET_VAR_PREFIX}BOB"]


def test_preflight_provisions_on_first_run_then_is_idempotent(monkeypatch, tmp_path):
    """End to end on a real project root: the first deploy establishes every
    hash and both signing secrets; the second changes nothing — byte-identical
    ``.env.auth``, so the re-rendered digest label is unchanged too and a no-op
    redeploy recreates nothing."""
    config = _auth_config("password", users=("alice", "bob-2"))

    _run_preflight(monkeypatch, tmp_path, config)

    env_auth = tmp_path / AUTH_ENV_FILENAME
    stored = parse_dotenv_file(env_auth)
    assert stored[f"{PW_HASH_VAR_PREFIX}ALICE"]
    assert stored[f"{PW_HASH_VAR_PREFIX}BOB_2"]
    assert all(stored[var] for var in SESSION_SECRET_VARS)

    before = env_auth.read_text(encoding="utf-8")
    _run_preflight(monkeypatch, tmp_path, config)

    assert env_auth.read_text(encoding="utf-8") == before


def test_preflight_in_oidc_mode_creates_env_auth_with_secrets_and_no_hashes(monkeypatch, tmp_path):
    """oidc mode still has to WRITE .env.auth: the sidecar's compose service
    declares `env_file: .env.auth`, so `compose up` hard-fails outright when the
    file is absent — an oidc stack could not start at all. What it must not
    contain is password hashes: oidc authenticates at the IdP, and minting
    passwords nobody will ever type would put credentials on disk for nothing."""
    _run_preflight(monkeypatch, tmp_path, _auth_config("oidc", users=("alice", "bob")))

    env_auth = tmp_path / AUTH_ENV_FILENAME
    assert env_auth.is_file()
    stored = parse_dotenv_file(env_auth)
    assert all(stored[var] for var in SESSION_SECRET_VARS)
    assert not [var for var in stored if var.startswith(PW_HASH_VAR_PREFIX)]


def test_preflight_does_not_swallow_a_roster_that_cannot_be_keyed(monkeypatch, tmp_path):
    """`ensure_auth_credentials` raises (writing nothing) when two usernames
    normalize onto one credential variable — they would share a password, so one
    operator's credentials would open the other's terminal. Deploy never runs
    lint, so that raise IS the abort: the preflight must let it propagate."""
    config = _auth_config("password", users=("alice-b", "alice_b"))

    with pytest.raises(RuntimeError, match="one credential variable"):
        _run_preflight(monkeypatch, tmp_path, config)

    assert not (tmp_path / AUTH_ENV_FILENAME).exists()


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_preflight_gate_raises_naming_every_user_left_without_a_hash(monkeypatch, tmp_path):
    """The post-mint invariant: with .env.auth unwritable no password hash can
    be established, so the deploy aborts HERE — before any compose invocation —
    naming each password-mode roster user, rather than bringing up terminals
    nobody can log into."""
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text("# pre-existing\n", encoding="utf-8")
    os.chmod(env_auth, 0o400)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _run_preflight(monkeypatch, tmp_path, _auth_config("password", users=("alice", "bob")))
    finally:
        os.chmod(env_auth, 0o600)

    message = str(excinfo.value)
    assert "alice" in message
    assert "bob" in message
    assert AUTH_ENV_FILENAME in message
    assert "auth.method: password" in message


@pytest.mark.skipif(running_as_root, reason="root ignores the read-only mode this test relies on")
def test_preflight_gate_raises_naming_missing_secret_vars_in_oidc_mode(monkeypatch, tmp_path):
    """A signing secret is required in EVERY method, so an unwritable .env.auth
    aborts an oidc deploy too, naming the variables. No user is named: oidc
    provisions no password hash in the first place, and refusing over one would
    block a deployment that would have worked."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
    os.chmod(project_root, 0o500)  # nothing can be created inside it
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _run_preflight(monkeypatch, project_root, _auth_config("oidc", users=("alice",)))
    finally:
        os.chmod(project_root, 0o700)

    message = str(excinfo.value)
    for var in SESSION_SECRET_VARS:
        assert var in message
    assert "alice" not in message


@pytest.mark.parametrize(
    "gitignore",
    [
        "# nothing\n",
        ".env\n.env.production\n",
        "!.env.auth\n",
        # git resolves a path against the LAST matching pattern, so this file
        # is TRACKED despite the earlier glob — the warning must still fire.
        ".env*\n!.env.auth\n",
    ],
)
def test_preflight_warns_when_gitignore_does_not_cover_env_auth(
    monkeypatch, tmp_path, caplog, gitignore
):
    """Projects scaffolded before auth existed ignore .env but not .env.auth —
    and gitignore matches literal names, so .env does not cover it. Warn (never
    block) that a file about to hold hashes and signing secrets is committable."""
    _stub_clean_provisioning(monkeypatch, tmp_path)
    (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")

    with caplog.at_level("WARNING"):
        _run_preflight(monkeypatch, tmp_path, _auth_config("password"))  # warn-only: no raise

    assert AUTH_ENV_FILENAME in caplog.text
    assert ".gitignore" in caplog.text


@pytest.mark.parametrize(
    "gitignore",
    [".env.auth\n", "/.env.auth\n", ".env*\n", "!.env.auth\n.env.auth\n"],
)
def test_preflight_does_not_warn_when_gitignore_covers_env_auth(
    monkeypatch, tmp_path, caplog, gitignore
):
    """A literal entry, a root-anchored one, or a glob that matches it all count
    as covered — the warning must not become background noise."""
    _stub_clean_provisioning(monkeypatch, tmp_path)
    (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")

    with caplog.at_level("WARNING"):
        _run_preflight(monkeypatch, tmp_path, _auth_config("password"))

    assert AUTH_ENV_FILENAME not in caplog.text


def test_preflight_does_not_warn_without_a_gitignore_at_all(monkeypatch, tmp_path, caplog):
    """No .gitignore is ordinarily a root nobody commits from; warning there
    would fire on every deploy about a risk that does not exist."""
    _stub_clean_provisioning(monkeypatch, tmp_path)

    with caplog.at_level("WARNING"):
        _run_preflight(monkeypatch, tmp_path, _auth_config("password"))

    assert AUTH_ENV_FILENAME not in caplog.text


# ---------------------------------------------------------------------------
# Auth sidecar image: the registry-mode requirement and the local-mode build
# ---------------------------------------------------------------------------


def _sidecar_config(image_source: str, *, auth_image=None, method="password") -> dict:
    """A config in `image_source` mode, with auth.image only when given."""
    config = _auth_config(method, users=("alice",), auth_image=auth_image)
    config["modules"]["web_terminals"]["image_source"] = image_source
    return config


def test_registry_mode_without_auth_image_fails_preflight(monkeypatch, tmp_path):
    """Registry mode builds nothing locally, and the compose overlay falls back
    to a `:local` tag that mode never produces — so a deployment that forgot
    auth.image must abort here, naming the key, instead of dying at `pull` on a
    tag no registry has."""
    _stub_clean_provisioning(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match="auth.image"):
        _run_preflight(monkeypatch, tmp_path, _sidecar_config("registry"))


def test_registry_mode_with_auth_image_passes_preflight(monkeypatch, tmp_path):
    _stub_clean_provisioning(monkeypatch, tmp_path)

    _run_preflight(
        monkeypatch, tmp_path, _sidecar_config("registry", auth_image="reg/osprey-auth:1.2.3")
    )


def test_local_mode_needs_no_auth_image_at_preflight(monkeypatch, tmp_path):
    """Local mode produces the tag itself, so the key is optional there."""
    _stub_clean_provisioning(monkeypatch, tmp_path)
    monkeypatch.setattr(provision, "verify_persona_renders", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "resolve_personas", lambda *a, **kw: [])

    _run_preflight(monkeypatch, tmp_path, _sidecar_config("local"))


def test_auth_method_none_needs_no_auth_image(monkeypatch, tmp_path):
    """With authentication off no sidecar service is rendered at all, so a
    registry deploy that never opted in must not be asked for an image."""
    _stub_clean_provisioning(monkeypatch, tmp_path)

    _run_preflight(monkeypatch, tmp_path, _sidecar_config("registry", method="none"))


def _capture_build(monkeypatch):
    recorded: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        return _FakeCompletedProcess()

    # Stubbed at `run_captured`, not at `subprocess.run`: the sidecar build is
    # watched (`on_line=`), and a watched capture reads its child through a pipe
    # rather than writing it straight to the spool — so a `subprocess.run` stub
    # would be walked straight past and this test would run a real build.
    monkeypatch.setattr(provision, "run_captured", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    return recorded


def test_local_mode_builds_the_auth_image_the_compose_overlay_references(monkeypatch, tmp_path):
    """The tag, the context and the ownership label in one place: compose
    declares the sidecar with an `image:` and no `build:` block, so this is its
    only producer — and the tag must be exactly what the overlay renders."""
    monkeypatch.chdir(tmp_path)
    recorded = _capture_build(monkeypatch)

    provision.build_auth_sidecar_image(_sidecar_config("local"), False, {})

    assert len(recorded) == 1
    cmd = recorded[0]
    context_dir = tmp_path / provision.AUTH_BUILD_CONTEXT
    assert cmd[:4] == ["podman", "build", "-t", "als-assistant-auth:local"]
    assert cmd[-1] == str(context_dir)
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == str(context_dir / "Dockerfile")
    # OSPREY_PROJECT_NAME is what stamps com.osprey.project on the image, the
    # ownership label `nuke` verifies before removing a tag.
    assert any(arg.startswith("OSPREY_PROJECT_NAME=") and arg.split("=", 1)[1] for arg in cmd)
    assert any(arg.startswith("OSPREY_VERSION=") and arg.split("=", 1)[1] for arg in cmd)
    assert "OSPREY_DEV=1" not in cmd
    # The context is materialized from the bundled template package, including
    # the .dockerignore the Dockerfile COPYs as its guaranteed glob sibling.
    assert (context_dir / "Dockerfile").is_file()
    assert (context_dir / ".dockerignore").is_file()


def test_local_mode_dev_build_of_the_auth_image_passes_the_dev_build_arg(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    recorded = _capture_build(monkeypatch)
    monkeypatch.setattr(provision, "_stage_dev_wheel_for_context", lambda out_dir, dev: True)

    provision.build_auth_sidecar_image(_sidecar_config("local"), True, {})

    assert "OSPREY_DEV=1" in recorded[0]


def test_local_mode_with_auth_image_builds_nothing(monkeypatch, tmp_path):
    """auth.image pins an externally supplied image; building over that pin
    would produce a tag nothing references."""
    monkeypatch.chdir(tmp_path)
    recorded = _capture_build(monkeypatch)

    provision.build_auth_sidecar_image(_sidecar_config("local", auth_image="reg/auth:1"), False, {})

    assert recorded == []


def test_registry_mode_builds_no_auth_image_locally(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    recorded = _capture_build(monkeypatch)

    provision.build_auth_sidecar_image(
        _sidecar_config("registry", auth_image="reg/auth:1"), False, {}
    )

    assert recorded == []


def test_auth_method_none_builds_no_auth_image_locally(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    recorded = _capture_build(monkeypatch)

    provision.build_auth_sidecar_image(_sidecar_config("local", method="none"), False, {})

    assert recorded == []


# ---------------------------------------------------------------------------
# Web-stack invocations: pull scoping and the force-recreate primitive
# ---------------------------------------------------------------------------


def _stub_web_stack(monkeypatch, tmp_path):
    """Neuter every collaborator of deploy_up_web_terminals and capture argv."""
    monkeypatch.chdir(tmp_path)
    recorded: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])
    monkeypatch.setattr(provision, "runtime_env", lambda config, env, **kw: dict(env))
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda config, dest_dir=".": [])
    monkeypatch.setattr(provision, "ensure_env_production", lambda config, root: None)
    monkeypatch.setattr(provision, "resolve_personas", lambda *a, **kw: [])
    monkeypatch.setattr(provision, "verify_persona_renders", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "build_persona_images", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "build_auth_sidecar_image", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "_reconcile_web_stack_recreates", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "reload_nginx_config", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "enable_linger", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "seed_user_containers", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "run_verify_script", lambda *a, **kw: None)
    monkeypatch.setattr(provision, "warn_if_web_stack_unreachable", lambda *a, **kw: None)
    return recorded


def test_local_mode_web_stack_never_pulls_the_local_auth_image(monkeypatch, tmp_path):
    """`compose pull` hard-fails on a tag no registry can serve, and the
    sidecar's local tag is exactly that — so local mode's web stack must issue
    no pull at all, which is what keeps the locally built image out of it."""
    recorded = _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(_sidecar_config("local"), [], False, {}, [])

    assert not any("pull" in cmd for cmd in recorded)
    assert any(cmd[-2:] == ["up", "-d"] for cmd in recorded)


def test_registry_mode_web_stack_still_pulls(monkeypatch, tmp_path):
    """Registry mode's sidecar image is a published one (preflight required
    auth.image), so it belongs in the pull like every other web-stack image."""
    recorded = _stub_web_stack(monkeypatch, tmp_path)

    provision.deploy_up_web_terminals(
        _sidecar_config("registry", auth_image="reg/auth:1"), [], False, {}, []
    )

    assert any("pull" in cmd for cmd in recorded)


def test_local_mode_builds_the_auth_image_before_any_compose_invocation(monkeypatch, tmp_path):
    """Ordering is the point: compose `up` on an unbuilt local tag fails with an
    opaque 'no such image'."""
    order: list[str] = []
    recorded = _stub_web_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(
        provision, "build_auth_sidecar_image", lambda *a, **kw: order.append("build")
    )

    def _fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        order.append("compose")
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision.deploy_up_web_terminals(_sidecar_config("local"), [], False, {}, [])

    assert order[0] == "build"


def test_force_recreate_auth_sidecar_targets_only_the_sidecar_service(monkeypatch, tmp_path):
    """env_file content is baked in at container creation, so only a recreate
    puts a changed .env.auth in force — scoped to the `auth` service, because
    recreating the whole stack would bounce every live terminal."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path, "services: {}\n")
    recorded: list[dict] = []

    def _fake_run(cmd, **kwargs):
        recorded.append({"cmd": list(cmd), "env": kwargs.get("env")})
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    # The pre-recreate render is pinned by its own tests below; this one is
    # about the compose argv, so keep the stub config renderable-free.
    monkeypatch.setattr(provision, "write_web_terminal_artifacts", lambda *a, **k: [])

    provision.force_recreate_auth_sidecar(
        {"project_name": "myproj"}, ["--env-file", ".env"], env={"X": "1"}
    )

    assert len(recorded) == 1
    assert recorded[0]["cmd"] == [
        "podman",
        "compose",
        # The pinned base, identical to the `up` that created the stack.
        "--project-directory",
        str(tmp_path),
        "-f",
        str(tmp_path / "build" / "docker-compose.web.yml"),
        "--env-file",
        ".env",
        "up",
        "-d",
        "--force-recreate",
        "auth",
    ]
    # Same COMPOSE_PROJECT_NAME pin as the `up` that created the stack —
    # without it compose would address a different project entirely.
    assert recorded[0]["env"]["COMPOSE_PROJECT_NAME"] == "myproj"


def test_force_recreate_auth_sidecar_rerenders_before_the_recreate(monkeypatch, tmp_path):
    """Every caller of this primitive (passwd/decommission/prune) mutates
    .env.auth AFTER the last render, so the compose file's digest label
    describes the file's previous content. The recreate must run against a
    freshly-digested compose file, or the created container carries a label
    that is not the digest of the env it baked — costing a spurious bounce at
    the next deploy, and masking a byte-exact revert from the label diff."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path, "services: {}\n")
    order: list[str] = []
    monkeypatch.setattr(
        provision,
        "write_web_terminal_artifacts",
        lambda config, dest_dir=".": order.append("render") or [],
    )

    def _fake_run(cmd, **kwargs):
        order.append("recreate")
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])

    provision.force_recreate_auth_sidecar({"project_name": "myproj"}, [])

    assert order == ["render", "recreate"]


def test_force_recreate_auth_sidecar_still_recreates_when_the_render_fails(
    monkeypatch, tmp_path, caplog
):
    """The render is a label-hygiene step; the recreate is the security step
    (it puts a credential purge into force). A render failure must degrade to
    a warning + stale label, never to a skipped recreate."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path, "services: {}\n")
    recreated: list[list[str]] = []

    def _failing_render(config, dest_dir="."):
        raise ValueError("unrenderable config")

    monkeypatch.setattr(provision, "write_web_terminal_artifacts", _failing_render)

    def _fake_run(cmd, **kwargs):
        recreated.append(list(cmd))
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])

    with caplog.at_level("WARNING"):
        provision.force_recreate_auth_sidecar({"project_name": "myproj"}, [])

    assert len(recreated) == 1 and "--force-recreate" in recreated[0]
    assert "Could not re-render" in caplog.text


def test_deploy_up_renders_the_artifacts_before_any_web_stack_compose_invocation(
    monkeypatch, tmp_path
):
    """Ordering is what makes the digest label trustworthy: the render digests
    `.env.auth` into the sidecar's service definition, so it must run before
    the web stack's `up -d` — that `up` is what compares definitions and
    recreates the sidecar after any content change (mint or hand-edit)."""
    order: list[str] = []
    _stub_web_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(
        provision,
        "write_web_terminal_artifacts",
        lambda config, dest_dir=".": order.append("render") or [],
    )

    def _fake_run(cmd, **kwargs):
        order.append("compose")
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision.deploy_up_web_terminals(
        _sidecar_config("registry", auth_image="reg/auth:1"), [], False, {}, []
    )

    assert order[0] == "render"
    assert "compose" in order


def test_force_recreate_auth_sidecar_is_a_warning_not_a_failure_without_a_stack(
    monkeypatch, tmp_path, caplog
):
    """Nothing was ever deployed from this root: there is no container to
    recreate, and raising would turn that into a deploy error."""
    monkeypatch.chdir(tmp_path)

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    with caplog.at_level("WARNING"):
        provision.force_recreate_auth_sidecar({"project_name": "myproj"}, [])

    assert "docker-compose.web.yml" in caplog.text


# ---------------------------------------------------------------------------
# preflight_web_terminals -- the Bash/launch-token conflict gate
#
# The guard also lives at the render seam (tests/.../test_artifacts.py), which is
# what covers the lifecycle re-render paths that never reach a preflight. What
# only THIS placement can give is timing: the refusal has to land before
# `ensure_env_production`, or a doomed deploy has already paid for the project
# image build and minted -- and printed -- credentials for a stack that will
# never come up. A regression that dropped this call would leave the render-seam
# tests green, so it needs its own assertion.
# ---------------------------------------------------------------------------


def _persona_project(root: Path, name: str, *, writes: bool, denies_bash: bool) -> str:
    """Write a persona project under *root*; return its relative project_path.

    Every project here runs the bluesky server, spelled out rather than left to
    a default: the server is opt-in in the registry, so a project that omits the
    key runs no server and can never hold the launch token. `writes` alone is
    what moves a project across the tier boundary.
    """
    import json

    import yaml

    project_dir = root / "profiles" / name
    (project_dir / ".claude").mkdir(parents=True)
    (project_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "project_name": name,
                "control_system": {"writes_enabled": writes},
                "claude_code": {"servers": {"bluesky": {"enabled": True}}},
            }
        ),
        encoding="utf-8",
    )
    deny = ["Bash", "Edit"] if denies_bash else ["Edit"]
    (project_dir / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"deny": deny}}), encoding="utf-8"
    )
    return f"profiles/{name}"


def _persona_roster_config(root: Path, *, denies_bash: bool) -> dict:
    """A registry-mode roster whose single persona both enables writes and runs bluesky."""
    return {
        "facility": {"prefix": "als"},
        "modules": {
            "web_terminals": {
                "users": [{"name": "alice", "index": 0, "persona": "readwrite"}],
                "auth": {"method": "none"},
                "personas": {
                    "readwrite": {
                        "project": "rw",
                        "project_path": _persona_project(
                            root, "rw", writes=True, denies_bash=denies_bash
                        ),
                    }
                },
            }
        },
    }


def test_preflight_refuses_a_bash_permitting_launch_token_persona(monkeypatch, tmp_path):
    """The fail-fast gate: an entitled persona shipping no shell deny stops the
    deploy in seconds, naming the persona.

    `ensure_env_production` is deliberately NOT stubbed here. It raises on this
    root too, so demanding specifically a `BashLaunchTokenConflictError` proves
    the conflict is what the operator is told about — a guard moved below it
    would surface the wrong error and fail this test.
    """
    monkeypatch.chdir(tmp_path)
    config = _persona_roster_config(tmp_path, denies_bash=False)

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        provision.preflight_web_terminals(config)

    assert "readwrite" in str(excinfo.value)


def test_preflight_refuses_before_any_credential_is_minted(monkeypatch, tmp_path):
    """THE reason this placement exists. `preflight_web_terminals`' docstring
    commits to not writing (and printing) passwords for a stack that never comes
    up; the conflict refusal has to sit ahead of `ensure_env_production` to honour
    that, not merely somewhere in the function."""
    monkeypatch.chdir(tmp_path)
    reached: list[str] = []
    monkeypatch.setattr(
        provision, "ensure_env_production", lambda config, root: reached.append("env_production")
    )
    monkeypatch.setattr(
        provision, "_provision_auth_secrets", lambda wt, root: reached.append("auth_secrets")
    )

    with pytest.raises(BashLaunchTokenConflictError):
        provision.preflight_web_terminals(_persona_roster_config(tmp_path, denies_bash=False))

    assert reached == []


def test_preflight_passes_a_persona_that_denies_bash(monkeypatch, tmp_path):
    """The negative control: the same entitled persona shipping the shell deny
    clears the gate, so the guard cannot be passing by refusing everything."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provision, "ensure_env_production", lambda config, root: None)

    provision.preflight_web_terminals(_persona_roster_config(tmp_path, denies_bash=True))


def test_preflight_with_dangerously_allow_bash_proceeds_and_says_so(monkeypatch, tmp_path, caplog):
    """The waiver must never be silent: every `osprey up` that runs under it
    prints a warning naming the key and the personas it waved through, at the
    exact point the refusal would otherwise have fired."""
    import logging

    monkeypatch.chdir(tmp_path)
    reached: list[str] = []
    monkeypatch.setattr(
        provision, "ensure_env_production", lambda config, root: reached.append("env_production")
    )
    monkeypatch.setattr(
        provision, "_provision_auth_secrets", lambda wt, root: reached.append("auth_secrets")
    )
    monkeypatch.setattr(
        provision, "_provision_terminal_secrets", lambda wt, root: reached.append("terminal")
    )
    config = _persona_roster_config(tmp_path, denies_bash=False)
    config["dangerously_allow_bash"] = True

    with caplog.at_level(logging.WARNING):
        provision.preflight_web_terminals(config)

    assert reached == ["env_production", "auth_secrets", "terminal"]
    banner = [r.getMessage() for r in caplog.records if "dangerously_allow_bash" in r.getMessage()]
    assert banner, caplog.text
    assert "readwrite" in banner[0]


def test_preflight_stays_quiet_when_dangerously_allow_bash_waives_nothing(
    monkeypatch, tmp_path, caplog
):
    """The key on a conflict-free roster is inert and the banner is not printed --
    a banner with nobody named would train operators to ignore it."""
    import logging

    monkeypatch.chdir(tmp_path)
    for name in ("ensure_env_production", "_provision_auth_secrets", "_provision_terminal_secrets"):
        monkeypatch.setattr(provision, name, lambda *a, **k: None)
    config = _persona_roster_config(tmp_path, denies_bash=True)
    config["dangerously_allow_bash"] = True

    with caplog.at_level(logging.WARNING):
        provision.preflight_web_terminals(config)

    assert "dangerously_allow_bash" not in caplog.text
