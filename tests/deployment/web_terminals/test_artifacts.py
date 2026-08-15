"""Tests for the render-and-write seam shared by bring-up and the lifecycle verbs."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from osprey.deployment.web_terminals.artifacts import (
    BashLaunchTokenConflictError,
    auth_env_digest,
    write_web_terminal_artifacts,
)
from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.render import AUTH_ENV_DIGEST_LABEL


def _config(users):
    return {
        "facility": {"prefix": "als", "name": "ALS"},
        "registry": {"url": "registry.example.org"},
        "deploy": {"fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 8080,
                "web_base_port": 9000,
                "artifact_base_port": 9100,
                "ariel_base_port": 9200,
                "lattice_base_port": 9300,
                "users": users,
            }
        },
    }


def test_write_web_terminal_artifacts_writes_three_files_under_the_build_zone(tmp_path):
    written = write_web_terminal_artifacts(_config(["alice", "bob"]), tmp_path)

    names = {p.relative_to(tmp_path / "build").as_posix() for p in written}
    assert names == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }
    for path in written:
        assert path.is_file()
        assert path.read_text(encoding="utf-8")  # non-empty


def test_write_web_terminal_artifacts_creates_nginx_parent_dir(tmp_path):
    write_web_terminal_artifacts(_config(["alice"]), tmp_path)
    assert (tmp_path / "build" / "nginx").is_dir()
    assert (tmp_path / "build" / "nginx" / "nginx.conf").is_file()


def test_write_web_terminal_artifacts_defaults_to_the_repos_build_zone(tmp_path, monkeypatch):
    """With no destination given, the artifacts land in the repo's build/ zone.

    Never at the repo root: they are render output, and a compose file or an
    nginx/ tree at the root would be untracked clutter in the source zone that
    the next `rm -rf build/` would not clean up — while the running stack kept
    reading the copy it was started from.
    """
    monkeypatch.chdir(tmp_path)
    written = write_web_terminal_artifacts(_config(["alice"]))
    assert (tmp_path / "build" / "docker-compose.web.yml").is_file()
    assert not (tmp_path / "docker-compose.web.yml").exists()
    assert not (tmp_path / "nginx").exists()
    assert {p.parent for p in written} <= {tmp_path / "build", tmp_path / "build" / "nginx"}


def test_write_web_terminal_artifacts_reflects_object_form_users(tmp_path):
    """Object-form users with explicit indices render into the compose overlay."""
    write_web_terminal_artifacts(
        _config([{"name": "alice", "index": 0}, {"name": "bob", "index": 1}]), tmp_path
    )
    compose = (tmp_path / "build" / "docker-compose.web.yml").read_text(encoding="utf-8")
    assert "web-alice" in compose
    assert "web-bob" in compose


def test_write_web_terminal_artifacts_propagates_render_valueerror(tmp_path):
    """An unrenderable config (TLS enabled without cert/key) surfaces as ValueError."""
    config = _config(["alice"])
    config["modules"]["web_terminals"]["tls"] = {"enabled": True}
    with pytest.raises(ValueError):
        write_web_terminal_artifacts(config, tmp_path)


# ---------------------------------------------------------------------------
# The .env.auth digest: this seam is where file content meets the rendered
# sidecar definition, so it is where the digest is computed
# ---------------------------------------------------------------------------


def _auth_config(users):
    """The base config with authentication on (no TLS, so opt into HTTP)."""
    config = _config(users)
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }
    return config


def _rendered_auth_service(dest) -> dict:
    return yaml.safe_load((dest / "docker-compose.web.yml").read_text(encoding="utf-8"))[
        "services"
    ]["auth"]


def test_write_stamps_the_auth_sidecar_with_the_env_auth_content_digest(tmp_path):
    """The label is the sha256 of the file's exact bytes at the REPO ROOT — the
    directory compose resolves `env_file: .env.auth` against once the project
    directory is pinned there, so the digest is a faithful stand-in for what the
    sidecar will actually read. The artifacts themselves land in build/."""
    content = b"OSPREY_AUTH_SESSION_SECRET=abc123\n"
    (tmp_path / AUTH_ENV_FILENAME).write_bytes(content)

    write_web_terminal_artifacts(_auth_config(["alice"]), repo_root=tmp_path)

    auth = _rendered_auth_service(tmp_path / "build")
    assert auth["labels"][AUTH_ENV_DIGEST_LABEL] == hashlib.sha256(content).hexdigest()


def test_hand_edit_of_env_auth_changes_the_rendered_sidecar_definition(tmp_path):
    """THE BUG this label exists to fix: an operator hand-appends OIDC client
    credentials to `.env.auth` (the documented workflow) and redeploys. The
    mint is idempotent, so nothing else about the deploy changes — the re-render
    itself must change the sidecar's service definition, because a definition
    change is the only recreate trigger every compose implementation honours.
    An unchanged file must keep the render byte-identical (no-op redeploys
    recreate nothing)."""
    config = _auth_config(["alice"])
    env_auth = tmp_path / AUTH_ENV_FILENAME
    env_auth.write_text("OSPREY_AUTH_SESSION_SECRET=abc123\n", encoding="utf-8")
    build = tmp_path / "build"

    write_web_terminal_artifacts(config, repo_root=tmp_path)
    before = _rendered_auth_service(build)
    compose_before = (build / "docker-compose.web.yml").read_bytes()

    # No-op redeploy first: same file, byte-identical render.
    write_web_terminal_artifacts(config, repo_root=tmp_path)
    assert (build / "docker-compose.web.yml").read_bytes() == compose_before

    # The hand-edit, exactly as documented for OIDC deployments.
    with env_auth.open("a", encoding="utf-8") as handle:
        handle.write("OSPREY_AUTH_OIDC_CLIENT_SECRET=idp-issued-secret\n")
    write_web_terminal_artifacts(config, repo_root=tmp_path)
    after = _rendered_auth_service(build)

    assert before != after
    assert before["labels"][AUTH_ENV_DIGEST_LABEL] != after["labels"][AUTH_ENV_DIGEST_LABEL]


def test_missing_env_auth_digests_the_empty_string_instead_of_crashing(tmp_path):
    """A render from a root with no `.env.auth` yet (e.g. re-rendering artifacts
    outside a full deploy) must never crash — it stamps the digest of empty
    content, which the first real deploy's re-render then supersedes."""
    write_web_terminal_artifacts(_auth_config(["alice"]), tmp_path)

    auth = _rendered_auth_service(tmp_path / "build")
    assert auth["labels"][AUTH_ENV_DIGEST_LABEL] == hashlib.sha256(b"").hexdigest()


def test_auth_env_digest_reads_the_file_under_the_given_root(tmp_path):
    """The helper the compose-level proof reuses: digest of the exact bytes,
    empty-content sentinel when the file is absent."""
    assert auth_env_digest(tmp_path) == hashlib.sha256(b"").hexdigest()

    (tmp_path / AUTH_ENV_FILENAME).write_bytes(b"A=1\n")
    assert auth_env_digest(tmp_path) == hashlib.sha256(b"A=1\n").hexdigest()


# ---------------------------------------------------------------------------
# Per-persona credentials: this seam resolves them, the render only formats them
#
# `render_web_terminals` reads no filesystem, so every persona entitlement is
# decided HERE, where the deploy root is in scope, and handed down as a set. The
# proof below is therefore end-to-end in a way no render-level test can be: it
# starts from persona projects actually on disk and ends at the rendered compose.
# `BLUESKY_LAUNCH_TOKEN` is the credential worth pinning this way, because it
# arms physical hardware motion — the one grant where handing the set to the
# wrong persona has consequences that a redeploy cannot take back.
# ---------------------------------------------------------------------------

_LAUNCH_TOKEN_LINE = "BLUESKY_LAUNCH_TOKEN=${BLUESKY_LAUNCH_TOKEN:-}"


#: The shell-relevant subset of `settings.json.j2`'s `deny_defaults` (the template also
#: ships the playwright and context7 MCP wildcards, which no guard here reads). Every
#: OSPREY project denies the shell unless its config removed the entry, so a persona
#: project fixture that shipped no `.claude/settings.json` at all would be modelling a
#: deployment that cannot exist — and would trip the Bash/launch-token conflict guard.
_SHIPPED_DENY = ["Bash", "Edit", "WebFetch", "WebSearch"]


def _write_persona_project(
    tmp_path, name: str, project_config: dict, *, denies_bash: bool = True
) -> str:
    """Render a persona project under *tmp_path*; return its relative project_path.

    Writes both halves of what the guard reads: the `config.yml` the entitlement
    predicates walk, and the built `.claude/settings.json` artifact the Bash check
    reads. `denies_bash=False` renders the artifact a project whose config carried
    `claude_code.permissions.remove_deny: ["Bash"]` would produce.
    """
    project_dir = tmp_path / "profiles" / name
    project_dir.mkdir(parents=True)
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, **project_config}), encoding="utf-8"
    )
    deny = [entry for entry in _SHIPPED_DENY if denies_bash or entry != "Bash"]
    (project_dir / ".claude").mkdir()
    (project_dir / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": [], "deny": deny, "ask": []}}), encoding="utf-8"
    )
    return f"profiles/{name}"


def _tiered_roster_config(tmp_path, *, rw_denies_bash: bool = True, ro_denies_bash: bool = True):
    """A two-tier roster whose persona projects are really on disk.

    `alice` runs a read-write persona that also runs the bluesky server, so its
    project satisfies both halves of the launch-token predicate. `bob` runs a
    read-only persona — `writes_enabled: false` — which can never satisfy it.
    Both ship the shell deny by default; the `*_denies_bash` switches drop it to
    construct the conflict the deploy guard refuses.
    """
    config = _config(
        [
            {"name": "alice", "index": 0, "persona": "readwrite"},
            {"name": "bob", "index": 1, "persona": "readonly"},
        ]
    )
    config["modules"]["web_terminals"]["personas"] = {
        "readwrite": {
            "project": "rw",
            "project_path": _write_persona_project(
                tmp_path,
                "rw",
                {"control_system": {"writes_enabled": True}},
                denies_bash=rw_denies_bash,
            ),
        },
        "readonly": {
            "project": "ro",
            "project_path": _write_persona_project(
                tmp_path,
                "ro",
                {"control_system": {"writes_enabled": False}},
                denies_bash=ro_denies_bash,
            ),
        },
    }
    return config


def _rendered_services(dest) -> dict:
    return yaml.safe_load((dest / "docker-compose.web.yml").read_text(encoding="utf-8"))["services"]


def test_write_grants_the_launch_token_to_the_entitled_persona(tmp_path):
    """The wiring, proven from disk: a persona project that enables writes and runs
    the bluesky server gets the launch token in its OWN `environment:` block,
    interpolated from the deploy .env at compose time so no secret is ever written
    into a rendered artifact."""
    write_web_terminal_artifacts(_tiered_roster_config(tmp_path), tmp_path)

    services = _rendered_services(tmp_path / "build")
    assert _LAUNCH_TOKEN_LINE in services["web-alice"]["environment"]


def test_write_never_grants_the_launch_token_to_a_read_only_persona(tmp_path):
    """The tier boundary, asserted POSITIVELY on the persona that must not have it.
    A test that only checks the entitled persona passes just as happily when the
    token leaks to the whole roster — which is the failure mode this seam exists
    to prevent, since this token arms hardware motion."""
    write_web_terminal_artifacts(_tiered_roster_config(tmp_path), tmp_path)

    bob_env = _rendered_services(tmp_path / "build")["web-bob"]["environment"]
    assert _LAUNCH_TOKEN_LINE not in bob_env
    assert not any("BLUESKY_LAUNCH_TOKEN" in value for value in bob_env)


def test_launch_token_is_granted_per_user_and_never_through_the_shared_env_file(tmp_path):
    """`.env.production` is ROSTERWIDE — every persona's container reads it, read-only
    ones included — so the grant must reach exactly one `environment:` block and
    nothing else. Rendering the artifacts must also not create or touch that file:
    it is a durable secret store this seam never writes.
    """
    config = _tiered_roster_config(tmp_path)

    write_web_terminal_artifacts(config, tmp_path)

    compose_text = (tmp_path / "build" / "docker-compose.web.yml").read_text(encoding="utf-8")
    assert compose_text.count(_LAUNCH_TOKEN_LINE) == 1
    services = _rendered_services(tmp_path / "build")
    assert services["web-alice"]["env_file"] == ".env.production"
    assert services["web-bob"]["env_file"] == services["web-alice"]["env_file"]
    assert not (tmp_path / ".env.production").exists()


# ---------------------------------------------------------------------------
# The Bash/launch-token conflict guard
#
# The one accepted cost of granting BLUESKY_LAUNCH_TOKEN is that the chat
# approval gates the `queue_start` tool and nothing else: a persona holding the
# token whose agent may also run a shell reads it out of its own environment and
# arms hardware with no approval at any point. The deploy refuses that pairing
# rather than shipping it, which is what makes the grant defensible — so these
# tests pin the refusal, and pin that it names every offender.
# ---------------------------------------------------------------------------


def test_entitled_persona_permitting_bash_refuses_the_deploy_and_is_named(tmp_path):
    """THE conflict: alice's persona is entitled to the token and its shipped
    settings.json omits the shell deny. The deploy must stop, and the message must
    name the persona — an operator hitting this at `osprey up` has no other context."""
    config = _tiered_roster_config(tmp_path, rw_denies_bash=False)

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert "readwrite" in str(excinfo.value)
    assert excinfo.value.personas == ["readwrite"]


def test_refusing_the_deploy_writes_no_artifacts(tmp_path):
    """The guard runs BEFORE the render, so a refusal leaves nothing half-written.
    A build/ directory carrying a compose file from a refused deploy is worse than
    no deploy: the next `compose up -f build/...` would start the stack the guard
    rejected."""
    with pytest.raises(BashLaunchTokenConflictError):
        write_web_terminal_artifacts(
            _tiered_roster_config(tmp_path, rw_denies_bash=False), tmp_path
        )

    assert not (tmp_path / "build").exists()


def test_the_refusal_states_both_conditions_and_the_remedy(tmp_path):
    """The message IS the deliverable. It has to say what made the persona
    entitled, what made it unsafe, and — because the settings.json read here is
    the one baked into the image at build time — that restoring the deny requires
    a REBUILD, not just a re-render."""
    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(
            _tiered_roster_config(tmp_path, rw_denies_bash=False), tmp_path
        )

    message = str(excinfo.value)
    assert "writes_enabled" in message
    assert "bluesky" in message
    assert "permissions.deny" in message
    assert "REBUILD" in message


def test_a_correctly_configured_deployment_renders_without_the_guard_firing(tmp_path):
    """The negative control. Every persona ships the shell deny — the state every
    OSPREY build produces unless its config removed the entry — so the entitled
    persona still gets its token and the artifacts are written normally."""
    written = write_web_terminal_artifacts(_tiered_roster_config(tmp_path), tmp_path)

    assert written
    assert _LAUNCH_TOKEN_LINE in _rendered_services(tmp_path / "build")["web-alice"]["environment"]


def test_a_read_only_persona_permitting_bash_is_not_a_conflict(tmp_path):
    """Permitting the shell is only a conflict for a persona that HOLDS the token.
    A read-only persona is never handed one, so there is nothing for a shell to
    read — refusing here would block deployments that are not exposed."""
    config = _tiered_roster_config(tmp_path, ro_denies_bash=False)

    write_web_terminal_artifacts(config, tmp_path)

    bob_env = _rendered_services(tmp_path / "build")["web-bob"]["environment"]
    assert not any("BLUESKY_LAUNCH_TOKEN" in value for value in bob_env)


def test_every_offending_persona_is_named_not_just_the_first(tmp_path):
    """An operator fixing one offender at a time would redeploy once per persona to
    discover the next. The guard intersects SETS precisely so one refusal reports
    the whole conflict."""
    config = _tiered_roster_config(tmp_path, rw_denies_bash=False)
    config["modules"]["web_terminals"]["users"].append(
        {"name": "carol", "index": 2, "persona": "alsobad"}
    )
    config["modules"]["web_terminals"]["personas"]["alsobad"] = {
        "project": "bad2",
        "project_path": _write_persona_project(
            tmp_path, "bad2", {"control_system": {"writes_enabled": True}}, denies_bash=False
        ),
    }

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == ["alsobad", "readwrite"]
    message = str(excinfo.value)
    assert "alsobad" in message
    assert "readwrite" in message


def test_the_guard_follows_the_shipped_artifact_not_the_config_intent(tmp_path):
    """`osprey up` does not rebuild. A persona whose config.yml was edited to drop
    the shell deny AFTER its last build still ships an image that denies it — the
    edit reaches nothing until someone rebuilds. Reading intent here would refuse a
    deployment that is not actually exposed, and (in the mirror case) would clear
    one that is."""
    config = _tiered_roster_config(tmp_path)
    rw_config = tmp_path / "profiles" / "rw" / "config.yml"
    parsed = yaml.safe_load(rw_config.read_text(encoding="utf-8"))
    parsed["claude_code"] = {"permissions": {"remove_deny": ["Bash"]}}
    rw_config.write_text(yaml.safe_dump(parsed), encoding="utf-8")

    # The built artifact — what the image actually carries — still denies Bash.
    write_web_terminal_artifacts(config, tmp_path)

    assert _LAUNCH_TOKEN_LINE in _rendered_services(tmp_path / "build")["web-alice"]["environment"]
