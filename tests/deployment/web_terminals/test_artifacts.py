"""Tests for the render-and-write seam shared by bring-up and the lifecycle verbs."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from osprey.cli.templates.claude_code import DENY_DEFAULTS
from osprey.deployment.web_terminals.artifacts import (
    OPEN_MODE_EGRESS_TOOLS,
    UNRENDERED_SETTINGS,
    ZERO_MIGRATION_OFFENDER,
    BashLaunchTokenConflictError,
    DangerouslyAllowBashValueError,
    OpenModeEgressError,
    auth_env_digest,
    bash_launch_token_offenders,
    check_bash_launch_token_conflict,
    check_open_mode_requirements,
    dangerously_allowed_bash_personas,
    open_mode_missing_by_persona,
    open_mode_offenders,
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


#: What `settings.json.j2` actually writes into a rendered project: the template's
#: `deny_defaults`, verbatim. Every OSPREY project denies these unless its config
#: removed an entry, so a persona project fixture that shipped no
#: `.claude/settings.json` at all would be modelling a deployment that cannot exist —
#: and would trip both the Bash/launch-token conflict guard and the open-mode egress
#: gate, each of which reads this artifact and fails closed on its absence.
_SHIPPED_DENY = list(DENY_DEFAULTS)


def _write_persona_project(
    tmp_path,
    name: str,
    project_config: dict,
    *,
    denies_bash: bool = True,
    deny: list[str] | None = None,
) -> str:
    """Render a persona project under *tmp_path*; return its relative project_path.

    Writes both halves of what the guards read: the `config.yml` the entitlement
    predicates walk, and the built `.claude/settings.json` artifact the deny checks
    read. `denies_bash=False` renders the artifact a project whose config carried
    `claude_code.permissions.remove_deny: ["Bash"]` would produce; `deny` sets the
    whole list instead, for the open-mode gate, whose question is about four entries
    rather than one.
    """
    project_dir = tmp_path / "profiles" / name
    project_dir.mkdir(parents=True)
    (project_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": name, **project_config}), encoding="utf-8"
    )
    if deny is None:
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
    Both spell out `claude_code.servers.bluesky.enabled`: the server is opt-in in
    the registry, so a project that omits the key runs no server and would sit
    outside the predicate for the wrong reason. Both ship the shell deny by
    default; the `*_denies_bash` switches drop it to construct the conflict the
    deploy guard refuses.
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
                {
                    "control_system": {"writes_enabled": True},
                    "claude_code": {"servers": {"bluesky": {"enabled": True}}},
                },
                denies_bash=rw_denies_bash,
            ),
        },
        "readonly": {
            "project": "ro",
            "project_path": _write_persona_project(
                tmp_path,
                "ro",
                {
                    "control_system": {"writes_enabled": False},
                    "claude_code": {"servers": {"bluesky": {"enabled": True}}},
                },
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
    assert services["web-alice"]["env_file"] == ".env.users"
    assert services["web-bob"]["env_file"] == services["web-alice"]["env_file"]
    assert not (tmp_path / ".env.users").exists()


# ---------------------------------------------------------------------------
# services.graphdb -> per-user graph-store password, resolved from disk
#
# The same seam as above, for the fourth grant. `config_needs_graphdb_password`
# and `render_web_terminals` are each pinned on their own in test_personas.py
# and test_render.py; what only this file can prove is the COMPOSITION --- that
# the predicate's answer for a persona project really on disk is the answer the
# rendered compose file carries, for every YAML spelling of the block that the
# predicate distinguishes.
#
# The spellings are not interchangeable, and the difference is invisible at a
# glance: a mapping-valued `graphdb: {}` resolves to a fully-defaulted store and
# renders the graph server, so it must grant; a bare `graphdb:` parses as None
# and configures nothing, so it must not. The tier boundary does NOT apply here,
# unlike the launch token above --- reading the graph is a read, so a read-only
# persona that configures a store is entitled to its credential.
# ---------------------------------------------------------------------------

_GRAPHDB_PASSWORD_LINE = "GRAPHDB_PASSWORD=${GRAPHDB_PASSWORD:-ospreygraph}"


def _graphdb_roster_config(tmp_path):
    """A four-user roster covering every spelling the grant predicate separates.

    One persona per cell, each project really on disk:

    * ``alice`` --- a READ-ONLY persona whose config carries a mapping-valued
      ``services.graphdb: {}``. Entitled: the block resolves to a defaulted
      store, and the write switch is not part of this grant.
    * ``bob`` --- configures a store but vetoes the server with
      ``claude_code.servers.graph.enabled: false``. Not entitled: nothing in the
      container would dial it.
    * ``carol`` --- a bare null-valued ``services.graphdb:`` key. Not entitled:
      the key is present but configures no store.
    * ``dave`` --- a malformed block (a non-numeric ``port_host``). Not entitled,
      and the render must complete rather than raise: a deploy is not the place
      to discover a typo by traceback.
    """
    config = _config(
        [
            {"name": "alice", "index": 0, "persona": "readonly_graph"},
            {"name": "bob", "index": 1, "persona": "graph_vetoed"},
            {"name": "carol", "index": 2, "persona": "null_block"},
            {"name": "dave", "index": 3, "persona": "malformed"},
        ]
    )
    config["modules"]["web_terminals"]["personas"] = {
        "readonly_graph": {
            "project": "ro-graph",
            "project_path": _write_persona_project(
                tmp_path,
                "ro-graph",
                {"control_system": {"writes_enabled": False}, "services": {"graphdb": {}}},
            ),
        },
        "graph_vetoed": {
            "project": "vetoed",
            "project_path": _write_persona_project(
                tmp_path,
                "vetoed",
                {
                    "services": {"graphdb": {"port_host": 7687}},
                    "claude_code": {"servers": {"graph": {"enabled": False}}},
                },
            ),
        },
        "null_block": {
            "project": "null-block",
            "project_path": _write_persona_project(
                tmp_path, "null-block", {"services": {"graphdb": None}}
            ),
        },
        "malformed": {
            "project": "malformed",
            "project_path": _write_persona_project(
                tmp_path, "malformed", {"services": {"graphdb": {"port_host": "not-a-port"}}}
            ),
        },
    }
    return config


def test_write_grants_the_graph_password_to_the_persona_that_configures_a_store(tmp_path):
    """The wiring, proven from disk: a persona project carrying a resolvable
    ``services.graphdb`` block gets the store's password in its OWN
    ``environment:`` block, interpolated from the deploy ``.env`` at compose time
    so no secret is written into a rendered artifact.

    Asserted on a READ-ONLY persona deliberately. This grant crosses the tier
    boundary the launch token above defends, and a fixture that granted it to a
    write-armed persona would leave that difference untested.
    """
    write_web_terminal_artifacts(_graphdb_roster_config(tmp_path), tmp_path)

    services = _rendered_services(tmp_path / "build")
    assert _GRAPHDB_PASSWORD_LINE in services["web-alice"]["environment"]


@pytest.mark.parametrize(
    ("user", "why"),
    [
        ("bob", "the graph server is switched off, so nothing would dial the store"),
        ("carol", "a bare `graphdb:` key parses as None and configures no store"),
        ("dave", "a malformed block resolves to no store at all"),
    ],
)
def test_write_grants_the_graph_password_to_no_one_else(tmp_path, user: str, why: str):
    """Every non-entitled spelling, asserted positively on the persona that must
    not hold the credential. A test that only checks the entitled persona passes
    just as happily when the password leaks to the whole roster.
    """
    write_web_terminal_artifacts(_graphdb_roster_config(tmp_path), tmp_path)

    env = _rendered_services(tmp_path / "build")[f"web-{user}"]["environment"]
    assert not any("GRAPHDB_PASSWORD" in value for value in env), why


def test_a_malformed_graphdb_block_renders_instead_of_raising(tmp_path):
    """A typo in one persona's block must not take the whole deploy down.

    The predicate reads a malformed block as "no store", so the render completes
    and every other persona's entitlement is decided normally --- which is what
    makes the parametrized denial above a statement about entitlement rather
    than about the deploy having failed before it got there.
    """
    written = write_web_terminal_artifacts(_graphdb_roster_config(tmp_path), tmp_path)

    assert (tmp_path / "build" / "docker-compose.web.yml") in written
    assert set(_rendered_services(tmp_path / "build")) >= {
        "web-alice",
        "web-bob",
        "web-carol",
        "web-dave",
    }


def test_graph_password_is_granted_per_user_and_never_through_the_shared_env_file(tmp_path):
    """``.env.users`` is ROSTERWIDE, so the grant must reach exactly one
    ``environment:`` block and nothing else --- the same placement argument the
    launch token makes, and the reason this credential is absent from the
    ``.env.users`` subset (see ``test_env_production.py``).
    """
    write_web_terminal_artifacts(_graphdb_roster_config(tmp_path), tmp_path)

    compose_text = (tmp_path / "build" / "docker-compose.web.yml").read_text(encoding="utf-8")
    # Counted by LINE, not by occurrence: the grant spells the variable twice in
    # its own line (`VAR=${VAR:-default}`), so a bare substring count would read
    # the single correct grant as two.
    mentions = [line for line in compose_text.splitlines() if "GRAPHDB_PASSWORD" in line]
    assert mentions == [f"      - {_GRAPHDB_PASSWORD_LINE}"]
    assert not (tmp_path / ".env.users").exists()


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


def _zero_migration_config(tmp_path, *, writes_enabled: bool = True, denies_bash: bool = True):
    """A persona-less roster: the web image IS the deploy project.

    No persona catalog and no default_persona, so every entry runs the deploy
    project itself; entitlement is answered by ``config_needs_launch_token_for``
    on the deploy config, and the shipped settings artifact is
    ``<project_root>/.claude/settings.json``. The deploy config spells out
    ``claude_code.servers.bluesky.enabled`` for the same reason every persona
    fixture here does: the server is opt-in in the registry, so a config that
    omits the key runs no server and would sit outside the predicate for the
    wrong reason.
    """
    config = _config(["alice"])
    config["claude_code"] = {"servers": {"bluesky": {"enabled": True}}}
    if writes_enabled:
        config["control_system"] = {"writes_enabled": True}
    if denies_bash:
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text(
            json.dumps({"permissions": {"allow": [], "deny": list(_SHIPPED_DENY), "ask": []}}),
            encoding="utf-8",
        )
    return config


def test_an_entitled_personaless_roster_without_the_bash_deny_refuses(tmp_path):
    """The zero-migration half of the same conflict: no persona is in effect, the
    deploy config itself entitles every entry to the token, and the deploy
    project ships no shell deny (here: no settings.json at all, which counts the
    same — absence is not evidence of a deny). The guard must refuse rather than
    leaving the persona-less path silently unbound."""
    config = _zero_migration_config(tmp_path, denies_bash=False)

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == [ZERO_MIGRATION_OFFENDER]
    assert "no persona" in str(excinfo.value)
    assert not (tmp_path / "build").exists()


def test_an_entitled_personaless_roster_shipping_the_bash_deny_deploys(tmp_path):
    """The negative control: the deploy project ships the shell deny every OSPREY
    build produces, so the entitled persona-less entry keeps its token."""
    written = write_web_terminal_artifacts(_zero_migration_config(tmp_path), tmp_path)

    assert written
    assert _LAUNCH_TOKEN_LINE in _rendered_services(tmp_path / "build")["web-alice"]["environment"]


def test_an_unentitled_personaless_roster_is_not_a_conflict(tmp_path):
    """No entitlement, nothing for a shell to read — a bare zero-migration deploy
    with no write grant must keep deploying exactly as before."""
    config = _zero_migration_config(tmp_path, writes_enabled=False, denies_bash=False)

    written = write_web_terminal_artifacts(config, tmp_path)

    assert written
    env = _rendered_services(tmp_path / "build")["web-alice"]["environment"]
    assert not any("BLUESKY_LAUNCH_TOKEN" in value for value in env)


def test_a_default_persona_roster_is_covered_by_the_persona_check_not_the_sentinel(tmp_path):
    """Entries with no explicit persona but a default_persona resolve to the
    default, so the persona-keyed intersection already binds them — the
    persona-less check must not double-report (or misattribute) them."""
    config = _tiered_roster_config(tmp_path, rw_denies_bash=False)
    config["modules"]["web_terminals"]["default_persona"] = "readwrite"
    config["modules"]["web_terminals"]["users"] = [{"name": "alice", "index": 0}]
    # Entitle the deploy root too — writes AND the bluesky server, or the
    # entitlement would not be real — and the sentinel must still not appear,
    # because no entry actually runs the deploy project.
    config["control_system"] = {"writes_enabled": True}
    config["claude_code"] = {"servers": {"bluesky": {"enabled": True}}}

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == ["readwrite"]


def test_a_role_only_roster_is_covered_by_the_persona_check_not_the_sentinel(tmp_path):
    """An entry that names a ``role:`` and no ``persona:`` still runs a persona —
    the one the ``authorization`` block binds the role to — so the persona-keyed
    intersection binds it and the persona-less sentinel must not fire. A guard
    that read the raw ``persona`` key would call this roster persona-less and
    judge the deploy project's own settings instead of the persona's."""
    config = _tiered_roster_config(tmp_path, rw_denies_bash=False)
    config["modules"]["web_terminals"]["authorization"] = {
        "roles": {"operator": {"persona": "readwrite"}}
    }
    config["modules"]["web_terminals"]["users"] = [
        {"name": "alice", "index": 0, "role": "operator"}
    ]
    config["control_system"] = {"writes_enabled": True}

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == ["readwrite"]


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
            tmp_path,
            "bad2",
            {
                "control_system": {"writes_enabled": True},
                "claude_code": {"servers": {"bluesky": {"enabled": True}}},
            },
            denies_bash=False,
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
    # Merged, not assigned: the same section already carries the bluesky server
    # this persona is entitled through, and replacing it would disarm the persona
    # the test is about rather than only editing its permissions.
    parsed["claude_code"]["permissions"] = {"remove_deny": ["Bash"]}
    rw_config.write_text(yaml.safe_dump(parsed), encoding="utf-8")

    # The built artifact — what the image actually carries — still denies Bash.
    write_web_terminal_artifacts(config, tmp_path)

    assert _LAUNCH_TOKEN_LINE in _rendered_services(tmp_path / "build")["web-alice"]["environment"]


# ---------------------------------------------------------------------------
# The guard is per LANE
#
# Each plan lane carries its own launch token, armed by the write posture of
# the control target that lane drives. A persona can therefore hold the VA
# lane's token on a deployment whose baseline is a live machine — and a shell
# in that container reaches the VA hardware just as directly, so the conflict
# is refused on any lane, not only on lane 1.
# ---------------------------------------------------------------------------

#: A persona project whose baseline is a live machine with writes disarmed, that
#: arms the VA connector alone and renders the VA second lane. Lane 1 declares no
#: target and so drives `live`, which this config leaves unarmed.
_VA_ARMED_PROJECT: dict = {
    "control_system": {
        "type": "epics",
        "writes_enabled": False,
        "connector": {"epics": {}, "virtual_accelerator": {"writes_enabled": True}},
    },
    "services": {"bluesky_va": {"target": "va", "port": 10081}},
    "claude_code": {"servers": {"bluesky": {"enabled": True}}},
}


def _va_lane_roster_config(tmp_path, *, denies_bash: bool = True):
    """A one-user roster whose persona is entitled on the VA lane only."""
    config = _config([{"name": "alice", "index": 0, "persona": "va_operator"}])
    config["modules"]["web_terminals"]["personas"] = {
        "va_operator": {
            "project": "va",
            "project_path": _write_persona_project(
                tmp_path, "va", _VA_ARMED_PROJECT, denies_bash=denies_bash
            ),
        }
    }
    return config


def test_the_guard_returns_the_entitlement_map_keyed_by_lane(tmp_path):
    """The guard hands its caller what it cleared, so the render grants exactly the
    set that passed. Keyed by lane, because each lane's token is a separate grant:
    the tiered roster arms lane 1 and renders no other."""
    result = check_bash_launch_token_conflict(_tiered_roster_config(tmp_path), tmp_path)

    assert result == {"bluesky": {"readwrite"}}


def test_a_persona_entitled_on_the_va_lane_alone_is_still_a_conflict(tmp_path):
    """The refusal is not about lane 1. A persona holding only the VA lane's token
    can read it out of its environment and arm the virtual accelerator's queue with
    no approval, which is the same bypass on a different machine."""
    config = _va_lane_roster_config(tmp_path, denies_bash=False)

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == ["va_operator"]
    assert excinfo.value.personas_by_lane == {"bluesky_va": ["va_operator"]}


def test_the_refusal_names_the_lane_and_the_key_that_disarms_it(tmp_path):
    """The remedy has to name the key an operator actually edits. Sending them to
    the deployment-wide key would have them disarm every target at once — and on
    this deployment that key is already false, so it would read as no remedy at
    all."""
    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        check_bash_launch_token_conflict(
            _va_lane_roster_config(tmp_path, denies_bash=False), tmp_path
        )

    message = str(excinfo.value)
    assert "bluesky_va" in message
    assert "BLUESKY_VA_LAUNCH_TOKEN" in message
    assert "control_system.connector.virtual_accelerator.writes_enabled" in message
    assert "claude_code.servers.bluesky.enabled" in message


def test_the_offender_set_is_the_union_over_every_lane(tmp_path):
    """`bash_launch_token_offenders` answers one question — who is in conflict —
    for a caller that must not raise. One token is one shell away from arming its
    own lane, so entitlement on ANY lane puts a persona in the set."""
    assert bash_launch_token_offenders(_va_lane_roster_config(tmp_path), tmp_path) == set()
    assert bash_launch_token_offenders(
        _va_lane_roster_config(tmp_path / "conflicted", denies_bash=False), tmp_path / "conflicted"
    ) == {"va_operator"}


def test_a_lane_nobody_may_arm_is_absent_from_the_entitlement_map(tmp_path):
    """Lane 1 drives the live machine here and is disarmed, so it is missing from
    the map rather than present with an empty set — the deployment renders no
    lane-1 grant at all."""
    result = check_bash_launch_token_conflict(_va_lane_roster_config(tmp_path), tmp_path)

    assert result == {"bluesky_va": {"va_operator"}}


def test_the_va_lane_grant_reaches_no_lane_one_token(tmp_path):
    """The tier boundary, per target: a persona armed only for the virtual
    accelerator must never be handed lane 1's token, which arms the live machine."""
    write_web_terminal_artifacts(_va_lane_roster_config(tmp_path), tmp_path)

    alice_env = _rendered_services(tmp_path / "build")["web-alice"]["environment"]
    assert not any("BLUESKY_LAUNCH_TOKEN" in value for value in alice_env)


def test_a_personaless_roster_armed_on_the_va_lane_alone_is_refused_by_lane(tmp_path):
    """The two halves together: no persona is in effect, and the deploy config
    arms the virtual accelerator alone while its baseline live machine stays
    read-only. The persona-less entry therefore holds the VA lane's token and
    nothing else — and with no shell deny shipped, the refusal must name that
    lane and the per-connector key that disarms it, not lane 1 and not the
    deployment-wide key (which is already false here and would read as no
    remedy at all)."""
    config = _config(["alice"])
    config.update(
        {
            "control_system": {
                "type": "epics",
                "writes_enabled": False,
                "connector": {"epics": {}, "virtual_accelerator": {"writes_enabled": True}},
            },
            "services": {"bluesky_va": {"target": "va", "port": 10081}},
            "claude_code": {"servers": {"bluesky": {"enabled": True}}},
        }
    )

    with pytest.raises(BashLaunchTokenConflictError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == [ZERO_MIGRATION_OFFENDER]
    assert excinfo.value.personas_by_lane == {"bluesky_va": [ZERO_MIGRATION_OFFENDER]}
    message = str(excinfo.value)
    assert "BLUESKY_VA_LAUNCH_TOKEN" in message
    assert "control_system.connector.virtual_accelerator.writes_enabled" in message
    assert "no persona" in message
    assert not (tmp_path / "build").exists()
    # The ask-only reader binds the same entry: one shared predicate, so the
    # collect-all preflight cannot clear what the raising guard refuses.
    assert bash_launch_token_offenders(config, tmp_path) == {ZERO_MIGRATION_OFFENDER}


# ---------------------------------------------------------------------------
# dangerously_allow_bash -- the one key that waives the Bash/launch-token refusal
# ---------------------------------------------------------------------------


def _dangerous_va_config(tmp_path, value=True):
    config = _va_lane_roster_config(tmp_path, denies_bash=False)
    config["dangerously_allow_bash"] = value
    return config


def test_dangerously_allow_bash_waives_the_refusal_and_still_grants_the_token(tmp_path):
    """The key changes ONE thing: the deploy no longer refuses. The entitlement is
    untouched -- the same persona is granted the same lane's token by the same
    rule -- so the render must carry BLUESKY_VA_LAUNCH_TOKEN for a persona whose
    shipped settings permit Bash."""
    config = _dangerous_va_config(tmp_path)

    assert bash_launch_token_offenders(config, tmp_path) == set()
    assert check_bash_launch_token_conflict(config, tmp_path) == {"bluesky_va": {"va_operator"}}
    write_web_terminal_artifacts(config, tmp_path)

    alice_env = _rendered_services(tmp_path / "build")["web-alice"]["environment"]
    assert any("BLUESKY_VA_LAUNCH_TOKEN" in value for value in alice_env)


def test_dangerously_allow_bash_names_the_personas_it_waved_through(tmp_path):
    """What the banner prints: the personas the guard WOULD have refused. Computed
    by the same predicate, so the banner and the refusal cannot disagree about
    who is in conflict. Empty whenever the key is off -- including on a
    conflict-free roster with the key on."""
    assert dangerously_allowed_bash_personas(_dangerous_va_config(tmp_path), tmp_path) == {
        "va_operator"
    }
    assert (
        dangerously_allowed_bash_personas(
            _va_lane_roster_config(tmp_path / "off", denies_bash=False), tmp_path / "off"
        )
        == set()
    )
    clean = _va_lane_roster_config(tmp_path / "clean")
    clean["dangerously_allow_bash"] = True
    assert dangerously_allowed_bash_personas(clean, tmp_path / "clean") == set()


@pytest.mark.parametrize("value", [False, None])
def test_dangerously_allow_bash_off_is_byte_for_byte_the_refusal(tmp_path, value):
    with pytest.raises(BashLaunchTokenConflictError):
        check_bash_launch_token_conflict(_dangerous_va_config(tmp_path, value), tmp_path)


@pytest.mark.parametrize("value", ["true", 1, "yes", "I-understand-the-risk"])
def test_dangerously_allow_bash_accepts_only_the_boolean_true(tmp_path, value):
    """A key this loud cannot be half-set. Anything but a literal YAML `true` is
    a config error naming the key -- not a silent refusal (which would read as
    the key not working) and not a silent waiver."""
    with pytest.raises(DangerouslyAllowBashValueError) as excinfo:
        check_bash_launch_token_conflict(_dangerous_va_config(tmp_path, value), tmp_path)

    assert "dangerously_allow_bash" in str(excinfo.value)
    assert repr(value) in str(excinfo.value)


# ---------------------------------------------------------------------------
# The OPEN-mode egress gate
#
# `auth.method: none` means nginx vouches for every terminal it proxies. An
# agent inside one terminal reaches nginx over loopback and is indistinguishable
# from the operator's browser, so open mode is refused unless every persona's
# shipped settings deny the host-network egress tools the python executor's own
# socket guard cannot cover.
# ---------------------------------------------------------------------------

#: A persona project that is entitled to no launch token at all — writes off and
#: no bluesky server — so these tests exercise the open-mode gate alone and a
#: failure here can never be the Bash/launch-token guard firing instead.
_UNARMED_PROJECT: dict = {"control_system": {"writes_enabled": False}}


def _open_roster_config(tmp_path, *, method: str = "none", deny: list[str] | None = None):
    """A one-user roster on *method*, whose persona ships exactly *deny*."""
    config = _config([{"name": "alice", "index": 0, "persona": "operator"}])
    config["modules"]["web_terminals"]["auth"] = {"method": method}
    config["modules"]["web_terminals"]["personas"] = {
        "operator": {
            "project": "op",
            "project_path": _write_persona_project(
                tmp_path, "op", _UNARMED_PROJECT, deny=deny or list(_SHIPPED_DENY)
            ),
        }
    }
    return config


def _without(*tools: str) -> list[str]:
    """The shipped deny list with *tools* lifted, as `remove_deny` would render it."""
    return [entry for entry in _SHIPPED_DENY if entry not in tools]


def test_the_open_mode_egress_tools_are_spelled_as_the_template_ships_them(tmp_path):
    """The gate compares literal `permissions.deny` entries against the artifact
    `settings.json.j2` writes from `deny_defaults`. A rename there that this tuple
    did not follow would not fail loudly — it would silently stop matching, and the
    gate would clear a persona that still holds the tool. So the subset relationship
    is pinned rather than left to be noticed."""
    assert set(OPEN_MODE_EGRESS_TOOLS) <= set(DENY_DEFAULTS)
    # And it is a STRICT subset on purpose: `Edit` writes files and the context7
    # server reaches a documentation host, neither of which is a route back to
    # this deployment's own terminals.
    assert set(DENY_DEFAULTS) - set(OPEN_MODE_EGRESS_TOOLS) == {
        "Edit",
        "mcp__plugin_context7_context7__*",
    }


def test_open_mode_refuses_a_persona_that_may_run_a_shell(tmp_path):
    """The headline case. A shell reaches every port on the host, so it walks
    straight past the executor's in-process socket guard into a neighbour's
    terminal — which open mode hands out to anything that asks."""
    config = _open_roster_config(tmp_path, deny=_without("Bash"))

    with pytest.raises(OpenModeEgressError) as excinfo:
        check_open_mode_requirements(config, tmp_path)

    assert excinfo.value.personas == ["operator"]
    message = str(excinfo.value)
    assert "'operator'" in message
    # The remedy an operator can take without touching any persona at all.
    assert "modules.web_terminals.auth.method to 'token'" in message


def test_open_mode_refuses_a_persona_that_lifted_only_one_web_tool(tmp_path):
    """`Bash` is not the whole perimeter, and a gate that only asked about it would
    clear a persona whose agent can still GET a neighbour's terminal. The refusal
    names the one tool that is missing rather than sending the operator through all
    four — three of which are already denied here."""
    config = _open_roster_config(tmp_path, deny=_without("WebFetch"))

    with pytest.raises(OpenModeEgressError) as excinfo:
        check_open_mode_requirements(config, tmp_path)

    assert excinfo.value.missing_by_persona == {"operator": ["WebFetch"]}
    message = str(excinfo.value)
    assert "may still reach the host network via 'WebFetch'." in message
    assert "'Bash'," not in message.split("may still reach")[1].split("\n")[0]


def test_open_mode_passes_when_every_persona_denies_the_whole_egress_set(tmp_path):
    """The shipped default: a project rendered from `deny_defaults` denies all four,
    so the ordinary open deployment starts. A gate that refused this would be a gate
    nobody could satisfy without hand-editing an artifact."""
    check_open_mode_requirements(_open_roster_config(tmp_path), tmp_path)

    assert open_mode_offenders(_open_roster_config(tmp_path / "twin"), tmp_path / "twin") == set()


@pytest.mark.parametrize("method", ["token", "password", "oidc"])
def test_a_walled_or_token_deployment_is_not_asked_the_open_question(tmp_path, method):
    """The gate is about what nginx vouches for, not about what a persona may run.
    Under `token` the browser still has to present the per-user magic link, and
    `password`/`oidc` put a login wall in front of the roster — so a persona with a
    shell is a deliberate, documented posture there and must not be refused."""
    config = _open_roster_config(tmp_path, method=method, deny=_without("Bash"))

    check_open_mode_requirements(config, tmp_path)

    assert open_mode_offenders(config, tmp_path) == set()


def test_open_mode_fails_closed_on_a_settings_artifact_it_cannot_read(tmp_path):
    """An unparseable artifact is not evidence of a deny. Answering "safe" here
    would make a corrupt or truncated settings.json the easiest way through the
    gate — and the operator would never see it happen."""
    config = _open_roster_config(tmp_path)
    (tmp_path / "profiles" / "op" / ".claude" / "settings.json").write_text(
        "{ not json", encoding="utf-8"
    )

    with pytest.raises(OpenModeEgressError) as excinfo:
        check_open_mode_requirements(config, tmp_path)

    # Nothing was read, so every tool in the set is reported missing.
    assert excinfo.value.missing_by_persona == {"operator": list(OPEN_MODE_EGRESS_TOOLS)}


def test_open_mode_names_the_missing_render_rather_than_all_four_tools(tmp_path):
    """A persona with no rendered project on this host fails every deny check for
    a reason no `permissions.deny` edit can fix. Listing the four entries there
    sends the operator to a file that is not on the disk — so that case is
    reported as the render it actually is, with the remedy that clears it.

    This is the state a registry-mode open deployment is in by default, which is
    why `persona_render_problem` now demands the render in that mode too."""
    config = _open_roster_config(tmp_path)
    (tmp_path / "profiles" / "op" / ".claude" / "settings.json").unlink()

    with pytest.raises(OpenModeEgressError) as excinfo:
        check_open_mode_requirements(config, tmp_path)

    assert excinfo.value.missing_by_persona == {"operator": list(UNRENDERED_SETTINGS)}
    message = str(excinfo.value)
    assert "'operator' has no rendered .claude/settings.json on this host" in message
    assert "osprey build" in message
    # The whole set is still what the deployment must eventually deny -- an
    # unrendered persona denies nothing -- so the headline names all four.
    assert "'Bash'" in message
    # And the remedy no longer claims a re-pull alone clears this: what is read
    # here is THIS host's render, in either image-source mode.
    assert "a re-pull alone is not enough" in message


def test_open_mode_reads_the_settings_artifact_once_per_offender(tmp_path, monkeypatch):
    """The gate names four entries per offender off ONE read of the artifact,
    not one roster walk per entry. Four reads of the same small JSON file per
    persona is affordable, but it is also four chances for the walks to disagree
    about which personas a deployment has."""
    from osprey.deployment.web_terminals import artifacts as artifacts_module

    reads: list[str] = []
    real = artifacts_module.settings_json_deny_entries

    def counted(project_dir):
        reads.append(str(project_dir))
        return real(project_dir)

    monkeypatch.setattr(artifacts_module, "settings_json_deny_entries", counted)

    assert open_mode_missing_by_persona(_open_roster_config(tmp_path), tmp_path) == {}
    assert len(reads) == 1


def test_open_mode_refuses_a_persona_that_denies_one_playwright_tool_by_name(tmp_path):
    """The near miss that looks safe. The artifact is compared by EXACT entry, so
    a persona denying `...__browser_navigate` still ships every other browser
    tool — and any of them reaches a neighbour's terminal just as well. The
    refusal names the wildcard, which is the entry that actually closes it."""
    wildcard = "mcp__plugin_playwright_playwright__*"
    config = _open_roster_config(
        tmp_path,
        deny=[
            *_without(wildcard),
            "mcp__plugin_playwright_playwright__browser_navigate",
        ],
    )

    with pytest.raises(OpenModeEgressError) as excinfo:
        check_open_mode_requirements(config, tmp_path)

    assert excinfo.value.missing_by_persona == {"operator": [wildcard]}
    assert f"via {wildcard!r}." in str(excinfo.value)


def test_open_mode_binds_the_roster_entries_that_run_no_persona(tmp_path):
    """The zero-migration path: a bare-string roster entry runs the deploy project
    itself, so it appears in no persona set and would otherwise walk through the
    gate untouched. Its artifact is the deploy project's own settings.json — absent
    here, which fails closed under the sentinel name."""
    config = _config(["alice"])
    config["modules"]["web_terminals"]["auth"] = {"method": "none"}

    with pytest.raises(OpenModeEgressError) as excinfo:
        write_web_terminal_artifacts(config, tmp_path)

    assert excinfo.value.personas == [ZERO_MIGRATION_OFFENDER]
    assert "no persona" in str(excinfo.value)
    # Refused BEFORE the render, so a rejected deploy leaves nothing half-written.
    assert not (tmp_path / "build").exists()
    # The ask-only reader binds the same entry: one shared predicate, so the
    # collect-all preflight cannot clear what the raising gate refuses.
    assert open_mode_offenders(config, tmp_path) == {ZERO_MIGRATION_OFFENDER}


def test_the_render_seam_refuses_an_open_deployment_before_writing_anything(tmp_path):
    """`force_recreate_auth_sidecar` re-renders through neither `osprey up`'s
    preflight nor `decommission_user`, so the seam has to answer for itself — and
    has to refuse before the first artifact lands."""
    config = _open_roster_config(tmp_path, deny=_without("Bash"))

    with pytest.raises(OpenModeEgressError):
        write_web_terminal_artifacts(config, tmp_path)

    assert not (tmp_path / "build").exists()


def test_an_unknown_auth_method_is_not_treated_as_open(tmp_path):
    """A method nothing supports renders no deployment at all — the render raises on
    it and lint reports it. Reporting it here as well would give the operator a
    second, confusing refusal about personas, and would put a raise inside the
    ask-only reader its callers are promised will not raise."""
    config = _open_roster_config(tmp_path, method="sso", deny=_without("Bash"))

    check_open_mode_requirements(config, tmp_path)

    assert open_mode_offenders(config, tmp_path) == set()


def test_the_collect_all_preflight_reports_the_open_mode_refusal(tmp_path, monkeypatch):
    """The gate refuses one deploy at a time; the preflight REPORT exists to say
    everything wrong at once. It reads the same predicate without raising, so an
    operator sees the open-mode problem in the same pass as the others."""
    from osprey.deployment.web_terminals import provision

    monkeypatch.chdir(tmp_path)
    config = _open_roster_config(tmp_path, deny=_without("Bash"))

    findings, _advisories = provision.web_terminal_preflight_report(config, repo_root=tmp_path)

    assert any("may still reach the host network" in problem for problem, _ in findings)
