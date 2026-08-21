"""Unit tests for ``.env.users`` generation.

Covers ``osprey.deployment.web_terminals.env_production`` in isolation: the
module-conditional subset generator and its claude_code provider auth-secret
coverage.
"""

from __future__ import annotations

import os

import pytest
import yaml

from osprey.deployment.web_terminals import env_production

# ---------------------------------------------------------------------------
# ensure_env_production -- module-conditional subset generator for local-mode
# web-terminal deploys.
# ---------------------------------------------------------------------------


def _write_dotenv(path, values: dict) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


# A facility config with every relevant module enabled, plus every excluded
# secret's config-declared name present too -- this is the fixture the
# security spec (the exclusion list) gets unit-tested against.
_FULL_CONFIG = {
    "facility": {"name": "Test Facility", "prefix": "test", "timezone": "America/Los_Angeles"},
    "llm": {"provider": "cborg", "api_key_env_var": "CBORG_API_KEY"},
    "ci": {"provider": "gitlab", "token_env_var": "TEST_CI_TOKEN"},
    "registry": {
        "url": "registry.example.org/test",
        "token_env_var": "TEST_REGISTRY_TOKEN",
        "external_projects": [
            {
                "name": "beam-viewer",
                "url": "registry.example.org/beam-viewer",
                "image": "beam-viewer:latest",
                "token_env_var": "BEAM_VIEWER_DEPLOY_TOKEN",
            }
        ],
    },
    "modules": {
        "web_terminals": {"enabled": True, "image_source": "local"},
        "olog": {
            "enabled": True,
            "username_env_var": "OLOG_USERNAME",
            "password_env_var": "OLOG_PASSWORD",
        },
        "wiki_search": {"enabled": True, "token_env_var": "CONFLUENCE_ACCESS_TOKEN"},
        # Enabled, and carrying a legacy token-name knob on purpose: the
        # generator must ignore both. The dispatcher's tokens are OSPREY's own
        # service-to-service credentials, minted under fixed names, and no web
        # terminal ever presents one.
        "event_dispatcher": {"enabled": True, "token_env_var": "EVENT_DISPATCHER_TOKEN"},
        "ariel": {
            "enabled": True,
            "dsn": "postgresql://ariel:ariel@ariel-postgres:5432/ariel",
        },
    },
}

# Every secret .env.users must NEVER contain -- the build-time
# credentials (CI, registry, external-project pulls) and the fixed-name tokens
# OSPREY's own deployed services authenticate to each other with. This
# exclusion list is the security spec for the generator, so a service token
# that is granted per-persona elsewhere belongs here too: BLUESKY_LAUNCH_TOKEN
# reaches its entitled containers through the per-user compose `environment:`
# block, and the whole point of that placement is that this rosterwide file
# never carries it.
_EXCLUDED_ENV = {
    "TEST_CI_TOKEN": "ci-secret",
    "TEST_REGISTRY_TOKEN": "registry-secret",
    "BEAM_VIEWER_DEPLOY_TOKEN": "external-project-secret",
    "EVENT_DISPATCHER_TOKEN": "dispatcher-secret",
    "DISPATCH_WORKER_TOKEN": "worker-secret",
    "BLUESKY_LAUNCH_TOKEN": "launch-token-secret",
}

# Credentials the agent inside a web terminal presents to systems outside the
# deploy -- the only kind that earns a place in .env.users.
_INCLUDED_ENV = {
    "CBORG_API_KEY": "llm-secret",
    "OLOG_USERNAME": "olog-user",
    "OLOG_PASSWORD": "olog-pass",
    "CONFLUENCE_ACCESS_TOKEN": "wiki-secret",
}


def test_env_production_present_returned_as_is(tmp_path):
    marker = "# operator-authored, do not touch\nFOO=bar\n"
    (tmp_path / ".env.users").write_text(marker, encoding="utf-8")

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result == tmp_path / ".env.users"
    assert result.read_text(encoding="utf-8") == marker


def test_env_production_present_in_registry_mode_returned_as_is(tmp_path):
    marker = "FOO=bar\n"
    (tmp_path / ".env.users").write_text(marker, encoding="utf-8")
    config = {**_FULL_CONFIG, "modules": {**_FULL_CONFIG["modules"], "web_terminals": {}}}

    result = env_production.ensure_env_production(config, tmp_path)

    assert result.read_text(encoding="utf-8") == marker


def test_env_production_neither_present_raises_actionably(tmp_path):
    with pytest.raises(RuntimeError, match=r"\.env\.users.*\.env"):
        env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert not (tmp_path / ".env.users").exists()


def test_env_production_registry_mode_never_generates_even_with_env_present(tmp_path):
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})
    config = {**_FULL_CONFIG, "modules": {**_FULL_CONFIG["modules"], "web_terminals": {}}}

    with pytest.raises(RuntimeError, match="Registry-mode"):
        env_production.ensure_env_production(config, tmp_path)

    assert not (tmp_path / ".env.users").exists()


def test_env_production_local_mode_generates_from_env(tmp_path):
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result == tmp_path / ".env.users"
    generated = env_production.parse_dotenv_file(result)

    # Included: llm key, module-gated olog/wiki credentials, ARIEL_DSN, TZ.
    assert generated["CBORG_API_KEY"] == "llm-secret"
    assert generated["OLOG_USERNAME"] == "olog-user"
    assert generated["OLOG_PASSWORD"] == "olog-pass"
    assert generated["CONFLUENCE_ACCESS_TOKEN"] == "wiki-secret"
    assert generated["ARIEL_DSN"] == "postgresql://ariel:ariel@ariel-postgres:5432/ariel"
    assert generated["TZ"] == "America/Los_Angeles"


def test_env_production_never_includes_excluded_secrets(tmp_path):
    """The security spec: CI, registry, and external-project tokens, plus the
    tokens OSPREY's own services authenticate to each other with, must never
    appear in the generated file -- neither their key nor their value, even
    though the source .env contains all of them."""
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    generated = env_production.parse_dotenv_file(result)
    raw_text = result.read_text(encoding="utf-8")

    for excluded_key, excluded_value in _EXCLUDED_ENV.items():
        assert excluded_key not in generated
        assert excluded_key not in raw_text
        assert excluded_value not in raw_text


def test_env_production_omits_service_tokens_even_with_module_enabled(tmp_path):
    """A native-service token is excluded on its own merits, not because some
    module happened to be off: _FULL_CONFIG enables event_dispatcher AND still
    carries a legacy knob naming EVENT_DISPATCHER_TOKEN, the minted values sit
    in .env, and the generated file must still contain neither the caller-facing
    token nor the internal dispatcher-to-worker one."""
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    generated = env_production.parse_dotenv_file(result)
    assert _FULL_CONFIG["modules"]["event_dispatcher"]["enabled"] is True
    assert "EVENT_DISPATCHER_TOKEN" not in generated
    assert "DISPATCH_WORKER_TOKEN" not in generated


def test_env_production_generated_file_is_mode_0600(tmp_path):
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert (result.stat().st_mode & 0o777) == 0o600


def test_env_production_created_with_restrictive_mode_atomically(monkeypatch, tmp_path):
    """Regression guard for the write-then-chmod umask race: the file must be
    opened with mode 0600 from the very first os.open call (O_CREAT with an
    explicit restrictive mode), never created at the process umask (e.g.
    0644) and tightened only after every secret has already been written."""
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)

    captured: dict = {}
    real_open = os.open

    def _spy_open(path, flags, mode=0o777):
        if str(path).endswith(".env.users"):
            captured["flags"] = flags
            captured["mode"] = mode
        return real_open(path, flags, mode)

    monkeypatch.setattr(env_production.os, "open", _spy_open)

    env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert captured, "os.open was never called for .env.users"
    assert captured["mode"] == 0o600
    assert captured["flags"] & os.O_CREAT


def test_env_production_module_disabled_omits_its_vars(tmp_path):
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)
    config = {
        "facility": {"timezone": "UTC"},
        "llm": {"api_key_env_var": "CBORG_API_KEY"},
        "modules": {
            "web_terminals": {"image_source": "local"},
            "olog": {"enabled": False, "username_env_var": "OLOG_USERNAME"},
            "wiki_search": {"enabled": False, "token_env_var": "CONFLUENCE_ACCESS_TOKEN"},
            "ariel": {"enabled": False, "dsn": "postgresql://ariel:ariel@ariel-postgres/ariel"},
        },
    }

    result = env_production.ensure_env_production(config, tmp_path)
    generated = env_production.parse_dotenv_file(result)

    assert generated == {"CBORG_API_KEY": "llm-secret", "TZ": "UTC"}


def test_env_production_missing_var_in_env_is_skipped_not_fabricated(tmp_path):
    # .env exists but doesn't set the olog vars -- never fabricated.
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": "llm-secret"})
    config = {
        "facility": {},
        "llm": {"api_key_env_var": "CBORG_API_KEY"},
        "modules": {
            "web_terminals": {"image_source": "local"},
            "olog": {
                "enabled": True,
                "username_env_var": "OLOG_USERNAME",
                "password_env_var": "OLOG_PASSWORD",
            },
        },
    }

    result = env_production.ensure_env_production(config, tmp_path)
    generated = env_production.parse_dotenv_file(result)

    assert generated == {"CBORG_API_KEY": "llm-secret", "TZ": "UTC"}


def test_env_production_local_mode_defaults_when_image_source_absent_is_registry(tmp_path):
    """No modules.web_terminals.image_source at all -> defaults to registry
    (fail-closed), so an absent .env.users still raises rather than
    silently generating from a stray .env."""
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)
    config = {"facility": {}, "llm": {}, "modules": {"web_terminals": {}}}

    with pytest.raises(RuntimeError, match="Registry-mode"):
        env_production.ensure_env_production(config, tmp_path)


# ---------------------------------------------------------------------------
# ensure_env_production -- claude_code provider auth-secret coverage. The
# generator must ship the auth secret of every claude_code.provider a web
# container will actually authenticate with (deploy config's own on the
# zero-migration path, each referenced persona project's under a catalog),
# and must fail loudly -- not generate a dead file -- when one is missing.
# ---------------------------------------------------------------------------


def _write_persona_project(tmp_path, name, provider):
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "config.yml").write_text(
        f"project_name: {name}\nclaude_code:\n  provider: {provider}\n", encoding="utf-8"
    )
    return name  # catalog project_path, relative to the deploy project root


def _persona_config(tmp_path, personas: dict[str, str]) -> dict:
    """A local-mode deploy config whose catalog references rendered persona
    projects, one per ``{persona_name: provider}`` entry."""
    catalog = {
        persona: {
            "project": f"{persona}-proj",
            "project_path": _write_persona_project(tmp_path, f"{persona}-proj", provider),
        }
        for persona, provider in personas.items()
    }
    first = next(iter(personas))
    return {
        "facility": {"timezone": "UTC"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_persona": first,
                "personas": catalog,
                "users": [
                    {"name": "alice", "index": 0, "persona": persona} for persona in personas
                ],
            },
        },
    }


def test_env_production_zero_migration_copies_own_claude_code_secret(tmp_path):
    """No persona catalog: the deploy config's own claude_code.provider is what
    the web container runs, so its auth secret is copied -- and required."""
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": "cc-secret"})
    config = {
        "facility": {},
        "claude_code": {"provider": "cborg"},
        "modules": {"web_terminals": {"image_source": "local"}},
    }

    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(config, tmp_path)
    )

    assert generated["CBORG_API_KEY"] == "cc-secret"


def test_env_production_copies_each_persona_projects_claude_code_secret(tmp_path):
    """Persona catalog: every referenced persona project's own provider secret
    ships, even when the deploy config's provider differs."""
    _write_dotenv(
        tmp_path / ".env",
        {"ALS_APG_API_KEY": "persona-secret", "CBORG_API_KEY": "deploy-secret"},
    )
    config = _persona_config(tmp_path, {"operator": "als-apg"})
    config["claude_code"] = {"provider": "cborg"}

    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(config, tmp_path)
    )

    assert generated["ALS_APG_API_KEY"] == "persona-secret"
    # The deploy config's own provider secret is copied too (extra, not required).
    assert generated["CBORG_API_KEY"] == "deploy-secret"


def test_env_production_missing_persona_claude_code_secret_raises_actionably(tmp_path):
    """A referenced persona's provider secret absent from .env must raise --
    naming the var, the provider, and the persona -- never generate a file
    that produces healthy-looking, unauthenticated terminals."""
    _write_dotenv(tmp_path / ".env", {"SOMETHING_ELSE": "x"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with pytest.raises(RuntimeError, match=r"ALS_APG_API_KEY.*\.env") as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    assert "als-apg" in str(excinfo.value)
    assert "operator" in str(excinfo.value)
    assert not (tmp_path / ".env.users").exists()


def test_env_production_deploy_configs_own_secret_not_required_under_catalog(tmp_path):
    """With a persona catalog in play the per-user containers run persona
    projects, so the deploy config's own provider secret is copy-if-present
    but its absence must NOT fail the deploy."""
    _write_dotenv(tmp_path / ".env", {"ALS_APG_API_KEY": "persona-secret"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})
    config["claude_code"] = {"provider": "anthropic"}  # ANTHROPIC_API_KEY not in .env

    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(config, tmp_path)
    )

    assert generated["ALS_APG_API_KEY"] == "persona-secret"
    assert "ANTHROPIC_API_KEY" not in generated


def test_env_production_custom_provider_secret_derived_from_api_providers(tmp_path):
    """A custom proxy provider (defined under api.providers, not built in)
    derives <NAME>_API_KEY -- the same rule the launch-time resolver uses."""
    _write_dotenv(tmp_path / ".env", {"MY_PROXY_API_KEY": "custom-secret"})
    config = {
        "facility": {},
        "api": {"providers": {"my-proxy": {"base_url": "https://proxy.example.org"}}},
        "claude_code": {"provider": "my-proxy"},
        "modules": {"web_terminals": {"image_source": "local"}},
    }

    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(config, tmp_path)
    )

    assert generated["MY_PROXY_API_KEY"] == "custom-secret"


def test_env_production_unknown_provider_is_skipped_not_raised(tmp_path):
    """A provider name known neither to CLAUDE_CODE_PROVIDERS nor to
    api.providers contributes nothing here -- rejecting it is the launch-time
    resolver's job, with its own actionable error."""
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": "x"})
    config = {
        "facility": {},
        "claude_code": {"provider": "frobnicator"},
        "modules": {"web_terminals": {"image_source": "local"}},
    }

    result = env_production.ensure_env_production(config, tmp_path)

    assert result.is_file()


def test_env_production_stale_existing_file_without_credentials_warns(tmp_path, caplog):
    """The never-clobber rule keeps a stale pre-provider-change file in
    service; the deploy must at least say so, naming the missing var."""
    (tmp_path / ".env.users").write_text("TZ=UTC\n", encoding="utf-8")
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with caplog.at_level("WARNING"):
        result = env_production.ensure_env_production(config, tmp_path)

    assert result.read_text(encoding="utf-8") == "TZ=UTC\n"  # still never clobbered
    assert "ALS_APG_API_KEY" in caplog.text
    assert "none of the LLM credential" in caplog.text


def test_env_production_existing_file_with_credential_does_not_warn(tmp_path, caplog):
    (tmp_path / ".env.users").write_text("ALS_APG_API_KEY=ok\n", encoding="utf-8")
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with caplog.at_level("WARNING"):
        env_production.ensure_env_production(config, tmp_path)

    assert "none of the LLM credential" not in caplog.text


def test_env_production_missing_secret_present_in_shell_env_names_the_fix(tmp_path, monkeypatch):
    """The gate deliberately never reads the ambient shell env as a secret
    source -- but when the missing var IS exported there, the error must say
    so and hand the operator the exact copy-in command, instead of leaving
    them to discover the .env-only rule by archaeology."""
    monkeypatch.setenv("ALS_APG_API_KEY", "exported-in-shell")
    _write_dotenv(tmp_path / ".env", {"SOMETHING_ELSE": "x"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with pytest.raises(RuntimeError, match="ALS_APG_API_KEY") as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    message = str(excinfo.value)
    assert "exported in the current shell" in message
    assert f">> {tmp_path / '.env'}" in message
    # The secret VALUE itself must never appear in the error.
    assert "exported-in-shell" not in message


def test_env_production_missing_secret_hint_names_only_the_repo_env(tmp_path, monkeypatch):
    """One secret store, so one remedy — even with a legacy sibling profile recorded.

    This message used to name two files, because a built project's ``.env`` was
    DERIVED from a separate profile directory's: a write to the project copy
    unblocked the deploy and was then dropped by the next build, so the operator
    had to be told both. Under the three-zone layout the profile and the secret
    store share the repo root, so there is one file and a write to it survives.

    Driven with a manifest recording a sibling ``<name>-profile/`` — the retired
    layout — precisely because that is what would resurrect the second path: if
    the hint ever starts naming a sibling directory again, the operator is being
    sent to a file this deploy does not read, and the "dropped by the next
    build" warning would be false besides.
    """
    import json

    profile_dir = tmp_path / "proj-profile"
    profile_dir.mkdir()
    (tmp_path / ".osprey-manifest.json").write_text(
        json.dumps({"build_args": {"profile_path_abs": str(profile_dir / "profile.yml")}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALS_APG_API_KEY", "exported-in-shell")
    _write_dotenv(tmp_path / ".env", {"SOMETHING_ELSE": "x"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with pytest.raises(RuntimeError, match="ALS_APG_API_KEY") as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    message = str(excinfo.value)
    assert f">> {tmp_path / '.env'}" in message
    assert str(profile_dir) not in message
    assert "dropped by the next build" not in message
    assert "exported-in-shell" not in message


def test_env_production_missing_secret_absent_everywhere_has_no_shell_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("ALS_APG_API_KEY", raising=False)
    _write_dotenv(tmp_path / ".env", {"SOMETHING_ELSE": "x"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with pytest.raises(RuntimeError) as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    assert "exported in the current shell" not in str(excinfo.value)


def test_env_production_never_carries_the_dispatcher_token(tmp_path):
    """The tier boundary, pinned. One `.env.production` is handed to EVERY per-user
    container, so a dispatcher bearer here would reach read-only personas too --
    and that credential can fire triggers. The EVENTS panel is wired through the
    PER-USER compose `environment:` block instead (see
    `render_web_terminals(dispatcher_personas=...)`); this asserts the shared file
    stays clean even when the operator's .env is full of service tokens."""
    # Arrange
    _write_dotenv(
        tmp_path / ".env",
        {
            "ALS_APG_API_KEY": "cc-secret",
            "EVENT_DISPATCHER_TOKEN": "fire-any-trigger",
            "DISPATCH_WORKER_TOKEN": "worker",
            "BLUESKY_LAUNCH_TOKEN": "arm-the-queue",
            "GRAPHDB_PASSWORD": "rewrite-the-graph",
        },
    )
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    # Act
    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(config, tmp_path)
    )

    # Assert — the LLM credential crosses, the service tokens never do
    assert generated["ALS_APG_API_KEY"] == "cc-secret"
    for service_token in (
        "EVENT_DISPATCHER_TOKEN",
        "DISPATCH_WORKER_TOKEN",
        "BLUESKY_LAUNCH_TOKEN",
        # The graphdb store's only credential, and a write-capable one. It is
        # granted per-persona through the compose `environment:` block for a
        # different reason than the tier boundary -- the read-only tier is meant
        # to have it -- but it stays out of this rosterwide file all the same,
        # which would otherwise hand it to personas configuring no graph store.
        "GRAPHDB_PASSWORD",
    ):
        assert service_token not in generated


# ---------------------------------------------------------------------------
# ensure_env_production -- env-chain derivation. The generator reads the whole
# chain (.env.shared then .env, later winning), not the root .env alone, so a
# key the committed defaults carry reaches the per-user containers and a
# required auth var living only in the shared half is not a hard error.
# ---------------------------------------------------------------------------


def test_env_production_without_shared_matches_the_local_env_alone(tmp_path):
    """The no-.env.shared shape is the pre-chain shape, byte for byte.

    Pinned as literal bytes rather than a parsed dict: the chain change must be
    invisible to every deployment that never adopts .env.shared, and only the
    exact file content can say that.
    """
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result.read_text(encoding="utf-8") == (
        "CBORG_API_KEY=llm-secret\n"
        "OLOG_USERNAME=olog-user\n"
        "OLOG_PASSWORD=olog-pass\n"
        "CONFLUENCE_ACCESS_TOKEN=wiki-secret\n"
        "ARIEL_DSN=postgresql://ariel:ariel@ariel-postgres:5432/ariel\n"
        "TZ=America/Los_Angeles\n"
    )


def test_env_production_copies_a_key_only_the_shared_half_sets(tmp_path):
    """A credential the committed defaults carry is as real a source for a web
    terminal as one the host-local .env carries."""
    _write_dotenv(tmp_path / ".env.shared", {"CONFLUENCE_ACCESS_TOKEN": "wiki-from-shared"})
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": "llm-secret"})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    generated = env_production.parse_dotenv_file(result)
    assert generated["CONFLUENCE_ACCESS_TOKEN"] == "wiki-from-shared"
    assert generated["CBORG_API_KEY"] == "llm-secret"


def test_env_production_local_env_wins_over_shared_on_a_shared_key(tmp_path):
    _write_dotenv(tmp_path / ".env.shared", {"CBORG_API_KEY": "from-shared"})
    _write_dotenv(tmp_path / ".env", {"CBORG_API_KEY": "from-local"})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert env_production.parse_dotenv_file(result)["CBORG_API_KEY"] == "from-local"


def test_env_production_generates_from_the_shared_half_alone(tmp_path):
    """No host-local .env at all: the chain is still non-empty, so there is
    something to derive from and the deploy is not refused."""
    _write_dotenv(tmp_path / ".env.shared", {"CBORG_API_KEY": "llm-secret"})

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert env_production.parse_dotenv_file(result)["CBORG_API_KEY"] == "llm-secret"


def test_env_production_required_auth_secret_in_shared_half_does_not_raise(tmp_path):
    """The refusal asks whether the MERGED chain sets the var, so a provider
    secret kept in the shared defaults produces authenticated terminals rather
    than a refused deploy."""
    _write_dotenv(tmp_path / ".env.shared", {"ALS_APG_API_KEY": "shared-secret"})
    _write_dotenv(tmp_path / ".env", {"SOMETHING_ELSE": "x"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    result = env_production.ensure_env_production(config, tmp_path)

    assert env_production.parse_dotenv_file(result)["ALS_APG_API_KEY"] == "shared-secret"


def test_env_production_missing_from_the_whole_chain_still_raises(tmp_path):
    """Present in neither half: still a refusal, and the message names both
    files it read plus the ONE file to add the variable to -- the host-local
    .env, since the shared half is committed and holds no secrets."""
    _write_dotenv(tmp_path / ".env.shared", {"SOMETHING_SHARED": "y"})
    _write_dotenv(tmp_path / ".env", {"SOMETHING_ELSE": "x"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with pytest.raises(RuntimeError, match="ALS_APG_API_KEY") as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    message = str(excinfo.value)
    assert str(tmp_path / ".env.shared") in message
    assert f"variable(s) to {tmp_path / '.env'}" in message
    assert not (tmp_path / ".env.users").exists()


def test_env_production_neither_chain_file_present_raises_naming_both(tmp_path):
    with pytest.raises(RuntimeError, match=r"\.env\.shared, \.env"):
        env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert not (tmp_path / ".env.users").exists()


def test_env_production_quotes_a_value_that_would_not_survive_a_re_read(tmp_path):
    """Written through format_env_line, so a value whose boundaries are
    whitespace arrives at the container intact instead of stripped."""
    (tmp_path / ".env").write_text('CBORG_API_KEY="  padded-secret  "\n', encoding="utf-8")

    result = env_production.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert 'CBORG_API_KEY="  padded-secret  "' in result.read_text(encoding="utf-8")
    assert env_production.parse_dotenv_file(result)["CBORG_API_KEY"] == "  padded-secret  "


# ---------------------------------------------------------------------------
# Observability-store credentials. The store's account NAME crosses into
# .env.users; its admin PASSWORD never does, because one file is handed to
# every persona alike and admin access reads every transcript in the store.
# ---------------------------------------------------------------------------


#: How the shipped telemetry block spells its two credentials: reference
#: literals carrying their own fallbacks, not env-var NAMES the way
#: ``llm.api_key_env_var`` does.
_SHIPPED_STORE_USER = "${ZO_ROOT_USER_EMAIL:-root@example.com}"
_SHIPPED_STORE_PASSWORD = "${ZO_ROOT_USER_PASSWORD:-Complexpass#123}"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("${ZO_ROOT_USER_PASSWORD}", ("ZO_ROOT_USER_PASSWORD", False)),
        (_SHIPPED_STORE_PASSWORD, ("ZO_ROOT_USER_PASSWORD", True)),
        (_SHIPPED_STORE_USER, ("ZO_ROOT_USER_EMAIL", True)),
        ("root@example.com", None),  # a plain literal references nothing
        ("$ZO_ROOT_USER_PASSWORD", None),  # unbraced: outside the dialect
        ("", None),
        (None, None),
        (5080, None),
    ],
)
def test_env_reference_tells_the_two_reference_forms_from_a_literal(value, expected):
    """The whole point of the matcher: a bare reference and a defaulted one
    fail differently in a container, and neither looks like a plain value."""
    assert env_production._env_reference(value) == expected


def _write_telemetry_persona(
    tmp_path, name="operator-proj", *, provider="anthropic", **credentials
):
    """Write a rendered persona project carrying a telemetry block.

    ``credentials`` are placed under ``claude_code.telemetry.openobserve``
    verbatim, so a test can spell each one the way a real config would.
    """
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "project_name": name,
                "claude_code": {
                    "provider": provider,
                    "telemetry": {
                        "enabled": True,
                        "backend": "openobserve",
                        "openobserve": {"org": "default", **credentials},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return name  # catalog project_path, relative to the deploy project root


def _catalog_config(project_path, persona="operator"):
    """A local-mode deploy config whose roster runs one catalogued persona."""
    return {
        "facility": {"timezone": "UTC"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": "local",
                "default_persona": persona,
                "personas": {persona: {"project": project_path, "project_path": project_path}},
                "users": [{"name": "alice", "index": 0, "persona": persona}],
            },
        },
    }


def test_env_production_never_carries_the_store_admin_password(tmp_path):
    """The security decision this category exists to make.

    ZO_ROOT_USER_PASSWORD is the observability store's single admin credential,
    and .env.users is handed to every persona alike -- a copy here would grant
    every read-only terminal admin read of every transcript in the store. The
    account NAME is not a secret and does cross, which is what makes the
    omission of the password a decision rather than an oversight.
    """
    _write_dotenv(
        tmp_path / ".env",
        {
            "ANTHROPIC_API_KEY": "cc-secret",
            "ZO_ROOT_USER_EMAIL": "store-account@example.org",
            "ZO_ROOT_USER_PASSWORD": "store-admin-secret",
        },
    )
    config = _catalog_config(
        _write_telemetry_persona(
            tmp_path, user=_SHIPPED_STORE_USER, password=_SHIPPED_STORE_PASSWORD
        )
    )

    result = env_production.ensure_env_production(config, tmp_path)

    generated = env_production.parse_dotenv_file(result)
    raw_text = result.read_text(encoding="utf-8")
    assert generated["ZO_ROOT_USER_EMAIL"] == "store-account@example.org"
    assert "ZO_ROOT_USER_PASSWORD" not in generated
    assert "ZO_ROOT_USER_PASSWORD" not in raw_text
    assert "store-admin-secret" not in raw_text


def test_env_production_omits_the_store_password_a_config_declares_it_needs(tmp_path):
    """Even a config asserting the variable is set -- a bare reference, with no
    fallback of its own -- and a chain that sets it: the requirement is
    REPORTED, never satisfied by copying the credential into this file."""
    _write_dotenv(
        tmp_path / ".env",
        {
            "ANTHROPIC_API_KEY": "cc-secret",
            "ZO_ROOT_USER_PASSWORD": "store-admin-secret",
        },
    )
    config = _catalog_config(
        _write_telemetry_persona(tmp_path, password="${ZO_ROOT_USER_PASSWORD}")
    )

    result = env_production.ensure_env_production(config, tmp_path)

    raw_text = result.read_text(encoding="utf-8")
    assert "ZO_ROOT_USER_PASSWORD" not in raw_text
    assert "store-admin-secret" not in raw_text


def test_env_production_copies_the_store_account_name_when_the_chain_sets_it(tmp_path):
    """A fixed-key enumerated category, like TZ and ARIEL_DSN: present in the
    chain, copied; absent, silently skipped."""
    _write_dotenv(
        tmp_path / ".env",
        {**_INCLUDED_ENV, "ZO_ROOT_USER_EMAIL": "store-account@example.org"},
    )

    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(_FULL_CONFIG, tmp_path)
    )

    assert generated["ZO_ROOT_USER_EMAIL"] == "store-account@example.org"


def test_env_production_omits_the_store_account_name_when_the_chain_lacks_it(tmp_path):
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)

    generated = env_production.parse_dotenv_file(
        env_production.ensure_env_production(_FULL_CONFIG, tmp_path)
    )

    assert "ZO_ROOT_USER_EMAIL" not in generated


def test_env_production_bare_telemetry_password_absent_from_the_chain_refuses(tmp_path):
    """A bare reference resolves to nothing at all inside the container -- the
    placeholder reaches the store verbatim -- so the deploy is refused rather
    than generated, naming the variable, where it is asked for, and why this
    file will not be carrying it either way."""
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": "cc-secret"})
    config = _catalog_config(
        _write_telemetry_persona(tmp_path, password="${ZO_ROOT_USER_PASSWORD}")
    )

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    message = str(excinfo.value)
    assert "claude_code.telemetry.openobserve.password" in message
    assert "operator" in message
    assert "single admin credential" in message
    assert not (tmp_path / ".env.users").exists()


def test_env_production_bare_telemetry_password_in_the_deploy_config_refuses_too(tmp_path):
    """The deploy config is read the same way a persona project is: on the
    zero-migration path it IS the project the web image runs."""
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": "cc-secret"})
    config = {
        "facility": {},
        "claude_code": {
            "provider": "anthropic",
            "telemetry": {"openobserve": {"password": "${ZO_ROOT_USER_PASSWORD}"}},
        },
        "modules": {"web_terminals": {"image_source": "local"}},
    }

    with pytest.raises(RuntimeError, match="ZO_ROOT_USER_PASSWORD") as excinfo:
        env_production.ensure_env_production(config, tmp_path)

    assert "deploy config" in str(excinfo.value)


def test_env_production_defaulted_telemetry_password_is_not_required(tmp_path):
    """The shipped spelling carries its own fallback, so it asks nothing of the
    env chain and must not turn a working deploy into a refusal."""
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": "cc-secret"})
    config = _catalog_config(_write_telemetry_persona(tmp_path, password=_SHIPPED_STORE_PASSWORD))

    result = env_production.ensure_env_production(config, tmp_path)

    assert result.is_file()


def test_env_production_a_literal_telemetry_password_asks_nothing_of_the_chain(tmp_path):
    """A hand-written password is a value, not a reference: nothing to require,
    and nothing this file copies either."""
    _write_dotenv(tmp_path / ".env", {"ANTHROPIC_API_KEY": "cc-secret"})
    config = _catalog_config(_write_telemetry_persona(tmp_path, password="hand-written"))

    result = env_production.ensure_env_production(config, tmp_path)

    assert "hand-written" not in result.read_text(encoding="utf-8")


def test_env_production_existing_file_without_the_store_account_name_warns(tmp_path, caplog):
    """The advisory arm for the never-clobbered file: it names the variable the
    config's telemetry references and nothing about any value."""
    (tmp_path / ".env.users").write_text("ANTHROPIC_API_KEY=ok\nTZ=UTC\n", encoding="utf-8")
    config = _catalog_config(_write_telemetry_persona(tmp_path, user=_SHIPPED_STORE_USER))

    with caplog.at_level("WARNING"):
        env_production.ensure_env_production(config, tmp_path)

    assert "ZO_ROOT_USER_EMAIL" in caplog.text
    assert "claude_code.telemetry.openobserve.user" in caplog.text


def test_env_production_the_two_advisory_arms_are_evaluated_independently(tmp_path, caplog):
    """A file carrying the LLM credential satisfies the all-or-nothing arm and
    still gets the telemetry advisory; the LLM arm stays silent."""
    (tmp_path / ".env.users").write_text("ANTHROPIC_API_KEY=ok\n", encoding="utf-8")
    config = _catalog_config(_write_telemetry_persona(tmp_path, user=_SHIPPED_STORE_USER))

    with caplog.at_level("WARNING"):
        env_production.ensure_env_production(config, tmp_path)

    assert "none of the LLM credential" not in caplog.text
    assert "ZO_ROOT_USER_EMAIL" in caplog.text


def test_env_production_existing_file_with_the_store_account_name_does_not_warn(tmp_path, caplog):
    (tmp_path / ".env.users").write_text(
        "ANTHROPIC_API_KEY=ok\nZO_ROOT_USER_EMAIL=store-account@example.org\n",
        encoding="utf-8",
    )
    config = _catalog_config(_write_telemetry_persona(tmp_path, user=_SHIPPED_STORE_USER))

    with caplog.at_level("WARNING"):
        env_production.ensure_env_production(config, tmp_path)

    assert "ZO_ROOT_USER_EMAIL" not in caplog.text


def test_env_production_no_telemetry_block_means_no_telemetry_advisory(tmp_path, caplog):
    """A config that references no store account gets no advisory about one."""
    (tmp_path / ".env.users").write_text("ANTHROPIC_API_KEY=ok\n", encoding="utf-8")
    config = _catalog_config(_write_telemetry_persona(tmp_path))

    with caplog.at_level("WARNING"):
        env_production.ensure_env_production(config, tmp_path)

    assert "observability account" not in caplog.text
