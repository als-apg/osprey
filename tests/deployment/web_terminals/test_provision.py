"""Unit tests for the web-terminal host-side provisioning module.

Covers ``osprey.deployment.web_terminals.provision`` in isolation: the
rootless-podman linger step, ``.env.production`` generation, the per-persona
local image builder, on-demand persona project rendering, and the advisory
post-up verify script. The deploy_up-entry orchestration that wires these
together lives in ``tests/deployment/test_container_lifecycle.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.deployment.web_terminals import provision

# ---------------------------------------------------------------------------
# _enable_linger -- rootless-podman persistence via loginctl
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_enable_linger_skips_on_docker_runtime(monkeypatch):
    """Docker has no per-user systemd session -- linger never applies there."""
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(
        provision.shutil,
        "which",
        lambda name: pytest.fail("loginctl should not be probed for docker"),
    )
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: calls.append(a))

    provision._enable_linger({}, {})

    assert calls == []


def test_enable_linger_skips_when_loginctl_absent(monkeypatch):
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: None)
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: calls.append(a))

    provision._enable_linger({}, {})

    assert calls == []


def test_enable_linger_noop_when_already_enabled(monkeypatch):
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(provision.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd == ["loginctl", "show-user", "deployuser", "--property=Linger"]
        return _FakeCompletedProcess(returncode=0, stdout="Linger=yes\n")

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._enable_linger({}, {})

    # Only the status check ran -- enable-linger is never invoked once we
    # already know it's on, so a no-op deploy stays quiet.
    assert len(calls) == 1


def test_enable_linger_enables_when_not_yet_enabled(monkeypatch):
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(provision.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._enable_linger({}, {})

    assert calls == [
        ["loginctl", "show-user", "deployuser", "--property=Linger"],
        ["loginctl", "enable-linger", "deployuser"],
    ]


def test_enable_linger_enable_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(provision.getpass, "getuser", lambda: "deployuser")

    def _fake_run(cmd, **kwargs):
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="Permission denied")

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._enable_linger({}, {})  # must not raise


def test_enable_linger_status_check_error_still_attempts_enable(monkeypatch):
    """A broken status check (loginctl show-user itself errors) must not
    prevent the enable attempt -- only a confirmed already-enabled state
    short-circuits it."""
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(provision.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            raise OSError("boom")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._enable_linger({}, {})  # must not raise

    assert calls[-1] == ["loginctl", "enable-linger", "deployuser"]


def test_enable_linger_enable_call_error_does_not_raise(monkeypatch):
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(provision.getpass, "getuser", lambda: "deployuser")

    def _fake_run(cmd, **kwargs):
        if "show-user" in cmd:
            return _FakeCompletedProcess(returncode=0, stdout="Linger=no\n")
        raise OSError("no systemd")

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._enable_linger({}, {})  # must not raise


def test_enable_linger_getuser_keyerror_does_not_raise(monkeypatch):
    """getpass.getuser() falls back to pwd.getpwuid(os.getuid()) when
    USER/LOGNAME/LNAME/USERNAME are all unset, which raises KeyError (<=3.12)
    or OSError (3.13+) for a uid with no passwd entry -- e.g. an LDAP/NSS
    user under a stripped-env systemd/cron context. That must be caught
    here, not propagate through the post-up hook and abort the deploy after
    `up -d` already succeeded."""
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")

    def _raise_keyerror():
        raise KeyError("getpwuid(): uid not found: 1234")

    monkeypatch.setattr(provision.getpass, "getuser", _raise_keyerror)
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: calls.append(a))

    provision._enable_linger({}, {})  # must not raise

    assert calls == []  # no loginctl call was ever attempted


def test_enable_linger_status_check_timeout_still_attempts_enable(monkeypatch):
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["podman", "compose"])
    monkeypatch.setattr(provision.shutil, "which", lambda name: "/usr/bin/loginctl")
    monkeypatch.setattr(provision.getpass, "getuser", lambda: "deployuser")
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "show-user" in cmd:
            raise provision.subprocess.TimeoutExpired(cmd=cmd, timeout=10)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._enable_linger({}, {})  # must not raise

    assert calls[-1] == ["loginctl", "enable-linger", "deployuser"]


# ---------------------------------------------------------------------------
# ensure_env_production -- module-conditional CI-subset generator for
# local-mode web-terminal deploys.
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
        "event_dispatcher": {
            "enabled": True,
            "token_env_var": "EVENT_DISPATCHER_TOKEN",
            "sidecar_token_env_var": "DISPATCH_SIDECAR_TOKEN",
        },
        "ariel": {
            "enabled": True,
            "dsn": "postgresql://ariel:ariel@ariel-postgres:5432/ariel",
        },
    },
}

# Every secret .env.production must NEVER contain, keyed by the config path
# that names it -- the exclusion list is the security spec for this task.
_EXCLUDED_ENV = {
    "TEST_CI_TOKEN": "ci-secret",
    "TEST_REGISTRY_TOKEN": "registry-secret",
    "BEAM_VIEWER_DEPLOY_TOKEN": "external-project-secret",
    "DISPATCH_SIDECAR_TOKEN": "sidecar-secret",
}

_INCLUDED_ENV = {
    "CBORG_API_KEY": "llm-secret",
    "OLOG_USERNAME": "olog-user",
    "OLOG_PASSWORD": "olog-pass",
    "CONFLUENCE_ACCESS_TOKEN": "wiki-secret",
    "EVENT_DISPATCHER_TOKEN": "dispatcher-secret",
}


def test_env_production_present_returned_as_is(tmp_path):
    marker = "# operator-authored, do not touch\nFOO=bar\n"
    (tmp_path / ".env.production").write_text(marker, encoding="utf-8")

    result = provision.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result == tmp_path / ".env.production"
    assert result.read_text(encoding="utf-8") == marker


def test_env_production_present_in_registry_mode_returned_as_is(tmp_path):
    marker = "FOO=bar\n"
    (tmp_path / ".env.production").write_text(marker, encoding="utf-8")
    config = {**_FULL_CONFIG, "modules": {**_FULL_CONFIG["modules"], "web_terminals": {}}}

    result = provision.ensure_env_production(config, tmp_path)

    assert result.read_text(encoding="utf-8") == marker


def test_env_production_neither_present_raises_actionably(tmp_path):
    with pytest.raises(RuntimeError, match=r"\.env\.production.*\.env"):
        provision.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert not (tmp_path / ".env.production").exists()


def test_env_production_registry_mode_never_generates_even_with_env_present(tmp_path):
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})
    config = {**_FULL_CONFIG, "modules": {**_FULL_CONFIG["modules"], "web_terminals": {}}}

    with pytest.raises(RuntimeError, match="Registry-mode"):
        provision.ensure_env_production(config, tmp_path)

    assert not (tmp_path / ".env.production").exists()


def test_env_production_local_mode_generates_from_env(tmp_path):
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = provision.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert result == tmp_path / ".env.production"
    generated = provision.parse_dotenv_file(result)

    # Included: llm key, module-gated olog/wiki/dispatcher, ARIEL_DSN, TZ.
    assert generated["CBORG_API_KEY"] == "llm-secret"
    assert generated["OLOG_USERNAME"] == "olog-user"
    assert generated["OLOG_PASSWORD"] == "olog-pass"
    assert generated["CONFLUENCE_ACCESS_TOKEN"] == "wiki-secret"
    assert generated["EVENT_DISPATCHER_TOKEN"] == "dispatcher-secret"
    assert generated["ARIEL_DSN"] == "postgresql://ariel:ariel@ariel-postgres:5432/ariel"
    assert generated["TZ"] == "America/Los_Angeles"


def test_env_production_never_includes_excluded_secrets(tmp_path):
    """The security spec: registry token, sidecar token, and external-project
    tokens must never appear in the generated file -- neither their key nor
    their value, even though the source .env contains all of them."""
    _write_dotenv(tmp_path / ".env", {**_INCLUDED_ENV, **_EXCLUDED_ENV})

    result = provision.ensure_env_production(_FULL_CONFIG, tmp_path)

    generated = provision.parse_dotenv_file(result)
    raw_text = result.read_text(encoding="utf-8")

    for excluded_key, excluded_value in _EXCLUDED_ENV.items():
        assert excluded_key not in generated
        assert excluded_key not in raw_text
        assert excluded_value not in raw_text

    # And the CI/registry token vars named in config are also never copied,
    # confirming the omission is by construction, not incidental.
    assert "TEST_CI_TOKEN" not in generated
    assert "TEST_REGISTRY_TOKEN" not in generated


def test_env_production_generated_file_is_mode_0600(tmp_path):
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)

    result = provision.ensure_env_production(_FULL_CONFIG, tmp_path)

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
        if str(path).endswith(".env.production"):
            captured["flags"] = flags
            captured["mode"] = mode
        return real_open(path, flags, mode)

    monkeypatch.setattr(provision.os, "open", _spy_open)

    provision.ensure_env_production(_FULL_CONFIG, tmp_path)

    assert captured, "os.open was never called for .env.production"
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
            "event_dispatcher": {"enabled": False, "token_env_var": "EVENT_DISPATCHER_TOKEN"},
            "ariel": {"enabled": False, "dsn": "postgresql://ariel:ariel@ariel-postgres/ariel"},
        },
    }

    result = provision.ensure_env_production(config, tmp_path)
    generated = provision.parse_dotenv_file(result)

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

    result = provision.ensure_env_production(config, tmp_path)
    generated = provision.parse_dotenv_file(result)

    assert generated == {"CBORG_API_KEY": "llm-secret", "TZ": "UTC"}


def test_env_production_local_mode_defaults_when_image_source_absent_is_registry(tmp_path):
    """No modules.web_terminals.image_source at all -> defaults to registry
    (fail-closed), so an absent .env.production still raises rather than
    silently generating from a stray .env."""
    _write_dotenv(tmp_path / ".env", _INCLUDED_ENV)
    config = {"facility": {}, "llm": {}, "modules": {"web_terminals": {}}}

    with pytest.raises(RuntimeError, match="Registry-mode"):
        provision.ensure_env_production(config, tmp_path)


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

    generated = provision.parse_dotenv_file(provision.ensure_env_production(config, tmp_path))

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

    generated = provision.parse_dotenv_file(provision.ensure_env_production(config, tmp_path))

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
        provision.ensure_env_production(config, tmp_path)

    assert "als-apg" in str(excinfo.value)
    assert "operator" in str(excinfo.value)
    assert not (tmp_path / ".env.production").exists()


def test_env_production_deploy_configs_own_secret_not_required_under_catalog(tmp_path):
    """With a persona catalog in play the per-user containers run persona
    projects, so the deploy config's own provider secret is copy-if-present
    but its absence must NOT fail the deploy."""
    _write_dotenv(tmp_path / ".env", {"ALS_APG_API_KEY": "persona-secret"})
    config = _persona_config(tmp_path, {"operator": "als-apg"})
    config["claude_code"] = {"provider": "anthropic"}  # ANTHROPIC_API_KEY not in .env

    generated = provision.parse_dotenv_file(provision.ensure_env_production(config, tmp_path))

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

    generated = provision.parse_dotenv_file(provision.ensure_env_production(config, tmp_path))

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

    result = provision.ensure_env_production(config, tmp_path)

    assert result.is_file()


def test_env_production_stale_existing_file_without_credentials_warns(tmp_path, caplog):
    """The never-clobber rule keeps a stale pre-provider-change file in
    service; the deploy must at least say so, naming the missing var."""
    (tmp_path / ".env.production").write_text("TZ=UTC\n", encoding="utf-8")
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with caplog.at_level("WARNING"):
        result = provision.ensure_env_production(config, tmp_path)

    assert result.read_text(encoding="utf-8") == "TZ=UTC\n"  # still never clobbered
    assert "ALS_APG_API_KEY" in caplog.text
    assert "none of the LLM credential" in caplog.text


def test_env_production_existing_file_with_credential_does_not_warn(tmp_path, caplog):
    (tmp_path / ".env.production").write_text("ALS_APG_API_KEY=ok\n", encoding="utf-8")
    config = _persona_config(tmp_path, {"operator": "als-apg"})

    with caplog.at_level("WARNING"):
        provision.ensure_env_production(config, tmp_path)

    assert "none of the LLM credential" not in caplog.text


# ---------------------------------------------------------------------------
# _warn_if_web_stack_unreachable -- advisory post-up host-reachability probe
# (the Docker Desktop network_mode:host trap: healthy stack, unreachable host).
# ---------------------------------------------------------------------------

_PROBE_CONFIG = {"modules": {"web_terminals": {"enabled": True, "nginx_port": 9080}}}


def test_web_stack_reachable_no_warning(monkeypatch, caplog):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(provision.urllib.request, "urlopen", lambda url, timeout: _Resp())

    with caplog.at_level("WARNING"):
        provision._warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "not reachable" not in caplog.text


def test_web_stack_unreachable_warns_with_docker_desktop_hint(monkeypatch, caplog):
    def _refuse(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(provision.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(provision.sys, "platform", "darwin")
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])

    with caplog.at_level("WARNING"):
        provision._warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "http://127.0.0.1:9080/" in caplog.text
    assert "Enable host networking" in caplog.text


def test_web_stack_unreachable_on_linux_warns_without_desktop_hint(monkeypatch, caplog):
    def _refuse(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(provision.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(provision.sys, "platform", "linux")
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])

    with caplog.at_level("WARNING"):
        provision._warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "not reachable" in caplog.text
    assert "Enable host networking" not in caplog.text


def test_web_stack_http_error_counts_as_reachable(monkeypatch, caplog):
    def _http_error(url, timeout):
        raise provision.urllib.error.HTTPError(url, 502, "Bad Gateway", None, None)

    monkeypatch.setattr(provision.urllib.request, "urlopen", _http_error)

    with caplog.at_level("WARNING"):
        provision._warn_if_web_stack_unreachable(_PROBE_CONFIG, attempts=2, delay=0)

    assert "not reachable" not in caplog.text


# ---------------------------------------------------------------------------
# build_persona_images -- local-mode per-persona image builder
# ---------------------------------------------------------------------------


def _make_persona_project(tmp_path, name, cli_version=None):
    """Create a minimal persona project dir with a Dockerfile + config.yml."""
    project_dir = tmp_path / name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    if cli_version is not None:
        (project_dir / "config.yml").write_text(
            f"claude_code:\n  cli_version: {cli_version!r}\n", encoding="utf-8"
        )
    else:
        (project_dir / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    return str(project_dir)


@pytest.fixture
def _no_dev_wheel_staging(monkeypatch):
    """Stub out the dev-wheel staging collaborator (its own coverage lives with
    _build_project_image's tests) so build_persona_images tests never touch a
    real wheel build. Reports SUCCESS (True) — the OSPREY_DEV build-arg is
    keyed on staging success, so simulating a successful staging keeps the
    dev-path assertions meaningful; the failure path has its own test."""
    monkeypatch.setattr(provision, "_copy_local_framework_for_override", lambda project_root: True)


def test_build_persona_images_noop_in_registry_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: calls.append(a))
    config = {"modules": {"web_terminals": {"image_source": "registry"}}}

    provision.build_persona_images(config, [{"persona": "ops"}], False, {})

    assert calls == []


def test_build_persona_images_local_without_catalog_raises(tmp_path):
    config = {"modules": {"web_terminals": {"image_source": "local"}}}

    with pytest.raises(ValueError, match="requires both"):
        provision.build_persona_images(config, [], False, {})


def test_build_persona_images_local_without_default_persona_raises(tmp_path):
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "personas": {"ops": {"project": "ops-app", "project_path": str(tmp_path)}},
            }
        }
    }

    with pytest.raises(ValueError, match="requires both"):
        provision.build_persona_images(config, [], False, {})


def test_build_persona_images_builds_each_referenced_persona_once(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    sci_path = _make_persona_project(tmp_path, "sci-app")

    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {"project": "ops-app", "project_path": ops_path},
                    "sci": {"project": "sci-app", "project_path": sci_path},
                },
            }
        },
    }
    resolved_users = [
        {"name": "alice", "persona": "ops", "project": "ops-app"},
        {"name": "bob", "persona": "ops", "project": "ops-app"},  # shares ops -- must not rebuild
        {"name": "carol", "persona": "sci", "project": "sci-app"},
    ]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, False, {})

    assert len(calls) == 2  # one build per DISTINCT persona, not per user

    ops_cmd = next(c for c in calls if "ops-app-ops:local" in c)
    sci_cmd = next(c for c in calls if "sci-app-sci:local" in c)

    assert ops_cmd[0] == "docker"
    assert "-f" in ops_cmd
    assert os.path.join(ops_path, "Dockerfile") == ops_cmd[ops_cmd.index("-f") + 1]
    assert ops_path == ops_cmd[-1]  # context is project_path
    assert "--label" in ops_cmd
    assert "com.osprey.project=myfacility" in ops_cmd

    assert "com.osprey.project=myfacility" in sci_cmd
    assert sci_path == sci_cmd[-1]


def test_build_persona_images_never_builds_zero_migration_entries(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """An entry with persona=None (no persona system in effect) is skipped --
    it never contributes a build unit, even in local mode."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "legacy", "persona": None, "project": "myfacility-assistant"}]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, False, {})

    assert calls == []


def test_build_persona_images_includes_cli_version_from_persona_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version="2.1.99")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "CLAUDE_CLI_VERSION=2.1.99" in cmd


def test_build_persona_images_omits_cli_version_when_unset_in_persona_config(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """The persona's own config.yml has no claude_code.cli_version -- the
    build-arg must be omitted entirely (never falls back to the framework
    default the facility/dispatch-worker path uses)."""
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any(str(arg).startswith("CLAUDE_CLI_VERSION=") for arg in cmd)
    # The facility config's own claude_code.cli_version (if any) must never
    # leak into a persona build either -- there is none set here, but the
    # generic OSPREY_PIP_SPEC build-arg is still present.
    assert any(str(arg).startswith("OSPREY_PIP_SPEC=") for arg in cmd)


def test_build_persona_images_never_reads_facility_cli_version(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """A claude_code.cli_version set on the FACILITY config must never leak
    into a persona build -- only the persona's own project_path/config.yml is
    consulted."""
    ops_path = _make_persona_project(tmp_path, "ops-app", cli_version=None)
    config = {
        "project_name": "myfacility",
        "claude_code": {"cli_version": "9.9.9"},  # facility-level pin -- must be ignored
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert not any("9.9.9" in str(arg) for arg in cmd)


def test_build_persona_images_dev_mode_adds_osprey_dev_build_arg(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Under --dev the persona build argv carries OSPREY_DEV=1 (mirroring the
    dispatch-worker project-image dev path)."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, True, {})

    (cmd,) = calls
    assert "OSPREY_DEV=1" in cmd
    assert cmd[cmd.index("OSPREY_DEV=1") - 1] == "--build-arg"


def test_build_persona_images_dev_mode_omits_osprey_dev_when_staging_fails(monkeypatch, tmp_path):
    """--dev with a FAILED wheel staging must build WITHOUT OSPREY_DEV: the
    pin-relaxing arg would otherwise silently install the latest published
    release instead of the local code the flag promises (fail-closed)."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(provision, "_copy_local_framework_for_override", lambda project_root: False)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, True, {})

    (cmd,) = calls  # the image is still built -- just without the dev relaxation
    assert "OSPREY_DEV=1" not in cmd


def test_build_persona_images_non_dev_omits_osprey_dev_build_arg(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, resolved_users, False, {})

    (cmd,) = calls
    assert "OSPREY_DEV=1" not in cmd


def test_build_persona_images_dev_mode_stages_and_cleans_wheel(monkeypatch, tmp_path):
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
        (Path(project_root) / "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
        return True

    monkeypatch.setattr(provision, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: None)

    provision.build_persona_images(config, resolved_users, True, {})

    # Staged artifacts (wheel AND its requirements manifest) must be cleaned
    # up after the build so neither can poison a later non-dev build.
    assert list(Path(ops_path).glob("*.whl")) == []
    assert not (Path(ops_path) / "osprey-local-requirements.txt").exists()


def test_build_persona_images_dev_mode_cleans_staged_artifacts_on_build_failure(
    monkeypatch, tmp_path
):
    """The persona cleanup runs in a finally: a failing image build must still
    remove the staged wheel + manifest from the persona's context."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }
    resolved_users = [{"name": "alice", "persona": "ops", "project": "ops-app"}]

    def _fake_stage(project_root):
        (Path(project_root) / "osprey_framework-0.0.0-py3-none-any.whl").write_text("wheel")
        (Path(project_root) / "osprey-local-requirements.txt").write_text("softioc>=4.5\n")
        return True

    def _failing_build(cmd, **k):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(provision, "_copy_local_framework_for_override", _fake_stage)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    monkeypatch.setattr(provision.subprocess, "run", _failing_build)

    with pytest.raises(subprocess.CalledProcessError):
        provision.build_persona_images(config, resolved_users, True, {})

    assert list(Path(ops_path).glob("*.whl")) == []
    assert not (Path(ops_path) / "osprey-local-requirements.txt").exists()


def test_build_persona_images_no_referenced_personas_runs_no_build(
    monkeypatch, tmp_path, _no_dev_wheel_staging
):
    """Local mode + catalog + default_persona configured, but resolved_users
    references no catalog entry (e.g. empty roster) -- no-op, no crash."""
    ops_path = _make_persona_project(tmp_path, "ops-app")
    config = {
        "project_name": "myfacility",
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": {"project": "ops-app", "project_path": ops_path}},
            }
        },
    }

    monkeypatch.setattr(provision, "get_runtime_command", lambda config: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision.build_persona_images(config, [], False, {})

    assert calls == []


# ---------------------------------------------------------------------------
# _auto_render_missing_personas -- render a referenced persona's project on
# demand when its project_path directory is absent, BEFORE build_persona_images
# builds its image. Renders network-free (--skip-deps), never overwrites a
# complete (user-owned) render, and hard-errors on a partial render or a
# missing build_profile.
# ---------------------------------------------------------------------------


def _auto_render_config(tmp_path, **persona_overrides):
    """A local-mode config whose single persona 'ops' renders to <tmp_path>/ops-app.

    Defaults to a usable build_profile so the render path is exercised; pass
    ``build_profile=None`` to drop it.
    """
    persona = {
        "project": "ops-app",
        "project_path": str(tmp_path / "ops-app"),
        "build_profile": "control-assistant",
    }
    persona.update(persona_overrides)
    return {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {"ops": persona},
            }
        }
    }


_AUTO_RENDER_USERS = [{"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"}]


def test_auto_render_renders_when_project_path_missing(monkeypatch, tmp_path):
    """No directory at project_path -> exactly one `osprey build` render, argv
    verbatim: <project> --preset <build_profile> -o <parent(project_path)>
    --skip-deps (rendered into the parent so it lands AT project_path). The CLI
    is re-entered via the RUNNING interpreter (`python -m osprey`), never a
    bare `osprey` that PATH could resolve to a different install."""
    config = _auto_render_config(tmp_path)  # <tmp_path>/ops-app does not exist
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == [
        [
            sys.executable,
            "-m",
            "osprey",
            "build",
            "ops-app",
            "--preset",
            "control-assistant",
            "-o",
            str(tmp_path),
            "--skip-deps",
        ]
    ]


def test_auto_render_partial_render_raises(monkeypatch, tmp_path):
    """project_path exists but is missing its Dockerfile -> a partial render;
    raise (naming the dir) rather than silently rebuild over it."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    # Dockerfile deliberately absent -> partial render.
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    with pytest.raises(ValueError, match="partial render") as excinfo:
        provision._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert str(project_path) in str(excinfo.value)
    assert "Dockerfile" in str(excinfo.value)
    assert calls == []  # never rendered over the partial tree


def test_auto_render_complete_render_is_noop(monkeypatch, tmp_path):
    """project_path exists with both config.yml and Dockerfile -> user-owned
    complete render; never overwrite it, run no `osprey build`."""
    project_path = tmp_path / "ops-app"
    project_path.mkdir()
    (project_path / "config.yml").write_text("project_name: whatever\n", encoding="utf-8")
    (project_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    config = _auto_render_config(tmp_path)
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == []


def test_auto_render_missing_build_profile_raises(monkeypatch, tmp_path):
    """project_path absent (a render IS needed) but the catalog entry has no
    build_profile -> raise, since there's nothing to render from."""
    config = _auto_render_config(tmp_path, build_profile=None)
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    with pytest.raises(ValueError, match="build_profile"):
        provision._auto_render_missing_personas(config, _AUTO_RENDER_USERS, {})

    assert calls == []  # nothing rendered


def test_auto_render_renders_each_distinct_persona_once(monkeypatch, tmp_path):
    """Two users sharing a persona collapse to one render; a second, distinct
    persona renders separately -- one `osprey build` per DISTINCT persona."""
    config = {
        "modules": {
            "web_terminals": {
                "image_source": "local",
                "default_persona": "ops",
                "personas": {
                    "ops": {
                        "project": "ops-app",
                        "project_path": str(tmp_path / "ops-app"),
                        "build_profile": "control-assistant",
                    },
                    "sci": {
                        "project": "sci-app",
                        "project_path": str(tmp_path / "sci-app"),
                        "build_profile": "physicist",
                    },
                },
            }
        }
    }
    resolved_users = [
        {"name": "alice", "index": 0, "persona": "ops", "project": "ops-app"},
        {"name": "bob", "index": 1, "persona": "ops", "project": "ops-app"},  # shares ops
        {"name": "carol", "index": 2, "persona": "sci", "project": "sci-app"},
    ]
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda cmd, **k: calls.append(cmd))

    provision._auto_render_missing_personas(config, resolved_users, {})

    assert len(calls) == 2
    assert any("ops-app" in c and "control-assistant" in c for c in calls)
    assert any("sci-app" in c and "physicist" in c for c in calls)


# ---------------------------------------------------------------------------
# _run_verify_script -- advisory post-up smoke check in isolation
# ---------------------------------------------------------------------------


def test_run_verify_script_skips_silently_when_absent(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(provision.subprocess, "run", lambda *a, **k: calls.append(a))

    provision._run_verify_script(str(tmp_path), {})

    assert calls == []


def test_run_verify_script_runs_via_bash_with_cwd_and_env(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    verify_path = scripts_dir / "verify.sh"
    verify_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    run_env = {"COMPOSE_PROJECT_NAME": "test"}
    provision._run_verify_script(str(tmp_path), run_env)

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == ["bash", str(verify_path)]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"] == run_env


def test_run_verify_script_nonzero_exit_does_not_raise(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")

    monkeypatch.setattr(
        provision.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="boom"),
    )

    provision._run_verify_script(str(tmp_path), {})  # must not raise


def test_run_verify_script_oserror_does_not_raise(monkeypatch, tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "verify.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    def _raise(*a, **k):
        raise OSError("no bash")

    monkeypatch.setattr(provision.subprocess, "run", _raise)

    provision._run_verify_script(str(tmp_path), {})  # must not raise


# ---------------------------------------------------------------------------
# deploy_down_web_terminals -- the web stack's own `compose down`
# ---------------------------------------------------------------------------


def test_deploy_down_web_terminals_runs_compose_down_on_web_file(monkeypatch, tmp_path):
    """With a rendered docker-compose.web.yml at the project root, the web
    stack gets its own `compose -f docker-compose.web.yml down` under the
    pinned compose project — the mirror of deploy_up_web_terminals' second
    invocation. Without it the fixed-name `<prefix>-web-<user>`/`<prefix>-nginx`
    containers outlive every `osprey deploy down` and the next web-terminals
    deploy on the host dies at `up` with a container-name Conflict."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")

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
        "-f",
        "docker-compose.web.yml",
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
