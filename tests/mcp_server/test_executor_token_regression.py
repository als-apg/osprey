"""Pentest-style regression: agent-run code in the python-executor sandbox
cannot read write-arming tokens from its environment.

Unlike ``test_executor_env_scrub.py`` (which unit-tests the scrub function and
inspects the ``env=`` kwarg passed to a *mocked* subprocess call), these tests
spawn a REAL local subprocess through ``execute_code()`` end-to-end and have
the sandboxed code itself introspect ``os.environ`` and report back what it
can see. This guards the actual write-safety property: even if the scrub
function had a bug that only manifested at runtime (encoding issue, wrong
dict identity, etc.), a real execution catches it where a mocked one would
not.

Threat model: ``BLUESKY_PROMOTE_TOKEN`` authenticates callers of the Bluesky
bridge's ``/runs/{id}/promote`` endpoint. If agent-generated code running in
this sandbox could read that token from its environment, it could bypass
``launch_run``'s in-tool ``writes_enabled`` re-check and POST directly to
``/promote``. This test proves that path is closed at the environment layer.
"""

import json

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import execute_code


@pytest.fixture(autouse=True)
def _reset_all_config_caches(monkeypatch):
    """Reset ALL config caches before each test (see test_executor_adapter.py)."""
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()

    import osprey.utils.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    saved_cache = _cfg._config_cache.copy()
    _cfg._config_cache.clear()

    yield

    reset_config_cache()
    _cfg._config_cache.clear()
    _cfg._config_cache.update(saved_cache)


def _write_local_config(tmp_path):
    config = {
        "control_system": {"type": "mock", "limits_checking": {"enabled": False}},
        "execution": {"execution_method": "local"},
        "python_executor": {"execution_timeout_seconds": 60},
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))


_DUMP_ENV_CODE = (
    "import json, os\n"
    "print('ENV_KEYS_JSON_START')\n"
    "print(json.dumps(sorted(os.environ.keys())))\n"
    "print('ENV_KEYS_JSON_END')\n"
)


def _extract_env_keys(stdout: str) -> list[str]:
    """Pull the JSON-encoded sorted env key list out of the sandbox's stdout."""
    start = stdout.index("ENV_KEYS_JSON_START") + len("ENV_KEYS_JSON_START")
    end = stdout.index("ENV_KEYS_JSON_END")
    return json.loads(stdout[start:end])


async def test_sandbox_code_cannot_see_bluesky_promote_token(tmp_path, monkeypatch):
    """Real sandboxed execution: BLUESKY_PROMOTE_TOKEN is absent from os.environ."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "super-secret-promote-value")
    _write_local_config(tmp_path)

    result = await execute_code(_DUMP_ENV_CODE, "readonly", "pentest env dump")

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    env_keys = _extract_env_keys(result.stdout)
    assert "BLUESKY_PROMOTE_TOKEN" not in env_keys
    # The literal secret value must not leak into stdout/stderr either.
    assert "super-secret-promote-value" not in result.stdout
    assert "super-secret-promote-value" not in result.stderr


async def test_sandbox_code_cannot_see_event_dispatcher_token(tmp_path, monkeypatch):
    """Real sandboxed execution: EVENT_DISPATCHER_TOKEN is absent from os.environ."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "super-secret-dispatch-value")
    _write_local_config(tmp_path)

    result = await execute_code(_DUMP_ENV_CODE, "readonly", "pentest env dump")

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    env_keys = _extract_env_keys(result.stdout)
    assert "EVENT_DISPATCHER_TOKEN" not in env_keys
    # The literal secret value must not leak into stdout/stderr either.
    assert "super-secret-dispatch-value" not in result.stdout
    assert "super-secret-dispatch-value" not in result.stderr


async def test_sandbox_code_cannot_see_either_token_simultaneously(tmp_path, monkeypatch):
    """Both tokens set in the parent process — neither reaches the sandbox."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "promote-secret")
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "dispatch-secret")
    _write_local_config(tmp_path)

    result = await execute_code(_DUMP_ENV_CODE, "readonly", "pentest env dump")

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    env_keys = _extract_env_keys(result.stdout)
    assert "BLUESKY_PROMOTE_TOKEN" not in env_keys
    assert "EVENT_DISPATCHER_TOKEN" not in env_keys
    # Neither literal secret value may leak into stdout/stderr either.
    for secret in ("promote-secret", "dispatch-secret"):
        assert secret not in result.stdout
        assert secret not in result.stderr


async def test_sandbox_still_has_a_usable_environment(tmp_path, monkeypatch):
    """Negative control: scrubbing must not wipe the whole env (e.g. PATH survives).

    Without this, the token-absence assertions above could pass vacuously if
    a bug handed the subprocess an empty environment instead of a scrubbed one.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "promote-secret")
    _write_local_config(tmp_path)

    result = await execute_code(_DUMP_ENV_CODE, "readonly", "pentest env dump")

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    env_keys = _extract_env_keys(result.stdout)
    assert "PATH" in env_keys
    assert len(env_keys) > 1


async def test_sandbox_code_cannot_reach_promote_endpoint_via_env_token(tmp_path, monkeypatch):
    """End-to-end pentest: code that tries to read the token to forge a promote
    call finds nothing to read — the attack surface this task closes.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "promote-secret")
    _write_local_config(tmp_path)

    attack_code = (
        "import os\n"
        "token = os.environ.get('BLUESKY_PROMOTE_TOKEN')\n"
        "print('TOKEN_VALUE:', repr(token))\n"
    )

    result = await execute_code(attack_code, "readonly", "pentest promote token grab")

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    assert "TOKEN_VALUE: None" in result.stdout
    assert "promote-secret" not in result.stdout
