"""The environment an agent-code execution child is spawned with is an allowlist.

Three spawn sites run agent-authored Python in a child process: the python
executor, the visualization sandbox and plan validation. Each builds the
child's environment with :func:`osprey.mcp_server.sandbox_env.scrub_sandbox_child_env`,
which keeps only the names the child's own code reads (plus the names a
deployment lists in ``python_executor.child_env_passthrough``), then drops the
credential set and the sandbox-only names on top, so neither the allowlist nor
the config key can re-admit them.

The child is also stamped as handed a resolved environment, so it does not load
the project's env chain back into itself: without that stamp the allowlist is
inert, because the wrapper's own registry setup would read ``.env`` from the
child's working directory.
"""

from __future__ import annotations

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import execute_code
from osprey.mcp_server.sandbox_env import (
    CHILD_ENV_PASSTHROUGH_KEY,
    SANDBOX_CHILD_ENV_ALLOW_NAMES,
    SANDBOX_CHILD_ENV_ALLOW_PREFIXES,
    configured_child_env_passthrough,
    scrub_sandbox_child_env,
)
from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV

# --------------------------------------------------------------------------- #
# The allowlist
# --------------------------------------------------------------------------- #


def test_unlisted_secret_shaped_names_are_absent():
    """A name the child's code does not read never reaches it, whatever it holds."""
    parent = {
        "EXAMPLE_PROVIDER_API_KEY": "provider",
        "EXAMPLE_BRIDGE_SA_KEY": "bridge",
        "EXAMPLE_STORE_PASSWORD": "store",
        "EXAMPLE_FORGE_TOKEN": "forge",
        "OSPREY_AUTH_PW_ALICE": "alice",
        "OSPREY_ARCHIVER_PASSWORD": "archiver",
        "PATH": "/usr/bin",
    }

    assert scrub_sandbox_child_env(parent) == {"PATH": "/usr/bin", ENV_CHAIN_APPLIED_ENV: "1"}


@pytest.mark.parametrize(
    "name",
    [
        "HOME",
        "LANG",
        "VIRTUAL_ENV",
        "SSL_CERT_FILE",
        "HTTPS_PROXY",
        "https_proxy",
        "MPLCONFIGDIR",
        "OSPREY_CONFIG",
        "EPICS_CA_ADDR_LIST",
        "LC_ALL",
        "PYTHONPATH",
        "TANGO_HOST",
        "ENSHOST",
    ],
)
def test_allowlisted_names_and_prefixes_are_present(name):
    assert scrub_sandbox_child_env({name: "value"})[name] == "value"


def test_osprey_prefix_is_not_an_allow_prefix():
    """``OSPREY_`` holds the web-auth passwords and the archiver password."""
    assert not any("OSPREY_".startswith(prefix) for prefix in SANDBOX_CHILD_ENV_ALLOW_PREFIXES)
    assert not any(prefix.startswith("OSPREY") for prefix in SANDBOX_CHILD_ENV_ALLOW_PREFIXES)


def _child_read_names() -> list[tuple[str, str]]:
    from osprey import runtime
    from osprey.audit import posture
    from osprey.mcp_server.python_executor import executor
    from osprey_connectors import identity, posture_store

    return [
        ("posture_store.AGENT_DATA_ROOT_ENV_VAR", posture_store.AGENT_DATA_ROOT_ENV_VAR),
        ("posture_store.CONTROL_CONTEXT_DIR_ENV_VAR", posture_store.CONTROL_CONTEXT_DIR_ENV_VAR),
        ("posture_store.CONTROL_CONTEXT_TREE_ENV_VAR", posture_store.CONTROL_CONTEXT_TREE_ENV_VAR),
        ("posture_store.CONTROL_OWNER_ENV_VAR", posture_store.CONTROL_OWNER_ENV_VAR),
        ("posture_store.LAUNCH_POSTURE_ENV_VAR", posture_store.LAUNCH_POSTURE_ENV_VAR),
        ("identity.AUDIT_IDENTITY_ENV", identity.AUDIT_IDENTITY_ENV),
        ("posture.POSTURE_ENV_VAR", posture.POSTURE_ENV_VAR),
        ("posture.POSTURE_SOURCE_ENV_VAR", posture.POSTURE_SOURCE_ENV_VAR),
        ("posture.POSTURE_SESSION_ENV_VAR", posture.POSTURE_SESSION_ENV_VAR),
        ("posture.CONTROL_TARGET_ENV_VAR", posture.CONTROL_TARGET_ENV_VAR),
        ("runtime.ENV_CONTROL_TARGET", runtime.ENV_CONTROL_TARGET),
        ("runtime.ENV_CONTROL_TARGET_GENERATION", runtime.ENV_CONTROL_TARGET_GENERATION),
        ("runtime.ENV_CONTROL_TARGET_REFUSAL", runtime.ENV_CONTROL_TARGET_REFUSAL),
        ("executor.ENV_LAUNCH_POSTURE", executor.ENV_LAUNCH_POSTURE),
    ]


@pytest.mark.parametrize(("source", "name"), _child_read_names(), ids=lambda v: str(v))
def test_every_name_the_child_reads_is_allowed(source, name):
    """Each osprey literal on the allowlist is pinned to the constant its reader uses."""
    assert name in SANDBOX_CHILD_ENV_ALLOW_NAMES, f"{source} ({name}) is not allowlisted"
    assert scrub_sandbox_child_env({name: "value"}).get(name) == "value"


def test_passthrough_adds_a_name():
    child = scrub_sandbox_child_env(
        {"HDF5_PLUGIN_PATH": "/opt/plugins", "OTHER": "x"}, passthrough=("HDF5_PLUGIN_PATH",)
    )
    assert child["HDF5_PLUGIN_PATH"] == "/opt/plugins"
    assert "OTHER" not in child


def test_passthrough_cannot_readmit_a_credential():
    parent = {
        "OSPREY_PANEL_TOKEN": "secret",
        "OSPREY_WEB_PORT": "8080",
        "OSPREY_TERMINAL_USER": "alice",
        "PATH": "/usr/bin",
    }
    child = scrub_sandbox_child_env(parent, passthrough=tuple(parent))

    assert "OSPREY_PANEL_TOKEN" not in child
    assert "OSPREY_WEB_PORT" not in child
    assert "OSPREY_TERMINAL_USER" not in child
    assert child["PATH"] == "/usr/bin"


def test_chain_stamp_is_set():
    assert scrub_sandbox_child_env({})[ENV_CHAIN_APPLIED_ENV] == "1"


def test_input_is_not_mutated():
    parent = {"PATH": "/usr/bin", "EXAMPLE_PROVIDER_API_KEY": "secret"}
    original = dict(parent)
    scrub_sandbox_child_env(parent, passthrough=("EXAMPLE_PROVIDER_API_KEY",))
    assert parent == original


# --------------------------------------------------------------------------- #
# The config key
# --------------------------------------------------------------------------- #


def test_passthrough_key_absent_gives_nothing():
    assert configured_child_env_passthrough({}) == ()
    assert configured_child_env_passthrough({"python_executor": {}}) == ()
    assert (
        configured_child_env_passthrough({"python_executor": {"child_env_passthrough": None}}) == ()
    )


def test_passthrough_key_list_gives_a_tuple():
    config = {"python_executor": {"child_env_passthrough": ["HDF5_PLUGIN_PATH", "OMP_NUM_THREADS"]}}
    assert configured_child_env_passthrough(config) == ("HDF5_PLUGIN_PATH", "OMP_NUM_THREADS")


def test_passthrough_key_rejects_a_non_mapping_section():
    with pytest.raises(ValueError, match="python_executor must be a mapping"):
        configured_child_env_passthrough({"python_executor": 42})


@pytest.mark.parametrize(
    "value",
    [
        "HDF5_PLUGIN_PATH",
        ["HDF5_PLUGIN_PATH", 3],
        ["NOT-A-NAME"],
        ["1LEADING_DIGIT"],
        ["TRAILING_NEWLINE\n"],
        ["OSPREY_PANEL_TOKEN"],
        ["BLUESKY_LAUNCH_TOKEN"],
        ["OSPREY_TERMINAL_USER"],
    ],
    ids=[
        "not-a-list",
        "non-string",
        "dash",
        "leading-digit",
        "trailing-newline",
        "exact",
        "suffix",
        "terminal",
    ],
)
def test_passthrough_key_rejects_a_bad_entry(value):
    with pytest.raises(ValueError, match=CHILD_ENV_PASSTHROUGH_KEY):
        configured_child_env_passthrough({"python_executor": {"child_env_passthrough": value}})


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def reset_config_caches(monkeypatch):
    """Reset every config cache around a test that writes its own ``config.yml``."""
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


def _write_mock_project(root) -> None:
    (root / "config.yml").write_text(
        yaml.dump(
            {
                "control_system": {"type": "mock", "limits_checking": {"enabled": False}},
                "execution": {"execution_method": "subprocess"},
                "python_executor": {"execution_timeout_seconds": 60},
            }
        )
    )


# --------------------------------------------------------------------------- #
# The executor child does not load the env chain itself
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("reset_config_caches")
async def test_executor_child_does_not_reload_the_env_chain(tmp_path, monkeypatch):
    """A value that lives only in the project ``.env`` never reaches executed code.

    The child's working directory is the project root, and its registry setup
    builds a ``ConfigBuilder`` there; without the stamp that load reads ``.env``
    back into the child's environment and hands it the very value the parent
    withheld.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXAMPLE_PROVIDER_API_KEY", raising=False)
    _write_mock_project(tmp_path)
    (tmp_path / ".env").write_text("EXAMPLE_PROVIDER_API_KEY=from-dotenv\n")

    result = await execute_code(
        "import os\nprint('KEY=' + repr(os.environ.get('EXAMPLE_PROVIDER_API_KEY')))\n",
        "readonly",
        "env chain probe",
    )

    assert result.success, f"execution failed: {result.error_message}\n{result.stderr}"
    assert "KEY=None" in result.stdout
    assert "from-dotenv" not in result.stdout
