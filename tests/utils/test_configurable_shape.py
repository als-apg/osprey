"""Pins the exact key set of the loader's pre-computed ``configurable`` dict.

``ConfigBuilder.configurable`` is the single pre-computed view every
``get_config_value`` / ``get_full_configuration`` call reads first, so any key it
carries reads as a supported configuration surface. Keys that nothing consumes
therefore have to stay out of it — the retired multi-application scoping
(``applications`` / ``current_application``) and the identity block
(``user_id`` / ``chat_id`` / ``session_id`` / ``thread_id`` / ``session_url`` /
``agent_control_defaults``) are pinned as absent below so they cannot drift back.

The autouse ``reset_state_between_tests`` fixture clears the config singleton and
``CONFIG_FILE`` around every test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.utils.config import ConfigBuilder

EXPECTED_CONFIGURABLE_KEYS = {
    "model_configs",
    "provider_configs",
    "service_configs",
    "execution",
    "python_executor",
    "logging",
    "development",
    "project_root",
    "registry_path",
    "facility_timezone",
}

RETIRED_CONFIGURABLE_KEYS = (
    "applications",
    "current_application",
    "user_id",
    "chat_id",
    "session_id",
    "thread_id",
    "session_url",
    "agent_control_defaults",
)


def _write_config(dir_path: Path, body: str) -> Path:
    cfg = dir_path / "config.yml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


@pytest.fixture
def minimal_configurable(tmp_path):
    cfg = _write_config(tmp_path, f"project_root: {tmp_path}\n")
    return ConfigBuilder(str(cfg), load_env=False).configurable


def test_configurable_key_set_is_exactly_pinned(minimal_configurable):
    assert set(minimal_configurable) == EXPECTED_CONFIGURABLE_KEYS


@pytest.mark.parametrize("key", RETIRED_CONFIGURABLE_KEYS)
def test_retired_key_absent(minimal_configurable, key):
    assert key not in minimal_configurable


@pytest.mark.parametrize("key", RETIRED_CONFIGURABLE_KEYS)
def test_retired_key_absent_even_when_config_declares_it(tmp_path, key):
    """A legacy config that still declares the retired keys keeps loading.

    The values are ignored rather than promoted back into ``configurable``.
    """
    cfg = _write_config(
        tmp_path,
        f"project_root: {tmp_path}\n"
        "applications:\n"
        "  - legacy_app\n"
        "user_id: someone\n"
        "chat_id: c-1\n"
        "session_id: s-1\n"
        "thread_id: t-1\n"
        "session_url: http://example.invalid/session\n"
        "agent_control_defaults:\n"
        "  control_system_writes_enabled: true\n",
    )
    configurable = ConfigBuilder(str(cfg), load_env=False).configurable
    assert key not in configurable
    assert set(configurable) == EXPECTED_CONFIGURABLE_KEYS


def test_writes_enabled_is_not_mirrored_into_configurable(tmp_path):
    """``control_system.writes_enabled`` has its own reader; the loader must not copy it."""
    cfg = _write_config(
        tmp_path,
        f"project_root: {tmp_path}\ncontrol_system:\n  type: mock\n  writes_enabled: true\n",
    )
    configurable = ConfigBuilder(str(cfg), load_env=False).configurable
    assert "control_system" not in configurable
    flattened = str(configurable)
    assert "control_system_writes_enabled" not in flattened
