"""The store's compose fragment and server config files, rendered.

Regenerate the goldens with:

    PYTHONPATH=src ./.venv/bin/python tests/deployment/telemetry_store/test_store_render.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.templates.test_render_defaults_golden import (
    _pinned_context,
    _probe_repo,
    _raw_default_config,
    _render_templates,
)

_GOLDENS = Path(__file__).parent / "goldens"
_FILES = ("docker-compose.yml", "osprey.xml", "osprey-defaults.xml", "zz-osprey-admin.xml")

#: name -> deployment overlay. The store publishes on loopback under every one.
SCENARIOS = {
    "defaults": {},
    "bind_all": {"bind_address": "0.0.0.0"},
    "bind_concrete": {"bind_address": "10.0.0.5"},
}


def _render(scenario: str) -> dict[str, str]:
    with _probe_repo() as repo_root:
        config = _raw_default_config(repo_root)
        config["deployment"] = dict(SCENARIOS[scenario])
        config["deployed_services"] = ["telemetry_store"]
        context = _pinned_context(config)
    # _render_templates keys by service directory, so one file per call.
    return {
        name: _render_templates(context, [f"services/telemetry_store/{name}.j2"])[
            "telemetry_store.yml"
        ]
        for name in _FILES
    }


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_golden(scenario: str) -> None:
    for name, text in _render(scenario).items():
        assert text == (_GOLDENS / f"render_{scenario}" / name).read_text(), name


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_store_publishes_on_loopback_only(scenario: str) -> None:
    compose = yaml.safe_load(_render(scenario)["docker-compose.yml"])
    store = compose["services"]["telemetry_store"]
    assert len(store["ports"]) == 1
    assert store["ports"][0].startswith("127.0.0.1:") and store["ports"][0].endswith(":8123")
    assert store["env_file"] == [".env.telemetry-admin"]
    assert store["labels"] == {
        "osprey.telemetry.cred.digest": "${OSPREY_TELEMETRY_DIGEST_TELEMETRY_STORE:-}"
    }
    assert store["mem_limit"] == 2 * 1024**3


def test_server_config_pins() -> None:
    rendered = _render("defaults")
    config = rendered["osprey.xml"]
    assert (
        "<max_server_memory_usage_to_ram_ratio>0.9</max_server_memory_usage_to_ram_ratio>" in config
    )
    assert "<users_without_row_policies_can_read_rows>true</" in config
    assert "<allow_implicit_no_password>0</allow_implicit_no_password>" in config
    assert "<max_concurrent_select_queries>4</max_concurrent_select_queries>" in config
    admin = rendered["zz-osprey-admin.xml"]
    assert "<osprey_admin>" in admin
    assert '<networks replace="replace">' in admin
    assert "<ip>127.0.0.1</ip>" in admin and "<ip>::1</ip>" in admin
    assert "<access_management>1</access_management>" in admin


if __name__ == "__main__":
    for scenario in SCENARIOS:
        target = _GOLDENS / f"render_{scenario}"
        target.mkdir(parents=True, exist_ok=True)
        for name, text in _render(scenario).items():
            (target / name).write_text(text)
