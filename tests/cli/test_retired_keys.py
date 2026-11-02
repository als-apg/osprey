"""A profile that spells a key the agent record replaced is refused, naming the replacement.

The retired keys live in one table,
:data:`~osprey.agent_record.config.RETIRED_CONFIG_KEYS`, beside the block that
replaces them. The profile parser refuses each of them — the key itself and
anything below it, in a nested, dotted or mixed spelling — and every
``deployed_services`` entry the record replaced, all offenders at once, each
with what to write instead. The manifest lists every one of them as deleted
and none as a live key, so nothing renders, defaults or documents them.

The operator's path through it: build, read the refusal, edit the profile as
it says, build again.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record.config import (
    RETIRED_CONFIG_KEYS,
    RETIRED_DEPLOYED_SERVICES,
    refuse_retired_config_keys,
)
from osprey.cli.build_cmd import build
from osprey.cli.build_profile_load import _parse_profile, manifest_rows
from osprey.cli.channel_finder_cmd import FILE_DATABASE_PARADIGMS
from osprey.cli.config_cmd import packaged_manifest_path
from osprey_connectors.errors import BuildProfileError
from tests._preset_data import bundle_data_root

#: One leaf the tests spell under each retired key, and the text of its
#: replacement the refusal must carry.
_SPELLED: dict[str, tuple[str, Any, str]] = {
    "claude_code.telemetry": ("claude_code.telemetry.enabled", False, "agent record"),
    "claude_code.transcripts": (
        "claude_code.transcripts.retention_days",
        30,
        "agent_record.retention_days",
    ),
    "audit.tool_call": ("audit.tool_call.enabled", True, "agent-record entry"),
    "services.telemetry_store": (
        "services.telemetry_store.retention_days",
        30,
        "agent_record:",
    ),
    "services.telemetry_gateway": ("services.telemetry_gateway.port", 4318, "agent record"),
    "web.feedback.max_store_bytes": (
        "web.feedback.max_store_bytes",
        1048576,
        "agent_record.retention_days",
    ),
    "modules.web_terminals.reader_port": (
        "modules.web_terminals.reader_port",
        8095,
        "modules.web_terminals.history_port",
    ),
    "modules.web_terminals.reader_origin": (
        "modules.web_terminals.reader_origin",
        "https://history.example.org",
        "modules.web_terminals.history_origin",
    ),
}


def _nested(path: str, value: Any) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    return {head: _nested(rest, value) if rest else value}


def _mixed(path: str, value: Any) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    return {head: {rest: value}} if rest else {head: value}


def _refusal(config: dict[str, Any]) -> str:
    with pytest.raises(BuildProfileError) as caught:
        _parse_profile({"name": "probe", "config": config})
    return str(caught.value)


def test_every_retired_key_has_a_spelled_case() -> None:
    assert set(_SPELLED) == set(RETIRED_CONFIG_KEYS)


@pytest.mark.parametrize("spelling", [pytest.param(s, id=s) for s in ("dotted", "nested", "mixed")])
@pytest.mark.parametrize("retired", sorted(_SPELLED))
def test_a_retired_key_is_refused_in_every_spelling_naming_its_replacement(
    retired: str, spelling: str
) -> None:
    leaf, value, replacement = _SPELLED[retired]
    config = {
        "dotted": {leaf: value},
        "nested": _nested(leaf, value),
        "mixed": _mixed(leaf, value),
    }[spelling]

    message = _refusal(config)

    assert leaf in message
    assert replacement in message


def test_a_retired_block_spelled_whole_is_refused() -> None:
    message = _refusal({"claude_code": {"telemetry": {}}})

    assert "claude_code.telemetry —" in message


@pytest.mark.parametrize("service", sorted(RETIRED_DEPLOYED_SERVICES))
def test_a_retired_deployed_services_entry_is_refused(service: str) -> None:
    message = _refusal({"deployed_services": ["qmd", service]})

    assert f"deployed_services: {service}" in message
    assert "never a `deployed_services` entry" in message
    assert "'qmd'" not in message


def test_every_offender_is_named_at_once() -> None:
    message = _refusal(
        {
            "claude_code": {"telemetry": {"enabled": False}, "transcripts.retention_days": 30},
            "modules.web_terminals.reader_port": 8095,
            "deployed_services": ["telemetry_store", "telemetry_gateway"],
        }
    )

    for named in (
        "claude_code.telemetry.enabled",
        "claude_code.transcripts.retention_days",
        "modules.web_terminals.reader_port",
        "deployed_services: telemetry_gateway, telemetry_store",
    ):
        assert named in message


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"claude_code.telemetry_extra": 1}, id="a-sibling-sharing-a-prefix"),
        pytest.param({"modules.web_terminals.history_port": 8095}, id="the-replacement-key"),
        pytest.param({"web.feedback.github_repo": "org/repo"}, id="a-live-sibling"),
        pytest.param({"audit": {"other": 1}}, id="a-key-beside-the-retired-block"),
        pytest.param({"deployed_services": ["qmd", "graphdb"]}, id="live-services"),
    ],
)
def test_a_key_beside_a_retired_one_is_not_refused(config: dict[str, Any]) -> None:
    refuse_retired_config_keys(config)


def test_the_manifest_lists_every_retired_key_as_deleted_and_none_as_live() -> None:
    manifest = yaml.safe_load(packaged_manifest_path().read_text(encoding="utf-8"))
    deleted = set(manifest["deleted"])
    live = set(manifest_rows())

    for retired in RETIRED_CONFIG_KEYS:
        assert retired in deleted, retired
        assert not [key for key in live if key == retired or key.startswith(f"{retired}.")]


# --- FR6: refused, edited as directed, then builds -------------------------


def _repo(tmp_path: Path, config: dict[str, Any]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "profile.yml").write_text(
        yaml.dump(
            {
                "name": "Retired Keys",
                "project_name": "retired-keys",
                "data": "data",
                "provider": "cborg",
                "model": "claude-haiku-4-5",
                "channel_finder_mode": FILE_DATABASE_PARADIGMS[0],
                "config": config,
            },
            default_flow_style=False,
        )
    )
    shutil.copytree(bundle_data_root("control_assistant"), repo / "data")
    (repo / "data" / "facility" / "knowledge").mkdir(parents=True, exist_ok=True)
    return repo


def _build(repo: Path):
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_a_profile_with_retired_keys_is_refused_then_edited_as_directed_then_builds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    posture = {
        "control_system.type": "virtual_accelerator",
        "control_system.connector.virtual_accelerator.serving": "in_process",
        "archiver.type": "mock",
        # The bundle's own hooks, approval among them, are what a profile that
        # names none renders with, so the approval floor is stated here.
        "approval.enabled": False,
        "approval.default_policy": "always",
        "hooks.debug": False,
    }
    retired = {
        "claude_code.telemetry.enabled": False,
        "claude_code.transcripts.retention_days": 45,
    }
    repo = _repo(tmp_path, {**posture, **retired})

    with caplog.at_level(logging.ERROR):
        refused = _build(repo)

    assert refused.exit_code != 0
    reported = caplog.text
    assert "claude_code.telemetry.enabled — remove it" in reported
    assert "claude_code.transcripts.retention_days — set `agent_record.retention_days`" in reported

    # As directed: the telemetry block goes, the retention moves to the record.
    profile = yaml.safe_load((repo / "profile.yml").read_text())
    for key in retired:
        del profile["config"][key]
    profile["agent_record"] = {"retention_days": 45}
    (repo / "profile.yml").write_text(yaml.dump(profile, default_flow_style=False))

    built = _build(repo)

    assert built.exit_code == 0, built.output
