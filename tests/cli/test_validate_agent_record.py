"""The ``agent_record:`` block, as ``osprey validate`` judges it.

``osprey validate`` is the verb a CI pipeline runs before ``osprey build``, so a
block the build refuses must fail validate first, with the build's own message —
the block's value rules and its named-reader rules both. A reader finding the
build only prints is printed here too, and passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record.config import (
    agent_record_errors,
    parse_agent_record_block,
    reader_errors,
    reader_warnings,
)
from osprey.cli.validate_cmd import validate


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_profile(tmp_path: Path, agent_record: Any) -> Path:
    body: dict[str, Any] = {"name": "Facility", "extends": "hello-world", "data": "data"}
    if agent_record is not None:
        body["agent_record"] = agent_record
    (tmp_path / "data").mkdir(exist_ok=True)
    path = tmp_path / "profile.yml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _run(runner: CliRunner, profile: Path):
    return runner.invoke(validate, [str(profile)])


def _block(raw: dict[str, Any]):
    return parse_agent_record_block({"agent_record": raw})


def test_validate_refuses_a_bad_backend_with_the_rule_message(
    runner: CliRunner, tmp_path: Path
) -> None:
    raw = {"backend": "sqlite"}
    result = _run(runner, _write_profile(tmp_path, raw))
    assert result.exit_code == 2, result.output
    expected = agent_record_errors(_block(raw))
    assert expected
    for message in expected:
        assert message in result.output


def test_validate_refuses_a_bad_retention(runner: CliRunner, tmp_path: Path) -> None:
    raw = {"retention_days": 0}
    result = _run(runner, _write_profile(tmp_path, raw))
    assert result.exit_code == 2, result.output
    assert "agent_record.retention_days" in result.output


def test_validate_refuses_a_bad_reader_entry_with_the_build_message(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Reader rules only reach the command through the deploy-aware lint."""
    raw = {"readers": [{"name": "ops", "kinds": ["not_a_kind"]}]}
    result = _run(runner, _write_profile(tmp_path, raw))
    assert result.exit_code == 2, result.output
    assert "Profile validation failed" in result.output
    expected = reader_errors(_block(raw), {})
    assert expected
    for message in expected:
        assert message in result.output


def test_validate_refuses_a_reader_window_beyond_retention(
    runner: CliRunner, tmp_path: Path
) -> None:
    raw = {"retention_days": 30, "readers": [{"name": "ops", "days_back": 90}]}
    result = _run(runner, _write_profile(tmp_path, raw))
    assert result.exit_code == 2, result.output
    assert "days_back" in result.output


def test_validate_passes_a_valid_block(runner: CliRunner, tmp_path: Path) -> None:
    raw = {"retention_days": 30, "readers": [{"name": "ops", "days_back": 7}]}
    result = _run(runner, _write_profile(tmp_path, raw))
    assert result.exit_code == 0, result.output
    assert "Profile is valid" in result.output


def test_validate_prints_a_reader_warning_and_passes(runner: CliRunner, tmp_path: Path) -> None:
    raw = {"readers": [{"name": "ops-script", "oidc_subject": "abc-123"}]}
    result = _run(runner, _write_profile(tmp_path, raw))
    assert result.exit_code == 0, result.output
    warnings = reader_warnings(_block(raw), {})
    assert warnings
    for message in warnings:
        assert message in result.output
    assert "Profile is valid" in result.output
