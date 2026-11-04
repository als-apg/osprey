"""Tests for a profile's ``optimizer:`` block: its closed key set, its parse and
the targets rule.

The block is closed like ``bluesky:``: a misspelled ``target:`` must fail the
build rather than leave the deployment on its default target. Which targets
are allowed is a validation rule, so it is reported with every other profile
problem instead of stopping the parse at the first one.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile_load import _KNOWN_OPTIMIZER_KEYS, _parse_profile
from osprey.cli.build_profile_schema import OptimizerConfig, optimizer_errors
from osprey.port_layout import default_port
from osprey_connectors.errors import BuildProfileError

#: A port an author pinned by hand, clear of the layout's own default.
AUTHORED_PORT = default_port("optimizer") + 1000


def _profile(block: Any) -> Any:
    return _parse_profile({"name": "x", "data": "data", "optimizer": block})


def test_known_keys_are_the_schema_fields() -> None:
    assert _KNOWN_OPTIMIZER_KEYS == {f.name for f in fields(OptimizerConfig)}


def test_every_valid_key_is_accepted_and_read() -> None:
    profile = _profile(
        {"port": AUTHORED_PORT, "targets": ["va", "standin"], "environments_dir": "envs"}
    )
    assert profile.optimizer == OptimizerConfig(
        port=AUTHORED_PORT, targets=["va", "standin"], environments_dir="envs"
    )


def test_an_empty_block_takes_the_defaults() -> None:
    profile = _profile({})
    assert profile.optimizer == OptimizerConfig()
    assert profile.optimizer.port == default_port("optimizer")
    assert profile.optimizer.targets == ["va"]
    assert profile.optimizer.environments_dir is None


def test_no_block_means_no_optimizer() -> None:
    assert _parse_profile({"name": "x", "data": "data"}).optimizer is None


def test_an_unknown_key_is_refused() -> None:
    with pytest.raises(BuildProfileError, match="target"):
        _profile({"target": ["va"]})


@pytest.mark.parametrize(
    ("block", "match"),
    [
        pytest.param([], "must be a mapping", id="not_a_mapping"),
        pytest.param({"targets": "va"}, "list of target names", id="targets_not_a_list"),
        pytest.param({"environments_dir": ""}, "non-empty path", id="empty_environments_dir"),
    ],
)
def test_a_malformed_block_is_refused(block: Any, match: str) -> None:
    with pytest.raises(BuildProfileError, match=match):
        _profile(block)


@pytest.mark.parametrize(
    ("targets", "match"),
    [
        pytest.param(["live"], "may not include 'live'", id="live"),
        pytest.param(["va", "live"], "may not include 'live'", id="live_beside_va"),
        pytest.param(["sim"], "unknown target", id="unknown"),
        pytest.param([], "at least one", id="none"),
        pytest.param(["va", "va"], "twice", id="duplicate"),
    ],
)
def test_targets_the_optimizer_may_not_drive_are_errors(targets: list[str], match: str) -> None:
    errors = optimizer_errors(OptimizerConfig(targets=targets))
    assert any(match in e for e in errors), errors


def test_allowed_targets_validate_cleanly() -> None:
    assert optimizer_errors(OptimizerConfig(targets=["va", "standin"])) == []


def test_the_live_refusal_reaches_profile_validation(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    profile = _profile({"targets": ["live"]})
    with pytest.raises(BuildProfileError, match="may not include 'live'"):
        profile.validate(tmp_path)


def test_a_valid_block_passes_profile_validation(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    _profile({"targets": ["va", "standin"]}).validate(tmp_path)


def test_an_out_of_range_port_is_an_error() -> None:
    assert any("1..65535" in e for e in optimizer_errors(OptimizerConfig(port=0)))
