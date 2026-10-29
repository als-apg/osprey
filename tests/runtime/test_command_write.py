"""A run's limits posture is its target's, and a command write is a write the run lock gates.

An :class:`ActionContext` says whether limits checking is off for the run's
target, and that is never the caller's to say: the lock derives it from the
target's limits posture (complete, and ``enabled: false``) and refuses a
context that disagrees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import (
    ENV_CONTROL_IDENTITY,
    ActionContext,
    lock,
    target_limits_off,
)
from osprey_connectors import posture_store
from osprey_connectors.workspace import reset_config_cache
from tests._guarded_run import LIVE_BASELINE, reset_run_state

ALICE_LIVE = ActionContext(target="live", owner="alice", approved_target="live")
ALICE_LIVE_LIMITS_OFF = ActionContext(
    target="live", owner="alice", approved_target="live", limits_off=True
)

#: Limits blocks by what they say, as ``control_system.limits_checking``.
LIMITS_OFF: dict[str, Any] = {"enabled": False}
STRICT: dict[str, Any] = {"enabled": True, "mode": "exclusive", "on_violation": "error"}
OPTIONAL: dict[str, Any] = {"enabled": True, "mode": "optional", "on_violation": "error"}
UNSTATED: dict[str, Any] = {"mode": "exclusive"}


def _deploy(root: Path, limits_checking: Any = None) -> None:
    """Write ``build/config.yml`` on a live baseline with *limits_checking*, none when ``None``."""
    section: dict[str, Any] = dict(LIVE_BASELINE)
    if limits_checking is not None:
        section["limits_checking"] = limits_checking
    config = {"control_system": section, "approval": {"enabled": False}}
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host repo on a live baseline with no limits block, stamped ``live``, owner alice.

    The run state is fresh: nothing is held.
    """
    reset_run_state(monkeypatch)
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    _deploy(root)
    monkeypatch.chdir(root)
    for name in ("OSPREY_CONFIG", "CONFIG_FILE", "OSPREY_EXECUTION_MODE", ENV_CONTROL_IDENTITY):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_OWNER_ENV_VAR, "alice")
    yield root
    reset_config_cache()


# --------------------------------------------------------------------------
# limits_off is the target's
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limits_checking", "off"),
    [
        (LIMITS_OFF, True),
        (STRICT, False),
        (OPTIONAL, False),
        (UNSTATED, False),
        (None, False),
        ("not a block", False),
    ],
    ids=["enabled-false", "strict", "optional", "unstated-enabled", "no-block", "unreadable"],
)
def test_limits_are_off_only_for_a_complete_posture_that_says_enabled_false(
    repo: Path, limits_checking: Any, off: bool
) -> None:
    _deploy(repo, limits_checking)

    assert target_limits_off("live") is off


def test_a_context_saying_limits_are_off_holds_the_lock_on_a_limits_off_target(
    repo: Path,
) -> None:
    _deploy(repo, LIMITS_OFF)

    with lock(None, context=ALICE_LIVE_LIMITS_OFF):
        pass


@pytest.mark.parametrize(
    "limits_checking",
    [STRICT, OPTIONAL, UNSTATED, None],
    ids=["strict", "optional", "unstated-enabled", "no-block"],
)
def test_a_caller_built_context_claiming_limits_off_is_refused(
    repo: Path, limits_checking: Any
) -> None:
    _deploy(repo, limits_checking)

    with pytest.raises(RuntimeError, match="limits checking is off"):
        with lock(None, context=ALICE_LIVE_LIMITS_OFF):
            pytest.fail("a run whose context claims limits are off never starts")


def test_a_context_saying_limits_are_on_is_refused_on_a_limits_off_target(repo: Path) -> None:
    _deploy(repo, LIMITS_OFF)

    with pytest.raises(RuntimeError, match="limits checking is on"):
        with lock(None, context=ALICE_LIVE):
            pytest.fail("a run whose context disagrees with its target never starts")
