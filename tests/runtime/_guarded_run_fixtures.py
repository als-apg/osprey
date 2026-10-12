"""Shared fixtures of the guarded-run tests: a deployment config, a planted journal, a bound call.

A planted journal is written by the product's own writer
(:class:`~osprey.runtime.journal.DurableJournal`), so its bytes are the bytes a
killed run leaves; :func:`plant_raw` is only for content that writer never
produces.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime.guarded_run as guarded_run
from osprey.runtime.guarded_run import (
    ENV_APPROVED_JOURNAL_SHA256,
    ENV_APPROVED_TARGET,
    JOURNAL_FILE_NAME,
    guarded_run_dir,
    lock,
)
from osprey.runtime.journal import DurableJournal
from osprey_connectors.workspace import reset_config_cache

#: The generation this process is stamped with.
GENERATION = 4

#: An ``approval`` block that puts every tool's call to a human.
ASKING = {"enabled": True, "default_policy": "always"}

#: A deployment on a real control system, whose own target is ``live``.
LIVE_BASELINE = {"type": "epics", "connector": {"epics": {"gateways": {}}}}


def write_config(root: Path, approval: dict[str, Any] | None = ASKING) -> None:
    """Write ``build/config.yml`` on :data:`LIVE_BASELINE` with ``approval``, and drop the cache."""
    config: dict[str, Any] = {"control_system": LIVE_BASELINE}
    if approval is not None:
        config["approval"] = approval
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    reset_config_cache()


def journal_path(target: str = "live") -> Path:
    """The journal file of ``target``'s guarded run."""
    return guarded_run_dir(target) / JOURNAL_FILE_NAME


def plant(
    values: dict[str, Any],
    *,
    target: str = "live",
    generation: int = GENERATION,
    identity: str = "alice",
) -> Path:
    """A journal a killed run on ``target`` left, holding ``values``."""
    journal = DurableJournal(journal_path(target))
    try:
        journal.start(
            target=target,
            generation=generation,
            identity=identity,
            pid=4242,
            started="2026-01-01T00:00:00Z",
        )
        for address, value in values.items():
            journal.record(address, value)
    finally:
        journal.close()
    return journal.path


def plant_raw(lines: Iterable[str], *, target: str = "live") -> Path:
    """A journal on ``target`` holding exactly ``lines``, for content the writer never produces."""
    path = journal_path(target)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def digest(path: Path) -> str:
    """The sha256 hex digest of the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approve(
    monkeypatch: pytest.MonkeyPatch,
    digest: str | None,
    target: str | None = "live",
    tool: str = "execute",
    *,
    enter_lock: bool = False,
) -> None:
    """Bind a fresh approved ``tool`` call, as the wrapper does before user code.

    ``None`` hands the sandbox no such field. With ``enter_lock`` the call's lock
    is entered once, as the wrapper does before user code runs.
    """
    monkeypatch.setattr(guarded_run, "_APPROVED", None)
    if digest is not None:
        monkeypatch.setenv(ENV_APPROVED_JOURNAL_SHA256, digest)
    if target is not None:
        monkeypatch.setenv(ENV_APPROVED_TARGET, target)
    guarded_run._open_approved_call(tool)
    if enter_lock:
        with lock(None):
            pass
