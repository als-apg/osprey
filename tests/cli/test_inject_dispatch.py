"""Tests for the ``_inject_dispatch`` build step in ``osprey.cli.build_cmd``.

Covers the four responsibilities of the dispatch-injection step: resolving and
copying the triggers file (bundled preset name or profile-relative path),
copying the bundled event_dispatcher + dispatch_worker compose templates,
writing the ``services.*`` config + registering both in ``deployed_services``
(additively), and raising on an unresolvable triggers name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_dispatch
from osprey.cli.build_profile import DispatchConfig
from osprey.errors import BuildProfileError


def _write_config(project_path: Path) -> None:
    """Write a minimal config.yml with a pre-existing deployed service."""
    yaml = YAML()
    config = {
        "deployed_services": ["postgresql"],
        "services": {"postgresql": {}},
    }
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)


def _read_config(project_path: Path) -> dict:
    """Reload config.yml as a plain dict."""
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)


def _dispatch(**overrides: object) -> DispatchConfig:
    """Build a DispatchConfig with sensible defaults overridable per-test."""
    base: dict = {
        "triggers": "tutorial_triggers.yml",
        "worker_count": 1,
        "workspace_mode": "isolated",
        "max_concurrent_runs": 2,
        "max_queue_depth": 50,
        "dispatcher_port": 8020,
        "worker_port_base": 9190,
        "timeout_sec": 300,
        "facility_name": "ALS",
        "pv_strip_prefix": "ALS:",
    }
    base.update(overrides)
    return DispatchConfig(**base)  # type: ignore[arg-type]


def test_inject_dispatch_bundled_triggers(tmp_path: Path) -> None:
    """A bundled triggers name resolves from the presets dir; full injection."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"  # empty — forces bundled resolution
    profile_dir.mkdir()
    _write_config(project_path)

    _inject_dispatch(_dispatch(), profile_dir=profile_dir, project_path=project_path)

    # Triggers copied from bundled preset.
    triggers = project_path / "triggers.yml"
    assert triggers.is_file()
    assert "hello-dispatch" in triggers.read_text(encoding="utf-8")

    # Compose templates copied, plus the shared Dockerfile (used to build the
    # local dispatch image on `osprey deploy up`).
    assert (project_path / "services" / "event_dispatcher" / "docker-compose.yml.j2").is_file()
    assert (project_path / "services" / "dispatch_worker" / "docker-compose.yml.j2").is_file()
    assert (project_path / "services" / "event_dispatcher" / "Dockerfile").is_file()

    # Config entries written with the correct contract.
    config = _read_config(project_path)
    ed = config["services"]["event_dispatcher"]
    assert ed["port"] == 8020
    assert ed["facility_name"] == "ALS"
    assert ed["pv_strip_prefix"] == "ALS:"
    assert ed["path"] == "./services/event_dispatcher"
    # No pinned image: the service builds the shared local image (the compose
    # template defaults to ``osprey-dispatch:local`` + a ``build:`` section).
    assert "image" not in ed

    dw = config["services"]["dispatch_worker"]
    assert dw["worker_count"] == 1
    assert dw["workspace_mode"] == "isolated"
    assert dw["worker_port_base"] == 9190
    assert dw["timeout_sec"] == 300
    assert dw["path"] == "./services/dispatch_worker"
    assert "image" not in dw

    # deployed_services is additive — keeps postgresql, adds both dispatch services.
    deployed = [str(s) for s in config["deployed_services"]]
    assert "postgresql" in deployed
    assert "event_dispatcher" in deployed
    assert "dispatch_worker" in deployed


def test_inject_dispatch_profile_relative_triggers(tmp_path: Path) -> None:
    """A profile-relative triggers file takes precedence and is copied."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "triggers.yml").write_text("triggers:\n  - name: local-only\n", encoding="utf-8")
    _write_config(project_path)

    _inject_dispatch(
        _dispatch(triggers="triggers.yml"),
        profile_dir=profile_dir,
        project_path=project_path,
    )

    triggers = project_path / "triggers.yml"
    assert triggers.is_file()
    assert "local-only" in triggers.read_text(encoding="utf-8")


def test_inject_dispatch_unresolvable_triggers_raises(tmp_path: Path) -> None:
    """An unresolvable triggers name raises BuildProfileError."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    _write_config(project_path)

    with pytest.raises(BuildProfileError, match="dispatch.triggers not found"):
        _inject_dispatch(
            _dispatch(triggers="does-not-exist.yml"),
            profile_dir=profile_dir,
            project_path=project_path,
        )


def test_inject_dispatch_propagates_inactivity_sec(tmp_path: Path) -> None:
    """A dispatch.inactivity_sec lands in services.dispatch_worker so the compose
    template can render DISPATCH_INACTIVITY_SEC for the worker (mirrors timeout_sec)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"  # empty — forces bundled resolution
    profile_dir.mkdir()
    _write_config(project_path)

    _inject_dispatch(
        _dispatch(inactivity_sec=45),
        profile_dir=profile_dir,
        project_path=project_path,
    )

    dw = _read_config(project_path)["services"]["dispatch_worker"]
    assert dw["inactivity_sec"] == 45


def test_inject_dispatch_propagates_pool_limits(tmp_path: Path) -> None:
    """A non-default dispatch.max_concurrent_runs/max_queue_depth lands in triggers.yml.

    The dispatcher reads pool limits from triggers.yml at runtime, while the
    build profile validates them on DispatchConfig — so the copied triggers.yml
    must be patched to the preset's values, not keep the bundled defaults.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    profile_dir = tmp_path / "profile"  # empty — forces bundled resolution
    profile_dir.mkdir()
    _write_config(project_path)

    _inject_dispatch(
        _dispatch(max_concurrent_runs=7, max_queue_depth=33),
        profile_dir=profile_dir,
        project_path=project_path,
    )

    yaml = YAML()
    with open(project_path / "triggers.yml") as fh:
        triggers_doc = yaml.load(fh)

    assert triggers_doc["dispatcher"]["max_concurrent_runs"] == 7
    assert triggers_doc["dispatcher"]["max_queue_depth"] == 33
    # dispatch_target (and the triggers themselves) are preserved by the patch.
    assert triggers_doc["dispatcher"]["dispatch_target"]
    assert any(t["name"] == "hello-dispatch" for t in triggers_doc["triggers"])
