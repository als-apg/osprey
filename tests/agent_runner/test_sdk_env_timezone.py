"""`sdk_env` runs the SDK subprocess on the facility clock.

A deployed agent runs inside a container whose ``TZ`` is the facility timezone
(the compose templates set it from ``system.timezone``), so the date Claude Code
puts in its own system prompt — what the agent takes "today" and "four days ago"
to mean — agrees with every timestamp the deployment's data carries: archiver
events and seeded logbook entries are both anchored in the facility zone.

A headless run on a host — the e2e suite, `osprey audit`, the benchmark
runner — launches the same CLI through :func:`sdk_env` and, before this, left
``TZ`` to the host. On a Pacific host after 17:00 the agent's "today" was one
calendar day behind the facility's (UTC) today, so "four days ago" pointed one
day short of the night the scenario placed its trip on, and a correct diagnosis
read as a date mismatch. The project's config decides the zone, exactly as it
does for the container; a project that names none runs on UTC, the same default
the facility-timezone resolver falls back to, so the two clocks agree either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.agent_runner import sdk_env


def _write_config(project_dir: Path, *, timezone: str | None) -> None:
    """A minimal config.yml: a resolvable provider, plus an optional facility zone."""
    system = f"system:\n  timezone: {timezone}\n" if timezone is not None else ""
    (project_dir / "config.yml").write_text(
        "claude_code:\n  provider: cborg\n  model: claude-haiku-4-5\n" + system
    )


@pytest.fixture(autouse=True)
def _provider_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CBORG_API_KEY", "sk-cborg-secret")


def test_sdk_env_pins_tz_to_the_facility_timezone(tmp_path: Path) -> None:
    _write_config(tmp_path, timezone="America/Los_Angeles")

    env = sdk_env(tmp_path)

    assert env["TZ"] == "America/Los_Angeles"


def test_sdk_env_tz_is_utc_when_the_project_names_no_timezone(tmp_path: Path) -> None:
    """Same default the facility-timezone resolver uses, so both clocks agree."""
    _write_config(tmp_path, timezone=None)

    env = sdk_env(tmp_path)

    assert env["TZ"] == "UTC"


def test_sdk_env_without_a_project_sets_no_tz() -> None:
    """No project, no facility — nothing to pin the clock to."""
    assert "TZ" not in sdk_env()
