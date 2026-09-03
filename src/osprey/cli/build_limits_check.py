"""The build's read of the channel-limits database.

The only code that parses ``control_system.limits_checking.database_path``
against the limits schema used to be the bluesky bridge's own startup gate,
inside a container, at the very end of a deploy: a limits file written against
a superseded schema passed ``osprey build`` and surfaced twenty minutes into
``osprey up`` as an unhealthy container whose compose log names nothing. The
validator's message is the actionable one, and it was being produced in the
one place nobody could read it.

This is the same gate, run where the file is cheap to fix. It mirrors
``_assert_limits_readable_if_writable`` in the bridge exactly — same posture
readers, same resolution anchor (the render's own ``config.yml``), same loader
— so a file the build accepts is a file the bridge starts on, and a file the
bridge refuses is refused here first. A missing file is the bridge's refusal
too, and stays one here: the compose renderer's mount-time check already says
so for a deployment that arms writes, and this adds the parse.

Writes-gated for the same reason the bridge and the mount are. A deployment
that leaves every target read-only may name a database it stages later, and
a build that refused it would be refusing a shape the deploy path supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

LIMITS_DATABASE_KEY = "control_system.limits_checking.database_path"


def _rendered_config(render_dir: Path) -> dict[str, Any]:
    from osprey.utils.config_writer import flush_config_edits
    from osprey_connectors import yaml_loader

    # The render edits config.yml under one session; read what it holds.
    flush_config_edits(render_dir / "config.yml")
    with (render_dir / "config.yml").open("r", encoding="utf-8") as fh:
        return yaml_loader.safe_load(fh) or {}


def limits_database_errors(render_dir: Path) -> list[str]:
    """The limits database a render will start on, parsed; one line when it cannot be.

    Read after the profile's ``data/`` tree has been applied into the render,
    because that is where a relative ``database_path`` resolves to and where
    the file arrives from. The gate is the bridge's: some configured target
    arms writes *and* its resolved limits posture states ``enabled: true``.
    The database is deployment-wide, so it is loaded once however many targets
    qualify, and the line names the first target that made it load.

    Args:
        render_dir: The rendered project directory, after the conventions.

    Returns:
        One line carrying the validator's own message, naming the key and the
        posture that armed it; empty when nothing armed the read or the file
        loads. A ``database_path`` that is unset or not a string is left to
        the compose renderer, whose refusal already names it.
    """
    from osprey_connectors.types import (
        configured_targets,
        target_limits_posture,
        target_writes_enabled,
        target_writes_enabled_key,
    )

    config = _rendered_config(render_dir)
    section = config.get("control_system")
    if not isinstance(section, dict):
        return []

    armed = None
    for target in configured_targets(section):
        if not target_writes_enabled(section, target):
            continue
        posture = target_limits_posture(section, target)
        if posture.incomplete or posture.enabled is not True:
            continue
        armed = (target, target_writes_enabled_key(section, target), posture.key("enabled"))
        break
    if armed is None:
        return []
    target, writes_key, limits_key = armed

    limits_checking = section.get("limits_checking")
    raw = limits_checking.get("database_path") if isinstance(limits_checking, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return []

    from osprey.connectors.control_system.limits_validator import LimitsValidator

    db_path = LimitsValidator.resolve_database_path(
        raw, config.get("project_root"), config_path=str(render_dir / "config.yml")
    )
    try:
        LimitsValidator._load_limits_database(db_path)
    except Exception as exc:  # noqa: BLE001 - the validator's message is the report
        return [
            f"{writes_key} and {limits_key} are both set for target {target}, but "
            f"{LIMITS_DATABASE_KEY} ({raw}) could not be read or parsed: {exc}"
        ]
    return []
