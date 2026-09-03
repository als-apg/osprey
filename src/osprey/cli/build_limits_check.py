"""The build's read of the channel-limits database.

The only code that parses ``control_system.limits_checking.database_path``
against the limits schema used to be the bluesky bridge's own startup gate,
inside a container, at the very end of a deploy: a limits file written against
a superseded schema passed ``osprey build`` and surfaced twenty minutes into
``osprey up`` as an unhealthy container whose compose log names nothing. The
validator's message is the actionable one, and it was being produced in the
one place nobody could read it.

This is the same gate, run where the file is cheap to fix. It IS the bridge's
read, not a mirror of it: the same posture readers, and the one resolver and
loader every reader of the key shares
(:meth:`~osprey_connectors.control_system.limits_validator.LimitsValidator.load_configured_database`),
anchored on the render's own ``config.yml`` — so a file the build accepts is
a file the bridge starts on, and a file the bridge refuses is refused here
first. A missing file is the bridge's refusal too, and stays one here: the
compose renderer's mount-time check already says so for a deployment that
arms writes, and this adds the parse.

Writes-gated for the same reason the bridge and the mount are. A deployment
that leaves every target read-only may name a database it stages later, and
a build that refused it would be refusing a shape the deploy path supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _rendered_config(render_dir: Path) -> dict[str, Any]:
    from osprey_connectors import yaml_loader

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

    from osprey_connectors.control_system.limits_validator import (
        LimitsValidator,
        mapping_config_lookup,
    )

    lookup = mapping_config_lookup(config)
    anchor = str(render_dir / "config.yml")
    if LimitsValidator.configured_database_path(config_lookup=lookup, config_path=anchor) is None:
        return []

    loaded, reason = LimitsValidator.load_configured_database(
        config_lookup=lookup, config_path=anchor
    )
    if loaded is None:
        # The loader's wording, not a paraphrase: it names the field that moved.
        return [f"{writes_key} and {limits_key} are both set for target {target}, but {reason}"]
    return []
