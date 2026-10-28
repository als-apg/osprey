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
(:meth:`~osprey_connectors.control_system.LimitsValidator.load_configured_database`),
anchored on the render's own ``config.yml`` — so a file the build accepts is
a file the bridge starts on, and a file the bridge refuses is refused here
first. A missing file is the bridge's refusal too, and stays one here: the
compose renderer's mount-time check already says so for a deployment that
arms writes, and this adds the parse.

Tier-gated for the same reason the mount is: a render whose login tier is
``control_system.writes: never`` writes nowhere and never opens the limits
database, so no reader of it has anything to refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _rendered_config(render_dir: Path) -> dict[str, Any]:
    from osprey.utils.config_writer import flush_config_edits
    from osprey_connectors import yaml_loader

    # The render edits config.yml under one session; read what it holds.
    flush_config_edits(render_dir / "config.yml")
    with (render_dir / "config.yml").open("r", encoding="utf-8") as fh:
        return yaml_loader.safe_load(fh) or {}


def _tier_key(section: dict[str, Any]) -> str:
    """The key that set an ``allowed`` tier, as the render spells it.

    ``control_system.writes`` whenever the render carries it. A hand-written
    render that still spells only the retired leaves answers through
    :func:`~osprey_connectors.types.tier_writes`' fallback, and the line names
    the leaf that armed it rather than a key the file does not contain.
    """
    from osprey.cli.build_profile_document import RETIRED_WRITES_LEAF
    from osprey_connectors.types import WRITES_KEY

    section_name, _, tier_leaf = WRITES_KEY.rpartition(".")
    if tier_leaf in section:
        return WRITES_KEY
    if section.get(RETIRED_WRITES_LEAF) is True:
        return f"{section_name}.{RETIRED_WRITES_LEAF}"
    connector = section.get("connector")
    if isinstance(connector, dict):
        for connector_type, block in connector.items():
            if isinstance(block, dict) and block.get(RETIRED_WRITES_LEAF) is True:
                return f"{section_name}.connector.{connector_type}.{RETIRED_WRITES_LEAF}"
    return WRITES_KEY


def limits_database_errors(render_dir: Path) -> list[str]:
    """The limits database a render will start on, parsed; one line when it cannot be.

    Read after the profile's ``data/`` tree has been applied into the render,
    because that is where a relative ``database_path`` resolves to and where
    the file arrives from. The gate is the render's tier: the login tier allows
    writes (:func:`~osprey_connectors.types.tier_allows`) *and* some configured
    target's resolved limits posture requires a database
    (:func:`~osprey_connectors.types.limits_database_required` — limits checking
    on, and a mode other than ``optional`` or a path configured); the list is
    :func:`~osprey_connectors.types.tier_limits_database_targets`, the same one
    the compose mount reads.
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
    from osprey_connectors.control_system import (
        LimitsValidator,
        mapping_config_lookup,
    )
    from osprey_connectors.types import target_limits_posture, tier_limits_database_targets

    config = _rendered_config(render_dir)
    section = config.get("control_system")
    if not isinstance(section, dict):
        return []

    lookup = mapping_config_lookup(config)
    anchor = str(render_dir / "config.yml")
    path_configured = (
        LimitsValidator.configured_database_path(config_lookup=lookup, config_path=anchor)
        is not None
    )

    armed = None
    for target in tier_limits_database_targets(section, path_configured=path_configured):
        posture = target_limits_posture(section, target)
        if posture.incomplete:
            continue
        armed = (target, _tier_key(section), posture.key("enabled"))
        break
    if armed is None or not path_configured:
        return []
    target, writes_key, limits_key = armed

    loaded, reason = LimitsValidator.load_configured_database(
        config_lookup=lookup, config_path=anchor
    )
    if loaded is None:
        # The loader's wording, not a paraphrase: it names the field that moved.
        return [f"{writes_key} and {limits_key} are both set for target {target}, but {reason}"]
    return []
