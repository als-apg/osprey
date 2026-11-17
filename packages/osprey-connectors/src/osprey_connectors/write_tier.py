"""How far a deployment may write: the login tier, ``control_system.writes``.

:func:`tier_writes` answers "how far may this deployment write" from the one tier key,
``control_system.writes``. The lint that reads the config, the profile card
that tells an operator what they hold, and the connector that would perform
the write all have to agree about the answer. Two readings of one posture is
one of them being wrong about a machine somebody can move.
"""

from typing import Literal, get_args

__all__ = [
    "TIER_WORDS",
    "TierWord",
    "WRITES_KEY",
    "tier_allows",
    "tier_writes",
]

#: The login tier's write ceiling, dotted as a caller spells it for
#: ``get_config_value``: ``"never"``, ``"asked"`` or ``"allowed"``, the furthest
#: position a login may pick. Target-blind — a tier that allows writes allows
#: them for every machine, and which machine is actually written is the
#: session's switch to decide, not the tier's. Read by :func:`tier_writes`; when
#: the key is absent the tier is ``never``.
WRITES_KEY = "control_system.writes"

#: :data:`WRITES_KEY`'s leaf inside the ``control_system:`` section.
_WRITES_LEAF = "writes"

#: One tier word, as a type: the three values of :data:`WRITES_KEY`, narrowest
#: first — every answer :func:`tier_writes` gives.
TierWord = Literal["never", "asked", "allowed"]

#: The three tier words, narrowest first: a rank, read by ``.index``.
TIER_WORDS: tuple[TierWord, ...] = get_args(TierWord)


def tier_writes(section: object) -> TierWord:
    """The write ceiling this deployment's login tier sets: ``never``, ``asked`` or ``allowed``.

    :data:`WRITES_KEY` decides when the section carries it, and then only the
    literal strings ``"asked"`` and ``"allowed"`` arm, matched exactly; ``"never"``,
    any other string (the switch's retired ``"on"`` among them), ``True``, ``None``
    and every other value answer ``never``. When the key is absent the tier is
    ``never``: the retired arming leaves are not read at run time, so a render
    built before the tier key existed writes nowhere until it is rebuilt (the
    build's normalizer writes the key into every render).

    Target-blind by design: a tier that allows writes allows them for every
    machine. The tier is a ceiling, not a selection — the session's switch picks
    the machine.

    Never raises. A section that is not a mapping answers ``never``.
    """
    value = section.get(_WRITES_LEAF) if isinstance(section, dict) else None
    if value == "asked":
        return "asked"
    if value == "allowed":
        return "allowed"
    return "never"


def tier_allows(section: object) -> bool:
    """Whether the tier arms anything: :func:`tier_writes` is not ``never``. Never raises.

    This reads the ceiling only: ``asked`` and ``allowed`` both answer true, so
    whether a write prompts is not decided here.
    """
    return tier_writes(section) != "never"
