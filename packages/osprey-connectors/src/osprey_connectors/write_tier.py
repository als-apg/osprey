"""How far a deployment may write: the login tier, ``control_system.writes``.

:func:`tier_writes` answers "how far may this deployment write" from the one tier key,
``control_system.writes``. The lint that reads the config, the profile card
that tells an operator what they hold, and the connector that would perform
the write all have to agree about the answer. Two readings of one posture is
one of them being wrong about a machine somebody can move.
"""

from typing import get_args

from osprey_connectors.config_sections import ControlSystemSection, TierWord

__all__ = [
    "TIER_WORDS",
    "WRITES_KEY",
    "tier_allows",
    "tier_writes",
]

#: The login tier's write ceiling, dotted as a caller spells it for
#: ``LoadedConfig.get``: ``"never"``, ``"asked"`` or ``"allowed"``, the furthest
#: position a login may pick. Target-blind — a tier that allows writes allows
#: them for every machine, and which machine is actually written is the
#: session's switch to decide, not the tier's. Read by :func:`tier_writes`; when
#: the key is absent the tier is ``never``.
WRITES_KEY = "control_system.writes"

#: The three tier words
#: (:data:`~osprey_connectors.config_sections.TierWord`), narrowest first: a
#: rank, read by ``.index``.
TIER_WORDS: tuple[TierWord, ...] = get_args(TierWord)


def tier_writes(section: ControlSystemSection) -> TierWord:
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

    Never raises: :attr:`~osprey_connectors.config_sections.ControlSystemSection.writes`
    is that reading, made once when the section is read.
    """
    return section.writes


def tier_allows(section: ControlSystemSection) -> bool:
    """Whether the tier arms anything: :func:`tier_writes` is not ``never``. Never raises.

    This reads the ceiling only: ``asked`` and ``allowed`` both answer true, so
    whether a write prompts is not decided here.
    """
    return tier_writes(section) != "never"
