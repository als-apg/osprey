"""Heuristic read/write direction vote for MML signal groups.

Every channel-bearing ``(raw_family, field)`` of a merged ``ao`` dict gets a
:class:`Vote`. Within one system the vote is decided in this order:

1. The field's own ``MemberOf`` tags, split on whitespace into words, so a
   multi-word tag such as ``Boolean Monitor`` or ``Multi-Bit Boolean Monitor``
   still speaks. A word from :data:`MEMBEROF_WORDS` ``"read"`` means read, one
   from ``"write"`` means write, and both present leave the system undecided
   (the field name is then not consulted, since the tags contradict each other).
2. The field name, by suffix only (:data:`GRAMMAR_SUFFIXES`). Bare ``On`` is
   never a suffix: in MML ``On`` is the monitor and ``OnControl`` the setpoint.
3. Otherwise undecided.

The vote is the union over sub-machines: systems with no evidence add nothing,
systems that decide and agree yield that direction, and systems that disagree
yield ``None``. ``per_system`` always records every system's own verdict so the
profile can print disagreements.

The voter proposes directions for ``map --init`` and checks stated ones in
``map --check``; ``emit`` never calls it, taking directions from the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from osprey.services.mml.family import family_views, system_bodies

__all__ = [
    "GRAMMAR_SUFFIXES",
    "MEMBEROF_WORDS",
    "Direction",
    "Vote",
    "VoteSource",
    "field_vote",
    "vote_directions",
]

Direction = Literal["read", "write"]
VoteSource = Literal["memberof", "grammar", "undecided"]

#: Whole words inside a field's ``MemberOf`` tags that mark its direction.
#: Source: the ``MemberOf`` tags MML exports put on fields, as seen in a
#: production export (``Monitor``, ``Setpoint``, ``MachineConfig``,
#: ``Boolean Monitor``, ``Boolean Control``, ``Multi-Bit Boolean Monitor``).
#: The tag vocabulary is open; words outside this table carry no direction.
MEMBEROF_WORDS: dict[Direction, tuple[str, ...]] = {
    "read": ("Monitor",),
    "write": ("Setpoint", "Control", "MachineConfig"),
}

#: Field-name suffixes that mark a direction when the tags do not.
GRAMMAR_SUFFIXES: dict[Direction, tuple[str, ...]] = {
    "read": ("Monitor", "Readback", "RBV", "RB"),
    "write": ("Setpoint", "Control", "Reset", "SP"),
}


@dataclass(frozen=True)
class Vote:
    """The direction vote for one ``(raw_family, field)``.

    Attributes:
        direction: ``"read"``, ``"write"``, or ``None`` when undecided or when
            systems disagree.
        per_system: Each carrying system's own verdict, keyed by system token.
        source: ``"memberof"`` when any system decided from tags, ``"grammar"``
            when every deciding system used the field name, ``"undecided"``
            when ``direction`` is ``None``.
    """

    direction: Direction | None
    per_system: dict[str, Direction | None]
    source: VoteSource


def _tag_words(member_of: Any) -> set[str]:
    """Return the whitespace-split words of every string tag, at any nesting."""
    if isinstance(member_of, str):
        return set(member_of.split())
    if isinstance(member_of, (list, tuple)):
        words: set[str] = set()
        for item in member_of:
            words |= _tag_words(item)
        return words
    return set()


def field_vote(name: str, member_of: Any) -> tuple[Direction | None, VoteSource]:
    """Vote one field's direction from its name and its ``MemberOf`` tags.

    The rule itself, taken apart from where the field was read: the module
    docstring states it, this function is it, and every caller -- the mapping
    voter below, and the VA channel manifest, which votes the same fields off
    the emitted channel database rather than off the raw export -- reaches
    exactly one copy of it.

    Args:
        name: The field's own name, e.g. ``Setpoint``.
        member_of: The field's ``MemberOf`` value, in any of the shapes an
            export writes it (a string, a nested list, or absent).

    Returns:
        ``(direction, source)`` -- the direction, or ``None`` when the tags
        contradict each other and when neither tags nor name decide.
    """
    words = _tag_words(member_of)
    hits = {direction for direction, table in MEMBEROF_WORDS.items() if words.intersection(table)}
    if len(hits) == 2:
        return None, "undecided"
    if hits:
        return hits.pop(), "memberof"
    for direction, suffixes in GRAMMAR_SUFFIXES.items():
        if name.endswith(suffixes):
            return direction, "grammar"
    return None, "undecided"


def _system_vote(name: str, body: dict) -> tuple[Direction | None, VoteSource]:
    """Vote one field in one system."""
    return field_vote(name, body.get("MemberOf"))


def vote_directions(ao: dict) -> dict[tuple[str, str], Vote]:
    """Vote a direction for every channel-bearing field of a merged export.

    Args:
        ao: ``{system: {family: body}}``; top-level and system-level keys
            starting with ``_`` (``_exports``, ``_import_order``,
            ``_description``) and non-dict values are skipped. Not modified.

    Returns:
        Votes keyed by ``(raw_family, field)``, in sorted key order.
    """
    collected: dict[tuple[str, str], dict[str, tuple[Direction | None, VoteSource]]] = {}
    for system, families in system_bodies(ao):
        for view in family_views(system, families):
            for field in view.fields.values():
                verdicts = collected.setdefault((view.raw_name, field.name), {})
                verdicts[system] = _system_vote(field.name, field.body)

    votes: dict[tuple[str, str], Vote] = {}
    for key in sorted(collected):
        verdicts = collected[key]
        per_system = {system: verdicts[system][0] for system in sorted(verdicts)}
        decided = {direction for direction, _ in verdicts.values() if direction is not None}
        if len(decided) == 1:
            sources = {source for direction, source in verdicts.values() if direction}
            source: VoteSource = "memberof" if "memberof" in sources else "grammar"
            votes[key] = Vote(decided.pop(), per_system, source)
        else:
            votes[key] = Vote(None, per_system, "undecided")
    return votes
