"""Facility references: a mention of a facility record in text.

A :class:`FacilityReference` is one span of a text that names a channel id or a
device id of the facility file, with the kinds of record that id resolves to.
A finder turns text into references: :func:`find_facility_links` for the markup
an author writes on a knowledge page, :func:`find_facility_references` for a
quoted id in logbook prose, looked up in the facility file. :func:`resolved_references`
turns references into the list every surface shows: each span with the records
its target resolves to.

A facility link is a standard markdown inline link whose destination uses the
``facility:`` scheme, ``[label](facility:ADDRESS)``. A target that holds
whitespace, a parenthesis, an angle bracket or a backslash is written as a
CommonMark angle-bracket destination around the whole destination,
``[label](<facility:ADDRESS WITH SPACES>)``.
:class:`osprey.facility.records.FacilityRecords` resolves the target.

The link finder reads markdown link syntax only: a link inside a fenced code
block or an inline code span is not a link, and nothing outside a link is ever
read as a facility name.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.facility.records import LINK_KINDS, LinkKind

if TYPE_CHECKING:
    from osprey.facility.records import FacilityRecords

__all__ = [
    "LINK_SCHEME",
    "FacilityReference",
    "find_facility_links",
    "find_facility_references",
    "resolved_references",
]

#: The URI scheme of a facility link's destination.
LINK_SCHEME = "facility"

_PREFIX = f"{LINK_SCHEME}:"

#: An inline link (not an image) whose destination is a facility link, bare or
#: in angle brackets.
_LINK = re.compile(
    r"(?<![!\\])\[(?P<label>(?:\\.|[^\\\[\]])*)\]\(\s*"
    rf"(?:<{_PREFIX}(?P<angle>(?:\\.|[^\\<>\n])*)>|{_PREFIX}(?P<bare>[^\s()<>\\]+))"
    r"\s*\)"
)

#: An opening code fence: up to three spaces, then three or more backticks or tildes.
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")

#: An inline code span: a run of backticks, its content, the same run again.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)", re.DOTALL)

_ESCAPE = re.compile(r"\\(.)", re.DOTALL)


def _link_kinds(kinds: Collection[str]) -> tuple[LinkKind, ...]:
    """The link kinds among ``kinds``, in ``LINK_KINDS`` order and distinct."""
    return tuple(kind for kind in LINK_KINDS if kind in kinds)


@dataclass(frozen=True)
class FacilityReference:
    """One mention of a facility record in a text.

    Attributes:
        start: The Unicode code-point offset into the text where the mention
            starts.
        end: The Unicode code-point offset into the text just past the
            mention; greater than ``start``.
        target: The channel id or device id the mention names; never empty.
        kinds: The kinds of record ``target`` resolves to, in ``LINK_KINDS``
            order and distinct; empty when it resolves to none or was not
            looked up.
        label: The text an author gave a facility link, with its backslash
            escapes undone; ``None`` for a mention that is not a link. A mention
            spans exactly its target (``end - start == len(target)``); a link
            spans its whole markup.

    Raises:
        ValueError: A field breaks one of the invariants above.
    """

    start: int
    end: int
    target: str
    kinds: tuple[LinkKind, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"a reference spans start < end, not {self.start}..{self.end}")
        if not self.target:
            raise ValueError("a reference names a target")
        if self.label is None and self.end - self.start != len(self.target):
            raise ValueError(
                f"a mention spans its target {self.target!r}, not {self.start}..{self.end}"
            )
        kinds = tuple(self.kinds)
        if any(kind not in LINK_KINDS for kind in kinds):
            raise ValueError(f"a reference resolves to channels or devices, not {kinds!r}")
        if kinds != _link_kinds(kinds):
            raise ValueError(
                f"a reference's kinds are channel then device, each once, not {kinds!r}"
            )
        object.__setattr__(self, "kinds", kinds)

    def to_json(self) -> dict[str, Any]:
        """This reference as a JSON object; ``label`` appears only on a link."""
        document: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "target": self.target,
            "kinds": list(self.kinds),
        }
        if self.label is not None:
            document["label"] = self.label
        return document

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> FacilityReference:
        """The reference :meth:`to_json` wrote.

        Args:
            document: A JSON object :meth:`to_json` returned.

        Returns:
            The reference.

        Raises:
            TypeError: A field is missing or unknown.
            ValueError: A field breaks one of the type's invariants.
        """
        fields: dict[str, Any] = {
            key: tuple(val) if key == "kinds" else val for key, val in document.items()
        }
        return cls(**fields)


def _blank_code(markdown: str) -> str:
    """``markdown`` with every fenced block and code span blanked, offsets kept."""
    lines = markdown.splitlines(keepends=True)
    kept: list[str] = []
    fence: str | None = None
    for line in lines:
        opening = _FENCE.match(line)
        if fence is None and opening:
            fence = opening.group(1)
            kept.append(_blank(line))
        elif fence is not None:
            closing = re.match(rf"^ {{0,3}}{fence[0]}{{{len(fence)},}}\s*$", line)
            if closing:
                fence = None
            kept.append(_blank(line))
        else:
            kept.append(line)
    return _CODE_SPAN.sub(lambda match: _blank(match.group(0)), "".join(kept))


def _blank(text: str) -> str:
    """``text`` with every character but a line break turned into a space."""
    return re.sub(r"[^\n]", " ", text)


def find_facility_links(
    text: str, records: FacilityRecords | None = None
) -> list[FacilityReference]:
    """Every facility link in the markdown ``text``, as references in text order.

    Args:
        text: A page body, or any markdown text.
        records: The facility records each target is looked up in; without
            them every reference's ``kinds`` stays empty.

    Returns:
        One :class:`FacilityReference` per inline link with a ``facility:``
        destination outside code, spanning the whole link and carrying its label.
    """
    found: list[FacilityReference] = []
    for match in _LINK.finditer(_blank_code(text)):
        source = text[match.start() : match.end()]
        parsed = _LINK.fullmatch(source)
        if parsed is None:
            continue
        angle = parsed.group("angle")
        target = _ESCAPE.sub(r"\1", angle) if angle is not None else parsed.group("bare")
        if not target:
            continue
        kinds: tuple[LinkKind, ...] = ()
        if records is not None:
            kinds = _link_kinds({record.kind for record in records.resolve(target)})
        found.append(
            FacilityReference(
                match.start(),
                match.end(),
                target,
                kinds,
                label=_ESCAPE.sub(r"\1", parsed.group("label")),
            )
        )
    return found


def _is_identifier_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def find_facility_references(text: str, records: FacilityRecords) -> list[FacilityReference]:
    """Every id of the facility file that ``text`` quotes, as references in text order.

    A quote is a target of ``records`` with no identifier character (alphanumeric
    or ``_``) directly before or after it; the longest target wins at a position.
    The text is not markdown: a quoted id is a reference wherever it stands.

    Args:
        text: Logbook prose.
        records: The facility records whose ids are looked for.

    Returns:
        One :class:`FacilityReference` per quote, with code-point offsets, no
        label, and the kinds its target resolves to in the order
        :meth:`~osprey.facility.records.FacilityRecords.resolve` lists them.
    """
    targets = records.targets()
    lengths = sorted({len(target) for target in targets if target}, reverse=True)
    found: list[FacilityReference] = []
    position = 0
    while position < len(text):
        target = None
        if position == 0 or not _is_identifier_character(text[position - 1]):
            target = _longest_quote(text, position, targets, lengths)
        if target is None:
            position += 1
            continue
        kinds = _link_kinds({record.kind for record in records.resolve(target)})
        found.append(FacilityReference(position, position + len(target), target, kinds))
        position += len(target)
    return found


def _longest_quote(
    text: str, position: int, targets: frozenset[str], lengths: list[int]
) -> str | None:
    """The longest target at ``position`` with no identifier character right after it."""
    for length in lengths:
        end = position + length
        if end > len(text) or text[position:end] not in targets:
            continue
        if end == len(text) or not _is_identifier_character(text[end]):
            return text[position:end]
    return None


def resolved_references(
    references: Iterable[FacilityReference], records: FacilityRecords | None
) -> list[dict[str, Any]]:
    """The references as a surface shows them: each span with its records.

    Args:
        references: Spans a finder returned, in text order.
        records: The facility records the targets resolve in, or ``None``
            when no facility file was read.

    Returns:
        One ``{start, end, target, kinds, label, records}`` per reference, in
        the order given. ``records`` lists one ``{kind, id, unit?,
        description?}`` per record the target resolves to (empty for a target
        the facility file does not hold), or is ``None`` without records.
    """
    resolved: list[dict[str, Any]] = []
    for found in references:
        described = records.describe(found.target) if records is not None else None
        kinds = found.kinds
        if described is not None:
            kinds = _link_kinds({entry["kind"] for entry in described})
        resolved.append(
            {
                "start": found.start,
                "end": found.end,
                "target": found.target,
                "kinds": list(kinds),
                "label": found.label,
                "records": described,
            }
        )
    return resolved
