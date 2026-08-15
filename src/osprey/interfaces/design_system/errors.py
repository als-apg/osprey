"""Shared fail-closed validation infrastructure.

The design system's three validators — the token validator
(``generator/validate.py``), the panel-manifest schema validator
(``panels/manifest.py``), and the panel validator (``panels/validator.py``)
— share one idiom: a domain-specific :class:`~enum.StrEnum` of
machine-readable rule ids, a frozen located *finding* dataclass rendering
``"{source}: {message}"``, and a ``ValueError`` subclass that bundles
*every* finding (a fail-closed door never reports just the first). The
rule enums and check functions stay domain-specific; the shared shape
lives here so the three validators render and bundle findings identically
by construction instead of by carefully-mirrored copies.

**Naming rule this package holds to:** the ``*Error`` suffix is reserved
for throwables. A validator's per-failure *record* — a frozen dataclass a
check appends to a list and hands back — is a ``*Finding``
(:class:`SourcedFinding`, ``TokenFinding``, ``ManifestFinding``,
``PanelFinding``). ``except SomeFinding`` is a ``TypeError`` waiting to
happen, so a finding never wears a name that invites it.

Concrete finding classes subclass :class:`SourcedFinding` (narrowing
``rule`` to their domain enum); the fail-closed doors raise a
:class:`BundledValidationError` subclass parametrized by their finding
type.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = ["SourcedFinding", "BundledValidationError"]

E = TypeVar("E")


@dataclass(frozen=True)
class SourcedFinding:
    """A single, located validation failure rendering ``"{source}: {message}"``.

    A plain record, NOT a throwable — see the naming rule in this module's
    docstring.

    Attributes:
        rule: Which check produced this finding. Subclasses re-declare this
            field with their domain's rule enum type (re-declaration keeps
            the ``(rule, message, source)`` field order).
        message: Human-readable description of the failure.
        source: Where the failure is — a file path string, a
            ``"path:line"`` location, or a placeholder for in-memory input.
    """

    rule: Any
    message: str
    source: str

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


class BundledValidationError(ValueError, Generic[E]):
    """Base for the fail-closed doors: a ``ValueError`` bundling every finding.

    This — and its subclasses — are the only ``*Error`` names in the family,
    and the only things here that can be caught.

    Attributes:
        errors: Every finding, in the order it was found.
    """

    def __init__(self, errors: Sequence[E]) -> None:
        self.errors: list[E] = list(errors)
        super().__init__("\n".join(str(error) for error in self.errors))
