"""Shared fail-closed validation-error infrastructure.

The design system's three validators — the token validator
(``generator/validate.py``), the panel-manifest schema validator
(``panels/manifest.py``), and the panel validator (``panels/validator.py``)
— share one error idiom: a domain-specific :class:`~enum.StrEnum` of
machine-readable rule ids, a frozen located-error dataclass rendering
``"{source}: {message}"``, and a ``ValueError`` subclass that bundles
*every* failure (a fail-closed door never reports just the first). The
rule enums and check functions stay domain-specific; the shared shape
lives here so the three validators render and bundle errors identically
by construction instead of by carefully-mirrored copies.

Concrete error classes subclass :class:`SourcedError` (narrowing ``rule``
to their domain enum) and :class:`BundledValidationError` (parametrized by
their error type), keeping every public name and constructor signature the
consuming test suites pin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = ["SourcedError", "BundledValidationError"]

E = TypeVar("E")


@dataclass(frozen=True)
class SourcedError:
    """A single, located validation failure rendering ``"{source}: {message}"``.

    Attributes:
        rule: Which check produced this error. Subclasses re-declare this
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
    """Base for the fail-closed doors: a ``ValueError`` bundling every failure.

    Attributes:
        errors: Every validation failure, in the order it was found.
    """

    def __init__(self, errors: Sequence[E]) -> None:
        self.errors: list[E] = list(errors)
        super().__init__("\n".join(str(error) for error in self.errors))
